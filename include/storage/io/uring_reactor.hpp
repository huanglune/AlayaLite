// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file uring_reactor.hpp
 * @brief Shared io_uring with cooperatively polled completions for libcoro.
 *
 * `UringReactor` complements `IOUringEngine` (thread-local rings, same-thread
 * batch submit+wait): a shared ring lets any pool thread drive any wave, so a
 * coroutine can start a read on one thread and finish it on another.
 *
 * Completion delivery is COOPERATIVE POLLING, not a reaper thread. The first
 * design used a dedicated reaper (io_uring_wait_cqe -> pool.resume per wave);
 * profiling the disk-bound update benchmark showed why that cannot keep up:
 * every wave paid a cross-thread wake chain (reaper wakeup -> pool queue ->
 * condition-variable wake -> requeue), ~230K voluntary context switches/s and
 * 41% kernel time, capping thread utilization at ~57%. Polling removes the
 * chain entirely: the waiting coroutine reaps the CQ from user space (no
 * syscall) and yields through its pool between polls, so completions ride the
 * executor's existing scheduling instead of waking it. This is the same
 * philosophy as the archived poll-flag IOUringEngine design (workers poll,
 * nothing sleeps per-I/O), adapted to coroutines.
 *
 * Mechanics:
 *  - One shared ring. SQE prep under `sq_mutex_`, ONE io_uring_submit syscall
 *    per wave (not per read). Under the lock the SQ is always flushed before
 *    release and waves are chunked to the ring depth, so get_sqe can never
 *    find the SQ full.
 *  - `try_reap()` drains ready CQEs under a try-locked `cq_mutex_` (peek is
 *    pure user-space). Completion callbacks only store results into the wave
 *    frame — they run on whichever thread polls, never resume anything.
 *  - `read()` / `read_batch()` are coro::tasks: submit, then poll-yield until
 *    the wave's atomic counter hits zero. All state lives in the coroutine
 *    frame — no heap allocation per I/O, and no resume handoff to race with.
 *  - Kernel >= 5.5 (IORING_FEAT_NODROP) buffers completions past CQ capacity,
 *    so many concurrent waves are safe.
 *
 * Shutdown contract: callers must drain all awaited reads before destroying
 * the reactor (the update paths sync_wait their batches). The destructor
 * inline-reaps any stragglers so the ring is quiesced before queue_exit.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "storage/io/io_engine.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/platform.hpp"

#ifdef ALAYA_OS_LINUX
  #include <liburing.h>
#endif

// ThreadSanitizer cannot see the submit -> kernel -> CQE handoff (liburing is
// an uninstrumented prebuilt and the ring lives in shared memory), so the
// writes a submitter makes to wave state before io_uring_submit look
// unordered against the reads a reaping thread makes after peeking the CQE.
// These annotations hand TSan exactly that edge; they compile to nothing in
// normal builds.
#if defined(__SANITIZE_THREAD__)
extern "C" void __tsan_acquire(void *addr);
extern "C" void __tsan_release(void *addr);
  #define ALAYA_TSAN_RELEASE(addr) __tsan_release(addr)
  #define ALAYA_TSAN_ACQUIRE(addr) __tsan_acquire(addr)
#else
  #define ALAYA_TSAN_RELEASE(addr) ((void)0)
  #define ALAYA_TSAN_ACQUIRE(addr) ((void)0)
#endif

namespace alaya {

#ifdef ALAYA_OS_LINUX

class UringReactor {
 public:
  /// Completion context carried in SQE user_data, embedded in the awaiting
  /// coroutine's frame. `fn(arg, result)` runs on WHICHEVER thread reaps the
  /// CQE and must only store into the wave state — never block, never resume.
  struct Completion {
    AsyncIOCallback fn;
    void *arg;
  };

  explicit UringReactor(unsigned queue_depth = kDefaultQueueDepth) : depth_(queue_depth) {
    const int ret = io_uring_queue_init(queue_depth, &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error("UringReactor: io_uring_queue_init failed: " +
                               std::string(strerror(-ret)));
    }
  }

  ~UringReactor() {
    // Quiesce the ring: callers should have drained, but never queue_exit with
    // I/O in flight. Waves that leaked would have dangling frames either way.
    while (in_flight_.load(std::memory_order_acquire) > 0) {
      struct io_uring_cqe *cqe = nullptr;
      const int ret = io_uring_wait_cqe(&ring_, &cqe);
      if (ret == -EINTR) {
        continue;
      }
      if (ret < 0) {
        break;
      }
      dispatch_cqe(cqe);
    }
    io_uring_queue_exit(&ring_);
  }

  ALAYA_NON_COPYABLE_NON_MOVABLE(UringReactor);

  /// True when the kernel supports what the reactor needs (io_uring with READ).
  static auto is_available() -> bool {
    struct io_uring_probe *probe = io_uring_get_probe();
    if (probe == nullptr) {
      return false;
    }
    const bool ok = io_uring_opcode_supported(probe, IORING_OP_READ) != 0;
    io_uring_free_probe(probe);
    return ok;
  }

  /// Drain ready completions (user-space CQ peek, no syscall). Returns true if
  /// this call dispatched at least one CQE or another thread holds the CQ —
  /// i.e. "progress is being made"; false means the CQ was idle.
  auto try_reap() -> bool {
    std::unique_lock<std::mutex> lock(cq_mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      return true;
    }
    bool any = false;
    struct io_uring_cqe *cqe = nullptr;
    while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
      dispatch_cqe(cqe);
      any = true;
    }
    return any;
  }

  /// Wave of positioned reads: submit all (one syscall per depth-sized chunk),
  /// then cooperatively poll until every read lands. Runs on the caller's
  /// pool; between polls the coroutine yields so other update work executes.
  /// co_await yields the number of failed reads (result != requested size);
  /// per-read results are in `requests[i].result_`.
  auto read_batch(coro::thread_pool &pool, int fd, IORequest *requests, uint32_t count)
      -> coro::task<uint32_t> {
    if (count == 0) {
      co_return 0;
    }
    WaveState wave;
    wave.remaining.store(count, std::memory_order_release);
    std::vector<Slot> slots(count);
    std::vector<Completion> completions(count);
    for (uint32_t i = 0; i < count; ++i) {
      slots[i] = Slot{&wave, &requests[i]};
      completions[i] = Completion{&UringReactor::on_slot_complete, &slots[i]};
    }
    // Chunked submission with exact accounting: submit_reads() reports how
    // many reads actually reached the kernel. On a short count (reactor
    // poisoned) the never-submitted remainder is deducted, the in-kernel reads
    // are drained against this (still-alive) frame, and only then does the
    // error propagate.
    uint32_t submitted = 0;
    bool poisoned = false;
    while (submitted < count) {
      const uint32_t chunk = std::min(count - submitted, depth_);
      const uint32_t flushed =
          submit_reads(fd, &requests[submitted], &completions[submitted], chunk);
      submitted += flushed;
      if (flushed < chunk) {
        poisoned = true;
        break;
      }
    }
    if (poisoned) {
      wave.remaining.fetch_sub(count - submitted, std::memory_order_acq_rel);
    }
    while (wave.remaining.load(std::memory_order_acquire) != 0) {
      (void)try_reap();
      if (wave.remaining.load(std::memory_order_acquire) == 0) {
        break;
      }
      co_await pool.schedule();
    }
    if (poisoned) {
      throw std::runtime_error(
          "UringReactor: io_uring submission failed (reactor poisoned; see log)");
    }
    co_return wave.failures.load(std::memory_order_acquire);
  }

  /// One polled read; co_await yields bytes-read (or -errno). Callers validate
  /// the length exactly like the blocking pread paths do.
  auto read(coro::thread_pool &pool, int fd, void *buf, uint32_t len, uint64_t off)
      -> coro::task<int32_t> {
    IORequest request{buf, len, off};
    (void)co_await read_batch(pool, fd, &request, 1);
    co_return request.result_;
  }

 private:
  static constexpr unsigned kDefaultQueueDepth = 4096;

  struct WaveState {
    std::atomic<uint32_t> remaining{0};
    std::atomic<uint32_t> failures{0};
  };

  struct Slot {
    WaveState *wave = nullptr;
    IORequest *request = nullptr;
  };

  /// Runs on the reaping thread. Store-only: the wave's poll loop observes
  /// remaining == 0 with acquire ordering, so all stores below are visible to
  /// the frame before it resumes past the loop.
  static void on_slot_complete(void *arg, int32_t result) {
    auto *slot = static_cast<Slot *>(arg);
    slot->request->result_ = result;
    if (result < 0 || static_cast<size_t>(result) != slot->request->size_) {
      slot->wave->failures.fetch_add(1, std::memory_order_acq_rel);
    }
    slot->wave->remaining.fetch_sub(1, std::memory_order_acq_rel);
  }

  void dispatch_cqe(struct io_uring_cqe *cqe) {
    ALAYA_TSAN_ACQUIRE(&ring_);  // pairs with the release in submit_reads
    auto *completion = static_cast<Completion *>(io_uring_cqe_get_data(cqe));
    const int32_t res = cqe->res;
    io_uring_cqe_seen(&ring_, cqe);
    in_flight_.fetch_sub(1, std::memory_order_acq_rel);
    if (completion != nullptr) {
      completion->fn(completion->arg, res);
    }
  }

  /// Prep + submit @p n reads as ONE io_uring_submit (bounded retries on
  /// transient errors). Returns how many reads actually reached the kernel —
  /// n on success. Any hard failure POISONS the reactor: every later call
  /// returns 0 without prepping, so SQEs parked by a failed flush are never
  /// handed to the kernel and their (long-dead) slot pointers are never
  /// dispatched. Caller guarantees n <= depth_; under sq_mutex_ the SQ is
  /// always flushed before release, so get_sqe cannot fail while healthy.
  auto submit_reads(int fd, IORequest *requests, Completion *completions, uint32_t n) -> uint32_t {
    std::lock_guard<std::mutex> lock(sq_mutex_);
    if (poisoned_) {
      return 0;
    }
    for (uint32_t i = 0; i < n; ++i) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) {
        // Unreachable while healthy (waves chunked to depth_, SQ flushed every
        // submit). Poison: the i SQEs prepped above are parked forever.
        poisoned_ = true;
        LOG_ERROR("UringReactor: SQ unexpectedly full — reactor poisoned");
        return 0;
      }
      io_uring_prep_read(sqe,
                         fd,
                         requests[i].buffer_,
                         static_cast<uint32_t>(requests[i].size_),
                         requests[i].offset_);
      io_uring_sqe_set_data(sqe, &completions[i]);
    }
    ALAYA_TSAN_RELEASE(&ring_);  // publish frame writes to whoever reaps the CQE
    uint32_t flushed = 0;
    int spins = 0;
    while (flushed < n) {
      const int ret = io_uring_submit(&ring_);
      if (ret > 0) {
        flushed += static_cast<uint32_t>(ret);
        in_flight_.fetch_add(static_cast<uint32_t>(ret), std::memory_order_acq_rel);
        continue;
      }
      if (ret == -EINTR) {
        continue;
      }
      if ((ret == 0 || ret == -EAGAIN || ret == -EBUSY) && ++spins < kSubmitRetries) {
        (void)try_reap();  // cq_mutex_ != sq_mutex_: relieve CQ pressure
        continue;
      }
      poisoned_ = true;
      LOG_ERROR("UringReactor: io_uring_submit failed ({}) after {}/{} — reactor poisoned",
                ret < 0 ? strerror(-ret) : "no progress",
                flushed,
                n);
      break;
    }
    return flushed;
  }

  static constexpr int kSubmitRetries = 1024;

  struct io_uring ring_{};
  uint32_t depth_ = kDefaultQueueDepth;
  std::mutex sq_mutex_;
  std::mutex cq_mutex_;
  bool poisoned_ = false;  ///< guarded by sq_mutex_
  std::atomic<uint64_t> in_flight_{0};
};

#else  // !ALAYA_OS_LINUX — stub so call sites can compile; is_available() gates use.

class UringReactor {
 public:
  struct Completion {
    AsyncIOCallback fn;
    void *arg;
  };
  explicit UringReactor(unsigned /*queue_depth*/ = 0) {
    throw std::runtime_error("UringReactor requires Linux io_uring");
  }
  static auto is_available() -> bool { return false; }
};

#endif  // ALAYA_OS_LINUX

}  // namespace alaya
