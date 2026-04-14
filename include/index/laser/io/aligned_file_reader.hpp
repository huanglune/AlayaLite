/**
 * @file aligned_file_reader.hpp
 * @brief Linux AIO (Asynchronous I/O) wrapper for high-performance disk access.
 *
 * This module provides direct disk I/O capabilities using Linux kernel AIO,
 * which is essential for achieving predictable, low-latency disk reads in
 * disk-based vector search systems.
 *
 * Key Design Decisions:
 * - Uses O_DIRECT to bypass the OS page cache
 * - Uses Linux AIO (libaio) for asynchronous, non-blocking I/O
 * - Requires 512-byte alignment for all buffers, offsets, and lengths
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <libaio.h>
#include <unistd.h>

namespace symqg::aio {

constexpr size_t kDefaultAioEventsPerThread = 128;
constexpr size_t kMaxIoDepth = 128;
constexpr size_t kSectorAlignment = 512;

[[nodiscard]] constexpr auto round_up(uint64_t value, uint64_t alignment) -> uint64_t {
  return ((value / alignment) + static_cast<uint64_t>(value % alignment != 0)) * alignment;
}

[[nodiscard]] constexpr auto is_aligned(uint64_t value, uint64_t alignment) -> bool {
  return value % alignment == 0;
}

[[nodiscard]] constexpr auto is_sector_aligned(uint64_t value) -> bool {
  return is_aligned(value, kSectorAlignment);
}

using IOContext = io_context_t;

/**
 * @brief Represents a single aligned read request for O_DIRECT I/O.
 *
 * All fields must be 512-byte aligned to satisfy O_DIRECT requirements.
 */
struct AlignedRead {
  uint64_t offset_{0};
  uint64_t len_{0};
  uint64_t id_{0};
  void *buf_{nullptr};

  AlignedRead() = default;

  AlignedRead(uint64_t offset, uint64_t len, uint64_t id, void *buf)
      : offset_(offset), len_(len), id_(id), buf_(buf) {
    assert(is_sector_aligned(offset_));
    assert(is_sector_aligned(len_));
    assert(is_sector_aligned(reinterpret_cast<uint64_t>(buf_)));  // NOLINT
  }
};

namespace detail {

inline void prepare_iocbs(int fd,
                          AlignedRead *reqs,
                          size_t count,
                          iocb *cb_buf,
                          iocb **cbs_buf) {
  for (size_t j = 0; j < count; ++j) {
    io_prep_pread(&cb_buf[j], fd, reqs[j].buf_, reqs[j].len_,
                  static_cast<off_t>(reqs[j].offset_));
    cb_buf[j].data = reinterpret_cast<void *>(static_cast<uintptr_t>(reqs[j].id_));  // NOLINT
    cbs_buf[j] = &cb_buf[j];
  }
}

inline auto submit_and_check(io_context_t ctx, size_t count, iocb **cbs) -> int {
  int ret = io_submit(ctx, static_cast<int64_t>(count), cbs);
  if (ret != static_cast<int>(count)) {
    throw std::runtime_error(
        std::string("io_submit() failed; returned ") + std::to_string(ret) +
        ", expected=" + std::to_string(count) +
        ", errno=" + std::to_string(errno) + "=" + ::strerror(errno));
  }
  return ret;
}

}  // namespace detail

class LinuxAlignedFileReader {
 public:
  LinuxAlignedFileReader() = default;

  ~LinuxAlignedFileReader() {
    int64_t ret = ::fcntl(file_desc_, F_GETFD);
    if (ret == -1) {
      if (errno != EBADF) {
        ::close(file_desc_);
      }
    }
  }

  LinuxAlignedFileReader(const LinuxAlignedFileReader &) = delete;
  auto operator=(const LinuxAlignedFileReader &) -> LinuxAlignedFileReader & = delete;
  LinuxAlignedFileReader(LinuxAlignedFileReader &&) = delete;
  auto operator=(LinuxAlignedFileReader &&) -> LinuxAlignedFileReader & = delete;

  auto get_ctx() -> IOContext & {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    auto it = ctx_map_.find(std::this_thread::get_id());
    if (it == ctx_map_.end()) {
      return bad_ctx_;
    }
    return it->second;
  }

  void register_thread() { register_thread(kDefaultAioEventsPerThread); }

  void register_thread(size_t max_events) {
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    if (ctx_map_.find(my_id) != ctx_map_.end()) {
      return;
    }
    io_context_t ctx = nullptr;
    int ret = io_setup(static_cast<int>(max_events), &ctx);
    if (ret != 0) {
      lk.unlock();
      if (errno == EAGAIN) {
        throw std::runtime_error(
            "io_setup() failed with EAGAIN: AIO context limit reached. "
            "Check: cat /proc/sys/fs/aio-nr /proc/sys/fs/aio-max-nr. "
            "Fix: sudo sysctl -w fs.aio-max-nr=1048576");
      }
      if (errno == ENOMEM) {
        throw std::runtime_error("io_setup() failed: insufficient kernel memory for AIO context");
      }
      throw std::runtime_error(std::string("io_setup() failed; returned ") + std::to_string(ret) +
                               ", errno=" + std::to_string(errno) + ":" + ::strerror(errno));
    }
    ctx_map_[my_id] = ctx;
  }

  void deregister_thread() {
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut_);
    auto it = ctx_map_.find(my_id);
    assert(it != ctx_map_.end());
    io_context_t ctx = it->second;
    ctx_map_.erase(it);
    lk.unlock();
    io_destroy(ctx);
  }

  void deregister_all_threads() {
    std::unique_lock<std::mutex> lk(ctx_mut_);
    for (auto &[tid, ctx] : ctx_map_) {
      io_destroy(ctx);
    }
    ctx_map_.clear();
  }

  void open(const std::string &fname) { open(fname, /*direct_io=*/true); }

  void open(const std::string &fname, bool direct_io) {
    int flags = O_RDONLY;
    if (direct_io) {
      flags |= O_DIRECT;
    }
    file_desc_ = ::open(fname.c_str(), flags);
    if (file_desc_ == -1) {
      throw std::runtime_error("Failed to open file: " + fname +
                               ", errno=" + std::to_string(errno) + ":" + ::strerror(errno));
    }
  }

  void close() {
    if (file_desc_ != -1) {
      ::close(file_desc_);
      file_desc_ = -1;
    }
  }

  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool /*async*/ = false) {
    assert(file_desc_ != -1);
    execute_io(ctx, read_reqs);
  }

  /**
   * @brief Submit read requests asynchronously using pre-allocated iocb buffers (zero-alloc).
   *
   * This is the hot-path overload used during beam search. The caller provides
   * pre-allocated iocb arrays to avoid any heap allocation.
   */
  auto submit_reqs(AlignedRead *read_reqs,
                   size_t count,
                   IOContext &ctx,
                   iocb *cb_buf,
                   iocb **cbs_buf) -> int {
    assert(file_desc_ != -1);
    detail::prepare_iocbs(file_desc_, read_reqs, count, cb_buf, cbs_buf);
    return detail::submit_and_check(ctx, count, cbs_buf);
  }

  /**
   * @brief Submit read requests asynchronously (allocates temporary iocb arrays).
   *
   * Non-hot-path overload for convenience when pre-allocated buffers aren't available.
   */
  auto submit_reqs(AlignedRead *read_reqs, size_t count, IOContext &ctx) -> int {
    assert(file_desc_ != -1);
    std::vector<iocb> cb(count);
    std::vector<iocb *> cbs(count);
    detail::prepare_iocbs(file_desc_, read_reqs, count, cb.data(), cbs.data());
    return detail::submit_and_check(ctx, count, cbs.data());
  }

  auto submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) -> int {
    return submit_reqs(read_reqs.data(), read_reqs.size(), ctx);
  }

  void get_events(IOContext &ctx, int n_ops) {
    std::vector<io_event> evts(n_ops);
    auto ret = io_getevents(ctx, static_cast<int64_t>(n_ops), static_cast<int64_t>(n_ops),
                            evts.data(), nullptr);
    if (ret != static_cast<int64_t>(n_ops)) {
      throw std::runtime_error(
          std::string("io_getevents() failed; returned ") + std::to_string(ret) +
          ", expected=" + std::to_string(n_ops));
    }
  }

 private:
  int file_desc_{-1};
  io_context_t bad_ctx_{nullptr};
  std::map<std::thread::id, IOContext> ctx_map_;
  std::mutex ctx_mut_;

  void execute_io(io_context_t ctx, std::vector<AlignedRead> &read_reqs) {
#ifdef DEBUG
    for (auto &req : read_reqs) {
      assert(is_sector_aligned(req.len_));
      assert(is_sector_aligned(req.offset_));
      assert(is_sector_aligned(reinterpret_cast<uint64_t>(req.buf_)));  // NOLINT
    }
#endif

    uint64_t n_iters = round_up(read_reqs.size(), kDefaultAioEventsPerThread) /
                        kDefaultAioEventsPerThread;
    for (uint64_t iter = 0; iter < n_iters; ++iter) {
      uint64_t n_ops = std::min(
          static_cast<uint64_t>(read_reqs.size()) - (iter * kDefaultAioEventsPerThread),
          static_cast<uint64_t>(kDefaultAioEventsPerThread));

      std::vector<iocb> cb(n_ops);
      std::vector<iocb *> cbs(n_ops);
      auto *batch_start = read_reqs.data() + iter * kDefaultAioEventsPerThread;
      detail::prepare_iocbs(file_desc_, batch_start, n_ops, cb.data(), cbs.data());
      detail::submit_and_check(ctx, n_ops, cbs.data());

      std::vector<io_event> evts(n_ops);
      auto ret = io_getevents(ctx, static_cast<int64_t>(n_ops), static_cast<int64_t>(n_ops),
                              evts.data(), nullptr);
      if (ret != static_cast<int64_t>(n_ops)) {
        throw std::runtime_error(
            std::string("io_getevents() failed; returned ") + std::to_string(ret) +
            ", expected=" + std::to_string(n_ops));
      }
    }
  }
};

}  // namespace symqg::aio

// Backward-compatible aliases in global scope for existing callers
using AlignedRead = symqg::aio::AlignedRead;
using LinuxAlignedFileReader = symqg::aio::LinuxAlignedFileReader;
using IOContext = symqg::aio::IOContext;
