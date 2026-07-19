// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#if !defined(__linux__)
  #error "LibaioPageReader is available only on Linux"
#endif

#include <libaio.h>

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>

#include "storage/io/alignment.hpp"

namespace alaya::storage::io {

class LibaioPageReader final : public PageReader {
 private:
  struct BatchState {
    std::atomic_size_t remaining{0};
    std::atomic_bool cancel_requested{false};
  };
  struct RequestState {
    iocb cb{};
    ReadRequest request;
    Completion completion;
    std::shared_ptr<BatchState> batch;
    bool suppress_completion = false;
  };

 public:
  explicit LibaioPageReader(const std::filesystem::path &path, ReaderOptions options = {})
      : constraints_(direct_constraints(path, std::max<std::uint32_t>(1, options.queue_depth))) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open O_DIRECT");
    const int setup = ::io_setup(constraints_.max_batch, &context_);
    if (setup < 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::system_error(-setup, std::generic_category(), "io_setup");
    }
    try {
      reaper_ = std::thread([this] {
        reap_loop();
      });
    } catch (...) {
      ::io_destroy(context_);
      ::close(fd_);
      throw;
    }
  }

  ~LibaioPageReader() override { shutdown(); }

  [[nodiscard]] auto constraints() const noexcept -> ReadConstraints override {
    return constraints_;
  }

  [[nodiscard]] auto submit(std::span<const ReadRequest> requests, Completion completion)
      -> BatchHandle override {
    if ((completion.fn == nullptr && !requests.empty()) || requests.size() > constraints_.max_batch)
      throw std::invalid_argument("invalid PageReader batch");
    for (const auto &request : requests) {
      if (!validate_read_request(request, constraints_))
        throw std::invalid_argument("read request violates alignment constraints");
    }

    auto batch = std::make_shared<BatchState>();
    batch->remaining.store(requests.size(), std::memory_order_relaxed);
    auto handle = make_batch_handle([weak = std::weak_ptr(batch)]() noexcept {
      const auto state = weak.lock();
      if (!state || state->remaining.load(std::memory_order_acquire) == 0)
        return CancelResult::already_complete;
      state->cancel_requested.store(true, std::memory_order_release);
      return CancelResult::requested;
    });
    if (requests.empty()) {
      std::lock_guard lock(mutex_);
      if (stopping_) throw std::runtime_error("PageReader is shut down");
      return handle;
    }

    std::vector<std::unique_ptr<RequestState>> states;
    std::vector<iocb *> cbs;
    states.reserve(requests.size());
    cbs.reserve(requests.size());
    for (const auto &request : requests) {
      auto state = std::make_unique<RequestState>();
      state->request = request;
      state->completion = completion;
      state->batch = batch;
      ::io_prep_pread(&state->cb,
                      fd_,
                      request.buffer.data(),
                      request.buffer.size(),
                      request.offset);
      state->cb.data = state.get();
      cbs.push_back(&state->cb);
      states.push_back(std::move(state));
    }

    std::unique_lock lock(mutex_);
    capacity_cv_.wait(lock, [&] {
      return stopping_ || outstanding_ + requests.size() <= constraints_.max_batch;
    });
    if (stopping_) throw std::runtime_error("PageReader is shut down");
    const int submitted =
        ::io_submit(context_, static_cast<long>(cbs.size()), cbs.data());  // NOLINT(runtime/int)
    if (submitted != static_cast<int>(cbs.size())) {
      if (submitted > 0) {
        outstanding_ += static_cast<std::size_t>(submitted);
        for (int i = 0; i < submitted; ++i) {
          states[static_cast<std::size_t>(i)]->suppress_completion = true;
          states[static_cast<std::size_t>(i)].release();
        }
        drain_cv_.wait(lock, [&] {
          return outstanding_ == 0;
        });
      }
      const int error = submitted < 0 ? -submitted : EAGAIN;
      throw std::system_error(error, std::generic_category(), "partial io_submit");
    }
    outstanding_ += requests.size();
    for (auto &state : states) state.release();
    lock.unlock();
    return handle;
  }

  void shutdown() noexcept override {
    {
      std::unique_lock lock(mutex_);
      if (stopping_) return;
      stopping_ = true;
      capacity_cv_.notify_all();
      drain_cv_.wait(lock, [&] {
        return outstanding_ == 0;
      });
      reaper_stop_ = true;
    }
    if (reaper_.joinable()) reaper_.join();
    if (context_ != 0) {
      ::io_destroy(context_);
      context_ = 0;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  static auto direct_constraints(const std::filesystem::path &path, std::uint32_t depth)
      -> ReadConstraints {
    std::size_t memory = 4096;
    std::size_t offset = 4096;
#if defined(STATX_DIOALIGN) && defined(AT_STATX_DONT_SYNC)
    struct statx info{};
    if (::statx(AT_FDCWD, path.c_str(), AT_STATX_DONT_SYNC, STATX_DIOALIGN, &info) == 0 &&
        (info.stx_mask & STATX_DIOALIGN) != 0) {
      if (info.stx_dio_mem_align != 0) memory = info.stx_dio_mem_align;
      if (info.stx_dio_offset_align != 0) offset = info.stx_dio_offset_align;
    }
#endif
    return {memory, offset, offset, depth, true};
  }

  void reap_loop() noexcept {
    std::vector<io_event> events(constraints_.max_batch);
    for (;;) {
      timespec timeout{0, 10'000'000};
      const int count = ::io_getevents(context_,
                                       0,
                                       static_cast<long>(events.size()),  // NOLINT(runtime/int)
                                       events.data(),
                                       &timeout);
      if (count > 0) {
        for (int i = 0; i < count; ++i) complete(events[static_cast<std::size_t>(i)]);
      }
      std::lock_guard lock(mutex_);
      if (reaper_stop_ && outstanding_ == 0) return;
    }
  }

  void complete(const io_event &event) noexcept {
    // io_getevents() can publish a completion immediately after io_submit().
    // Pair with submit()'s mutex-held ownership hand-off before adopting or
    // reading RequestState: otherwise the reaper can race the submitter's
    // unique_ptr release (and, on a partial submit, suppress_completion).
    std::unique_lock lock(mutex_);
    std::unique_ptr<RequestState> state(static_cast<RequestState *>(event.data));
    lock.unlock();
    ReadResult result{.id = state->request.id};
    if (event.res < 0) {
      result.status = ReadStatus::io_error;
      result.error = {-static_cast<int>(event.res), std::generic_category()};
    } else {
      result.bytes = static_cast<std::size_t>(event.res);
      if (state->batch->cancel_requested.load(std::memory_order_acquire))
        result.status = ReadStatus::cancelled;
      else if (Clock::now() > state->request.deadline)
        result.status = ReadStatus::timed_out;
      else
        result.status =
            result.bytes == state->request.buffer.size() ? ReadStatus::ok : ReadStatus::short_read;
    }
    state->batch->remaining.fetch_sub(1, std::memory_order_acq_rel);
    if (!state->suppress_completion) state->completion.fn(state->completion.context, result);
    lock.lock();
    --outstanding_;
    lock.unlock();
    capacity_cv_.notify_all();
    drain_cv_.notify_all();
  }

  int fd_ = -1;
  io_context_t context_ = 0;
  ReadConstraints constraints_{};
  std::thread reaper_;
  std::mutex mutex_;
  std::condition_variable capacity_cv_;
  std::condition_variable drain_cv_;
  std::size_t outstanding_ = 0;
  bool stopping_ = false;
  bool reaper_stop_ = false;
};

}  // namespace alaya::storage::io
