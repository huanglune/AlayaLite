// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file threadpool_file_reader.hpp
 * @brief Portable AlignedFileReader backend built from pread and worker threads.
 *
 * This AlayaDB.AI implementation follows the AlignedFileReader interface
 * lineage originating in Microsoft DiskANN; the backend implementation is
 * original to AlayaLite and remains AGPL-3.0-only.
 */

#pragma once

#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>

#include "concurrentqueue.h"  // NOLINT
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "utils/thread_config.hpp"

struct ThreadPoolContext {
  moodycamel::ConcurrentQueue<AlignedReadEvent> completions;
  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  std::atomic<bool> active{true};
};

class ThreadPoolFileReader : public AlignedFileReader {
 private:
  struct SubmittedRead {
    AlignedRead read;
    ThreadPoolContext *ctx = nullptr;
  };

  int file_desc_ = -1;
  std::map<std::thread::id, std::unique_ptr<ThreadPoolContext>> owned_contexts_;
  moodycamel::ConcurrentQueue<SubmittedRead> submitted_reads_;
  std::vector<std::thread> workers_;
  std::atomic<bool> stop_{false};
  std::mutex submit_mutex_;
  std::condition_variable submit_cv_;

  static size_t worker_count_from_env();
  void worker_loop();
  void notify_completion(ThreadPoolContext *ctx, const AlignedReadEvent &event);

 public:
  ThreadPoolFileReader() = default;
  ~ThreadPoolFileReader() override { close(); }

  IOContext &get_ctx() override;
  void register_thread() override;
  void deregister_thread() override;
  void deregister_all_threads() override;

  void open(const std::string &fname) override;
  void close() override;

  void read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async = false) override;
  int submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) override;
  int get_events(IOContext &ctx, int n_ops, std::vector<AlignedReadEvent> &out) override;
  int poll_events(IOContext &ctx, int max_events, std::vector<AlignedReadEvent> &out) override;
};

inline size_t ThreadPoolFileReader::worker_count_from_env() {
  const size_t hardware = ::alaya::system_thread_count();
  size_t worker_count =
      std::min(static_cast<size_t>(MAX_IO_DEPTH), static_cast<size_t>(2U) * hardware);

  const char *env = std::getenv("ALAYA_LASER_IO_THREADS");
  if (env == nullptr || *env == '\0') {
    return worker_count;
  }

  char *end = nullptr;
  errno = 0;
  const auto requested = std::strtoull(env, &end, 10);
  if (errno != 0 || end == env || *end != '\0' || requested == 0) {
    std::cerr << "ThreadPoolFileReader: ignoring invalid ALAYA_LASER_IO_THREADS=" << env
              << std::endl;
    return worker_count;
  }
  return std::min(static_cast<size_t>(MAX_IO_DEPTH), static_cast<size_t>(requested));
}

inline IOContext &ThreadPoolFileReader::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut_);
  auto it = ctx_map_.find(std::this_thread::get_id());
  if (it == ctx_map_.end()) {
    throw std::runtime_error(
        "ThreadPoolFileReader::get_ctx: calling thread is not registered "
        "(call register_thread() first)");
  }
  return it->second;
}

inline void ThreadPoolFileReader::register_thread() {
  const auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut_);
  if (ctx_map_.find(my_id) != ctx_map_.end()) {
    throw std::runtime_error("ThreadPoolFileReader::register_thread: thread is already registered");
  }

  auto ctx = std::make_unique<ThreadPoolContext>();
  IOContext raw_ctx = ctx.get();
  owned_contexts_[my_id] = std::move(ctx);
  ctx_map_[my_id] = raw_ctx;
}

inline void ThreadPoolFileReader::deregister_thread() {
  const auto my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut_);
  auto ctx_it = ctx_map_.find(my_id);
  if (ctx_it == ctx_map_.end()) {
    return;
  }
  if (ctx_it->second != nullptr) {
    ctx_it->second->active.store(false, std::memory_order_release);
    ctx_it->second->completion_cv.notify_all();
  }
  ctx_map_.erase(ctx_it);
  owned_contexts_.erase(my_id);
}

inline void ThreadPoolFileReader::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut_);
  for (auto &entry : ctx_map_) {
    if (entry.second != nullptr) {
      entry.second->active.store(false, std::memory_order_release);
      entry.second->completion_cv.notify_all();
    }
  }
  ctx_map_.clear();
  owned_contexts_.clear();
}

inline void ThreadPoolFileReader::open(const std::string &fname) {
  close();

  file_desc_ = ::open(fname.c_str(), O_RDONLY);
  if (file_desc_ == -1) {
    throw std::runtime_error("ThreadPoolFileReader::open: open() failed for " + fname +
                             ", errno=" + std::to_string(errno) + "=" + ::strerror(errno));
  }

  stop_.store(false, std::memory_order_release);
  const size_t worker_count = worker_count_from_env();
  workers_.reserve(worker_count);
  for (size_t i = 0; i < worker_count; ++i) {
    workers_.emplace_back([this]() {
      worker_loop();
    });
  }
}

inline void ThreadPoolFileReader::close() {
  stop_.store(true, std::memory_order_release);
  submit_cv_.notify_all();
  for (auto &worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();

  SubmittedRead ignored;
  while (submitted_reads_.try_dequeue(ignored)) {
  }

  if (file_desc_ != -1) {
    const int ret = ::close(file_desc_);
    if (ret == -1) {
      std::cerr << "ThreadPoolFileReader::close: close() failed; errno=" << errno << ":"
                << ::strerror(errno) << std::endl;
    }
    file_desc_ = -1;
  }
  deregister_all_threads();
}

inline void ThreadPoolFileReader::read(std::vector<AlignedRead> &read_reqs,
                                       IOContext &ctx,
                                       bool async) {
  const int submitted = submit_reqs(read_reqs, ctx);
  if (async) {
    return;
  }

  std::vector<AlignedReadEvent> ignored;
  const int completed = get_events(ctx, submitted, ignored);
  if (completed != submitted) {
    throw std::runtime_error("ThreadPoolFileReader::read: completed " + std::to_string(completed) +
                             " reads, expected " + std::to_string(submitted));
  }
}

inline int ThreadPoolFileReader::submit_reqs(std::vector<AlignedRead> &read_reqs, IOContext &ctx) {
  if (file_desc_ == -1) {
    throw std::runtime_error("ThreadPoolFileReader::submit_reqs: reader is not open");
  }
  if (ctx == nullptr) {
    throw std::runtime_error("ThreadPoolFileReader::submit_reqs: invalid thread context");
  }
  if (read_reqs.size() > MAX_EVENTS) {
    throw std::runtime_error("ThreadPoolFileReader::submit_reqs: request count " +
                             std::to_string(read_reqs.size()) +
                             " exceeds MAX_EVENTS=" + std::to_string(MAX_EVENTS));
  }

  for (const auto &read : read_reqs) {
    submitted_reads_.enqueue(SubmittedRead{read, ctx});
  }
  submit_cv_.notify_all();
  return static_cast<int>(read_reqs.size());
}

inline int ThreadPoolFileReader::get_events(IOContext &ctx,
                                            int n_ops,
                                            std::vector<AlignedReadEvent> &out) {
  out.clear();
  if (n_ops <= 0) {
    return 0;
  }
  if (ctx == nullptr) {
    throw std::runtime_error("ThreadPoolFileReader::get_events: invalid thread context");
  }

  out.reserve(static_cast<size_t>(n_ops));
  while (static_cast<int>(out.size()) < n_ops) {
    AlignedReadEvent event;
    while (static_cast<int>(out.size()) < n_ops && ctx->completions.try_dequeue(event)) {
      out.push_back(event);
    }
    if (static_cast<int>(out.size()) >= n_ops) {
      break;
    }

    std::unique_lock<std::mutex> lk(ctx->completion_mutex);
    ctx->completion_cv.wait_for(lk, std::chrono::microseconds{50});
  }
  return static_cast<int>(out.size());
}

inline int ThreadPoolFileReader::poll_events(IOContext &ctx,
                                             int max_events,
                                             std::vector<AlignedReadEvent> &out) {
  out.clear();
  if (max_events <= 0) {
    return 0;
  }
  if (ctx == nullptr) {
    throw std::runtime_error("ThreadPoolFileReader::poll_events: invalid thread context");
  }

  out.resize(static_cast<size_t>(max_events));
  const size_t count =
      ctx->completions.try_dequeue_bulk(out.data(), static_cast<size_t>(max_events));
  out.resize(count);
  return static_cast<int>(count);
}

inline void ThreadPoolFileReader::worker_loop() {
  while (true) {
    SubmittedRead submitted;
    if (submitted_reads_.try_dequeue(submitted)) {
      int64_t result = 0;
      errno = 0;
      const ssize_t bytes_read = ::pread(file_desc_,
                                         submitted.read.buf,
                                         static_cast<size_t>(submitted.read.len),
                                         static_cast<off_t>(submitted.read.offset));
      if (bytes_read == -1) {
        result = -static_cast<int64_t>(errno);
      } else {
        result = static_cast<int64_t>(bytes_read);
      }
      notify_completion(submitted.ctx, AlignedReadEvent{submitted.read.id, result});
      continue;
    }

    if (stop_.load(std::memory_order_acquire)) {
      break;
    }

    std::unique_lock<std::mutex> lk(submit_mutex_);
    submit_cv_.wait_for(lk, std::chrono::microseconds{50});
  }
}

inline void ThreadPoolFileReader::notify_completion(ThreadPoolContext *ctx,
                                                    const AlignedReadEvent &event) {
  if (ctx == nullptr || !ctx->active.load(std::memory_order_acquire)) {
    return;
  }
  ctx->completions.enqueue(event);
  ctx->completion_cv.notify_one();
}
