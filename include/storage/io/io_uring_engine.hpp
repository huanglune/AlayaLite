/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "io_engine.hpp"
#include "utils/log.hpp"
#include "utils/platform.hpp"

#ifdef ALAYA_OS_LINUX

  #include <liburing.h>

  #include <cerrno>
  #include <cstring>
  #include <stdexcept>
  #include <vector>

namespace alaya {

// ============================================================================
// IOUringEngine - High-performance async I/O using io_uring
// ============================================================================

/**
 * @brief I/O engine using Linux io_uring for async operations.
 *
 * io_uring provides high-performance async I/O with minimal syscall overhead.
 * Falls back to synchronous operations if io_uring is not available.
 */
class IOUringEngine final : public IOEngine {
 public:
  /// Default submission queue depth
  static constexpr size_t kDefaultQueueDepth = 128;

  /**
   * @brief Construct IOUringEngine with specified queue depth.
   * @param queue_depth Size of submission/completion queues
   * @throws std::runtime_error if io_uring initialization fails
   */
  explicit IOUringEngine(size_t queue_depth = kDefaultQueueDepth) : queue_depth_(queue_depth) {
    if (!is_available()) {
      throw std::runtime_error("io_uring is not available on this system");
    }
    init_ring();
  }

  ~IOUringEngine() override { cleanup_ring(); }

  // ---- Synchronous operations (delegate to pread/pwrite for simplicity) ----

  auto pread(int fd, void *buf, size_t size, uint64_t offset) -> ssize_t override {
    return ::pread(fd, buf, size, static_cast<off_t>(offset));
  }

  auto pwrite(int fd, const void *buf, size_t size, uint64_t offset) -> ssize_t override {
    return ::pwrite(fd, buf, size, static_cast<off_t>(offset));
  }

  // ---- Async batch operations ----

  auto submit_reads(int fd, std::span<IORequest> requests) -> size_t override {
    return submit_batch(fd, requests, false);
  }

  auto submit_writes(int fd, std::span<IORequest> requests) -> size_t override {
    return submit_batch(fd, requests, true);
  }

  auto wait(size_t min_complete, int timeout_ms) -> size_t override {
    if (!initialized_) {
      return 0;
    }

    size_t completed = 0;

    if (timeout_ms < 0) {
      // Blocking wait for min_complete requests
      while (completed < min_complete) {
        struct io_uring_cqe *cqe = nullptr;
        int ret = io_uring_wait_cqe(&ring_, &cqe);
        if (ret < 0) {
          LOG_ERROR("io_uring_wait_cqe failed: {}", strerror(-ret));
          break;
        }

        process_cqe(cqe);
        io_uring_cqe_seen(&ring_, cqe);
        ++completed;
      }
    } else {
      // Non-blocking or timed wait
      struct __kernel_timespec ts = {.tv_sec = timeout_ms / 1000,
                                     .tv_nsec = (timeout_ms % 1000) * 1000000L};

      while (completed < min_complete) {
        struct io_uring_cqe *cqe = nullptr;
        int ret = io_uring_wait_cqe_timeout(&ring_, &cqe, &ts);
        if (ret == -ETIME) {
          break;  // Timeout
        }
        if (ret < 0) {
          LOG_ERROR("io_uring_wait_cqe_timeout failed: {}", strerror(-ret));
          break;
        }

        process_cqe(cqe);
        io_uring_cqe_seen(&ring_, cqe);
        ++completed;
      }
    }

    // Drain any additional completions without blocking
    drain_completions(completed);

    return completed;
  }

  [[nodiscard]] auto supports_async() const -> bool override { return initialized_; }

  [[nodiscard]] auto name() const -> std::string_view override { return "io_uring"; }

  /**
   * @brief Check if io_uring is available on this system.
   * @return true if io_uring can be used
   */
  static auto is_available() -> bool {
    struct io_uring_probe *probe = io_uring_get_probe();
    if (probe == nullptr) {
      return false;
    }

    bool read_supported = io_uring_opcode_supported(probe, IORING_OP_READ) != 0;
    bool write_supported = io_uring_opcode_supported(probe, IORING_OP_WRITE) != 0;
    io_uring_free_probe(probe);

    return read_supported && write_supported;
  }

 private:
  struct io_uring ring_{};
  size_t queue_depth_;
  bool initialized_{false};
  std::vector<IORequest *> pending_requests_;

  void init_ring() {
    int ret = io_uring_queue_init(static_cast<unsigned>(queue_depth_), &ring_, 0);
    if (ret < 0) {
      throw std::runtime_error("io_uring_queue_init failed: " + std::string(strerror(-ret)));
    }

    initialized_ = true;
    pending_requests_.reserve(queue_depth_);
    LOG_INFO("IOUringEngine initialized with queue_depth={}", queue_depth_);
  }

  void cleanup_ring() {
    if (initialized_) {
      io_uring_queue_exit(&ring_);
      initialized_ = false;
    }
  }

  auto submit_batch(int fd, std::span<IORequest> requests, bool is_write) -> size_t {
    if (!initialized_ || requests.empty()) {
      return 0;
    }

    size_t prepared = 0;

    for (auto &req : requests) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring_);
      if (sqe == nullptr) {
        // SQ full, submit what we have and try to get more
        int submitted = io_uring_submit(&ring_);
        if (submitted < 0) {
          LOG_ERROR("io_uring_submit failed: {}", strerror(-submitted));
          break;
        }

        sqe = io_uring_get_sqe(&ring_);
        if (sqe == nullptr) {
          LOG_WARN("io_uring SQ still full after submit");
          break;
        }
      }

      if (is_write) {
        io_uring_prep_write(sqe, fd, req.buffer_, static_cast<unsigned>(req.size_), req.offset_);
      } else {
        io_uring_prep_read(sqe, fd, req.buffer_, static_cast<unsigned>(req.size_), req.offset_);
      }

      io_uring_sqe_set_data(sqe, &req);
      req.result_ = 0;  // Reset result
      ++prepared;
    }

    if (prepared > 0) {
      int submitted = io_uring_submit(&ring_);
      if (submitted < 0) {
        LOG_ERROR("io_uring_submit failed: {}", strerror(-submitted));
        return 0;
      }
      return static_cast<size_t>(submitted);
    }

    return 0;
  }

  void process_cqe(struct io_uring_cqe *cqe) {
    auto *req = static_cast<IORequest *>(io_uring_cqe_get_data(cqe));
    if (req != nullptr) {
      req->result_ = cqe->res;
    }
  }

  void drain_completions(size_t &completed) {
    struct io_uring_cqe *cqe = nullptr;
    while (io_uring_peek_cqe(&ring_, &cqe) == 0) {
      process_cqe(cqe);
      io_uring_cqe_seen(&ring_, cqe);
      ++completed;
    }
  }
};

}  // namespace alaya

#endif  // ALAYA_OS_LINUX
