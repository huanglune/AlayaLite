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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>

#include "utils/macros.hpp"
#include "utils/platform.hpp"

#ifdef ALAYA_OS_LINUX
  #include <unistd.h>
#elif defined(ALAYA_OS_WINDOWS)
  #include <windows.h>
#else
  #include <unistd.h>
#endif

namespace alaya {

// ============================================================================
// IORequest - Simplified I/O request structure
// ============================================================================

/**
 * @brief I/O request for batch operations.
 */
struct IORequest {
  void *buffer_{nullptr};     ///< Buffer for data transfer (must be aligned)
  size_t size_{0};            ///< Number of bytes to read/write
  uint64_t offset_{0};        ///< File offset
  int32_t result_{0};         ///< Bytes transferred or negative error code
  void *user_data_{nullptr};  ///< User-provided context

  IORequest() = default;

  IORequest(void *buf, size_t sz, uint64_t off, void *ud = nullptr)
      : buffer_(buf), size_(sz), offset_(off), user_data_(ud) {}

  [[nodiscard]] auto is_success() const -> bool {
    return result_ > 0 && static_cast<size_t>(result_) == size_;
  }
};

// ============================================================================
// IOEngine - Abstract base class for I/O engines
// ============================================================================

/**
 * @brief Abstract I/O engine interface.
 *
 * Provides both synchronous and asynchronous I/O operations.
 * Implementations can use different backends (pread/pwrite, io_uring, etc.)
 */
class IOEngine {
 public:
  virtual ~IOEngine() = default;
  IOEngine() = default;
  ALAYA_NON_COPYABLE_NON_MOVABLE(IOEngine);

  /**
   * @brief Synchronous positioned read.
   * @param fd File descriptor
   * @param buf Destination buffer (must be aligned for Direct IO)
   * @param size Number of bytes to read
   * @param offset File offset
   * @return Bytes read, or -1 on error
   */
  virtual auto pread(int fd, void *buf, size_t size, uint64_t offset) -> ssize_t = 0;

  /**
   * @brief Synchronous positioned write.
   * @param fd File descriptor
   * @param buf Source buffer (must be aligned for Direct IO)
   * @param size Number of bytes to write
   * @param offset File offset
   * @return Bytes written, or -1 on error
   */
  virtual auto pwrite(int fd, const void *buf, size_t size, uint64_t offset) -> ssize_t = 0;

  /**
   * @brief Submit batch of read requests.
   * @param fd File descriptor
   * @param requests Span of IORequest to submit
   * @return Number of requests successfully submitted
   */
  virtual auto submit_reads(int fd, std::span<IORequest> requests) -> size_t = 0;

  /**
   * @brief Submit batch of write requests.
   * @param fd File descriptor
   * @param requests Span of IORequest to submit
   * @return Number of requests successfully submitted
   */
  virtual auto submit_writes(int fd, std::span<IORequest> requests) -> size_t = 0;

  /**
   * @brief Wait for completion of submitted requests.
   * @param min_complete Minimum number of completions to wait for
   * @param timeout_ms Timeout in milliseconds (-1 for infinite)
   * @return Number of completed requests
   */
  virtual auto wait(size_t min_complete, int timeout_ms) -> size_t = 0;

  /**
   * @brief Check if engine supports true async I/O.
   * @return true if async operations are non-blocking
   */
  [[nodiscard]] virtual auto supports_async() const -> bool = 0;

  /**
   * @brief Get engine name for logging/debugging.
   * @return Engine name string
   */
  [[nodiscard]] virtual auto name() const -> std::string_view = 0;
};

// ============================================================================
// SyncEngine - Synchronous fallback implementation
// ============================================================================

/**
 * @brief Synchronous I/O engine using pread/pwrite.
 *
 * This is the fallback engine used on platforms without io_uring support.
 * All "async" operations complete synchronously.
 */
class SyncEngine final : public IOEngine {
 public:
  SyncEngine() = default;
  ~SyncEngine() override = default;

  auto pread(int fd, void *buf, size_t size, uint64_t offset) -> ssize_t override {
#ifdef ALAYA_OS_WINDOWS
    return pread_windows(fd, buf, size, offset);
#else
    return ::pread(fd, buf, size, static_cast<off_t>(offset));
#endif
  }

  auto pwrite(int fd, const void *buf, size_t size, uint64_t offset) -> ssize_t override {
#ifdef ALAYA_OS_WINDOWS
    return pwrite_windows(fd, buf, size, offset);
#else
    return ::pwrite(fd, buf, size, static_cast<off_t>(offset));
#endif
  }

  auto submit_reads(int fd, std::span<IORequest> requests) -> size_t override {
    size_t completed = 0;
    for (auto &req : requests) {
      auto bytes = pread(fd, req.buffer_, req.size_, req.offset_);
      req.result_ = static_cast<int32_t>(bytes);
      if (bytes > 0) {
        ++completed;
      }
    }
    return completed;
  }

  auto submit_writes(int fd, std::span<IORequest> requests) -> size_t override {
    size_t completed = 0;
    for (auto &req : requests) {
      auto bytes = pwrite(fd, req.buffer_, req.size_, req.offset_);
      req.result_ = static_cast<int32_t>(bytes);
      if (bytes > 0) {
        ++completed;
      }
    }
    return completed;
  }

  auto wait([[maybe_unused]] size_t min_complete, [[maybe_unused]] int timeout_ms)
      -> size_t override {
    // All operations complete synchronously in submit_*
    return 0;
  }

  [[nodiscard]] auto supports_async() const -> bool override { return false; }

  [[nodiscard]] auto name() const -> std::string_view override { return "sync"; }

 private:
#ifdef ALAYA_OS_WINDOWS
  static auto pread_windows(int fd, void *buf, size_t size, uint64_t offset) -> ssize_t {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
      return -1;
    }

    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

    DWORD bytes_read = 0;
    if (!ReadFile(handle, buf, static_cast<DWORD>(size), &bytes_read, &overlapped)) {
      if (GetLastError() != ERROR_HANDLE_EOF) {
        return -1;
      }
    }
    return static_cast<ssize_t>(bytes_read);
  }

  static auto pwrite_windows(int fd, const void *buf, size_t size, uint64_t offset) -> ssize_t {
    auto handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (handle == INVALID_HANDLE_VALUE) {
      return -1;
    }

    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

    DWORD bytes_written = 0;
    if (!WriteFile(handle, buf, static_cast<DWORD>(size), &bytes_written, &overlapped)) {
      return -1;
    }
    return static_cast<ssize_t>(bytes_written);
  }
#endif
};

// Forward declaration for factory function
auto create_io_engine() -> std::unique_ptr<IOEngine>;

}  // namespace alaya
