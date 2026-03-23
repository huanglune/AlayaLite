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
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "io_engine.hpp"
#include "utils/log.hpp"
#include "utils/platform.hpp"

#ifdef ALAYA_OS_LINUX
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
  #include "io_uring_engine.hpp"
#elif defined(ALAYA_OS_MACOS)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#elif defined(ALAYA_OS_WINDOWS)
  #include <io.h>
  #include <windows.h>
#endif

namespace alaya {

// ============================================================================
// Constants
// ============================================================================

/// Default sector size for alignment (4KB)
constexpr size_t kDefaultSectorSize = 4096;

// ============================================================================
// Factory function implementation
// ============================================================================

/**
 * @brief Create the best available I/O engine for this platform.
 * @return Unique pointer to an IOEngine instance
 */
inline auto create_io_engine() -> std::unique_ptr<IOEngine> {
#ifdef ALAYA_OS_LINUX
  if (IOUringEngine::is_available()) {
    try {
      auto engine = std::make_unique<IOUringEngine>();
      LOG_INFO("Using io_uring I/O engine");
      return engine;
    } catch (const std::exception &e) {
      LOG_WARN("Failed to initialize io_uring: {}, falling back to sync", e.what());
    }
  }
#endif
  LOG_INFO("Using synchronous I/O engine");
  return std::make_unique<SyncEngine>();
}

// ============================================================================
// DirectFileIO - High-level file I/O interface
// ============================================================================

/**
 * @brief Direct file I/O with automatic backend selection.
 *
 * DirectFileIO provides both synchronous and asynchronous I/O operations.
 * On Linux with kernel 5.1+, it uses io_uring for high-performance async I/O.
 * On other platforms, it falls back to synchronous Direct I/O.
 *
 * All operations support Direct I/O mode for bypassing the page cache.
 * Buffer addresses and file offsets must be sector-aligned (typically 4KB).
 *
 * Example usage:
 * @code
 * DirectFileIO file("/path/to/file", DirectFileIO::Mode::kRead);
 *
 * // Synchronous read
 * auto buf = AlignedAlloc<char, 4096>(4096);
 * file.read(buf.data(), 4096, 0);
 *
 * // Batch async read
 * std::vector<IORequest> requests = {...};
 * file.submit_reads(requests);
 * file.wait_completions(requests.size());
 * @endcode
 */
class DirectFileIO {
 public:
  /// File open mode
  enum class Mode : uint8_t {
    kRead = 0,       ///< Open for reading only
    kWrite = 1,      ///< Open for writing only
    kReadWrite = 2,  ///< Open for reading and writing
  };

  /**
   * @brief Default constructor - creates closed reader.
   */
  DirectFileIO() : engine_(create_io_engine()) {}

  /**
   * @brief Construct and open a file.
   * @param path File path to open
   * @param mode Open mode (read, write, or read-write)
   * @throws std::runtime_error if file cannot be opened
   */
  explicit DirectFileIO(std::string_view path, Mode mode = Mode::kRead)
      : engine_(create_io_engine()) {
    if (!open(path, mode)) {
      throw std::runtime_error("Failed to open file: " + std::string(path));
    }
  }

  ~DirectFileIO() { close(); }

  // Non-copyable
  DirectFileIO(const DirectFileIO &) = delete;
  auto operator=(const DirectFileIO &) -> DirectFileIO & = delete;

  // Movable
  DirectFileIO(DirectFileIO &&other) noexcept
      : fd_(other.fd_),
        path_(std::move(other.path_)),
        file_size_(other.file_size_),
        sector_size_(other.sector_size_),
        bypass_page_cache_(other.bypass_page_cache_),
        engine_(std::move(other.engine_)) {
    other.fd_ = -1;
    other.file_size_ = 0;
    other.bypass_page_cache_ = false;
  }

  auto operator=(DirectFileIO &&other) noexcept -> DirectFileIO & {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      path_ = std::move(other.path_);
      file_size_ = other.file_size_;
      sector_size_ = other.sector_size_;
      bypass_page_cache_ = other.bypass_page_cache_;
      engine_ = std::move(other.engine_);
      other.fd_ = -1;
      other.file_size_ = 0;
      other.bypass_page_cache_ = false;
    }
    return *this;
  }

  // ==========================================================================
  // File Operations
  // ==========================================================================

  /**
   * @brief Open a file.
   * @param path File path
   * @param mode Open mode
   * @return true if successful
   */
  auto open(std::string_view path, Mode mode = Mode::kRead) -> bool {
    if (is_open()) {
      close();
    }

    path_ = std::string(path);

#ifdef ALAYA_OS_LINUX
    return open_linux(mode);
#elif defined(ALAYA_OS_MACOS)
    return open_macos(mode);
#elif defined(ALAYA_OS_WINDOWS)
    return open_windows(mode);
#else
    return open_fallback(mode);
#endif
  }

  /**
   * @brief Close the file.
   */
  void close() {
    if (fd_ >= 0) {
#ifdef ALAYA_OS_WINDOWS
      _close(fd_);
#else
      ::close(fd_);
#endif
      fd_ = -1;
    }
    path_.clear();
    file_size_ = 0;
    bypass_page_cache_ = false;
  }

  /**
   * @brief Check if file is open.
   * @return true if open
   */
  [[nodiscard]] auto is_open() const -> bool { return fd_ >= 0; }

  /**
   * @brief Get the file path.
   * @return File path
   */
  [[nodiscard]] auto path() const -> const std::string & { return path_; }

  /**
   * @brief Get the file size.
   * @return File size in bytes
   */
  [[nodiscard]] auto file_size() const -> uint64_t { return file_size_; }

  /**
   * @brief Get the sector size for alignment.
   * @return Sector size in bytes
   */
  [[nodiscard]] auto sector_size() const -> size_t { return sector_size_; }
  [[nodiscard]] auto bypasses_page_cache() const -> bool { return bypass_page_cache_; }

  // ==========================================================================
  // Synchronous I/O
  // ==========================================================================

  /**
   * @brief Synchronous read at offset.
   * @param buffer Destination buffer (must be aligned for Direct IO)
   * @param size Bytes to read
   * @param offset File offset
   * @return Bytes read, or -1 on error
   */
  auto read(void *buffer, size_t size, uint64_t offset) -> ssize_t {
    if (!is_open()) {
      return -1;
    }
    return engine_->pread(fd_, buffer, size, offset);
  }

  /**
   * @brief Synchronous write at offset.
   * @param buffer Source buffer (must be aligned for Direct IO)
   * @param size Bytes to write
   * @param offset File offset
   * @return Bytes written, or -1 on error
   */
  auto write(const void *buffer, size_t size, uint64_t offset) -> ssize_t {
    if (!is_open()) {
      return -1;
    }
    return engine_->pwrite(fd_, buffer, size, offset);
  }

  // ==========================================================================
  // Async Batch I/O
  // ==========================================================================

  /**
   * @brief Submit batch read requests and wait for completion.
   * @param requests Span of IORequest to submit
   * @return Number of requests completed successfully
   */
  auto submit_reads(std::span<IORequest> requests) -> size_t {
    if (!is_open() || requests.empty()) {
      return 0;
    }
    size_t submitted = engine_->submit_reads(fd_, requests);
    if (submitted > 0 && engine_->supports_async()) {
      engine_->wait(submitted, -1);
    }
    return submitted;
  }

  /**
   * @brief Submit batch write requests and wait for completion.
   * @param requests Span of IORequest to submit
   * @return Number of requests completed successfully
   */
  auto submit_writes(std::span<IORequest> requests) -> size_t {
    if (!is_open() || requests.empty()) {
      return 0;
    }
    size_t submitted = engine_->submit_writes(fd_, requests);
    if (submitted > 0 && engine_->supports_async()) {
      engine_->wait(submitted, -1);
    }
    return submitted;
  }

  /**
   * @brief Wait for request completions.
   * @param min_complete Minimum completions to wait for
   * @param timeout_ms Timeout in milliseconds (-1 for infinite)
   * @return Number of completed requests
   */
  auto wait_completions(size_t min_complete = 1, int timeout_ms = -1) -> size_t {
    return engine_->wait(min_complete, timeout_ms);
  }

  // ==========================================================================
  // Alignment Helpers
  // ==========================================================================

  /**
   * @brief Round up value to sector alignment.
   * @param value Value to align
   * @return Aligned value
   */
  [[nodiscard]] auto align_up(uint64_t value) const -> uint64_t {
    return (value + sector_size_ - 1) & ~(sector_size_ - 1);
  }

  /**
   * @brief Round down value to sector alignment.
   * @param value Value to align
   * @return Aligned value
   */
  [[nodiscard]] auto align_down(uint64_t value) const -> uint64_t {
    return value & ~(sector_size_ - 1);
  }

  /**
   * @brief Check if pointer is sector-aligned.
   * @param ptr Pointer to check
   * @return true if aligned
   */
  [[nodiscard]] auto is_aligned(const void *ptr) const -> bool {
    return (reinterpret_cast<uintptr_t>(ptr) % sector_size_) == 0;
  }

  /**
   * @brief Check if offset is sector-aligned.
   * @param offset Offset to check
   * @return true if aligned
   */
  [[nodiscard]] auto is_aligned(uint64_t offset) const -> bool {
    return (offset % sector_size_) == 0;
  }

  // ==========================================================================
  // Engine Info
  // ==========================================================================

  /**
   * @brief Get the I/O engine name.
   * @return Engine name string
   */
  [[nodiscard]] auto engine_name() const -> std::string_view { return engine_->name(); }

  /**
   * @brief Check if async operations are truly async.
   * @return true if using io_uring
   */
  [[nodiscard]] auto supports_async() const -> bool { return engine_->supports_async(); }

 private:
  int fd_{-1};
  std::string path_;
  uint64_t file_size_{0};
  size_t sector_size_{kDefaultSectorSize};
  bool bypass_page_cache_{false};
  std::unique_ptr<IOEngine> engine_;

  // ==========================================================================
  // Platform-specific open implementations
  // ==========================================================================

#ifdef ALAYA_OS_LINUX
  auto open_linux(Mode mode) -> bool {
    int flags = get_posix_flags(mode) | O_DIRECT;

    fd_ = ::open(path_.c_str(), flags, 0644);
    if (fd_ < 0) {
      // Try without O_DIRECT
      flags &= ~O_DIRECT;
      fd_ = ::open(path_.c_str(), flags, 0644);
      if (fd_ < 0) {
        LOG_ERROR("Failed to open file: {} (errno={})", path_, errno);
        return false;
      }
      bypass_page_cache_ = false;
      LOG_WARN("Opened {} without Direct IO", path_);
    } else {
      bypass_page_cache_ = true;
    }

    update_file_size();
    LOG_INFO("Opened {} (fd={}, size={}, engine={})", path_, fd_, file_size_, engine_->name());
    return true;
  }
#endif

#ifdef ALAYA_OS_MACOS
  auto open_macos(Mode mode) -> bool {
    int flags = get_posix_flags(mode);

    fd_ = ::open(path_.c_str(), flags, 0644);
    if (fd_ < 0) {
      LOG_ERROR("Failed to open file: {} (errno={})", path_, errno);
      return false;
    }

    // Enable F_NOCACHE for Direct IO behavior
    if (fcntl(fd_, F_NOCACHE, 1) == -1) {
      bypass_page_cache_ = false;
      LOG_WARN("Failed to set F_NOCACHE on {}", path_);
    } else {
      bypass_page_cache_ = true;
    }

    update_file_size();
    LOG_INFO("Opened {} (fd={}, size={})", path_, fd_, file_size_);
    return true;
  }
#endif

#ifdef ALAYA_OS_WINDOWS
  auto open_windows(Mode mode) -> bool {
    DWORD access = 0;
    DWORD creation = OPEN_EXISTING;
    DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING;

    switch (mode) {
      case Mode::kRead:
        access = GENERIC_READ;
        break;
      case Mode::kWrite:
        access = GENERIC_WRITE;
        creation = CREATE_ALWAYS;
        break;
      case Mode::kReadWrite:
        access = GENERIC_READ | GENERIC_WRITE;
        creation = OPEN_ALWAYS;
        break;
    }

    HANDLE handle =
        CreateFileA(path_.c_str(), access, FILE_SHARE_READ, nullptr, creation, flags, nullptr);

    if (handle == INVALID_HANDLE_VALUE) {
      // Try without FILE_FLAG_NO_BUFFERING
      flags = FILE_ATTRIBUTE_NORMAL;
      handle =
          CreateFileA(path_.c_str(), access, FILE_SHARE_READ, nullptr, creation, flags, nullptr);
      if (handle == INVALID_HANDLE_VALUE) {
        LOG_ERROR("Failed to open file: {} (error={})", path_, GetLastError());
        return false;
      }
      bypass_page_cache_ = false;
    } else {
      bypass_page_cache_ = true;
    }

    fd_ = _open_osfhandle(reinterpret_cast<intptr_t>(handle), 0);
    if (fd_ < 0) {
      CloseHandle(handle);
      return false;
    }

    update_file_size();
    LOG_INFO("Opened {} (fd={}, size={})", path_, fd_, file_size_);
    return true;
  }
#endif

  auto open_fallback(Mode mode) -> bool {
    int flags = get_posix_flags(mode);

    fd_ = ::open(path_.c_str(), flags, 0644);
    if (fd_ < 0) {
      LOG_ERROR("Failed to open file: {} (errno={})", path_, errno);
      return false;
    }

    update_file_size();
    LOG_WARN("Opened {} without Direct IO support", path_);
    bypass_page_cache_ = false;
    return true;
  }

  [[nodiscard]] static auto get_posix_flags(Mode mode) -> int {
    switch (mode) {
      case Mode::kRead:
        return O_RDONLY;
      case Mode::kWrite:
        return O_WRONLY | O_CREAT | O_TRUNC;
      case Mode::kReadWrite:
        return O_RDWR | O_CREAT;
    }
    return O_RDONLY;
  }

  void update_file_size() {
#ifdef ALAYA_OS_WINDOWS
    HANDLE handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
    LARGE_INTEGER size;
    if (GetFileSizeEx(handle, &size)) {
      file_size_ = static_cast<uint64_t>(size.QuadPart);
    }
#else
    struct stat st{};
    if (fstat(fd_, &st) == 0) {
      file_size_ = static_cast<uint64_t>(st.st_size);
    }
#endif
  }
};

}  // namespace alaya
