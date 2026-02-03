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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "utils/log.hpp"
#include "utils/memory.hpp"

// Platform-specific includes for Direct IO
#ifdef ALAYA_OS_LINUX
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
#elif defined(ALAYA_OS_MACOS)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
#elif defined(ALAYA_OS_WINDOWS)
  #include <windows.h>
#endif

namespace alaya {

// ============================================================================
// Constants for aligned I/O
// ============================================================================

/// Default sector size for alignment (4KB, standard for most SSDs/HDDs)
constexpr size_t kDefaultSectorSize = 4096;

/// Maximum number of concurrent async I/O requests
constexpr size_t kMaxAsyncIODepth = 128;

// ============================================================================
// IOStatus - Result status for I/O operations
// ============================================================================

/**
 * @brief Enumeration of possible I/O operation results.
 */
enum class IOStatus : uint8_t {
  kSuccess = 0,       ///< Operation completed successfully
  kError = 1,         ///< Generic error
  kEOF = 2,           ///< End of file reached
  kNotAligned = 3,    ///< Buffer or offset not properly aligned
  kFileNotOpen = 4,   ///< File is not open
  kInvalidArg = 5,    ///< Invalid argument provided
  kPending = 6,       ///< Async operation is still pending
  kCancelled = 7,     ///< Operation was cancelled
  kNotSupported = 8,  ///< Operation not supported on this platform
};

/**
 * @brief Convert IOStatus to string for logging/debugging.
 */
inline auto io_status_to_string(IOStatus status) -> const char * {
  switch (status) {
    case IOStatus::kSuccess:
      return "Success";
    case IOStatus::kError:
      return "Error";
    case IOStatus::kEOF:
      return "EOF";
    case IOStatus::kNotAligned:
      return "NotAligned";
    case IOStatus::kFileNotOpen:
      return "FileNotOpen";
    case IOStatus::kInvalidArg:
      return "InvalidArg";
    case IOStatus::kPending:
      return "Pending";
    case IOStatus::kCancelled:
      return "Cancelled";
    case IOStatus::kNotSupported:
      return "NotSupported";
    default:
      return "Unknown";
  }
}

// ============================================================================
// IORequest - Request structure for async I/O operations
// ============================================================================

/**
 * @brief Request structure for asynchronous I/O operations.
 *
 * This structure encapsulates all information needed for an async I/O request.
 * It is designed to be compatible with future io_uring integration.
 */
struct IORequest {
  void *buffer_{nullptr};     ///< Aligned buffer for data transfer
  size_t size_{0};            ///< Number of bytes to read/write
  uint64_t offset_{0};        ///< File offset (must be sector-aligned for Direct IO)
  bool is_write_{false};      ///< True for write, false for read
  void *user_data_{nullptr};  ///< User-provided context pointer
  std::atomic<IOStatus> status_{IOStatus::kPending};  ///< Current status of the request
  ssize_t bytes_transferred_{0};                      ///< Bytes actually transferred

  IORequest() = default;

  IORequest(void *buf, size_t sz, uint64_t off, bool write = false, void *ud = nullptr)
      : buffer_(buf), size_(sz), offset_(off), is_write_(write), user_data_(ud) {}

  // std::atomic is not copyable, so we need custom copy/move operations
  IORequest(const IORequest &other)
      : buffer_(other.buffer_),
        size_(other.size_),
        offset_(other.offset_),
        is_write_(other.is_write_),
        user_data_(other.user_data_),
        status_(other.status_.load(std::memory_order_relaxed)),
        bytes_transferred_(other.bytes_transferred_) {}

  IORequest(IORequest &&other) noexcept
      : buffer_(other.buffer_),
        size_(other.size_),
        offset_(other.offset_),
        is_write_(other.is_write_),
        user_data_(other.user_data_),
        status_(other.status_.load(std::memory_order_relaxed)),
        bytes_transferred_(other.bytes_transferred_) {}

  auto operator=(const IORequest &other) -> IORequest & {
    if (this != &other) {
      buffer_ = other.buffer_;
      size_ = other.size_;
      offset_ = other.offset_;
      is_write_ = other.is_write_;
      user_data_ = other.user_data_;
      status_.store(other.status_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bytes_transferred_ = other.bytes_transferred_;
    }
    return *this;
  }

  auto operator=(IORequest &&other) noexcept -> IORequest & {
    if (this != &other) {
      buffer_ = other.buffer_;
      size_ = other.size_;
      offset_ = other.offset_;
      is_write_ = other.is_write_;
      user_data_ = other.user_data_;
      status_.store(other.status_.load(std::memory_order_relaxed), std::memory_order_relaxed);
      bytes_transferred_ = other.bytes_transferred_;
    }
    return *this;
  }

  /**
   * @brief Check if the request has completed.
   * @return true if completed (success or error), false if still pending
   */
  [[nodiscard]] auto is_complete() const -> bool {
    auto s = status_.load(std::memory_order_acquire);
    return s != IOStatus::kPending;
  }

  /**
   * @brief Check if the request completed successfully.
   * @return true if completed with success status
   */
  [[nodiscard]] auto is_success() const -> bool {
    return status_.load(std::memory_order_acquire) == IOStatus::kSuccess;
  }
};

// ============================================================================
// IOContext - Context for batch async I/O operations (io_uring preparation)
// ============================================================================

/**
 * @brief Context for managing batch asynchronous I/O operations.
 *
 * This class is designed as a placeholder for future io_uring integration.
 * It manages a pool of IORequest objects and provides batch submission/completion.
 */
class IOContext {
 public:
  using CompletionCallback = std::function<void(IORequest *)>;

 private:
  std::vector<IORequest> requests_;  ///< Pool of requests
  size_t max_depth_;                 ///< Maximum number of concurrent requests
  size_t pending_count_{0};          ///< Number of pending requests
  CompletionCallback callback_;      ///< Callback for completed requests

 public:
  explicit IOContext(size_t max_depth = kMaxAsyncIODepth) : max_depth_(max_depth) {
    requests_.reserve(max_depth);
  }

  /**
   * @brief Set the completion callback.
   * @param cb Callback function to invoke when a request completes
   */
  void set_callback(CompletionCallback cb) { callback_ = std::move(cb); }

  /**
   * @brief Get the maximum depth (number of concurrent requests).
   * @return Maximum depth
   */
  [[nodiscard]] auto max_depth() const -> size_t { return max_depth_; }

  /**
   * @brief Get the number of pending requests.
   * @return Number of pending requests
   */
  [[nodiscard]] auto pending_count() const -> size_t { return pending_count_; }

  /**
   * @brief Check if there is room for more requests.
   * @return true if more requests can be submitted
   */
  [[nodiscard]] auto can_submit() const -> bool { return pending_count_ < max_depth_; }

  /**
   * @brief Prepare a read request (placeholder for io_uring integration).
   *
   * @param buffer Aligned buffer to read into
   * @param size Number of bytes to read
   * @param offset File offset to read from
   * @param user_data Optional user context
   * @return Pointer to the prepared IORequest, or nullptr if queue is full
   */
  auto prepare_read(void *buffer, size_t size, uint64_t offset, void *user_data = nullptr)
      -> IORequest * {
    if (!can_submit()) {
      return nullptr;
    }
    requests_.emplace_back(buffer, size, offset, false, user_data);
    return &requests_.back();
  }

  /**
   * @brief Prepare a write request (placeholder for io_uring integration).
   *
   * @param buffer Aligned buffer to write from
   * @param size Number of bytes to write
   * @param offset File offset to write to
   * @param user_data Optional user context
   * @return Pointer to the prepared IORequest, or nullptr if queue is full
   */
  auto prepare_write(void *buffer, size_t size, uint64_t offset, void *user_data = nullptr)
      -> IORequest * {
    if (!can_submit()) {
      return nullptr;
    }
    requests_.emplace_back(buffer, size, offset, true, user_data);
    return &requests_.back();
  }

  /**
   * @brief Clear all requests from the context.
   */
  void clear() {
    requests_.clear();
    pending_count_ = 0;
  }

  /**
   * @brief Get access to the request pool.
   * @return Reference to the request vector
   */
  [[nodiscard]] auto requests() -> std::vector<IORequest> & { return requests_; }
  [[nodiscard]] auto requests() const -> const std::vector<IORequest> & { return requests_; }
};

// ============================================================================
// AlignedFileReader - Aligned file I/O with Direct IO support
// ============================================================================

/**
 * @brief File reader/writer with Direct IO support for DiskANN.
 *
 * This class provides aligned file I/O operations suitable for DiskANN's
 * disk-based index access patterns. It supports:
 *
 * - Direct IO (O_DIRECT on Linux) to bypass kernel page cache
 * - Aligned read/write operations (required for Direct IO)
 * - Synchronous positioned I/O (pread/pwrite)
 * - Placeholder interface for future async I/O (io_uring)
 *
 * All buffer addresses and file offsets must be aligned to the sector size
 * (typically 4KB) when using Direct IO mode.
 *
 * @note Direct IO is automatically enabled on Linux when the file is opened.
 *       On other platforms, a fallback to buffered I/O is used.
 */
class AlignedFileReader {
 public:
  /**
   * @brief File open mode flags.
   */
  enum class OpenMode : uint32_t {  // NOLINT
    kReadOnly = 1 << 0,             ///< Open for reading only
    kWriteOnly = 1 << 1,            ///< Open for writing only
    kReadWrite = 1 << 2,            ///< Open for reading and writing
    kCreate = 1 << 3,               ///< Create file if it doesn't exist
    kTruncate = 1 << 4,             ///< Truncate file to zero length
    kAppend = 1 << 5,               ///< Append to file
    kDirectIO = 1 << 6,             ///< Enable Direct IO (bypass page cache)
  };
  using OpenModeType = std::underlying_type_t<OpenMode>;  // get OpenMode type(uint32_t)

  /**
   * @brief Combine OpenMode flags using bitwise OR.
   */
  friend constexpr auto operator|(OpenMode a, OpenMode b) -> OpenMode {
    return static_cast<OpenMode>(static_cast<OpenModeType>(a) | static_cast<OpenModeType>(b));
  }

  /**
   * @brief Check if a flag is set in the mode.
   */
  static constexpr auto has_flag(OpenMode mode, OpenMode flag) -> bool {
    return (static_cast<OpenModeType>(mode) & static_cast<OpenModeType>(flag)) != 0;
  }

 private:
#ifdef ALAYA_OS_WINDOWS
  HANDLE fd_{INVALID_HANDLE_VALUE};  ///< Windows file handle
#else
  int fd_{-1};  ///< POSIX file descriptor
#endif

  std::string filepath_;                    ///< Path to the open file
  size_t sector_size_{kDefaultSectorSize};  ///< Sector size for alignment
  uint64_t file_size_{0};                   ///< Cached file size
  bool is_open_{false};                     ///< Whether file is currently open
  bool direct_io_enabled_{false};           ///< Whether Direct IO is enabled
  OpenMode mode_{};                         ///< Current open mode

 public:
  AlignedFileReader() = default;

  /**
   * @brief Constructor that opens a file.
   *
   * @param filepath Path to the file
   * @param mode Open mode flags
   */
  explicit AlignedFileReader(std::string_view filepath,
                             OpenMode mode = OpenMode::kReadOnly | OpenMode::kDirectIO) {
    auto status = open(filepath, mode);
    if (status != IOStatus::kSuccess) {
      throw std::runtime_error("Failed to open file: " + std::string(filepath) + " - " +
                               io_status_to_string(status));
    }
  }

  ~AlignedFileReader() { close(); }

  // Non-copyable
  AlignedFileReader(const AlignedFileReader &) = delete;
  auto operator=(const AlignedFileReader &) -> AlignedFileReader & = delete;

  // Movable
  AlignedFileReader(AlignedFileReader &&other) noexcept
      : fd_(other.fd_),
        filepath_(std::move(other.filepath_)),
        sector_size_(other.sector_size_),
        file_size_(other.file_size_),
        is_open_(other.is_open_),
        direct_io_enabled_(other.direct_io_enabled_),
        mode_(other.mode_) {
#ifdef ALAYA_OS_WINDOWS
    other.fd_ = INVALID_HANDLE_VALUE;
#else
    other.fd_ = -1;
#endif
    other.is_open_ = false;
  }

  auto operator=(AlignedFileReader &&other) noexcept -> AlignedFileReader & {
    if (this != &other) {
      close();
      fd_ = other.fd_;
      filepath_ = std::move(other.filepath_);
      sector_size_ = other.sector_size_;
      file_size_ = other.file_size_;
      is_open_ = other.is_open_;
      direct_io_enabled_ = other.direct_io_enabled_;
      mode_ = other.mode_;
#ifdef ALAYA_OS_WINDOWS
      other.fd_ = INVALID_HANDLE_VALUE;
#else
      other.fd_ = -1;
#endif
      other.is_open_ = false;
    }
    return *this;
  }

  // ==========================================================================
  // File Operations
  // ==========================================================================

  /**
   * @brief Open a file for aligned I/O.
   *
   * @param filepath Path to the file
   * @param mode Open mode flags
   * @return IOStatus indicating success or failure
   */
  auto open(std::string_view filepath, OpenMode mode = OpenMode::kReadOnly | OpenMode::kDirectIO)
      -> IOStatus {
    if (is_open_) {
      close();
    }

    filepath_ = std::string(filepath);
    mode_ = mode;
    direct_io_enabled_ = has_flag(mode, OpenMode::kDirectIO);

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
    if (!is_open_) {
      return;
    }

#ifdef ALAYA_OS_WINDOWS
    if (fd_ != INVALID_HANDLE_VALUE) {
      CloseHandle(fd_);
      fd_ = INVALID_HANDLE_VALUE;
    }
#else
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
#endif

    is_open_ = false;
    file_size_ = 0;
    filepath_.clear();
  }

  /**
   * @brief Check if the file is open.
   * @return true if file is open
   */
  [[nodiscard]] auto is_open() const -> bool { return is_open_; }

  /**
   * @brief Check if Direct IO is enabled.
   * @return true if Direct IO is enabled
   */
  [[nodiscard]] auto is_direct_io() const -> bool { return direct_io_enabled_; }

  /**
   * @brief Get the file path.
   * @return Reference to the file path string
   */
  [[nodiscard]] auto filepath() const -> const std::string & { return filepath_; }

  /**
   * @brief Get the sector size used for alignment.
   * @return Sector size in bytes
   */
  [[nodiscard]] auto sector_size() const -> size_t { return sector_size_; }

  /**
   * @brief Get the file size.
   * @return File size in bytes
   */
  [[nodiscard]] auto file_size() const -> uint64_t { return file_size_; }

  /**
   * @brief Update the cached file size.
   * @return IOStatus indicating success or failure
   */
  auto update_file_size() -> IOStatus {
    if (!is_open_) {
      return IOStatus::kFileNotOpen;
    }

#ifdef ALAYA_OS_WINDOWS
    LARGE_INTEGER size;
    if (!GetFileSizeEx(fd_, &size)) {
      return IOStatus::kError;
    }
    file_size_ = static_cast<uint64_t>(size.QuadPart);
#else
    struct stat st;
    if (fstat(fd_, &st) != 0) {
      return IOStatus::kError;
    }
    file_size_ = static_cast<uint64_t>(st.st_size);
#endif

    return IOStatus::kSuccess;
  }

  // ==========================================================================
  // Alignment Helpers
  // ==========================================================================

  /**
   * @brief Check if a pointer is properly aligned.
   *
   * @param ptr Pointer to check
   * @return true if aligned to sector size
   */
  [[nodiscard]] auto is_aligned(const void *ptr) const -> bool {
    return (reinterpret_cast<uintptr_t>(ptr) % sector_size_) == 0;
  }

  /**
   * @brief Check if an offset is properly aligned.
   *
   * @param offset Offset to check
   * @return true if aligned to sector size
   */
  [[nodiscard]] auto is_aligned(uint64_t offset) const -> bool {
    return (offset % sector_size_) == 0;
  }

  /**
   * @brief Round up a value to the next sector boundary.
   *
   * @param value Value to round up
   * @return Value rounded up to sector alignment
   */
  [[nodiscard]] auto align_up(uint64_t value) const -> uint64_t {
    return (value + sector_size_ - 1) & ~(sector_size_ - 1);
  }

  /**
   * @brief Round down a value to the previous sector boundary.
   *
   * @param value Value to round down
   * @return Value rounded down to sector alignment
   */
  [[nodiscard]] auto align_down(uint64_t value) const -> uint64_t {
    return value & ~(sector_size_ - 1);
  }

  // ==========================================================================
  // Synchronous I/O Operations
  // ==========================================================================

  /**
   * @brief Read data from file at a specific offset.
   *
   * For Direct IO mode, both buffer and offset must be sector-aligned.
   *
   * @param buffer Destination buffer (must be aligned for Direct IO)
   * @param size Number of bytes to read (must be multiple of sector size for Direct IO)
   * @param offset File offset to read from (must be aligned for Direct IO)
   * @return Number of bytes read, or -1 on error
   */
  auto read(void *buffer, size_t size, uint64_t offset) -> ssize_t {
    if (!is_open_) {
      LOG_ERROR("File not open for read");
      return -1;
    }

    // Check alignment for Direct IO
    if (direct_io_enabled_) {
      if (!is_aligned(buffer)) {
        LOG_ERROR("Buffer not aligned: ptr={}, sector_size={}", fmt::ptr(buffer), sector_size_);
        return -1;
      }
      if (!is_aligned(offset)) {
        LOG_ERROR("Offset not aligned: offset={}, sector_size={}", offset, sector_size_);
        return -1;
      }
      if (!is_aligned(size)) {
        LOG_ERROR("Size not aligned: size={}, sector_size={}", size, sector_size_);
        return -1;
      }
    }

#ifdef ALAYA_OS_WINDOWS
    return read_windows(buffer, size, offset);
#else
    return read_posix(buffer, size, offset);
#endif
  }

  /**
   * @brief Write data to file at a specific offset.
   *
   * For Direct IO mode, both buffer and offset must be sector-aligned.
   *
   * @param buffer Source buffer (must be aligned for Direct IO)
   * @param size Number of bytes to write (must be multiple of sector size for Direct IO)
   * @param offset File offset to write to (must be aligned for Direct IO)
   * @return Number of bytes written, or -1 on error
   */
  auto write(const void *buffer, size_t size, uint64_t offset) -> ssize_t {
    if (!is_open_) {
      LOG_ERROR("File not open for write");
      return -1;
    }

    // Check alignment for Direct IO
    if (direct_io_enabled_) {
      if (!is_aligned(buffer)) {
        LOG_ERROR("Buffer not aligned: ptr={}, sector_size={}", fmt::ptr(buffer), sector_size_);
        return -1;
      }
      if (!is_aligned(offset)) {
        LOG_ERROR("Offset not aligned: offset={}, sector_size={}", offset, sector_size_);
        return -1;
      }
      if (!is_aligned(size)) {
        LOG_ERROR("Size not aligned: size={}, sector_size={}", size, sector_size_);
        return -1;
      }
    }

#ifdef ALAYA_OS_WINDOWS
    return write_windows(buffer, size, offset);
#else
    return write_posix(buffer, size, offset);
#endif
  }

  /**
   * @brief Synchronize file data to disk.
   *
   * @param data_only If true, only sync data (not metadata)
   * @return IOStatus indicating success or failure
   */
  auto sync(bool data_only = false) -> IOStatus {
    if (!is_open_) {
      return IOStatus::kFileNotOpen;
    }

#ifdef ALAYA_OS_WINDOWS
    if (!FlushFileBuffers(fd_)) {
      return IOStatus::kError;
    }
#else
    int ret;
  #ifdef ALAYA_OS_LINUX
    if (data_only) {
      ret = fdatasync(fd_);
    } else {
      ret = fsync(fd_);
    }
  #else
    ret = fsync(fd_);
  #endif
    if (ret != 0) {
      return IOStatus::kError;
    }
#endif

    return IOStatus::kSuccess;
  }

  // ==========================================================================
  // Async I/O Interface (Placeholder for io_uring integration)
  // ==========================================================================

  /**
   * @brief Submit an asynchronous read request.
   *
   * @note This is a placeholder for future io_uring integration.
   *       Currently falls back to synchronous read.
   *
   * @param request The I/O request to submit
   * @return IOStatus::kSuccess if submitted, error status otherwise
   */
  auto read_async(IORequest *request) -> IOStatus {
    if (!is_open_) {
      request->status_.store(IOStatus::kFileNotOpen, std::memory_order_release);
      return IOStatus::kFileNotOpen;
    }

    // TODO(future): Implement io_uring based async read
    // For now, fall back to synchronous read
    ssize_t bytes = read(request->buffer_, request->size_, request->offset_);
    if (bytes < 0) {
      request->status_.store(IOStatus::kError, std::memory_order_release);
      request->bytes_transferred_ = 0;
      return IOStatus::kError;
    }

    request->bytes_transferred_ = bytes;
    if (static_cast<size_t>(bytes) < request->size_) {
      request->status_.store(IOStatus::kEOF, std::memory_order_release);
    } else {
      request->status_.store(IOStatus::kSuccess, std::memory_order_release);
    }
    return IOStatus::kSuccess;
  }

  /**
   * @brief Submit an asynchronous write request.
   *
   * @note This is a placeholder for future io_uring integration.
   *       Currently falls back to synchronous write.
   *
   * @param request The I/O request to submit
   * @return IOStatus::kSuccess if submitted, error status otherwise
   */
  auto write_async(IORequest *request) -> IOStatus {
    if (!is_open_) {
      request->status_.store(IOStatus::kFileNotOpen, std::memory_order_release);
      return IOStatus::kFileNotOpen;
    }

    // TODO(future): Implement io_uring based async write
    // For now, fall back to synchronous write
    ssize_t bytes = write(request->buffer_, request->size_, request->offset_);
    if (bytes < 0) {
      request->status_.store(IOStatus::kError, std::memory_order_release);
      request->bytes_transferred_ = 0;
      return IOStatus::kError;
    }

    request->bytes_transferred_ = bytes;
    request->status_.store(IOStatus::kSuccess, std::memory_order_release);
    return IOStatus::kSuccess;
  }

  /**
   * @brief Submit a batch of async I/O requests.
   *
   * @note This is a placeholder for future io_uring integration.
   *       Currently processes requests sequentially.
   *
   * @param ctx IOContext containing the requests to submit
   * @return Number of requests successfully submitted
   */
  auto submit_batch(IOContext *ctx) -> size_t {
    if (!is_open_ || ctx == nullptr) {
      return 0;
    }

    size_t submitted = 0;
    for (auto &req : ctx->requests()) {
      IOStatus status;
      if (req.is_write_) {
        status = write_async(&req);
      } else {
        status = read_async(&req);
      }
      if (status == IOStatus::kSuccess || status == IOStatus::kEOF) {
        ++submitted;
      }
    }
    return submitted;
  }

  /**
   * @brief Wait for and reap completed async I/O requests.
   *
   * @note This is a placeholder for future io_uring integration.
   *       Currently returns immediately as all requests complete synchronously.
   *
   * @param ctx IOContext to check for completions
   * @param min_completions Minimum number of completions to wait for
   * @param timeout_ms Timeout in milliseconds (-1 for infinite)
   * @return Number of completed requests
   */
  auto reap_completions([[maybe_unused]] IOContext *ctx,
                        [[maybe_unused]] size_t min_completions = 1,
                        [[maybe_unused]] int timeout_ms = -1) -> size_t {
    // TODO(future): Implement io_uring completion reaping
    // For now, all requests complete synchronously in submit_batch
    if (ctx == nullptr) {
      return 0;
    }
    return ctx->requests().size();
  }

 private:
  // ==========================================================================
  // Platform-specific Open Implementations
  // ==========================================================================

#ifdef ALAYA_OS_LINUX
  auto open_linux(OpenMode mode) -> IOStatus {
    int flags = 0;

    // Access mode
    if (has_flag(mode, OpenMode::kReadWrite)) {
      flags |= O_RDWR;
    } else if (has_flag(mode, OpenMode::kWriteOnly)) {
      flags |= O_WRONLY;
    } else {
      flags |= O_RDONLY;
    }

    // Creation flags
    if (has_flag(mode, OpenMode::kCreate)) {
      flags |= O_CREAT;
    }
    if (has_flag(mode, OpenMode::kTruncate)) {
      flags |= O_TRUNC;
    }
    if (has_flag(mode, OpenMode::kAppend)) {
      flags |= O_APPEND;
    }

    // Direct IO flag
    if (has_flag(mode, OpenMode::kDirectIO)) {
      flags |= O_DIRECT;
    }

    fd_ = ::open(filepath_.c_str(), flags, 0644);
    if (fd_ < 0) {
      LOG_ERROR("Failed to open file: {} (errno={})", filepath_, errno);
      return IOStatus::kError;
    }

    is_open_ = true;
    direct_io_enabled_ = has_flag(mode, OpenMode::kDirectIO);

    // Get file size
    update_file_size();

    LOG_INFO("Opened file: {} (fd={}, direct_io={})", filepath_, fd_, direct_io_enabled_);
    return IOStatus::kSuccess;
  }
#endif

#ifdef ALAYA_OS_MACOS
  auto open_macos(OpenMode mode) -> IOStatus {
    int flags = 0;

    // Access mode
    if (has_flag(mode, OpenMode::kReadWrite)) {
      flags |= O_RDWR;
    } else if (has_flag(mode, OpenMode::kWriteOnly)) {
      flags |= O_WRONLY;
    } else {
      flags |= O_RDONLY;
    }

    // Creation flags
    if (has_flag(mode, OpenMode::kCreate)) {
      flags |= O_CREAT;
    }
    if (has_flag(mode, OpenMode::kTruncate)) {
      flags |= O_TRUNC;
    }
    if (has_flag(mode, OpenMode::kAppend)) {
      flags |= O_APPEND;
    }

    fd_ = ::open(filepath_.c_str(), flags, 0644);
    if (fd_ < 0) {
      LOG_ERROR("Failed to open file: {} (errno={})", filepath_, errno);
      return IOStatus::kError;
    }

    // macOS: Use F_NOCACHE to disable caching (similar to O_DIRECT)
    if (has_flag(mode, OpenMode::kDirectIO)) {
      if (fcntl(fd_, F_NOCACHE, 1) == -1) {
        LOG_WARN("Failed to set F_NOCACHE on {}, falling back to buffered IO", filepath_);
        direct_io_enabled_ = false;
      } else {
        direct_io_enabled_ = true;
      }
    }

    is_open_ = true;
    update_file_size();

    LOG_INFO("Opened file: {} (fd={}, direct_io={})", filepath_, fd_, direct_io_enabled_);
    return IOStatus::kSuccess;
  }
#endif

#ifdef ALAYA_OS_WINDOWS
  auto open_windows(OpenMode mode) -> IOStatus {
    DWORD access = 0;
    DWORD share = FILE_SHARE_READ;
    DWORD creation = OPEN_EXISTING;
    DWORD flags = FILE_ATTRIBUTE_NORMAL;

    // Access mode
    if (has_flag(mode, OpenMode::kReadWrite)) {
      access = GENERIC_READ | GENERIC_WRITE;
    } else if (has_flag(mode, OpenMode::kWriteOnly)) {
      access = GENERIC_WRITE;
    } else {
      access = GENERIC_READ;
    }

    // Creation disposition
    if (has_flag(mode, OpenMode::kCreate) && has_flag(mode, OpenMode::kTruncate)) {
      creation = CREATE_ALWAYS;
    } else if (has_flag(mode, OpenMode::kCreate)) {
      creation = OPEN_ALWAYS;
    } else if (has_flag(mode, OpenMode::kTruncate)) {
      creation = TRUNCATE_EXISTING;
    }

    // Direct IO flag
    if (has_flag(mode, OpenMode::kDirectIO)) {
      flags |= FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH;
      direct_io_enabled_ = true;
    }

    fd_ = CreateFileA(filepath_.c_str(), access, share, nullptr, creation, flags, nullptr);

    if (fd_ == INVALID_HANDLE_VALUE) {
      LOG_ERROR("Failed to open file: {} (error={})", filepath_, GetLastError());
      return IOStatus::kError;
    }

    is_open_ = true;
    update_file_size();

    LOG_INFO("Opened file: {} (direct_io={})", filepath_, direct_io_enabled_);
    return IOStatus::kSuccess;
  }
#endif

  // Fallback for unknown platforms
  auto open_fallback(OpenMode mode) -> IOStatus {
    int flags = 0;

    if (has_flag(mode, OpenMode::kReadWrite)) {
      flags |= O_RDWR;
    } else if (has_flag(mode, OpenMode::kWriteOnly)) {
      flags |= O_WRONLY;
    } else {
      flags |= O_RDONLY;
    }

    if (has_flag(mode, OpenMode::kCreate)) {
      flags |= O_CREAT;
    }
    if (has_flag(mode, OpenMode::kTruncate)) {
      flags |= O_TRUNC;
    }

    fd_ = ::open(filepath_.c_str(), flags, 0644);
    if (fd_ < 0) {
      return IOStatus::kError;
    }

    is_open_ = true;
    direct_io_enabled_ = false;  // No Direct IO on fallback
    update_file_size();

    LOG_WARN("Opened file without Direct IO support: {}", filepath_);
    return IOStatus::kSuccess;
  }

  // ==========================================================================
  // Platform-specific Read/Write Implementations
  // ==========================================================================

#ifndef ALAYA_OS_WINDOWS
  auto read_posix(void *buffer, size_t size, uint64_t offset) -> ssize_t {
    return ::pread(fd_, buffer, size, static_cast<off_t>(offset));
  }

  auto write_posix(const void *buffer, size_t size, uint64_t offset) -> ssize_t {
    return ::pwrite(fd_, buffer, size, static_cast<off_t>(offset));
  }
#endif

#ifdef ALAYA_OS_WINDOWS
  auto read_windows(void *buffer, size_t size, uint64_t offset) -> ssize_t {
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

    DWORD bytes_read = 0;
    if (!ReadFile(fd_, buffer, static_cast<DWORD>(size), &bytes_read, &overlapped)) {
      if (GetLastError() != ERROR_HANDLE_EOF) {
        return -1;
      }
    }
    return static_cast<ssize_t>(bytes_read);
  }

  auto write_windows(const void *buffer, size_t size, uint64_t offset) -> ssize_t {
    OVERLAPPED overlapped = {};
    overlapped.Offset = static_cast<DWORD>(offset & 0xFFFFFFFF);
    overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);

    DWORD bytes_written = 0;
    if (!WriteFile(fd_, buffer, static_cast<DWORD>(size), &bytes_written, &overlapped)) {
      return -1;
    }
    return static_cast<ssize_t>(bytes_written);
  }
#endif
};

}  // namespace alaya
