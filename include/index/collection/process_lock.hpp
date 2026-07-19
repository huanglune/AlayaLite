// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#ifndef _WIN32
  #include <sys/file.h>
#endif

#include "core/status.hpp"
#include "platform/fs.hpp"

namespace alaya::internal::collection {

inline constexpr std::string_view kCollectionProcessLockFilename{"collection.lock"};

// A stable lock file gives the facade a cross-process mode contract:
// exactly one read-write handle, or any number of read-only handles.  The
// file is never removed or written, so acquiring a shared lock is byte-stable.
class CollectionProcessLock {
 public:
  CollectionProcessLock(const CollectionProcessLock &) = delete;
  auto operator=(const CollectionProcessLock &) -> CollectionProcessLock & = delete;
  CollectionProcessLock(CollectionProcessLock &&) = delete;
  auto operator=(CollectionProcessLock &&) -> CollectionProcessLock & = delete;

  ~CollectionProcessLock() { release(); }

  [[nodiscard]] static auto acquire(const std::filesystem::path &root,
                                    bool read_only,
                                    bool create_if_missing)
      -> core::Result<std::unique_ptr<CollectionProcessLock>> {
    const auto path = root / ".alaya_internal" / kCollectionProcessLockFilename;
    try {
      if (create_if_missing) {
        std::filesystem::create_directories(path.parent_path());
      } else if (!std::filesystem::is_regular_file(path)) {
        return core::Status::
            error(core::StatusCode::not_supported,
                  core::OperationStage::open,
                  core::StatusDetail::readonly_instance,
                  "read-only Collection open requires an initialized process lock; "
                  "open once in read-write mode first");
      }

#ifdef _WIN32
      const auto access = read_only ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE);
      const auto sharing = read_only ? FILE_SHARE_READ : 0;
      const auto disposition = create_if_missing ? OPEN_ALWAYS : OPEN_EXISTING;
      auto handle = ::CreateFileW(path.c_str(),
                                  access,
                                  sharing,
                                  nullptr,
                                  disposition,
                                  FILE_ATTRIBUTE_NORMAL,
                                  nullptr);
      if (handle == INVALID_HANDLE_VALUE) {
        const auto error = ::GetLastError();
        if (error == ERROR_SHARING_VIOLATION || error == ERROR_LOCK_VIOLATION) {
          return conflict_status();
        }
        return io_status("CreateFileW failed with error " + std::to_string(error));
      }
      return std::unique_ptr<CollectionProcessLock>(
          new CollectionProcessLock(path, read_only, handle));
#else
      auto flags = read_only ? O_RDONLY : O_RDWR;
      if (create_if_missing) {
        flags |= O_CREAT;
      }
  #ifdef O_CLOEXEC
      flags |= O_CLOEXEC;
  #endif
  #ifdef O_NOFOLLOW
      flags |= O_NOFOLLOW;
  #endif
      const auto descriptor = ::open(path.c_str(), flags, 0600);
      if (descriptor < 0) {
        return io_status("open failed: " + std::string(std::strerror(errno)));
      }
      const auto operation = (read_only ? LOCK_SH : LOCK_EX) | LOCK_NB;
      if (::flock(descriptor, operation) != 0) {
        const auto error = errno;
        ::close(descriptor);
        if (error == EWOULDBLOCK || error == EAGAIN) {
          return conflict_status();
        }
        return io_status("flock failed: " + std::string(std::strerror(error)));
      }
      return std::unique_ptr<CollectionProcessLock>(
          new CollectionProcessLock(path, read_only, descriptor));
#endif
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  void release() noexcept {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(handle_);
      handle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (descriptor_ >= 0) {
      (void)::flock(descriptor_, LOCK_UN);
      (void)::close(descriptor_);
      descriptor_ = -1;
    }
#endif
  }

  [[nodiscard]] auto read_only() const noexcept -> bool { return read_only_; }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
#ifdef _WIN32
  CollectionProcessLock(std::filesystem::path path, bool read_only, HANDLE handle)
      : path_(std::move(path)), read_only_(read_only), handle_(handle) {}
#else
  CollectionProcessLock(std::filesystem::path path, bool read_only, int descriptor)
      : path_(std::move(path)), read_only_(read_only), descriptor_(descriptor) {}
#endif

  [[nodiscard]] static auto conflict_status() -> core::Status {
    return core::Status::error(core::StatusCode::conflict,
                               core::OperationStage::open,
                               core::StatusDetail::already_exists,
                               "Collection lock conflict: one writer or multiple read-only "
                               "handles are allowed",
                               core::Retryability::retryable);
  }

  [[nodiscard]] static auto io_status(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::io_error,
                               core::OperationStage::open,
                               core::StatusDetail::engine_exception,
                               "Collection process lock " + std::move(diagnostic));
  }

  std::filesystem::path path_{};
  bool read_only_{};
#ifdef _WIN32
  HANDLE handle_{INVALID_HANDLE_VALUE};
#else
  int descriptor_{-1};
#endif
};

}  // namespace alaya::internal::collection
