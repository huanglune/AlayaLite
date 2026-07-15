// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <filesystem>
#include <utility>

namespace alaya {

class FileLock {
 public:
  explicit FileLock(std::filesystem::path lock_file) : lock_file_(std::move(lock_file)) {
    fd_ = open(lock_file_.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ >= 0) {
      flock(fd_, LOCK_EX);
    }
  }

  ~FileLock() {
    if (fd_ >= 0) {
      flock(fd_, LOCK_UN);
      close(fd_);
    }
  }

  FileLock(const FileLock &) = delete;
  auto operator=(const FileLock &) -> FileLock & = delete;

 private:
  std::filesystem::path lock_file_;
  int fd_ = -1;
};

}  // namespace alaya
