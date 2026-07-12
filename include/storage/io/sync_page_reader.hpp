// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

#include "storage/io/alignment.hpp"

namespace alaya::storage::io {

class SyncPageReader final : public PageReader {
 public:
  explicit SyncPageReader(const std::filesystem::path &path, ReaderOptions options = {})
      : constraints_(conservative_constraints(options.mode == OpenMode::automatic
                                                  ? OpenMode::buffered
                                                  : options.mode,
                                              options.queue_depth)) {
    int flags = O_RDONLY;
#ifdef O_DIRECT
    if (constraints_.direct) flags |= O_DIRECT;
#endif
    fd_ = ::open(path.c_str(), flags);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open");
  }

  ~SyncPageReader() override { shutdown(); }

  [[nodiscard]] auto constraints() const noexcept -> ReadConstraints override {
    return constraints_;
  }

  [[nodiscard]] auto submit(std::span<const ReadRequest> requests, Completion completion)
      -> BatchHandle override {
    if (shutdown_) throw std::runtime_error("PageReader is shut down");
    if ((completion.fn == nullptr && !requests.empty()) ||
        requests.size() > constraints_.max_batch) {
      throw std::invalid_argument("invalid PageReader batch");
    }
    for (const auto &request : requests) {
      if (!validate_read_request(request, constraints_)) {
        throw std::invalid_argument("read request violates alignment constraints");
      }
    }

    auto completed = std::make_shared<std::atomic_bool>(requests.empty());
    auto handle = make_batch_handle([completed]() noexcept {
      return completed->load(std::memory_order_acquire) ? CancelResult::already_complete
                                                        : CancelResult::unsupported;
    });
    for (const auto &request : requests) {
      ReadResult result{.id = request.id};
      const auto bytes = ::pread(fd_,
                                 request.buffer.data(),
                                 request.buffer.size(),
                                 static_cast<off_t>(request.offset));
      if (bytes < 0) {
        result.status = ReadStatus::io_error;
        result.error = {errno, std::generic_category()};
      } else {
        result.bytes = static_cast<std::size_t>(bytes);
        if (Clock::now() > request.deadline) {
          result.status = ReadStatus::timed_out;
        } else {
          result.status =
              result.bytes == request.buffer.size() ? ReadStatus::ok : ReadStatus::short_read;
        }
      }
      completion.fn(completion.context, result);  // Intentionally inline (D8).
    }
    completed->store(true, std::memory_order_release);
    return handle;
  }

  void shutdown() noexcept override {
    if (!std::exchange(shutdown_, true) && fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_ = -1;
  ReadConstraints constraints_{};
  bool shutdown_ = false;
};

}  // namespace alaya::storage::io
