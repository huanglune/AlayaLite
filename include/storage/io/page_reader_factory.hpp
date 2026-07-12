// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <memory>

#if defined(__linux__) && defined(ALAYA_LASER_USE_LIBAIO) && ALAYA_LASER_USE_LIBAIO
  #include "storage/io/backends/libaio_page_reader.hpp"
#endif
#include "storage/io/backends/threadpool_page_reader.hpp"
#include "storage/io/sync_page_reader.hpp"

namespace alaya::storage::io {

enum class PageReaderBackend : std::uint8_t { sync, libaio, threadpool, io_uring, iocp };

[[nodiscard]] inline auto open_page_reader(const std::filesystem::path &path,
                                           const ReaderOptions &options = {},
                                           PageReaderBackend backend = PageReaderBackend::sync)
    -> std::unique_ptr<PageReader> {
  switch (backend) {
    case PageReaderBackend::sync:
      return std::make_unique<SyncPageReader>(path, options);
    case PageReaderBackend::threadpool:
      return std::make_unique<ThreadpoolPageReader>(path, options);
    case PageReaderBackend::libaio:
#if defined(__linux__) && defined(ALAYA_LASER_USE_LIBAIO) && ALAYA_LASER_USE_LIBAIO
      return std::make_unique<LibaioPageReader>(path, options);
#else
      throw std::runtime_error("libaio PageReader is unavailable on this platform");
#endif
    case PageReaderBackend::io_uring:
      throw std::runtime_error("io_uring PageReader is not implemented");
    case PageReaderBackend::iocp:
      throw std::runtime_error("IOCP PageReader is not implemented");
  }
  throw std::runtime_error("unknown PageReader backend");
}

}  // namespace alaya::storage::io
