// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include "storage/io/page_reader_factory.hpp"

namespace alaya::laser {

struct PageReadCompletions {
  std::mutex mutex;
  std::deque<storage::io::ReadResult> results;
};

[[nodiscard]] inline auto make_laser_page_reader(const std::filesystem::path &path,
                                                 std::uint32_t queue_depth = 128)
    -> std::unique_ptr<storage::io::PageReader> {
  const storage::io::ReaderOptions options{.mode = storage::io::OpenMode::automatic,
                                           .queue_depth = queue_depth};
#if defined(__linux__)
  return storage::io::open_page_reader(path, options, storage::io::PageReaderBackend::libaio);
#else
  return storage::io::open_page_reader(path, options, storage::io::PageReaderBackend::threadpool);
#endif
}

[[nodiscard]] inline auto allocate_page_read_buffer(const storage::io::PageReader &reader,
                                                    std::size_t bytes) -> void * {
  const auto constraints = reader.constraints();
  if (constraints.buffer_alignment == 0 || constraints.size_alignment == 0 ||
      bytes % constraints.size_alignment != 0) {
    throw std::invalid_argument("LASER page geometry violates PageReader constraints");
  }
  const auto alignment = std::max(constraints.buffer_alignment, alignof(void *));
  void *buffer = nullptr;
  if (::posix_memalign(&buffer, alignment, bytes) != 0) throw std::bad_alloc();
  std::memset(buffer, 0, bytes);
  return buffer;
}

inline void collect_page_read(void *context, storage::io::ReadResult result) noexcept {
  auto &completions = *static_cast<PageReadCompletions *>(context);
  {
    std::lock_guard lock(completions.mutex);
    completions.results.push_back(result);
  }
}

[[nodiscard]] inline auto submit_page_reads(storage::io::PageReader &reader,
                                            std::span<const storage::io::ReadRequest> requests,
                                            PageReadCompletions &completions) -> std::size_t {
  [[maybe_unused]] auto handle =
      reader.submit(requests,
                    storage::io::Completion{.fn = collect_page_read, .context = &completions});
  return requests.size();
}

inline void validate_page_read(const storage::io::ReadResult &result) {
  if (result.status != storage::io::ReadStatus::ok) {
    throw std::system_error(result.error,
                            "LASER page read failed for request " + std::to_string(result.id));
  }
}

inline auto poll_page_reads(PageReadCompletions &completions,
                            std::size_t max_results,
                            std::vector<storage::io::ReadResult> &out) -> std::size_t {
  std::lock_guard lock(completions.mutex);
  out.clear();
  const auto count = std::min(max_results, completions.results.size());
  for (std::size_t i = 0; i < count; ++i) {
    validate_page_read(completions.results.front());
    out.push_back(completions.results.front());
    completions.results.pop_front();
  }
  return count;
}

}  // namespace alaya::laser
