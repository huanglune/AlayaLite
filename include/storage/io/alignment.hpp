// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

#include "storage/io/page_reader.hpp"

namespace alaya::storage::io {

[[nodiscard]] constexpr auto is_aligned(std::uint64_t value, std::size_t alignment) noexcept
    -> bool {
  return alignment != 0 && value % alignment == 0;
}

[[nodiscard]] inline auto validate_read_request(const ReadRequest &request,
                                                const ReadConstraints &constraints) noexcept
    -> bool {
  const auto address = reinterpret_cast<std::uintptr_t>(request.buffer.data());
  return constraints.buffer_alignment != 0 && constraints.offset_alignment != 0 &&
         constraints.size_alignment != 0 && is_aligned(address, constraints.buffer_alignment) &&
         is_aligned(request.offset, constraints.offset_alignment) &&
         is_aligned(request.buffer.size(), constraints.size_alignment);
}

[[nodiscard]] constexpr auto align_down(std::uint64_t value, std::size_t alignment) noexcept
    -> std::optional<std::uint64_t> {
  if (alignment == 0) return std::nullopt;
  return value - value % alignment;
}

[[nodiscard]] constexpr auto align_up(std::uint64_t value, std::size_t alignment) noexcept
    -> std::optional<std::uint64_t> {
  if (alignment == 0) return std::nullopt;
  const auto remainder = value % alignment;
  if (remainder == 0) return value;
  const auto delta = alignment - remainder;
  if (value > std::numeric_limits<std::uint64_t>::max() - delta) return std::nullopt;
  return value + delta;
}

// Conservative Linux policy for the selected open mode. A backend may replace
// these values with statx/device-derived constraints after opening the file.
[[nodiscard]] constexpr auto conservative_constraints(OpenMode mode,
                                                      std::uint32_t max_batch = 128) noexcept
    -> ReadConstraints {
  if (mode == OpenMode::buffered) return {1, 1, 1, max_batch, false};
  return {4096, 4096, 4096, max_batch, true};
}

// Windows no-buffering constraints are volume-dependent; the future IOCP
// backend supplies them through this platform-neutral contract.
[[nodiscard]] constexpr auto constraints_from_sector_sizes(std::size_t logical_sector,
                                                           std::size_t physical_sector,
                                                           std::uint32_t max_batch = 128) noexcept
    -> ReadConstraints {
  const auto alignment = std::max(logical_sector, physical_sector);
  return {alignment, logical_sector, logical_sector, max_batch, true};
}

}  // namespace alaya::storage::io
