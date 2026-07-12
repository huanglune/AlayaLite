// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace alaya::core {

// Contract v3 is frozen. Engines must not add private fields to an erased
// boundary. New fields are introduced here through a versioned tail or a new
// operation-table version.
inline constexpr std::uint32_t kContractAbiVersion = 3;
inline constexpr std::uint32_t kOperationTableVersion = 3;

struct VersionedStructHeader {
  std::uint32_t struct_size{};
  std::uint32_t abi_version{kContractAbiVersion};
};

template <class T>
[[nodiscard]] constexpr auto current_struct_header() noexcept -> VersionedStructHeader {
  static_assert(sizeof(T) <= std::numeric_limits<std::uint32_t>::max());
  return {static_cast<std::uint32_t>(sizeof(T)), kContractAbiVersion};
}

[[nodiscard]] constexpr auto has_versioned_prefix(const VersionedStructHeader &header,
                                                  std::size_t required_size) noexcept -> bool {
  return header.abi_version == kContractAbiVersion &&
         header.struct_size >= sizeof(VersionedStructHeader) && header.struct_size >= required_size;
}

template <class T>
[[nodiscard]] constexpr auto is_current_struct(const T &value) noexcept -> bool {
  return has_versioned_prefix(value.header, sizeof(T));
}

[[nodiscard]] constexpr auto checked_add(std::uint64_t lhs,
                                         std::uint64_t rhs,
                                         std::uint64_t &result) noexcept -> bool {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return false;
  }
  result = lhs + rhs;
  return true;
}

[[nodiscard]] constexpr auto checked_multiply(std::uint64_t lhs,
                                              std::uint64_t rhs,
                                              std::uint64_t &result) noexcept -> bool {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

static_assert(std::is_standard_layout_v<VersionedStructHeader>);
static_assert(sizeof(VersionedStructHeader) == 8,
              "same-toolchain layout regression canary for the v3 prefix");

}  // namespace alaya::core
