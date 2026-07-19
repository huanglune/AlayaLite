// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <memory>

#include "core/algorithm_registry.hpp"

namespace alaya {

struct QgSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  QgSearchExtension() : header(core::current_struct_header<QgSearchExtension>()) {}
};

[[nodiscard]] inline auto make_qg_search_extension(const QgSearchExtension &options)
    -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::qg;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

}  // namespace alaya
