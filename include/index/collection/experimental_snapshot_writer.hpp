// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string_view>

#include "index/collection/routing_snapshot.hpp"

namespace alaya::internal::collection {

// This is intentionally not a checkpoint, WAL, replay source, or durability
// promise. Gate 4 only provides a manually invoked, feature-gated diagnostic
// snapshot so every experimental writer is confined to a new namespace. Gate
// 7 owns the durable protocol and any future reader.
class ExperimentalSnapshotWriter {
 public:
  static constexpr std::string_view kNamespace{"collection_shell_v1"};

  [[nodiscard]] static auto write(const PersistenceOptions &options,
                                  const RoutingSnapshot &snapshot) -> core::Status {
    if (options.root.empty() || options.namespace_name != kNamespace) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "experimental writer requires the collection_shell_v1 namespace");
    }
    try {
      const auto directory = options.root / ".alaya_internal" / options.namespace_name;
      std::filesystem::create_directories(directory);
      const auto output_path =
          directory / ("version_map_" + std::to_string(snapshot.generation) + ".snapshot");
      std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
      if (!output) {
        return core::Status::error(core::StatusCode::io_error,
                                   core::OperationStage::save,
                                   core::StatusDetail::none,
                                   "cannot open experimental version-map snapshot");
      }
      output << "schema=collection-shell-v1\n"
             << "generation=" << snapshot.generation << '\n'
             << "watermark=" << snapshot.visibility_watermark << '\n'
             << "metadata_epoch=" << snapshot.metadata_epoch << '\n';
      for (const auto &[logical_id, version] : snapshot.versions) {
        output << "id=" << static_cast<unsigned>(logical_id.kind()) << ':';
        for (const auto value : logical_id.canonical_bytes()) {
          output << std::hex << std::setw(2) << std::setfill('0')
                 << std::to_integer<unsigned>(value);
        }
        output << std::dec << ",segment=" << version.address.segment_id
               << ",generation=" << version.address.generation
               << ",row=" << static_cast<std::uint64_t>(version.address.row_id)
               << ",sequence=" << version.upsert_sequence
               << ",state=" << (version.state == VersionState::live ? "live" : "tombstone") << '\n';
      }
      output.flush();
      if (!output) {
        return core::Status::error(core::StatusCode::io_error,
                                   core::OperationStage::save,
                                   core::StatusDetail::none,
                                   "cannot write experimental version-map snapshot");
      }
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }
};

}  // namespace alaya::internal::collection
