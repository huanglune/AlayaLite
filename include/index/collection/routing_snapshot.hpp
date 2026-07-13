// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "index/collection/types.hpp"

namespace alaya::internal::collection {

struct SegmentEntry {
  SegmentEntry(std::uint64_t id,
               std::uint64_t segment_generation,
               SegmentRole segment_role,
               core::AnySegment erased_segment,
               ExactRerank rerank,
               std::uint64_t first_unused_row,
               bool supports_atomic_bundle)
      : segment_id(id),
        generation(segment_generation),
        role(segment_role),
        segment(std::move(erased_segment)),
        exact_rerank(std::move(rerank)),
        next_row_id(first_unused_row),
        atomic_mutation_bundle(supports_atomic_bundle) {}

  std::uint64_t segment_id{};
  std::uint64_t generation{};
  SegmentRole role{SegmentRole::sealed};
  core::AnySegment segment{};
  ExactRerank exact_rerank{};
  std::atomic_uint64_t next_row_id{};
  bool atomic_mutation_bundle{};

  // Collection uses this lock only when an instance declares a weaker
  // ConcurrencyProfile. It never changes the engine's operation table.
  mutable std::shared_mutex operation_mutex{};
};

using VersionMap = std::map<core::LogicalId, VersionEntry, LogicalIdLess>;
using ReverseMap = std::map<RowAddress, ReverseEntry>;

struct RoutingSnapshot {
  std::uint64_t generation{1};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{1};
  std::vector<std::shared_ptr<SegmentEntry>> segments{};
  VersionMap versions{};
  ReverseMap reverse{};
  core::RowCount searchable_live_count{};
  core::RowCount tombstone_count{};

  [[nodiscard]] auto find_segment(std::uint64_t segment_id, std::uint64_t segment_generation) const
      -> std::shared_ptr<SegmentEntry> {
    for (const auto &entry : segments) {
      if (entry->segment_id == segment_id && entry->generation == segment_generation) {
        return entry;
      }
    }
    return {};
  }

  [[nodiscard]] auto find_active_mutable() const -> std::shared_ptr<SegmentEntry> {
    for (const auto &entry : segments) {
      if (entry->role == SegmentRole::active_mutable &&
          entry->segment.capabilities().supports(core::OperationCapability::mutation)) {
        return entry;
      }
    }
    return {};
  }

  [[nodiscard]] auto known_rows_for(const SegmentEntry &segment) const -> core::RowCount {
    core::RowCount count{};
    for (const auto &[address, unused] : reverse) {
      (void)unused;
      if (address.segment_id == segment.segment_id && address.generation == segment.generation) {
        ++count;
      }
    }
    return count;
  }
};

using RoutingSnapshotPtr = std::shared_ptr<const RoutingSnapshot>;

}  // namespace alaya::internal::collection
