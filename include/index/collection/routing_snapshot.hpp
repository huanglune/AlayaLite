// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <unordered_map>
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
               bool supports_atomic_bundle,
               SegmentMaintenanceHook maintenance_hook = {})
      : segment_id(id),
        generation(segment_generation),
        role(segment_role),
        segment(std::move(erased_segment)),
        exact_rerank(std::move(rerank)),
        next_row_id(first_unused_row),
        atomic_mutation_bundle(supports_atomic_bundle),
        maintenance(std::move(maintenance_hook)) {}

  std::uint64_t segment_id{};
  std::uint64_t generation{};
  SegmentRole role{SegmentRole::sealed};
  core::AnySegment segment{};
  ExactRerank exact_rerank{};
  std::atomic_uint64_t next_row_id{};
  bool atomic_mutation_bundle{};
  SegmentMaintenanceHook maintenance{};

  // Collection uses this lock only when an instance declares a weaker
  // ConcurrencyProfile. It never changes the engine's operation table.
  mutable std::shared_mutex operation_mutex{};
};

using VersionMap = std::map<core::LogicalId, VersionEntry, LogicalIdLess>;
using ReverseMap = std::map<RowAddress, ReverseEntry>;

struct SegmentIdentity {
  std::uint64_t segment_id{};
  std::uint64_t generation{};

  auto operator==(const SegmentIdentity &) const -> bool = default;
};

struct SegmentIdentityHash {
  [[nodiscard]] auto operator()(const SegmentIdentity &identity) const noexcept -> std::size_t {
    auto seed = std::hash<std::uint64_t>{}(identity.segment_id);
    seed ^= std::hash<std::uint64_t>{}(identity.generation) + 0x9e3779b97f4a7c15ULL + (seed << 6U) +
            (seed >> 2U);
    return seed;
  }
};

using KnownRowCounts = std::unordered_map<SegmentIdentity, core::RowCount, SegmentIdentityHash>;

struct RoutingSnapshot {
  std::uint64_t generation{1};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{1};
  std::vector<std::shared_ptr<SegmentEntry>> segments{};
  VersionMap versions{};
  ReverseMap reverse{};
  KnownRowCounts known_row_counts{};
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
    const auto found =
        known_row_counts.find(SegmentIdentity{segment.segment_id, segment.generation});
    return found == known_row_counts.end() ? 0 : found->second;
  }

  void rebuild_known_row_counts() {
    known_row_counts.clear();
    for (const auto &[address, unused] : reverse) {
      (void)unused;
      ++known_row_counts[SegmentIdentity{address.segment_id, address.generation}];
    }
  }
};

using RoutingSnapshotPtr = std::shared_ptr<const RoutingSnapshot>;

}  // namespace alaya::internal::collection
