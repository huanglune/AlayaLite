// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace alaya::core {

using ExternalId = std::uint64_t;
using RowCount = std::uint64_t;

// All distances are smaller-is-better. L2 means squared L2, inner_product
// means negative dot product, and cosine means negative dot product after L2
// normalization. NaN is reserved for engines that expose rank-only results.
enum class Metric : std::uint8_t { l2 = 0, inner_product = 1, cosine = 2 };
enum class Medium : std::uint8_t { memory = 0, disk = 1 };
enum class SegmentState : std::uint8_t {
  building = 0,
  mutable_live = 1,
  sealing = 2,
  sealed = 3,
  failed = 4,
};
enum class FilterSupport : std::uint8_t {
  none = 0,
  prefilter = 1,
  traversal = 2,
  postfilter = 3,
};

struct QueryView {
  const float *data{};
  std::uint32_t dim{};
};

struct QueryBatchView {
  const float *data{};
  RowCount rows{};
  std::uint32_t dim{};
};

struct VectorBatchView {
  const float *data{};
  const ExternalId *ids{};
  RowCount rows{};
  std::uint32_t dim{};
};

// filter and request are non-owning, type-erased tokens. Their concrete plans
// and runtime contexts belong to later metadata/runtime layers.
struct SearchOptions {
  std::uint32_t top_k{10};
  std::uint32_t effort{100};
  std::uint32_t beam_width{4};
  bool exact_rerank{true};
  const void *filter{};
  const void *request{};
};

struct SearchHit {
  ExternalId id{};
  float distance{};
};

using SearchSink = std::span<SearchHit>;

struct SearchResult {
  RowCount count{};
  RowCount visited{};
};

struct BatchSearchResult {
  RowCount query_count{};
  RowCount hit_count{};
};

// Stable numeric identities are assigned by Collection/registries. They are
// deliberately not C++ strings or engine enums.
struct Descriptor {
  std::uint64_t id{};
  std::uint64_t algorithm_id{};
  std::uint32_t format_version{};
  std::uint32_t dim{};
  RowCount rows{};
  Metric metric{Metric::l2};
  Medium medium{Medium::memory};
  SegmentState state{SegmentState::building};
  std::uint8_t reserved{};
};

struct Artifact {
  std::string_view name{};
  std::uint64_t size_bytes{};
  std::uint64_t checksum{};
};

struct ArtifactManifest {
  std::uint32_t schema_version{1};
  std::uint32_t format_version{};
  std::uint64_t algorithm_id{};
  std::span<const Artifact> artifacts{};
};

// Non-owning filesystem views used by memory segments.  The writer names the
// destinations selected by the caller; engines keep ownership and atomic
// publication policy outside the segment.
struct ArtifactView {
  std::string_view graph_path{};
  std::string_view data_path{};
  std::string_view quant_path{};
};

using ArtifactWriter = ArtifactView;

struct OpenOptions {};
struct OpenContext {
  const void *opaque{};
};
struct SaveOptions {};

struct MutationContext {
  const void *opaque{};
};
struct CheckpointContext {
  const void *opaque{};
};
struct SealContext {
  const void *opaque{};
};
struct BuildContext {
  const void *opaque{};
};

struct MutationResult {
  RowCount affected{};
};
struct CheckpointToken {
  std::uint64_t value{};
};
struct SealedArtifact {
  ArtifactManifest manifest{};
};

}  // namespace alaya::core
