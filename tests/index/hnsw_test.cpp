// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <queue>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "core/capabilities.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "executor/jobs/graph_update_job.hpp"
#include "index/graph/hnsw/detail/hnsw_builder_kernel.hpp"
#include "index/graph/hnsw/detail/hnsw_segment_bridge.hpp"
#include "index/graph/hnsw/hnsw_segment.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"

namespace alaya {
namespace {

using Space = RawSpace<>;
using Segment = HnswSegment<Space>;
using Sq8Segment = HnswSegment<SQ8Space<>, Space>;
using ByteSpace = RawSpace<std::int8_t, float, std::uint32_t>;
using ByteSegment = HnswSegment<ByteSpace>;
using UnsignedByteSpace = RawSpace<std::uint8_t, float, std::uint32_t>;
using UnsignedByteSegment = HnswSegment<UnsignedByteSpace>;
using WideIdSegment = HnswSegment<RawSpace<float, float, std::uint64_t>>;

template <typename T>
concept CompleteHnswLifecycle = requires(typename T::BuildInput input,
                                         const HnswBuildOptions &build_options,
                                         core::BuildContext &build_context,
                                         core::ArtifactView artifact,
                                         const core::OpenOptions &open_options,
                                         core::OpenContext &open_context) {
  { T::build(input, build_options, build_context) } -> std::same_as<std::unique_ptr<T>>;
  { T::open(artifact, open_options, open_context) } -> std::same_as<std::unique_ptr<T>>;
};

static_assert(CompleteHnswLifecycle<Segment>);
static_assert(core::Searchable<Segment>);
static_assert(core::BatchSearchable<Segment>);
static_assert(core::Saveable<Segment>);
static_assert(core::StatsProvider<Segment>);
static_assert(!core::Mutable<Segment>);
static_assert(core::Searchable<Sq8Segment>);
static_assert(core::BatchSearchable<Sq8Segment>);
static_assert(core::Saveable<Sq8Segment>);
static_assert(core::Searchable<ByteSegment>);
static_assert(core::Searchable<UnsignedByteSegment>);
static_assert(core::Searchable<WideIdSegment>);

template <typename SegmentType>
auto artifact_locations(std::string_view graph, std::string_view data, std::string_view quant = {})
    -> std::array<core::ArtifactLocation, 3> {
  return {core::ArtifactLocation(SegmentType::kGraphArtifactName, graph),
          core::ArtifactLocation(SegmentType::kDataArtifactName, data),
          core::ArtifactLocation(SegmentType::kQuantArtifactName, quant)};
}

struct SearchCall {
  template <class T>
  SearchCall(const T *data,
             core::RowCount rows,
             std::uint32_t dim,
             std::uint64_t top_k,
             std::uint32_t effort)
      : hits(static_cast<std::size_t>(rows * top_k)),
        offsets(static_cast<std::size_t>(rows + 1)),
        counts(static_cast<std::size_t>(rows)),
        statuses(static_cast<std::size_t>(rows)),
        completeness(static_cast<std::size_t>(rows)) {
    hnsw.effort = effort;
    extension = make_hnsw_search_extension(hnsw);
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    request.queries = core::TypedTensorView::contiguous(data, rows, dim);
    request.options.top_k = top_k;
    request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = &context;
    request.response = &response;
  }

  core::SearchContext context;
  HnswSearchExtension hnsw;
  core::AlgorithmSearchExtension extension;
  std::vector<core::SearchHit> hits;
  std::vector<core::RowCount> offsets;
  std::vector<core::RowCount> counts;
  std::vector<core::Status> statuses;
  std::vector<core::SearchCompleteness> completeness;
  core::SearchResponse response;
  core::SearchRequest request;
};

auto read_bytes(const std::filesystem::path &path) -> std::vector<char> {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read test artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class HnswSegmentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alayalite-hnsw-segment-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);

    data_.resize(kRows * kDim);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t col = 0; col < kDim; ++col) {
        data_[row * kDim + col] = static_cast<float>((row * 17 + col * 5) % 101) / 101.0F;
      }
    }
    space_ = std::make_shared<Space>(kCapacity, kDim, MetricType::L2);
    space_->fit(data_.data(), kRows);
    core::BuildContext context;
    segment_ = Segment::build({space_, space_}, build_options(), context);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static constexpr auto build_options() -> HnswBuildOptions {
    return {.max_neighbors = 8, .ef_construction = 32, .thread_count = 1};
  }

  static constexpr std::uint32_t kRows = 64;
  static constexpr std::uint32_t kCapacity = 80;
  static constexpr std::uint32_t kDim = 8;
  std::filesystem::path root_;
  std::vector<float> data_;
  std::shared_ptr<Space> space_;
  std::unique_ptr<Segment> segment_;
};

TEST_F(HnswSegmentTest, BuildReturnsFullySearchableSegment) {
  const auto descriptor = segment_->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::hnsw);
  EXPECT_EQ(descriptor.format_version, Segment::kFormatVersion);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, core::Medium::memory);

  core::SegmentStats stats;
  ASSERT_TRUE(segment_->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kCapacity);

  SearchCall call(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(segment_->search(call.request).ok());
  ASSERT_EQ(call.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(call.hits.front().score, 0.0F);
  EXPECT_EQ(call.hits.front().score_kind, core::ScoreKind::distance);
  EXPECT_EQ(call.hits.front().comparable_metric, core::Metric::l2);
}

TEST_F(HnswSegmentTest, BatchSearchWritesCallerOwnedSink) {
  SearchCall call(data_.data(), 2, kDim, 3, 24);
  ASSERT_TRUE(segment_->batch_search(call.request).ok());
  EXPECT_EQ(call.response.query_count, 2U);
  EXPECT_EQ(call.offsets, (std::vector<core::RowCount>{0, 3, 6}));
  EXPECT_EQ(call.counts, (std::vector<core::RowCount>{3, 3}));
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits[0].row_id), 0U);
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits[3].row_id), 1U);
  EXPECT_TRUE(call.statuses[0].ok());
  EXPECT_TRUE(call.statuses[1].ok());

  SearchCall too_small(data_.data(), 2, kDim, 3, 24);
  too_small.response.hits = std::span<core::SearchHit>(too_small.hits.data(), 5);
  EXPECT_EQ(segment_->batch_search(too_small.request).detail(), core::StatusDetail::sink_too_small);
}

TEST_F(HnswSegmentTest, RejectsInvalidUnifiedSearchRequests) {
  SearchCall missing(static_cast<const float *>(nullptr), 1, kDim, 4, 24);
  EXPECT_EQ(segment_->search(missing.request).detail(), core::StatusDetail::null_data);

  SearchCall wrong_dim(data_.data(), 1, kDim - 1, 4, 24);
  EXPECT_EQ(segment_->search(wrong_dim.request).detail(), core::StatusDetail::dimension_mismatch);

  SearchCall low_effort(data_.data(), 1, kDim, 4, 3);
  EXPECT_EQ(segment_->search(low_effort.request).code(), core::StatusCode::invalid_argument);

  SearchCall filtered(data_.data(), 1, kDim, 4, 24);
  filtered.request.filter.kind = core::SegmentFilterKind::bitmap;
  EXPECT_EQ(segment_->search(filtered.request).code(), core::StatusCode::not_supported);
}

TEST_F(HnswSegmentTest, EmptyBatchAndZeroScratchBudgetFollowV3Rules) {
  SearchCall empty(static_cast<const float *>(nullptr), 0, kDim, 3, 24);
  ASSERT_TRUE(segment_->batch_search(empty.request).ok());
  EXPECT_EQ(empty.offsets, (std::vector<core::RowCount>{0}));

  SearchCall denied(data_.data(), 1, kDim, 4, 24);
  denied.context.query_scratch_lease = core::MemoryLease(0);
  const auto status = segment_->search(denied.request);
  EXPECT_EQ(status.code(), core::StatusCode::resource_exhausted);
  EXPECT_EQ(status.detail(), core::StatusDetail::budget_denied);
}

TEST_F(HnswSegmentTest, SaveReturnsVersionedManifestAndPreservesLegacyBytes) {
  const auto segment_graph = (root_ / "segment.graph").string();
  const auto segment_data = (root_ / "segment.data").string();
  const auto legacy_graph = (root_ / "legacy.graph").string();
  const auto legacy_data = (root_ / "legacy.data").string();

  const auto locations = artifact_locations<Segment>(segment_graph, segment_data);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.format_version, Segment::kFormatVersion);
  ASSERT_EQ(manifest.algorithm_id, core::algorithm::hnsw);
  ASSERT_EQ(manifest.artifacts.size(), 2U);
  EXPECT_EQ(manifest.artifacts[0].name, Segment::kGraphArtifactName);
  EXPECT_EQ(manifest.artifacts[1].name, Segment::kDataArtifactName);
  EXPECT_EQ(manifest.artifacts[0].size_bytes, std::filesystem::file_size(segment_graph));
  EXPECT_EQ(manifest.artifacts[1].size_bytes, std::filesystem::file_size(segment_data));

  // The internal bridge is the old Graph/Space writer used only to prove byte
  // compatibility. Segment save must be an orchestration layer, not a codec.
  detail::HnswSegmentBridge<Space, Space>::graph(*segment_)->save(legacy_graph);
  space_->save(legacy_data);
  EXPECT_EQ(read_bytes(segment_graph), read_bytes(legacy_graph));
  EXPECT_EQ(read_bytes(segment_data), read_bytes(legacy_data));

  Graph<> legacy_graph_reader;
  legacy_graph_reader.load(segment_graph);
  Space legacy_data_reader;
  legacy_data_reader.load(segment_data);
  EXPECT_EQ(legacy_graph_reader.max_nodes_, kCapacity);
  EXPECT_NE(legacy_graph_reader.overlay_graph_, nullptr);
  EXPECT_EQ(legacy_data_reader.get_data_num(), kRows);
}

TEST_F(HnswSegmentTest, OpenReadsLegacyRawArtifacts) {
  const auto graph = (root_ / "legacy-open.graph").string();
  const auto data = (root_ / "legacy-open.data").string();
  detail::HnswSegmentBridge<Space, Space>::graph(*segment_)->save(graph);
  space_->save(data);

  const auto locations = artifact_locations<Segment>(graph, data);
  core::OpenContext context;
  auto reopened =
      Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                    {},
                    context);
  SearchCall call(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(reopened->search(call.request).ok());
  ASSERT_EQ(call.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(call.hits.front().score, 0.0F);
}

TEST_F(HnswSegmentTest, OpenReadsLegacySq8GraphDataQuantFamily) {
  auto sq8 = std::make_shared<SQ8Space<>>(kCapacity, kDim, MetricType::L2);
  sq8->fit(data_.data(), kRows);

  detail::HnswBuilderKernel<Space> legacy_builder(space_, 8, 32);
  auto legacy_graph = legacy_builder.build_graph(1);
  const auto graph = (root_ / "legacy-sq8.graph").string();
  const auto data = (root_ / "legacy-sq8.data").string();
  const auto quant = (root_ / "legacy-sq8.quant").string();
  legacy_graph->save(graph);
  space_->save(data);
  sq8->save(quant);

  const auto locations = artifact_locations<Sq8Segment>(graph, data, quant);
  core::OpenContext context;
  auto reopened =
      Sq8Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                       {},
                       context);
  core::SegmentStats stats;
  ASSERT_TRUE(reopened->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  SearchCall call(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(reopened->search(call.request).ok());
  ASSERT_EQ(call.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits.front().row_id), 0U);

  const auto roundtrip_graph = (root_ / "roundtrip-sq8.graph").string();
  const auto roundtrip_data = (root_ / "roundtrip-sq8.data").string();
  const auto roundtrip_quant = (root_ / "roundtrip-sq8.quant").string();
  const auto roundtrip_locations =
      artifact_locations<Sq8Segment>(roundtrip_graph, roundtrip_data, roundtrip_quant);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(roundtrip_locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(reopened->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.artifacts.size(), 3U);
  EXPECT_EQ(manifest.artifacts[2].name, Sq8Segment::kQuantArtifactName);
  EXPECT_EQ(read_bytes(graph), read_bytes(roundtrip_graph));
  EXPECT_EQ(read_bytes(data), read_bytes(roundtrip_data));
  EXPECT_EQ(read_bytes(quant), read_bytes(roundtrip_quant));
}

template <class T, class TypedSpace, class TypedSegment>
void verify_native_byte_query() {
  constexpr std::uint32_t rows = 16;
  constexpr std::uint32_t dim = 4;
  std::vector<T> data(rows * dim);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint32_t col = 0; col < dim; ++col) {
      data[row * dim + col] = static_cast<T>(row + col);
    }
  }
  auto space = std::make_shared<TypedSpace>(rows, dim, MetricType::L2);
  space->fit(data.data(), rows);
  core::BuildContext context;
  auto segment = TypedSegment::build({space, space},
                                     {.max_neighbors = 8, .ef_construction = 24, .thread_count = 1},
                                     context);
  SearchCall native(data.data(), 1, dim, 3, 12);
  ASSERT_TRUE(segment->search(native.request).ok());
  ASSERT_EQ(native.counts[0], 3U);
  EXPECT_EQ(static_cast<std::uint64_t>(native.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(native.hits.front().score, 0.0F);

  std::vector<float> wrong_type(dim);
  SearchCall converted(wrong_type.data(), 1, dim, 3, 12);
  EXPECT_EQ(segment->search(converted.request).code(), core::StatusCode::not_supported);
}

TEST(HnswSegmentTypedQueryTest, Int8AndUint8RequireNativeTypedViews) {
  verify_native_byte_query<std::int8_t, ByteSpace, ByteSegment>();
  verify_native_byte_query<std::uint8_t, UnsignedByteSpace, UnsignedByteSegment>();
}

TEST_F(HnswSegmentTest, V3SearchMatchesV2KernelBitForBit) {
  constexpr std::uint32_t query_count = 4;
  constexpr std::uint32_t top_k = 5;
  constexpr std::uint32_t effort = 24;
  SearchCall v3(data_.data(), query_count, kDim, top_k, effort);
  ASSERT_TRUE(segment_->batch_search(v3.request).ok());

  auto graph = detail::HnswSegmentBridge<Space, Space>::graph(*segment_);
  GraphSearchJob<Space> legacy(space_, graph, nullptr, space_);
  std::array<std::uint32_t, top_k> legacy_ids{};
  std::array<float, top_k> legacy_distances{};
  for (std::uint32_t row = 0; row < query_count; ++row) {
    legacy.search_solo(data_.data() + row * kDim,
                       legacy_ids.data(),
                       legacy_distances.data(),
                       top_k,
                       effort);
    ASSERT_EQ(v3.counts[row], top_k);
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto offset = static_cast<std::size_t>(v3.offsets[row] + hit);
      EXPECT_EQ(static_cast<std::uint64_t>(v3.hits[offset].row_id), legacy_ids[hit]);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(v3.hits[offset].score),
                std::bit_cast<std::uint32_t>(legacy_distances[hit]));
    }
  }
}

TEST_F(HnswSegmentTest, RegistersAsFirstRealAnySegmentProducer) {
  auto erased = Segment::into_any(std::move(segment_));
  ASSERT_TRUE(erased.ok());
  auto any = std::move(erased).value();
  const auto capabilities = any.capabilities();
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::save));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(core::OperationCapability::mutation));

  SearchCall call(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(any.search(call.request).ok());
  EXPECT_EQ(call.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits.front().row_id), 0U);
}

struct InvariantReport {
  bool connected{};
  bool layer_structure{};
  bool degree_constraints{};
  bool search_reachability{};
};

auto is_active(const Space &space, std::uint32_t id) -> bool {
  return id < space.get_data_num() && space.data_storage_.is_valid(id);
}

auto live_count(const Space &space) -> std::uint32_t {
  std::uint32_t count = 0;
  for (std::uint32_t id = 0; id < space.get_data_num(); ++id) {
    count += static_cast<std::uint32_t>(is_active(space, id));
  }
  return count;
}

auto connected_from_live_entry(const Graph<> &graph, const Space &space) -> bool {
  if (graph.overlay_graph_ == nullptr) {
    return false;
  }
  const auto entry = graph.overlay_graph_->ep_;
  if (!is_active(space, entry)) {
    return false;
  }

  std::vector<bool> visited(space.get_data_num(), false);
  std::queue<std::uint32_t> pending;
  visited[entry] = true;
  pending.push(entry);
  while (!pending.empty()) {
    const auto node = pending.front();
    pending.pop();
    for (std::uint32_t edge = 0; edge < graph.max_nbrs_; ++edge) {
      const auto neighbor = graph.at(node, edge);
      if (neighbor == Graph<>::kEmptyId) {
        break;
      }
      if (neighbor < visited.size() && is_active(space, neighbor) && !visited[neighbor]) {
        visited[neighbor] = true;
        pending.push(neighbor);
      }
    }
  }

  for (std::uint32_t id = 0; id < space.get_data_num(); ++id) {
    if (is_active(space, id) && !visited[id]) {
      return false;
    }
  }
  return true;
}

struct DegreeReport {
  bool live_endpoints{true};
  bool contiguous_lists{true};
  bool unique_neighbors{true};
  bool no_self_edges{true};
  bool nonzero_in_and_out{true};

  [[nodiscard]] auto valid() const -> bool {
    return live_endpoints && contiguous_lists && unique_neighbors && no_self_edges &&
           nonzero_in_and_out;
  }
};

auto inspect_in_and_out_degrees(const Graph<> &graph, const Space &space) -> DegreeReport {
  DegreeReport report;
  const auto rows = space.get_data_num();
  std::vector<std::uint32_t> in_degree(rows, 0);
  std::vector<std::uint32_t> out_degree(rows, 0);
  for (std::uint32_t node = 0; node < rows; ++node) {
    if (!is_active(space, node)) {
      continue;
    }
    std::unordered_set<std::uint32_t> unique;
    bool saw_empty = false;
    for (std::uint32_t edge = 0; edge < graph.max_nbrs_; ++edge) {
      const auto neighbor = graph.at(node, edge);
      if (neighbor == Graph<>::kEmptyId) {
        saw_empty = true;
        continue;
      }
      if (saw_empty) {
        report.contiguous_lists = false;
      }
      if (neighbor >= rows || !is_active(space, neighbor)) {
        report.live_endpoints = false;
        continue;
      }
      report.no_self_edges &= neighbor != node;
      report.unique_neighbors &= unique.insert(neighbor).second;
      ++out_degree[node];
      ++in_degree[neighbor];
    }
  }

  const auto active_count = live_count(space);
  if (active_count > 1) {
    for (std::uint32_t node = 0; node < rows; ++node) {
      if (is_active(space, node) && (in_degree[node] == 0 || out_degree[node] == 0)) {
        report.nonzero_in_and_out = false;
      }
    }
  }
  return report;
}

auto valid_hnsw_layers(const Graph<> &graph, const Space &space) -> bool {
  if (graph.overlay_graph_ == nullptr) {
    return false;
  }
  const auto &overlay = *graph.overlay_graph_;
  const auto rows = space.get_data_num();
  if (overlay.levels_.size() != graph.max_nodes_ || overlay.lists_.size() != graph.max_nodes_ ||
      overlay.node_num_ != graph.max_nodes_ || overlay.max_nbrs_ != graph.max_nbrs_ ||
      !is_active(space, overlay.ep_)) {
    return false;
  }

  std::uint32_t max_level = 0;
  for (std::uint32_t node = 0; node < rows; ++node) {
    if (is_active(space, node)) {
      max_level = std::max(max_level, overlay.levels_[node]);
    }
  }
  if (overlay.levels_[overlay.ep_] != max_level) {
    return false;
  }

  const auto upper_degree_bound = graph.max_nbrs_ / 2;
  for (std::uint32_t node = 0; node < rows; ++node) {
    if (!is_active(space, node)) {
      continue;
    }
    const auto level_count = overlay.levels_[node];
    if (overlay.lists_[node].size() != static_cast<std::size_t>(level_count) * graph.max_nbrs_) {
      return false;
    }
    for (std::uint32_t level = 1; level <= level_count; ++level) {
      std::unordered_set<std::uint32_t> unique;
      bool saw_empty = false;
      std::uint32_t degree = 0;
      for (std::uint32_t edge = 0; edge < graph.max_nbrs_; ++edge) {
        const auto neighbor = overlay.at(level, node, edge);
        if (neighbor == Graph<>::kEmptyId) {
          saw_empty = true;
          continue;
        }
        if (saw_empty || neighbor >= rows || !is_active(space, neighbor) || neighbor == node ||
            overlay.levels_[neighbor] < level || !unique.insert(neighbor).second) {
          return false;
        }
        ++degree;
      }
      if (degree > upper_degree_bound) {
        return false;
      }
    }
  }
  return true;
}

auto every_live_vector_is_search_reachable(GraphSearchJob<Space> &search,
                                           const Space &space,
                                           std::uint32_t top_k = 8) -> bool {
  const auto active_count = live_count(space);
  top_k = std::min(top_k, active_count);
  std::vector<std::uint32_t> ids(top_k);
  for (std::uint32_t id = 0; id < space.get_data_num(); ++id) {
    if (!is_active(space, id)) {
      continue;
    }
    search.search_solo(space.get_data_by_id(id), ids.data(), top_k, std::max(24U, top_k));
    if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
      return false;
    }
  }
  return true;
}

auto inspect_invariants(const Graph<> &graph, Space &space, GraphSearchJob<Space> &search)
    -> InvariantReport {
  return {.connected = connected_from_live_entry(graph, space),
          .layer_structure = valid_hnsw_layers(graph, space),
          .degree_constraints = inspect_in_and_out_degrees(graph, space).valid(),
          .search_reachability = every_live_vector_is_search_reachable(search, space)};
}

// Decision point 6: run the legacy generic update path as the mutable
// candidate, then keep Mutable absent because the HNSW-specific invariants do
// not survive.  A future native implementation must make every assertion in
// this characterization pass before adding insert/erase to HnswSegment.
TEST(HnswMutationInvariantTest, GenericUpdateCandidateFailsLayerAndDeletionInvariants) {
  constexpr std::uint32_t base_rows = 40;
  constexpr std::uint32_t insert_rows = 24;
  constexpr std::uint32_t capacity = 72;
  constexpr std::uint32_t dim = 6;
  constexpr std::uint32_t max_neighbors = 8;

  std::vector<float> all_data((base_rows + insert_rows) * dim);
  for (std::uint32_t row = 0; row < base_rows + insert_rows; ++row) {
    for (std::uint32_t col = 0; col < dim; ++col) {
      all_data[row * dim + col] =
          static_cast<float>((row * 29 + col * 11 + (row % 5) * col) % 127) / 127.0F;
    }
  }

  auto candidate_space = std::make_shared<Space>(capacity, dim, MetricType::L2);
  candidate_space->fit(all_data.data(), base_rows);
  detail::HnswBuilderKernel<Space> candidate_builder(candidate_space, max_neighbors, 32);
  auto candidate_graph = std::shared_ptr<Graph<>>(candidate_builder.build_graph(1).release());
  auto candidate_search = std::make_shared<GraphSearchJob<Space>>(candidate_space, candidate_graph);
  GraphUpdateJob<Space> candidate_update(candidate_search);

  const auto before = inspect_invariants(*candidate_graph, *candidate_space, *candidate_search);
  ASSERT_TRUE(before.connected);
  ASSERT_TRUE(before.layer_structure);
  ASSERT_TRUE(before.degree_constraints);
  ASSERT_TRUE(before.search_reachability);

  const auto overlay_before = candidate_graph->overlay_graph_->lists_;
  const auto levels_before = candidate_graph->overlay_graph_->levels_;
  const auto entry_before = candidate_graph->overlay_graph_->ep_;
  for (std::uint32_t row = 0; row < insert_rows; ++row) {
    auto *vector = all_data.data() + static_cast<std::size_t>(base_rows + row) * dim;
    ASSERT_EQ(candidate_update.insert_and_update(vector, 24), base_rows + row);
  }

  const auto after_insert =
      inspect_invariants(*candidate_graph, *candidate_space, *candidate_search);
  EXPECT_TRUE(after_insert.connected);
  EXPECT_TRUE(after_insert.layer_structure);
  const auto insert_degrees = inspect_in_and_out_degrees(*candidate_graph, *candidate_space);
  // Generic reciprocal repair admits duplicate neighbors and self-edges even
  // though every endpoint is live and every node retains non-zero in/out degree.
  EXPECT_TRUE(insert_degrees.live_endpoints);
  EXPECT_TRUE(insert_degrees.contiguous_lists);
  EXPECT_FALSE(insert_degrees.unique_neighbors);
  EXPECT_FALSE(insert_degrees.no_self_edges);
  EXPECT_TRUE(insert_degrees.nonzero_in_and_out);
  EXPECT_FALSE(after_insert.degree_constraints);
  EXPECT_TRUE(after_insert.search_reachability);

  // A native single-thread HNSW build with the same fixed level RNG assigns
  // upper levels to this deterministic insertion suffix. The generic updater
  // leaves the entire overlay unchanged and assigns level zero to every node.
  auto native_space = std::make_shared<Space>(capacity, dim, MetricType::L2);
  native_space->fit(all_data.data(), base_rows + insert_rows);
  detail::HnswBuilderKernel<Space> native_builder(native_space, max_neighbors, 32);
  auto native_graph = native_builder.build_graph(1);
  const auto native_has_new_upper_level =
      std::any_of(native_graph->overlay_graph_->levels_.begin() + base_rows,
                  native_graph->overlay_graph_->levels_.begin() + base_rows + insert_rows,
                  [](std::uint32_t level) {
                    return level > 0;
                  });
  ASSERT_TRUE(native_has_new_upper_level);
  EXPECT_EQ(candidate_graph->overlay_graph_->ep_, entry_before);
  EXPECT_EQ(std::vector(candidate_graph->overlay_graph_->levels_.begin(),
                        candidate_graph->overlay_graph_->levels_.begin() + base_rows),
            std::vector(levels_before.begin(), levels_before.begin() + base_rows));
  EXPECT_EQ(candidate_graph->overlay_graph_->lists_, overlay_before);
  EXPECT_TRUE(
      std::all_of(candidate_graph->overlay_graph_->levels_.begin() + base_rows,
                  candidate_graph->overlay_graph_->levels_.begin() + base_rows + insert_rows,
                  [](std::uint32_t level) {
                    return level == 0;
                  }));

  // Deleting the live HNSW entry point exposes all three structural failures:
  // the entry point is stale, active nodes retain inbound edges to the deleted
  // node, and live-only traversal no longer starts from a connected entry.
  const auto deleted_entry = candidate_graph->overlay_graph_->ep_;
  std::vector<float> deleted_query(candidate_space->get_data_by_id(deleted_entry),
                                   candidate_space->get_data_by_id(deleted_entry) + dim);
  candidate_update.remove(deleted_entry);
  const auto after_erase =
      inspect_invariants(*candidate_graph, *candidate_space, *candidate_search);
  EXPECT_FALSE(after_erase.connected);
  EXPECT_FALSE(after_erase.layer_structure);
  EXPECT_FALSE(after_erase.degree_constraints);

  std::vector<std::uint32_t> deleted_search_ids(8);
  candidate_search->search_solo(deleted_query.data(), deleted_search_ids.data(), 8, 24);
  EXPECT_EQ(std::find(deleted_search_ids.begin(), deleted_search_ids.end(), deleted_entry),
            deleted_search_ids.end());
  EXPECT_TRUE(after_erase.search_reachability);

  static_assert(!core::Mutable<Segment>);
}

}  // namespace
}  // namespace alaya
