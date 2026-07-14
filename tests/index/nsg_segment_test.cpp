// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/capabilities.hpp"
#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/detail/memory_graph_segment.hpp"
#include "index/graph/nsg/detail/nsg_builder_kernel.hpp"
#include "index/graph/nsg/nsg_segment.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"

namespace alaya {
namespace {

using Space = RawSpace<>;
using Segment = NsgSegment<Space>;
using Sq8Segment = NsgSegment<SQ8Space<>, Space>;
using ByteSegment = NsgSegment<RawSpace<std::int8_t, float, std::uint32_t>>;
using UnsignedByteSegment = NsgSegment<RawSpace<std::uint8_t, float, std::uint32_t>>;
using WideIdSegment = NsgSegment<RawSpace<float, float, std::uint64_t>>;

template <typename T>
concept CompleteNsgLifecycle = requires(typename T::BuildInput input,
                                        const NsgBuildOptions &build_options,
                                        core::BuildContext &build_context,
                                        core::ArtifactView artifact,
                                        const core::OpenOptions &open_options,
                                        core::OpenContext &open_context) {
  { T::build(input, build_options, build_context) } -> std::same_as<std::unique_ptr<T>>;
  { T::open(artifact, open_options, open_context) } -> std::same_as<std::unique_ptr<T>>;
};

static_assert(CompleteNsgLifecycle<Segment>);
static_assert(core::Searchable<Segment>);
static_assert(core::BatchSearchable<Segment>);
static_assert(core::Saveable<Segment>);
static_assert(core::StatsProvider<Segment>);
static_assert(!core::Mutable<Segment>);
static_assert(core::Searchable<Sq8Segment>);
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
    nsg.effort = effort;
    extension = make_nsg_search_extension(nsg);
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
  NsgSearchExtension nsg;
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

class NsgSegmentTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { spdlog::set_level(spdlog::level::warn); }

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alayalite-nsg-segment-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);

    data_.resize(kRows * kDim);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t col = 0; col < kDim; ++col) {
        data_[row * kDim + col] =
            static_cast<float>((row * 17 + col * 5 + (row % 7) * col) % 101) / 101.0F;
      }
    }
    space_ = std::make_shared<Space>(kCapacity, kDim, MetricType::L2);
    space_->fit(data_.data(), kRows);
    core::BuildContext context;
    segment_ = Segment::build({core::TypedTensorView::contiguous(data_.data(), kRows, kDim),
                               space_,
                               space_},
                              build_options(),
                              context);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static constexpr auto build_options() -> NsgBuildOptions {
    return {.max_neighbors = 8, .ef_construction = 32, .thread_count = 1};
  }

  static constexpr std::uint32_t kRows = 128;
  static constexpr std::uint32_t kCapacity = 144;
  static constexpr std::uint32_t kDim = 8;
  std::filesystem::path root_;
  std::vector<float> data_;
  std::shared_ptr<Space> space_;
  std::unique_ptr<Segment> segment_;
};

TEST_F(NsgSegmentTest, BuildSearchBatchAndDescriptorFollowV3Contract) {
  const auto descriptor = segment_->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::nsg);
  EXPECT_EQ(descriptor.format_version, Segment::kFormatVersion);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, core::Medium::memory);

  core::SegmentStats stats;
  ASSERT_TRUE(segment_->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kCapacity);

  SearchCall single(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(segment_->search(single.request).ok());
  ASSERT_EQ(single.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(single.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(single.hits.front().score, 0.0F);

  SearchCall batch(data_.data(), 3, kDim, 4, 24);
  ASSERT_TRUE(segment_->batch_search(batch.request).ok());
  EXPECT_EQ(batch.offsets, (std::vector<core::RowCount>{0, 4, 8, 12}));
  EXPECT_EQ(batch.counts, (std::vector<core::RowCount>{4, 4, 4}));
  EXPECT_EQ(static_cast<std::uint64_t>(batch.hits[4].row_id), 1U);

  SearchCall wrong_rows(data_.data(), 2, kDim, 4, 24);
  EXPECT_EQ(segment_->search(wrong_rows.request).code(), core::StatusCode::invalid_argument);
  std::array<std::int8_t, kDim> wrong_scalar{};
  SearchCall wrong_type(wrong_scalar.data(), 1, kDim, 4, 24);
  EXPECT_EQ(segment_->search(wrong_type.request).detail(),
            core::StatusDetail::unsupported_scalar_type);
}

TEST_F(NsgSegmentTest, SegmentMatchesDetailKernelBytesAndSearchBitForBit) {
  const auto segment_graph = (root_ / "segment.graph").string();
  const auto segment_data = (root_ / "segment.data").string();
  const auto direct_graph_path = (root_ / "direct.graph").string();
  const auto direct_data_path = (root_ / "direct.data").string();

  const auto locations = artifact_locations<Segment>(segment_graph, segment_data);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.format_version, Segment::kFormatVersion);
  ASSERT_EQ(manifest.algorithm_id, core::algorithm::nsg);
  ASSERT_EQ(manifest.artifacts.size(), 2U);

  detail::NsgBuilderKernel<Space> direct_builder(space_,
                                                 build_options().max_neighbors,
                                                 build_options().ef_construction);
  auto direct_graph =
      std::shared_ptr<Graph<>>(direct_builder.build_graph(build_options().thread_count).release());
  direct_graph->save(direct_graph_path);
  space_->save(direct_data_path);
  EXPECT_EQ(read_bytes(segment_graph), read_bytes(direct_graph_path));
  EXPECT_EQ(read_bytes(segment_data), read_bytes(direct_data_path));

  constexpr std::uint32_t query_count = 8;
  constexpr std::uint32_t top_k = 8;
  constexpr std::uint32_t effort = 32;
  SearchCall v3(data_.data(), query_count, kDim, top_k, effort);
  ASSERT_TRUE(segment_->batch_search(v3.request).ok());
  GraphSearchJob<Space> direct_search(space_, direct_graph, nullptr, space_);
  std::array<std::uint32_t, top_k> direct_ids{};
  std::array<float, top_k> direct_distances{};
  std::uint32_t exact_matches = 0;
  for (std::uint32_t row = 0; row < query_count; ++row) {
    direct_search.search_solo(data_.data() + row * kDim,
                              direct_ids.data(),
                              direct_distances.data(),
                              top_k,
                              effort);
    ASSERT_EQ(v3.counts[row], top_k);
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto offset = static_cast<std::size_t>(v3.offsets[row] + hit);
      EXPECT_EQ(static_cast<std::uint64_t>(v3.hits[offset].row_id), direct_ids[hit]);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(v3.hits[offset].score),
                std::bit_cast<std::uint32_t>(direct_distances[hit]));
    }

    std::vector<std::pair<float, std::uint32_t>> exact;
    exact.reserve(kRows);
    for (std::uint32_t candidate = 0; candidate < kRows; ++candidate) {
      exact.emplace_back(space_->get_distance(row, candidate), candidate);
    }
    std::partial_sort(exact.begin(), exact.begin() + top_k, exact.end());
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      exact_matches += static_cast<std::uint32_t>(
          std::find_if(exact.begin(), exact.begin() + top_k, [&](const auto &entry) {
            return entry.second == direct_ids[hit];
          }) != exact.begin() + top_k);
    }
  }
  const auto recall = static_cast<double>(exact_matches) / (query_count * top_k);
  EXPECT_GE(recall, 0.5);
}

TEST_F(NsgSegmentTest, OpenReadsLegacyArtifactsAndRoundTripsBytes) {
  detail::NsgBuilderKernel<Space> legacy_builder(space_,
                                                 build_options().max_neighbors,
                                                 build_options().ef_construction);
  auto legacy_graph = legacy_builder.build_graph(1);
  const auto graph = (root_ / "legacy.graph").string();
  const auto data = (root_ / "legacy.data").string();
  legacy_graph->save(graph);
  space_->save(data);

  const auto locations = artifact_locations<Segment>(graph, data);
  core::OpenContext context;
  auto reopened =
      Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                    {},
                    context);
  SearchCall call(data_.data(), 1, kDim, 4, 24);
  ASSERT_TRUE(reopened->search(call.request).ok());
  EXPECT_EQ(static_cast<std::uint64_t>(call.hits.front().row_id), 0U);

  const auto roundtrip_graph = (root_ / "roundtrip.graph").string();
  const auto roundtrip_data = (root_ / "roundtrip.data").string();
  const auto roundtrip_locations = artifact_locations<Segment>(roundtrip_graph, roundtrip_data);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(roundtrip_locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(reopened->save(writer, {}, manifest).ok());
  EXPECT_EQ(read_bytes(graph), read_bytes(roundtrip_graph));
  EXPECT_EQ(read_bytes(data), read_bytes(roundtrip_data));
}

TEST_F(NsgSegmentTest, QuantizedArtifactFamilyRoundTripsWithoutCodecChanges) {
  auto sq8 = std::make_shared<SQ8Space<>>(kCapacity, kDim, MetricType::L2);
  sq8->fit(data_.data(), kRows);
  core::BuildContext build_context;
  auto quantized =
      Sq8Segment::build({core::TypedTensorView::contiguous(data_.data(), kRows, kDim), sq8, space_},
                        build_options(),
                        build_context);

  const auto graph = (root_ / "sq8.graph").string();
  const auto data = (root_ / "sq8.data").string();
  const auto quant = (root_ / "sq8.quant").string();
  const auto locations = artifact_locations<Sq8Segment>(graph, data, quant);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(quantized->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.artifacts.size(), 3U);

  core::OpenContext open_context;
  auto reopened =
      Sq8Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                       {},
                       open_context);
  SearchCall call(data_.data(), 2, kDim, 5, 32);
  ASSERT_TRUE(reopened->batch_search(call.request).ok());
  EXPECT_EQ(call.counts, (std::vector<core::RowCount>{5, 5}));

  const auto rt_graph = (root_ / "sq8-rt.graph").string();
  const auto rt_data = (root_ / "sq8-rt.data").string();
  const auto rt_quant = (root_ / "sq8-rt.quant").string();
  const auto rt_locations = artifact_locations<Sq8Segment>(rt_graph, rt_data, rt_quant);
  core::ArtifactWriter rt_writer{std::span<const core::ArtifactLocation>(rt_locations)};
  ASSERT_TRUE(reopened->save(rt_writer, {}, manifest).ok());
  EXPECT_EQ(read_bytes(graph), read_bytes(rt_graph));
  EXPECT_EQ(read_bytes(data), read_bytes(rt_data));
  EXPECT_EQ(read_bytes(quant), read_bytes(rt_quant));
}

TEST_F(NsgSegmentTest, ContextBudgetsAndAnySegmentCapabilitiesAreEnforced) {
  core::BuildContext denied_context;
  denied_context.growing_reservation = core::MemoryReservation(0);
  EXPECT_THROW(Segment::build({core::TypedTensorView::contiguous(data_.data(), kRows, kDim),
                               space_,
                               space_},
                              build_options(),
                              denied_context),
               std::runtime_error);

  auto erased = Segment::into_any(std::move(segment_));
  ASSERT_TRUE(erased.ok());
  auto any = std::move(erased).value();
  const auto capabilities = any.capabilities();
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::save));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(core::OperationCapability::mutation));
}

TEST_F(NsgSegmentTest, ConcurrentSearchOnlyIsReentrant) {
  constexpr std::uint32_t thread_count = 8;
  constexpr std::uint32_t iterations = 64;
  std::atomic<std::uint32_t> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&, thread]() {
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto row = (thread * iterations + iteration) % kRows;
        SearchCall call(data_.data() + row * kDim, 1, kDim, 1, 24);
        const auto status = segment_->search(call.request);
        if (!status.ok() || call.counts[0] != 1 ||
            static_cast<std::uint64_t>(call.hits[0].row_id) != row) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  EXPECT_EQ(failures.load(), 0U);
}

}  // namespace
}  // namespace alaya
