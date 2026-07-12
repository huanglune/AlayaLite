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
#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/qg/detail/qg_builder_kernel.hpp"
#include "index/graph/qg/detail/qg_segment_bridge.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "space/rabitq_space.hpp"
#include "utils/openmp.hpp"

namespace alaya {
namespace {

using Space = RaBitQSpace<>;
using Segment = QgSegment<Space>;
using QgGraph = typename Segment::GraphType;
using SegmentSpaceView = typename Segment::SpaceViewType;

static_assert(!std::same_as<QgGraph, Space>);
static_assert(::alaya::Space<SegmentSpaceView>);
static_assert(is_rabitq_space_v<SegmentSpaceView>);

template <typename T>
concept CompleteQgLifecycle = requires(typename T::BuildInput input,
                                       const QgBuildOptions &build_options,
                                       core::BuildContext &build_context,
                                       core::ArtifactView artifact,
                                       const core::OpenOptions &open_options,
                                       core::OpenContext &open_context) {
  { T::build(input, build_options, build_context) } -> std::same_as<std::unique_ptr<T>>;
  { T::open(artifact, open_options, open_context) } -> std::same_as<std::unique_ptr<T>>;
};

static_assert(CompleteQgLifecycle<Segment>);
static_assert(core::Searchable<Segment>);
static_assert(core::BatchSearchable<Segment>);
static_assert(core::Saveable<Segment>);
static_assert(core::StatsProvider<Segment>);
static_assert(!core::Mutable<Segment>);

auto artifact_location(std::string_view path, std::string_view name = Segment::kArtifactName)
    -> std::array<core::ArtifactLocation, 1> {
  return {core::ArtifactLocation(name, path)};
}

struct SearchCall {
  SearchCall(const float *data,
             core::RowCount rows,
             std::uint32_t dim,
             std::uint64_t top_k,
             std::uint32_t effort)
      : hits(static_cast<std::size_t>(rows * top_k)),
        offsets(static_cast<std::size_t>(rows + 1)),
        counts(static_cast<std::size_t>(rows)),
        statuses(static_cast<std::size_t>(rows)),
        completeness(static_cast<std::size_t>(rows)) {
    qg.effort = effort;
    extension = make_qg_search_extension(qg);
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
  QgSearchExtension qg;
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
    throw std::runtime_error("cannot read QG test artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

class QgSegmentTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { spdlog::set_level(spdlog::level::warn); }

  void SetUp() override {
    // libgomp is not TSan-instrumented. Keep fixture fit/build serial so the
    // sanitizer run isolates the declared concurrent search-only profile.
    platform::set_openmp_thread_count(1);
    root_ = std::filesystem::temp_directory_path() /
            ("alayalite-qg-segment-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);

    data_.resize(kRows * kDim);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t col = 0; col < kDim; ++col) {
        data_[row * kDim + col] = std::sin(static_cast<float>((row + 1) * (col + 3)) * 0.03125F) +
                                  static_cast<float>(row % 11) * 0.01F;
      }
    }
    space_ = std::make_shared<Space>(kCapacity, kDim, MetricType::L2);
    space_->fit(data_.data(), kRows);
    core::BuildContext context;
    segment_ =
        Segment::build({core::TypedTensorView::contiguous(data_.data(), kRows, kDim), space_},
                       build_options(),
                       context);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static constexpr auto build_options() -> QgBuildOptions {
    return {.ef_build = 64, .thread_count = 1};
  }

  static constexpr std::uint32_t kRows = 128;
  static constexpr std::uint32_t kCapacity = 144;
  static constexpr std::uint32_t kDim = 64;
  std::filesystem::path root_;
  std::vector<float> data_;
  std::shared_ptr<Space> space_;
  std::unique_ptr<Segment> segment_;
};

TEST_F(QgSegmentTest, BuildSearchBatchDescriptorAndCapabilitiesFollowV3Contract) {
  const auto descriptor = segment_->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::qg);
  EXPECT_EQ(descriptor.engine_factory_id, core::algorithm::qg);
  EXPECT_EQ(descriptor.format_version, Segment::kFormatVersion);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, core::Medium::memory);
  EXPECT_EQ(descriptor.preprocessing, core::MetricPreprocessing::engine_quantized);

  core::SegmentStats stats;
  ASSERT_TRUE(segment_->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kCapacity);

  SearchCall single(data_.data(), 1, kDim, 4, 64);
  ASSERT_TRUE(segment_->search(single.request).ok());
  ASSERT_EQ(single.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(single.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(single.hits.front().score, 0.0F);

  SearchCall batch(data_.data(), 3, kDim, 4, 64);
  ASSERT_TRUE(segment_->batch_search(batch.request).ok());
  EXPECT_EQ(batch.offsets, (std::vector<core::RowCount>{0, 4, 8, 12}));
  EXPECT_EQ(batch.counts, (std::vector<core::RowCount>{4, 4, 4}));
  EXPECT_EQ(static_cast<std::uint64_t>(batch.hits[4].row_id), 1U);

  SearchCall wrong_rows(data_.data(), 2, kDim, 4, 64);
  EXPECT_EQ(segment_->search(wrong_rows.request).code(), core::StatusCode::invalid_argument);

  auto erased = Segment::into_any(std::move(segment_));
  ASSERT_TRUE(erased.ok());
  auto any = std::move(erased).value();
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::search));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::save));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::stats));
  EXPECT_FALSE(any.capabilities().supports(core::OperationCapability::mutation));
}

TEST_F(QgSegmentTest, RetainedCodecAndLegacyReaderRoundTripByteForByte) {
  const auto segment_path = (root_ / "segment.qg").string();
  const auto direct_path = (root_ / "direct.qg").string();
  const auto locations = artifact_location(segment_path);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.format_version, Segment::kFormatVersion);
  ASSERT_EQ(manifest.algorithm_id, core::algorithm::qg);
  ASSERT_EQ(manifest.artifacts.size(), 1U);
  EXPECT_EQ(manifest.artifacts[0].name, Segment::kArtifactName);

  // QgSegment delegates to the exact legacy RaBitQSpace codec.
  space_->save(direct_path);
  EXPECT_EQ(read_bytes(segment_path), read_bytes(direct_path));

  const auto legacy_locations = artifact_location(segment_path, Segment::kLegacyArtifactName);
  core::OpenContext open_context;
  auto reopened =
      Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(legacy_locations)),
                    {},
                    open_context);
  const auto roundtrip_path = (root_ / "roundtrip.qg").string();
  const auto roundtrip_locations = artifact_location(roundtrip_path);
  core::ArtifactWriter roundtrip_writer{
      std::span<const core::ArtifactLocation>(roundtrip_locations)};
  ASSERT_TRUE(reopened->save(roundtrip_writer, {}, manifest).ok());
  EXPECT_EQ(read_bytes(segment_path), read_bytes(roundtrip_path));
}

TEST_F(QgSegmentTest, SegmentOwnsGraphViewAndEntryPointWhileSpaceRetainsCodecBytes) {
  auto graph = detail::QgSegmentBridge<Space>::graph(*segment_);
  auto legacy_space = detail::QgSegmentBridge<Space>::space(*segment_);
  ASSERT_NE(graph, nullptr);
  ASSERT_NE(legacy_space, nullptr);

  // Adjacency remains zero-copy in the interleaved v1 codec slots, but the
  // Segment owns the graph object through which Segment build/search operate.
  EXPECT_EQ(graph->get_edges(0), legacy_space->get_edges(0));
  EXPECT_EQ(graph->get_ep(), legacy_space->get_ep());

  SearchCall before(data_.data(), 1, kDim, 8, 64);
  ASSERT_TRUE(segment_->search(before.request).ok());
  const auto owned_entry = graph->get_ep();
  legacy_space->set_ep((owned_entry + 1) % kRows);
  EXPECT_EQ(graph->get_ep(), owned_entry);
  EXPECT_NE(graph->get_ep(), legacy_space->get_ep());

  SearchCall after(data_.data(), 1, kDim, 8, 64);
  ASSERT_TRUE(segment_->search(after.request).ok());
  ASSERT_EQ(before.counts, after.counts);
  for (std::size_t hit = 0; hit < before.counts[0]; ++hit) {
    EXPECT_EQ(static_cast<std::uint64_t>(before.hits[hit].row_id),
              static_cast<std::uint64_t>(after.hits[hit].row_id));
    EXPECT_EQ(std::bit_cast<std::uint32_t>(before.hits[hit].score),
              std::bit_cast<std::uint32_t>(after.hits[hit].score));
  }

  // save() serializes the Segment authority back into the legacy v1 field.
  const auto artifact_path = (root_ / "owned-entry.qg").string();
  const auto locations = artifact_location(artifact_path);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  Space loaded;
  loaded.load(artifact_path);
  EXPECT_EQ(loaded.get_ep(), owned_entry);
}

TEST_F(QgSegmentTest, LoadedArtifactMatchesLegacySearchBitForBitAndMeetsRecallSanity) {
  const auto artifact_path = (root_ / "legacy.qg").string();
  space_->save(artifact_path);
  const auto locations = artifact_location(artifact_path, Segment::kLegacyArtifactName);
  core::OpenContext context;
  auto reopened =
      Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                    {},
                    context);
  auto direct_space = detail::QgSegmentBridge<Space>::space(*reopened);
  GraphSearchJob<Space> direct_search(direct_space, nullptr, nullptr, direct_space);

  constexpr std::uint32_t query_count = 8;
  constexpr std::uint32_t top_k = 8;
  constexpr std::uint32_t effort = 64;
  SearchCall v3(data_.data(), query_count, kDim, top_k, effort);
  ASSERT_TRUE(reopened->batch_search(v3.request).ok());
  std::array<std::uint32_t, top_k> direct_ids{};
  std::uint32_t exact_matches = 0;
  for (std::uint32_t row = 0; row < query_count; ++row) {
    direct_ids.fill(std::numeric_limits<std::uint32_t>::max());
    const auto *query = data_.data() + row * kDim;
    direct_search.rabitq_search_solo(query, top_k, direct_ids.data(), effort);
    ASSERT_EQ(v3.counts[row], top_k);
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto offset = static_cast<std::size_t>(v3.offsets[row] + hit);
      EXPECT_EQ(static_cast<std::uint64_t>(v3.hits[offset].row_id), direct_ids[hit]);
      const auto direct_score =
          direct_space->get_dist_func()(query, direct_space->get_data_by_id(direct_ids[hit]), kDim);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(v3.hits[offset].score),
                std::bit_cast<std::uint32_t>(direct_score));
    }

    std::vector<std::pair<float, std::uint32_t>> exact;
    exact.reserve(kRows);
    for (std::uint32_t candidate = 0; candidate < kRows; ++candidate) {
      exact.emplace_back(direct_space->get_dist_func()(query,
                                                       direct_space->get_data_by_id(candidate),
                                                       kDim),
                         candidate);
    }
    std::partial_sort(exact.begin(), exact.begin() + top_k, exact.end());
    for (const auto id : direct_ids) {
      exact_matches += static_cast<std::uint32_t>(
          std::find_if(exact.begin(), exact.begin() + top_k, [&](const auto &entry) {
            return entry.second == id;
          }) != exact.begin() + top_k);
    }
  }
  EXPECT_GE(static_cast<double>(exact_matches) / (query_count * top_k), 0.5);
}

TEST_F(QgSegmentTest, ContextBudgetsAreEnforced) {
  core::BuildContext denied_build;
  denied_build.growing_reservation = core::MemoryReservation(0);
  EXPECT_THROW(Segment::build({core::TypedTensorView::contiguous(data_.data(), kRows, kDim),
                               space_},
                              build_options(),
                              denied_build),
               std::runtime_error);

  const auto artifact_path = (root_ / "budget.qg").string();
  const auto locations = artifact_location(artifact_path);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  core::OpenContext denied_open;
  denied_open.resident_lease = core::MemoryLease(0);
  EXPECT_THROW(Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                             {},
                             denied_open),
               std::runtime_error);
}

TEST_F(QgSegmentTest, ConcurrentSearchOnlyIsReentrant) {
  constexpr std::uint32_t thread_count = 8;
  constexpr std::uint32_t iterations = 64;
  std::array<std::uint32_t, kRows> expected{};
  for (std::uint32_t row = 0; row < kRows; ++row) {
    SearchCall call(data_.data() + row * kDim, 1, kDim, 1, 64);
    ASSERT_TRUE(segment_->search(call.request).ok());
    expected[row] =
        static_cast<std::uint32_t>(static_cast<std::uint64_t>(call.hits.front().row_id));
  }

  std::atomic<std::uint32_t> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&, thread]() {
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto row = (thread * iterations + iteration) % kRows;
        SearchCall call(data_.data() + row * kDim, 1, kDim, 1, 64);
        const auto status = segment_->search(call.request);
        if (!status.ok() || call.counts[0] != 1 ||
            static_cast<std::uint64_t>(call.hits[0].row_id) != expected[row]) {
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
