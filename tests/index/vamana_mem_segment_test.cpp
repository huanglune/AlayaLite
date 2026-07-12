// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
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
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_greedy_search.hpp"
#include "index/graph/vamana/vamana_mem_segment.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "index/memory_engine_registry.hpp"

namespace alaya {
namespace {

using Segment = VamanaMemSegment;

template <typename T>
concept CompleteVamanaMemLifecycle = requires(typename T::BuildInput input,
                                              const VamanaMemBuildOptions &build_options,
                                              core::BuildContext &build_context,
                                              core::ArtifactView artifact,
                                              const core::OpenOptions &open_options,
                                              core::OpenContext &open_context) {
  { T::build(input, build_options, build_context) } -> std::same_as<std::unique_ptr<T>>;
  { T::open(artifact, open_options, open_context) } -> std::same_as<std::unique_ptr<T>>;
};

static_assert(CompleteVamanaMemLifecycle<Segment>);
static_assert(core::Searchable<Segment>);
static_assert(core::BatchSearchable<Segment>);
static_assert(core::Saveable<Segment>);
static_assert(core::StatsProvider<Segment>);
static_assert(!core::Mutable<Segment>);

auto artifact_locations(std::string_view graph, std::string_view data)
    -> std::array<core::ArtifactLocation, 2> {
  return {core::ArtifactLocation(Segment::kGraphArtifactName, graph),
          core::ArtifactLocation(Segment::kDataArtifactName, data)};
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
    vamana.effort = effort;
    extension = make_vamana_mem_search_extension(vamana);
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    request.queries = core::TypedTensorView::contiguous(data, rows, dim);
    request.options.top_k = top_k;
    request.options.extensions =
        std::span<const core::AlgorithmSearchExtension>(std::addressof(extension), 1);
    request.context = std::addressof(context);
    request.response = std::addressof(response);
  }

  core::SearchContext context;
  VamanaMemSearchExtension vamana;
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
    throw std::runtime_error("cannot read Vamana-memory test artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_fbin(const std::filesystem::path &path,
                const std::vector<float> &vectors,
                std::uint32_t rows,
                std::uint32_t dim) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char *>(std::addressof(rows)), sizeof(rows));
  output.write(reinterpret_cast<const char *>(std::addressof(dim)), sizeof(dim));
  output.write(reinterpret_cast<const char *>(vectors.data()),
               static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!output) {
    throw std::runtime_error("cannot write direct fbin fixture");
  }
}

auto read_fbin(const std::filesystem::path &path,
               std::uint32_t expected_rows,
               std::uint32_t expected_dim) -> std::vector<float> {
  std::ifstream input(path, std::ios::binary);
  std::uint32_t rows = 0;
  std::uint32_t dim = 0;
  input.read(reinterpret_cast<char *>(std::addressof(rows)), sizeof(rows));
  input.read(reinterpret_cast<char *>(std::addressof(dim)), sizeof(dim));
  if (!input || rows != expected_rows || dim != expected_dim) {
    throw std::runtime_error("unexpected direct fbin fixture header");
  }
  std::vector<float> vectors(static_cast<std::size_t>(rows) * dim);
  input.read(reinterpret_cast<char *>(vectors.data()),
             static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("unexpected direct fbin fixture payload");
  }
  return vectors;
}

class VamanaMemSegmentTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { spdlog::set_level(spdlog::level::warn); }

  void SetUp() override {
    root_ =
        std::filesystem::temp_directory_path() /
        ("alayalite-vamana-mem-segment-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    data_.resize(kRows * kDim);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      for (std::uint32_t col = 0; col < kDim; ++col) {
        data_[row * kDim + col] = std::sin(static_cast<float>((row + 3) * (col + 5)) * 0.041F) +
                                  static_cast<float>((row * 13 + col * 7) % 31) * 0.007F;
      }
    }
    core::BuildContext context;
    segment_ = Segment::build(Segment::BuildInput(
                                  core::TypedTensorView::contiguous(data_.data(), kRows, kDim)),
                              build_options(),
                              context);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static constexpr auto build_options() -> VamanaMemBuildOptions {
    return {.max_neighbors = 8,
            .construction_effort = 32,
            .alpha = 1.2F,
            .thread_count = 1,
            .max_candidates = 64,
            .seed = 424242};
  }

  static constexpr std::uint32_t kRows = 128;
  static constexpr std::uint32_t kDim = 8;
  std::filesystem::path root_;
  std::vector<float> data_;
  std::unique_ptr<Segment> segment_;
};

TEST_F(VamanaMemSegmentTest, BuildSearchBatchDescriptorAndCapabilitiesFollowV3Contract) {
  const auto descriptor = segment_->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::vamana);
  EXPECT_EQ(descriptor.engine_factory_id, core::algorithm::vamana);
  EXPECT_EQ(descriptor.format_version, Segment::kFormatVersion);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, core::Medium::memory);
  EXPECT_EQ(descriptor.preprocessing, core::MetricPreprocessing::none);

  core::SegmentStats stats;
  ASSERT_TRUE(segment_->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kRows);
  EXPECT_GT(stats.resident_bytes, kRows * kDim * sizeof(float));

  SearchCall single(data_.data(), 1, kDim, 4, 32);
  ASSERT_TRUE(segment_->search(single.request).ok());
  ASSERT_EQ(single.counts[0], 4U);
  EXPECT_EQ(static_cast<std::uint64_t>(single.hits.front().row_id), 0U);
  EXPECT_FLOAT_EQ(single.hits.front().score, 0.0F);

  SearchCall batch(data_.data(), 3, kDim, 4, 32);
  ASSERT_TRUE(segment_->batch_search(batch.request).ok());
  EXPECT_EQ(batch.offsets, (std::vector<core::RowCount>{0, 4, 8, 12}));
  EXPECT_EQ(batch.counts, (std::vector<core::RowCount>{4, 4, 4}));
  EXPECT_EQ(static_cast<std::uint64_t>(batch.hits[4].row_id), 1U);

  SearchCall wrong_rows(data_.data(), 2, kDim, 4, 32);
  EXPECT_EQ(segment_->search(wrong_rows.request).code(), core::StatusCode::invalid_argument);
  std::array<std::int8_t, kDim> wrong_scalar{};
  SearchCall wrong_type(reinterpret_cast<const float *>(wrong_scalar.data()), 1, kDim, 4, 32);
  wrong_type.request.queries = core::TypedTensorView::contiguous(wrong_scalar.data(), 1, kDim);
  EXPECT_EQ(segment_->search(wrong_type.request).detail(),
            core::StatusDetail::unsupported_scalar_type);

  auto erased = Segment::into_any(std::move(segment_));
  ASSERT_TRUE(erased.ok());
  auto any = std::move(erased).value();
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::search));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::save));
  EXPECT_TRUE(any.capabilities().supports(core::OperationCapability::stats));
  EXPECT_FALSE(any.capabilities().supports(core::OperationCapability::mutation));
}

TEST_F(VamanaMemSegmentTest, DirectKernelDifferentialIsByteAndSearchBitExactWithRecallSanity) {
  const auto segment_graph = root_ / "segment.index";
  const auto segment_data = root_ / "segment.fbin";
  const auto direct_graph = root_ / "direct.index";
  const auto direct_data = root_ / "direct.fbin";
  const auto segment_graph_string = segment_graph.string();
  const auto segment_data_string = segment_data.string();
  const auto locations = artifact_locations(segment_graph_string, segment_data_string);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  ASSERT_EQ(manifest.schema_version, 1U);
  ASSERT_EQ(manifest.format_version, Segment::kFormatVersion);
  ASSERT_EQ(manifest.algorithm_id, core::algorithm::vamana);
  ASSERT_EQ(manifest.artifacts.size(), 2U);
  EXPECT_EQ(manifest.artifacts[0].name, Segment::kGraphArtifactName);
  EXPECT_EQ(manifest.artifacts[1].name, Segment::kDataArtifactName);

  vamana::VamanaBuildParams direct_params;
  direct_params.R = build_options().max_neighbors;
  direct_params.L = build_options().construction_effort;
  direct_params.alpha = build_options().alpha;
  direct_params.num_threads = build_options().thread_count;
  direct_params.maxc = build_options().max_candidates;
  direct_params.seed = build_options().seed;
  vamana::VamanaBuilder direct_builder(data_.data(), kRows, kDim, direct_params);
  direct_builder.build();
  vamana::save_graph(direct_builder.graph(),
                     direct_graph,
                     direct_params.R,
                     direct_builder.medoid());
  write_fbin(direct_data, data_, kRows, kDim);
  EXPECT_EQ(read_bytes(segment_graph), read_bytes(direct_graph));
  EXPECT_EQ(read_bytes(segment_data), read_bytes(direct_data));

  // Direct writer output plus the original fbin opens in the Segment.
  const auto direct_graph_string = direct_graph.string();
  const auto direct_data_string = direct_data.string();
  const auto direct_locations = artifact_locations(direct_graph_string, direct_data_string);
  core::OpenContext open_context;
  auto opened_direct =
      Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(direct_locations)),
                    {},
                    open_context);

  // Segment writer output opens through the retained reader/search kernels.
  vamana::VamanaReader direct_reader(segment_graph);
  auto direct_vectors = read_fbin(segment_data, kRows, kDim);
  vamana::VamanaGreedySearch direct_search(direct_reader, direct_vectors.data(), kDim);

  constexpr std::uint32_t query_count = 8;
  constexpr std::uint32_t top_k = 8;
  constexpr std::uint32_t effort = 64;
  SearchCall segment_results(data_.data(), query_count, kDim, top_k, effort);
  ASSERT_TRUE(opened_direct->batch_search(segment_results.request).ok());
  std::uint32_t exact_matches = 0;
  for (std::uint32_t row = 0; row < query_count; ++row) {
    const auto *query = data_.data() + row * kDim;
    const auto direct_hits = direct_search.search(query, top_k, effort);
    ASSERT_EQ(direct_hits.size(), top_k);
    ASSERT_EQ(segment_results.counts[row], top_k);
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto offset = static_cast<std::size_t>(segment_results.offsets[row] + hit);
      EXPECT_EQ(static_cast<std::uint64_t>(segment_results.hits[offset].row_id),
                direct_hits[hit].id);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(segment_results.hits[offset].score),
                std::bit_cast<std::uint32_t>(direct_hits[hit].distance));
    }

    std::vector<std::pair<float, std::uint32_t>> exact;
    exact.reserve(kRows);
    for (std::uint32_t candidate = 0; candidate < kRows; ++candidate) {
      float distance = 0.0F;
      for (std::uint32_t col = 0; col < kDim; ++col) {
        const auto diff = query[col] - data_[candidate * kDim + col];
        distance += diff * diff;
      }
      exact.emplace_back(distance, candidate);
    }
    std::partial_sort(exact.begin(), exact.begin() + top_k, exact.end());
    for (const auto &hit : direct_hits) {
      exact_matches += static_cast<std::uint32_t>(
          std::find_if(exact.begin(), exact.begin() + top_k, [&](const auto &entry) {
            return entry.second == hit.id;
          }) != exact.begin() + top_k);
    }
  }
  const auto recall = static_cast<double>(exact_matches) / (query_count * top_k);
  RecordProperty("recall_at_8", recall);
  EXPECT_GE(recall, 0.7);

  const auto roundtrip_graph = root_ / "roundtrip.index";
  const auto roundtrip_data = root_ / "roundtrip.fbin";
  const auto roundtrip_graph_string = roundtrip_graph.string();
  const auto roundtrip_data_string = roundtrip_data.string();
  const auto roundtrip_locations =
      artifact_locations(roundtrip_graph_string, roundtrip_data_string);
  core::ArtifactWriter roundtrip_writer{
      std::span<const core::ArtifactLocation>(roundtrip_locations)};
  ASSERT_TRUE(opened_direct->save(roundtrip_writer, {}, manifest).ok());
  EXPECT_EQ(read_bytes(direct_graph), read_bytes(roundtrip_graph));
  EXPECT_EQ(read_bytes(direct_data), read_bytes(roundtrip_data));
}

TEST_F(VamanaMemSegmentTest, TypedTensorContextsAndArtifactTransactionAreEnforced) {
  std::vector<float> padded(kRows * (kDim + 3), -1000.0F);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    std::copy_n(data_.data() + row * kDim, kDim, padded.data() + row * (kDim + 3));
  }
  core::TypedTensorView strided(padded.data(),
                                core::ScalarType::float32,
                                kRows,
                                kDim,
                                (kDim + 3) * sizeof(float));
  core::BuildContext build_context;
  auto strided_segment =
      Segment::build(Segment::BuildInput(strided), build_options(), build_context);
  SearchCall strided_search(data_.data(), 1, kDim, 1, 32);
  ASSERT_TRUE(strided_segment->search(strided_search.request).ok());
  EXPECT_EQ(static_cast<std::uint64_t>(strided_search.hits[0].row_id), 0U);

  core::BuildContext denied_build;
  denied_build.growing_reservation = core::MemoryReservation(0);
  EXPECT_THROW(Segment::build(Segment::BuildInput(
                                  core::TypedTensorView::contiguous(data_.data(), kRows, kDim)),
                              build_options(),
                              denied_build),
               std::runtime_error);

  const auto graph = root_ / "budget.index";
  const auto data = root_ / "budget.fbin";
  const auto graph_string = graph.string();
  const auto data_string = data.string();
  const auto locations = artifact_locations(graph_string, data_string);
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  ASSERT_TRUE(segment_->save(writer, {}, manifest).ok());
  core::OpenContext denied_open;
  denied_open.resident_lease = core::MemoryLease(0);
  EXPECT_THROW(Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations)),
                             {},
                             denied_open),
               std::runtime_error);

  const auto uncommitted_graph = root_ / "must-not-exist.index";
  const auto uncommitted_graph_string = uncommitted_graph.string();
  const std::array incomplete{
      core::ArtifactLocation(Segment::kGraphArtifactName, uncommitted_graph_string)};
  core::ArtifactWriter incomplete_writer{std::span<const core::ArtifactLocation>(incomplete)};
  const auto status = segment_->save(incomplete_writer, {}, manifest);
  EXPECT_EQ(status.code(), core::StatusCode::invalid_argument);
  EXPECT_FALSE(std::filesystem::exists(uncommitted_graph));
}

TEST_F(VamanaMemSegmentTest, StandaloneFactoryHasNoLegacyFallback) {
  using internal::memory::EngineFeature;
  using internal::memory::EngineRole;
  EXPECT_EQ(VamanaMemSegmentFactory::registration.role, EngineRole::searchable_segment);
  EXPECT_EQ(VamanaMemSegmentFactory::registration.feature, EngineFeature::vamana_memory);
  EXPECT_FALSE(VamanaMemSegmentFactory::registration.has_legacy_factory);
  EXPECT_EQ(VamanaMemSegmentFactory::registration.legacy.implementation_key, "none");
  internal::memory::MemoryEngineFeatureFlags disabled;
  disabled.vamana_memory_segment = false;
  core::BuildContext disabled_context;
  auto unavailable =
      VamanaMemSegmentFactory::build(Segment::BuildInput(
                                         core::TypedTensorView::contiguous(data_.data(),
                                                                           kRows,
                                                                           kDim)),
                                     build_options(),
                                     disabled_context,
                                     disabled);
  EXPECT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), core::StatusCode::not_supported);

  core::BuildContext enabled_context;
  auto available =
      VamanaMemSegmentFactory::build(Segment::BuildInput(
                                         core::TypedTensorView::contiguous(data_.data(),
                                                                           kRows,
                                                                           kDim)),
                                     build_options(),
                                     enabled_context);
  ASSERT_TRUE(available.ok()) << available.status().diagnostic();
  EXPECT_EQ(std::move(available).value()->descriptor().algorithm_id, core::algorithm::vamana);
}

TEST_F(VamanaMemSegmentTest, ConcurrentSearchOnlyIsReentrant) {
  constexpr std::uint32_t thread_count = 8;
  constexpr std::uint32_t iterations = 64;
  std::array<std::uint32_t, kRows> expected{};
  for (std::uint32_t row = 0; row < kRows; ++row) {
    SearchCall call(data_.data() + row * kDim, 1, kDim, 1, 32);
    ASSERT_TRUE(segment_->search(call.request).ok());
    expected[row] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(call.hits[0].row_id));
  }

  std::atomic<std::uint32_t> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  for (std::uint32_t thread = 0; thread < thread_count; ++thread) {
    threads.emplace_back([&, thread]() {
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto row = (thread * iterations + iteration) % kRows;
        SearchCall call(data_.data() + row * kDim, 1, kDim, 1, 32);
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
