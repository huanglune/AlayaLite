// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// Production metric wiring for the sealed/read-only LASER Collection target.
// IP compares the direct LASER id with the public qg-to-LASER route on
// identical non-unit data;
// cosine includes a high-norm distractor that wins raw IP but must lose after
// row normalization. Both cases verify native/Collection proofs and reopen.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/detail/collection_target_builder.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/laser_segment_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/unified_laser_segment_searcher.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "platform/detect.hpp"
#include "simd/distance_ip.hpp"
#include "utils/evaluate.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr core::RowCount kRows = 192;
constexpr core::RowCount kQueryCount = 12;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-laser-metric-wiring-" + std::string(name) + "-" +
             std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  TemporaryDirectory(const TemporaryDirectory &) = delete;
  auto operator=(const TemporaryDirectory &) -> TemporaryDirectory & = delete;

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct Dataset {
  std::vector<float> vectors{};
  std::vector<core::LogicalId> ids{};
};

[[nodiscard]] auto splitmix64(std::uint64_t &state) -> std::uint64_t {
  state += 0x9E3779B97F4A7C15ULL;
  auto value = state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] auto make_unit_directions(core::RowCount rows, std::uint64_t seed) -> Dataset {
  Dataset result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{seed};
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("row-" + std::to_string(row)));
    double squared_norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto sample = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
      const auto value = static_cast<float>(sample) / static_cast<float>(1U << 23U) - 1.0F;
      result.vectors[static_cast<std::size_t>(row * kDim + column)] = value;
      squared_norm += static_cast<double>(value) * value;
    }
    const auto inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= inverse_norm;
    }
  }
  return result;
}

[[nodiscard]] auto make_ip_dataset() -> Dataset {
  auto result = make_unit_directions(kRows, 0xA11A2026'0718'1A01ULL);
  for (core::RowCount row = 0; row < kRows; ++row) {
    const auto scale = 0.25F + static_cast<float>((row * 17U) % 29U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale;
    }
  }
  return result;
}

[[nodiscard]] auto make_cosine_dataset() -> Dataset {
  auto result = make_unit_directions(kRows, 0xA11A2026'0718'C051ULL);
  for (core::RowCount row = 0; row < kRows; ++row) {
    const auto scale = 0.05F + static_cast<float>((row * 19U) % 37U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale;
    }
  }

  // Query direction is +e0. row-0 is the true cosine optimum despite its tiny
  // norm. row-1 has lower cosine (0.8) but raw dot 800, so an unnormalized
  // LASER build would rank the distractor ahead of row-0 (raw dot 0.01).
  std::fill_n(result.vectors.begin(), kDim * 2, 0.0F);
  result.vectors[0] = 0.01F;
  result.vectors[kDim] = 800.0F;
  result.vectors[kDim + 1] = 600.0F;
  return result;
}

[[nodiscard]] auto make_ip_queries(const Dataset &dataset) -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(kQueryCount * kDim));
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto source = (query * 23U + 7U) % kRows;
    const auto scale = 0.5F + static_cast<float>((query * 11U) % 13U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 7U) - 3) * 0.0005F;
      queries[static_cast<std::size_t>(query * kDim + column)] =
          scale * (dataset.vectors[static_cast<std::size_t>(source * kDim + column)] +
                   perturbation);
    }
  }
  return queries;
}

[[nodiscard]] auto make_cosine_queries(const Dataset &dataset) -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(kQueryCount * kDim));
  queries[0] = 7.0F;  // deliberately non-unit; Collection must normalize it.
  for (core::RowCount query = 1; query < kQueryCount; ++query) {
    const auto source = (query * 29U + 5U) % kRows;
    const auto scale = 0.2F + static_cast<float>(query * 3U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      queries[static_cast<std::size_t>(query * kDim + column)] =
          scale * dataset.vectors[static_cast<std::size_t>(source * kDim + column)];
    }
  }
  return queries;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root,
                                core::Metric metric,
                                core::AlgorithmId algorithm) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = metric;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = algorithm;
  options.quantization = CollectionQuantization::rabitq;
  options.max_neighbors = 32;
  options.ef_construction = 256;
  options.build_threads = 4;
  return options;
}

void insert_dataset(Collection &collection, const Dataset &dataset) {
  std::vector<CollectionItem> items;
  items.reserve(dataset.ids.size());
  for (core::RowCount row = 0; row < dataset.ids.size(); ++row) {
    CollectionItem item;
    item.logical_id = dataset.ids[static_cast<std::size_t>(row)];
    item.vector = core::TypedTensorView::contiguous(dataset.vectors.data() +
                                                        static_cast<std::ptrdiff_t>(row * kDim),
                                                    1,
                                                    kDim);
    items.push_back(std::move(item));
  }
  auto added = collection.add_batch(items, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  ASSERT_EQ(added.value().rows.size(), dataset.ids.size());
}

[[nodiscard]] auto exact_oracle(const Dataset &dataset,
                                const std::vector<float> &queries,
                                core::Metric metric) -> std::vector<std::uint32_t> {
  return find_exact_gt<float, float, std::uint32_t>(queries,
                                                    dataset.vectors,
                                                    kDim,
                                                    static_cast<std::uint32_t>(kTopK),
                                                    metric);
}

[[nodiscard]] auto recall_at_k(const CollectionSearchResponse &response,
                               const std::vector<std::uint32_t> &oracle,
                               const Dataset &dataset) -> double {
  std::uint64_t matches{};
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    for (core::RowCount rank = 0; rank < kTopK; ++rank) {
      const auto &expected =
          dataset.ids[oracle[static_cast<std::size_t>(query * kTopK + rank)]];
      if (std::find(response.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                    response.ids.begin() + static_cast<std::ptrdiff_t>(end),
                    expected) != response.ids.begin() + static_cast<std::ptrdiff_t>(end)) {
        ++matches;
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(kQueryCount * kTopK);
}

[[nodiscard]] auto exact_score(const float *query,
                               const float *candidate,
                               core::Metric metric) -> float {
  double dot{};
  double query_norm{};
  double candidate_norm{};
  for (std::uint32_t column = 0; column < kDim; ++column) {
    dot += static_cast<double>(query[column]) * candidate[column];
    query_norm += static_cast<double>(query[column]) * query[column];
    candidate_norm += static_cast<double>(candidate[column]) * candidate[column];
  }
  if (metric == core::Metric::inner_product) {
    return static_cast<float>(-dot);
  }
  if (query_norm == 0.0 || candidate_norm == 0.0) {
    return 0.0F;
  }
  return static_cast<float>(-dot / std::sqrt(query_norm * candidate_norm));
}

void expect_exact_sorted_distances(const CollectionSearchResponse &response,
                                   const Dataset &dataset,
                                   const std::vector<float> &queries,
                                   core::Metric metric) {
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    float previous = -std::numeric_limits<float>::infinity();
    for (core::RowCount index = begin; index < end; ++index) {
      const auto found = std::ranges::find(dataset.ids, response.ids[index]);
      ASSERT_NE(found, dataset.ids.end());
      const auto row = static_cast<std::size_t>(std::distance(dataset.ids.begin(), found));
      const auto expected = exact_score(queries.data() + query * kDim,
                                        dataset.vectors.data() + row * kDim,
                                        metric);
      EXPECT_NEAR(response.distances[index], expected, 2.0e-4F);
      EXPECT_LE(previous, response.distances[index]);
      previous = response.distances[index];
    }
  }
}

template <class Searcher>
void expect_native_result_contract(Searcher &searcher,
                                   const float *native_query,
                                   const float *oracle_query,
                                   const Dataset &dataset,
                                   core::Metric metric) {
  disk::DiskSearchOptions options;
  options.top_k = kTopK;
  options.ef = 256;
  const auto rank_only = searcher.search(native_query, options);
  ASSERT_EQ(rank_only.size(), kTopK);
  for (const auto &hit : rank_only) {
    const auto bits = std::bit_cast<std::uint32_t>(hit.distance);
    EXPECT_EQ(bits & 0x7F800000U, 0x7F800000U);
    EXPECT_NE(bits & 0x007FFFFFU, 0U);
  }

  options.return_distances = true;
  const auto numeric = searcher.search(native_query, options);
  ASSERT_EQ(numeric.size(), kTopK);
  std::vector<std::size_t> native_order(dataset.ids.size());
  std::iota(native_order.begin(), native_order.end(), std::size_t{0});
  std::sort(native_order.begin(), native_order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return dataset.ids[lhs].compare(dataset.ids[rhs]) < 0;
  });
  float previous = -std::numeric_limits<float>::infinity();
  for (const auto &hit : numeric) {
    ASSERT_LT(hit.label, native_order.size());
    const auto source_row = native_order[hit.label];
    const auto expected = exact_score(oracle_query,
                                      dataset.vectors.data() + source_row * kDim,
                                      metric);
    const float *native_candidate = dataset.vectors.data() + source_row * kDim;
    std::vector<float> normalized_candidate;
    if (metric == core::Metric::cosine) {
      normalized_candidate.assign(native_candidate, native_candidate + kDim);
      const auto normalize_status = internal::collection::detail::l2_normalize_float_rows(
          normalized_candidate, kDim, core::OperationStage::validation);
      ASSERT_TRUE(normalize_status.ok()) << normalize_status.diagnostic();
      native_candidate = normalized_candidate.data();
    }
    const auto qg_distance_domain =
        ::alaya::simd::ip_sqr<float, float>(native_query, native_candidate, kDim);
    EXPECT_TRUE(std::isfinite(hit.distance));
    EXPECT_NEAR(hit.distance, expected, 2.0e-4F);
    EXPECT_FLOAT_EQ(hit.distance, qg_distance_domain);
    EXPECT_LE(previous, hit.distance);
    previous = hit.distance;
  }
}

struct LaserArtifacts {
  internal::collection::SegmentEntryV2 entry{};
  std::filesystem::path segment_directory{};
  disk::SegmentManifest native{};
};

[[nodiscard]] auto load_laser_artifacts(const std::filesystem::path &root,
                                        const CollectionSealReceipt &receipt) -> LaserArtifacts {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  if (target == manifest.segments.end()) {
    throw std::runtime_error("sealed LASER entry is absent from Collection manifest");
  }
  LaserArtifacts result;
  result.entry = *target;
  result.segment_directory =
      root / "segments" /
      internal::collection::detail::collection_segment_name(receipt.sealed_segment_id);
  result.native = disk::SegmentManifest::load(result.segment_directory / "manifest.txt");
  return result;
}

void expect_laser_identity_and_semantics(const LaserArtifacts &artifacts,
                                         core::Metric metric,
                                         std::string_view preprocessing) {
  EXPECT_EQ(artifacts.entry.algorithm_id, core::algorithm::laser);
  EXPECT_EQ(artifacts.entry.factory_key, "laser");
  EXPECT_EQ(artifacts.entry.reader_compatibility.required_features,
            (std::vector<std::string>{"disk_laser_segment"}));
  ASSERT_TRUE(artifacts.entry.extensions.contains("preprocessing"));
  EXPECT_EQ(artifacts.entry.extensions.at("preprocessing"), preprocessing);
  EXPECT_EQ(artifacts.native.metric, metric);
  ASSERT_TRUE(artifacts.native.x_extras.contains("x_laser_preprocessing"));
  EXPECT_EQ(artifacts.native.x_extras.at("x_laser_preprocessing"), preprocessing);

  const auto index_path =
      artifacts.segment_directory / artifacts.native.x_extras.at("x_laser_index_file");
  EXPECT_NO_THROW(laser::qg_validate_native_semantics_file(
      index_path, metric, laser::qg_expected_preprocessing(metric)));
}

void expect_collection_descriptor(const std::filesystem::path &root,
                                  const internal::collection::SegmentEntryV2 &entry,
                                  core::Metric metric,
                                  core::MetricPreprocessing preprocessing) {
  core::OpenContext context;
  const internal::collection::CollectionSchema schema{kDim, metric, core::ScalarType::float32};
  auto opened = internal::collection::detail::open_laser_collection_target(root,
                                                                            entry,
                                                                            schema,
                                                                            context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto descriptor = opened.value().descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::laser);
  EXPECT_EQ(descriptor.metric, metric);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.preprocessing, preprocessing);
}

void expect_target_numeric_result_contract(
    const std::filesystem::path &root,
    const internal::collection::SegmentEntryV2 &entry,
    const Dataset &dataset,
    const float *query,
    core::Metric metric) {
  core::OpenContext open_context;
  const internal::collection::CollectionSchema schema{kDim, metric, core::ScalarType::float32};
  auto opened = internal::collection::detail::open_laser_collection_target(root,
                                                                            entry,
                                                                            schema,
                                                                            open_context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto target = std::move(opened).value();

  disk::LaserSegmentSearchExtension extension_options;
  extension_options.effort = 256;
  extension_options.return_distances = true;
  auto extension = disk::LaserSegment::make_search_extension(extension_options);
  std::vector<core::SearchHit> hits(kTopK);
  std::vector<core::RowCount> offsets(2);
  std::vector<core::RowCount> counts(1);
  std::vector<core::Status> statuses(1);
  std::vector<core::SearchCompleteness> completeness(1);
  core::SearchResponse response;
  response.hits = hits;
  response.offsets = offsets;
  response.valid_counts = counts;
  response.statuses = statuses;
  response.completeness = completeness;
  core::SearchContext search_context;
  core::SearchRequest request;
  request.queries = core::TypedTensorView::contiguous(query, 1, kDim);
  request.options.top_k = kTopK;
  request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
  request.context = &search_context;
  request.response = &response;
  const auto status = target.search(std::move(request));
  ASSERT_TRUE(status.ok()) << status.diagnostic();
  ASSERT_EQ(counts[0], kTopK);
  EXPECT_EQ(response.score_kind, core::ScoreKind::distance);
  EXPECT_EQ(response.comparable_metric, metric);

  std::vector<std::size_t> native_order(dataset.ids.size());
  std::iota(native_order.begin(), native_order.end(), std::size_t{0});
  std::sort(native_order.begin(), native_order.end(), [&](std::size_t lhs, std::size_t rhs) {
    return dataset.ids[lhs].compare(dataset.ids[rhs]) < 0;
  });
  for (std::size_t index = 0; index < counts[0]; ++index) {
    ASSERT_EQ(hits[index].score_kind, core::ScoreKind::distance);
    const auto native_row = static_cast<std::uint64_t>(hits[index].row_id);
    ASSERT_LT(native_row, native_order.size());
    const auto source_row = native_order[native_row];
    const auto expected = exact_score(query,
                                      dataset.vectors.data() + source_row * kDim,
                                      metric);
    EXPECT_NEAR(hits[index].score, expected, 2.0e-4F);
  }
}

TEST(CollectionLaserMetricWiring, InnerProductMatchesPublicQgRouteOnSameNonUnitDataAndReopens) {
  const auto dataset = make_ip_dataset();
  const auto queries = make_ip_queries(dataset);
  const auto oracle = exact_oracle(dataset, queries, core::Metric::inner_product);
  TemporaryDirectory laser_directory("ip-laser");
  TemporaryDirectory qg_directory("ip-qg");

  auto laser_created = Collection::create(
      make_options(laser_directory.path(), core::Metric::inner_product, core::algorithm::laser));
  ASSERT_TRUE(laser_created.ok()) << laser_created.status().diagnostic();
  auto laser_collection = std::move(laser_created).value();
  insert_dataset(*laser_collection, dataset);
  auto laser_sealed = laser_collection->seal();
  ASSERT_TRUE(laser_sealed.ok()) << laser_sealed.status().diagnostic();
  EXPECT_EQ(laser_sealed.value().built_algorithm, core::algorithm::laser);
  EXPECT_FALSE(laser_sealed.value().flat_fallback) << laser_sealed.value().fallback_reason;

  const auto artifacts = load_laser_artifacts(laser_directory.path(), laser_sealed.value());
  expect_laser_identity_and_semantics(artifacts, core::Metric::inner_product, "none");
  expect_collection_descriptor(laser_directory.path(),
                               artifacts.entry,
                               core::Metric::inner_product,
                               core::MetricPreprocessing::none);
  expect_target_numeric_result_contract(laser_directory.path(),
                                        artifacts.entry,
                                        dataset,
                                        queries.data(),
                                        core::Metric::inner_product);
  disk::LaserSegmentSearcher native_searcher(artifacts.segment_directory);
  expect_native_result_contract(native_searcher,
                                queries.data(),
                                queries.data(),
                                dataset,
                                core::Metric::inner_product);
  disk::UnifiedLaserSegmentSearcher arena_searcher(artifacts.segment_directory,
                                                    laser::ResidencyMode::kResidentArena);
  expect_native_result_contract(arena_searcher,
                                queries.data(),
                                queries.data(),
                                dataset,
                                core::Metric::inner_product);

  auto laser_response = laser_collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(laser_response.ok()) << laser_response.status().diagnostic();
  expect_exact_sorted_distances(laser_response.value(),
                                dataset,
                                queries,
                                core::Metric::inner_product);
  const auto laser_recall = recall_at_k(laser_response.value(), oracle, dataset);

  ASSERT_TRUE(laser_collection->close().ok());
  laser_collection.reset();
  auto reopened_result = Collection::open(laser_directory.path());
  ASSERT_TRUE(reopened_result.ok()) << reopened_result.status().diagnostic();
  auto reopened = std::move(reopened_result).value();
  auto reopened_response = reopened->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(reopened_response.ok()) << reopened_response.status().diagnostic();
  const auto reopened_recall = recall_at_k(reopened_response.value(), oracle, dataset);
  expect_exact_sorted_distances(reopened_response.value(),
                                dataset,
                                queries,
                                core::Metric::inner_product);

  auto qg_created = Collection::create(
      make_options(qg_directory.path(), core::Metric::inner_product, core::algorithm::qg));
  ASSERT_TRUE(qg_created.ok()) << qg_created.status().diagnostic();
  auto qg_collection = std::move(qg_created).value();
  insert_dataset(*qg_collection, dataset);
  auto qg_sealed = qg_collection->seal();
  ASSERT_TRUE(qg_sealed.ok()) << qg_sealed.status().diagnostic();
  EXPECT_EQ(qg_sealed.value().built_algorithm, core::algorithm::qg);
  EXPECT_FALSE(qg_sealed.value().flat_fallback);
  auto qg_response = qg_collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(qg_response.ok()) << qg_response.status().diagnostic();
  const auto qg_recall = recall_at_k(qg_response.value(), oracle, dataset);

  std::cout << "measured_collection_laser_ip_recall_at_10=" << std::fixed
            << std::setprecision(4) << laser_recall << '\n';
  std::cout << "measured_collection_laser_ip_reopen_recall_at_10=" << reopened_recall << '\n';
  std::cout << "measured_collection_qg_same_data_ip_recall_at_10=" << qg_recall << '\n';

  // The same-data public qg route is the topology/quality comparator. The
  // absolute floor catches a shared collapse; the relative margin tolerates
  // disk traversal scheduling while still preventing a materially weaker
  // LASER path.
  EXPECT_GE(laser_recall, 0.70);
  EXPECT_GE(reopened_recall, 0.70);
  EXPECT_GE(laser_recall, qg_recall - 0.20);
  EXPECT_GE(reopened_recall, qg_recall - 0.20);
  EXPECT_TRUE(reopened->close().ok());
  EXPECT_TRUE(qg_collection->close().ok());
}

TEST(CollectionLaserMetricWiring, CosineNormalizesRowsWrapsQueriesAndReopens) {
  const auto dataset = make_cosine_dataset();
  const auto queries = make_cosine_queries(dataset);
  const auto oracle = exact_oracle(dataset, queries, core::Metric::cosine);
  TemporaryDirectory temporary("cosine");

  auto created = Collection::create(
      make_options(temporary.path(), core::Metric::cosine, core::algorithm::laser));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::laser);
  EXPECT_FALSE(sealed.value().flat_fallback) << sealed.value().fallback_reason;

  const auto artifacts = load_laser_artifacts(temporary.path(), sealed.value());
  expect_laser_identity_and_semantics(artifacts, core::Metric::cosine, "l2_normalized");
  expect_collection_descriptor(temporary.path(),
                               artifacts.entry,
                               core::Metric::cosine,
                               core::MetricPreprocessing::l2_normalized);
  expect_target_numeric_result_contract(temporary.path(),
                                        artifacts.entry,
                                        dataset,
                                        queries.data(),
                                        core::Metric::cosine);

  auto response = collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(response.ok()) << response.status().diagnostic();
  ASSERT_FALSE(response.value().ids.empty());
  EXPECT_EQ(response.value().ids.front(), dataset.ids[0]);
  expect_exact_sorted_distances(response.value(), dataset, queries, core::Metric::cosine);
  const auto recall = recall_at_k(response.value(), oracle, dataset);

  // Exercise the native ranking before Collection exact-rerank: normalize the
  // non-unit query explicitly because the raw searcher sits below the adapter.
  std::vector<float> normalized_query(kDim, 0.0F);
  normalized_query[0] = 1.0F;
  disk::LaserSegmentSearcher native_searcher(artifacts.segment_directory);
  disk::DiskSearchOptions native_options;
  native_options.top_k = kTopK;
  native_options.ef = 256;
  const auto native_hits = native_searcher.search(normalized_query.data(), native_options);
  ASSERT_FALSE(native_hits.empty());
  EXPECT_EQ(native_hits.front().label, 0U)
      << "raw-IP high-norm distractor row-1 must not beat normalized row-0";
  expect_native_result_contract(native_searcher,
                                normalized_query.data(),
                                queries.data(),
                                dataset,
                                core::Metric::cosine);

  auto retained = collection->get_by_id(dataset.ids[1], CollectionProjection::vector);
  ASSERT_TRUE(retained.ok()) << retained.status().diagnostic();
  ASSERT_TRUE(retained.value().vector.has_value());
  const auto retained_view = retained.value().vector->view();
  EXPECT_FLOAT_EQ(retained_view.row<float>(0)[0], 800.0F);
  EXPECT_FLOAT_EQ(retained_view.row<float>(0)[1], 600.0F);

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened_result = Collection::open(temporary.path());
  ASSERT_TRUE(reopened_result.ok()) << reopened_result.status().diagnostic();
  auto reopened = std::move(reopened_result).value();
  auto reopened_response = reopened->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(reopened_response.ok()) << reopened_response.status().diagnostic();
  ASSERT_FALSE(reopened_response.value().ids.empty());
  EXPECT_EQ(reopened_response.value().ids.front(), dataset.ids[0]);
  const auto reopened_recall = recall_at_k(reopened_response.value(), oracle, dataset);
  expect_exact_sorted_distances(reopened_response.value(),
                                dataset,
                                queries,
                                core::Metric::cosine);

  std::cout << "measured_collection_laser_cosine_recall_at_10=" << std::fixed
            << std::setprecision(4) << recall << '\n';
  std::cout << "measured_collection_laser_cosine_reopen_recall_at_10=" << reopened_recall << '\n';
  EXPECT_GE(recall, 0.85);
  EXPECT_GE(reopened_recall, 0.85);
  EXPECT_TRUE(reopened->close().ok());
}

}  // namespace
}  // namespace alaya
