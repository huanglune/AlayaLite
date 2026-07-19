// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// QG recall floor lock (HNSW retirement wave, decision 4). This file used to
// build one HNSW segment Collection and one QG segment Collection side by
// side and assert recall parity between them (see git history for
// collection_hnsw_qg_parity_test.cpp, and REPORT-u4-preflight.md item 3, for
// the original HNSW-vs-QG numbers). HNSW is retired in this same wave, so
// this is now a QG-only absolute-floor lock: each case's floor is derived
// from what this exact fixture measured at HEAD of this wave (l2_unit
// qg=0.9900, l2_nonunit qg=0.8650, ip_unit qg=0.9950, ip_nonunit qg=0.9650)
// minus a margin generous enough to absorb ordinary build/platform noise.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "platform/detect.hpp"
#include "utils/evaluate.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim =
    64;  // QG's FhtKacRotator requires floor_log2(dim) in [6,11], i.e. dim >= 64
constexpr core::RowCount kRows = 300;
constexpr core::RowCount kQueryCount = 20;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-qg-recall-floor-" + std::string(name) + "-" +
             std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

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

// Unit-norm dataset, same recipe as collection_qg_seal_test.cpp's
// make_float_dataset.
[[nodiscard]] auto make_unit_dataset(core::RowCount rows, std::uint64_t seed) -> Dataset {
  Dataset result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{seed};
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("row-" + std::to_string(row)));
    double norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto sample = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
      const auto value = static_cast<float>(sample) / static_cast<float>(1U << 23U) - 1.0F;
      result.vectors[static_cast<std::size_t>(row * kDim + column)] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale;
    }
  }
  return result;
}

// hnsw_seal's (now retired) make_cosine_dataset style: unit vectors with a
// deliberately-varying per-row magnitude (0.25x .. 28.25x), fed under l2 or
// inner_product (neither of which Collection normalizes away, unlike
// cosine) so the norm variation actually reaches the index.
[[nodiscard]] auto make_nonunit_dataset(core::RowCount rows, std::uint64_t seed) -> Dataset {
  auto result = make_unit_dataset(rows, seed);
  for (core::RowCount row = 0; row < rows; ++row) {
    const auto scale = 0.25F + static_cast<float>((row * 17U) % 29U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale;
    }
  }
  return result;
}

[[nodiscard]] auto make_queries(const Dataset &dataset, core::RowCount query_count, bool normalize)
    -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(query_count * kDim));
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 23U + 7U) % dataset.ids.size();
    const auto query_scale = normalize ? 1.0F : (0.5F + static_cast<float>((query * 11U) % 13U));
    double norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 7U) - 3) * 0.0005F;
      const auto value =
          dataset.vectors[static_cast<std::size_t>(source * kDim + column)] + perturbation;
      queries[static_cast<std::size_t>(query * kDim + column)] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = normalize ? static_cast<float>(1.0 / std::sqrt(norm)) : query_scale;
    for (std::uint32_t column = 0; column < kDim; ++column) {
      queries[static_cast<std::size_t>(query * kDim + column)] *= scale;
    }
  }
  return queries;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root, core::Metric metric)
    -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = metric;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::qg;
  options.quantization = CollectionQuantization::rabitq;
  options.build_threads = 1;
  options.ef_construction = 400;
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

// Metric-aware exact ground truth, via tests/include/utils/evaluate.hpp's
// find_exact_gt, mapped from row-index space to Collection LogicalIds.
[[nodiscard]] auto exact_oracle_ids(const Dataset &dataset,
                                    const std::vector<float> &queries,
                                    core::Metric metric) -> std::vector<uint32_t> {
  return find_exact_gt<float, float, uint32_t>(queries,
                                               dataset.vectors,
                                               kDim,
                                               static_cast<uint32_t>(kTopK),
                                               metric);
}

[[nodiscard]] auto recall_at_k(const CollectionSearchResponse &response,
                               const std::vector<uint32_t> &oracle_row_ids,
                               const Dataset &dataset) -> double {
  std::uint64_t matches{};
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    for (core::RowCount rank = 0; rank < kTopK; ++rank) {
      const auto row = oracle_row_ids[static_cast<std::size_t>(query * kTopK + rank)];
      const auto &expected = dataset.ids[row];
      if (std::find(response.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                    response.ids.begin() + static_cast<std::ptrdiff_t>(end),
                    expected) != response.ids.begin() + static_cast<std::ptrdiff_t>(end)) {
        ++matches;
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(kQueryCount * kTopK);
}

// Engine identity must come from the segment that seal() actually published.
// The algorithm id intentionally cannot distinguish this wave's same-id swap,
// and Collection::target_implementation_key() describes configured intent,
// not the implementation that won support resolution.
void expect_qg_laser_engine_identity(const std::filesystem::path &root) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->factory_key, "qg");
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{"qg_laser_segment"}));
  EXPECT_EQ(target->extensions.at("score_kind"), "distance");
  EXPECT_EQ(target->extensions.at("numeric_score_comparable"), "true");
  EXPECT_NE(target->factory_key, "flat");
  EXPECT_EQ(std::ranges::find(target->reader_compatibility.required_features,
                             "disk_flat_segment"),
            target->reader_compatibility.required_features.end());
}

// Builds a QG-target Collection, seals it, searches with `queries`, and
// returns recall@kTopK against `oracle_row_ids`.
[[nodiscard]] auto measure_qg_recall(std::string_view name,
                                     core::Metric metric,
                                     const Dataset &dataset,
                                     const std::vector<float> &queries,
                                     const std::vector<uint32_t> &oracle_row_ids) -> double {
  TemporaryDirectory temporary(name);
  auto created = Collection::create(make_options(temporary.path(), metric));
  if (!created.ok()) {
    ADD_FAILURE() << name << ": Collection::create failed: " << created.status().diagnostic();
    return 0.0;
  }
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  if (!sealed.ok()) {
    ADD_FAILURE() << name << ": seal() failed: " << sealed.status().diagnostic();
    return 0.0;
  }
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::qg)
      << name << ": expected a real qg segment, not a flat fallback ("
      << sealed.value().fallback_reason << ")";
  EXPECT_FALSE(sealed.value().flat_fallback) << name << ": " << sealed.value().fallback_reason;
  expect_qg_laser_engine_identity(temporary.path());

  QgSearchExtension effort;
  effort.effort = 400;
  const auto extension = make_qg_search_extension(effort);
  core::SearchOptions options(kTopK);
  options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
  core::SearchStats runtime_stats;
  core::SearchContext search_context;
  search_context.stats = &runtime_stats;
  auto response = collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
      options,
      search_context);
  if (!response.ok()) {
    ADD_FAILURE() << name << ": batch_search failed: " << response.status().diagnostic();
    return 0.0;
  }
  EXPECT_EQ(response.value().search_stats.rerank_nanoseconds, 0U);
  EXPECT_EQ(runtime_stats.rerank_count, 0U);

  const auto recall = recall_at_k(response.value(), oracle_row_ids, dataset);
  EXPECT_TRUE(collection->close().ok());
  return recall;
}

struct RecallFloorCase {
  core::Metric metric;
  bool unit_norm;
  // Measured qg recall@10 minus a margin generous enough to absorb ordinary
  // build/platform noise (see file header comment for the raw numbers this
  // wave measured them from).
  double floor;
};

[[nodiscard]] auto case_name(const RecallFloorCase &c) -> std::string {
  // Underscore-separated: this doubles as the GTest parameterized test-case
  // name, which rejects dashes.
  return std::string(c.metric == core::Metric::l2 ? "l2" : "ip") + "_" +
         (c.unit_norm ? "unit" : "nonunit");
}

class CollectionQgRecallFloorTest : public ::testing::TestWithParam<RecallFloorCase> {};

TEST_P(CollectionQgRecallFloorTest, QgRecallStaysAboveFloor) {
  const auto param = GetParam();
  const auto name = case_name(param);
  constexpr std::uint64_t kSeed = 0xA11A2026'0716'7A01ULL;

  const auto dataset =
      param.unit_norm ? make_unit_dataset(kRows, kSeed) : make_nonunit_dataset(kRows, kSeed);
  const auto queries = make_queries(dataset, kQueryCount, /*normalize=*/param.unit_norm);
  const auto oracle_row_ids = exact_oracle_ids(dataset, queries, param.metric);

  const auto qg_recall = measure_qg_recall(name, param.metric, dataset, queries, oracle_row_ids);

  std::cout << "measured_qg_recall_floor_" << name << "_recall_at_10=" << std::fixed
            << std::setprecision(4) << qg_recall << std::endl;

  EXPECT_GE(qg_recall, param.floor) << name;
}

INSTANTIATE_TEST_SUITE_P(
    L2AndInnerProductUnitAndNonUnit,
    CollectionQgRecallFloorTest,
    ::testing::Values(RecallFloorCase{core::Metric::l2, true, 0.85},
                      RecallFloorCase{core::Metric::l2, false, 0.75},
                      RecallFloorCase{core::Metric::inner_product, true, 0.85},
                      RecallFloorCase{core::Metric::inner_product, false, 0.80}),
    [](const ::testing::TestParamInfo<RecallFloorCase> &info) {
      return case_name(info.param);
    });

}  // namespace
}  // namespace alaya
