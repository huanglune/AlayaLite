// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 64;
constexpr core::RowCount kRows = 384;
constexpr core::RowCount kQueryCount = 20;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-qg-" + std::string(name) + "-" +
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

template <typename T>
struct Dataset {
  std::vector<T> vectors{};
  std::vector<core::LogicalId> ids{};
};

[[nodiscard]] auto splitmix64(std::uint64_t &state) -> std::uint64_t {
  state += 0x9E3779B97F4A7C15ULL;
  auto value = state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

[[nodiscard]] auto make_float_dataset(core::RowCount rows) -> Dataset<float> {
  Dataset<float> result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{0xA11A'2026'0715'4A01ULL};
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

[[nodiscard]] auto make_cosine_dataset(core::RowCount rows) -> Dataset<float> {
  auto result = make_float_dataset(rows);
  for (core::RowCount row = 0; row < rows; ++row) {
    const auto scale = 0.25F + static_cast<float>((row * 17U) % 29U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale;
    }
  }
  std::fill_n(result.vectors.begin(), kDim, 0.0F);
  return result;
}

[[nodiscard]] auto make_int8_dataset(core::RowCount rows) -> Dataset<std::int8_t> {
  Dataset<std::int8_t> result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("int8-" + std::to_string(row)));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] =
          static_cast<std::int8_t>((row * 17U + column * 11U) % 127U);
    }
  }
  return result;
}

[[nodiscard]] auto make_queries(const Dataset<float> &dataset) -> std::vector<float> {
  std::vector<float> result(static_cast<std::size_t>(kQueryCount * kDim));
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto source = (query * 29U + 5U) % dataset.ids.size();
    double norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query * 3U + column) % 9U) - 4) * 0.0007F;
      const auto value =
          dataset.vectors[static_cast<std::size_t>(source * kDim + column)] + perturbation;
      result[static_cast<std::size_t>(query * kDim + column)] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result[static_cast<std::size_t>(query * kDim + column)] *= scale;
    }
  }
  return result;
}

[[nodiscard]] auto make_cosine_queries(const Dataset<float> &dataset, core::RowCount query_count)
    -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(query_count * kDim));
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 23U + 7U) % dataset.ids.size();
    const auto query_scale = 0.5F + static_cast<float>((query * 11U) % 13U);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 7U) - 3) * 0.002F;
      queries[static_cast<std::size_t>(query * kDim + column)] =
          (dataset.vectors[static_cast<std::size_t>(source * kDim + column)] + perturbation) *
          query_scale;
    }
  }
  return queries;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root,
                                core::Metric metric,
                                core::ScalarType scalar_type = core::ScalarType::float32)
    -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = metric;
  options.scalar_type = scalar_type;
  options.target_algorithm = core::algorithm::qg;
  options.quantization = CollectionQuantization::rabitq;
  options.build_threads = 1;
  options.ef_construction = 400;
  return options;
}

template <typename T>
void insert_dataset(Collection &collection, const Dataset<T> &dataset) {
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

[[nodiscard]] auto exact_score(const float *query, const float *candidate, core::Metric metric)
    -> float {
  double l2{};
  double dot{};
  double query_norm{};
  double candidate_norm{};
  for (std::uint32_t column = 0; column < kDim; ++column) {
    const auto lhs = static_cast<double>(query[column]);
    const auto rhs = static_cast<double>(candidate[column]);
    const auto difference = lhs - rhs;
    l2 += difference * difference;
    dot += lhs * rhs;
    query_norm += lhs * lhs;
    candidate_norm += rhs * rhs;
  }
  if (metric == core::Metric::l2) {
    return static_cast<float>(l2);
  }
  if (metric == core::Metric::inner_product) {
    return static_cast<float>(-dot);
  }
  if (query_norm == 0 || candidate_norm == 0) {
    return 0.0F;
  }
  return static_cast<float>(-dot / std::sqrt(query_norm * candidate_norm));
}

[[nodiscard]] auto exact_oracle(const Dataset<float> &dataset,
                                std::span<const float> queries,
                                core::RowCount query_count,
                                core::RowCount top_k,
                                core::Metric metric) -> std::vector<std::vector<core::LogicalId>> {
  std::vector<std::vector<core::LogicalId>> result(static_cast<std::size_t>(query_count));
  for (core::RowCount query = 0; query < query_count; ++query) {
    std::vector<std::pair<float, std::size_t>> ranked;
    ranked.reserve(dataset.ids.size());
    for (std::size_t row = 0; row < dataset.ids.size(); ++row) {
      ranked.emplace_back(exact_score(queries.data() + query * kDim,
                                      dataset.vectors.data() + row * kDim,
                                      metric),
                          row);
    }
    std::sort(ranked.begin(), ranked.end(), [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first) {
        return lhs.first < rhs.first;
      }
      return dataset.ids[lhs.second].compare(dataset.ids[rhs.second]) < 0;
    });
    auto &ids = result[static_cast<std::size_t>(query)];
    const auto count = std::min<core::RowCount>(top_k, ranked.size());
    for (core::RowCount rank = 0; rank < count; ++rank) {
      ids.push_back(dataset.ids[ranked[static_cast<std::size_t>(rank)].second]);
    }
  }
  return result;
}

[[nodiscard]] auto recall_at_k(const CollectionSearchResponse &response,
                               const std::vector<std::vector<core::LogicalId>> &oracle) -> double {
  std::uint64_t matches{};
  for (std::size_t query = 0; query < oracle.size(); ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    for (const auto &expected : oracle[query]) {
      if (std::find(response.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                    response.ids.begin() + static_cast<std::ptrdiff_t>(end),
                    expected) != response.ids.begin() + static_cast<std::ptrdiff_t>(end)) {
        ++matches;
      }
    }
  }
  return static_cast<double>(matches) / static_cast<double>(oracle.size() * oracle.front().size());
}

[[nodiscard]] auto find_row(const Dataset<float> &dataset, const core::LogicalId &logical_id)
    -> std::size_t {
  const auto found = std::ranges::find(dataset.ids, logical_id);
  if (found == dataset.ids.end()) {
    throw std::runtime_error("QG test response contains an unknown logical ID");
  }
  return static_cast<std::size_t>(std::distance(dataset.ids.begin(), found));
}

void expect_contract_a_scores(const CollectionSearchResponse &response,
                              const Dataset<float> &dataset,
                              std::span<const float> queries,
                              core::Metric metric) {
  ASSERT_EQ(response.offsets.size(), response.valid_counts.size() + 1);
  for (std::size_t query = 0; query < response.valid_counts.size(); ++query) {
    auto previous = -std::numeric_limits<float>::infinity();
    for (auto index = response.offsets[query]; index < response.offsets[query + 1]; ++index) {
      const auto row = find_row(dataset, response.ids[static_cast<std::size_t>(index)]);
      const auto exact =
          exact_score(queries.data() + query * kDim, dataset.vectors.data() + row * kDim, metric);
      EXPECT_NEAR(response.distances[static_cast<std::size_t>(index)], exact, 2.0e-5F);
      EXPECT_LE(previous, exact + 2.0e-5F);
      previous = exact;
    }
  }
}

void expect_response_ids(const CollectionSearchResponse &response,
                         const std::vector<std::vector<core::LogicalId>> &expected) {
  ASSERT_EQ(response.offsets.size(), expected.size() + 1);
  for (std::size_t query = 0; query < expected.size(); ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    ASSERT_EQ(end - begin, expected[query].size());
    for (std::size_t rank = 0; rank < expected[query].size(); ++rank) {
      EXPECT_EQ(response.ids[static_cast<std::size_t>(begin + rank)], expected[query][rank]);
    }
  }
}

void expect_qg_manifest(const std::filesystem::path &root,
                        std::string_view preprocessing = "engine_quantized") {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::qg);
  EXPECT_EQ(target->factory_key, "qg");
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{"qg_segment"}));
  EXPECT_EQ(target->extensions.at("stored_scalar_type"), "float32");
  EXPECT_EQ(target->extensions.at("preprocessing"), preprocessing);
  EXPECT_EQ(target->extensions.at("quantization"), "rabitq");

  std::set<std::string> artifact_names;
  for (const auto &artifact : target->artifacts) {
    ASSERT_TRUE(std::filesystem::is_regular_file(root / artifact.relative_path));
    if (artifact.logical_name != "artifact_manifest_v2") {
      artifact_names.insert(artifact.logical_name);
    }
  }
  EXPECT_EQ(artifact_names, (std::set<std::string>{"qg"}));
}

void expect_flat_manifest(const std::filesystem::path &root) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::flat);
  EXPECT_EQ(target->factory_key, "flat");
}

[[nodiscard]] auto make_active_vector(std::span<const float> query,
                                      float target_score,
                                      core::Metric metric) -> std::vector<float> {
  std::vector<float> result(kDim);
  double query_norm{};
  for (const auto value : query) {
    query_norm += static_cast<double>(value) * value;
  }
  if (metric == core::Metric::inner_product) {
    const auto scale = static_cast<float>(-static_cast<double>(target_score) / query_norm);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result[column] = scale * query[column];
    }
    return result;
  }

  std::vector<float> direction(kDim, 0.0F);
  direction[0] = 1.0F;
  const auto projection = static_cast<double>(query[0]) / query_norm;
  double direction_norm{};
  for (std::uint32_t column = 0; column < kDim; ++column) {
    direction[column] -= static_cast<float>(projection * query[column]);
    direction_norm += static_cast<double>(direction[column]) * direction[column];
  }
  const auto direction_scale = static_cast<float>(1.0 / std::sqrt(direction_norm));
  const auto displacement = std::sqrt(std::max(0.0F, target_score));
  for (std::uint32_t column = 0; column < kDim; ++column) {
    result[column] = query[column] + displacement * direction[column] * direction_scale;
  }
  return result;
}

class CollectionQgSealTest : public ::testing::TestWithParam<core::Metric> {};

TEST_P(CollectionQgSealTest, PublishesRecallsReopensAndExactlyOrdersMixedCandidates) {
  const auto metric = GetParam();
  const auto metric_name = metric == core::Metric::l2 ? "l2" : "inner_product";
  TemporaryDirectory temporary(metric_name);
  const auto dataset = make_float_dataset(kRows);
  const auto queries = make_queries(dataset);
  const auto oracle =
      exact_oracle(dataset, std::span<const float>(queries), kQueryCount, kTopK, metric);

  auto created = Collection::create(make_options(temporary.path(), metric));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::qg);
  EXPECT_EQ(sealed.value().effective_ef_construction, 400U);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_qg_manifest(temporary.path());

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_qg_" << metric_name << "_recall_at_10=" << std::fixed
            << std::setprecision(4) << recall << '\n';
  EXPECT_GE(recall, 0.80);
  expect_contract_a_scores(before.value(), dataset, queries, metric);

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto opened = Collection::open(temporary.path());
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto reopened = std::move(opened).value();
  auto after =
      reopened->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                             kTopK);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().offsets, before.value().offsets);
  EXPECT_EQ(after.value().valid_counts, before.value().valid_counts);
  EXPECT_EQ(after.value().ids, before.value().ids);
  EXPECT_EQ(after.value().distances, before.value().distances);

  core::SearchOptions large_options(128);
  core::SearchContext large_context;
  auto large = reopened->search(core::TypedTensorView::contiguous(queries.data(), 1, kDim),
                                large_options,
                                large_context);
  ASSERT_TRUE(large.ok()) << large.status().diagnostic();
  EXPECT_EQ(large.value().valid_counts, (std::vector<core::RowCount>{128}));

  const std::span<const float> mixed_query(queries.data(), kDim);
  auto sealed_candidates =
      reopened->search(core::TypedTensorView::contiguous(mixed_query.data(), 1, kDim), kTopK);
  ASSERT_TRUE(sealed_candidates.ok()) << sealed_candidates.status().diagnostic();
  ASSERT_EQ(sealed_candidates.value().valid_counts, (std::vector<core::RowCount>{kTopK}));

  Dataset<float> active;
  active.ids.reserve(3);
  active.vectors.reserve(3 * kDim);
  constexpr std::array<std::pair<std::size_t, std::size_t>, 3>
      kBrackets{std::pair<std::size_t, std::size_t>{0, 1}, {3, 4}, {6, 7}};
  for (std::size_t index = 0; index < kBrackets.size(); ++index) {
    const auto [lower, upper] = kBrackets[index];
    const auto target = std::midpoint(sealed_candidates.value().distances[lower],
                                      sealed_candidates.value().distances[upper]);
    auto vector = make_active_vector(mixed_query, target, metric);
    active.ids.push_back(core::LogicalId::from_utf8("active-" + std::to_string(index)));
    active.vectors.insert(active.vectors.end(), vector.begin(), vector.end());
  }
  insert_dataset(*reopened, active);

  auto mixed =
      reopened->search(core::TypedTensorView::contiguous(mixed_query.data(), 1, kDim), kTopK);
  ASSERT_TRUE(mixed.ok()) << mixed.status().diagnostic();
  ASSERT_EQ(mixed.value().valid_counts, (std::vector<core::RowCount>{kTopK}));

  Dataset<float> combined = dataset;
  combined.ids.insert(combined.ids.end(), active.ids.begin(), active.ids.end());
  combined.vectors.insert(combined.vectors.end(), active.vectors.begin(), active.vectors.end());
  expect_contract_a_scores(mixed.value(), combined, mixed_query, metric);

  Dataset<float> candidate_pool;
  for (const auto &logical_id : sealed_candidates.value().ids) {
    const auto row = find_row(dataset, logical_id);
    candidate_pool.ids.push_back(logical_id);
    candidate_pool.vectors.insert(candidate_pool.vectors.end(),
                                  dataset.vectors.begin() + static_cast<std::ptrdiff_t>(row * kDim),
                                  dataset.vectors.begin() +
                                      static_cast<std::ptrdiff_t>((row + 1) * kDim));
  }
  candidate_pool.ids.insert(candidate_pool.ids.end(), active.ids.begin(), active.ids.end());
  candidate_pool.vectors.insert(candidate_pool.vectors.end(),
                                active.vectors.begin(),
                                active.vectors.end());
  const auto candidate_oracle = exact_oracle(candidate_pool, mixed_query, 1, kTopK, metric);
  expect_response_ids(mixed.value(), candidate_oracle);

  const auto mixed_oracle = exact_oracle(combined, mixed_query, 1, kTopK, metric);
  const auto mixed_recall = recall_at_k(mixed.value(), mixed_oracle);
  std::cout << "measured_qg_" << metric_name << "_mixed_recall_at_10=" << std::fixed
            << std::setprecision(4) << mixed_recall << '\n';
  std::cout << "measured_qg_" << metric_name << "_mixed_candidate_order_exact="
            << (mixed.value().ids == candidate_oracle.front() ? "true" : "false") << '\n';
  EXPECT_GE(mixed_recall, 0.80);

  std::vector<core::LogicalId> expected_recalled_order;
  std::vector<core::LogicalId> actual_recalled_order;
  for (const auto &logical_id : mixed_oracle.front()) {
    if (std::ranges::find(mixed.value().ids, logical_id) != mixed.value().ids.end()) {
      expected_recalled_order.push_back(logical_id);
    }
  }
  for (const auto &logical_id : mixed.value().ids) {
    if (std::ranges::find(mixed_oracle.front(), logical_id) != mixed_oracle.front().end()) {
      actual_recalled_order.push_back(logical_id);
    }
  }
  EXPECT_EQ(actual_recalled_order, expected_recalled_order);
  ASSERT_TRUE(reopened->close().ok());
}

INSTANTIATE_TEST_SUITE_P(QgFloat32,
                         CollectionQgSealTest,
                         ::testing::Values(core::Metric::l2, core::Metric::inner_product),
                         [](const ::testing::TestParamInfo<core::Metric> &info) {
                           return info.param == core::Metric::l2 ? "L2" : "InnerProduct";
                         });

TEST(CollectionQgCosineSeal, NormalizesRecallsReopensHandlesZeroAndMergesWithActiveFlat) {
  TemporaryDirectory temporary("cosine");
  const auto dataset = make_cosine_dataset(kRows);
  const auto queries = make_cosine_queries(dataset, kQueryCount);
  const auto oracle = exact_oracle(dataset,
                                   std::span<const float>(queries),
                                   kQueryCount,
                                   kTopK,
                                   core::Metric::cosine);

  auto created = Collection::create(make_options(temporary.path(), core::Metric::cosine));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::qg);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_qg_manifest(temporary.path(), "l2_normalized");

  const auto manifest = internal::collection::ArtifactManifestV2::load(
      temporary.path() / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  core::OpenContext open_context;
  const internal::collection::CollectionSchema schema{kDim,
                                                      core::Metric::cosine,
                                                      core::ScalarType::float32};
  auto opened = internal::collection::detail::open_qg_collection_target(temporary.path(),
                                                                        *target,
                                                                        schema,
                                                                        open_context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto descriptor = opened.value().descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::qg);
  EXPECT_EQ(descriptor.metric, core::Metric::cosine);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(descriptor.preprocessing, core::MetricPreprocessing::l2_normalized);

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_qg_cosine_recall_at_10=" << std::fixed << std::setprecision(4) << recall
            << '\n';
  EXPECT_GE(recall, 0.95);

  std::vector<float> zero_query(kDim, 0.0F);
  const auto zero_oracle =
      exact_oracle(dataset, std::span<const float>(zero_query), 1, kRows, core::Metric::cosine);
  auto zero_before =
      collection->search(core::TypedTensorView::contiguous(zero_query.data(), 1, kDim), kRows);
  ASSERT_TRUE(zero_before.ok()) << zero_before.status().diagnostic();
  ASSERT_EQ(zero_before.value().valid_counts, (std::vector<core::RowCount>{kRows}));
  expect_response_ids(zero_before.value(), zero_oracle);
  for (const auto score : zero_before.value().distances) {
    EXPECT_FLOAT_EQ(score, 0.0F);
  }

  auto retained = collection->get_by_id(dataset.ids[1], CollectionProjection::vector);
  ASSERT_TRUE(retained.ok()) << retained.status().diagnostic();
  ASSERT_TRUE(retained.value().vector.has_value());
  const auto retained_view = retained.value().vector->view();
  for (std::uint32_t column = 0; column < kDim; ++column) {
    EXPECT_FLOAT_EQ(retained_view.row<float>(0)[column], dataset.vectors[kDim + column]);
  }

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened_result = Collection::open(temporary.path());
  ASSERT_TRUE(reopened_result.ok()) << reopened_result.status().diagnostic();
  auto reopened = std::move(reopened_result).value();
  auto after =
      reopened->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                             kTopK);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().offsets, before.value().offsets);
  EXPECT_EQ(after.value().valid_counts, before.value().valid_counts);
  EXPECT_EQ(after.value().ids, before.value().ids);
  EXPECT_EQ(after.value().distances, before.value().distances);
  EXPECT_GE(recall_at_k(after.value(), oracle), 0.95);

  auto zero_after =
      reopened->search(core::TypedTensorView::contiguous(zero_query.data(), 1, kDim), kRows);
  ASSERT_TRUE(zero_after.ok()) << zero_after.status().diagnostic();
  EXPECT_EQ(zero_after.value().ids, zero_before.value().ids);
  EXPECT_EQ(zero_after.value().distances, zero_before.value().distances);

  auto retained_after = reopened->get_by_id(dataset.ids[1], CollectionProjection::vector);
  ASSERT_TRUE(retained_after.ok()) << retained_after.status().diagnostic();
  ASSERT_TRUE(retained_after.value().vector.has_value());
  const auto retained_after_view = retained_after.value().vector->view();
  for (std::uint32_t column = 0; column < kDim; ++column) {
    EXPECT_FLOAT_EQ(retained_after_view.row<float>(0)[column], dataset.vectors[kDim + column]);
  }

  std::vector<float> active_vector(queries.begin(), queries.begin() + kDim);
  CollectionItem active;
  active.logical_id = core::LogicalId::from_utf8("active-best");
  active.vector = core::TypedTensorView::contiguous(active_vector.data(), 1, kDim);
  auto added = reopened->add(std::move(active));
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  auto mixed =
      reopened->search(core::TypedTensorView::contiguous(active_vector.data(), 1, kDim), 5);
  ASSERT_TRUE(mixed.ok()) << mixed.status().diagnostic();
  ASSERT_EQ(mixed.value().valid_counts, (std::vector<core::RowCount>{5}));
  ASSERT_EQ(mixed.value().ids.front(), core::LogicalId::from_utf8("active-best"));
  EXPECT_NEAR(mixed.value().distances.front(), -1.0F, 1.0e-5F);
  const auto sealed_oracle =
      exact_oracle(dataset, std::span<const float>(active_vector), 1, 4, core::Metric::cosine);
  for (std::size_t rank = 0; rank < sealed_oracle.front().size(); ++rank) {
    EXPECT_EQ(mixed.value().ids[rank + 1], sealed_oracle.front()[rank]);
  }
  for (std::size_t index = 1; index < mixed.value().distances.size(); ++index) {
    EXPECT_LE(mixed.value().distances[index - 1], mixed.value().distances[index]);
    const auto found = std::ranges::find(dataset.ids, mixed.value().ids[index]);
    ASSERT_NE(found, dataset.ids.end());
    const auto row = static_cast<std::size_t>(std::distance(dataset.ids.begin(), found));
    EXPECT_NEAR(mixed.value().distances[index],
                exact_score(active_vector.data(),
                            dataset.vectors.data() + row * kDim,
                            core::Metric::cosine),
                1.0e-4F);
  }
  ASSERT_TRUE(reopened->close().ok());
}

TEST(CollectionQgFallback, RejectsForeignRabitqAndHonestlyFallsBackForUnsupportedQgSchemas) {
  // HNSW, NSG, and Fusion are all retired: their algorithm ids stay reserved
  // (never reused) but are no longer accepted at all, so the capability gate
  // rejects them outright instead of reaching the rabitq/qg cross-check.
  // (HNSW+rabitq used to be the invalid_argument "explicit index_type=qg"
  // cross-check case; now it -- like nsg/fusion -- never reaches that check.)
  for (const auto algorithm :
       {core::algorithm::hnsw, core::algorithm::nsg, core::algorithm::fusion}) {
    TemporaryDirectory temporary("retired-algorithm-" + std::to_string(algorithm));
    auto options = make_options(temporary.path(), core::Metric::l2);
    options.target_algorithm = algorithm;
    auto rejected = Collection::create(options);
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.status().code(), core::StatusCode::not_supported);
    EXPECT_NE(rejected.status().diagnostic().find("target algorithm is unsupported"),
              std::string::npos);
  }

  {
    TemporaryDirectory temporary("int8-fallback");
    const auto dataset = make_int8_dataset(kRows);
    auto created = Collection::create(
        make_options(temporary.path(), core::Metric::l2, core::ScalarType::int8));
    ASSERT_TRUE(created.ok()) << created.status().diagnostic();
    auto collection = std::move(created).value();
    insert_dataset(*collection, dataset);
    auto sealed = collection->seal();
    ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
    EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
    EXPECT_TRUE(sealed.value().flat_fallback);
    EXPECT_NE(sealed.value().fallback_reason.find("float32"), std::string::npos);
    expect_flat_manifest(temporary.path());
    ASSERT_TRUE(collection->close().ok());
  }

  // Cosine used to be a qg fallback-to-flat trigger; it is now a supported
  // qg target (see CollectionQgCosineSeal), so that scenario moved out of
  // this "honest fallback" test and is no longer exercised here.

  {
    TemporaryDirectory temporary("small-fallback");
    const auto dataset = make_float_dataset(32);
    auto created = Collection::create(make_options(temporary.path(), core::Metric::l2));
    ASSERT_TRUE(created.ok()) << created.status().diagnostic();
    auto collection = std::move(created).value();
    insert_dataset(*collection, dataset);
    auto sealed = collection->seal();
    ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
    EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
    EXPECT_TRUE(sealed.value().flat_fallback);
    EXPECT_NE(sealed.value().fallback_reason.find(">32"), std::string::npos);
    expect_flat_manifest(temporary.path());
    ASSERT_TRUE(collection->close().ok());
  }
}

}  // namespace
}  // namespace alaya
