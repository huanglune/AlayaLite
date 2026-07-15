// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 32;
constexpr core::RowCount kRows = 640;
constexpr core::RowCount kQueryCount = 24;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-hnsw-" + std::string(name) + "-" +
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

template <class T>
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
  std::uint64_t state{0xA11A'2026'0715'2A01ULL};
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

template <typename T>
[[nodiscard]] auto make_byte_dataset(core::RowCount rows) -> Dataset<T> {
  static_assert(std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>);
  Dataset<T> result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state =
      std::is_same_v<T, std::int8_t> ? 0x1A8A'2026'0715'2B01ULL : 0x8A8A'2026'0715'2B02ULL;
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8(
        std::string(std::is_same_v<T, std::int8_t> ? "int8-" : "uint8-") + std::to_string(row)));
    std::vector<T> values(kDim);
    for (std::uint32_t column = 0; column < kDim; ++column) {
      if constexpr (std::is_same_v<T, std::int8_t>) {
        const auto magnitude = static_cast<int>(8U + column * 3U);
        values[column] = static_cast<T>((splitmix64(state) & 1U) == 0U ? magnitude : -magnitude);
      } else {
        values[column] = static_cast<T>(16U + column * 6U);
      }
    }
    for (std::uint32_t column = kDim - 1; column > 0; --column) {
      const auto swap_column = static_cast<std::uint32_t>(splitmix64(state) % (column + 1U));
      std::swap(values[column], values[swap_column]);
    }
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] = values[column];
    }
  }
  return result;
}

[[nodiscard]] auto make_queries(const Dataset<float> &dataset, core::RowCount query_count)
    -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(query_count * kDim));
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 23U + 7U) % dataset.ids.size();
    double norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 7U) - 3) * 0.0005F;
      const auto value =
          dataset.vectors[static_cast<std::size_t>(source * kDim + column)] + perturbation;
      queries[static_cast<std::size_t>(query * kDim + column)] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      queries[static_cast<std::size_t>(query * kDim + column)] *= scale;
    }
  }
  return queries;
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

template <typename T>
[[nodiscard]] auto make_byte_queries(const Dataset<T> &dataset, core::RowCount query_count)
    -> std::vector<T> {
  std::vector<T> queries(static_cast<std::size_t>(query_count * kDim));
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 23U + 7U) % dataset.ids.size();
    std::copy_n(dataset.vectors.data() + static_cast<std::ptrdiff_t>(source * kDim),
                kDim,
                queries.data() + static_cast<std::ptrdiff_t>(query * kDim));
    const auto lhs = static_cast<std::uint32_t>((query * 5U + 1U) % kDim);
    const auto rhs = static_cast<std::uint32_t>((query * 11U + 9U) % kDim);
    std::swap(queries[static_cast<std::size_t>(query * kDim + lhs)],
              queries[static_cast<std::size_t>(query * kDim + rhs)]);
  }
  return queries;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root,
                                core::Metric metric,
                                core::ScalarType scalar_type = core::ScalarType::float32,
                                CollectionQuantization quantization = CollectionQuantization::none)
    -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = metric;
  options.scalar_type = scalar_type;
  options.target_algorithm = core::algorithm::hnsw;
  options.quantization = quantization;
  options.build_threads = 1;
  options.ef_construction = 400;
  return options;
}

template <class T>
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
  for (const auto &row : added.value().rows) {
    ASSERT_EQ(row.row_status, CollectionRowMutationStatus::inserted);
  }
}

template <class T>
[[nodiscard]] auto exact_score(const T *query, const T *candidate, core::Metric metric) -> float {
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

template <class T>
[[nodiscard]] auto exact_oracle(const Dataset<T> &dataset,
                                std::span<const T> queries,
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
    for (core::RowCount rank = 0; rank < top_k; ++rank) {
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

void expect_exact_ids(const CollectionSearchResponse &response,
                      const std::vector<std::vector<core::LogicalId>> &oracle) {
  ASSERT_EQ(response.offsets.size(), oracle.size() + 1);
  for (std::size_t query = 0; query < oracle.size(); ++query) {
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    ASSERT_EQ(end - begin, oracle[query].size());
    for (std::size_t rank = 0; rank < oracle[query].size(); ++rank) {
      EXPECT_EQ(response.ids[begin + rank], oracle[query][rank]);
    }
  }
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

[[nodiscard]] auto scalar_type_name(core::ScalarType scalar_type) -> std::string_view {
  switch (scalar_type) {
    case core::ScalarType::float32:
      return "float32";
    case core::ScalarType::int8:
      return "int8";
    case core::ScalarType::uint8:
      return "uint8";
  }
  return "unknown";
}

void expect_hnsw_manifest(const std::filesystem::path &root,
                          core::ScalarType scalar_type,
                          std::string_view quantization,
                          std::string_view preprocessing = {}) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::hnsw);
  EXPECT_EQ(target->factory_key, "hnsw");
  EXPECT_EQ(target->format_version, 1U);
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{"hnsw_segment"}));
  ASSERT_TRUE(target->extensions.contains("stored_scalar_type"));
  ASSERT_TRUE(target->extensions.contains("preprocessing"));
  ASSERT_TRUE(target->extensions.contains("quantization"));
  EXPECT_EQ(target->extensions.at("stored_scalar_type"), scalar_type_name(scalar_type));
  EXPECT_EQ(target->extensions.at("quantization"), quantization);
  const auto expected_preprocessing = preprocessing.empty()
                                          ? (quantization == "none" ? "none" : "engine_quantized")
                                          : preprocessing;
  EXPECT_EQ(target->extensions.at("preprocessing"), expected_preprocessing);

  std::set<std::string> artifact_names;
  for (const auto &artifact : target->artifacts) {
    ASSERT_TRUE(std::filesystem::is_regular_file(root / artifact.relative_path));
    if (artifact.logical_name != "artifact_manifest_v2") {
      artifact_names.insert(artifact.logical_name);
    }
  }
  const auto expected_artifacts = quantization == "none"
                                      ? std::set<std::string>{"data", "graph"}
                                      : std::set<std::string>{"data", "graph", "quant"};
  EXPECT_EQ(artifact_names, expected_artifacts);
}

class CollectionHnswSealTest : public ::testing::TestWithParam<core::Metric> {};

TEST_P(CollectionHnswSealTest, PublishesRecallsReopensAndMergesWithActiveFlat) {
  const auto metric = GetParam();
  TemporaryDirectory temporary(metric == core::Metric::l2 ? "l2" : "ip");
  const auto dataset = make_float_dataset(kRows);
  const auto queries = make_queries(dataset, kQueryCount);
  const auto oracle =
      exact_oracle(dataset, std::span<const float>(queries), kQueryCount, kTopK, metric);

  auto created = Collection::create(make_options(temporary.path(), metric));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::hnsw);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);

  expect_hnsw_manifest(temporary.path(), core::ScalarType::float32, "none");

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_hnsw_" << (metric == core::Metric::l2 ? "l2" : "inner_product")
            << "_recall_at_10=" << std::fixed << std::setprecision(4) << recall << '\n';
  EXPECT_GE(recall, 0.95);

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

  core::AlgorithmSearchExtension foreign_extension;
  foreign_extension.algorithm_id = core::algorithm::nsg;
  core::SearchOptions large_options(128);
  large_options.extensions = std::span<const core::AlgorithmSearchExtension>(&foreign_extension, 1);
  core::SearchContext large_context;
  auto large = reopened->search(core::TypedTensorView::contiguous(queries.data(), 1, kDim),
                                large_options,
                                large_context);
  ASSERT_TRUE(large.ok()) << large.status().diagnostic();
  ASSERT_EQ(large.value().valid_counts, (std::vector<core::RowCount>{128}));

  std::vector<float> active_vector(kDim, 4.0F);
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
  EXPECT_FLOAT_EQ(mixed.value().distances.front(),
                  exact_score(active_vector.data(), active_vector.data(), metric));
  const auto sealed_oracle =
      exact_oracle(dataset, std::span<const float>(active_vector), 1, 4, metric);
  for (std::size_t rank = 0; rank < sealed_oracle.front().size(); ++rank) {
    EXPECT_EQ(mixed.value().ids[rank + 1], sealed_oracle.front()[rank]);
  }
  for (std::size_t index = 1; index < mixed.value().distances.size(); ++index) {
    EXPECT_LE(mixed.value().distances[index - 1], mixed.value().distances[index]);
    const auto found = std::ranges::find(dataset.ids, mixed.value().ids[index]);
    ASSERT_NE(found, dataset.ids.end());
    const auto row = static_cast<std::size_t>(std::distance(dataset.ids.begin(), found));
    EXPECT_NEAR(mixed.value().distances[index],
                exact_score(active_vector.data(), dataset.vectors.data() + row * kDim, metric),
                1.0e-4F);
  }
  ASSERT_TRUE(reopened->close().ok());
}

INSTANTIATE_TEST_SUITE_P(RawFloat32,
                         CollectionHnswSealTest,
                         ::testing::Values(core::Metric::l2, core::Metric::inner_product),
                         [](const ::testing::TestParamInfo<core::Metric> &info) {
                           return info.param == core::Metric::l2 ? "L2" : "InnerProduct";
                         });

TEST(CollectionHnswCosineSeal, NormalizesRecallsReopensHandlesZeroAndMergesWithActiveFlat) {
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
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::hnsw);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_hnsw_manifest(temporary.path(), core::ScalarType::float32, "none", "l2_normalized");

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
  auto opened = internal::collection::detail::open_hnsw_collection_target(temporary.path(),
                                                                          *target,
                                                                          schema,
                                                                          open_context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto descriptor = opened.value().descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::hnsw);
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
  std::cout << "measured_hnsw_cosine_recall_at_10=" << std::fixed << std::setprecision(4) << recall
            << '\n';
  EXPECT_GE(recall, 0.95);

  std::vector<float> zero_query(kDim, 0.0F);
  const auto zero_oracle =
      exact_oracle(dataset, std::span<const float>(zero_query), 1, kRows, core::Metric::cosine);
  auto zero_before =
      collection->search(core::TypedTensorView::contiguous(zero_query.data(), 1, kDim), kRows);
  ASSERT_TRUE(zero_before.ok()) << zero_before.status().diagnostic();
  ASSERT_EQ(zero_before.value().valid_counts, (std::vector<core::RowCount>{kRows}));
  expect_exact_ids(zero_before.value(), zero_oracle);
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

enum class Phase2bVariant : std::uint8_t { sq8, sq4, int8, uint8 };

struct Phase2bParam {
  Phase2bVariant variant{};
  core::Metric metric{core::Metric::l2};
};

[[nodiscard]] auto phase2b_variant_name(Phase2bVariant variant) -> std::string_view {
  switch (variant) {
    case Phase2bVariant::sq8:
      return "sq8";
    case Phase2bVariant::sq4:
      return "sq4";
    case Phase2bVariant::int8:
      return "int8";
    case Phase2bVariant::uint8:
      return "uint8";
  }
  return "unknown";
}

template <typename T>
void run_phase2b_hnsw_case(const Phase2bParam &param,
                           const Dataset<T> &dataset,
                           const std::vector<T> &queries,
                           core::ScalarType scalar_type,
                           CollectionQuantization quantization,
                           double recall_floor) {
  const auto variant_name = phase2b_variant_name(param.variant);
  const auto metric_name = param.metric == core::Metric::l2 ? "l2" : "inner_product";
  TemporaryDirectory temporary(std::string(variant_name) + "-" + metric_name);
  const auto oracle =
      exact_oracle(dataset, std::span<const T>(queries), kQueryCount, kTopK, param.metric);

  auto created =
      Collection::create(make_options(temporary.path(), param.metric, scalar_type, quantization));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::hnsw);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_hnsw_manifest(temporary.path(),
                       scalar_type,
                       quantization == CollectionQuantization::none ? "none" : variant_name);

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_hnsw_" << variant_name << '_' << metric_name
            << "_recall_at_10=" << std::fixed << std::setprecision(4) << recall << '\n';
  EXPECT_GE(recall, recall_floor);

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
  EXPECT_GE(recall_at_k(after.value(), oracle), recall_floor);

  core::AlgorithmSearchExtension foreign_extension;
  foreign_extension.algorithm_id = core::algorithm::nsg;
  core::SearchOptions large_options(128);
  large_options.extensions = std::span<const core::AlgorithmSearchExtension>(&foreign_extension, 1);
  core::SearchContext large_context;
  auto large = reopened->search(core::TypedTensorView::contiguous(queries.data(), 1, kDim),
                                large_options,
                                large_context);
  ASSERT_TRUE(large.ok()) << large.status().diagnostic();
  EXPECT_EQ(large.value().valid_counts, (std::vector<core::RowCount>{128}));
  ASSERT_TRUE(reopened->close().ok());
}

class CollectionHnswPhase2bSealTest : public ::testing::TestWithParam<Phase2bParam> {};

TEST_P(CollectionHnswPhase2bSealTest, PublishesTypedArtifactsRecallsAndReopens) {
  const auto param = GetParam();
  if (param.variant == Phase2bVariant::sq8 || param.variant == Phase2bVariant::sq4) {
    const auto dataset = make_float_dataset(kRows);
    const auto queries = make_queries(dataset, kQueryCount);
    const auto quantization = param.variant == Phase2bVariant::sq8 ? CollectionQuantization::sq8
                                                                   : CollectionQuantization::sq4;
    const auto floor = param.variant == Phase2bVariant::sq8 ? 0.90 : 0.80;
    run_phase2b_hnsw_case(param, dataset, queries, core::ScalarType::float32, quantization, floor);
  } else if (param.variant == Phase2bVariant::int8) {
    const auto dataset = make_byte_dataset<std::int8_t>(kRows);
    const auto queries = make_byte_queries(dataset, kQueryCount);
    run_phase2b_hnsw_case(param,
                          dataset,
                          queries,
                          core::ScalarType::int8,
                          CollectionQuantization::none,
                          0.95);
  } else {
    const auto dataset = make_byte_dataset<std::uint8_t>(kRows);
    const auto queries = make_byte_queries(dataset, kQueryCount);
    run_phase2b_hnsw_case(param,
                          dataset,
                          queries,
                          core::ScalarType::uint8,
                          CollectionQuantization::none,
                          0.95);
  }
}

INSTANTIATE_TEST_SUITE_P(
    QuantizedAndByte,
    CollectionHnswPhase2bSealTest,
    ::testing::Values(Phase2bParam{Phase2bVariant::sq8, core::Metric::l2},
                      Phase2bParam{Phase2bVariant::sq8, core::Metric::inner_product},
                      Phase2bParam{Phase2bVariant::sq4, core::Metric::l2},
                      Phase2bParam{Phase2bVariant::sq4, core::Metric::inner_product},
                      Phase2bParam{Phase2bVariant::int8, core::Metric::l2},
                      Phase2bParam{Phase2bVariant::int8, core::Metric::inner_product},
                      Phase2bParam{Phase2bVariant::uint8, core::Metric::l2},
                      Phase2bParam{Phase2bVariant::uint8, core::Metric::inner_product}),
    [](const ::testing::TestParamInfo<Phase2bParam> &info) {
      std::string name(phase2b_variant_name(info.param.variant));
      name += info.param.metric == core::Metric::l2 ? "_L2" : "_InnerProduct";
      return name;
    });

void run_float_fallback_case(std::string_view name,
                             core::Metric metric,
                             CollectionQuantization quantization) {
  TemporaryDirectory temporary(name);
  constexpr core::RowCount kFallbackRows = 72;
  constexpr core::RowCount kFallbackTopK = 6;
  const auto dataset = make_float_dataset(kFallbackRows);
  const std::span<const float> query(dataset.vectors.data() + 5 * kDim, kDim);
  const auto oracle = exact_oracle(dataset, query, 1, kFallbackTopK, metric);
  auto created = Collection::create(
      make_options(temporary.path(), metric, core::ScalarType::float32, quantization));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_FALSE(sealed.value().fallback_reason.empty());
  expect_flat_manifest(temporary.path());
  auto searched =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, kDim), kFallbackTopK);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  expect_exact_ids(searched.value(), oracle);
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionHnswFallback, CosineSq8IsHonestlyExactFlat) {
  run_float_fallback_case("cosine-sq8", core::Metric::cosine, CollectionQuantization::sq8);
}

TEST(CollectionHnswFallback, Int8CosineIsHonestlyExactFlat) {
  TemporaryDirectory temporary("int8-cosine");
  constexpr core::RowCount kFallbackRows = 72;
  constexpr core::RowCount kFallbackTopK = 6;
  const auto dataset = make_byte_dataset<std::int8_t>(kFallbackRows);
  const std::span<const std::int8_t> query(dataset.vectors.data() + 9 * kDim, kDim);
  const auto oracle = exact_oracle(dataset, query, 1, kFallbackTopK, core::Metric::cosine);
  auto created = Collection::create(
      make_options(temporary.path(), core::Metric::cosine, core::ScalarType::int8));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_FALSE(sealed.value().fallback_reason.empty());
  expect_flat_manifest(temporary.path());
  auto searched =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, kDim), kFallbackTopK);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  expect_exact_ids(searched.value(), oracle);
  ASSERT_TRUE(collection->close().ok());
}

}  // namespace
}  // namespace alaya
