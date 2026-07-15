// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
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
constexpr core::RowCount kRows = 256;
constexpr core::RowCount kQueryCount = 16;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-nsg-fusion-" + std::string(name) + "-" +
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

[[nodiscard]] auto make_float_dataset(core::RowCount rows, bool scaled = false) -> Dataset<float> {
  Dataset<float> result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{0xA11A'2026'0715'3A01ULL};
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("float-" + std::to_string(row)));
    double norm{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto sample = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
      const auto value = static_cast<float>(sample) / static_cast<float>(1U << 23U) - 1.0F;
      result.vectors[static_cast<std::size_t>(row * kDim + column)] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    const auto row_scale = scaled ? 0.25F + static_cast<float>((row * 17U) % 29U) : 1.0F;
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row * kDim + column)] *= scale * row_scale;
    }
  }
  return result;
}

template <typename T>
[[nodiscard]] auto make_byte_dataset(core::RowCount rows) -> Dataset<T> {
  static_assert(std::is_same_v<T, std::int8_t> || std::is_same_v<T, std::uint8_t>);
  Dataset<T> result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state =
      std::is_same_v<T, std::int8_t> ? 0x1A8A'2026'0715'3B01ULL : 0x8A8A'2026'0715'3B02ULL;
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
      const auto other = static_cast<std::uint32_t>(splitmix64(state) % (column + 1U));
      std::swap(values[column], values[other]);
    }
    std::copy(values.begin(),
              values.end(),
              result.vectors.begin() + static_cast<std::ptrdiff_t>(row * kDim));
  }
  return result;
}

[[nodiscard]] auto make_float_queries(const Dataset<float> &dataset,
                                      core::RowCount query_count,
                                      bool scaled = false) -> std::vector<float> {
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
    if (!scaled) {
      const auto inverse_norm = static_cast<float>(1.0 / std::sqrt(norm));
      for (std::uint32_t column = 0; column < kDim; ++column) {
        queries[static_cast<std::size_t>(query * kDim + column)] *= inverse_norm;
      }
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

template <typename T>
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
  return query_norm == 0.0 || candidate_norm == 0.0
             ? 0.0F
             : static_cast<float>(-dot / std::sqrt(query_norm * candidate_norm));
}

template <typename T>
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
      matches += static_cast<std::uint64_t>(
          std::find(response.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                    response.ids.begin() + static_cast<std::ptrdiff_t>(end),
                    expected) != response.ids.begin() + static_cast<std::ptrdiff_t>(end));
    }
  }
  return static_cast<double>(matches) / static_cast<double>(oracle.size() * oracle.front().size());
}

enum class Variant : std::uint8_t {
  raw_float32_l2,
  raw_float32_inner_product,
  sq8,
  sq4,
  int8,
  uint8,
  cosine_float32,
};

struct VariantConfig {
  core::Metric metric{core::Metric::l2};
  core::ScalarType scalar_type{core::ScalarType::float32};
  CollectionQuantization quantization{CollectionQuantization::none};
  std::string_view quantization_name{"none"};
  std::string_view preprocessing{"none"};
  double recall_floor{0.90};
};

[[nodiscard]] auto variant_config(Variant variant) -> VariantConfig {
  switch (variant) {
    case Variant::raw_float32_l2:
      return {};
    case Variant::raw_float32_inner_product:
      return {.metric = core::Metric::inner_product};
    case Variant::sq8:
      return {.quantization = CollectionQuantization::sq8,
              .quantization_name = "sq8",
              .preprocessing = "engine_quantized",
              .recall_floor = 0.85};
    case Variant::sq4:
      return {.quantization = CollectionQuantization::sq4,
              .quantization_name = "sq4",
              .preprocessing = "engine_quantized",
              .recall_floor = 0.75};
    case Variant::int8:
      return {.scalar_type = core::ScalarType::int8};
    case Variant::uint8:
      return {.scalar_type = core::ScalarType::uint8};
    case Variant::cosine_float32:
      return {.metric = core::Metric::cosine, .preprocessing = "l2_normalized"};
  }
  return {};
}

[[nodiscard]] auto variant_name(Variant variant) -> std::string_view {
  switch (variant) {
    case Variant::raw_float32_l2:
      return "raw_f32_l2";
    case Variant::raw_float32_inner_product:
      return "raw_f32_inner_product";
    case Variant::sq8:
      return "sq8";
    case Variant::sq4:
      return "sq4";
    case Variant::int8:
      return "int8";
    case Variant::uint8:
      return "uint8";
    case Variant::cosine_float32:
      return "cosine_f32";
  }
  return "unknown";
}

[[nodiscard]] auto engine_name(core::AlgorithmId algorithm) -> std::string_view {
  return algorithm == core::algorithm::nsg ? "nsg" : "fusion";
}

[[nodiscard]] auto scalar_name(core::ScalarType scalar_type) -> std::string_view {
  if (scalar_type == core::ScalarType::int8) {
    return "int8";
  }
  if (scalar_type == core::ScalarType::uint8) {
    return "uint8";
  }
  return "float32";
}

[[nodiscard]] auto make_options(const std::filesystem::path &root,
                                core::AlgorithmId algorithm,
                                const VariantConfig &config) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = config.metric;
  options.scalar_type = config.scalar_type;
  options.target_algorithm = algorithm;
  options.quantization = config.quantization;
  options.build_threads = 1;
  options.ef_construction = 400;
  return options;
}

void expect_target_manifest(const std::filesystem::path &root,
                            core::AlgorithmId algorithm,
                            const VariantConfig &config) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, algorithm);
  EXPECT_EQ(target->factory_key, engine_name(algorithm));
  EXPECT_EQ(target->format_version, 1U);
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{std::string(engine_name(algorithm)) + "_segment"}));
  EXPECT_EQ(target->extensions.at("stored_scalar_type"), scalar_name(config.scalar_type));
  EXPECT_EQ(target->extensions.at("quantization"), config.quantization_name);
  EXPECT_EQ(target->extensions.at("preprocessing"), config.preprocessing);
  EXPECT_EQ(target->extensions.at("ef_construction_requested"), "400");
  EXPECT_EQ(target->extensions.at("ef_construction_effective"), std::to_string(kRows));
  EXPECT_EQ(target->extensions.at("max_neighbors"), "32");

  std::set<std::string> artifact_names;
  for (const auto &artifact : target->artifacts) {
    ASSERT_TRUE(std::filesystem::is_regular_file(root / artifact.relative_path));
    if (artifact.logical_name != "artifact_manifest_v2") {
      artifact_names.insert(artifact.logical_name);
    }
  }
  const auto expected = config.quantization == CollectionQuantization::none
                            ? std::set<std::string>{"data", "graph"}
                            : std::set<std::string>{"data", "graph", "quant"};
  EXPECT_EQ(artifact_names, expected);
}

template <typename T>
void run_sealed_case(core::AlgorithmId algorithm,
                     Variant variant,
                     const Dataset<T> &dataset,
                     const std::vector<T> &queries) {
  const auto config = variant_config(variant);
  TemporaryDirectory temporary(std::string(engine_name(algorithm)) + "-" +
                               std::string(variant_name(variant)));
  const auto oracle =
      exact_oracle(dataset, std::span<const T>(queries), kQueryCount, kTopK, config.metric);
  auto created = Collection::create(make_options(temporary.path(), algorithm, config));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, algorithm);
  EXPECT_EQ(sealed.value().effective_ef_construction, kRows);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_target_manifest(temporary.path(), algorithm, config);

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_" << engine_name(algorithm) << '_' << variant_name(variant)
            << "_recall_at_10=" << std::fixed << std::setprecision(4) << recall << '\n';
  EXPECT_GE(recall, config.recall_floor);

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
  EXPECT_GE(recall_at_k(after.value(), oracle), config.recall_floor);

  core::AlgorithmSearchExtension foreign_extension;
  foreign_extension.algorithm_id =
      algorithm == core::algorithm::nsg ? core::algorithm::fusion : core::algorithm::nsg;
  core::SearchOptions large_options(128);
  large_options.extensions = std::span<const core::AlgorithmSearchExtension>(&foreign_extension, 1);
  core::SearchContext large_context;
  auto large = reopened->search(core::TypedTensorView::contiguous(queries.data(), 1, kDim),
                                large_options,
                                large_context);
  ASSERT_TRUE(large.ok()) << large.status().diagnostic();
  EXPECT_EQ(large.value().valid_counts, (std::vector<core::RowCount>{128}));

  std::vector<T> active_vector(queries.begin(), queries.begin() + kDim);
  CollectionItem active;
  active.logical_id = core::LogicalId::from_utf8("active-best");
  active.vector = core::TypedTensorView::contiguous(active_vector.data(), 1, kDim);
  auto added = reopened->add(std::move(active));
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  auto mixed =
      reopened->search(core::TypedTensorView::contiguous(active_vector.data(), 1, kDim), kTopK);
  ASSERT_TRUE(mixed.ok()) << mixed.status().diagnostic();
  ASSERT_EQ(mixed.value().valid_counts, (std::vector<core::RowCount>{kTopK}));
  ASSERT_EQ(mixed.value().ids.front(), core::LogicalId::from_utf8("active-best"));
  EXPECT_NEAR(mixed.value().distances.front(),
              exact_score(active_vector.data(), active_vector.data(), config.metric),
              1.0e-4F);
  for (std::size_t lhs = 0; lhs < mixed.value().ids.size(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < mixed.value().ids.size(); ++rhs) {
      EXPECT_NE(mixed.value().ids[lhs], mixed.value().ids[rhs]);
    }
  }
  for (std::size_t index = 1; index < mixed.value().ids.size(); ++index) {
    EXPECT_LE(mixed.value().distances[index - 1], mixed.value().distances[index]);
    EXPECT_NE(std::ranges::find(dataset.ids, mixed.value().ids[index]), dataset.ids.end());
  }
  ASSERT_TRUE(reopened->close().ok());
}

using EngineVariant = std::tuple<core::AlgorithmId, Variant>;
class CollectionNsgFusionSealTest : public ::testing::TestWithParam<EngineVariant> {};

TEST_P(CollectionNsgFusionSealTest, PublishesRecallsReopensScalesEffortAndMergesActiveFlat) {
  const auto [algorithm, variant] = GetParam();
  if (variant == Variant::int8) {
    const auto dataset = make_byte_dataset<std::int8_t>(kRows);
    run_sealed_case(algorithm, variant, dataset, make_byte_queries(dataset, kQueryCount));
  } else if (variant == Variant::uint8) {
    const auto dataset = make_byte_dataset<std::uint8_t>(kRows);
    run_sealed_case(algorithm, variant, dataset, make_byte_queries(dataset, kQueryCount));
  } else {
    const auto cosine = variant == Variant::cosine_float32;
    const auto dataset = make_float_dataset(kRows, cosine);
    run_sealed_case(algorithm, variant, dataset, make_float_queries(dataset, kQueryCount, cosine));
  }
}

INSTANTIATE_TEST_SUITE_P(Phase3Matrix,
                         CollectionNsgFusionSealTest,
                         ::testing::Combine(::testing::Values(core::algorithm::nsg,
                                                              core::algorithm::fusion),
                                            ::testing::Values(Variant::raw_float32_l2,
                                                              Variant::raw_float32_inner_product,
                                                              Variant::sq8,
                                                              Variant::sq4,
                                                              Variant::int8,
                                                              Variant::uint8,
                                                              Variant::cosine_float32)),
                         [](const ::testing::TestParamInfo<EngineVariant> &info) {
                           return std::string(engine_name(std::get<0>(info.param))) + "_" +
                                  std::string(variant_name(std::get<1>(info.param)));
                         });

class CollectionNsgFusionFallbackTest : public ::testing::TestWithParam<core::AlgorithmId> {};

void expect_flat_manifest(const std::filesystem::path &root) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::flat);
  EXPECT_EQ(target->factory_key, "flat");
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{"disk_flat_segment"}));
}

TEST_P(CollectionNsgFusionFallbackTest, FewerThan65RowsFallsBackToHonestFlat) {
  const auto algorithm = GetParam();
  constexpr core::RowCount rows = 64;
  const auto dataset = make_float_dataset(rows);
  TemporaryDirectory temporary(std::string(engine_name(algorithm)) + "-small");
  auto created = Collection::create(make_options(temporary.path(), algorithm, VariantConfig{}));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_NE(sealed.value().fallback_reason.find(">=65 live rows"), std::string::npos);
  expect_flat_manifest(temporary.path());

  const std::span<const float> query(dataset.vectors.data() + 7 * kDim, kDim);
  const auto oracle = exact_oracle(dataset, query, 1, kTopK, core::Metric::l2);
  auto searched =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, kDim), kTopK);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  EXPECT_EQ(searched.value().ids, oracle.front());
  ASSERT_TRUE(collection->close().ok());
}

TEST_P(CollectionNsgFusionFallbackTest, Int8CosineFallsBackToHonestFlat) {
  const auto algorithm = GetParam();
  constexpr core::RowCount rows = 200;
  const auto dataset = make_byte_dataset<std::int8_t>(rows);
  auto config = VariantConfig{};
  config.metric = core::Metric::cosine;
  config.scalar_type = core::ScalarType::int8;
  TemporaryDirectory temporary(std::string(engine_name(algorithm)) + "-int8-cosine");
  auto created = Collection::create(make_options(temporary.path(), algorithm, config));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_NE(sealed.value().fallback_reason.find("raw float32"), std::string::npos);
  expect_flat_manifest(temporary.path());

  const std::span<const std::int8_t> query(dataset.vectors.data() + 9 * kDim, kDim);
  const auto oracle = exact_oracle(dataset, query, 1, kTopK, core::Metric::cosine);
  auto searched =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, kDim), kTopK);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  EXPECT_EQ(searched.value().ids, oracle.front());
  ASSERT_TRUE(collection->close().ok());
}

INSTANTIATE_TEST_SUITE_P(Phase3Fallback,
                         CollectionNsgFusionFallbackTest,
                         ::testing::Values(core::algorithm::nsg, core::algorithm::fusion),
                         [](const ::testing::TestParamInfo<core::AlgorithmId> &info) {
                           return std::string(engine_name(info.param));
                         });

}  // namespace
}  // namespace alaya
