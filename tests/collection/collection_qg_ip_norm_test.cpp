// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// U4-preflight IP audit (amendment v2): characterization + regression lock for
// QG (memory RaBitQ) recall on non-unit-norm inner_product data.
//
// Every pre-existing rabitq/QG test only ever fed unit vectors
// (collection_qg_seal_test.cpp's make_float_dataset:85-88 always
// L2-normalizes), so the "does inner_product recall hold up for non-unit-norm
// data" question had never actually been exercised. The original hypothesis
// (RaBitQCore::memory_factors' inner-product branch has a hidden "||o||=1"
// assumption via a literal `1` in its base term) was falsified by a math
// review: that `1` is a candidate-independent constant, and the resulting
// estimator 1-<q,o> is strictly order-preserving for inner_product ranking at
// any norm (see the derivation comment on RaBitQCore::memory_factors, and
// tests/space/rabitq_space_test.cpp's RaBitQCoreTest.
// InnerProductBranchLocksToOneMinusDot + RaBitQSpaceIpNormTest.
// NonUnitNormRecallDoesNotCollapse for the space-level version of this test).
//
// This is the Collection-level (full QG seal + search pipeline) counterpart:
// it does not try to "catch a bug" any more -- it locks in that recall does
// not collapse for non-unit-norm inner_product data, as a regression guard.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 64;
constexpr core::RowCount kRows = 256;
constexpr core::RowCount kQueryCount = 20;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-qg-ip-norm-" + std::string(name) + "-" +
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

// Same generation as collection_qg_seal_test.cpp's make_float_dataset: every
// row is L2-normalized to a unit vector.
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

// Start from unit vectors, then deliberately vary magnitude per row (0.25x
// to 28.25x). This is fed to QG under inner_product, so the varying norm is
// NOT normalized away anywhere in the pipeline (Collection only
// L2-normalizes for cosine, not inner_product).
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

[[nodiscard]] auto make_options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = core::Metric::inner_product;
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

// Exact, un-normalized -dot oracle: matches the ip_sqr convention (smaller is
// more similar) that RaBitQCore::memory_factors' inner-product branch targets
// -- see the derivation comment on that function for why this is the correct
// oracle regardless of ||o||.
[[nodiscard]] auto exact_oracle(const Dataset &dataset,
                                std::span<const float> queries,
                                core::RowCount query_count,
                                core::RowCount top_k) -> std::vector<std::vector<core::LogicalId>> {
  std::vector<std::vector<core::LogicalId>> result(static_cast<std::size_t>(query_count));
  for (core::RowCount query = 0; query < query_count; ++query) {
    std::vector<std::pair<float, std::size_t>> ranked;
    ranked.reserve(dataset.ids.size());
    for (std::size_t row = 0; row < dataset.ids.size(); ++row) {
      double dot{};
      for (std::uint32_t column = 0; column < kDim; ++column) {
        dot += static_cast<double>(queries[query * kDim + column]) *
               static_cast<double>(dataset.vectors[row * kDim + column]);
      }
      ranked.emplace_back(static_cast<float>(-dot), row);
    }
    std::sort(ranked.begin(), ranked.end(), [&](const auto &lhs, const auto &rhs) {
      if (lhs.first != rhs.first) {
        return lhs.first < rhs.first;
      }
      return dataset.ids[lhs.second].compare(dataset.ids[rhs.second]) < 0;
    });
    auto &ids = result[static_cast<std::size_t>(query)];
    const auto count = std::min<core::RowCount>(top_k, static_cast<core::RowCount>(ranked.size()));
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

// Builds a QG(inner_product) Collection over `dataset`, seals it, searches
// with `queries`, and returns recall@kTopK against the exact -dot oracle.
[[nodiscard]] auto measure_qg_ip_recall(std::string_view name,
                                        const Dataset &dataset,
                                        const std::vector<float> &queries) -> double {
  TemporaryDirectory temporary(name);
  const auto oracle = exact_oracle(dataset, std::span<const float>(queries), kQueryCount, kTopK);

  auto created = Collection::create(make_options(temporary.path()));
  if (!created.ok()) {
    ADD_FAILURE() << "Collection::create failed: " << created.status().diagnostic();
    return 0.0;
  }
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  if (!sealed.ok()) {
    ADD_FAILURE() << "seal() failed: " << sealed.status().diagnostic();
    return 0.0;
  }
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::qg)
      << "expected a real QG segment (not a flat fallback) for " << name;
  EXPECT_FALSE(sealed.value().flat_fallback) << sealed.value().fallback_reason;

  auto response =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  if (!response.ok()) {
    ADD_FAILURE() << "batch_search failed: " << response.status().diagnostic();
    return 0.0;
  }

  const auto recall = recall_at_k(response.value(), oracle);
  EXPECT_TRUE(collection->close().ok());
  return recall;
}

TEST(CollectionQgIpNormTest, NonUnitNormRecallDoesNotCollapse) {
  constexpr std::uint64_t kSeed = 0xA11A2026'07165A01ULL;

  const auto unit_dataset = make_unit_dataset(kRows, kSeed);
  const auto unit_queries = make_queries(unit_dataset, kQueryCount, /*normalize=*/true);
  const auto unit_recall = measure_qg_ip_recall("unit", unit_dataset, unit_queries);

  const auto nonunit_dataset = make_nonunit_dataset(kRows, kSeed);
  const auto nonunit_queries = make_queries(nonunit_dataset, kQueryCount, /*normalize=*/false);
  const auto nonunit_recall = measure_qg_ip_recall("nonunit", nonunit_dataset, nonunit_queries);

  std::cout << "measured_collection_qg_ip_unit_recall_at_10=" << std::fixed << std::setprecision(4)
            << unit_recall << std::endl;
  std::cout << "measured_collection_qg_ip_nonunit_recall_at_10=" << std::fixed
            << std::setprecision(4) << nonunit_recall << std::endl;

  // Sanity: matches collection_qg_seal_test.cpp's own inner_product threshold
  // for the same (unit-norm) data shape.
  EXPECT_GE(unit_recall, 0.80);

  // Characterization + regression lock (U4-preflight IP audit, amendment v2):
  // recall on non-unit-norm inner_product data must not collapse relative to
  // the unit-norm baseline measured immediately above with the identical
  // pipeline. If this ever fires, it means a *different*, unknown bug --
  // amendment v2's math review already proved the "literal 1" formula in
  // RaBitQCore::memory_factors is order-preserving for any norm. Stop and
  // report; do not "fix" it by touching memory_factors.
  EXPECT_GE(nonunit_recall, unit_recall - 0.25);
  EXPECT_GE(nonunit_recall, 0.40);
}

}  // namespace
}  // namespace alaya
