// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/disk/segment_manifest.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 16;
constexpr core::RowCount kRows = 256;
constexpr core::RowCount kQueryCount = 32;
constexpr core::RowCount kTopK = 10;
constexpr std::uint32_t kMaxNeighbors = 24;
constexpr std::uint32_t kEfConstruction = 128;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-vamana-" + std::string(name) + "-" +
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

[[nodiscard]] auto make_dataset(core::RowCount rows) -> Dataset {
  Dataset result;
  result.vectors.resize(static_cast<std::size_t>(rows * kDim));
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{0xA11A'2026'0715'0001ULL};
  for (core::RowCount row = 0; row < rows; ++row) {
    auto digits = std::to_string(row);
    result.ids.push_back(
        core::LogicalId::from_utf8("row-" + std::string(8 - digits.size(), '0') + digits));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto sample = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
      result.vectors[static_cast<std::size_t>(row * kDim + column)] =
          static_cast<float>(sample) / static_cast<float>(1U << 23U) - 1.0F;
    }
  }
  return result;
}

[[nodiscard]] auto make_queries(const Dataset &dataset, core::RowCount query_count)
    -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(query_count * kDim));
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 7U + 3U) % dataset.ids.size();
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 5U) - 2) * 0.0025F;
      queries[static_cast<std::size_t>(query * kDim + column)] =
          dataset.vectors[source * kDim + column] + perturbation;
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
  options.target_algorithm = core::algorithm::vamana;
  options.build_threads = 1;
  options.max_neighbors = kMaxNeighbors;
  options.ef_construction = kEfConstruction;
  return options;
}

void insert_dataset(Collection &collection, const Dataset &dataset) {
  for (core::RowCount row = 0; row < dataset.ids.size(); ++row) {
    CollectionItem item;
    item.logical_id = dataset.ids[static_cast<std::size_t>(row)];
    item.vector = core::TypedTensorView::contiguous(dataset.vectors.data() +
                                                        static_cast<std::ptrdiff_t>(row * kDim),
                                                    1,
                                                    kDim);
    auto added = collection.add(std::move(item));
    ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  }
}

[[nodiscard]] auto exact_score(const float *query, const float *candidate, core::Metric metric)
    -> float {
  double l2{};
  double dot{};
  for (std::uint32_t column = 0; column < kDim; ++column) {
    const auto lhs = static_cast<double>(query[column]);
    const auto rhs = static_cast<double>(candidate[column]);
    const auto difference = lhs - rhs;
    l2 += difference * difference;
    dot += lhs * rhs;
  }
  return metric == core::Metric::inner_product ? static_cast<float>(-dot) : static_cast<float>(l2);
}

[[nodiscard]] auto exact_oracle(const Dataset &dataset,
                                const std::vector<float> &queries,
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

TEST(CollectionVamanaSeal, PublishesSearchesAndReopensWithRecallCanary) {
  TemporaryDirectory temporary("sealed");
  const auto dataset = make_dataset(kRows);
  const auto queries = make_queries(dataset, kQueryCount);
  const auto oracle = exact_oracle(dataset, queries, kQueryCount, kTopK, core::Metric::l2);

  auto created = Collection::create(make_options(temporary.path(), core::Metric::l2));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(collection->options().max_neighbors, kMaxNeighbors);
  EXPECT_EQ(collection->options().ef_construction, kEfConstruction);
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::vamana);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);

  const auto manifest = internal::collection::ArtifactManifestV2::load(
      temporary.path() / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::vamana);
  EXPECT_EQ(target->factory_key, "vamana");
  EXPECT_EQ(target->lifecycle, internal::collection::SegmentLifecycleV2::sealed);
  std::set<std::string> artifact_names;
  for (const auto &artifact : target->artifacts) {
    if (artifact.logical_name != "artifact_manifest_v2") {
      artifact_names.insert(artifact.logical_name);
    }
  }
  ASSERT_EQ(artifact_names.size(), 4U);
  EXPECT_EQ(artifact_names, (std::set<std::string>{"graph", "ids", "manifest", "vectors"}));
  const auto native_manifest_artifact =
      std::ranges::find_if(target->artifacts, [](const auto &artifact) {
        return artifact.logical_name == "manifest";
      });
  ASSERT_NE(native_manifest_artifact, target->artifacts.end());
  const auto native_manifest =
      disk::SegmentManifest::load(temporary.path() / native_manifest_artifact->relative_path);
  EXPECT_EQ(native_manifest.x_extras.at("x_R"), std::to_string(kMaxNeighbors));
  EXPECT_EQ(native_manifest.x_extras.at("x_L"), std::to_string(kEfConstruction));

  auto before =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim),
                               kTopK);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  ASSERT_EQ(before.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  const auto recall = recall_at_k(before.value(), oracle);
  std::cout << "measured_vamana_recall_at_10=" << std::fixed << std::setprecision(4) << recall
            << '\n';
  EXPECT_GE(recall, 0.90);

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->options().max_neighbors, kMaxNeighbors);
  EXPECT_EQ(reopened.value()->options().ef_construction, kEfConstruction);
  auto after = reopened.value()->batch_search(core::TypedTensorView::contiguous(queries.data(),
                                                                                kQueryCount,
                                                                                kDim),
                                              kTopK);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().offsets, before.value().offsets);
  EXPECT_EQ(after.value().valid_counts, before.value().valid_counts);
  EXPECT_EQ(after.value().ids, before.value().ids);
  EXPECT_EQ(after.value().distances, before.value().distances);
  EXPECT_GE(recall_at_k(after.value(), oracle), 0.90);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionVamanaSeal, UnsupportedMetricFallsBackHonestlyToExactFlat) {
  TemporaryDirectory temporary("fallback");
  constexpr core::RowCount kFallbackRows = 32;
  constexpr core::RowCount kFallbackQueries = 4;
  constexpr core::RowCount kFallbackTopK = 5;
  const auto dataset = make_dataset(kFallbackRows);
  const auto queries = make_queries(dataset, kFallbackQueries);
  const auto oracle =
      exact_oracle(dataset, queries, kFallbackQueries, kFallbackTopK, core::Metric::inner_product);

  auto created = Collection::create(make_options(temporary.path(), core::Metric::inner_product));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_FALSE(sealed.value().fallback_reason.empty());

  const auto manifest = internal::collection::ArtifactManifestV2::load(
      temporary.path() / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::flat);
  EXPECT_EQ(target->factory_key, "flat");

  auto searched = collection->batch_search(core::TypedTensorView::contiguous(queries.data(),
                                                                             kFallbackQueries,
                                                                             kDim),
                                           kFallbackTopK);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  expect_exact_ids(searched.value(), oracle);
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionVamanaOptions, RejectsInvalidGraphBuildBounds) {
  TemporaryDirectory zero_neighbors("zero-neighbors");
  auto invalid_neighbors = make_options(zero_neighbors.path(), core::Metric::l2);
  invalid_neighbors.max_neighbors = 0;
  auto rejected_neighbors = Collection::create(invalid_neighbors);
  ASSERT_FALSE(rejected_neighbors.ok());
  EXPECT_EQ(rejected_neighbors.status().code(), core::StatusCode::invalid_argument);

  TemporaryDirectory low_effort("low-effort");
  auto invalid_effort = make_options(low_effort.path(), core::Metric::l2);
  invalid_effort.ef_construction = invalid_effort.max_neighbors - 1;
  auto rejected_effort = Collection::create(invalid_effort);
  ASSERT_FALSE(rejected_effort.ok());
  EXPECT_EQ(rejected_effort.status().code(), core::StatusCode::invalid_argument);
  EXPECT_NE(rejected_effort.status().diagnostic().find("ef_construction >= max_neighbors"),
            std::string::npos);
}

}  // namespace
}  // namespace alaya
