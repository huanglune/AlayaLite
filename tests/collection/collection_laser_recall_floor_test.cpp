// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// LASER recall floor lock (D-lite manifest W2). Mirrors
// collection_qg_recall_floor_test.cpp's methodology (build a real Collection,
// seal it, measure recall@10 against an exact oracle, lock a floor at
// measured-minus-margin) for the LASER on-disk QG target, which previously
// had no recall regression guard: collection_laser_target_test.cpp's
// lifecycle test only asserts a loose self-recall sanity floor (>=75% exact
// self-match on a tiny random dataset), not a locked recall@10 number.
//
// Differences from the qg fixture this mirrors:
//  - this remains the established L2 regression lane (unit/nonunit plus
//    residency parity). IP same-data public-qg parity and cosine normalization
//    semantics live in collection_laser_metric_wiring_test.cpp, so this file
//    keeps its historical two-tier dataset and thresholds unchanged.
//  - the baseline tiers retain dim=128 so their historical floor stays
//    comparable. A dedicated dim=768 tier below proves the non-power-of-two
//    Collection path (build -> importer -> open -> search) and checks recall
//    against the same exact-oracle methodology.
//  - rows = 400, clear of both LASER's own floor
//    (> RaBitQSpace<>::kDegreeBound == 32) and qg's fixture's 300.
//  - every tier additionally asserts engine identity against the actually
//    persisted sealed-segment manifest on disk (see expect_laser_manifest()
//    below). This is the first recall-floor test in this repo to guard
//    against a silent flat fallback masquerading as a passing recall
//    number, and the reasoning is subtle enough to spell out for whoever
//    copies this pattern next:
//
//      Collection::target_implementation_key() (collection.hpp) is NOT
//      sufficient, despite the name being the obvious grep hit. It resolves
//      options_.target_algorithm -- what the Collection is *configured* to
//      try to build -- through a static registration table and returns that
//      registration's implementation_key. It never looks at what actually
//      got built. A Collection configured for laser that silently fell back
//      to flat (wrong dim, wrong metric, too few rows) still reports
//      target_implementation_key() == "disk_laser_segment", because that's
//      a property of the *request*, not the *result*. The only signal that
//      can't be fooled this way is reading the on-disk ArtifactManifestV2
//      (collection_manifest.txt) back after seal() and checking the
//      *sealed segment entry's own* factory_key/
//      reader_compatibility.required_features -- which is what
//      collection_laser_target_test.cpp's expect_laser_manifest() does, and
//      what this file's copy of it does too. sealed.value().
//      built_algorithm/flat_fallback (also asserted below, in
//      measure_laser_recall()) is an equally trustworthy, cheaper first
//      check that needs no disk read -- this file asserts both.
//  - one extra tier: resident_arena residency (ALAYA_LASER_RESIDENCY=
//    resident_arena env override; precedent: collection_laser_target_test.
//    cpp's ResidentArenaResidencyViaEnvOverrideThroughCollection) reuses the
//    l2_unit case's exact dataset/queries/oracle (same seed -- "same data"
//    is literal here, not just same-recipe), builds a fresh paged-pool
//    (default residency) reference and a fresh resident-arena instance side
//    by side, and checks both clear the l2_unit floor and agree with each
//    other within a tolerance wide enough to absorb the paged-pool kernel's
//    own run-to-run nondeterminism (see the precedent test's comment: paged
//    search interleaves computation with async page-read completions and
//    can explore the graph in a different, individually valid order between
//    runs; resident-arena has no async I/O and is deterministic given the
//    same graph state and query).
//
// Floors are this fixture's measured recall@10 minus a margin generous
// enough to absorb ordinary build/platform noise (same methodology as
// collection_qg_recall_floor_test.cpp). Raw measured values, 7 back-to-back
// runs (see REPORT-allocator-merge.md W2 for the raw log):
//   l2_unit        measured=1.0000 every run                    floor=0.85
//   l2_nonunit     measured=0.9700-0.9800 (paged-pool run noise) floor=0.85
//   resident_arena measured=1.0000 every run, paged_reference=1.0000 every
//                  run (same floor as l2_unit; kConsistencyTolerance=0.10)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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
#include "index/collection/detail/collection_flat_target.hpp"
#include "index/disk/segment_manifest.hpp"
#include "platform/detect.hpp"
#include "utils/evaluate.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kBaselineDim = 128;
constexpr std::uint32_t kNonPowerOfTwoDim = 768;
constexpr core::RowCount kRows = 400;  // > RaBitQSpace<>::kDegreeBound (32).
constexpr core::RowCount kQueryCount = 20;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-laser-recall-floor-" + std::string(name) + "-" +
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
  std::uint32_t dim{};
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

// Unit-norm dataset -- same recipe as collection_qg_recall_floor_test.cpp's
// make_unit_dataset (itself from collection_qg_seal_test.cpp), parameterized
// so the same recipe and exact oracle exercise both the historical 128d floor
// fixture and the new 768d non-power-of-two path.
[[nodiscard]] auto make_unit_dataset(core::RowCount rows,
                                     std::uint64_t seed,
                                     std::uint32_t dimension = kBaselineDim) -> Dataset {
  Dataset result;
  result.dim = dimension;
  result.vectors.resize(static_cast<std::size_t>(rows) * dimension);
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::uint64_t state{seed};
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("row-" + std::to_string(row)));
    double norm{};
    for (std::uint32_t column = 0; column < dimension; ++column) {
      const auto sample = static_cast<std::uint32_t>(splitmix64(state) >> 40U);
      const auto value = static_cast<float>(sample) / static_cast<float>(1U << 23U) - 1.0F;
      result.vectors[static_cast<std::size_t>(row) * dimension + column] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = static_cast<float>(1.0 / std::sqrt(norm));
    for (std::uint32_t column = 0; column < dimension; ++column) {
      result.vectors[static_cast<std::size_t>(row) * dimension + column] *= scale;
    }
  }
  return result;
}

// hnsw_seal's (now retired) make_cosine_dataset style: unit vectors with a
// deliberately-varying per-row magnitude, fed under l2 (which does not
// normalize the magnitude away) so the variation actually reaches the
// index. Same recipe as collection_qg_recall_floor_test.cpp's
// make_nonunit_dataset.
[[nodiscard]] auto make_nonunit_dataset(core::RowCount rows,
                                        std::uint64_t seed,
                                        std::uint32_t dimension = kBaselineDim) -> Dataset {
  auto result = make_unit_dataset(rows, seed, dimension);
  for (core::RowCount row = 0; row < rows; ++row) {
    const auto scale = 0.25F + static_cast<float>((row * 17U) % 29U);
    for (std::uint32_t column = 0; column < dimension; ++column) {
      result.vectors[static_cast<std::size_t>(row) * dimension + column] *= scale;
    }
  }
  return result;
}

[[nodiscard]] auto make_queries(const Dataset &dataset, core::RowCount query_count, bool normalize)
    -> std::vector<float> {
  std::vector<float> queries(static_cast<std::size_t>(query_count) * dataset.dim);
  for (core::RowCount query = 0; query < query_count; ++query) {
    const auto source = (query * 23U + 7U) % dataset.ids.size();
    const auto query_scale = normalize ? 1.0F : (0.5F + static_cast<float>((query * 11U) % 13U));
    double norm{};
    for (std::uint32_t column = 0; column < dataset.dim; ++column) {
      const auto perturbation =
          static_cast<float>(static_cast<int>((query + column) % 7U) - 3) * 0.0005F;
      const auto value =
          dataset.vectors[static_cast<std::size_t>(source) * dataset.dim + column] + perturbation;
      queries[static_cast<std::size_t>(query) * dataset.dim + column] = value;
      norm += static_cast<double>(value) * value;
    }
    const auto scale = normalize ? static_cast<float>(1.0 / std::sqrt(norm)) : query_scale;
    for (std::uint32_t column = 0; column < dataset.dim; ++column) {
      queries[static_cast<std::size_t>(query) * dataset.dim + column] *= scale;
    }
  }
  return queries;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root,
                                std::uint32_t dimension) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = dimension;
  options.metric = core::Metric::l2;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::laser;
  options.quantization = CollectionQuantization::rabitq;
  options.build_threads = 4;
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
                                                        static_cast<std::ptrdiff_t>(row * dataset.dim),
                                                    1,
                                                    dataset.dim);
    items.push_back(std::move(item));
  }
  auto added = collection.add_batch(items, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  ASSERT_EQ(added.value().rows.size(), dataset.ids.size());
}

// Exact l2 ground truth via tests/include/utils/evaluate.hpp's find_exact_gt,
// mapped from row-index space to Collection LogicalIds.
[[nodiscard]] auto exact_oracle_ids(const Dataset &dataset, const std::vector<float> &queries)
    -> std::vector<uint32_t> {
  return find_exact_gt<float, float, uint32_t>(
      queries, dataset.vectors, dataset.dim, static_cast<uint32_t>(kTopK), core::Metric::l2);
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

// Engine-identity guard against the *persisted* sealed-segment manifest --
// see the file header comment for why Collection::target_implementation_key()
// is not a substitute (it reflects configured intent, not built reality).
void expect_laser_manifest(const std::filesystem::path &root) {
  const auto manifest = internal::collection::ArtifactManifestV2::load(
      root / internal::collection::kCollectionManifestFilename);
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::laser);
  EXPECT_EQ(target->factory_key, "laser");
  EXPECT_EQ(target->reader_compatibility.required_features,
            (std::vector<std::string>{"disk_laser_segment"}));
}

// Builds a laser-target Collection (optionally under a residency env
// override), seals it, asserts engine identity against the persisted
// manifest (plus the residency override actually landing in the native
// segment manifest, when one was requested), searches with `queries`, and
// returns recall@kTopK against `oracle_row_ids`.
[[nodiscard]] auto measure_laser_recall(std::string_view name,
                                        const Dataset &dataset,
                                        const std::vector<float> &queries,
                                        const std::vector<uint32_t> &oracle_row_ids,
                                        const char *residency_env_override = nullptr) -> double {
  struct EnvGuard {
    bool active{};
    ~EnvGuard() {
      if (active) {
        ::unsetenv("ALAYA_LASER_RESIDENCY");
      }
    }
  } env_guard;
  if (residency_env_override != nullptr) {
    ::setenv("ALAYA_LASER_RESIDENCY", residency_env_override, 1);
    env_guard.active = true;
  }

  TemporaryDirectory temporary(name);
  auto created = Collection::create(make_options(temporary.path(), dataset.dim));
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
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::laser)
      << name << ": expected a real laser segment, not a flat fallback ("
      << sealed.value().fallback_reason << ")";
  EXPECT_FALSE(sealed.value().flat_fallback) << name << ": " << sealed.value().fallback_reason;
  expect_laser_manifest(temporary.path());

  if (residency_env_override != nullptr) {
    // Confirm the native manifest actually recorded the override -- proof
    // this is a residency variant, not "it happens to pass the same way
    // regardless of residency" (collection_laser_target_test.cpp precedent).
    const auto seg_dir =
        temporary.path() / "segments" /
        internal::collection::detail::collection_segment_name(sealed.value().sealed_segment_id);
    const auto native_manifest = ::alaya::disk::SegmentManifest::load(seg_dir / "manifest.txt");
    const auto residency_extra = native_manifest.x_extras.find("x_laser_residency");
    // ASSERT_* expands to `return;` on failure, which does not typecheck in
    // this double-returning function -- EXPECT_NE plus a manual guard is the
    // non-void-function equivalent (same shape as the created.ok()/sealed.ok()
    // checks above).
    if (residency_extra == native_manifest.x_extras.end()) {
      ADD_FAILURE() << name << ": native segment manifest missing x_laser_residency extra";
    } else {
      EXPECT_EQ(residency_extra->second, residency_env_override);
    }
  }

  auto response =
      collection->batch_search(
          core::TypedTensorView::contiguous(queries.data(), kQueryCount, dataset.dim), kTopK);
  if (!response.ok()) {
    ADD_FAILURE() << name << ": batch_search failed: " << response.status().diagnostic();
    return 0.0;
  }

  const auto recall = recall_at_k(response.value(), oracle_row_ids, dataset);
  EXPECT_TRUE(collection->close().ok());
  return recall;
}

struct RecallFloorCase {
  bool unit_norm;
  // Measured laser recall@10 minus a margin generous enough to absorb
  // ordinary build/platform noise (see file header comment for the raw
  // numbers this wave measured them from).
  double floor;
};

[[nodiscard]] auto case_name(const RecallFloorCase &c) -> std::string {
  // Underscore-separated: this doubles as the GTest parameterized test-case
  // name, which rejects dashes.
  return std::string("l2_") + (c.unit_norm ? "unit" : "nonunit");
}

class CollectionLaserRecallFloorTest : public ::testing::TestWithParam<RecallFloorCase> {};

TEST_P(CollectionLaserRecallFloorTest, LaserRecallStaysAboveFloor) {
  const auto param = GetParam();
  const auto name = case_name(param);
  constexpr std::uint64_t kSeed = 0xA11A2026'0717'7A02ULL;

  const auto dataset =
      param.unit_norm ? make_unit_dataset(kRows, kSeed) : make_nonunit_dataset(kRows, kSeed);
  const auto queries = make_queries(dataset, kQueryCount, /*normalize=*/param.unit_norm);
  const auto oracle_row_ids = exact_oracle_ids(dataset, queries);

  const auto laser_recall = measure_laser_recall(name, dataset, queries, oracle_row_ids);

  std::cout << "measured_laser_recall_floor_" << name << "_recall_at_10=" << std::fixed
            << std::setprecision(4) << laser_recall << std::endl;

  EXPECT_GE(laser_recall, param.floor) << name;
}

INSTANTIATE_TEST_SUITE_P(L2UnitAndNonUnit,
                         CollectionLaserRecallFloorTest,
                         ::testing::Values(RecallFloorCase{true, 0.85},
                                           RecallFloorCase{false, 0.85}),
                         [](const ::testing::TestParamInfo<RecallFloorCase> &info) {
                           return case_name(info.param);
                         });

TEST(CollectionLaserDimensionGate, NonPowerOfTwo768BuildSearchRecall) {
  constexpr std::uint64_t kSeed = 0xA11A2026'0718'0768ULL;
  // The bare LASER gte768 full-main-dimension probe measured recall@10=0.98637
  // at ef=200 (docs/research/LASER_UPDATE_EXPLORATION.md section 21.2). This tiny
  // synthetic Collection fixture is not a dataset-equivalent benchmark, so it
  // uses the existing Collection LASER floor (0.85) as a noise-tolerant
  // regression threshold while printing the measured value for the report.
  constexpr double kRecallFloor = 0.85;

  const auto dataset = make_unit_dataset(kRows, kSeed, kNonPowerOfTwoDim);
  const auto queries = make_queries(dataset, kQueryCount, /*normalize=*/true);
  const auto oracle_row_ids = exact_oracle_ids(dataset, queries);

  const auto recall =
      measure_laser_recall("non-power-of-two-768d", dataset, queries, oracle_row_ids);

  std::cout << "measured_laser_collection_dim768_recall_at_10=" << std::fixed
            << std::setprecision(4) << recall << std::endl;
  EXPECT_GE(recall, kRecallFloor);
}

TEST(CollectionLaserDimensionGate, DimensionsOutsideFhtRangeFallBackToFlat) {
  constexpr core::RowCount kFallbackRows = 40;  // Clear the independent >32-row gate.
  for (const std::uint32_t dimension : {32U, 2049U}) {
    TemporaryDirectory temporary("out-of-range-" + std::to_string(dimension));
    const auto dataset = make_unit_dataset(kFallbackRows, 0xD1A6A7EULL + dimension, dimension);

    auto created = Collection::create(make_options(temporary.path(), dimension));
    ASSERT_TRUE(created.ok()) << "dim=" << dimension << ": "
                              << created.status().diagnostic();
    auto collection = std::move(created).value();
    insert_dataset(*collection, dataset);

    auto sealed = collection->seal();
    ASSERT_TRUE(sealed.ok()) << "dim=" << dimension << ": " << sealed.status().diagnostic();
    EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat) << "dim=" << dimension;
    EXPECT_TRUE(sealed.value().flat_fallback) << "dim=" << dimension;
    EXPECT_NE(sealed.value().fallback_reason.find("[33, 2048]"), std::string::npos)
        << "dim=" << dimension << ": " << sealed.value().fallback_reason;

    const auto manifest = internal::collection::ArtifactManifestV2::load(
        temporary.path() / internal::collection::kCollectionManifestFilename);
    const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
      return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
    });
    ASSERT_NE(target, manifest.segments.end());
    EXPECT_EQ(target->algorithm_id, core::algorithm::flat) << "dim=" << dimension;
    EXPECT_EQ(target->factory_key, "flat") << "dim=" << dimension;
    EXPECT_TRUE(collection->close().ok());
  }
}

// Resident-arena tier: reuses the l2_unit case's exact dataset/queries/oracle
// (same seed as the TEST_P case above -- literal same data, not just
// same-recipe). See the file header comment for the paged-vs-resident-arena
// nondeterminism tolerance rationale.
TEST(CollectionLaserRecallFloorResidentArena, ResidentArenaRecallMatchesPaged) {
  constexpr std::uint64_t kSeed = 0xA11A2026'0717'7A02ULL;  // == l2_unit's seed: same data.
  constexpr double kFloor = 0.85;                           // == l2_unit's floor.
  constexpr double kConsistencyTolerance = 0.10;

  const auto dataset = make_unit_dataset(kRows, kSeed);
  const auto queries = make_queries(dataset, kQueryCount, /*normalize=*/true);
  const auto oracle_row_ids = exact_oracle_ids(dataset, queries);

  const auto paged_recall =
      measure_laser_recall("resident-arena-paged-reference", dataset, queries, oracle_row_ids);
  const auto resident_recall = measure_laser_recall(
      "resident-arena-override", dataset, queries, oracle_row_ids, "resident_arena");

  std::cout << "measured_laser_recall_floor_resident_arena_recall_at_10=" << std::fixed
            << std::setprecision(4) << resident_recall << " (paged_reference=" << paged_recall
            << ")" << std::endl;

  EXPECT_GE(paged_recall, kFloor) << "paged reference";
  EXPECT_GE(resident_recall, kFloor) << "resident_arena";
  EXPECT_NEAR(resident_recall, paged_recall, kConsistencyTolerance)
      << "resident_arena recall should track the paged-pool default on identical data";
}

}  // namespace
}  // namespace alaya
