// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// End-to-end Collection-facade coverage for LASER as a Collection target
// algorithm (U2-c manifest, W1): create(target_algorithm=laser) -> upsert
// -> seal -> plain search + filter search -> close -> reopen (through the
// manifest-driven CollectionSegmentFactory::open_entry() registry path,
// i.e. open_laser_collection_target()) -> search still works -> one
// rotate_to_successor -> metric=inner_product falls back to flat (LASER is
// L2-only).
//
// This exercises build_laser_collection_target()/open_laser_collection_target()
// end to end (collection_target_builder.hpp), unlike
// tests/collection/segmented_collection_laser_filter_test.cpp, which builds
// its LASER segment out of band (raw VamanaBuilder/QGBuilder/
// LaserSegmentImporter calls) and hand-registers it into a bare
// SegmentedCollection -- that file is the right place for the lower-level
// per-hit ResultFlag::filtered assertion (see its
// BitmapFilterTraversalExecutesOnLaserSegment test), which this file cannot
// reproduce: the Collection facade's CollectionSearchResponse has no
// per-hit flags field, only aggregate search_stats.filter_execution. This
// file checks search_stats.filter_execution == traversal (proof the
// admission path actually ran, not a prefilter/postfilter fallback) plus
// "every surviving hit satisfies the predicate" as the facade-level analog.
//
// A note on why this file's seal()+rotate() sequence itself doubles as a
// regression check for the QGBuilder out-of-core heap overflow fixed
// earlier in this series (see tests/laser/qg/test_qg_builder_oom_regression.cpp):
// seal() builds one LASER index, and prepare_successor() inside the rotate
// step below builds a second one in the same process -- exactly the
// back-to-back-builds shape that used to crash intermittently.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/detail/collection_flat_target.hpp"
#include "index/disk/segment_manifest.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 128;    // Historical lifecycle fixture dimension.
constexpr core::RowCount kRows = 384;  // > RaBitQSpace<>::kDegreeBound (32).
constexpr core::RowCount kQueryCount = 16;
constexpr core::RowCount kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-laser-" + std::string(name) + "-" +
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

// ~30% pass rate: lands the Gate10 planner in the traversal (admission) band
// rather than prefilter/postfilter, matching
// segmented_collection_laser_filter_test.cpp's selected_for_row().
[[nodiscard]] auto selected_for_row(core::RowCount row) -> bool { return (row % 10) < 3; }

[[nodiscard]] auto make_dataset(core::RowCount rows, std::uint32_t seed) -> Dataset {
  Dataset result;
  result.vectors.resize(static_cast<std::size_t>(rows) * kDim);
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("row-" + std::to_string(row)));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row) * kDim + column] = dist(gen);
    }
  }
  return result;
}

// Queries are lightly perturbed copies of dataset rows, so an approximate
// index should usually (not always -- LASER is rank_only/approximate)
// recall the source row near the top of its results. This is a sanity
// check on search correctness, not a recall benchmark.
[[nodiscard]] auto make_queries(const Dataset &dataset, core::RowCount count, std::uint32_t seed)
    -> std::vector<float> {
  std::vector<float> result(static_cast<std::size_t>(count) * kDim);
  std::mt19937 gen(seed);
  std::normal_distribution<float> noise(0.0F, 0.02F);
  for (core::RowCount query = 0; query < count; ++query) {
    const auto source = (query * 37U + 5U) % dataset.ids.size();
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result[static_cast<std::size_t>(query) * kDim + column] =
          dataset.vectors[source * kDim + column] + noise(gen);
    }
  }
  return result;
}

[[nodiscard]] auto make_options(const std::filesystem::path &root, core::Metric metric)
    -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = metric;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::laser;
  options.quantization = CollectionQuantization::rabitq;
  options.build_threads = 4;
  options.ef_construction = 128;
  return options;
}

void insert_dataset(Collection &collection, const Dataset &dataset) {
  std::vector<CollectionItem> items;
  items.reserve(dataset.ids.size());
  for (core::RowCount row = 0; row < dataset.ids.size(); ++row) {
    CollectionItem item;
    item.logical_id = dataset.ids[static_cast<std::size_t>(row)];
    item.vector = core::TypedTensorView::contiguous(
        dataset.vectors.data() + static_cast<std::ptrdiff_t>(row) * kDim, 1, kDim);
    item.metadata = {{"selected", selected_for_row(row)}};
    items.push_back(std::move(item));
  }
  auto added = collection.add_batch(items, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  ASSERT_EQ(added.value().rows.size(), dataset.ids.size());
}

[[nodiscard]] auto selected_filter() -> CollectionFilter {
  return CollectionFilter(
      [](const core::LogicalId &, const CollectionMetadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("selected"));
      },
      /*selectivity_estimate=*/0.30);
}

[[nodiscard]] auto find_row(const Dataset &dataset, const core::LogicalId &logical_id)
    -> core::RowCount {
  const auto found = std::ranges::find(dataset.ids, logical_id);
  if (found == dataset.ids.end()) {
    throw std::runtime_error("collection_laser_target_test: unknown logical id in response");
  }
  return static_cast<core::RowCount>(std::distance(dataset.ids.begin(), found));
}

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

// Every hit's logical id must be a known dataset row, and (when queries.rows
// == valid_counts.size() and each row got exactly top_k, as constructed
// here) offsets must be contiguous multiples of top_k.
void expect_well_formed(const CollectionSearchResponse &response,
                        const Dataset &dataset,
                        core::RowCount expected_queries) {
  ASSERT_EQ(response.offsets.size(), expected_queries + 1);
  ASSERT_EQ(response.valid_counts.size(), expected_queries);
  for (core::RowCount query = 0; query < expected_queries; ++query) {
    EXPECT_TRUE(response.statuses[query].ok()) << response.statuses[query].diagnostic();
    const auto begin = response.offsets[query];
    const auto end = response.offsets[query + 1];
    EXPECT_EQ(end - begin, response.valid_counts[query]);
    for (auto index = begin; index < end; ++index) {
      EXPECT_NO_THROW((void)find_row(dataset, response.ids[static_cast<std::size_t>(index)]));
    }
  }
}

TEST(CollectionLaserTargetTest, CreateUpsertSealSearchFilterReopenAndRotate) {
  TemporaryDirectory temporary("lifecycle");
  const auto dataset = make_dataset(kRows, /*seed=*/20260716U);
  const auto queries = make_queries(dataset, kQueryCount, /*seed=*/7U);

  auto created = Collection::create(make_options(temporary.path(), core::Metric::l2));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::laser);
  EXPECT_FALSE(sealed.value().flat_fallback);
  EXPECT_TRUE(sealed.value().fallback_reason.empty());
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  expect_laser_manifest(temporary.path());

  // ---- Plain search --------------------------------------------------
  auto plain = collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(plain.ok()) << plain.status().diagnostic();
  ASSERT_EQ(plain.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  expect_well_formed(plain.value(), dataset, kQueryCount);

  std::size_t self_recalled = 0;
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto source = (query * 37U + 5U) % dataset.ids.size();
    const auto begin = plain.value().offsets[query];
    const auto end = plain.value().offsets[query + 1];
    for (auto index = begin; index < end; ++index) {
      if (plain.value().ids[static_cast<std::size_t>(index)] == dataset.ids[source]) {
        ++self_recalled;
        break;
      }
    }
  }
  // Loose sanity floor, not a recall benchmark: LASER is an approximate,
  // RaBitQ-quantized index, so exact top-1 self-match is not guaranteed for
  // every query, but a small/well-separated random dataset like this one
  // should recall it for the large majority.
  EXPECT_GE(self_recalled, static_cast<std::size_t>(kQueryCount) * 3 / 4)
      << "self_recalled=" << self_recalled << "/" << kQueryCount;

  // ---- Filter search: traversal admission, decision-7 style ----------
  core::SearchContext filter_context;
  auto filtered = collection->search(core::TypedTensorView::contiguous(queries.data(), 1, kDim),
                                     kTopK,
                                     selected_filter());
  ASSERT_TRUE(filtered.ok()) << filtered.status().diagnostic();
  EXPECT_EQ(filtered.value().search_stats.filter_execution, core::FilterExecution::traversal)
      << "selectivity 0.30 must land in the traversal band (0.15, 0.60]";
  ASSERT_FALSE(filtered.value().ids.empty());
  for (const auto &logical_id : filtered.value().ids) {
    const auto row = find_row(dataset, logical_id);
    EXPECT_TRUE(selected_for_row(row)) << "hit row " << row << " fails the predicate";
  }

  // ---- close -> reopen (through the manifest-driven registry path,
  // open_laser_collection_target()) -> search still works -------------
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto opened = Collection::open(temporary.path());
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto reopened = std::move(opened).value();

  auto after = reopened->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  ASSERT_EQ(after.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  expect_well_formed(after.value(), dataset, kQueryCount);
  // Not a byte-identical check: LASER's default residency is the paged-pool
  // kernel (disk_search_qg), whose beam search interleaves computation with
  // asynchronous page-read completions -- tests/laser/qg/test_admission_contract.cpp
  // and tests/disk/test_unified_laser_admission.cpp both document (and
  // gdb-confirmed) that repeated searches, even against the very same
  // QuantizedGraph instance, can explore the graph in a different order and
  // land on a different (individually valid) top-k. Comparing two
  // *different* instances (pre-close vs. post-reopen) is at least as
  // exposed to that nondeterminism, so this checks a meaningful overlap
  // floor instead of set equality.
  std::size_t overlap = 0;
  std::size_t total = 0;
  for (core::RowCount query = 0; query < kQueryCount; ++query) {
    const auto before_begin = plain.value().offsets[query];
    const auto before_end = plain.value().offsets[query + 1];
    std::set<core::LogicalId, internal::collection::LogicalIdLess> before_ids;
    for (auto index = before_begin; index < before_end; ++index) {
      before_ids.insert(plain.value().ids[static_cast<std::size_t>(index)]);
    }
    const auto after_begin = after.value().offsets[query];
    const auto after_end = after.value().offsets[query + 1];
    for (auto index = after_begin; index < after_end; ++index) {
      ++total;
      if (before_ids.count(after.value().ids[static_cast<std::size_t>(index)]) > 0) {
        ++overlap;
      }
    }
  }
  std::cout << "laser_reopen_overlap," << overlap << "/" << total << "\n";
  EXPECT_GT(overlap, total / 2) << "reopened search should mostly agree with the pre-close one";

  // ---- rotate_to_successor: also a second same-process LASER build ---
  // prepare_successor() seals the *active* mutable segment into a new
  // target; immediately after seal() that segment is empty (every row that
  // existed was moved into the segment seal() just sealed), and
  // prepare_successor() refuses to seal an empty active segment. So a few
  // new rows have to land in it first -- mirrors
  // collection_qg_seal_test.cpp's post-reopen `active` dataset insert.
  // prepare_successor() resolves the successor's target algorithm from the
  // *active* segment's own live row count (laser_target_support() gates on
  // count > RaBitQSpace<>::kDegreeBound == 32, same as qg_target_support()),
  // not the whole Collection's total -- so this batch has to individually
  // clear that floor for the successor to build as LASER too rather than
  // silently falling back to flat for being "too small" (a legitimate,
  // separate outcome this test does not want to exercise here).
  constexpr int kActiveRows = 48;
  Dataset active;
  active.ids.reserve(kActiveRows);
  active.vectors.reserve(static_cast<std::size_t>(kActiveRows) * kDim);
  {
    std::mt19937 gen(31415U);
    std::normal_distribution<float> dist(0.0F, 1.0F);
    for (int index = 0; index < kActiveRows; ++index) {
      active.ids.push_back(core::LogicalId::from_utf8("active-" + std::to_string(index)));
      for (std::uint32_t column = 0; column < kDim; ++column) {
        active.vectors.push_back(dist(gen));
      }
    }
  }
  insert_dataset(*reopened, active);

  auto handle = reopened->prepare_successor();
  ASSERT_TRUE(handle.ok()) << handle.status().diagnostic();
  ASSERT_TRUE(handle.value().ready());
  auto rotated = reopened->rotate_to_successor(handle.value());
  ASSERT_TRUE(rotated.ok()) << rotated.status().diagnostic();
  EXPECT_EQ(rotated.value().built_algorithm, core::algorithm::laser);
  EXPECT_FALSE(rotated.value().flat_fallback);

  Dataset combined = dataset;
  combined.ids.insert(combined.ids.end(), active.ids.begin(), active.ids.end());
  combined.vectors.insert(combined.vectors.end(), active.vectors.begin(), active.vectors.end());

  auto after_rotate = reopened->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(after_rotate.ok()) << after_rotate.status().diagnostic();
  ASSERT_EQ(after_rotate.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  // Hits can legitimately come from either the just-rotated LASER successor
  // (the original 384 rows) or the freshly inserted `active` rows above, so
  // this checks against their union, not `dataset` alone.
  expect_well_formed(after_rotate.value(), combined, kQueryCount);

  ASSERT_TRUE(reopened->close().ok());
}

TEST(CollectionLaserTargetTest, ResidentArenaResidencyViaEnvOverrideThroughCollection) {
  // Decision 6 (U2-c manifest): ALAYA_LASER_RESIDENCY drives both
  // build_laser_collection_target()'s manifest hint (what LaserSegmentImportParams::residency
  // gets persisted as) and LaserSegment::open()'s residency selection (env
  // overrides the manifest either way) -- this is the "unified arena
  // through Collection" residency variant of the lifecycle test above,
  // which never sets this env var and stays on the legacy default path.
  TemporaryDirectory temporary("resident-arena");
  const auto dataset = make_dataset(kRows, /*seed=*/271828U);
  const auto queries = make_queries(dataset, kQueryCount, /*seed=*/17U);

  ::setenv("ALAYA_LASER_RESIDENCY", "resident_arena", 1);
  struct EnvGuard {
    ~EnvGuard() { ::unsetenv("ALAYA_LASER_RESIDENCY"); }
  } env_guard;

  auto created = Collection::create(make_options(temporary.path(), core::Metric::l2));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::laser);
  EXPECT_FALSE(sealed.value().flat_fallback);

  // Confirm the native manifest actually recorded resident_arena -- this is
  // what makes the rest of this test a residency proof rather than "it
  // happens to pass the same way regardless of residency."
  const auto seg_dir =
      temporary.path() / "segments" /
      internal::collection::detail::collection_segment_name(sealed.value().sealed_segment_id);
  const auto native_manifest = ::alaya::disk::SegmentManifest::load(seg_dir / "manifest.txt");
  const auto residency_extra = native_manifest.x_extras.find("x_laser_residency");
  ASSERT_NE(residency_extra, native_manifest.x_extras.end());
  EXPECT_EQ(residency_extra->second, "resident_arena");

  auto plain = collection->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(plain.ok()) << plain.status().diagnostic();
  ASSERT_EQ(plain.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  expect_well_formed(plain.value(), dataset, kQueryCount);

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto opened = Collection::open(temporary.path());
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto reopened = std::move(opened).value();

  auto after = reopened->batch_search(
      core::TypedTensorView::contiguous(queries.data(), kQueryCount, kDim), kTopK);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  ASSERT_EQ(after.value().valid_counts,
            std::vector<core::RowCount>(static_cast<std::size_t>(kQueryCount), kTopK));
  expect_well_formed(after.value(), dataset, kQueryCount);
  // Unlike the paged-pool default lifecycle test above, the resident-arena
  // kernel has no async I/O (fully deterministic given the same graph state
  // and query), so this can assert byte-identical results across the
  // reopen instead of an overlap floor.
  EXPECT_EQ(after.value().ids, plain.value().ids);

  ASSERT_TRUE(reopened->close().ok());
}

TEST(CollectionLaserTargetFallback, InnerProductMetricFallsBackToFlat) {
  // LASER is L2-only (include/index/graph/laser/space/ has only l2.hpp);
  // laser_target_support() must reject any other metric so
  // resolve_build_algorithm() falls back to flat instead of attempting (and
  // failing) a LASER build.
  TemporaryDirectory temporary("ip-fallback");
  const auto dataset = make_dataset(kRows, /*seed=*/4104U);

  auto created = Collection::create(make_options(temporary.path(), core::Metric::inner_product));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().built_algorithm, core::algorithm::flat);
  EXPECT_TRUE(sealed.value().flat_fallback);
  EXPECT_NE(sealed.value().fallback_reason.find("laser"), std::string::npos)
      << sealed.value().fallback_reason;
  expect_flat_manifest(temporary.path());

  ASSERT_TRUE(collection->close().ok());
}

}  // namespace
}  // namespace alaya
