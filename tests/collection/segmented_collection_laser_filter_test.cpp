// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// Collection-level end-to-end tests for the segment admission contract
// (docs/design/segment-admission-contract.md), decisions 7-8 of the U2-b
// manifest:
//   - contract acceptance #3: a bitmap-filter recall test through the
//     Gate10 planner's mid-band (traversal) route, on a real LASER
//     segment -- not_supported no longer occurs, results satisfy the
//     predicate, and ResultFlag::filtered is set.
//   - the auto-policy fallback (decision 7): a mixed collection (a QG
//     memory segment, which still rejects any non-none filter kind, plus
//     the LASER segment) does not fail the whole query under the default
//     `automatic` filter policy.
//
// LaserSegment::open()/open_directory() always constructs the legacy
// (paged-pool) LaserSegmentSearcher -- there is no production path that
// routes a Collection-owned LASER segment through the resident-arena
// kernel today (UnifiedLaserSegmentSearcher is a lower-level primitive
// nothing in the Collection/AnySegment layer currently selects). So unlike
// tests/disk/test_unified_laser_admission.cpp, which exercises both
// residencies directly against UnifiedLaserSegmentSearcher, this file's
// "through Collection" tests necessarily exercise the one residency
// reachable that way (paged pool).

#include "index/collection/segmented_collection.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/laser_segment_importer.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "platform/detect.hpp"
#include "space/rabitq_space.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace alaya::internal::collection {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint64_t kLaserCount = 512;
constexpr std::uint32_t kR = 64;
constexpr std::uint64_t kLaserLabelBase = 500'000;
constexpr std::uint32_t kQgRows = 200;
constexpr std::uint32_t kQgCapacity = 256;
constexpr std::uint64_t kLaserSegmentId = 1;
constexpr std::uint64_t kQgSegmentId = 2;

std::vector<float> make_data(std::uint64_t n, std::uint32_t dim, std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(static_cast<std::size_t>(n) * dim);
  for (auto &v : data) {
    v = dist(gen);
  }
  return data;
}

// ~30% pass rate, matching the selectivity hint the tests below use --
// lands the Gate10 planner in the traversal band (0.15, 0.60].
[[nodiscard]] auto selected_for_row(std::uint64_t row) -> bool { return (row % 10) < 3; }

[[nodiscard]] auto row_payload(const float *vector, std::uint32_t dim, bool selected)
    -> RecordPayload {
  auto owned = OwnedVector::copy_row(core::TypedTensorView::contiguous(vector, 1, dim), 0);
  RecordPayload payload;
  if (owned.ok()) {
    payload.vector = std::move(owned).value();
  }
  payload.metadata = {{"selected", selected}};
  return payload;
}

[[nodiscard]] auto selected_filter() -> LogicalFilter {
  return LogicalFilter(
      [](const core::LogicalId &, const Metadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("selected"));
      },
      /*selectivity_estimate=*/0.30);
}

// The on-disk LASER segment is built once per process: QGBuilder::build()'s
// out-of-memory patch path has a pre-existing, data-dependent crash
// unrelated to this change (see tests/laser/qg/test_admission_contract.cpp
// and tests/disk/test_unified_laser_admission.cpp for the gdb-confirmed
// root cause: a memmove inside QuantizedGraph::update_qg_out_of_memory,
// reached only through the build path) when several small indices are
// built back to back in one process; building once avoids it.
struct LaserFixture {
  std::filesystem::path root;
  std::filesystem::path seg_dir;
  std::vector<float> data;
  std::vector<std::uint64_t> labels;

  LaserFixture() = default;
  LaserFixture(const LaserFixture &) = delete;
  auto operator=(const LaserFixture &) -> LaserFixture & = delete;

  ~LaserFixture() {
    if (root.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }

  static auto build() -> std::unique_ptr<LaserFixture> {
    auto fx = std::make_unique<LaserFixture>();
    fx->root = std::filesystem::temp_directory_path() /
               ("segmented_collection_laser_filter_test_" + std::to_string(platform::get_pid()));
    std::error_code ec;
    std::filesystem::remove_all(fx->root, ec);
    std::filesystem::create_directories(fx->root / "segments");
    fx->seg_dir = fx->root / "segments/seg_00000001";

    fx->data = make_data(kLaserCount, kDim, /*seed=*/4242);
    fx->labels.resize(kLaserCount);
    for (std::uint64_t pid = 0; pid < kLaserCount; ++pid) {
      fx->labels[pid] = kLaserLabelBase + pid;
    }

    const auto raw_dir = fx->root / "raw";
    std::filesystem::create_directories(raw_dir);
    const std::string raw_prefix = (raw_dir / "dsqg_seg_00000001").string();

    alaya::vamana::VamanaBuildParams vp;
    vp.R = kR;
    vp.L = 96;
    vp.alpha = 1.2F;
    vp.num_threads = 4;
    alaya::vamana::VamanaBuilder vb(fx->data.data(), kLaserCount, kDim, vp);
    vb.build();
    const std::string vamana_path = raw_prefix + "_vamana.index";
    alaya::vamana::save_graph(vb.graph(), vamana_path, kR, vb.medoid());

    alaya::laser::QuantizedGraph qg(kLaserCount, kR, kDim, kDim, /*rotator_seed=*/7);
    alaya::laser::QGBuilder builder(qg, /*ef_build=*/96, /*num_threads=*/4);
    builder.build(vamana_path.c_str(), raw_prefix.c_str());

    ::alaya::disk::LaserSegmentImportParams params;
    params.R = kR;
    ::alaya::disk::LaserSegmentImporter importer(kDim, core::Metric::l2, params);
    (void)importer.import_from(raw_dir, fx->labels.data(), fx->labels.size(), fx->seg_dir);
    return fx;
  }
};

// LASER hits are rank_only (approximate, no comparable distance) --
// mirrors the exact_rerank seam laser_segment_test.cpp's
// DifferentialRankOnlyManifestGate... test uses so Collection can normalize
// them into comparable scores (a pre-existing Collection requirement, not
// specific to filtering: mixing a rank_only-scoring segment with anything
// else, or even alone, needs this to sort/merge hits at all).
[[nodiscard]] auto open_laser_registration(const LaserFixture &fixture)
    -> SegmentRegistration {
  core::OpenContext open_context;
  auto opened =
      ::alaya::disk::LaserSegment::open_directory(fixture.seg_dir, core::OpenOptions{}, open_context);
  EXPECT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto any = ::alaya::disk::LaserSegment::into_any(std::move(opened).value());
  EXPECT_TRUE(any.ok()) << any.status().diagnostic();

  SegmentRegistration registration;
  registration.segment_id = kLaserSegmentId;
  registration.role = SegmentRole::sealed;
  registration.segment = std::move(any).value();
  registration.rows.reserve(kLaserCount);
  for (std::uint64_t pid = 0; pid < kLaserCount; ++pid) {
    const auto label = fixture.labels[pid];
    registration.rows.push_back(
        {core::LogicalId::from_legacy_uint64(label),
         core::SegmentRowId(label),
         pid + 1,
         VersionState::live,
         row_payload(fixture.data.data() + pid * kDim, kDim, selected_for_row(pid))});
  }
  registration.exact_rerank = [&fixture](const core::TypedTensorView &query,
                                         core::SegmentRowId row_id) -> core::Result<float> {
    const auto label = static_cast<std::uint64_t>(row_id);
    if (label < kLaserLabelBase || (label - kLaserLabelBase) >= kLaserCount) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::search,
                                 core::StatusDetail::malformed_struct,
                                 "LASER row_id outside the fixture's label range");
    }
    const auto pid = label - kLaserLabelBase;
    const float *vec = fixture.data.data() + pid * kDim;
    float distance = 0.0F;
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto delta = query.row<float>(0)[column] - vec[column];
      distance += delta * delta;
    }
    return distance;
  };
  return registration;
}

using QgSpace = RaBitQSpace<>;
using QgMemorySegment = QgSegment<QgSpace>;

[[nodiscard]] auto build_qg_registration(std::vector<float> &qg_data) -> SegmentRegistration {
  qg_data = make_data(kQgRows, kDim, /*seed=*/777);
  auto space = std::make_shared<QgSpace>(kQgCapacity, kDim, core::Metric::l2);
  space->fit(qg_data.data(), kQgRows);
  core::BuildContext build_context;
  auto segment = QgMemorySegment::build(
      {core::TypedTensorView::contiguous(qg_data.data(), kQgRows, kDim), space},
      {.ef_build = 64, .thread_count = 1},
      build_context);
  auto any = QgMemorySegment::into_any(std::move(segment));
  EXPECT_TRUE(any.ok()) << any.status().diagnostic();

  SegmentRegistration registration;
  registration.segment_id = kQgSegmentId;
  registration.role = SegmentRole::sealed;
  registration.segment = std::move(any).value();
  registration.rows.reserve(kQgRows);
  for (std::uint32_t row = 0; row < kQgRows; ++row) {
    registration.rows.push_back(
        {core::LogicalId::from_utf8("qg-" + std::to_string(row)),
         core::SegmentRowId(row),
         row + 1,
         VersionState::live,
         row_payload(qg_data.data() + row * kDim, kDim, selected_for_row(row))});
  }
  return registration;
}

class LaserFilterCollectionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { fixture_ = LaserFixture::build(); }
  static void TearDownTestSuite() { fixture_.reset(); }

  static std::unique_ptr<LaserFixture> fixture_;
};

std::unique_ptr<LaserFixture> LaserFilterCollectionTest::fixture_;

// ---------------------------------------------------------------------------
// Contract acceptance #3: bitmap-filter recall through the Gate10 planner's
// mid-band (traversal) route, on a real LASER segment.
// ---------------------------------------------------------------------------

TEST_F(LaserFilterCollectionTest, BitmapFilterTraversalExecutesOnLaserSegment) {
  auto registration = open_laser_registration(*fixture_);
  auto opened = SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();

  core::SearchContext context;
  CollectionSearchStats stats;
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(fixture_->data.data(), 1, kDim);
  request.options.top_k = 10;
  request.filter = selected_filter();
  request.context = &context;
  request.stats = &stats;

  auto result = collection->search(request);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  EXPECT_EQ(stats.filter_execution, core::FilterExecution::traversal)
      << "selectivity 0.30 must land in the traversal band (0.15, 0.60]";
  ASSERT_FALSE(result.value().queries.empty());
  const auto &query_result = result.value().queries[0];
  EXPECT_TRUE(query_result.status.ok());
  EXPECT_GT(query_result.hits.size(), 0U);
  for (const auto &hit : query_result.hits) {
    ASSERT_EQ(hit.source.segment_id, kLaserSegmentId);
    const auto label = static_cast<std::uint64_t>(hit.source.row_id);
    ASSERT_GE(label, kLaserLabelBase);
    const auto pid = label - kLaserLabelBase;
    EXPECT_TRUE(selected_for_row(pid)) << "hit pid " << pid << " fails the predicate";
    EXPECT_NE(static_cast<std::uint32_t>(hit.result_flags) &
                  static_cast<std::uint32_t>(core::ResultFlag::filtered),
              0U)
        << "hit must carry ResultFlag::filtered under an active traversal admission";
  }
}

// select_filter_execution() maps FilterPolicy::strict to prefilter
// unconditionally (a pre-existing, decision-8-protected planner rule this
// change does not touch), so a strict-policy query never reaches the
// traversal/admission code path at all -- it takes Collection's own
// brute-force exact_search scan instead. This just confirms that policy
// still gets a correct answer through this change, on a LASER-only
// collection (exact_search does not touch segments, so this exercises no
// admission code -- it is a regression guard, not admission coverage).
TEST_F(LaserFilterCollectionTest, StrictPolicyStillWorksViaPrefilterNotTraversal) {
  auto registration = open_laser_registration(*fixture_);
  auto opened = SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();

  core::SearchContext context;
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(fixture_->data.data(), 1, kDim);
  request.options.top_k = 10;
  request.options.filter_policy = core::FilterPolicy::strict;
  request.filter = selected_filter();
  request.context = &context;

  auto result = collection->search(request);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  ASSERT_FALSE(result.value().queries.empty());
  EXPECT_GT(result.value().queries[0].hits.size(), 0U);
  for (const auto &hit : result.value().queries[0].hits) {
    const auto label = static_cast<std::uint64_t>(hit.source.row_id);
    ASSERT_GE(label, kLaserLabelBase);
    EXPECT_TRUE(selected_for_row(label - kLaserLabelBase));
  }
}

// ---------------------------------------------------------------------------
// Decision 7's auto-policy fallback: a mixed collection (a QG memory
// segment, which still rejects any non-none filter kind, plus the LASER
// segment, which executes it) must not fail the whole query.
// ---------------------------------------------------------------------------

TEST_F(LaserFilterCollectionTest, MixedQgAndLaserAutoPolicyDoesNotFailOverall) {
  std::vector<float> qg_data;
  auto qg_registration = build_qg_registration(qg_data);
  auto laser_registration = open_laser_registration(*fixture_);
  auto opened =
      SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                {std::move(qg_registration), std::move(laser_registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();

  core::SearchContext context;
  CollectionSearchStats stats;
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(fixture_->data.data(), 1, kDim);
  request.options.top_k = 10;
  // filter_policy left at its default: core::FilterPolicy::automatic.
  request.filter = selected_filter();
  request.context = &context;
  request.stats = &stats;

  auto result = collection->search(request);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic()
                           << " (auto policy must fall back on the qg segment's not_supported "
                              "rather than failing the whole query)";
  EXPECT_EQ(stats.filter_execution, core::FilterExecution::traversal);
  ASSERT_FALSE(result.value().queries.empty());
  const auto &query_result = result.value().queries[0];
  EXPECT_TRUE(query_result.status.ok());
  EXPECT_GT(query_result.hits.size(), 0U);
  bool saw_laser_hit = false;
  for (const auto &hit : query_result.hits) {
    const auto row_id_value = static_cast<std::uint64_t>(hit.source.row_id);
    std::uint64_t pid = 0;
    if (hit.source.segment_id == kLaserSegmentId) {
      ASSERT_GE(row_id_value, kLaserLabelBase);
      pid = row_id_value - kLaserLabelBase;
      saw_laser_hit = true;
    } else {
      ASSERT_EQ(hit.source.segment_id, kQgSegmentId);
      pid = row_id_value;
    }
    // Every surviving hit must satisfy the predicate regardless of which
    // segment it came from: the LASER segment actually executed the
    // bitmap (traversal-filtered), and the qg segment's rows (returned
    // unfiltered after the automatic-policy retry, since qg still rejects
    // any non-none filter kind) were weeded out by Collection's own
    // traversal re-verify step -- the same safety net that already runs
    // unconditionally for every traversal hit.
    EXPECT_TRUE(selected_for_row(pid))
        << "hit from segment " << hit.source.segment_id << " row " << pid
        << " fails the predicate";
  }
  EXPECT_TRUE(saw_laser_hit) << "expected at least one hit to come from the LASER segment";
}

// allow_partial is the *other* non-automatic policy this change's retry
// deliberately excludes (the retry only fires for
// FilterPolicy::automatic). Before this change, a segment that rejects a
// traversal filter under allow_partial already degrades gracefully
// (exhaustive=false, continue, no error) rather than failing -- confirm
// this change did not disturb that pre-existing path: the query must
// still not fail overall, via that older mechanism rather than the new
// retry.
TEST_F(LaserFilterCollectionTest, MixedQgAndLaserAllowPartialAlsoSucceeds) {
  std::vector<float> qg_data;
  auto qg_registration = build_qg_registration(qg_data);
  auto laser_registration = open_laser_registration(*fixture_);
  auto opened =
      SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                {std::move(qg_registration), std::move(laser_registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();

  core::SearchContext context;
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(fixture_->data.data(), 1, kDim);
  request.options.top_k = 10;
  request.options.filter_policy = core::FilterPolicy::allow_partial;
  request.filter = selected_filter();
  request.context = &context;

  auto result = collection->search(request);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  ASSERT_FALSE(result.value().queries.empty());
  for (const auto &hit : result.value().queries[0].hits) {
    const auto row_id_value = static_cast<std::uint64_t>(hit.source.row_id);
    if (hit.source.segment_id == kLaserSegmentId) {
      ASSERT_GE(row_id_value, kLaserLabelBase);
      EXPECT_TRUE(selected_for_row(row_id_value - kLaserLabelBase));
    } else {
      ASSERT_EQ(hit.source.segment_id, kQgSegmentId);
      EXPECT_TRUE(selected_for_row(row_id_value));
    }
  }
}

}  // namespace
}  // namespace alaya::internal::collection
