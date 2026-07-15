// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/disk/disk_vamana_segment.hpp"
#include "simd/distance_l2.hpp"

namespace alaya::disk {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-disk-vamana-segment-test-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct FixtureRows {
  static constexpr std::uint32_t kDim = 8;
  static constexpr std::uint64_t kRows = 128;

  FixtureRows() {
    vectors.resize(kRows * kDim);
    labels.resize(kRows);
    std::mt19937 random(20260712);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    for (std::uint64_t row = 0; row < kRows; ++row) {
      labels[row] = 20'000 + row * 11;
      for (std::uint32_t column = 0; column < kDim; ++column) {
        vectors[row * kDim + column] = distribution(random);
      }
    }
  }

  [[nodiscard]] auto input() const -> DiskVamanaBuildInput {
    return {core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), labels};
  }

  [[nodiscard]] static auto params() -> VamanaSegmentBuildParams {
    VamanaSegmentBuildParams value;
    value.R = 8;
    value.L = 24;
    value.alpha = 1.2F;
    value.num_threads = 1;
    value.seed = 424242;
    return value;
  }

  std::vector<float> vectors{};
  std::vector<std::uint64_t> labels{};
};

struct SearchCall {
  SearchCall(const float *queries,
             core::RowCount rows,
             std::uint32_t dim,
             std::uint64_t top_k,
             std::uint32_t effort)
      : hits(static_cast<std::size_t>(rows * top_k)),
        offsets(static_cast<std::size_t>(rows + 1)),
        counts(static_cast<std::size_t>(rows)),
        statuses(static_cast<std::size_t>(rows)),
        completeness(static_cast<std::size_t>(rows)) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    search_options.effort = effort;
    extension = DiskVamanaSegment::make_search_extension(search_options);
    request.queries = core::TypedTensorView::contiguous(queries, rows, dim);
    request.options.top_k = top_k;
    request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = std::addressof(context);
    request.response = std::addressof(response);
  }

  core::SearchContext context{};
  DiskVamanaSearchExtension search_options{};
  core::AlgorithmSearchExtension extension{};
  std::vector<core::SearchHit> hits{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  core::SearchResponse response{};
  core::SearchRequest request{};
};

[[nodiscard]] auto bytes(const std::filesystem::path &path) -> std::vector<char> {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open test artifact: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] auto build_segment(const FixtureRows &rows,
                                 const std::filesystem::path &root,
                                 bool manifest_v2 = false) -> std::unique_ptr<DiskVamanaSegment> {
  DiskVamanaPublicationOptions options;
  options.collection_root = root;
  options.segment_id = "seg_00000001";
  options.collection_features.manifest_v2_writer = manifest_v2;
  core::BuildContext context;
  auto built = DiskVamanaSegmentFactory::build(rows.input(),
                                               core::Metric::l2,
                                               FixtureRows::params(),
                                               options,
                                               context);
  if (!built.ok()) {
    throw std::runtime_error(built.status().diagnostic());
  }
  return std::move(built).value();
}

void expect_l2_gate(const core::Status &status, [[maybe_unused]] std::string_view metric) {
  EXPECT_EQ(status.code(), core::StatusCode::not_supported);
  EXPECT_EQ(status.detail(), core::StatusDetail::operation_slot_absent);
  EXPECT_NE(status.diagnostic().find("L2 only"), std::string::npos) << status.diagnostic();
}

TEST(DiskVamanaSegment, DescriptorCapabilitiesRegistryAndFeatureGateAreExplicit) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  auto segment = build_segment(rows, temporary.path() / "enabled");
  const auto descriptor = segment->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::vamana);
  EXPECT_EQ(descriptor.format_version, 1);
  EXPECT_EQ(descriptor.factory_version, 1);
  EXPECT_EQ(descriptor.medium, core::Medium::disk);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);

  auto any_result = DiskVamanaSegment::into_any(std::move(segment));
  ASSERT_TRUE(any_result.ok()) << any_result.status().diagnostic();
  auto any = std::move(any_result).value();
  const auto capabilities = any.capabilities();
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::save));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(core::OperationCapability::export_rows));
  EXPECT_FALSE(capabilities.supports(core::OperationCapability::mutation));
  core::OpaqueOperationRequest opaque_request;
  core::ExportCursor export_cursor;
  EXPECT_EQ(any.export_rows(opaque_request, export_cursor).code(), core::StatusCode::not_supported);
  core::MutationContext mutation_context;
  core::MutationToken mutation_token;
  EXPECT_EQ(any.prepare_mutation(opaque_request, mutation_context, mutation_token).code(),
            core::StatusCode::not_supported);

  internal::disk::DiskEngineFeatureFlags disabled;
  disabled.disk_vamana_segment = false;
  DiskVamanaPublicationOptions options;
  options.collection_root = temporary.path() / "disabled";
  options.segment_id = "seg_00000001";
  core::BuildContext context;
  auto rejected = DiskVamanaSegmentFactory::build(rows.input(),
                                                  core::Metric::l2,
                                                  FixtureRows::params(),
                                                  options,
                                                  context,
                                                  disabled);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::not_supported);
  EXPECT_FALSE(std::filesystem::exists(options.collection_root));
  core::OpenContext disabled_open_context;
  auto rejected_open = DiskVamanaSegmentFactory::open(core::ArtifactView{},
                                                      core::OpenOptions{},
                                                      disabled_open_context,
                                                      disabled);
  ASSERT_FALSE(rejected_open.ok());
  EXPECT_EQ(rejected_open.status().code(), core::StatusCode::not_supported);

  EXPECT_EQ(DiskVamanaSegmentFactory::registration.current.declared_index_type, "disk_vamana");
  EXPECT_EQ(DiskVamanaSegmentFactory::registration.current.implementation_key,
            "disk_vamana_segment");
  EXPECT_EQ(DiskVamanaSegmentFactory::registration.current.engine_factory_key, "vamana");
}

TEST(DiskVamanaSegment, NonL2BuildAndOpenReturnTheFirstVersionGate) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  for (const auto [metric, name] :
       {std::pair{core::Metric::inner_product, std::string_view("inner_product")},
        std::pair{core::Metric::cosine, std::string_view("cosine")}}) {
    DiskVamanaPublicationOptions options;
    options.collection_root = temporary.path() / std::string(name);
    options.segment_id = "seg_00000001";
    core::BuildContext context;
    auto rejected = DiskVamanaSegmentFactory::build(rows.input(),
                                                    metric,
                                                    FixtureRows::params(),
                                                    options,
                                                    context);
    ASSERT_FALSE(rejected.ok());
    expect_l2_gate(rejected.status(), name);
    EXPECT_FALSE(std::filesystem::exists(options.collection_root));
  }

}

TEST(DiskVamanaSegment, TypedTensorAndContextResourcesAreEnforced) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  DiskVamanaPublicationOptions options;
  options.collection_root = temporary.path() / "budget-build";
  options.segment_id = "seg_00000001";
  core::BuildContext build_context;
  build_context.growing_reservation = core::MemoryReservation(1);
  auto denied_build = DiskVamanaSegmentFactory::build(rows.input(),
                                                      core::Metric::l2,
                                                      FixtureRows::params(),
                                                      options,
                                                      build_context);
  ASSERT_FALSE(denied_build.ok());
  EXPECT_EQ(denied_build.status().code(), core::StatusCode::resource_exhausted);

  std::vector<std::int8_t> int8_vectors(FixtureRows::kRows * FixtureRows::kDim, 1);
  DiskVamanaBuildInput int8_input(core::TypedTensorView::contiguous(int8_vectors.data(),
                                                                    FixtureRows::kRows,
                                                                    FixtureRows::kDim),
                                  rows.labels);
  DiskVamanaPublicationOptions int8_options;
  int8_options.collection_root = temporary.path() / "int8";
  int8_options.segment_id = "seg_00000001";
  core::BuildContext int8_context;
  auto denied_scalar = DiskVamanaSegmentFactory::build(int8_input,
                                                       core::Metric::l2,
                                                       FixtureRows::params(),
                                                       int8_options,
                                                       int8_context);
  ASSERT_FALSE(denied_scalar.ok());
  EXPECT_EQ(denied_scalar.status().code(), core::StatusCode::not_supported);

  const auto root = temporary.path() / "built";
  auto segment = build_segment(rows, root);
  core::OpenContext open_context;
  open_context.resident_lease = core::MemoryLease(1);
  auto denied_open = DiskVamanaSegment::open_directory(root / "segments/seg_00000001",
                                                       core::OpenOptions{},
                                                       open_context);
  ASSERT_FALSE(denied_open.ok());
  EXPECT_EQ(denied_open.status().code(), core::StatusCode::resource_exhausted);

  SearchCall scratch_denied(rows.vectors.data(), 1, FixtureRows::kDim, 4, 24);
  scratch_denied.context.query_scratch_lease = core::MemoryLease(1);
  EXPECT_EQ(segment->search(scratch_denied.request).code(), core::StatusCode::resource_exhausted);
  SearchCall io_denied(rows.vectors.data(), 1, FixtureRows::kDim, 4, 24);
  io_denied.context.io_credits.available_bytes = 1;
  EXPECT_EQ(segment->search(io_denied.request).code(), core::StatusCode::resource_exhausted);

  std::array<std::int8_t, FixtureRows::kDim> int8_query{};
  SearchCall typed_search(rows.vectors.data(), 1, FixtureRows::kDim, 4, 24);
  typed_search.request.queries =
      core::TypedTensorView::contiguous(int8_query.data(), 1, FixtureRows::kDim);
  EXPECT_EQ(segment->search(typed_search.request).code(), core::StatusCode::not_supported);
}

TEST(DiskVamanaSegment, ManifestV2RoundTripsAndRejectsDamagedNativeArtifact) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  const auto root = temporary.path() / "v2";
  auto segment = build_segment(rows, root, true);
  const auto segment_dir = root / "segments/seg_00000001";
  EXPECT_TRUE(std::filesystem::is_regular_file(root / "collection_manifest.txt"));
  EXPECT_TRUE(std::filesystem::is_regular_file(segment_dir / "READY"));
  EXPECT_TRUE(std::filesystem::is_regular_file(segment_dir / "ARTIFACTS.v2"));

  auto unified = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(unified.ok()) << unified.status().diagnostic();
  ASSERT_EQ(unified.value().manifest.segments.size(), 1);
  const auto &entry = unified.value().manifest.segments[0];
  EXPECT_EQ(entry.algorithm_id, core::algorithm::vamana);
  EXPECT_EQ(entry.factory_key, "vamana");
  EXPECT_TRUE((entry.capabilities.operations &
               core::capability_bit(core::OperationCapability::search)) != 0);
  EXPECT_TRUE(
      (entry.capabilities.operations & core::capability_bit(core::OperationCapability::save)) != 0);
  EXPECT_EQ((entry.capabilities.operations &
             core::capability_bit(core::OperationCapability::export_rows)),
            0);

  core::OpenContext context;
  auto reopened =
      DiskVamanaSegment::open_collection(root, "seg_00000001", core::OpenOptions{}, context);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->descriptor().algorithm_id, core::algorithm::vamana);

  auto graph = bytes(segment_dir / "graph.index");
  ASSERT_GT(graph.size(), 32);
  graph.back() = static_cast<char>(graph.back() ^ 0x5A);
  platform::write_all_fsync(segment_dir / "graph.index", graph.data(), graph.size());
  auto rejected =
      DiskVamanaSegment::open_collection(root, "seg_00000001", core::OpenOptions{}, context);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::corruption);
  (void)segment;
}

TEST(DiskVamanaSegment, BuildCrashCutsNeverRouteAndRestartCleanupRemovesOrphans) {
  FixtureRows rows;
  const std::array
      fail_points{internal::collection::ArtifactTransactionFailPoint::after_staging_write,
                  internal::collection::ArtifactTransactionFailPoint::before_ready,
                  internal::collection::ArtifactTransactionFailPoint::after_ready_before_publish,
                  internal::collection::ArtifactTransactionFailPoint::
                      after_payload_publish_before_manifest};
  for (const auto fail_point : fail_points) {
    SCOPED_TRACE(static_cast<unsigned>(fail_point));
    TemporaryDirectory temporary;
    DiskVamanaPublicationOptions options;
    options.collection_root = temporary.path() / "collection";
    options.segment_id = "seg_00000001";
    options.collection_features.manifest_v2_writer = true;
    options.abort_policy = internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
    options.fail_point = fail_point;
    core::BuildContext context;
    auto rejected = DiskVamanaSegmentFactory::build(rows.input(),
                                                    core::Metric::l2,
                                                    FixtureRows::params(),
                                                    options,
                                                    context);
    ASSERT_FALSE(rejected.ok());
    EXPECT_FALSE(std::filesystem::exists(options.collection_root / "collection_manifest.txt"));
    auto cleanup = internal::collection::ArtifactControlPlaneTransaction::cleanup_orphans(
        options.collection_root);
    ASSERT_TRUE(cleanup.ok()) << cleanup.diagnostic();
    EXPECT_FALSE(std::filesystem::exists(options.collection_root / ".alaya_staging"));
    EXPECT_FALSE(std::filesystem::exists(options.collection_root / "segments/seg_00000001"));
  }
}

}  // namespace
}  // namespace alaya::disk
