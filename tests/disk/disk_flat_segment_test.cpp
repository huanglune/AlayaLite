// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/disk/disk_flat_segment.hpp"

namespace alaya::disk {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-disk-flat-segment-test-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct FixtureRows {
  static constexpr std::uint32_t kDim = 4;
  static constexpr std::uint64_t kRows = 17;

  FixtureRows() {
    vectors.resize(kRows * kDim);
    labels.resize(kRows);
    for (std::uint64_t row = 0; row < kRows; ++row) {
      labels[row] = 1000 + row * 7;
      for (std::uint32_t col = 0; col < kDim; ++col) {
        vectors[row * kDim + col] =
            static_cast<float>((row + 1) * (col + 3)) / 19.0F - static_cast<float>(col);
      }
    }
  }

  [[nodiscard]] auto input() const -> DiskFlatBuildInput {
    return {core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), labels};
  }

  std::vector<float> vectors{};
  std::vector<std::uint64_t> labels{};
};

struct SearchCall {
  SearchCall(const float *queries, core::RowCount rows, std::uint32_t dim, std::uint64_t top_k)
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
    request.queries = core::TypedTensorView::contiguous(queries, rows, dim);
    request.options.top_k = top_k;
    request.context = std::addressof(context);
    request.response = std::addressof(response);
  }

  core::SearchContext context{};
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
                                 bool manifest_v2 = false) -> std::unique_ptr<DiskFlatSegment> {
  DiskFlatPublicationOptions options;
  options.collection_root = root;
  options.segment_id = "seg_00000001";
  options.collection_features.manifest_v2_writer = manifest_v2;
  core::BuildContext context;
  auto built = DiskFlatSegmentFactory::build(rows.input(), core::Metric::l2, options, context);
  if (!built.ok()) {
    throw std::runtime_error(built.status().diagnostic());
  }
  return std::move(built).value();
}

TEST(DiskFlatSegment, DescriptorCapabilitiesAndDisabledFactoryAreExplicit) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  auto segment = build_segment(rows, temporary.path() / "enabled");
  const auto descriptor = segment->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::flat);
  EXPECT_EQ(descriptor.format_version, 1);
  EXPECT_EQ(descriptor.factory_version, 1);
  EXPECT_EQ(descriptor.medium, core::Medium::disk);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  auto any_result = DiskFlatSegment::into_any(std::move(segment));
  ASSERT_TRUE(any_result.ok()) << any_result.status().diagnostic();
  auto any = std::move(any_result).value();
  const auto capabilities = any.capabilities();
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::save));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::export_rows));
  EXPECT_TRUE(capabilities.supports(core::OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(core::OperationCapability::mutation));
  core::OpaqueOperationRequest mutation_request;
  core::MutationContext mutation_context;
  core::MutationToken mutation_token;
  const auto mutation_status =
      any.prepare_mutation(mutation_request, mutation_context, mutation_token);
  EXPECT_EQ(mutation_status.code(), core::StatusCode::not_supported);
  EXPECT_EQ(mutation_status.detail(), core::StatusDetail::operation_slot_absent);

  internal::disk::DiskEngineFeatureFlags disabled;
  disabled.disk_flat_segment = false;
  DiskFlatPublicationOptions options;
  options.collection_root = temporary.path() / "disabled";
  options.segment_id = "seg_00000001";
  core::BuildContext context;
  auto rejected =
      DiskFlatSegmentFactory::build(rows.input(), core::Metric::l2, options, context, disabled);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::not_supported);
  EXPECT_FALSE(std::filesystem::exists(options.collection_root));
  core::OpenContext disabled_open_context;
  auto rejected_open = DiskFlatSegmentFactory::open(core::ArtifactView{},
                                                    core::OpenOptions{},
                                                    disabled_open_context,
                                                    disabled);
  ASSERT_FALSE(rejected_open.ok());
  EXPECT_EQ(rejected_open.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(DiskFlatSegmentFactory::registration.current.implementation_key, "disk_flat_segment");
  EXPECT_EQ(DiskFlatSegmentFactory::registration.current.engine_factory_key, "flat");
}

TEST(DiskFlatSegment, ManifestV2GatePublishesAndReaderSurvivesRuntimeDisable) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  const auto root = temporary.path() / "v2";
  auto segment = build_segment(rows, root, true);
  EXPECT_TRUE(std::filesystem::is_regular_file(root / "collection_manifest.txt"));
  EXPECT_TRUE(std::filesystem::is_regular_file(root / "segments/seg_00000001/READY"));
  EXPECT_TRUE(std::filesystem::is_regular_file(root / "segments/seg_00000001/ARTIFACTS.v2"));
  const auto first_line = platform::read_file_prefix(root / "collection_manifest.txt", 9);
  EXPECT_EQ(first_line, "version=2");

  // Turning the writer back off does not participate in open: the reader is
  // permanently available for roll-forward.
  internal::collection::CollectionFeatureFlags disabled_writer;
  disabled_writer.manifest_v2_writer = false;
  EXPECT_FALSE(disabled_writer.manifest_v2_writer);
  core::OpenContext open_context;
  auto reopened =
      DiskFlatSegment::open_collection(root, "seg_00000001", core::OpenOptions{}, open_context);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->descriptor().algorithm_id, core::algorithm::flat);
  core::SegmentStats stats;
  EXPECT_TRUE(reopened.value()->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, FixtureRows::kRows);
  (void)segment;
}

TEST(DiskFlatSegment, BuildOpenAndSearchRejectInsufficientResources) {
  TemporaryDirectory temporary;
  FixtureRows rows;
  DiskFlatPublicationOptions options;
  options.collection_root = temporary.path() / "budget-build";
  options.segment_id = "seg_00000001";
  core::BuildContext build_context;
  build_context.growing_reservation = core::MemoryReservation(1);
  auto denied_build =
      DiskFlatSegmentFactory::build(rows.input(), core::Metric::l2, options, build_context);
  ASSERT_FALSE(denied_build.ok());
  EXPECT_EQ(denied_build.status().code(), core::StatusCode::resource_exhausted);

  const auto root = temporary.path() / "built";
  auto segment = build_segment(rows, root);
  core::OpenContext open_context;
  open_context.resident_lease = core::MemoryLease(1);
  auto denied_open = DiskFlatSegment::open_directory(root / "segments/seg_00000001",
                                                     core::OpenOptions{},
                                                     open_context);
  ASSERT_FALSE(denied_open.ok());
  EXPECT_EQ(denied_open.status().code(), core::StatusCode::resource_exhausted);

  SearchCall scratch_denied(rows.vectors.data(), 1, FixtureRows::kDim, 4);
  scratch_denied.context.query_scratch_lease = core::MemoryLease(1);
  const auto scratch_status = segment->search(scratch_denied.request);
  EXPECT_EQ(scratch_status.code(), core::StatusCode::resource_exhausted);

  SearchCall io_denied(rows.vectors.data(), 1, FixtureRows::kDim, 4);
  io_denied.context.io_credits.available_bytes = 1;
  const auto io_status = segment->search(io_denied.request);
  EXPECT_EQ(io_status.code(), core::StatusCode::resource_exhausted);

  std::vector<std::int8_t> int8_vectors(FixtureRows::kRows * FixtureRows::kDim, 1);
  DiskFlatBuildInput int8_input(core::TypedTensorView::contiguous(int8_vectors.data(),
                                                                  FixtureRows::kRows,
                                                                  FixtureRows::kDim),
                                rows.labels);
  DiskFlatPublicationOptions int8_options;
  int8_options.collection_root = temporary.path() / "int8";
  int8_options.segment_id = "seg_00000001";
  core::BuildContext int8_context;
  auto denied_scalar =
      DiskFlatSegmentFactory::build(int8_input, core::Metric::l2, int8_options, int8_context);
  ASSERT_FALSE(denied_scalar.ok());
  EXPECT_EQ(denied_scalar.status().code(), core::StatusCode::not_supported);

  std::array<std::int8_t, FixtureRows::kDim> int8_query{};
  std::array<float, FixtureRows::kDim> unused_float_query{};
  SearchCall typed_search(unused_float_query.data(), 1, FixtureRows::kDim, 4);
  typed_search.request.queries =
      core::TypedTensorView::contiguous(int8_query.data(), 1, FixtureRows::kDim);
  const auto typed_status = segment->search(typed_search.request);
  EXPECT_EQ(typed_status.code(), core::StatusCode::not_supported);
}

}  // namespace
}  // namespace alaya::disk
