// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/segmented_collection.hpp"
#include "index/collection/sha256.hpp"
#include "index/disk/disk_flat_segment.hpp"
#include "index/disk/laser_segment.hpp"
#include "core/platform.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

using internal::collection::CollectionSearchRequest;
using internal::collection::SegmentedCollection;
using internal::collection::SegmentRegistration;
using internal::collection::SegmentRole;
using internal::collection::VersionState;

constexpr std::uint32_t kDim = 128;
constexpr std::uint64_t kCount = 2048;
constexpr std::uint32_t kR = 64;
constexpr std::uint64_t kTopK = 10;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-laser-segment-test-" + std::to_string(platform::get_pid()) + "-" +
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

[[nodiscard]] auto fixture_directory() -> std::filesystem::path {
  return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
}

[[nodiscard]] auto fixture_prefix() -> std::string {
  return std::string(ALAYA_LASER_FIXTURE_PREFIX);
}

[[nodiscard]] auto fixture_index_name() -> std::string {
  return fixture_prefix() + "_R" + std::to_string(kR) + "_MD" + std::to_string(kDim) + ".index";
}

[[nodiscard]] auto fixture_available() -> bool {
  if (!engine_supported_v1(DiskIndexType::Laser) || fixture_directory().empty()) {
    return false;
  }
  const auto index = fixture_index_name();
  const std::array required{fixture_prefix() + "_input.fbin",
                            index,
                            index + "_rotator",
                            index + "_cache_ids",
                            index + "_cache_nodes"};
  return std::ranges::all_of(required, [](const auto &name) {
    std::error_code error;
    const auto path = fixture_directory() / name;
    return std::filesystem::is_regular_file(path, error) && !error &&
           std::filesystem::file_size(path, error) > 0 && !error;
  });
}

[[nodiscard]] auto fixture_vectors() -> std::vector<float> {
  const auto path = fixture_directory() / (fixture_prefix() + "_input.fbin");
  std::ifstream input(path, std::ios::binary);
  std::int32_t count{};
  std::int32_t dim{};
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!input || count != static_cast<std::int32_t>(kCount) ||
      dim != static_cast<std::int32_t>(kDim)) {
    throw std::runtime_error("LASER fixture vector header is invalid");
  }
  std::vector<float> vectors(static_cast<std::size_t>(count) * static_cast<std::size_t>(dim));
  input.read(reinterpret_cast<char *>(vectors.data()),
             static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("LASER fixture vector payload is truncated");
  }
  return vectors;
}

[[nodiscard]] auto fixture_labels() -> std::vector<std::uint64_t> {
  std::vector<std::uint64_t> labels(kCount);
  for (std::uint64_t row = 0; row < kCount; ++row) {
    labels[row] = 50'000 + row * 13;
  }
  return labels;
}

void import_fixture(const std::filesystem::path &segment_directory,
                    const std::vector<std::uint64_t> &labels) {
  LaserSegmentImporter importer(kDim, core::Metric::l2, {});
  (void)importer.import_from(fixture_directory(), labels.data(), labels.size(), segment_directory);
}

struct SearchCall {
  SearchCall(const float *queries,
             core::RowCount rows,
             std::uint64_t top_k,
             std::uint32_t effort,
             std::uint32_t beam_width = 4)
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
    extension_options.effort = effort;
    extension_options.beam_width = beam_width;
    extension = LaserSegment::make_search_extension(extension_options);
    request.queries = core::TypedTensorView::contiguous(queries, rows, kDim);
    request.options.top_k = top_k;
    request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = std::addressof(context);
    request.response = std::addressof(response);
  }

  core::SearchContext context{};
  LaserSegmentSearchExtension extension_options{};
  core::AlgorithmSearchExtension extension{};
  std::vector<core::SearchHit> hits{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  core::SearchResponse response{};
  core::SearchRequest request{};
};

[[nodiscard]] auto native_digests(const std::filesystem::path &segment_directory)
    -> std::map<std::string, internal::collection::Sha256Digest> {
  std::map<std::string, internal::collection::Sha256Digest> result;
  for (const auto &entry : std::filesystem::directory_iterator(segment_directory)) {
    if (entry.is_regular_file() && entry.path().filename() != "READY" &&
        entry.path().filename() != "ARTIFACTS.v2") {
      result.emplace(entry.path().filename().string(),
                     internal::collection::sha256_file(entry.path()));
    }
  }
  return result;
}

[[nodiscard]] auto median(std::vector<double> values) -> double {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

TEST(LaserSegment, DifferentialRankOnlyManifestGateCollectionRejectionAndPerformance) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER fixture is unavailable under " << fixture_directory();
  }
  static_assert(!core::Saveable<LaserSegment>);
  static_assert(core::BatchSearchable<LaserSegment>);
  static_assert(core::StatsProvider<LaserSegment>);

  TemporaryDirectory temporary;
  const auto root = temporary.path() / "collection";
  const auto segment_directory = root / "segments/seg_00000001";
  std::filesystem::create_directories(segment_directory.parent_path());
  const auto labels = fixture_labels();
  const auto vectors = fixture_vectors();
  import_fixture(segment_directory, labels);

  CollectionManifest legacy_collection;
  legacy_collection.dim = kDim;
  legacy_collection.metric = core::Metric::l2;
  legacy_collection.index_type = DiskIndexType::Laser;
  legacy_collection.next_segment_id = 2;
  legacy_collection.segment_ids = {"seg_00000001"};
  legacy_collection.save(root / "collection_manifest.txt");

  core::OpenContext open_context;
  auto differential = LaserSegmentLegacyFactory::open_differential(segment_directory, open_context);
  ASSERT_TRUE(differential.ok()) << differential.status().diagnostic();
  auto direct = differential.value().direct;
  auto segment = std::move(differential).value().segment;

  const auto descriptor = segment->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, core::algorithm::laser);
  EXPECT_EQ(descriptor.format_version, 1U);
  EXPECT_EQ(descriptor.factory_version, 1U);
  EXPECT_EQ(descriptor.metric, core::Metric::l2);
  EXPECT_EQ(descriptor.medium, core::Medium::disk);
  EXPECT_EQ(descriptor.stored_scalar_type, core::ScalarType::float32);
  EXPECT_EQ(LaserSegment::kFormatName, "disk_laser_qg");

  for (const auto effort : {32U, 64U, 128U}) {
    SearchCall call(vectors.data() + 31 * kDim, 1, kTopK, effort);
    std::vector<std::vector<DiskSearchHit>> raw;
    ASSERT_TRUE(LaserSegmentLegacyFactory::search_differential(*segment, call.request, raw).ok());
    ASSERT_EQ(raw.size(), 1U);
    const auto &expected = raw.front();
    ASSERT_EQ(call.counts[0], expected.size());
    EXPECT_EQ(call.response.score_kind, core::ScoreKind::rank_only);
    EXPECT_EQ(call.response.result_flags, core::ResultFlag::approximate);
    for (std::size_t index = 0; index < expected.size(); ++index) {
      EXPECT_EQ(static_cast<std::uint64_t>(call.hits[index].row_id), expected[index].label);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(call.hits[index].score),
                std::bit_cast<std::uint32_t>(expected[index].distance));
      EXPECT_EQ(call.hits[index].score_kind, core::ScoreKind::rank_only);
    }
  }

  SearchCall batch(vectors.data() + 7 * kDim, 2, kTopK, 64);
  std::vector<std::vector<DiskSearchHit>> raw_batch;
  ASSERT_TRUE(
      LaserSegmentLegacyFactory::search_differential(*segment, batch.request, raw_batch).ok());
  ASSERT_EQ(raw_batch.size(), 2U);
  for (std::uint64_t row = 0; row < 2; ++row) {
    const auto &expected = raw_batch[row];
    ASSERT_EQ(batch.counts[row], expected.size());
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const auto &actual = batch.hits[batch.offsets[row] + index];
      EXPECT_EQ(static_cast<std::uint64_t>(actual.row_id), expected[index].label);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.score),
                std::bit_cast<std::uint32_t>(expected[index].distance));
    }
  }

  SearchCall scratch_denied(vectors.data(), 1, kTopK, 64);
  scratch_denied.context.query_scratch_lease = core::MemoryLease(1);
  EXPECT_EQ(segment->search(scratch_denied.request).code(), core::StatusCode::resource_exhausted);
  SearchCall io_denied(vectors.data(), 1, kTopK, 64);
  io_denied.context.io_credits.available_bytes = 1;
  EXPECT_EQ(segment->search(io_denied.request).code(), core::StatusCode::resource_exhausted);
  core::OpenContext resident_denied;
  resident_denied.resident_lease = core::MemoryLease(1);
  auto denied_open =
      LaserSegment::open_directory(segment_directory, core::OpenOptions{}, resident_denied);
  ASSERT_FALSE(denied_open.ok());
  EXPECT_EQ(denied_open.status().code(), core::StatusCode::resource_exhausted);

  internal::disk::DiskEngineFeatureFlags disabled;
  disabled.disk_laser_segment = false;
  core::OpenContext disabled_context;
  auto disabled_open = LaserSegmentFactory::open(core::ArtifactView{},
                                                 core::OpenOptions{},
                                                 disabled_context,
                                                 disabled);
  ASSERT_FALSE(disabled_open.ok());
  EXPECT_EQ(disabled_open.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(LaserSegmentFactory::registration.current.implementation_key, "disk_laser_segment");
  EXPECT_EQ(LaserSegmentFactory::registration.current.engine_factory_key, "laser");
  EXPECT_EQ(LaserSegmentFactory::registration.legacy.implementation_key, "disk_laser_legacy");
  EXPECT_EQ(LaserSegmentFactory::registration.legacy.engine_factory_key, "disk_laser");

  const auto before_publication = native_digests(segment_directory);
  LaserSegmentReferenceOptions publication;
  publication.collection_root = root;
  publication.segment_id = "seg_00000001";
  core::BuildContext publication_context;
  EXPECT_TRUE(segment->publish_reference(publication, publication_context).ok());
  EXPECT_FALSE(std::filesystem::exists(segment_directory / "READY"));
  EXPECT_FALSE(std::filesystem::exists(segment_directory / "ARTIFACTS.v2"));
  EXPECT_EQ(platform::read_file_prefix(root / "collection_manifest.txt", 9), "version=1");
  EXPECT_EQ(native_digests(segment_directory), before_publication);

  publication.collection_features.manifest_v2_writer = true;
  publication.fail_point =
      internal::collection::ArtifactTransactionFailPoint::after_payload_publish_before_manifest;
  const auto interrupted = segment->publish_reference(publication, publication_context);
  ASSERT_FALSE(interrupted.ok());
  EXPECT_FALSE(std::filesystem::exists(segment_directory / "READY"));
  EXPECT_FALSE(std::filesystem::exists(segment_directory / "ARTIFACTS.v2"));
  EXPECT_EQ(platform::read_file_prefix(root / "collection_manifest.txt", 9), "version=1");
  EXPECT_EQ(native_digests(segment_directory), before_publication);

  publication.fail_point = internal::collection::ArtifactTransactionFailPoint::none;
  ASSERT_TRUE(segment->publish_reference(publication, publication_context).ok());
  EXPECT_TRUE(std::filesystem::is_regular_file(segment_directory / "READY"));
  EXPECT_TRUE(std::filesystem::is_regular_file(segment_directory / "ARTIFACTS.v2"));
  EXPECT_EQ(native_digests(segment_directory), before_publication);
  EXPECT_EQ(platform::read_file_prefix(root / "collection_manifest.txt", 9), "version=2");
  auto manifest = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(manifest.ok()) << manifest.status().diagnostic();
  ASSERT_EQ(manifest.value().manifest.segments.size(), 1U);
  const auto &entry = manifest.value().manifest.segments.front();
  EXPECT_EQ(entry.algorithm_id, core::algorithm::laser);
  EXPECT_EQ(entry.format_version, 1U);
  EXPECT_EQ(entry.factory_key, "laser");
  EXPECT_EQ(entry.extensions.at("score_kind"), "rank_only");
  EXPECT_EQ(entry.extensions.at("numeric_score_comparable"), "false");
  EXPECT_TRUE((entry.capabilities.operations &
               core::capability_bit(core::OperationCapability::search)) != 0);
  EXPECT_TRUE((entry.capabilities.operations &
               core::capability_bit(core::OperationCapability::batch_search)) != 0);
  EXPECT_TRUE((entry.capabilities.operations &
               core::capability_bit(core::OperationCapability::stats)) != 0);
  EXPECT_EQ(entry.capabilities.operations & core::capability_bit(core::OperationCapability::save),
            0U);
  ASSERT_FALSE(entry.artifacts.empty());
  for (const auto &artifact : entry.artifacts) {
    EXPECT_TRUE(artifact.ready);
    EXPECT_EQ(artifact.checksum_algorithm, internal::collection::ChecksumAlgorithmV2::sha256);
  }
  direct.reset();
  core::OpenContext collection_open_context;
  auto collection_reopened = LaserSegment::open_collection(root,
                                                           "seg_00000001",
                                                           core::OpenOptions{},
                                                           collection_open_context);
  ASSERT_TRUE(collection_reopened.ok()) << collection_reopened.status().diagnostic();

  // A numeric segment plus a rank-only LASER segment with no exact-vector
  // source must take Collection's existing incomparable-domain rejection.
  SearchCall laser_candidates(vectors.data() + 31 * kDim, 1, kTopK, 64);
  ASSERT_TRUE(segment->search(laser_candidates.request).ok());

  // A LASER-only route is valid when Collection is given the exact-vector
  // seam required to normalize rank-only results. This is deliberately a
  // separate snapshot from the incomparable mixed request below.
  auto standalone_any = LaserSegment::into_any(std::move(collection_reopened).value());
  ASSERT_TRUE(standalone_any.ok()) << standalone_any.status().diagnostic();
  SegmentRegistration standalone_registration;
  standalone_registration.segment_id = 3;
  standalone_registration.role = SegmentRole::sealed;
  standalone_registration.segment = std::move(standalone_any).value();
  for (std::size_t index = 0; index < laser_candidates.counts[0]; ++index) {
    const auto label = static_cast<std::uint64_t>(laser_candidates.hits[index].row_id);
    standalone_registration.rows.push_back({core::LogicalId::from_legacy_uint64(label),
                                            core::SegmentRowId(label),
                                            0,
                                            VersionState::live,
                                            {}});
  }
  standalone_registration.exact_rerank = [&vectors](const core::TypedTensorView &query,
                                                     core::SegmentRowId row_id)
      -> core::Result<float> {
    const auto label = static_cast<std::uint64_t>(row_id);
    if (label < 50'000 || (label - 50'000) % 13 != 0 ||
        (label - 50'000) / 13 >= kCount) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::search,
                                 core::StatusDetail::malformed_struct,
                                 "LASER label is outside the fixture vector map");
    }
    const auto source_row = (label - 50'000) / 13;
    float distance{};
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto delta = query.row<float>(0)[column] -
                         vectors[static_cast<std::size_t>(source_row) * kDim + column];
      distance += delta * delta;
    }
    return distance;
  };
  auto standalone =
      SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                {std::move(standalone_registration)});
  ASSERT_TRUE(standalone.ok()) << standalone.status().diagnostic();
  core::SearchContext standalone_context;
  CollectionSearchRequest standalone_request;
  standalone_request.queries =
      core::TypedTensorView::contiguous(vectors.data() + 31 * kDim, 1, kDim);
  standalone_request.options.top_k = 5;
  auto standalone_extension =
      LaserSegment::make_search_extension(laser_candidates.extension_options);
  standalone_extension.unknown_policy = core::UnknownExtensionPolicy::ignore_safe;
  standalone_request.options.extensions =
      std::span<const core::AlgorithmSearchExtension>(&standalone_extension, 1);
  standalone_request.context = &standalone_context;
  auto standalone_result = std::move(standalone).value()->search(standalone_request);
  ASSERT_TRUE(standalone_result.ok()) << standalone_result.status().diagnostic();
  ASSERT_EQ(standalone_result.value().queries.size(), 1U);
  ASSERT_TRUE(standalone_result.value().queries[0].status.ok());
  ASSERT_EQ(standalone_result.value().queries[0].hits.size(), 5U);
  for (const auto &hit : standalone_result.value().queries[0].hits) {
    EXPECT_NE(static_cast<std::uint32_t>(hit.result_flags) &
                  static_cast<std::uint32_t>(core::ResultFlag::exact_reranked),
              0U);
  }

  std::vector<float> flat_vectors(2 * kDim);
  std::copy_n(vectors.data(), kDim, flat_vectors.data());
  std::copy_n(vectors.data() + kDim, kDim, flat_vectors.data() + kDim);
  const std::array<std::uint64_t, 2> flat_labels{900'001, 900'002};
  DiskFlatBuildInput flat_input(core::TypedTensorView::contiguous(flat_vectors.data(), 2, kDim),
                                flat_labels);
  DiskFlatPublicationOptions flat_options;
  flat_options.collection_root = temporary.path() / "flat";
  flat_options.segment_id = "seg_00000002";
  core::BuildContext flat_context;
  auto flat =
      DiskFlatSegmentFactory::build(flat_input, core::Metric::l2, flat_options, flat_context);
  ASSERT_TRUE(flat.ok()) << flat.status().diagnostic();
  auto flat_any = DiskFlatSegment::into_any(std::move(flat).value());
  ASSERT_TRUE(flat_any.ok()) << flat_any.status().diagnostic();
  auto laser_any = LaserSegment::into_any(std::move(segment));
  ASSERT_TRUE(laser_any.ok()) << laser_any.status().diagnostic();
  EXPECT_TRUE(laser_any.value().capabilities().supports(core::OperationCapability::search));
  EXPECT_TRUE(laser_any.value().capabilities().supports(core::OperationCapability::batch_search));
  EXPECT_TRUE(laser_any.value().capabilities().supports(core::OperationCapability::stats));
  EXPECT_FALSE(laser_any.value().capabilities().supports(core::OperationCapability::save));
  EXPECT_FALSE(laser_any.value().capabilities().supports(core::OperationCapability::mutation));
  core::ArtifactWriter writer;
  core::ArtifactManifest save_manifest;
  EXPECT_EQ(laser_any.value().save(writer, core::SaveOptions{}, save_manifest).code(),
            core::StatusCode::not_supported);

  SegmentRegistration numeric_registration;
  numeric_registration.segment_id = 2;
  numeric_registration.segment = std::move(flat_any).value();
  for (const auto label : flat_labels) {
    numeric_registration.rows.push_back({core::LogicalId::from_legacy_uint64(label),
                                         core::SegmentRowId(label),
                                         0,
                                         VersionState::live,
                                         {}});
  }
  SegmentRegistration laser_registration;
  laser_registration.segment_id = 1;
  laser_registration.role = SegmentRole::sealed;
  laser_registration.segment = std::move(laser_any).value();
  for (std::size_t index = 0; index < laser_candidates.counts[0]; ++index) {
    const auto label = static_cast<std::uint64_t>(laser_candidates.hits[index].row_id);
    laser_registration.rows.push_back({core::LogicalId::from_legacy_uint64(label),
                                       core::SegmentRowId(label),
                                       0,
                                       VersionState::live,
                                       {}});
  }
  auto routed =
      SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                {std::move(numeric_registration), std::move(laser_registration)});
  ASSERT_TRUE(routed.ok()) << routed.status().diagnostic();
  core::SearchContext routed_context;
  CollectionSearchRequest routed_request;
  routed_request.queries = core::TypedTensorView::contiguous(vectors.data() + 31 * kDim, 1, kDim);
  routed_request.options.top_k = 5;
  auto routed_extension = LaserSegment::make_search_extension(laser_candidates.extension_options);
  routed_extension.unknown_policy = core::UnknownExtensionPolicy::ignore_safe;
  routed_request.options.extensions =
      std::span<const core::AlgorithmSearchExtension>(&routed_extension, 1);
  routed_request.context = &routed_context;
  auto mixed = std::move(routed).value()->search(routed_request);
  ASSERT_FALSE(mixed.ok());
  EXPECT_EQ(mixed.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(mixed.status().detail(), core::StatusDetail::operation_slot_absent);

  // The timing loop is last so functional failures cannot be hidden by a
  // noisy host. It compares the same fixed artifact, query, EF and beam.
  core::OpenContext perf_open_context;
  auto perf = LaserSegmentLegacyFactory::open_differential(segment_directory, perf_open_context);
  ASSERT_TRUE(perf.ok()) << perf.status().diagnostic();
  auto perf_direct = perf.value().direct;
  auto perf_segment = std::move(perf).value().segment;
  DiskSearchOptions perf_options;
  perf_options.top_k = kTopK;
  perf_options.ef = 64;
  perf_options.beam_width = 4;
  SearchCall perf_call(vectors.data() + 97 * kDim, 1, kTopK, 64);
  for (int warmup = 0; warmup < 3; ++warmup) {
    (void)perf_direct->search(vectors.data() + 97 * kDim, perf_options);
    ASSERT_TRUE(perf_segment->search(perf_call.request).ok());
  }
  std::vector<double> direct_us;
  std::vector<double> segment_us;
  constexpr int kPerformanceIterations = 240;
  direct_us.reserve(kPerformanceIterations);
  segment_us.reserve(kPerformanceIterations);
  std::uint64_t checksum{};
  for (int iteration = 0; iteration < kPerformanceIterations; ++iteration) {
    const auto time_direct = [&] {
      const auto begin = std::chrono::steady_clock::now();
      const auto hits = perf_direct->search(vectors.data() + 97 * kDim, perf_options);
      const auto end = std::chrono::steady_clock::now();
      checksum += hits.empty() ? 0 : hits.front().label;
      direct_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    };
    const auto time_segment = [&] {
      const auto begin = std::chrono::steady_clock::now();
      const auto status = perf_segment->search(perf_call.request);
      const auto end = std::chrono::steady_clock::now();
      EXPECT_TRUE(status.ok());
      checksum +=
          perf_call.counts[0] == 0 ? 0 : static_cast<std::uint64_t>(perf_call.hits[0].row_id);
      segment_us.push_back(std::chrono::duration<double, std::micro>(end - begin).count());
    };
    if ((iteration & 1) == 0) {
      time_direct();
      time_segment();
    } else {
      time_segment();
      time_direct();
    }
  }
  const auto direct_p50 = median(direct_us);
  const auto segment_p50 = median(segment_us);
  const auto overhead = (segment_p50 - direct_p50) / direct_p50;
  std::cout << "laser_segment_performance,direct_p50_us=" << direct_p50
            << ",segment_p50_us=" << segment_p50 << ",overhead_pct=" << overhead * 100.0
            << ",checksum=" << checksum << "\n";
  EXPECT_LE(overhead, 0.05) << "LaserSegment wrapper exceeds the 5% local sentinel";
}

}  // namespace
}  // namespace alaya::disk
