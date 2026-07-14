// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/manifest_dual_reader.hpp"
#include "index/disk/disk_flat_segment.hpp"
#include "index/disk/disk_vamana_segment.hpp"
#include "index/disk/laser_segment.hpp"
#include "platform/detect.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint64_t kLaserCount = 2048;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-gate6-rollback-drill-" + std::to_string(platform::get_pid()) + "-" +
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
  return fixture_prefix() + "_R64_MD128.index";
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
  if (!input || count != static_cast<std::int32_t>(kLaserCount) ||
      dim != static_cast<std::int32_t>(kDim)) {
    throw std::runtime_error("Gate 6 LASER fixture header is invalid");
  }
  std::vector<float> vectors(static_cast<std::size_t>(count) * kDim);
  input.read(reinterpret_cast<char *>(vectors.data()),
             static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("Gate 6 LASER fixture payload is truncated");
  }
  return vectors;
}

[[nodiscard]] auto sequential_ids(std::uint64_t count, std::uint64_t first)
    -> std::vector<std::uint64_t> {
  std::vector<std::uint64_t> ids(count);
  for (std::uint64_t row = 0; row < count; ++row) {
    ids[row] = first + row * 13;
  }
  return ids;
}

void import_laser(const std::filesystem::path &segment_directory,
                  const std::vector<std::uint64_t> &labels,
                  const std::filesystem::path &source_directory = fixture_directory()) {
  LaserSegmentImporter importer(kDim, core::Metric::l2, {});
  (void)importer.import_from(
      source_directory, labels.data(), labels.size(), segment_directory);
}

[[nodiscard]] auto fixture_for_segment(const std::filesystem::path &target,
                                       std::string_view segment_id)
    -> std::filesystem::path {
  std::filesystem::create_directories(target);
  const auto destination_prefix = "dsqg_" + std::string(segment_id);
  const std::array required_suffixes{"_R64_MD128.index",
                                     "_R64_MD128.index_rotator",
                                     "_R64_MD128.index_cache_ids",
                                     "_R64_MD128.index_cache_nodes"};
  for (const auto *suffix : required_suffixes) {
    std::filesystem::copy_file(fixture_directory() / (fixture_prefix() + suffix),
                               target / (destination_prefix + suffix));
  }
  const std::array optional_suffixes{"_medoids", "_medoids_indices", "_pca.bin"};
  for (const auto *suffix : optional_suffixes) {
    const auto source = fixture_directory() / (fixture_prefix() + suffix);
    std::error_code error;
    if (std::filesystem::is_regular_file(source, error) && !error) {
      std::filesystem::copy_file(source, target / (destination_prefix + suffix));
    }
  }
  return target;
}

[[nodiscard]] auto vamana_params() -> VamanaSegmentBuildParams {
  VamanaSegmentBuildParams params;
  params.R = 8;
  params.L = 32;
  params.alpha = 1.2F;
  params.num_threads = 1;
  params.seed = 424242;
  return params;
}

void expect_factory_visibility(const internal::disk::DiskEngineFeatureFlags &flags,
                               bool flat_enabled,
                               bool vamana_enabled,
                               bool laser_enabled) {
  DiskFlatPublicationOptions flat_options;
  core::BuildContext flat_context;
  auto flat = DiskFlatSegmentFactory::build(
      DiskFlatBuildInput{}, core::Metric::l2, flat_options, flat_context, flags);
  EXPECT_EQ(flat.status().code(),
            flat_enabled ? core::StatusCode::invalid_argument
                         : core::StatusCode::not_supported);

  DiskVamanaPublicationOptions vamana_options;
  core::BuildContext vamana_context;
  auto vamana = DiskVamanaSegmentFactory::build(DiskVamanaBuildInput{},
                                                core::Metric::l2,
                                                vamana_params(),
                                                vamana_options,
                                                vamana_context,
                                                flags);
  EXPECT_EQ(vamana.status().code(),
            vamana_enabled ? core::StatusCode::invalid_argument
                           : core::StatusCode::not_supported);

  core::OpenContext laser_context;
  auto laser = LaserSegmentFactory::open(
      core::ArtifactView{}, core::OpenOptions{}, laser_context, flags);
  EXPECT_EQ(laser.status().code(),
            laser_enabled ? core::StatusCode::invalid_argument
                          : core::StatusCode::not_supported);
}

TEST(Gate6RollbackDrill, RuntimeDisableAllAndEachEngineLeavesLegacyFactoriesIndependent) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER fixture is unavailable under " << fixture_directory();
  }
  internal::collection::CollectionFeatureFlags collection_flags;
  EXPECT_FALSE(collection_flags.manifest_v2_writer);

  internal::disk::DiskEngineFeatureFlags all_disabled;
  all_disabled.disk_flat_segment = false;
  all_disabled.disk_vamana_segment = false;
  all_disabled.disk_laser_segment = false;
  expect_factory_visibility(all_disabled, false, false, false);

  auto only_flat_disabled = internal::disk::DiskEngineFeatureFlags{};
  only_flat_disabled.disk_flat_segment = false;
  expect_factory_visibility(only_flat_disabled, false, true, true);
  auto only_vamana_disabled = internal::disk::DiskEngineFeatureFlags{};
  only_vamana_disabled.disk_vamana_segment = false;
  expect_factory_visibility(only_vamana_disabled, true, false, true);
  auto only_laser_disabled = internal::disk::DiskEngineFeatureFlags{};
  only_laser_disabled.disk_laser_segment = false;
  expect_factory_visibility(only_laser_disabled, true, true, false);

  TemporaryDirectory temporary;
  constexpr std::uint64_t kRows = 32;
  const auto vectors = fixture_vectors();
  const auto flat_ids = sequential_ids(kRows, 10'000);
  const auto vamana_ids = sequential_ids(kRows, 20'000);
  const auto laser_ids = sequential_ids(kLaserCount, 50'000);
  const auto flat_directory = temporary.path() / "legacy-flat/seg_00000001";
  const auto vamana_directory = temporary.path() / "legacy-vamana/seg_00000001";
  const auto laser_directory = temporary.path() / "legacy-laser/seg_00000001";
  std::filesystem::create_directories(flat_directory.parent_path());
  std::filesystem::create_directories(vamana_directory.parent_path());
  std::filesystem::create_directories(laser_directory.parent_path());
  auto legacy_flat = DiskFlatLegacyFactory::build(
      {core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), flat_ids},
      core::Metric::l2,
      flat_directory);
  ASSERT_TRUE(legacy_flat.ok()) << legacy_flat.status().diagnostic();
  auto legacy_vamana = DiskVamanaLegacyFactory::build(
      {core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), vamana_ids},
      core::Metric::l2,
      vamana_params(),
      vamana_directory);
  ASSERT_TRUE(legacy_vamana.ok()) << legacy_vamana.status().diagnostic();
  import_laser(laser_directory, laser_ids);
  auto legacy_laser = LaserSegmentLegacyFactory::open(laser_directory);
  ASSERT_TRUE(legacy_laser.ok()) << legacy_laser.status().diagnostic();
}

TEST(Gate6RollbackDrill, V2ThreeEngineRollForwardSurvivesWriterDisableAndRejectsCorruption) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER fixture is unavailable under " << fixture_directory();
  }
  TemporaryDirectory temporary;
  const auto root = temporary.path() / "collection";
  constexpr std::uint64_t kRows = 32;
  const auto vectors = fixture_vectors();
  const auto flat_ids = sequential_ids(kRows, 100'000);
  const auto vamana_ids = sequential_ids(kRows, 200'000);
  const auto laser_ids = sequential_ids(kLaserCount, 300'000);

  DiskFlatPublicationOptions flat_options;
  flat_options.collection_root = root;
  flat_options.segment_id = "seg_00000001";
  flat_options.manifest_generation = 1;
  flat_options.collection_features.manifest_v2_writer = true;
  core::BuildContext flat_context;
  auto flat = DiskFlatSegmentFactory::build(
      {core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), flat_ids},
      core::Metric::l2,
      flat_options,
      flat_context);
  ASSERT_TRUE(flat.ok()) << flat.status().diagnostic();
  auto after_flat = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(after_flat.ok()) << after_flat.status().diagnostic();
  ASSERT_EQ(after_flat.value().manifest.segments.size(), 1U);

  DiskVamanaPublicationOptions vamana_options;
  vamana_options.collection_root = root;
  vamana_options.segment_id = "seg_00000002";
  vamana_options.manifest_generation = 2;
  vamana_options.collection_features.manifest_v2_writer = true;
  vamana_options.base_manifest = after_flat.value().manifest;
  core::BuildContext vamana_context;
  auto vamana = DiskVamanaSegmentFactory::build(
      {core::TypedTensorView::contiguous(vectors.data() + kRows * kDim, kRows, kDim),
       vamana_ids},
      core::Metric::l2,
      vamana_params(),
      vamana_options,
      vamana_context);
  ASSERT_TRUE(vamana.ok()) << vamana.status().diagnostic();
  auto after_vamana = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(after_vamana.ok()) << after_vamana.status().diagnostic();
  ASSERT_EQ(after_vamana.value().manifest.segments.size(), 2U);

  const auto laser_directory = root / "segments/seg_00000003";
  const auto laser_source =
      fixture_for_segment(temporary.path() / "fixture-seg3", "seg_00000003");
  import_laser(laser_directory, laser_ids, laser_source);
  core::OpenContext laser_open_context;
  auto laser = LaserSegment::open_directory(
      laser_directory, core::OpenOptions{}, laser_open_context);
  ASSERT_TRUE(laser.ok()) << laser.status().diagnostic();
  LaserSegmentReferenceOptions laser_options;
  laser_options.collection_root = root;
  laser_options.segment_id = "seg_00000003";
  laser_options.manifest_generation = 3;
  laser_options.collection_features.manifest_v2_writer = true;
  laser_options.base_manifest = after_vamana.value().manifest;
  core::BuildContext laser_publish_context;
  auto published = laser.value()->publish_reference(laser_options, laser_publish_context);
  ASSERT_TRUE(published.ok()) << published.diagnostic();

  auto three_engine = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(three_engine.ok()) << three_engine.status().diagnostic();
  ASSERT_EQ(three_engine.value().manifest.segments.size(), 3U);
  std::set<core::AlgorithmId> algorithms;
  for (const auto &entry : three_engine.value().manifest.segments) {
    algorithms.insert(entry.algorithm_id);
  }
  EXPECT_EQ(algorithms,
            (std::set<core::AlgorithmId>{core::algorithm::flat,
                                         core::algorithm::vamana,
                                         core::algorithm::laser}));

  // The writer gate defaults off, while the dual reader has no corresponding
  // disable switch. Existing v2 artifacts therefore remain roll-forward-only.
  internal::collection::CollectionFeatureFlags writer_disabled;
  ASSERT_FALSE(writer_disabled.manifest_v2_writer);
  auto reader_with_writer_disabled =
      internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_TRUE(reader_with_writer_disabled.ok())
      << reader_with_writer_disabled.status().diagnostic();
  ASSERT_EQ(reader_with_writer_disabled.value().manifest.segments.size(), 3U);

  core::OpenContext flat_open_context;
  auto flat_reopened = DiskFlatSegment::open_collection(
      root, "seg_00000001", core::OpenOptions{}, flat_open_context);
  ASSERT_TRUE(flat_reopened.ok()) << flat_reopened.status().diagnostic();
  core::OpenContext vamana_open_context;
  auto vamana_reopened = DiskVamanaSegment::open_collection(
      root, "seg_00000002", core::OpenOptions{}, vamana_open_context);
  ASSERT_TRUE(vamana_reopened.ok()) << vamana_reopened.status().diagnostic();
  core::OpenContext laser_reopen_context;
  auto laser_reopened = LaserSegment::open_collection(
      root, "seg_00000003", core::OpenOptions{}, laser_reopen_context);
  ASSERT_TRUE(laser_reopened.ok()) << laser_reopened.status().diagnostic();

  const auto payload = root / "segments/seg_00000001/vectors.f32.bin";
  std::fstream stream(payload, std::ios::binary | std::ios::in | std::ios::out);
  ASSERT_TRUE(stream.is_open());
  char byte{};
  stream.read(&byte, 1);
  ASSERT_TRUE(stream.good());
  byte = static_cast<char>(byte ^ 0x5A);
  stream.seekp(0);
  stream.write(&byte, 1);
  stream.flush();
  stream.close();
  auto corrupted = internal::collection::CollectionManifestDualReader::open(root);
  ASSERT_FALSE(corrupted.ok());
  EXPECT_EQ(corrupted.status().code(), core::StatusCode::corruption);
  EXPECT_EQ(corrupted.status().detail(), core::StatusDetail::malformed_struct);
  EXPECT_NE(corrupted.status().diagnostic().find("SHA-256 mismatch"), std::string::npos);
}

}  // namespace
}  // namespace alaya::disk
