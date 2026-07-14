// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/manifest_dual_reader.hpp"
#include "index/collection/types.hpp"
#include "index/disk/disk_flat_builder.hpp"
#include "core/platform_fs.hpp"

namespace alaya::internal::collection {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-manifest-v2-test-" + std::to_string(platform::get_pid()) + "-" +
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

[[nodiscard]] auto sample_segment() -> SegmentEntryV2 {
  SegmentEntryV2 segment;
  segment.segment_id = "seg_00000001";
  segment.generation = 7;
  segment.role = SegmentRoleV2::build_target;
  segment.algorithm_id = core::algorithm::flat;
  segment.format_version = 1;
  segment.factory_key = "flat";
  segment.capabilities.operations = core::capability_bit(core::OperationCapability::search) |
                                    core::capability_bit(core::OperationCapability::batch_search) |
                                    core::capability_bit(core::OperationCapability::save) |
                                    core::capability_bit(core::OperationCapability::export_rows) |
                                    core::capability_bit(core::OperationCapability::stats);
  segment.lifecycle = SegmentLifecycleV2::sealed;
  segment.wal_cut = 11;
  segment.row_versions = {3, 9};
  segment.id_map_checkpoint = "id-map/checkpoint-11";
  segment.ready = true;
  segment.ready_marker = "segments/seg_00000001/READY";
  segment.ready_digest = sha256("ready");
  segment.reader_compatibility.required_features = {"manifest_v2"};
  segment.source_retention = {"source-6"};
  segment.extensions.emplace("x_test", "owned");
  OwnedArtifactV2 artifact;
  artifact.logical_name = "vectors";
  artifact.relative_path = "segments/seg_00000001/vectors.f32.bin";
  artifact.required = true;
  artifact.size_bytes = 7;
  artifact.digest = sha256("payload");
  artifact.ready = true;
  artifact.reader_compatibility.required_features = {"manifest_v2"};
  segment.artifacts.push_back(std::move(artifact));
  return segment;
}

[[nodiscard]] auto sample_manifest() -> ArtifactManifestV2 {
  ArtifactManifestV2 manifest;
  manifest.collection.schema_name = "test/schema";
  manifest.collection.schema_version = 4;
  manifest.collection.dim = 3;
  manifest.collection.metric = core::Metric::inner_product;
  manifest.collection.scalar_type = core::ScalarType::float32;
  manifest.collection.logical_id_encoding = LogicalIdEncodingV2::canonical_kind_and_bytes;
  manifest.collection.logical_id_encoding_version = 2;
  manifest.collection.metadata_checkpoint = "metadata/checkpoint-11";
  manifest.collection.metadata_epoch = 11;
  manifest.publication.generation = 12;
  manifest.publication.parent = "generation-11";
  manifest.wal_cut = 42;
  manifest.row_versions = {3, 9};
  manifest.id_map_checkpoint = "id-map/checkpoint-11";
  manifest.next_segment_id_hint = 2;
  manifest.segments.push_back(sample_segment());
  manifest.gc.phase = GcPhaseV2::pending;
  manifest.gc.generation = 8;
  manifest.gc.pending_segment_ids = {"seg_00000000"};
  manifest.gc.retained_sources = {"source-6"};
  manifest.extensions.emplace("x_collection", "value");
  return manifest;
}

[[nodiscard]] auto transaction_segment() -> SegmentEntryV2 {
  auto segment = sample_segment();
  segment.generation = 1;
  segment.role = SegmentRoleV2::searchable;
  segment.wal_cut = 0;
  segment.row_versions = {};
  segment.id_map_checkpoint.clear();
  segment.artifacts.clear();
  segment.ready = false;
  segment.ready_marker.clear();
  segment.ready_digest = {};
  segment.source_retention.clear();
  segment.extensions.clear();
  return segment;
}

[[nodiscard]] auto transaction_manifest() -> ArtifactManifestV2 {
  ArtifactManifestV2 manifest;
  manifest.collection.dim = 2;
  manifest.publication.generation = 1;
  manifest.next_segment_id_hint = 2;
  return manifest;
}

TEST(Sha256, KnownVectorsAndOwnedDigest) {
  EXPECT_EQ(sha256("").hex(), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256("abc").hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const auto parsed = Sha256Digest::from_hex(sha256("abc").hex());
  EXPECT_EQ(parsed, sha256("abc"));
  EXPECT_THROW((void)Sha256Digest::from_hex("deadbeef"), std::invalid_argument);
}

TEST(ArtifactManifestV2, DeterministicRoundTripReconstructsEveryField) {
  const auto manifest = sample_manifest();
  const auto encoded = manifest.serialize();
  EXPECT_EQ(encoded, manifest.serialize());
  const auto decoded = ArtifactManifestV2::deserialize(encoded);
  EXPECT_EQ(decoded, manifest);
  EXPECT_TRUE(decoded.validate().ok());
}

TEST(CollectionManifestDualReader, MapsV1ToExplicitDefaultedEquivalentView) {
  TemporaryDirectory temporary;
  const auto segments = temporary.path() / "segments";
  std::filesystem::create_directories(segments);
  constexpr std::array<float, 4> vectors{0.0F, 1.0F, 2.0F, 3.0F};
  constexpr std::array<std::uint64_t, 2> labels{100, 101};
  disk::DiskFlatBuilder builder(2, core::Metric::l2);
  builder.add_batch(vectors.data(), labels.data(), labels.size());
  (void)builder.finish(segments / "seg_00000001");
  disk::CollectionManifest legacy;
  legacy.dim = 2;
  legacy.metric = core::Metric::l2;
  legacy.index_type = disk::DiskIndexType::Flat;
  legacy.next_segment_id = 2;
  legacy.segment_ids = {"seg_00000001"};
  legacy.x_extras.emplace("x_preserved", "yes");
  legacy.save(temporary.path() / kCollectionManifestFilename);

  auto opened = CollectionManifestDualReader::open(temporary.path());
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto &view = opened.value();
  EXPECT_EQ(view.source_version, ManifestSourceVersion::disk_collection_v1);
  EXPECT_EQ(view.manifest.collection.dim, 2);
  EXPECT_EQ(view.manifest.collection.logical_id_encoding, LogicalIdEncodingV2::legacy_u64_le);
  EXPECT_EQ(view.manifest.next_segment_id_hint, 2);
  ASSERT_EQ(view.manifest.segments.size(), 1);
  EXPECT_EQ(view.manifest.segments[0].algorithm_id, core::algorithm::flat);
  EXPECT_EQ(view.manifest.extensions.at("x_preserved"), "yes");
  EXPECT_TRUE(view.field_was_defaulted("collection.metadata_epoch"));
  EXPECT_TRUE(view.field_was_defaulted("segments[0].artifacts[0].sha256_ready"));
  EXPECT_EQ(view.manifest.segments[0].artifacts[0].checksum_algorithm, ChecksumAlgorithmV2::none);
}

TEST(ArtifactControlPlaneTransaction, DefaultGatePublishesNoV2ControlFiles) {
  EXPECT_FALSE(CollectionFeatureFlags{}.manifest_v2_writer);
  TemporaryDirectory temporary;
  core::BuildContext context;
  ArtifactTransactionOptions options;
  options.collection_root = temporary.path();
  options.target_relative_directory = "segments/seg_00000001";
  options.transaction_id = "default_off";
  options.manifest_v2_writer = false;
  auto begun = ArtifactControlPlaneTransaction::begin(options, context);
  ASSERT_TRUE(begun.ok()) << begun.status().diagnostic();
  auto transaction = std::move(begun).value();
  auto writer = transaction->writer({{"data", "data.bin", true, {}}});
  ASSERT_TRUE(writer.ok()) << writer.status().diagnostic();
  constexpr std::string_view payload{"legacy-bytes"};
  platform::write_all_fsync(std::filesystem::path(writer.value().find("data")),
                            payload.data(),
                            payload.size());
  auto prepared = transaction->prepare(transaction_segment());
  ASSERT_TRUE(prepared.ok()) << prepared.status().diagnostic();
  ASSERT_TRUE(transaction->publish(transaction_manifest()).ok());
  const auto final = temporary.path() / "segments/seg_00000001";
  EXPECT_TRUE(std::filesystem::is_regular_file(final / "data.bin"));
  EXPECT_FALSE(std::filesystem::exists(final / "ARTIFACTS.v2"));
  EXPECT_FALSE(std::filesystem::exists(final / "READY"));
  EXPECT_FALSE(std::filesystem::exists(temporary.path() / kCollectionManifestFilename));
}

TEST(ArtifactControlPlaneTransaction, RestartRemovesReadyPayloadOrphanedBeforeRoutingPublish) {
  TemporaryDirectory temporary;
  core::BuildContext context;
  ArtifactTransactionOptions options;
  options.collection_root = temporary.path();
  options.target_relative_directory = "segments/seg_00000001";
  options.transaction_id = "payload_orphan";
  options.manifest_v2_writer = true;
  options.abort_policy = ArtifactAbortPolicy::retain_for_restart_cleanup;
  options.fail_point = ArtifactTransactionFailPoint::after_payload_publish_before_manifest;
  auto begun = ArtifactControlPlaneTransaction::begin(options, context);
  ASSERT_TRUE(begun.ok()) << begun.status().diagnostic();
  auto transaction = std::move(begun).value();
  auto writer = transaction->writer({{"data", "data.bin", true, {}}});
  ASSERT_TRUE(writer.ok()) << writer.status().diagnostic();
  constexpr std::string_view payload{"ready-orphan"};
  platform::write_all_fsync(std::filesystem::path(writer.value().find("data")),
                            payload.data(),
                            payload.size());
  ASSERT_TRUE(transaction->prepare(transaction_segment()).ok());
  auto published = transaction->publish(transaction_manifest());
  EXPECT_FALSE(published.ok());
  EXPECT_TRUE(std::filesystem::exists(temporary.path() / "segments/seg_00000001/READY"));
  EXPECT_FALSE(std::filesystem::exists(temporary.path() / kCollectionManifestFilename));
  transaction.reset();
  auto cleanup = ArtifactControlPlaneTransaction::cleanup_orphans(temporary.path());
  EXPECT_TRUE(cleanup.ok()) << cleanup.diagnostic();
  EXPECT_FALSE(std::filesystem::exists(temporary.path() / "segments/seg_00000001"));
}

TEST(ArtifactControlPlaneTransaction, CrashCutsNeverEnterRoutingAndRestartCleansOrphans) {
  const std::array fail_points{ArtifactTransactionFailPoint::after_staging_write,
                               ArtifactTransactionFailPoint::before_ready,
                               ArtifactTransactionFailPoint::after_ready_before_publish};
  std::uint64_t serial{};
  for (const auto fail_point : fail_points) {
    SCOPED_TRACE(static_cast<unsigned>(fail_point));
    TemporaryDirectory temporary;
    core::BuildContext context;
    ArtifactTransactionOptions options;
    options.collection_root = temporary.path();
    options.target_relative_directory = "segments/seg_00000001";
    options.transaction_id = "crash_" + std::to_string(++serial);
    options.manifest_v2_writer = true;
    options.abort_policy = ArtifactAbortPolicy::retain_for_restart_cleanup;
    options.fail_point = fail_point;
    auto begun = ArtifactControlPlaneTransaction::begin(options, context);
    ASSERT_TRUE(begun.ok()) << begun.status().diagnostic();
    auto transaction = std::move(begun).value();
    auto writer = transaction->writer({{"data", "data.bin", true, {}}});
    ASSERT_TRUE(writer.ok()) << writer.status().diagnostic();
    constexpr std::string_view payload{"partial"};
    platform::write_all_fsync(std::filesystem::path(writer.value().find("data")),
                              payload.data(),
                              payload.size());
    auto prepared = transaction->prepare(transaction_segment());
    EXPECT_FALSE(prepared.ok());
    EXPECT_FALSE(std::filesystem::exists(temporary.path() / kCollectionManifestFilename));
    EXPECT_FALSE(std::filesystem::exists(temporary.path() / "segments/seg_00000001"));
    transaction.reset();
    EXPECT_TRUE(std::filesystem::exists(temporary.path() / ".alaya_staging"));
    constexpr std::string_view stale_manifest{"not-routable"};
    const auto stale_path = temporary.path() / ".collection_manifest.v2.stale";
    platform::write_all_fsync(stale_path, stale_manifest.data(), stale_manifest.size());
    auto cleanup = ArtifactControlPlaneTransaction::cleanup_orphans(temporary.path());
    EXPECT_TRUE(cleanup.ok()) << cleanup.diagnostic();
    EXPECT_FALSE(std::filesystem::exists(temporary.path() / ".alaya_staging"));
    EXPECT_FALSE(std::filesystem::exists(stale_path));
  }
}

TEST(ArtifactControlPlaneTransaction, V2RollForwardReaderVerifiesReadyAndRejectsDamage) {
  TemporaryDirectory temporary;
  core::BuildContext context;
  ArtifactTransactionOptions options;
  options.collection_root = temporary.path();
  options.target_relative_directory = "segments/seg_00000001";
  options.transaction_id = "publish_v2";
  options.manifest_v2_writer = true;
  auto begun = ArtifactControlPlaneTransaction::begin(options, context);
  ASSERT_TRUE(begun.ok()) << begun.status().diagnostic();
  auto transaction = std::move(begun).value();
  auto writer = transaction->writer({{"data", "data.bin", true, {}}});
  ASSERT_TRUE(writer.ok()) << writer.status().diagnostic();
  constexpr std::string_view payload{"durable-payload"};
  const auto data_path = std::filesystem::path(writer.value().find("data"));
  platform::write_all_fsync(data_path, payload.data(), payload.size());
  auto prepared = transaction->prepare(transaction_segment());
  ASSERT_TRUE(prepared.ok()) << prepared.status().diagnostic();
  ASSERT_TRUE(transaction->publish(transaction_manifest()).ok());

  // Reader availability is independent of the writer gate: this is the
  // runtime-disable/roll-forward contract.
  auto opened = CollectionManifestDualReader::open(temporary.path());
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  EXPECT_EQ(opened.value().source_version, ManifestSourceVersion::artifact_manifest_v2);
  EXPECT_TRUE(opened.value().explicit_defaults.empty());
  ASSERT_EQ(opened.value().manifest.segments.size(), 1);
  EXPECT_EQ(opened.value().manifest.segments[0], prepared.value());

  constexpr std::string_view damaged{"damaged-payload"};
  platform::write_all_fsync(temporary.path() / "segments/seg_00000001/data.bin",
                            damaged.data(),
                            damaged.size());
  auto rejected = CollectionManifestDualReader::open(temporary.path());
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::corruption);
}

}  // namespace
}  // namespace alaya::internal::collection
