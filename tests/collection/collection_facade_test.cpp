// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifndef _WIN32
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/logical_wal.hpp"
#include "index/collection/sha256.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace internal::collection {

class CollectionTestAccess {
 public:
  [[nodiscard]] static auto pin_epoch(const Collection &collection) -> RoutingSnapshotPtr {
    return collection.implementation_->pin_routing_snapshot();
  }
};

}  // namespace internal::collection

namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-canonical-collection-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
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

[[nodiscard]] auto options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions result;
  result.root = root;
  result.dim = 2;
  result.metric = core::Metric::l2;
  result.scalar_type = core::ScalarType::float32;
  result.target_algorithm = core::algorithm::hnsw;
  result.build_threads = 3;
  return result;
}

[[nodiscard]] auto item(std::string id,
                        const std::array<float, 2> &vector,
                        std::string document = {},
                        CollectionMetadata metadata = {}) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(id);
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  result.metadata = std::move(metadata);
  result.document = std::move(document);
  return result;
}

[[nodiscard]] auto id_string(const core::LogicalId &id) -> std::string {
  const auto bytes = id.canonical_bytes();
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

TEST(CollectionFacade, CanonicalResultsReceiptsStatsCheckpointAndReopen) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(collection->target_algorithm(), core::algorithm::hnsw);
  EXPECT_EQ(collection->active_algorithm(), core::algorithm::flat);
  EXPECT_EQ(collection->target_implementation_key(), "hnsw_segment");
  EXPECT_EQ(collection->options().build_threads, 3U);

  const std::array<float, 2> first{0.0F, 0.0F};
  const std::array<float, 2> second{1.0F, 0.0F};
  auto first_receipt = collection->add(item("a", first, "A", {{"kind", std::string("first")}}));
  ASSERT_TRUE(first_receipt.ok()) << first_receipt.status().diagnostic();
  EXPECT_TRUE(first_receipt.value().searchable);
  EXPECT_EQ(first_receipt.value().durability, CollectionDurabilityState::wal_fsync);
  EXPECT_EQ(first_receipt.value().row_status, CollectionRowMutationStatus::inserted);
  auto second_receipt = collection->add(item("b", second, "B"));
  ASSERT_TRUE(second_receipt.ok()) << second_receipt.status().diagnostic();

  const std::array<float, 2> query{0.0F, 0.0F};
  auto single = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(single.ok()) << single.status().diagnostic();
  ASSERT_EQ(single.value().ids.size(), 2U);
  EXPECT_EQ(single.value().distances.size(), single.value().ids.size());
  EXPECT_EQ(single.value().offsets, (std::vector<core::RowCount>{0, 2}));
  EXPECT_EQ(single.value().valid_counts, (std::vector<core::RowCount>{2}));
  EXPECT_EQ(id_string(single.value().ids[0]), "a");
  EXPECT_EQ(single.value().distances[0], 0.0F);
  EXPECT_EQ(single.value().distances[1], 1.0F);

  const std::array<float, 4> queries{0.0F, 0.0F, 1.0F, 0.0F};
  auto batch = collection->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 7);
  ASSERT_TRUE(batch.ok()) << batch.status().diagnostic();
  EXPECT_EQ(batch.value().offsets, (std::vector<core::RowCount>{0, 2, 4}));
  EXPECT_EQ(batch.value().valid_counts, (std::vector<core::RowCount>{2, 2}));
  EXPECT_EQ(batch.value().ids.size(), batch.value().distances.size());

  const auto stats = collection->stats();
  EXPECT_EQ(stats.size, 2U);
  EXPECT_EQ(stats.accepted_count, 2U);
  EXPECT_EQ(stats.pending_count, 0U);
  EXPECT_EQ(stats.pending_bytes, 0U);
  EXPECT_EQ(stats.searchable_vector_bytes, 16U);
  EXPECT_EQ(stats.accepted_vector_bytes, 16U);

  auto checkpoint = collection->checkpoint();
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
  EXPECT_EQ(checkpoint.value().wal_cut, stats.visibility_watermark);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_FALSE(reopened.value()->imported_legacy_layout());
  auto record = reopened.value()->get_by_id(core::LogicalId::from_utf8("a"));
  ASSERT_TRUE(record.ok()) << record.status().diagnostic();
  ASSERT_TRUE(record.value().vector.has_value());
  EXPECT_EQ(record.value().document, "A");
  EXPECT_EQ(std::get<std::string>(record.value().metadata.at("kind")), "first");
  auto after = reopened.value()->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().offsets, single.value().offsets);
  EXPECT_EQ(after.value().distances, single.value().distances);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionFacade, BatchModesHaveStableStatusesAndNeverSilentlyDowngrade) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 2> zero{0.0F, 0.0F};
  ASSERT_TRUE(collection->add(item("existing", zero)).ok());

  std::vector<CollectionBatchRow> independent;
  independent.push_back({CollectionMutationAction::add,
                         core::LogicalId::from_utf8("existing"),
                         core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  independent.push_back({CollectionMutationAction::add,
                         core::LogicalId::from_utf8("new"),
                         core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  auto partial = collection->mutate_batch(independent);
  ASSERT_TRUE(partial.ok()) << partial.status().diagnostic();
  ASSERT_EQ(partial.value().rows.size(), 2U);
  EXPECT_EQ(partial.value().rows[0].row_status, CollectionRowMutationStatus::already_exists);
  EXPECT_FALSE(partial.value().rows[0].searchable);
  EXPECT_EQ(partial.value().rows[1].row_status, CollectionRowMutationStatus::inserted);
  EXPECT_TRUE(partial.value().rows[1].searchable);
  EXPECT_EQ(collection->size(), 2U);

  std::vector<CollectionBatchRow> duplicate;
  duplicate.push_back({CollectionMutationAction::upsert,
                       core::LogicalId::from_utf8("same"),
                       core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  duplicate.push_back({CollectionMutationAction::upsert,
                       core::LogicalId::from_utf8("same"),
                       core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  auto atomic = collection->mutate_batch(duplicate, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(atomic.ok()) << atomic.status().diagnostic();
  ASSERT_EQ(atomic.value().rows.size(), 2U);
  EXPECT_EQ(atomic.value().rows[0].row_status, CollectionRowMutationStatus::aborted);
  EXPECT_EQ(atomic.value().rows[1].row_status, CollectionRowMutationStatus::conflict);
  EXPECT_EQ(collection->size(), 2U);
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionFacade, RabitqRequiresExplicitQg) {
  TemporaryDirectory first;
  auto invalid = options(first.path());
  invalid.quantization = CollectionQuantization::rabitq;
  auto rejected = Collection::create(invalid);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::invalid_argument);
  EXPECT_NE(rejected.status().diagnostic().find("explicit index_type=qg"), std::string::npos);

  TemporaryDirectory second;
  auto valid = options(second.path());
  valid.quantization = CollectionQuantization::rabitq;
  valid.target_algorithm = core::algorithm::qg;
  auto created = Collection::create(valid);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  EXPECT_EQ(created.value()->target_algorithm(), core::algorithm::qg);
  EXPECT_EQ(created.value()->target_engine_factory_key(), "qg");
  ASSERT_TRUE(created.value()->close().ok());
}

TEST(CollectionFacade, CxxDirectParitySequenceUsesTheCanonicalWireShape) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 2> zero{0.0F, 0.0F};
  const std::array<float, 2> two{2.0F, 0.0F};
  const std::array<float, 2> one{1.0F, 0.0F};
  const std::array<float, 2> vertical{0.0F, 1.0F};
  ASSERT_TRUE(collection->add(item("a", zero, "A", {{"revision", std::int64_t{1}}})).ok());
  ASSERT_TRUE(collection->add(item("b", two, "B")).ok());
  ASSERT_TRUE(collection->upsert(item("a", one, "A2", {{"revision", std::int64_t{2}}})).ok());
  ASSERT_TRUE(collection->remove(core::LogicalId::from_utf8("b")).ok());
  ASSERT_TRUE(collection->add(item("c", vertical, "C")).ok());

  const std::array<float, 4> queries{1.0F, 0.0F, 0.0F, 1.0F};
  auto response =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 9);
  ASSERT_TRUE(response.ok()) << response.status().diagnostic();
  EXPECT_EQ(response.value().offsets, (std::vector<core::RowCount>{0, 2, 4}));
  EXPECT_EQ(response.value().valid_counts, (std::vector<core::RowCount>{2, 2}));
  ASSERT_EQ(response.value().ids.size(), 4U);
  EXPECT_EQ(id_string(response.value().ids[0]), "a");
  EXPECT_EQ(id_string(response.value().ids[1]), "c");
  EXPECT_EQ(id_string(response.value().ids[2]), "c");
  EXPECT_EQ(id_string(response.value().ids[3]), "a");
  EXPECT_EQ(response.value().distances, (std::vector<float>{0.0F, 2.0F, 0.0F, 2.0F}));
  EXPECT_EQ(collection->size(), 2U);
  ASSERT_TRUE(collection->checkpoint().ok());
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto replayed =
      reopened.value()->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 9);
  ASSERT_TRUE(replayed.ok()) << replayed.status().diagnostic();
  EXPECT_EQ(replayed.value().offsets, response.value().offsets);
  EXPECT_EQ(replayed.value().valid_counts, response.value().valid_counts);
  EXPECT_EQ(replayed.value().ids, response.value().ids);
  EXPECT_EQ(replayed.value().distances, response.value().distances);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionFacade, SealRotatesToSuccessorPublishesFlatAndReopens) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  std::vector<std::array<float, 2>> vectors{{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}};
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    ASSERT_TRUE(collection->add(item("seal-" + std::to_string(index), vectors[index])).ok());
  }

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().source_segment_id, 2U);
  EXPECT_EQ(sealed.value().successor_segment_id, 3U);
  EXPECT_EQ(sealed.value().sealed_segment_id, 4U);
  EXPECT_EQ(sealed.value().sealed_rows, 4U);
  EXPECT_GT(sealed.value().sealed_bytes, 0U);
  EXPECT_TRUE(std::filesystem::is_regular_file(temporary.path() / "collection_manifest.txt"));
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000004"));

  const auto manifest =
      internal::collection::ArtifactManifestV2::load(temporary.path() / "collection_manifest.txt");
  const auto target = std::ranges::find_if(manifest.segments, [](const auto &entry) {
    return entry.segment_id == "seg_00000004";
  });
  ASSERT_NE(target, manifest.segments.end());
  EXPECT_EQ(target->algorithm_id, core::algorithm::flat);
  EXPECT_EQ(target->lifecycle, internal::collection::SegmentLifecycleV2::sealed);
  EXPECT_EQ(target->wal_cut, sealed.value().wal_cut);
  EXPECT_EQ(manifest.gc.retained_sources, (std::vector<std::string>{"seg_00000004"}));

  const auto wal_scan = internal::collection::CollectionLogicalWal::scan_file(
      temporary.path() / ".alaya_internal" /
      std::string(internal::collection::kCollectionWalNamespace) /
      std::string(internal::collection::kCollectionWalFilename));
  ASSERT_TRUE(wal_scan.ok()) << wal_scan.status().diagnostic();
  ASSERT_EQ(wal_scan.value().frames.size(), 1U);
  EXPECT_EQ(wal_scan.value().frames[0].type,
            internal::collection::LogicalWalRecordType::checkpoint);
  EXPECT_EQ(wal_scan.value().frames[0].op_id, sealed.value().wal_cut);

  const std::array<float, 2> successor_vector{10.0F, 0.0F};
  ASSERT_TRUE(collection->add(item("successor", successor_vector)).ok());
  const std::array<float, 2> query{};
  auto live = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(live.ok()) << live.status().diagnostic();
  EXPECT_EQ(live.value().ids.size(), 5U);
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto after = reopened.value()->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().ids, live.value().ids);
  EXPECT_EQ(after.value().distances, live.value().distances);
  EXPECT_EQ(reopened.value()->stats().sealed_segments_count, 1U);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionFacade, FlatCompactPreservesRowsAndGcDeletesOnlyReleasedSources) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<std::array<float, 2>, 4> vectors{
      {{0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}}};
  for (std::size_t index = 0; index < 2; ++index) {
    ASSERT_TRUE(collection->add(item("compact-" + std::to_string(index), vectors[index])).ok());
  }
  ASSERT_TRUE(collection->seal().ok());
  ASSERT_TRUE(collection->gc().ok());  // Reclaims only the retired in-memory source marker.
  for (std::size_t index = 2; index < 4; ++index) {
    ASSERT_TRUE(collection->add(item("compact-" + std::to_string(index), vectors[index])).ok());
  }
  ASSERT_TRUE(collection->seal().ok());
  ASSERT_TRUE(collection->gc().ok());

  const std::array<float, 2> query{};
  auto before = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  std::map<std::string, std::vector<std::byte>> row_bytes;
  std::map<std::string, std::string> row_sha256;
  for (const auto &record : collection->records().value()) {
    ASSERT_TRUE(record.vector.has_value());
    row_bytes.emplace(id_string(record.logical_id),
                      std::vector<std::byte>(record.vector->bytes().begin(),
                                             record.vector->bytes().end()));
    row_sha256.emplace(id_string(record.logical_id),
                       internal::collection::sha256(record.vector->bytes()).hex());
  }

  auto held_epoch = internal::collection::CollectionTestAccess::pin_epoch(*collection);

  auto compacted = collection->compact();
  ASSERT_TRUE(compacted.ok()) << compacted.status().diagnostic();
  EXPECT_EQ(compacted.value().source_segment_ids, (std::vector<std::uint64_t>{4, 6}));
  EXPECT_EQ(compacted.value().compacted_segment_id, 7U);
  EXPECT_EQ(compacted.value().compacted_rows, 4U);
  EXPECT_GT(compacted.value().input_bytes, 0U);
  EXPECT_GT(compacted.value().output_bytes, 0U);
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000004"));
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000006"));
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000007"));
  const auto compact_manifest =
      internal::collection::ArtifactManifestV2::load(temporary.path() / "collection_manifest.txt");
  EXPECT_EQ(compact_manifest.gc.pending_segment_ids,
            (std::vector<std::string>{"seg_00000004", "seg_00000006"}));
  EXPECT_EQ(compact_manifest.gc.retained_sources, (std::vector<std::string>{"seg_00000007"}));

  auto after = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().ids, before.value().ids);
  EXPECT_EQ(after.value().distances, before.value().distances);
  for (const auto &record : collection->records().value()) {
    ASSERT_TRUE(record.vector.has_value());
    const auto found = row_bytes.find(id_string(record.logical_id));
    ASSERT_NE(found, row_bytes.end());
    EXPECT_EQ(std::vector<std::byte>(record.vector->bytes().begin(), record.vector->bytes().end()),
              found->second);
    EXPECT_EQ(internal::collection::sha256(record.vector->bytes()).hex(),
              row_sha256.at(id_string(record.logical_id)));
  }

  auto deferred = collection->gc();
  ASSERT_TRUE(deferred.ok()) << deferred.status().diagnostic();
  EXPECT_EQ(deferred.value().reclaimed, 0U);
  EXPECT_EQ(deferred.value().deferred, 2U);
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000004"));
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000006"));
  held_epoch.reset();

  auto reclaimed = collection->gc();
  ASSERT_TRUE(reclaimed.ok()) << reclaimed.status().diagnostic();
  EXPECT_EQ(reclaimed.value().reclaimed, 2U);
  EXPECT_GT(reclaimed.value().reclaimed_bytes, 0U);
  EXPECT_FALSE(std::filesystem::exists(temporary.path() / "segments" / "seg_00000004"));
  EXPECT_FALSE(std::filesystem::exists(temporary.path() / "segments" / "seg_00000006"));
  EXPECT_TRUE(std::filesystem::is_directory(temporary.path() / "segments" / "seg_00000007"));
  const auto stats = collection->stats();
  EXPECT_EQ(stats.sealed_segments_count, 1U);
  EXPECT_EQ(stats.gc_pending_count, 0U);
  EXPECT_EQ(stats.compacted_bytes, compacted.value().input_bytes);
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionFacade, AutoSealRotatesAtConfiguredRowThreshold) {
  TemporaryDirectory temporary;
  auto configured = options(temporary.path());
  configured.auto_seal_rows = 2;
  auto created = Collection::create(configured);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 2> first{0.0F, 0.0F};
  const std::array<float, 2> second{1.0F, 0.0F};
  ASSERT_TRUE(collection->add(item("auto-a", first)).ok());
  EXPECT_EQ(collection->stats().sealed_segments_count, 0U);
  ASSERT_TRUE(collection->add(item("auto-b", second)).ok());
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
  EXPECT_EQ(collection->options().auto_seal_rows, 2U);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->options().auto_seal_rows, 2U);
  EXPECT_EQ(reopened.value()->size(), 2U);
  ASSERT_TRUE(reopened.value()->close().ok());
}

#ifndef _WIN32
TEST(CollectionFacade, SealFourPointSigkillRecoveryRollsBackOrForward) {
  const std::array cases{CollectionSealFailPoint::after_cut_before_successor,
                         CollectionSealFailPoint::after_successor_switch,
                         CollectionSealFailPoint::during_export_build,
                         CollectionSealFailPoint::after_manifest_publish};
  auto battery_root = std::filesystem::path("/home/huangliang/md1/tmp") /
                      ("alaya-g10-seal-crash-" + std::to_string(platform::get_pid()));
  std::filesystem::remove_all(battery_root);
  std::filesystem::create_directories(battery_root);

  for (const auto failpoint : cases) {
    SCOPED_TRACE(static_cast<unsigned>(failpoint));
    const auto root = battery_root / std::to_string(static_cast<unsigned>(failpoint));
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      auto created = Collection::create(options(root));
      if (!created.ok()) {
        ::_exit(80);
      }
      const std::array<float, 2> first{0.0F, 0.0F};
      const std::array<float, 2> second{1.0F, 0.0F};
      const std::array<float, 2> third{2.0F, 0.0F};
      if (!created.value()->add(item("crash-a", first)).ok() ||
          !created.value()->add(item("crash-b", second)).ok() ||
          !created.value()->add(item("crash-c", third)).ok()) {
        ::_exit(81);
      }
      CollectionSealOptions seal_options;
      seal_options.fail_point = failpoint;
      seal_options.failpoint_hook = [failpoint](CollectionSealFailPoint observed) {
        if (observed == failpoint) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      (void)created.value()->seal(std::move(seal_options));
      ::_exit(82);
    }
    int child_status{};
    ASSERT_EQ(::waitpid(child, &child_status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(child_status));
    EXPECT_EQ(WTERMSIG(child_status), SIGKILL);
    if (failpoint == CollectionSealFailPoint::during_export_build) {
      const auto staging =
          root / internal::collection::ArtifactControlPlaneTransaction::kStagingDirectory;
      ASSERT_TRUE(std::filesystem::is_directory(staging));
      EXPECT_NE(std::filesystem::directory_iterator(staging),
                std::filesystem::directory_iterator{});
    }

    auto recovered = Collection::open(root);
    ASSERT_TRUE(recovered.ok()) << recovered.status().diagnostic();
    if (failpoint == CollectionSealFailPoint::during_export_build) {
      EXPECT_FALSE(std::filesystem::exists(
          root / internal::collection::ArtifactControlPlaneTransaction::kStagingDirectory));
    }
    auto collection = std::move(recovered).value();
    EXPECT_EQ(collection->size(), 3U);
    const std::array<float, 2> successor{3.0F, 0.0F};
    ASSERT_TRUE(collection->add(item("after-crash", successor)).ok());
    if (collection->stats().sealed_segments_count == 0) {
      auto resumed = collection->seal();
      ASSERT_TRUE(resumed.ok()) << resumed.status().diagnostic();
    }
    const std::array<float, 2> query{};
    auto result = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
    ASSERT_TRUE(result.ok()) << result.status().diagnostic();
    EXPECT_EQ(result.value().ids.size(), 4U);
    EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
    ASSERT_TRUE(collection->checkpoint().ok());
    ASSERT_TRUE(collection->close().ok());
    collection.reset();

    auto repeated = Collection::open(root);
    ASSERT_TRUE(repeated.ok()) << repeated.status().diagnostic();
    EXPECT_EQ(repeated.value()->size(), 4U);
    EXPECT_EQ(repeated.value()->stats().sealed_segments_count, 1U);
    ASSERT_TRUE(repeated.value()->close().ok());
  }
  std::filesystem::remove_all(battery_root);
}
#endif

}  // namespace
}  // namespace alaya
