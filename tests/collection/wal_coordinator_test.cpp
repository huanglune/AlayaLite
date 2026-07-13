// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#ifndef _WIN32
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include "fake_mutable_segment.hpp"

namespace alaya::internal::collection {
namespace {

using test::FakeMutableSegment;

class WalCoordinatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alaya-wal-coordinator-" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::filesystem::remove_all(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_{};
};

[[nodiscard]] auto open_collection(const std::filesystem::path &root,
                                   std::shared_ptr<FakeMutableSegment> &producer,
                                   bool atomic_bundle = true,
                                   MutationFailPoint fail_point = MutationFailPoint::none,
                                   std::function<void(MutationFailPoint)> hook = {})
    -> core::Result<std::shared_ptr<SegmentedCollection>> {
  producer = std::make_shared<FakeMutableSegment>();
  auto erased = test::make_fake_mutable_any(producer);
  if (!erased.ok()) {
    return erased.status();
  }
  SegmentRegistration registration;
  registration.segment_id = FakeMutableSegment::kSegmentId;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.atomic_mutation_bundle = atomic_bundle;
  CollectionConfig config;
  config.features.wal_coordinator = true;
  config.wal.root = root;
  config.fail_point = fail_point;
  config.failpoint_hook = std::move(hook);
  return SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                   {std::move(registration)},
                                   std::move(config));
}

[[nodiscard]] auto write_request(std::string id,
                                 const std::array<float, 2> &vector,
                                 std::string retry_token = {},
                                 WriteDurability durability = WriteDurability::wal_fsync,
                                 WriteMode mode = WriteMode::upsert) -> WriteRequest {
  WriteRequest request;
  request.logical_id = core::LogicalId::from_utf8(id);
  request.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  request.mode = mode;
  request.options.durability = durability;
  request.options.retry_token = std::move(retry_token);
  return request;
}

[[nodiscard]] auto get(const std::shared_ptr<SegmentedCollection> &collection, std::string id)
    -> core::Result<CollectionRecord> {
  return collection->get_by_id(core::LogicalId::from_utf8(id));
}

TEST_F(WalCoordinatorTest, DurableStringUpsertDeleteRecoveryAndRetryAreIdempotent) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();
  core::MutationContext context;
  const std::array<float, 2> first{1.0F, 2.0F};
  auto inserted = collection->write(write_request("durable", first, "insert-token"), context);
  ASSERT_TRUE(inserted.ok()) << inserted.status().diagnostic();
  EXPECT_EQ(inserted.value().durability, DurabilityState::wal_fsync);
  EXPECT_TRUE(inserted.value().searchable);
  EXPECT_EQ(inserted.value().durable_watermark, inserted.value().op_id);
  const auto prepared_before_retry = producer->prepared_op_ids().size();
  auto retried = collection->write(write_request("durable", first, "insert-token"), context);
  ASSERT_TRUE(retried.ok());
  EXPECT_EQ(retried.value().op_id, inserted.value().op_id);
  EXPECT_EQ(producer->prepared_op_ids().size(), prepared_before_retry);

  collection.reset();
  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  collection = std::move(opened).value();
  auto recovered = get(collection, "durable");
  ASSERT_TRUE(recovered.ok());
  EXPECT_EQ(recovered.value().upsert_sequence, inserted.value().op_id);
  retried = collection->write(write_request("durable", first, "insert-token"), context);
  ASSERT_TRUE(retried.ok());
  EXPECT_EQ(retried.value().op_id, inserted.value().op_id);

  const std::array<float, 2> second{9.0F, 8.0F};
  auto updated = collection->write(write_request("durable", second, "update-token"), context);
  ASSERT_TRUE(updated.ok());
  EXPECT_EQ(updated.value().row_status, RowMutationStatus::updated);
  WriteOptions erase_options;
  erase_options.retry_token = "delete-token";
  auto erased = collection->erase(core::LogicalId::from_utf8("durable"), context, erase_options);
  ASSERT_TRUE(erased.ok());
  EXPECT_EQ(erased.value().row_status, RowMutationStatus::deleted);
  collection.reset();

  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  collection = std::move(opened).value();
  EXPECT_EQ(get(collection, "durable").status().code(), core::StatusCode::not_found);
  auto delete_retry =
      collection->erase(core::LogicalId::from_utf8("durable"), context, erase_options);
  ASSERT_TRUE(delete_retry.ok());
  EXPECT_EQ(delete_retry.value().op_id, erased.value().op_id);
}

TEST_F(WalCoordinatorTest, WeakSearchableReceiptExplicitlyDisclaimsCrashDurability) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  core::MutationContext context;
  const std::array<float, 2> vector{1.0F, 3.0F};
  auto receipt = opened.value()->write(write_request("weak",
                                                     vector,
                                                     "weak-token",
                                                     WriteDurability::searchable),
                                       context);
  ASSERT_TRUE(receipt.ok());
  EXPECT_TRUE(receipt.value().searchable);
  EXPECT_EQ(receipt.value().durability, DurabilityState::searchable_not_durable);
  EXPECT_EQ(receipt.value().durable_watermark, 0U);
  EXPECT_TRUE(get(opened.value(), "weak").ok());
}

TEST_F(WalCoordinatorTest, EveryRedoCrashBoundaryHasTheSpecifiedRecoveryOutcome) {
  struct Case {
    MutationFailPoint point;
    bool committed;
  };
  const std::array cases{Case{MutationFailPoint::before_prepare, false},
                         Case{MutationFailPoint::after_prepare, false},
                         Case{MutationFailPoint::after_stage, false},
                         Case{MutationFailPoint::after_commit, true},
                         Case{MutationFailPoint::after_publish, true}};
  const std::array<float, 2> vector{3.0F, 4.0F};
  for (const auto &test_case : cases) {
    SCOPED_TRACE(static_cast<unsigned>(test_case.point));
    const auto case_root = root_ / std::to_string(static_cast<unsigned>(test_case.point));
    std::shared_ptr<FakeMutableSegment> producer;
    auto opened = open_collection(case_root, producer, true, test_case.point);
    ASSERT_TRUE(opened.ok());
    auto collection = std::move(opened).value();
    core::MutationContext context;
    auto result = collection->write(write_request("crash", vector, "crash-token"), context);
    EXPECT_FALSE(result.ok());
    if (test_case.point == MutationFailPoint::after_publish) {
      auto retry = collection->write(write_request("crash", vector, "crash-token"), context);
      ASSERT_TRUE(retry.ok());
      EXPECT_EQ(retry.value().row_status, RowMutationStatus::inserted);
    }
    collection.reset();
    opened = open_collection(case_root, producer);
    ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
    collection = std::move(opened).value();
    EXPECT_EQ(get(collection, "crash").ok(), test_case.committed);
  }

  // Receipt-after is the normal acknowledged path: reopen at the exact
  // durable/searchable watermarks carried by the delivered receipt.
  const auto receipt_root = root_ / "after-receipt";
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(receipt_root, producer);
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  core::MutationContext context;
  auto receipt = collection->write(write_request("receipt", vector, "receipt-token"), context);
  ASSERT_TRUE(receipt.ok());
  collection.reset();
  opened = open_collection(receipt_root, producer);
  ASSERT_TRUE(opened.ok());
  EXPECT_EQ(opened.value()->stats().visibility_watermark, receipt.value().visibility_watermark);
  EXPECT_EQ(opened.value()->stats().durable_watermark, receipt.value().durable_watermark);
  EXPECT_TRUE(get(opened.value(), "receipt").ok());
}

TEST_F(WalCoordinatorTest, MetadataAndEngineStageFailuresNeverPublishOneSide) {
  const std::array<float, 2> vector{1.0F, 1.0F};
  for (const auto point : {MutationFailPoint::metadata_stage_failure, MutationFailPoint::none}) {
    const auto case_root = root_ / std::to_string(static_cast<unsigned>(point));
    std::shared_ptr<FakeMutableSegment> producer;
    auto opened = open_collection(case_root, producer, true, point);
    ASSERT_TRUE(opened.ok());
    if (point == MutationFailPoint::none) {
      producer->fail_next_stage();  // Metadata dark view was built first; engine stage fails.
    }
    core::MutationContext context;
    auto result = opened.value()->write(write_request("partial", vector), context);
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(opened.value()->stats().size, 0U);
    EXPECT_GE(producer->abort_count(), 1U);
  }
}

TEST_F(WalCoordinatorTest, TornTailKeepsOnlyTheCompleteCommittedPrefix) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  core::MutationContext context;
  const std::array<float, 2> vector{2.0F, 2.0F};
  ASSERT_TRUE(opened.value()->write(write_request("prefix", vector), context).ok());
  opened.value().reset();

  const auto wal_path =
      root_ / ".alaya_internal" / kCollectionWalNamespace / kCollectionWalFilename;
  const auto torn_frame =
      logical_wal_detail::make_frame(LogicalWalRecordType::prepare, 1, 2, 2, {});
  {
    std::ofstream append(wal_path, std::ios::binary | std::ios::app);
    append.write(reinterpret_cast<const char *>(torn_frame.data()), 17);
  }
  auto torn = CollectionLogicalWal::scan_file(wal_path);
  ASSERT_TRUE(torn.ok());
  EXPECT_TRUE(torn.value().stopped_at_corrupt_or_torn_tail);

  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  EXPECT_TRUE(get(opened.value(), "prefix").ok());
  EXPECT_EQ(opened.value()->stats().visibility_watermark, 1U);
  auto healed = CollectionLogicalWal::scan_file(wal_path);
  ASSERT_TRUE(healed.ok());
  EXPECT_FALSE(healed.value().stopped_at_corrupt_or_torn_tail);
}

TEST_F(WalCoordinatorTest, DuplicateCommittedFramesReplayWithoutAnotherVersion) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  core::MutationContext context;
  const std::array<float, 2> vector{5.0F, 6.0F};
  auto receipt = opened.value()->write(write_request("duplicate", vector), context);
  ASSERT_TRUE(receipt.ok());
  opened.value().reset();
  const auto wal_path =
      root_ / ".alaya_internal" / kCollectionWalNamespace / kCollectionWalFilename;
  std::vector<char> original(static_cast<std::size_t>(std::filesystem::file_size(wal_path)));
  {
    std::ifstream input(wal_path, std::ios::binary);
    input.read(original.data(), static_cast<std::streamsize>(original.size()));
  }
  {
    std::ofstream append(wal_path, std::ios::binary | std::ios::app);
    append.write(original.data(), static_cast<std::streamsize>(original.size()));
  }
  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto record = get(opened.value(), "duplicate");
  ASSERT_TRUE(record.ok());
  EXPECT_EQ(record.value().upsert_sequence, receipt.value().op_id);
  ASSERT_EQ(producer->published_op_ids().size(), 1U);
  EXPECT_EQ(opened.value()->stats().size, 1U);
}

TEST_F(WalCoordinatorTest, BatchModesCoverAllStableStatusesAndAtomicCapability) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  const std::array<float, 2> vector{1.0F, 2.0F};
  BatchRowMutation invalid;
  invalid.logical_id = core::LogicalId::from_utf8("invalid");
  invalid.vector = core::TypedTensorView::contiguous(vector.data(), 1, 1);
  BatchRowMutation missing_delete;
  missing_delete.action = RowMutationAction::erase;
  missing_delete.logical_id = core::LogicalId::from_utf8("missing");
  BatchRowMutation insert;
  insert.logical_id = core::LogicalId::from_utf8("ordered");
  insert.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  insert.write_mode = WriteMode::insert_only;
  auto duplicate_insert = insert;
  BatchRowMutation update = insert;
  update.write_mode = WriteMode::upsert;
  BatchRowMutation replace = insert;
  replace.write_mode = WriteMode::replace;
  BatchRowMutation erase;
  erase.action = RowMutationAction::erase;
  erase.logical_id = insert.logical_id;
  std::vector rows{invalid, missing_delete, insert, duplicate_insert, update, replace, erase};
  BatchMutationRequest request;
  request.rows = rows;
  request.options.retry_token = "independent-batch";
  core::MutationContext context;
  auto receipt = collection->mutate_batch(request, context);
  ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
  ASSERT_EQ(receipt.value().rows.size(), rows.size());
  const std::array expected{RowMutationStatus::invalid_argument,
                            RowMutationStatus::not_found,
                            RowMutationStatus::inserted,
                            RowMutationStatus::already_exists,
                            RowMutationStatus::updated,
                            RowMutationStatus::replaced,
                            RowMutationStatus::deleted};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(receipt.value().rows[index].row_status, expected[index]);
  }
  auto retried = collection->mutate_batch(request, context);
  ASSERT_TRUE(retried.ok());
  EXPECT_EQ(retried.value().batch_op_id, receipt.value().batch_op_id);
  collection.reset();
  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  collection = std::move(opened).value();
  retried = collection->mutate_batch(request, context);
  ASSERT_TRUE(retried.ok());
  EXPECT_EQ(retried.value().batch_op_id, receipt.value().batch_op_id);
  ASSERT_EQ(retried.value().rows.size(), expected.size());
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(retried.value().rows[index].row_status, expected[index]);
  }

  BatchRowMutation first = insert;
  first.logical_id = core::LogicalId::from_utf8("atomic-duplicate");
  BatchRowMutation second = first;
  std::vector duplicate_rows{first, second};
  BatchMutationRequest atomic;
  atomic.rows = duplicate_rows;
  atomic.mode = BatchMutationMode::all_or_nothing;
  atomic.options.retry_token = "atomic-conflict";
  auto rejected = collection->mutate_batch(atomic, context);
  ASSERT_TRUE(rejected.ok());
  ASSERT_EQ(rejected.value().rows.size(), 2U);
  EXPECT_EQ(rejected.value().rows[0].row_status, RowMutationStatus::aborted);
  EXPECT_EQ(rejected.value().rows[1].row_status, RowMutationStatus::conflict);
  EXPECT_FALSE(get(collection, "atomic-duplicate").ok());

  first.logical_id = core::LogicalId::from_utf8("atomic-a");
  second.logical_id = core::LogicalId::from_utf8("atomic-b");
  duplicate_rows = {first, second};
  atomic.rows = duplicate_rows;
  atomic.options.retry_token = "atomic-success";
  auto committed = collection->mutate_batch(atomic, context);
  ASSERT_TRUE(committed.ok());
  EXPECT_TRUE(committed.value().searchable);
  EXPECT_TRUE(get(collection, "atomic-a").ok());
  EXPECT_TRUE(get(collection, "atomic-b").ok());

  const auto unsupported_root = root_ / "unsupported";
  opened = open_collection(unsupported_root, producer, false);
  ASSERT_TRUE(opened.ok());
  auto unsupported = opened.value()->mutate_batch(atomic, context);
  ASSERT_FALSE(unsupported.ok());
  EXPECT_EQ(unsupported.status().code(), core::StatusCode::not_supported);
}

TEST_F(WalCoordinatorTest, AtomicEngineFailureAbortsEveryRowWithoutPublishing) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  const std::array<float, 2> vector{1.0F, 2.0F};
  BatchRowMutation first;
  first.logical_id = core::LogicalId::from_utf8("abort-a");
  first.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  BatchRowMutation second = first;
  second.logical_id = core::LogicalId::from_utf8("abort-b");
  std::vector rows{first, second};
  BatchMutationRequest request;
  request.rows = rows;
  request.mode = BatchMutationMode::all_or_nothing;
  producer->fail_next_stage();
  core::MutationContext context;
  auto receipt = opened.value()->mutate_batch(request, context);
  ASSERT_TRUE(receipt.ok());
  ASSERT_EQ(receipt.value().rows.size(), 2U);
  EXPECT_EQ(receipt.value().rows[0].row_status, RowMutationStatus::aborted);
  EXPECT_EQ(receipt.value().rows[1].row_status, RowMutationStatus::aborted);
  EXPECT_FALSE(get(opened.value(), "abort-a").ok());
  EXPECT_FALSE(get(opened.value(), "abort-b").ok());
}

TEST_F(WalCoordinatorTest, CheckpointAdvancesCutManifestAndPreservesRetryLedger) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  const std::array<float, 2> vector{7.0F, 7.0F};
  core::MutationContext mutation_context;
  auto first =
      collection->write(write_request("checkpoint-a", vector, "old-token"), mutation_context);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(collection->write(write_request("checkpoint-b", vector), mutation_context).ok());
  core::CheckpointContext checkpoint_context;
  checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
  auto checkpoint = collection->checkpoint(checkpoint_context);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
  EXPECT_EQ(checkpoint.value().wal_cut, collection->stats().visibility_watermark);
  EXPECT_EQ(checkpoint.value().durable_watermark, checkpoint.value().wal_cut);
  ArtifactManifestV2 manifest;
  SegmentedCollection::apply_checkpoint_to_manifest(checkpoint.value(), manifest);
  EXPECT_EQ(manifest.wal_cut, checkpoint.value().wal_cut);
  EXPECT_EQ(manifest.collection.metadata_checkpoint, checkpoint.value().checkpoint_name);
  EXPECT_EQ(manifest.id_map_checkpoint, checkpoint.value().checkpoint_name);
  const auto wal_path =
      root_ / ".alaya_internal" / kCollectionWalNamespace / kCollectionWalFilename;
  auto scan = CollectionLogicalWal::scan_file(wal_path);
  ASSERT_TRUE(scan.ok());
  ASSERT_EQ(scan.value().frames.size(), 1U);
  EXPECT_EQ(scan.value().frames[0].type, LogicalWalRecordType::checkpoint);
  collection.reset();

  opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  collection = std::move(opened).value();
  EXPECT_TRUE(get(collection, "checkpoint-a").ok());
  EXPECT_TRUE(get(collection, "checkpoint-b").ok());
  auto retry =
      collection->write(write_request("checkpoint-a", vector, "old-token"), mutation_context);
  ASSERT_TRUE(retry.ok());
  EXPECT_EQ(retry.value().op_id, first.value().op_id);
  auto next = collection->write(write_request("checkpoint-c", vector), mutation_context);
  ASSERT_TRUE(next.ok());
  EXPECT_GT(next.value().op_id, checkpoint.value().wal_cut);
}

TEST_F(WalCoordinatorTest, CheckpointClosesAdmissionAndDrainsAnAdmittedMutationBeforeItsCut) {
  std::shared_ptr<FakeMutableSegment> producer;
  auto opened = open_collection(root_, producer);
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  const std::array<float, 2> vector{8.0F, 8.0F};
  producer->gate_next_stage();
  std::atomic_bool mutation_ok{};
  std::thread mutation([&] {
    core::MutationContext context;
    mutation_ok.store(collection->write(write_request("before-cut", vector), context).ok(),
                      std::memory_order_release);
  });
  ASSERT_TRUE(producer->wait_for_stage());
  std::atomic_bool checkpoint_done{};
  std::atomic_bool checkpoint_ok{};
  std::uint64_t cut{};
  std::thread checkpoint([&] {
    core::CheckpointContext context;
    context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto result = collection->checkpoint(context);
    if (result.ok()) {
      cut = result.value().wal_cut;
      checkpoint_ok.store(true, std::memory_order_release);
    }
    checkpoint_done.store(true, std::memory_order_release);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(checkpoint_done.load(std::memory_order_acquire));
  producer->release_stage();
  mutation.join();
  checkpoint.join();
  EXPECT_TRUE(mutation_ok.load(std::memory_order_acquire));
  EXPECT_TRUE(checkpoint_ok.load(std::memory_order_acquire));
  EXPECT_EQ(cut, collection->stats().visibility_watermark);
  EXPECT_TRUE(get(collection, "before-cut").ok());
}

#ifndef _WIN32
TEST_F(WalCoordinatorTest, SigkillProvesDurableReplayAndWeakSearchableCrashLoss) {
  const std::array<float, 2> vector{4.0F, 4.0F};
  auto run_and_kill = [&](const std::filesystem::path &root,
                          WriteDurability durability,
                          MutationFailPoint stop_at) {
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      std::shared_ptr<FakeMutableSegment> producer;
      auto opened = open_collection(root,
                                    producer,
                                    true,
                                    MutationFailPoint::none,
                                    [stop_at](MutationFailPoint point) {
                                      if (point == stop_at) {
                                        ::raise(SIGSTOP);
                                      }
                                    });
      if (!opened.ok()) {
        ::_exit(70);
      }
      core::MutationContext context;
      auto result = opened.value()->write(write_request("killed", vector, {}, durability), context);
      ::_exit(result.ok() ? 0 : 71);
    }
    int stopped_status{};
    ASSERT_EQ(::waitpid(child, &stopped_status, WUNTRACED), child);
    ASSERT_TRUE(WIFSTOPPED(stopped_status));
    ASSERT_EQ(::kill(child, SIGKILL), 0);
    int killed_status{};
    ASSERT_EQ(::waitpid(child, &killed_status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(killed_status));
    EXPECT_EQ(WTERMSIG(killed_status), SIGKILL);
  };

  const auto durable_root = root_ / "durable-kill";
  run_and_kill(durable_root, WriteDurability::wal_fsync, MutationFailPoint::after_commit);
  std::shared_ptr<FakeMutableSegment> producer;
  auto reopened = open_collection(durable_root, producer);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_TRUE(get(reopened.value(), "killed").ok());

  const auto weak_root = root_ / "weak-kill";
  run_and_kill(weak_root, WriteDurability::searchable, MutationFailPoint::after_publish);
  reopened = open_collection(weak_root, producer);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_FALSE(get(reopened.value(), "killed").ok());
  EXPECT_EQ(reopened.value()->stats().durable_watermark, 0U);
}
#endif

}  // namespace
}  // namespace alaya::internal::collection
