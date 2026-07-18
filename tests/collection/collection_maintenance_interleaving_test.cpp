// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include "fake_mutable_segment.hpp"

namespace alaya::internal::collection {
namespace {

using test::FakeMutableSegment;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::atomic_uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-collection-maintenance-" + std::string(name) + "-" + std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }
  ~TemporaryDirectory() { std::filesystem::remove_all(path_); }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct MaintenanceProbe {
  test::Barrier barrier{};
  std::atomic_uint64_t calls{};
  std::atomic_bool recovery_required{};
  std::atomic_bool fail{};

  void gate_next() { barrier.enable(); }
  [[nodiscard]] auto wait_until_entered() -> bool { return barrier.wait_until_entered(); }
  void release() { barrier.release(); }

  [[nodiscard]] auto run(std::size_t, std::size_t, bool, bool) -> core::Status {
    calls.fetch_add(1, std::memory_order_acq_rel);
    barrier.arrive_and_wait();
    if (fail.load(std::memory_order_acquire)) {
      return core::Status::error(core::StatusCode::internal,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::engine_exception,
                                 "injected maintenance failure");
    }
    return core::Status::success();
  }
};

[[nodiscard]] auto registration(const std::shared_ptr<FakeMutableSegment> &producer,
                                const std::shared_ptr<MaintenanceProbe> &maintenance,
                                std::uint64_t segment_id = FakeMutableSegment::kSegmentId,
                                std::uint64_t generation = 1) -> SegmentRegistration {
  auto erased = test::make_fake_mutable_any(producer);
  EXPECT_TRUE(erased.ok());
  SegmentRegistration result;
  result.segment_id = segment_id;
  result.generation = generation;
  result.role = SegmentRole::active_mutable;
  result.segment = std::move(erased).value();
  result.atomic_mutation_bundle = true;
  if (maintenance != nullptr) {
    result.maintenance.consolidate =
        [maintenance](std::size_t threads, std::size_t target, bool reclaim, bool bloom) {
          return maintenance->run(threads, target, reclaim, bloom);
        };
    result.maintenance.recovery_required = [maintenance] {
      return maintenance->recovery_required.load(std::memory_order_acquire);
    };
  }
  return result;
}

[[nodiscard]] auto open_collection(const std::filesystem::path &root,
                                   const std::shared_ptr<FakeMutableSegment> &producer,
                                   const std::shared_ptr<MaintenanceProbe> &maintenance,
                                   MutationFailPoint fail_point = MutationFailPoint::none,
                                   std::function<void(MutationFailPoint)> hook = {})
    -> core::Result<std::shared_ptr<SegmentedCollection>> {
  CollectionConfig config;
  config.features.wal_coordinator = true;
  config.wal.root = root;
  config.fail_point = fail_point;
  config.failpoint_hook = std::move(hook);
  std::vector<SegmentRegistration> registrations;
  registrations.push_back(registration(producer, maintenance));
  return SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                   std::move(registrations),
                                   std::move(config));
}

[[nodiscard]] auto write_request(std::string id, const std::array<float, 2> &vector)
    -> WriteRequest {
  WriteRequest request;
  request.logical_id = core::LogicalId::from_utf8(std::move(id));
  request.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
  return request;
}

[[nodiscard]] auto search_request(const std::array<float, 2> &query, core::SearchContext &context)
    -> CollectionSearchRequest {
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(query.data(), 1, query.size());
  request.options.top_k = 1;
  request.context = &context;
  return request;
}

[[nodiscard]] auto checkpoint_context() -> core::CheckpointContext {
  core::CheckpointContext context;
  context.durability_target = core::DurabilityTarget::full_checkpoint;
  return context;
}

TEST(CollectionMaintenanceInterleaving, InvalidThreadsAndMissingHookFailBeforePhysicalWork) {
  TemporaryDirectory root("admission");
  auto producer = std::make_shared<FakeMutableSegment>();
  auto opened = open_collection(root.path(), producer, nullptr);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();

  auto invalid = opened.value()->consolidate(0, 0, true, false);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), core::StatusCode::invalid_argument);
  auto unsupported = opened.value()->consolidate(1, 0, true, false);
  ASSERT_FALSE(unsupported.ok());
  EXPECT_EQ(unsupported.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(unsupported.status().detail(), core::StatusDetail::operation_slot_absent);
}

TEST(CollectionMaintenanceInterleaving, PendingWriteCompletesBeforeQueuedMaintenance) {
  TemporaryDirectory root("aw1");
  auto producer = std::make_shared<FakeMutableSegment>();
  auto maintenance = std::make_shared<MaintenanceProbe>();
  auto opened = open_collection(root.path(), producer, maintenance);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  producer->gate_next_stage();
  core::MutationContext mutation_context;
  const std::array<float, 2> vector{1.0F, 2.0F};
  auto writer = std::async(std::launch::async, [&] {
    return collection->write(write_request("aw-row", vector), mutation_context);
  });
  ASSERT_TRUE(producer->wait_for_stage());
  auto consolidate = std::async(std::launch::async, [&] {
    return collection->consolidate(1, 0, true, false);
  });
  EXPECT_EQ(consolidate.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  EXPECT_EQ(maintenance->calls.load(std::memory_order_acquire), 0U);

  producer->release_stage();
  ASSERT_TRUE(writer.get().ok());
  auto receipt = consolidate.get();
  ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
  EXPECT_EQ(receipt.value().active_segment_id, FakeMutableSegment::kSegmentId);
  EXPECT_EQ(maintenance->calls.load(std::memory_order_acquire), 1U);
  EXPECT_TRUE(collection->get_by_id(core::LogicalId::from_utf8("aw-row")).ok());
}

TEST(CollectionMaintenanceInterleaving, MaintenanceBlocksWritesButNotSearch) {
  TemporaryDirectory root("aw4-si1");
  auto producer = std::make_shared<FakeMutableSegment>();
  auto maintenance = std::make_shared<MaintenanceProbe>();
  auto opened = open_collection(root.path(), producer, maintenance);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  core::MutationContext initial_context;
  const std::array<float, 2> seed{0.0F, 0.0F};
  ASSERT_TRUE(collection->write(write_request("seed", seed), initial_context).ok());

  maintenance->gate_next();
  auto consolidate = std::async(std::launch::async, [&] {
    return collection->consolidate(1, 0, true, false);
  });
  ASSERT_TRUE(maintenance->wait_until_entered());

  core::MutationContext writer_context;
  const std::array<float, 2> next{3.0F, 4.0F};
  auto writer = std::async(std::launch::async, [&] {
    return collection->write(write_request("after-maintenance", next), writer_context);
  });
  core::SearchContext search_context;
  auto search = std::async(std::launch::async, [&] {
    return collection->search(search_request(seed, search_context));
  });
  EXPECT_EQ(writer.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  EXPECT_EQ(search.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  ASSERT_TRUE(search.get().ok())
      << "search must remain admitted while maintenance holds its write lane";

  maintenance->release();
  ASSERT_TRUE(consolidate.get().ok());
  ASSERT_TRUE(writer.get().ok());
}

TEST(CollectionMaintenanceInterleaving, RecoveryLatchWonWhileQueuedMaintenanceRechecks) {
  TemporaryDirectory root("aw3f");
  auto producer = std::make_shared<FakeMutableSegment>();
  auto maintenance = std::make_shared<MaintenanceProbe>();
  test::Barrier post_publish;
  post_publish.enable();
  auto opened =
      open_collection(root.path(),
                      producer,
                      maintenance,
                      MutationFailPoint::after_engine_publish_before_snapshot,
                      [&](MutationFailPoint point) {
                        if (point == MutationFailPoint::after_engine_publish_before_snapshot) {
                          post_publish.arrive_and_wait();
                        }
                      });
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  core::MutationContext mutation_context;
  const std::array<float, 2> vector{5.0F, 6.0F};
  auto writer = std::async(std::launch::async, [&] {
    return collection->write(write_request("committed", vector), mutation_context);
  });
  ASSERT_TRUE(post_publish.wait_until_entered());
  auto consolidate = std::async(std::launch::async, [&] {
    return collection->consolidate(1, 0, true, false);
  });
  EXPECT_EQ(consolidate.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  post_publish.release();

  ASSERT_FALSE(writer.get().ok());
  auto maintenance_result = consolidate.get();
  ASSERT_FALSE(maintenance_result.ok());
  EXPECT_EQ(maintenance_result.status().detail(), core::StatusDetail::readonly_instance);
  EXPECT_EQ(maintenance->calls.load(std::memory_order_acquire), 0U);
  auto checkpoint_request = checkpoint_context();
  auto checkpoint = collection->checkpoint(checkpoint_request);
  EXPECT_FALSE(checkpoint.ok());
  EXPECT_EQ(checkpoint.status().detail(), core::StatusDetail::readonly_instance);
  EXPECT_FALSE(collection->recovery_gate(core::OperationStage::freeze).ok());
}

TEST(CollectionMaintenanceInterleaving, CheckpointAndRotateShareMaintenanceOrder) {
  TemporaryDirectory root("checkpoint-rotate");
  auto source = std::make_shared<FakeMutableSegment>();
  auto source_maintenance = std::make_shared<MaintenanceProbe>();
  auto opened = open_collection(root.path(), source, source_maintenance);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  // CP-2 / ROT-1: a source checkpoint owns checkpoint_mutex_; maintenance cannot
  // resolve an identity until the A->B routing switch has completed.
  source->gate_next_checkpoint();
  auto successor = std::make_shared<FakeMutableSegment>();
  auto successor_maintenance = std::make_shared<MaintenanceProbe>();
  auto context = checkpoint_context();
  auto rotate = std::async(std::launch::async, [&] {
    return collection->rotate_to_successor(registration(successor,
                                                        successor_maintenance,
                                                        /*segment_id=*/3,
                                                        /*generation=*/1),
                                           context,
                                           [](const ActiveRotationReceipt &) {
                                             return core::Status::success();
                                           });
  });
  ASSERT_TRUE(source->wait_for_checkpoint());
  auto queued = std::async(std::launch::async, [&] {
    return collection->consolidate(1, 0, true, false);
  });
  EXPECT_EQ(queued.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  EXPECT_EQ(source_maintenance->calls.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(successor_maintenance->calls.load(std::memory_order_acquire), 0U);
  source->release_checkpoint();

  ASSERT_TRUE(rotate.get().ok());
  auto receipt = queued.get();
  ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
  EXPECT_EQ(receipt.value().active_segment_id, 3U);
  EXPECT_EQ(source_maintenance->calls.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(successor_maintenance->calls.load(std::memory_order_acquire), 1U);
}

TEST(CollectionMaintenanceInterleaving, PoisonedMaintenanceLatchesCollectionControlGate) {
  TemporaryDirectory root("poison");
  auto producer = std::make_shared<FakeMutableSegment>();
  auto maintenance = std::make_shared<MaintenanceProbe>();
  maintenance->fail.store(true, std::memory_order_release);
  maintenance->recovery_required.store(true, std::memory_order_release);
  auto opened = open_collection(root.path(), producer, maintenance);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  auto failed = collection->consolidate(1, 0, true, false);
  ASSERT_FALSE(failed.ok());
  EXPECT_FALSE(collection->recovery_gate(core::OperationStage::freeze).ok());
  auto checkpoint_request = checkpoint_context();
  auto checkpoint = collection->checkpoint(checkpoint_request);
  EXPECT_FALSE(checkpoint.ok());
  EXPECT_EQ(checkpoint.status().detail(), core::StatusDetail::readonly_instance);
  EXPECT_EQ(maintenance->calls.load(std::memory_order_acquire), 1U);
}

}  // namespace
}  // namespace alaya::internal::collection
