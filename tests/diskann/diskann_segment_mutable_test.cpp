// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "diskann_segment_test_fixture.hpp"

namespace {
using namespace diskann_test;

TEST_F(DiskAnnSegmentTest, MutableGateKeepsStageDarkAndPublishesThroughNativeApi) {
  TemporaryDirectory collection_root;
  DiskAnnMutableSegmentOptions mutable_options;
  mutable_options.collection_root = collection_root.path();
  mutable_options.segment_id = "seg_00000002";
  mutable_options.collection_segment_id = 2;
  alaya::internal::disk::DiskEngineFeatureFlags disabled;
  alaya::core::OpenContext disabled_context;
  auto gated = DiskAnnMutableSegmentFactory::open_any(index_directory_,
                                                      alaya::core::OpenOptions{},
                                                      disabled_context,
                                                      mutable_options,
                                                      disabled);
  ASSERT_FALSE(gated.ok());
  EXPECT_EQ(gated.status().code(), alaya::core::StatusCode::not_supported);

  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  alaya::core::OpenContext open_context;
  auto opened = DiskAnnMutableSegmentFactory::open_directory(index_directory_,
                                                             alaya::core::OpenOptions{},
                                                             open_context,
                                                             mutable_options,
                                                             enabled);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto typed_segment = std::move(opened).value();
  auto *typed = typed_segment.get();
  auto erased = DiskAnnSegment::into_mutable_any(std::move(typed_segment));
  ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
  auto segment = std::move(erased).value();
  const auto capabilities = segment.capabilities();
  EXPECT_TRUE(capabilities.supports(OperationCapability::mutation));
  EXPECT_TRUE(capabilities.supports(OperationCapability::checkpoint));
  EXPECT_TRUE(capabilities.supports(OperationCapability::close));
  EXPECT_TRUE(capabilities.supports(OperationCapability::drain));
  EXPECT_TRUE(capabilities.concurrency.search_with_stage);
  EXPECT_FALSE(capabilities.concurrency.search_with_publish);

  std::array<float, kDim> inserted{};
  inserted.fill(500.0F);
  alaya::internal::collection::SegmentMutationPayload payload;
  payload.action = alaya::internal::collection::SegmentMutationAction::write;
  payload.op_id = 1;
  payload.upsert_sequence = 1;
  payload.target = {2, 1, alaya::core::SegmentRowId(900001)};
  payload.vector = alaya::core::TypedTensorView::contiguous(inserted.data(), 1, kDim);
  alaya::core::OpaqueOperationRequest opaque;
  opaque.payload = &payload;
  opaque.payload_size = sizeof(payload);
  alaya::core::MutationContext mutation_context;
  alaya::core::MutationToken token;
  ASSERT_TRUE(segment.prepare_mutation(opaque, mutation_context, token).ok());
  ASSERT_TRUE(segment.stage_mutation(token, mutation_context).ok());
  alaya::core::SegmentStats staged_stats;
  ASSERT_TRUE(segment.stats(staged_stats).ok());
  EXPECT_EQ(staged_stats.pending_rows, 1U);
  auto mutation_stats = typed->mutable_mutation_stats();
  EXPECT_EQ(mutation_stats.prepared, 1U);
  EXPECT_EQ(mutation_stats.staged, 1U);
  EXPECT_EQ(mutation_stats.committed, 0U);
  EXPECT_EQ(mutation_stats.applied, 0U);

  alaya::core::SearchContext search_context;
  ResponseStorage dark_storage(1, 1);
  auto dark_request = request(inserted, 1, 1, search_context, dark_storage);
  ASSERT_TRUE(segment.search(dark_request).ok());
  ASSERT_EQ(dark_storage.counts[0], 1);
  EXPECT_NE(dark_storage.hits[0].row_id.value, 900001U);

  std::atomic_bool leaked_dark_label{};
  std::array<std::thread, 4> pressure;
  for (auto &thread : pressure) {
    thread = std::thread([&] {
      for (int iteration = 0; iteration < 25; ++iteration) {
        alaya::core::SearchContext concurrent_context;
        ResponseStorage concurrent_storage(1, 8);
        auto concurrent_request = request(inserted, 1, 8, concurrent_context, concurrent_storage);
        if (!segment.search(concurrent_request).ok()) {
          leaked_dark_label.store(true, std::memory_order_release);
          return;
        }
        for (std::uint64_t hit = 0; hit < concurrent_storage.counts[0]; ++hit) {
          if (concurrent_storage.hits[hit].row_id.value == 900001U) {
            leaked_dark_label.store(true, std::memory_order_release);
          }
        }
      }
    });
  }
  for (auto &thread : pressure) {
    thread.join();
  }
  EXPECT_FALSE(leaked_dark_label.load(std::memory_order_acquire));

  ASSERT_TRUE(segment.publish_mutation(token, mutation_context).ok());
  ResponseStorage published_storage(1, 1);
  auto published_request = request(inserted, 1, 1, search_context, published_storage);
  ASSERT_TRUE(segment.search(published_request).ok());
  ASSERT_EQ(published_storage.counts[0], 1);
  EXPECT_EQ(published_storage.hits[0].row_id.value, 900001U);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(published_storage.hits[0].score),
            std::bit_cast<std::uint32_t>(0.0F));

  alaya::core::SegmentStats stats;
  ASSERT_TRUE(segment.stats(stats).ok());
  EXPECT_EQ(stats.snapshot_version, 1U);
  EXPECT_EQ(stats.live_rows, kRows + 1);
  EXPECT_EQ(stats.pending_rows, 0U);
  EXPECT_GT(stats.dirty_bytes, 0U);
  mutation_stats = typed->mutable_mutation_stats();
  EXPECT_EQ(mutation_stats.committed, 1U);
  EXPECT_EQ(mutation_stats.applied, 1U);
  EXPECT_EQ(mutation_stats.replayed, 0U);
  EXPECT_EQ(mutation_stats.aborted, 0U);
  ASSERT_TRUE(segment.close().ok());
  ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
}

TEST_F(DiskAnnSegmentTest, MutableCloseDrainsPublishAndCancelledSearchBufferLifetime) {
  TemporaryDirectory collection_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions options;
  options.collection_root = collection_root.path();
  options.segment_id = "seg_00000002";
  options.collection_segment_id = 2;
  alaya::core::OpenContext open_context;
  auto opened = DiskAnnMutableSegmentFactory::open_any(async_directory_,
                                                       alaya::core::OpenOptions{},
                                                       open_context,
                                                       options,
                                                       enabled);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto segment = std::move(opened).value();

  std::array<float, kDim> inserted{};
  inserted.fill(2100.0F);
  alaya::internal::collection::SegmentMutationPayload payload;
  payload.action = alaya::internal::collection::SegmentMutationAction::write;
  payload.op_id = 1;
  payload.upsert_sequence = 1;
  payload.target = {2, 1, alaya::core::SegmentRowId(3'000'000)};
  payload.vector = alaya::core::TypedTensorView::contiguous(inserted.data(), 1, kDim);
  alaya::core::OpaqueOperationRequest opaque;
  opaque.payload = &payload;
  opaque.payload_size = sizeof(payload);
  alaya::core::MutationContext mutation_context;
  alaya::core::MutationToken token;
  ASSERT_TRUE(segment.prepare_mutation(opaque, mutation_context, token).ok());
  ASSERT_TRUE(segment.stage_mutation(token, mutation_context).ok());

  using Buffers = PinnedSearchBuffers<SafePointCancellation>;
  const auto query = std::span(async_vectors_).subspan(31 * kDim, kDim);
  auto buffers = std::make_shared<Buffers>(query);
  std::weak_ptr<Buffers> weak_buffers = buffers;
  std::weak_ptr<SafePointCancellation> weak_control = buffers->control;
  alaya::core::SearchContext search_context;
  search_context.cancellation = buffers->control->token();
  search_context.stats = &buffers->stats;
  auto search_request = request(buffers->query, 1, kTopK, search_context, buffers->storage);
  search_request.lifetime_pin = buffers;
  CompletionWaiter completion;
  std::atomic_bool pin_alive_in_callback{};
  auto started =
      segment.start_search(std::move(search_request),
                           alaya::core::SearchCompletion([&](alaya::core::Status status) {
                             pin_alive_in_callback.store(!weak_buffers.expired(),
                                                         std::memory_order_release);
                             completion.complete(std::move(status));
                           }));
  ASSERT_TRUE(started.ok()) << started.status().diagnostic();
  auto handle = std::move(started).value();
  buffers.reset();
  for (std::uint32_t spin = 0; spin < 200000; ++spin) {
    auto control = weak_control.lock();
    ASSERT_NE(control, nullptr);
    if (control->paused.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  auto control = weak_control.lock();
  ASSERT_NE(control, nullptr);
  ASSERT_TRUE(control->paused.load(std::memory_order_acquire));

  alaya::core::Status publish_status;
  std::atomic_bool publish_started{};
  std::thread publish([&] {
    publish_started.store(true, std::memory_order_release);
    publish_status = segment.publish_mutation(token, mutation_context);
  });
  while (!publish_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
  ASSERT_TRUE(segment.close().ok());
  alaya::core::Status drain_status;
  std::atomic_bool drain_done{};
  std::thread drain([&] {
    drain_status = segment.drain(alaya::core::Deadline{});
    drain_done.store(true, std::memory_order_release);
  });
  for (std::uint32_t spin = 0; spin < 10000; ++spin) {
    std::this_thread::yield();
  }
  EXPECT_FALSE(drain_done.load(std::memory_order_acquire));
  control->cancel.store(true, std::memory_order_release);
  handle.cancel();
  control->release.store(true, std::memory_order_release);
  control.reset();
  const auto terminal = completion.wait();
  EXPECT_EQ(terminal.code(), alaya::core::StatusCode::cancelled);
  publish.join();
  drain.join();
  EXPECT_TRUE(publish_status.ok()) << publish_status.diagnostic();
  EXPECT_TRUE(drain_status.ok()) << drain_status.diagnostic();
  EXPECT_TRUE(pin_alive_in_callback.load(std::memory_order_acquire));
  for (std::uint32_t spin = 0; spin < 200000 && !weak_buffers.expired(); ++spin) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(weak_buffers.expired());
}

TEST_F(DiskAnnSegmentTest, ConcurrentMutableCollectionSearchMutationStress) {
  using namespace alaya::internal::collection;
  TemporaryDirectory collection_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions options;
  options.collection_root = collection_root.path();
  options.segment_id = "seg_00000002";
  options.collection_segment_id = 2;
  alaya::core::OpenContext open_context;
  auto native = DiskAnnMutableSegmentFactory::open_directory(async_directory_,
                                                             alaya::core::OpenOptions{},
                                                             open_context,
                                                             options,
                                                             enabled);
  ASSERT_TRUE(native.ok()) << native.status().diagnostic();
  auto erased = DiskAnnSegment::into_mutable_any(std::move(native).value());
  ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
  SegmentRegistration registration;
  registration.segment_id = 2;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.next_row_id = 4'000'000;
  registration.atomic_mutation_bundle = true;
  CollectionConfig config;
  config.features.wal_coordinator = true;
  config.wal.root = collection_root.path();
  auto opened =
      SegmentedCollection::open({kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32},
                                {std::move(registration)},
                                config);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  std::array<float, kDim> query{};
  query.fill(3000.0F);
  std::atomic_bool go{};
  std::atomic_uint32_t failures{};
  std::array<std::thread, 3> searchers;
  for (auto &searcher : searchers) {
    searcher = std::thread([&] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < 60; ++iteration) {
        alaya::core::SearchContext context;
        CollectionSearchRequest request;
        request.queries = alaya::core::TypedTensorView::contiguous(query.data(), 1, kDim);
        request.options.top_k = 4;
        request.context = &context;
        auto result = collection->search(request);
        if (!result.ok() || result.value().queries.size() != 1 ||
            !result.value().queries[0].status.ok()) {
          failures.fetch_add(1, std::memory_order_acq_rel);
        }
      }
    });
  }
  go.store(true, std::memory_order_release);
  std::array<std::string, 12> ids{};
  alaya::core::MutationContext mutation_context;
  for (std::size_t index = 0; index < ids.size(); ++index) {
    ids[index] = "concurrent-" + std::to_string(index);
    std::array<float, kDim> vector{};
    vector.fill(3000.0F + static_cast<float>(index));
    WriteRequest write;
    write.logical_id = alaya::core::LogicalId::from_utf8(ids[index]);
    write.vector = alaya::core::TypedTensorView::contiguous(vector.data(), 1, kDim);
    if (!collection->write(write, mutation_context).ok()) {
      failures.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  for (std::size_t index = 0; index < 4; ++index) {
    std::array<float, kDim> vector{};
    vector.fill(3200.0F + static_cast<float>(index));
    WriteRequest update;
    update.logical_id = alaya::core::LogicalId::from_utf8(ids[index]);
    update.vector = alaya::core::TypedTensorView::contiguous(vector.data(), 1, kDim);
    if (!collection->write(update, mutation_context).ok()) {
      failures.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  for (std::size_t index = 8; index < ids.size(); ++index) {
    if (!collection->erase(alaya::core::LogicalId::from_utf8(ids[index]), mutation_context).ok()) {
      failures.fetch_add(1, std::memory_order_acq_rel);
    }
  }
  for (auto &searcher : searchers) {
    searcher.join();
  }
  EXPECT_EQ(failures.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(collection->stats().size, 8U);
  EXPECT_EQ(collection->stats().tombstone_count, 4U);
  ASSERT_TRUE(collection->close().ok());
  ASSERT_TRUE(collection->drain().ok());
}

TEST_F(DiskAnnSegmentTest, CollectionWalCheckpointReopenUsesAppliedWatermark) {
  using namespace alaya::internal::collection;
  TemporaryDirectory collection_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions mutable_options;
  mutable_options.collection_root = collection_root.path();
  mutable_options.segment_id = "seg_00000002";
  mutable_options.collection_segment_id = 2;

  auto make_registration = [&](alaya::core::AnySegment segment,
                               SegmentRole role = SegmentRole::active_mutable) {
    SegmentRegistration registration;
    registration.segment_id = 2;
    registration.role = role;
    registration.segment = std::move(segment);
    registration.atomic_mutation_bundle = role == SegmentRole::active_mutable;
    registration.rows.reserve(kRows);
    for (std::uint64_t row = 0; row < kRows; ++row) {
      RegisteredRow registered;
      registered.logical_id = alaya::core::LogicalId::from_legacy_uint64(labels_[row]);
      registered.row_id = alaya::core::SegmentRowId(labels_[row]);
      registered.upsert_sequence = row + 1;
      const auto view =
          alaya::core::TypedTensorView::contiguous(vectors_.data() + row * kDim, 1, kDim);
      auto owned = OwnedVector::copy_row(view, 0);
      EXPECT_TRUE(owned.ok()) << owned.status().diagnostic();
      if (owned.ok()) {
        registered.payload.vector = std::move(owned).value();
      }
      registration.rows.push_back(std::move(registered));
    }
    return registration;
  };

  alaya::core::OpenContext open_context;
  auto mutable_segment = DiskAnnMutableSegmentFactory::open_directory(index_directory_,
                                                                      alaya::core::OpenOptions{},
                                                                      open_context,
                                                                      mutable_options,
                                                                      enabled);
  ASSERT_TRUE(mutable_segment.ok()) << mutable_segment.status().diagnostic();
  auto erased = DiskAnnSegment::into_mutable_any(std::move(mutable_segment).value());
  ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
  CollectionConfig config;
  config.features.wal_coordinator = true;
  config.wal.root = collection_root.path();
  auto opened =
      SegmentedCollection::open({kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32},
                                {make_registration(std::move(erased).value())},
                                config);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  std::array<float, kDim> inserted{};
  inserted.fill(750.0F);
  WriteRequest write;
  write.logical_id = alaya::core::LogicalId::from_utf8("checkpointed-diskann");
  write.vector = alaya::core::TypedTensorView::contiguous(inserted.data(), 1, kDim);
  write.options.retry_token = "diskann-checkpoint-token";
  alaya::core::MutationContext mutation_context;
  auto receipt = collection->write(write, mutation_context);
  ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
  EXPECT_EQ(receipt.value().row_status, RowMutationStatus::inserted);
  EXPECT_EQ(receipt.value().durability, DurabilityState::wal_fsync);
  const auto committed_op = receipt.value().op_id;

  std::array<float, kDim> updated{};
  updated.fill(800.0F);
  WriteRequest update = write;
  update.vector = alaya::core::TypedTensorView::contiguous(updated.data(), 1, kDim);
  update.options.retry_token = "diskann-update-token";
  auto updated_receipt = collection->write(update, mutation_context);
  ASSERT_TRUE(updated_receipt.ok()) << updated_receipt.status().diagnostic();
  EXPECT_EQ(updated_receipt.value().row_status, RowMutationStatus::updated);
  const auto updated_op = updated_receipt.value().op_id;
  auto deleted =
      collection->erase(alaya::core::LogicalId::from_legacy_uint64(labels_[5]), mutation_context);
  ASSERT_TRUE(deleted.ok()) << deleted.status().diagnostic();
  EXPECT_EQ(deleted.value().row_status, RowMutationStatus::deleted);
  const auto checkpoint_cut = deleted.value().op_id;

  alaya::core::CheckpointContext checkpoint_context;
  checkpoint_context.durability_target = alaya::core::DurabilityTarget::full_checkpoint;
  auto checkpoint = collection->checkpoint(checkpoint_context);
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
  EXPECT_EQ(checkpoint.value().wal_cut, checkpoint_cut);
  ASSERT_TRUE(std::filesystem::is_regular_file(collection_root.path() / "collection_manifest.txt"));
  ASSERT_TRUE(collection->close().ok());
  ASSERT_TRUE(collection->drain().ok());
  collection.reset();

  alaya::core::OpenContext rollback_context;
  auto rollback_reader =
      DiskAnnMutableSegmentFactory::open_checkpoint_readonly_any(mutable_options,
                                                                 alaya::core::OpenOptions{},
                                                                 rollback_context);
  ASSERT_TRUE(rollback_reader.ok()) << rollback_reader.status().diagnostic();
  EXPECT_FALSE(rollback_reader.value().capabilities().supports(OperationCapability::mutation));
  EXPECT_FALSE(rollback_reader.value().capabilities().supports(OperationCapability::checkpoint));
  EXPECT_TRUE(rollback_reader.value().capabilities().supports(OperationCapability::search));
  alaya::core::SegmentStats rollback_stats;
  ASSERT_TRUE(rollback_reader.value().stats(rollback_stats).ok());
  EXPECT_EQ(rollback_stats.live_rows, kRows);
  EXPECT_EQ(rollback_stats.tombstone_rows, 2U);
  ASSERT_TRUE(rollback_reader.value().close().ok());
  ASSERT_TRUE(rollback_reader.value().drain(alaya::core::Deadline{}).ok());

  alaya::core::OpenContext reopen_context;
  auto reopened_segment = DiskAnnMutableSegmentFactory::open_checkpoint(mutable_options,
                                                                        alaya::core::OpenOptions{},
                                                                        reopen_context,
                                                                        enabled);
  ASSERT_TRUE(reopened_segment.ok()) << reopened_segment.status().diagnostic();
  EXPECT_GT(reopened_segment.value()->minimum_next_op_id(), checkpoint_cut);
  auto reopened_any = DiskAnnSegment::into_mutable_any(std::move(reopened_segment).value());
  ASSERT_TRUE(reopened_any.ok()) << reopened_any.status().diagnostic();
  auto reopened =
      SegmentedCollection::open({kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32},
                                {make_registration(std::move(reopened_any).value())},
                                config);
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  collection = std::move(reopened).value();
  auto record = collection->get_by_id(write.logical_id);
  ASSERT_TRUE(record.ok()) << record.status().diagnostic();
  EXPECT_EQ(record.value().upsert_sequence, updated_op);
  auto deleted_record =
      collection->get_by_id(alaya::core::LogicalId::from_legacy_uint64(labels_[5]));
  EXPECT_FALSE(deleted_record.ok());
  EXPECT_EQ(deleted_record.status().code(), alaya::core::StatusCode::not_found);
  auto retried = collection->write(write, mutation_context);
  ASSERT_TRUE(retried.ok()) << retried.status().diagnostic();
  EXPECT_EQ(retried.value().op_id, committed_op);

  alaya::core::SearchContext search_context;
  CollectionSearchRequest search;
  search.queries = alaya::core::TypedTensorView::contiguous(updated.data(), 1, kDim);
  search.options.top_k = 1;
  search.context = &search_context;
  auto result = collection->search(search);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  ASSERT_EQ(result.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(result.value().queries[0].hits[0].logical_id, write.logical_id);

  // Leave one durable COMMIT newer than both checkpoints. Disabling the
  // writer gate must still allow the new reader to roll that physical tail
  // forward without advertising a mutation operation table.
  std::array<float, kDim> tail_vector{};
  tail_vector.fill(900.0F);
  WriteRequest tail_write;
  tail_write.logical_id = alaya::core::LogicalId::from_utf8("rollback-wal-tail");
  tail_write.vector = alaya::core::TypedTensorView::contiguous(tail_vector.data(), 1, kDim);
  tail_write.options.retry_token = "rollback-tail-token";
  auto tail_receipt = collection->write(tail_write, mutation_context);
  ASSERT_TRUE(tail_receipt.ok()) << tail_receipt.status().diagnostic();
  const auto tail_op = tail_receipt.value().op_id;
  ASSERT_TRUE(collection->close().ok());
  ASSERT_TRUE(collection->drain().ok());
  collection.reset();

  alaya::core::OpenContext tail_reader_context;
  auto tail_reader =
      DiskAnnMutableSegmentFactory::open_checkpoint_readonly_any(mutable_options,
                                                                 alaya::core::OpenOptions{},
                                                                 tail_reader_context);
  ASSERT_TRUE(tail_reader.ok()) << tail_reader.status().diagnostic();
  EXPECT_FALSE(tail_reader.value().capabilities().supports(OperationCapability::mutation));
  EXPECT_FALSE(tail_reader.value().capabilities().supports(OperationCapability::checkpoint));
  alaya::core::SegmentStats tail_stats;
  ASSERT_TRUE(tail_reader.value().stats(tail_stats).ok());
  EXPECT_EQ(tail_stats.snapshot_version, tail_op);
  EXPECT_EQ(tail_stats.live_rows, kRows + 1);
  auto readonly_collection =
      SegmentedCollection::open({kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32},
                                {make_registration(std::move(tail_reader).value(),
                                                   SegmentRole::sealed)},
                                config);
  ASSERT_TRUE(readonly_collection.ok()) << readonly_collection.status().diagnostic();
  collection = std::move(readonly_collection).value();
  auto tail_record = collection->get_by_id(tail_write.logical_id);
  ASSERT_TRUE(tail_record.ok()) << tail_record.status().diagnostic();
  EXPECT_EQ(tail_record.value().upsert_sequence, tail_op);
  CollectionSearchRequest tail_search;
  tail_search.queries = alaya::core::TypedTensorView::contiguous(tail_vector.data(), 1, kDim);
  tail_search.options.top_k = 1;
  tail_search.context = &search_context;
  auto tail_result = collection->search(tail_search);
  ASSERT_TRUE(tail_result.ok()) << tail_result.status().diagnostic();
  ASSERT_EQ(tail_result.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(tail_result.value().queries[0].hits[0].logical_id, tail_write.logical_id);
  WriteRequest disabled_write = tail_write;
  disabled_write.logical_id = alaya::core::LogicalId::from_utf8("writer-stays-disabled");
  disabled_write.options.retry_token = "disabled-writer-token";
  auto disabled = collection->write(disabled_write, mutation_context);
  ASSERT_FALSE(disabled.ok());
  EXPECT_EQ(disabled.status().code(), alaya::core::StatusCode::not_supported);
  ASSERT_TRUE(collection->close().ok());
  ASSERT_TRUE(collection->drain().ok());
}

TEST_F(DiskAnnSegmentTest, MutableCollectionBatchModesAndBudgetDenialAreStable) {
  using namespace alaya::internal::collection;
  TemporaryDirectory collection_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions mutable_options;
  mutable_options.collection_root = collection_root.path();
  mutable_options.segment_id = "seg_00000002";
  mutable_options.collection_segment_id = 2;
  alaya::core::OpenContext open_context;
  auto native = DiskAnnMutableSegmentFactory::open_directory(index_directory_,
                                                             alaya::core::OpenOptions{},
                                                             open_context,
                                                             mutable_options,
                                                             enabled);
  ASSERT_TRUE(native.ok()) << native.status().diagnostic();
  auto erased = DiskAnnSegment::into_mutable_any(std::move(native).value());
  ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
  SegmentRegistration registration;
  registration.segment_id = 2;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.next_row_id = 1'000'000;
  registration.atomic_mutation_bundle = true;
  CollectionConfig config;
  config.features.wal_coordinator = true;
  config.wal.root = collection_root.path();
  auto opened =
      SegmentedCollection::open({kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32},
                                {std::move(registration)},
                                config);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();

  std::array<float, kDim> first{};
  first.fill(1000.0F);
  WriteRequest denied_request;
  denied_request.logical_id = alaya::core::LogicalId::from_utf8("budgeted");
  denied_request.vector = alaya::core::TypedTensorView::contiguous(first.data(), 1, kDim);
  denied_request.options.retry_token = "budget-retry-token";
  alaya::core::MutationContext wal_denied_context;
  wal_denied_context.wal_io_credits.available_requests = 0;
  auto wal_denied = collection->write(denied_request, wal_denied_context);
  ASSERT_FALSE(wal_denied.ok());
  EXPECT_EQ(wal_denied.status().code(), alaya::core::StatusCode::resource_exhausted);
  EXPECT_EQ(collection->stats().size, 0U);
  const auto wal_path = collection_root.path() / ".alaya_internal" /
                        std::string(kCollectionWalNamespace) / std::string(kCollectionWalFilename);
  ASSERT_TRUE(std::filesystem::is_regular_file(wal_path));
  EXPECT_EQ(std::filesystem::file_size(wal_path), 0U);
  alaya::core::MutationContext denied_context;
  denied_context.stage_reservation = alaya::core::MemoryReservation(0);
  auto denied = collection->write(denied_request, denied_context);
  ASSERT_FALSE(denied.ok());
  EXPECT_EQ(denied.status().code(), alaya::core::StatusCode::resource_exhausted);
  EXPECT_EQ(collection->stats().size, 0U);
  alaya::core::MutationContext mutation_context;
  auto retried = collection->write(denied_request, mutation_context);
  ASSERT_TRUE(retried.ok()) << retried.status().diagnostic();
  EXPECT_EQ(retried.value().row_status, RowMutationStatus::inserted);
  EXPECT_EQ(collection->stats().size, 1U);

  std::array<float, kDim> second{};
  second.fill(1100.0F);
  std::array<BatchRowMutation, 3> independent_rows{};
  independent_rows[0].logical_id = alaya::core::LogicalId::from_utf8("per-row");
  independent_rows[0].vector = alaya::core::TypedTensorView::contiguous(second.data(), 1, kDim);
  independent_rows[0].write_mode = WriteMode::insert_only;
  independent_rows[1] = independent_rows[0];
  independent_rows[1].vector = alaya::core::TypedTensorView::contiguous(first.data(), 1, kDim);
  independent_rows[2].action = RowMutationAction::erase;
  independent_rows[2].logical_id = alaya::core::LogicalId::from_utf8("missing");
  BatchMutationRequest independent;
  independent.rows = independent_rows;
  auto independent_receipt = collection->mutate_batch(independent, mutation_context);
  ASSERT_TRUE(independent_receipt.ok()) << independent_receipt.status().diagnostic();
  ASSERT_EQ(independent_receipt.value().rows.size(), independent_rows.size());
  EXPECT_EQ(independent_receipt.value().rows[0].row_status, RowMutationStatus::inserted);
  EXPECT_EQ(independent_receipt.value().rows[1].row_status, RowMutationStatus::already_exists);
  EXPECT_EQ(independent_receipt.value().rows[2].row_status, RowMutationStatus::not_found);
  EXPECT_EQ(collection->stats().size, 2U);

  std::array<BatchRowMutation, 2> duplicate_rows{};
  for (auto &row : duplicate_rows) {
    row.logical_id = alaya::core::LogicalId::from_utf8("atomic-duplicate");
    row.vector = alaya::core::TypedTensorView::contiguous(first.data(), 1, kDim);
  }
  BatchMutationRequest rejected_atomic;
  rejected_atomic.rows = duplicate_rows;
  rejected_atomic.mode = BatchMutationMode::all_or_nothing;
  auto rejected = collection->mutate_batch(rejected_atomic, mutation_context);
  ASSERT_TRUE(rejected.ok()) << rejected.status().diagnostic();
  EXPECT_EQ(rejected.value().rows[0].row_status, RowMutationStatus::aborted);
  EXPECT_EQ(rejected.value().rows[1].row_status, RowMutationStatus::conflict);
  EXPECT_EQ(collection->stats().size, 2U);

  std::array<float, kDim> third{};
  third.fill(1200.0F);
  std::array<float, kDim> fourth{};
  fourth.fill(1300.0F);
  std::array<BatchRowMutation, 2> atomic_rows{};
  atomic_rows[0].logical_id = alaya::core::LogicalId::from_utf8("atomic-a");
  atomic_rows[0].vector = alaya::core::TypedTensorView::contiguous(third.data(), 1, kDim);
  atomic_rows[1].logical_id = alaya::core::LogicalId::from_utf8("atomic-b");
  atomic_rows[1].vector = alaya::core::TypedTensorView::contiguous(fourth.data(), 1, kDim);
  BatchMutationRequest accepted_atomic;
  accepted_atomic.rows = atomic_rows;
  accepted_atomic.mode = BatchMutationMode::all_or_nothing;
  accepted_atomic.options.retry_token = "atomic-batch-token";
  const auto wal_size_before_atomic_denial = std::filesystem::file_size(wal_path);
  alaya::core::MutationContext atomic_denied_context;
  atomic_denied_context.stage_reservation = alaya::core::MemoryReservation(0);
  auto denied_atomic = collection->mutate_batch(accepted_atomic, atomic_denied_context);
  ASSERT_FALSE(denied_atomic.ok());
  EXPECT_EQ(denied_atomic.status().code(), alaya::core::StatusCode::resource_exhausted);
  EXPECT_EQ(collection->stats().size, 2U);
  EXPECT_FALSE(collection->get_by_id(atomic_rows[0].logical_id).ok());
  EXPECT_FALSE(collection->get_by_id(atomic_rows[1].logical_id).ok());
  EXPECT_EQ(std::filesystem::file_size(wal_path), wal_size_before_atomic_denial);
  auto atomic_receipt = collection->mutate_batch(accepted_atomic, mutation_context);
  ASSERT_TRUE(atomic_receipt.ok()) << atomic_receipt.status().diagnostic();
  ASSERT_EQ(atomic_receipt.value().rows.size(), atomic_rows.size());
  EXPECT_EQ(atomic_receipt.value().rows[0].row_status, RowMutationStatus::inserted);
  EXPECT_EQ(atomic_receipt.value().rows[1].row_status, RowMutationStatus::inserted);
  EXPECT_TRUE(collection->get_by_id(atomic_rows[0].logical_id).ok());
  EXPECT_TRUE(collection->get_by_id(atomic_rows[1].logical_id).ok());
  EXPECT_EQ(collection->stats().size, 4U);
  auto duplicate_retry = collection->mutate_batch(accepted_atomic, mutation_context);
  ASSERT_TRUE(duplicate_retry.ok()) << duplicate_retry.status().diagnostic();
  EXPECT_EQ(duplicate_retry.value().batch_op_id, atomic_receipt.value().batch_op_id);
  EXPECT_EQ(collection->stats().size, 4U);

  ASSERT_TRUE(collection->close().ok());
  ASSERT_TRUE(collection->drain().ok());
}

#ifndef _WIN32
TEST_F(DiskAnnSegmentTest, MutableWalSixPointSigkillBatteryAndRepeatedReplayConverge) {
  using namespace alaya::internal::collection;
  struct CrashCase {
    const char *name;
    std::optional<MutationFailPoint> point;
    bool committed;
  };
  const std::array cases{
      CrashCase{"before_prepare", MutationFailPoint::before_prepare, false},
      CrashCase{"after_prepare", MutationFailPoint::after_prepare, false},
      CrashCase{"after_stage", MutationFailPoint::after_stage, false},
      CrashCase{"after_commit", MutationFailPoint::after_commit, true},
      CrashCase{"after_publish", MutationFailPoint::after_publish, true},
      CrashCase{"after_receipt", std::nullopt, true},
  };
  TemporaryDirectory battery_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;

  auto open_collection = [&](const std::filesystem::path &root,
                             std::function<void(MutationFailPoint)> hook = {})
      -> alaya::core::Result<std::shared_ptr<SegmentedCollection>> {
    DiskAnnMutableSegmentOptions options;
    options.collection_root = root;
    options.segment_id = "seg_00000002";
    options.collection_segment_id = 2;
    alaya::core::OpenContext open_context;
    auto native = DiskAnnMutableSegmentFactory::open_directory(index_directory_,
                                                               alaya::core::OpenOptions{},
                                                               open_context,
                                                               options,
                                                               enabled);
    if (!native.ok()) {
      return native.status();
    }
    auto erased = DiskAnnSegment::into_mutable_any(std::move(native).value());
    if (!erased.ok()) {
      return erased.status();
    }
    SegmentRegistration registration;
    registration.segment_id = 2;
    registration.role = SegmentRole::active_mutable;
    registration.segment = std::move(erased).value();
    registration.next_row_id = 1'000'000;
    registration.atomic_mutation_bundle = true;
    CollectionConfig config;
    config.features.wal_coordinator = true;
    config.wal.root = root;
    config.failpoint_hook = std::move(hook);
    return SegmentedCollection::open({kDim,
                                      alaya::core::Metric::l2,
                                      alaya::core::ScalarType::float32},
                                     {std::move(registration)},
                                     std::move(config));
  };

  for (const auto &crash : cases) {
    const auto root = battery_root.path() / crash.name;
    std::filesystem::create_directories(root);
    const auto child = ::fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      auto hook = [point = crash.point](MutationFailPoint observed) {
        if (point.has_value() && observed == *point) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      auto opened = open_collection(root, hook);
      if (!opened.ok()) {
        ::_exit(80);
      }
      std::array<float, kDim> vector{};
      vector.fill(1400.0F);
      WriteRequest write;
      write.logical_id = alaya::core::LogicalId::from_utf8("crash-row");
      write.vector = alaya::core::TypedTensorView::contiguous(vector.data(), 1, kDim);
      write.options.retry_token = "crash-token";
      alaya::core::MutationContext context;
      auto receipt = opened.value()->write(write, context);
      if (!receipt.ok()) {
        ::_exit(81);
      }
      ::kill(::getpid(), SIGKILL);
      ::_exit(98);
    }
    int status{};
    ASSERT_EQ(::waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL) << crash.name;

    std::uint64_t recovered_op{};
    for (int replay_round = 0; replay_round < 2; ++replay_round) {
      auto recovered = open_collection(root);
      ASSERT_TRUE(recovered.ok()) << crash.name << ": " << recovered.status().diagnostic();
      auto collection = std::move(recovered).value();
      auto record = collection->get_by_id(alaya::core::LogicalId::from_utf8("crash-row"));
      EXPECT_EQ(record.ok(), crash.committed) << crash.name;
      if (crash.committed) {
        ASSERT_TRUE(record.ok());
        if (replay_round == 0) {
          recovered_op = record.value().upsert_sequence;
        } else {
          EXPECT_EQ(record.value().upsert_sequence, recovered_op);
        }
        EXPECT_EQ(collection->stats().size, 1U);
        EXPECT_EQ(collection->stats().allocated_count, kRows + 1);
        std::array<float, kDim> vector{};
        vector.fill(1400.0F);
        WriteRequest retry;
        retry.logical_id = alaya::core::LogicalId::from_utf8("crash-row");
        retry.vector = alaya::core::TypedTensorView::contiguous(vector.data(), 1, kDim);
        retry.options.retry_token = "crash-token";
        alaya::core::MutationContext context;
        auto receipt = collection->write(retry, context);
        ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
        EXPECT_EQ(receipt.value().op_id, recovered_op);
        EXPECT_EQ(collection->stats().size, 1U);
      }
      ASSERT_TRUE(collection->close().ok());
      ASSERT_TRUE(collection->drain().ok());
    }
  }
}
#endif

TEST_F(DiskAnnSegmentTest, MutableDifferentialSequenceAndCheckpointCutsMatchDirectKernel) {
  using namespace alaya::internal::collection;
  TemporaryDirectory root;
  const auto direct_directory = root.path() / "direct";
  std::filesystem::copy(index_directory_,
                        direct_directory,
                        std::filesystem::copy_options::recursive);
  auto direct = std::make_unique<alaya::diskann::DiskANNIndex>();
  alaya::diskann::DiskANNLoadParams load;
  load.num_threads = DiskAnnSegment::kSearchThreads;
  load.beam_width = DiskAnnSegment::kBeamWidth;
  load.scratch_search_list_size = DiskAnnSegment::kScratchSearchListSize;
  load.updatable = true;
  load.search_page_cache = true;
  direct->load(direct_directory.string(), load);

  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions options;
  options.collection_root = root.path() / "segment_collection";
  options.segment_id = "seg_00000002";
  options.collection_segment_id = 2;
  auto open_fresh = [&]() -> alaya::core::Result<alaya::core::AnySegment> {
    alaya::core::OpenContext context;
    auto opened = DiskAnnMutableSegmentFactory::open_directory(index_directory_,
                                                               alaya::core::OpenOptions{},
                                                               context,
                                                               options,
                                                               enabled);
    if (!opened.ok()) {
      return opened.status();
    }
    return DiskAnnSegment::into_mutable_any(std::move(opened).value());
  };
  auto opened = open_fresh();
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto segment = std::move(opened).value();

  auto apply_one = [&](std::uint64_t op_id,
                       SegmentMutationAction action,
                       std::uint64_t target_label,
                       const std::array<float, kDim> *vector,
                       std::optional<std::uint64_t> previous_label = std::nullopt) {
    SegmentMutationPayload payload;
    payload.action = action;
    payload.op_id = op_id;
    payload.upsert_sequence = op_id;
    payload.target = {2, 1, alaya::core::SegmentRowId(target_label)};
    if (previous_label.has_value()) {
      payload.previous = RowAddress{2, 1, alaya::core::SegmentRowId(*previous_label)};
    }
    if (vector != nullptr) {
      payload.vector = alaya::core::TypedTensorView::contiguous(vector->data(), 1, kDim);
    }
    alaya::core::OpaqueOperationRequest request;
    request.payload = &payload;
    request.payload_size = sizeof(payload);
    alaya::core::MutationContext context;
    alaya::core::MutationToken token;
    EXPECT_TRUE(segment.prepare_mutation(request, context, token).ok());
    EXPECT_TRUE(segment.stage_mutation(token, context).ok());
    EXPECT_TRUE(segment.publish_mutation(token, context).ok());
    return payload;
  };

  std::array<float, kDim> a{};
  std::array<float, kDim> b{};
  std::array<float, kDim> c{};
  std::array<float, kDim> d{};
  std::array<float, kDim> e{};
  a.fill(1500.0F);
  b.fill(1600.0F);
  c.fill(1700.0F);
  d.fill(1800.0F);
  e.fill(1900.0F);
  constexpr std::uint64_t kLabelA = 2'000'000;
  constexpr std::uint64_t kLabelB = 2'000'001;
  constexpr std::uint64_t kLabelC = 2'000'002;
  constexpr std::uint64_t kLabelD = 2'000'003;
  constexpr std::uint64_t kLabelE = 2'000'004;
  const auto direct_a = direct->insert(a.data(), kLabelA);
  (void)apply_one(1, SegmentMutationAction::write, kLabelA, &a);
  const auto direct_b = direct->insert(b.data(), kLabelB);
  (void)apply_one(2, SegmentMutationAction::write, kLabelB, &b);
  (void)direct->insert(c.data(), kLabelC);
  direct->remove(direct_a);
  const auto upsert_payload = apply_one(3, SegmentMutationAction::write, kLabelC, &c, kLabelA);

  alaya::core::CheckpointContext checkpoint_context;
  checkpoint_context.durability_target = alaya::core::DurabilityTarget::full_checkpoint;
  alaya::core::CheckpointToken checkpoint_token;
  ASSERT_TRUE(segment.checkpoint(checkpoint_context, checkpoint_token).ok());
  EXPECT_EQ(checkpoint_token.value, 3U);
  ASSERT_TRUE(segment.close().ok());
  ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
  segment = {};
  alaya::core::OpenContext reopen_context;
  auto reopened_native = DiskAnnMutableSegmentFactory::open_checkpoint(options,
                                                                       alaya::core::OpenOptions{},
                                                                       reopen_context,
                                                                       enabled);
  ASSERT_TRUE(reopened_native.ok()) << reopened_native.status().diagnostic();
  auto reopened_any = DiskAnnSegment::into_mutable_any(std::move(reopened_native).value());
  ASSERT_TRUE(reopened_any.ok()) << reopened_any.status().diagnostic();
  segment = std::move(reopened_any).value();
  alaya::core::OpaqueOperationRequest duplicate;
  duplicate.payload = &upsert_payload;
  duplicate.payload_size = sizeof(upsert_payload);
  alaya::core::MutationContext replay_context;
  ASSERT_TRUE(segment.replay_mutation(duplicate, replay_context).ok());
  alaya::core::SegmentStats after_duplicate;
  ASSERT_TRUE(segment.stats(after_duplicate).ok());
  EXPECT_EQ(after_duplicate.allocated_rows, kRows + 3);

  direct->remove(direct_b);
  (void)apply_one(4, SegmentMutationAction::erase, 2'100'000, nullptr, kLabelB);
  std::array<SegmentMutationPayload, 2> atomic_payloads{};
  atomic_payloads[0].action = SegmentMutationAction::write;
  atomic_payloads[0].op_id = 5;
  atomic_payloads[0].upsert_sequence = 5;
  atomic_payloads[0].target = {2, 1, alaya::core::SegmentRowId(kLabelD)};
  atomic_payloads[0].vector = alaya::core::TypedTensorView::contiguous(d.data(), 1, kDim);
  atomic_payloads[1].action = SegmentMutationAction::write;
  atomic_payloads[1].op_id = 6;
  atomic_payloads[1].upsert_sequence = 6;
  atomic_payloads[1].target = {2, 1, alaya::core::SegmentRowId(kLabelE)};
  atomic_payloads[1].vector = alaya::core::TypedTensorView::contiguous(e.data(), 1, kDim);
  SegmentMutationBundlePayload bundle;
  bundle.batch_op_id = 5;
  bundle.rows = atomic_payloads;
  alaya::core::OpaqueOperationRequest bundle_request;
  bundle_request.payload = &bundle;
  bundle_request.payload_size = sizeof(bundle);
  alaya::core::MutationContext bundle_context;
  alaya::core::MutationToken bundle_token;
  ASSERT_TRUE(segment.prepare_mutation(bundle_request, bundle_context, bundle_token).ok());
  ASSERT_TRUE(segment.stage_mutation(bundle_token, bundle_context).ok());
  ASSERT_TRUE(segment.publish_mutation(bundle_token, bundle_context).ok());
  (void)direct->insert(d.data(), kLabelD);
  (void)direct->insert(e.data(), kLabelE);

  DiskAnnSegmentSearchExtension extension_options;
  extension_options.search_list_size = 100;
  extension_options.use_pq = false;
  extension_options.rerank = false;
  extension_options.deterministic = true;
  const auto extension = DiskAnnSegment::make_search_extension(extension_options);
  alaya::diskann::DiskANNSearchParams direct_options;
  direct_options.search_list_size = 100;
  direct_options.use_pq = false;
  direct_options.rerank = false;
  direct_options.deterministic = true;
  std::array<float, kDim> initial_query{};
  std::copy_n(vectors_.data(), kDim, initial_query.data());
  const std::array queries{c, d, e, initial_query};
  for (const auto &query : queries) {
    constexpr std::uint32_t top_k = 10;
    std::array<std::uint64_t, top_k> direct_labels{};
    std::array<float, top_k> direct_scores{};
    const auto direct_count = direct->search(query.data(),
                                             top_k,
                                             direct_labels.data(),
                                             direct_scores.data(),
                                             direct_options);
    alaya::core::SearchContext search_context;
    ResponseStorage storage(1, top_k);
    const std::array extensions{extension};
    auto search_request = request(query, 1, top_k, search_context, storage, extensions);
    ASSERT_TRUE(segment.search(search_request).ok());
    ASSERT_EQ(storage.counts[0], direct_count);
    for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
      EXPECT_EQ(storage.hits[hit].row_id.value, direct_labels[hit]);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(storage.hits[hit].score),
                std::bit_cast<std::uint32_t>(direct_scores[hit]));
    }
  }
  alaya::core::SegmentStats final_stats;
  ASSERT_TRUE(segment.stats(final_stats).ok());
  EXPECT_EQ(final_stats.live_rows, direct->live_count());
  EXPECT_EQ(final_stats.tombstone_rows, direct->tombstone_count());
  EXPECT_EQ(final_stats.snapshot_version, 6U);
  ASSERT_TRUE(segment.close().ok());
  ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
}

TEST_F(DiskAnnSegmentTest, MutableRandomizedPublishCheckpointCutsRemainDifferential) {
  using namespace alaya::internal::collection;
  TemporaryDirectory root;
  const auto direct_directory = root.path() / "direct-random";
  std::filesystem::copy(index_directory_,
                        direct_directory,
                        std::filesystem::copy_options::recursive);
  auto direct = std::make_unique<alaya::diskann::DiskANNIndex>();
  alaya::diskann::DiskANNLoadParams load;
  load.num_threads = DiskAnnSegment::kSearchThreads;
  load.beam_width = DiskAnnSegment::kBeamWidth;
  load.scratch_search_list_size = DiskAnnSegment::kScratchSearchListSize;
  load.updatable = true;
  load.search_page_cache = true;
  direct->load(direct_directory.string(), load);

  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  DiskAnnMutableSegmentOptions options;
  options.collection_root = root.path() / "segment-random";
  options.segment_id = "seg_00000002";
  options.collection_segment_id = 2;
  alaya::core::OpenContext open_context;
  auto opened = DiskAnnMutableSegmentFactory::open_any(index_directory_,
                                                       alaya::core::OpenOptions{},
                                                       open_context,
                                                       options,
                                                       enabled);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto segment = std::move(opened).value();

  struct LiveRow {
    std::uint64_t label{};
    std::uint32_t slot{};
    std::array<float, kDim> vector{};
  };
  std::vector<LiveRow> live;
  std::mt19937_64 random(0xD15CA88ULL);
  std::uniform_real_distribution<float> jitter(-0.25F, 0.25F);
  std::array<bool, 16> checkpoint_cut{};
  std::size_t selected{};
  while (selected != 4) {
    const auto candidate = static_cast<std::size_t>(2 + random() % 13);
    if (!checkpoint_cut[candidate]) {
      checkpoint_cut[candidate] = true;
      ++selected;
    }
  }

  auto apply = [&](std::uint64_t op_id,
                   SegmentMutationAction action,
                   std::uint64_t target,
                   const std::array<float, kDim> *vector,
                   std::optional<std::uint64_t> previous) {
    SegmentMutationPayload payload;
    payload.action = action;
    payload.op_id = op_id;
    payload.upsert_sequence = op_id;
    payload.target = {2, 1, alaya::core::SegmentRowId(target)};
    if (previous.has_value()) {
      payload.previous = RowAddress{2, 1, alaya::core::SegmentRowId(*previous)};
    }
    if (vector != nullptr) {
      payload.vector = alaya::core::TypedTensorView::contiguous(vector->data(), 1, kDim);
    }
    alaya::core::OpaqueOperationRequest request;
    request.payload = &payload;
    request.payload_size = sizeof(payload);
    alaya::core::MutationContext context;
    alaya::core::MutationToken token;
    ASSERT_TRUE(segment.prepare_mutation(request, context, token).ok());
    ASSERT_TRUE(segment.stage_mutation(token, context).ok());
    ASSERT_TRUE(segment.publish_mutation(token, context).ok());
  };

  DiskAnnSegmentSearchExtension extension_options;
  extension_options.search_list_size = 100;
  extension_options.use_pq = false;
  extension_options.rerank = false;
  extension_options.deterministic = true;
  const auto extension = DiskAnnSegment::make_search_extension(extension_options);
  alaya::diskann::DiskANNSearchParams direct_options;
  direct_options.search_list_size = 100;
  direct_options.use_pq = false;
  direct_options.rerank = false;
  direct_options.deterministic = true;
  auto verify = [&] {
    alaya::core::SegmentStats stats;
    ASSERT_TRUE(segment.stats(stats).ok());
    EXPECT_EQ(stats.live_rows, direct->live_count());
    EXPECT_EQ(stats.tombstone_rows, direct->tombstone_count());
    std::array<float, kDim> baseline_query{};
    std::copy_n(vectors_.data() + 9 * kDim, kDim, baseline_query.data());
    std::vector<std::array<float, kDim>> queries{baseline_query};
    for (const auto &row : live) {
      queries.push_back(row.vector);
    }
    for (const auto &query : queries) {
      constexpr std::uint32_t top_k = 12;
      std::array<std::uint64_t, top_k> direct_labels{};
      std::array<float, top_k> direct_scores{};
      const auto direct_count = direct->search(query.data(),
                                               top_k,
                                               direct_labels.data(),
                                               direct_scores.data(),
                                               direct_options);
      alaya::core::SearchContext search_context;
      ResponseStorage storage(1, top_k);
      const std::array extensions{extension};
      auto search_request = request(query, 1, top_k, search_context, storage, extensions);
      ASSERT_TRUE(segment.search(search_request).ok());
      ASSERT_EQ(storage.counts[0], direct_count);
      for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
        EXPECT_EQ(storage.hits[hit].row_id.value, direct_labels[hit]);
        EXPECT_EQ(std::bit_cast<std::uint32_t>(storage.hits[hit].score),
                  std::bit_cast<std::uint32_t>(direct_scores[hit]));
      }
    }
  };

  std::uint64_t next_label = 6'000'000;
  std::uint64_t op_id = 1;
  for (std::size_t step = 0; step < checkpoint_cut.size(); ++step, ++op_id) {
    const auto choice = random() % 100;
    if (live.empty() || choice < 50) {
      LiveRow row;
      row.label = next_label++;
      for (auto &value : row.vector) {
        value = 5000.0F + static_cast<float>(step * 100) + jitter(random);
      }
      row.slot = direct->insert(row.vector.data(), row.label);
      apply(op_id, SegmentMutationAction::write, row.label, &row.vector, std::nullopt);
      live.push_back(row);
    } else if (choice < 80) {
      const auto index = static_cast<std::size_t>(random() % live.size());
      LiveRow replacement;
      replacement.label = next_label++;
      for (auto &value : replacement.vector) {
        value = 7000.0F + static_cast<float>(step * 100) + jitter(random);
      }
      replacement.slot = direct->insert(replacement.vector.data(), replacement.label);
      direct->remove(live[index].slot);
      apply(op_id,
            SegmentMutationAction::write,
            replacement.label,
            &replacement.vector,
            live[index].label);
      live[index] = replacement;
    } else {
      const auto index = static_cast<std::size_t>(random() % live.size());
      direct->remove(live[index].slot);
      apply(op_id, SegmentMutationAction::erase, next_label++, nullptr, live[index].label);
      live.erase(live.begin() + static_cast<std::ptrdiff_t>(index));
    }

    if (checkpoint_cut[step]) {
      verify();
      alaya::core::CheckpointContext checkpoint_context;
      checkpoint_context.durability_target = alaya::core::DurabilityTarget::full_checkpoint;
      alaya::core::CheckpointToken token;
      ASSERT_TRUE(segment.checkpoint(checkpoint_context, token).ok());
      EXPECT_EQ(token.value, op_id);
      ASSERT_TRUE(segment.close().ok());
      ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
      segment = {};
      alaya::core::OpenContext reopen_context;
      auto reopened = DiskAnnMutableSegmentFactory::open_checkpoint(options,
                                                                    alaya::core::OpenOptions{},
                                                                    reopen_context,
                                                                    enabled);
      ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
      auto erased = DiskAnnSegment::into_mutable_any(std::move(reopened).value());
      ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
      segment = std::move(erased).value();
      verify();
    }
  }
  verify();
  ASSERT_TRUE(segment.close().ok());
  ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
}

TEST_F(DiskAnnSegmentTest, MutableCheckpointFiveStepFailuresPreserveOldGeneration) {
  using alaya::internal::collection::ArtifactTransactionFailPoint;
  using alaya::internal::collection::SegmentMutationAction;
  using alaya::internal::collection::SegmentMutationPayload;
  const std::array failpoints{
      ArtifactTransactionFailPoint::after_staging_write,
      ArtifactTransactionFailPoint::before_ready,
      ArtifactTransactionFailPoint::after_ready_before_publish,
      ArtifactTransactionFailPoint::after_payload_publish_before_manifest,
  };
  TemporaryDirectory matrix_root;
  alaya::internal::disk::DiskEngineFeatureFlags enabled;
  enabled.diskann_mutable_segment = true;
  for (std::size_t index = 0; index < failpoints.size(); ++index) {
    DiskAnnMutableSegmentOptions options;
    options.collection_root = matrix_root.path() / std::to_string(index);
    options.segment_id = "seg_00000002";
    options.collection_segment_id = 2;
    alaya::core::OpenContext open_context;
    auto fresh = DiskAnnMutableSegmentFactory::open_any(index_directory_,
                                                        alaya::core::OpenOptions{},
                                                        open_context,
                                                        options,
                                                        enabled);
    ASSERT_TRUE(fresh.ok()) << fresh.status().diagnostic();
    auto segment = std::move(fresh).value();
    auto apply = [&](std::uint64_t op_id, std::uint64_t label, const auto &vector) {
      SegmentMutationPayload payload;
      payload.action = SegmentMutationAction::write;
      payload.op_id = op_id;
      payload.upsert_sequence = op_id;
      payload.target = {2, 1, alaya::core::SegmentRowId(label)};
      payload.vector = alaya::core::TypedTensorView::contiguous(vector.data(), 1, kDim);
      alaya::core::OpaqueOperationRequest request;
      request.payload = &payload;
      request.payload_size = sizeof(payload);
      alaya::core::MutationContext context;
      alaya::core::MutationToken token;
      EXPECT_TRUE(segment.prepare_mutation(request, context, token).ok());
      EXPECT_TRUE(segment.stage_mutation(token, context).ok());
      EXPECT_TRUE(segment.publish_mutation(token, context).ok());
    };
    std::array<float, kDim> first{};
    first.fill(3500.0F);
    apply(1, 5'000'000, first);
    alaya::core::CheckpointContext checkpoint_context;
    checkpoint_context.durability_target = alaya::core::DurabilityTarget::full_checkpoint;
    alaya::core::CheckpointToken token;
    ASSERT_TRUE(segment.checkpoint(checkpoint_context, token).ok());
    ASSERT_TRUE(segment.close().ok());
    ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
    segment = {};

    options.checkpoint_fail_point = failpoints[index];
    alaya::core::OpenContext failure_open_context;
    auto failing_native = DiskAnnMutableSegmentFactory::open_checkpoint(options,
                                                                        alaya::core::OpenOptions{},
                                                                        failure_open_context,
                                                                        enabled);
    ASSERT_TRUE(failing_native.ok()) << failing_native.status().diagnostic();
    auto failing_any = DiskAnnSegment::into_mutable_any(std::move(failing_native).value());
    ASSERT_TRUE(failing_any.ok()) << failing_any.status().diagnostic();
    segment = std::move(failing_any).value();
    std::array<float, kDim> second{};
    second.fill(4000.0F);
    apply(2, 5'000'001, second);
    alaya::core::CheckpointToken failed_token;
    auto failed = segment.checkpoint(checkpoint_context, failed_token);
    ASSERT_FALSE(failed.ok());
    EXPECT_EQ(failed.code(), alaya::core::StatusCode::io_error);
    ASSERT_TRUE(segment.close().ok());
    ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
    segment = {};

    options.checkpoint_fail_point = ArtifactTransactionFailPoint::none;
    alaya::core::OpenContext recovery_context;
    auto recovered_native =
        DiskAnnMutableSegmentFactory::open_checkpoint(options,
                                                      alaya::core::OpenOptions{},
                                                      recovery_context,
                                                      enabled);
    ASSERT_TRUE(recovered_native.ok()) << recovered_native.status().diagnostic();
    auto recovered_any = DiskAnnSegment::into_mutable_any(std::move(recovered_native).value());
    ASSERT_TRUE(recovered_any.ok()) << recovered_any.status().diagnostic();
    segment = std::move(recovered_any).value();
    alaya::core::SegmentStats recovered_stats;
    ASSERT_TRUE(segment.stats(recovered_stats).ok());
    EXPECT_EQ(recovered_stats.snapshot_version, 1U);
    EXPECT_EQ(recovered_stats.live_rows, kRows + 1);
    alaya::core::SearchContext search_context;
    ResponseStorage storage(1, 1);
    auto search_request = request(second, 1, 1, search_context, storage);
    ASSERT_TRUE(segment.search(search_request).ok());
    ASSERT_EQ(storage.counts[0], 1U);
    EXPECT_NE(storage.hits[0].row_id.value, 5'000'001U);

    // Recovery cleanup removed any READY orphan, so the same next generation
    // can be applied and published without a target-directory conflict.
    apply(2, 5'000'001, second);
    alaya::core::CheckpointToken recovered_token;
    ASSERT_TRUE(segment.checkpoint(checkpoint_context, recovered_token).ok());
    EXPECT_EQ(recovered_token.value, 2U);
    ASSERT_TRUE(segment.close().ok());
    ASSERT_TRUE(segment.drain(alaya::core::Deadline{}).ok());
  }
}

}  // namespace
