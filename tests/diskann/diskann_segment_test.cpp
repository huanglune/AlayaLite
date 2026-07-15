// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "diskann_segment_test_fixture.hpp"

namespace {
using namespace diskann_test;

TEST_F(DiskAnnSegmentTest, DescriptorCapabilitiesAndSyncWaitAdapterAreExplicit) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  const auto descriptor = segment->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, alaya::core::algorithm::diskann);
  EXPECT_EQ(descriptor.format_version, alaya::diskann::DiskANNIndex::kMetaVersion);
  EXPECT_EQ(descriptor.factory_version, 1);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, alaya::core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, alaya::core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, alaya::core::Medium::disk);

  alaya::core::SegmentStats stats;
  ASSERT_TRUE(segment->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kRows);
  EXPECT_GT(stats.resident_bytes, 0);
  EXPECT_EQ(stats.health, alaya::core::SegmentHealth::healthy);

  auto erased_result = DiskAnnSegment::into_any(std::move(segment));
  ASSERT_TRUE(erased_result.ok()) << erased_result.status().diagnostic();
  auto erased = std::move(erased_result).value();
  const auto capabilities = erased.capabilities();
  EXPECT_TRUE(capabilities.supports(OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(OperationCapability::mutation));
  EXPECT_FALSE(capabilities.supports(OperationCapability::save));
  EXPECT_FALSE(capabilities.supports(OperationCapability::export_rows));
  EXPECT_FALSE(capabilities.supports(OperationCapability::checkpoint));
  EXPECT_TRUE(capabilities.concurrency.native_async);
  EXPECT_TRUE(capabilities.concurrency.cooperative_cancel);

  const auto query = std::span(vectors_).subspan(5 * kDim, kDim);
  alaya::core::SearchContext context;
  ResponseStorage storage(1, kTopK);
  auto search_request = request(query, 1, kTopK, context, storage);
  const auto status = erased.search(search_request);
  ASSERT_TRUE(status.ok()) << status.diagnostic();
  EXPECT_EQ(storage.counts[0], kTopK);

  alaya::core::CheckpointContext checkpoint_context;
  alaya::core::CheckpointToken checkpoint_token;
  const auto checkpoint = erased.checkpoint(checkpoint_context, checkpoint_token);
  EXPECT_EQ(checkpoint.code(), alaya::core::StatusCode::not_supported);
  EXPECT_EQ(checkpoint.detail(), alaya::core::StatusDetail::operation_slot_absent);
}

TEST_F(DiskAnnSegmentTest, TypedInputEmptyAndCancelledPartialSemanticsAreExplicit) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  const auto query = std::span(vectors_).subspan(7 * kDim, kDim);

  alaya::core::SearchContext context;
  ResponseStorage empty_storage(1, 0);
  auto empty_request = request(query, 1, 0, context, empty_storage);
  ASSERT_TRUE(segment->search(empty_request).ok());
  EXPECT_EQ(empty_storage.offsets, (std::vector<alaya::core::RowCount>{0, 0}));
  EXPECT_EQ(empty_storage.counts[0], 0);
  EXPECT_EQ(empty_storage.completeness[0], alaya::core::SearchCompleteness::complete_k);

  std::array<std::int8_t, kDim> int8_query{};
  ResponseStorage typed_storage(1, kTopK);
  auto typed_request = request(query, 1, kTopK, context, typed_storage);
  typed_request.queries = alaya::core::TypedTensorView::contiguous(int8_query.data(), 1, kDim);
  const auto typed_status = segment->search(typed_request);
  EXPECT_EQ(typed_status.code(), alaya::core::StatusCode::not_supported);

  std::atomic_bool cancelled{true};
  alaya::core::SearchContext cancel_context;
  cancel_context.cancellation = alaya::core::CancellationToken::from_atomic(cancelled);
  ResponseStorage discard_storage(1, kTopK);
  auto discard_request = request(query, 1, kTopK, cancel_context, discard_storage);
  const auto discard = segment->search(discard_request);
  EXPECT_EQ(discard.code(), alaya::core::StatusCode::cancelled);
  EXPECT_FALSE(discard.partial());
  EXPECT_EQ(discard_storage.counts[0], 0);
  EXPECT_EQ(discard_storage.completeness[0], alaya::core::SearchCompleteness::failed);

  ResponseStorage retain_storage(1, kTopK);
  auto retain_request = request(query, 1, kTopK, cancel_context, retain_storage);
  retain_request.options.partial_result_policy = alaya::core::PartialResultPolicy::retain;
  const auto retain = segment->search(retain_request);
  EXPECT_EQ(retain.code(), alaya::core::StatusCode::cancelled);
  EXPECT_TRUE(retain.partial());
  EXPECT_EQ(retain_storage.counts[0], 0);
  EXPECT_EQ(retain_storage.completeness[0], alaya::core::SearchCompleteness::cancelled_partial);
}

}  // namespace

