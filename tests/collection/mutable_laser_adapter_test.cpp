// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// Unit coverage for MutableLaserCollectionAdapter driven directly (bypassing the
// Collection) so the 2B correctness spine is validated in isolation: explicit
// physical-txid idempotency (B-01/B-02), previous-inclusive tombstoning (B-05),
// and the post-commit failure latch that gates search/checkpoint (B-04).

#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "index/collection/detail/mutable_laser_collection_adapter.hpp"
#include "index/collection/mutation_wal_codec.hpp"
#include "index/disk/mutable_laser_segment.hpp"

namespace alaya::internal::collection::detail {
namespace {

constexpr std::uint32_t kDim = 64;
constexpr std::uint32_t kR = 32;
constexpr std::uint64_t kSegId = 2;
constexpr std::uint64_t kGen = 1;

std::filesystem::path scratch(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("mutable_laser_adapter_" + name + "_" + std::to_string(::getpid()));
}

std::shared_ptr<::alaya::disk::MutableLaserSegment> open_empty(const std::filesystem::path &dir) {
  std::filesystem::remove_all(dir);
  ::alaya::disk::MutableLaserSegment::create_empty(dir, "seg_00000002", kDim, kDim, kR,
                                                   core::Metric::l2);
  laser::UpdateParams params;
  params.max_points = 4096;
  params.ef_insert = 64;
  return std::make_shared<::alaya::disk::MutableLaserSegment>(
      dir, params, laser::ResidencyMode::kPagedPool, /*allow_empty=*/true);
}

CollectionSchema schema() {
  CollectionSchema s;
  s.dim = kDim;
  s.metric = core::Metric::l2;
  s.scalar_type = core::ScalarType::float32;
  return s;
}

std::vector<float> ramp(std::uint64_t row_id) {
  std::vector<float> v(kDim);
  for (std::uint32_t i = 0; i < kDim; ++i) {
    v[i] = static_cast<float>((row_id * 7 + i) % 97) * 0.01F;
  }
  return v;
}

SegmentMutationPayload make_write(std::uint64_t op_id, std::uint64_t row_id,
                                  const std::vector<float> &vec,
                                  std::optional<std::uint64_t> previous) {
  SegmentMutationPayload p;
  p.action = SegmentMutationAction::write;
  p.op_id = op_id;
  p.upsert_sequence = op_id;
  p.target = RowAddress{kSegId, kGen, core::SegmentRowId(row_id)};
  if (previous.has_value()) {
    p.previous = RowAddress{kSegId, kGen, core::SegmentRowId(*previous)};
  }
  p.vector = core::TypedTensorView::contiguous(const_cast<float *>(vec.data()), 1, kDim);
  return p;
}

// Drive one single-row transaction (prepare -> stage -> publish) with an explicit
// physical txid + max op (as Collection would set them).
core::Status drive(MutableLaserCollectionAdapter &adapter, const SegmentMutationPayload &payload,
                   std::uint64_t txid, std::uint64_t max_row_op_id,
                   WriteDurability durability = WriteDurability::wal_fsync) {
  WalMutationTransaction transaction;  // Collection's token target (durability tier)
  transaction.durability = durability;
  core::OpaqueOperationRequest request;
  request.payload = &payload;
  request.payload_size = sizeof(SegmentMutationPayload);
  core::MutationContext context;
  context.transaction_token = &transaction;
  context.transaction_id = txid;
  context.max_row_op_id = max_row_op_id;
  core::MutationToken token;
  auto status = adapter.prepare_mutation(request, context, token);
  if (!status.ok()) {
    return status;
  }
  status = adapter.stage_mutation(token, context);
  if (!status.ok()) {
    return status;
  }
  return adapter.publish_mutation(token, context);
}

bool searchable(const std::shared_ptr<::alaya::disk::MutableLaserSegment> &seg,
                const std::vector<float> &query, std::uint64_t label) {
  ::alaya::disk::DiskSearchOptions opts;
  opts.top_k = 16;
  opts.ef = 128;
  for (const auto &hit : seg->search(query.data(), opts)) {
    if (hit.label == label) {
      return true;
    }
  }
  return false;
}

// B-01/B-02: two single-row writes with DISTINCT physical txids (as a per-row batch
// yields via context.transaction_id) both apply; a replay with a txid/op already
// applied is skipped, not double-inserted.
TEST(MutableLaserAdapter, DistinctTxidsBothApplySameTxidSkips) {
  const auto dir = scratch("idempotency");
  auto seg = open_empty(dir);
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  const auto v10 = ramp(10);
  const auto v11 = ramp(11);
  ASSERT_TRUE(drive(adapter, make_write(101, 10, v10, std::nullopt), 101, 101).ok());
  ASSERT_TRUE(drive(adapter, make_write(102, 11, v11, std::nullopt), 102, 102).ok());
  EXPECT_EQ(seg->live_count(), 2U) << "distinct-txid per-row writes must both apply (B-01)";
  EXPECT_TRUE(searchable(seg, v10, 10));
  EXPECT_TRUE(searchable(seg, v11, 11));

  // Replay the second transaction again (same txid/op) -> idempotent skip.
  ASSERT_TRUE(drive(adapter, make_write(102, 11, v11, std::nullopt), 102, 102).ok());
  EXPECT_EQ(seg->live_count(), 2U) << "already-applied txid must be skipped, not re-inserted";
  std::filesystem::remove_all(dir);
}

// B-05: upsert v0 -> v1 tombstones v0's PID (its previous), so v0 is no longer live
// while v1 is searchable.
TEST(MutableLaserAdapter, UpsertTombstonesPreviousVersion) {
  const auto dir = scratch("previous");
  auto seg = open_empty(dir);
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  const auto v0 = ramp(20);
  const auto v1 = ramp(21);
  ASSERT_TRUE(drive(adapter, make_write(201, 20, v0, std::nullopt), 201, 201).ok());
  EXPECT_TRUE(seg->pid_for_label(20).has_value());
  // v1 (row_id 21) supersedes v0 (row_id 20): previous = 20.
  ASSERT_TRUE(drive(adapter, make_write(202, 21, v1, /*previous=*/20), 202, 202).ok());
  EXPECT_FALSE(seg->pid_for_label(20).has_value()) << "B-05: previous version must be tombstoned";
  EXPECT_TRUE(seg->pid_for_label(21).has_value());
  EXPECT_EQ(seg->live_count(), 1U);
  EXPECT_TRUE(searchable(seg, v1, 21));
  std::filesystem::remove_all(dir);
}

// B-04: a tombstone failure AFTER the write bundle committed latches the adapter and
// gates search / checkpoint (only abort/close pass).
TEST(MutableLaserAdapter, PostCommitTombstoneFailureLatchesAndGates) {
  const auto dir = scratch("latch");
  auto seg = open_empty(dir);
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  const auto v0 = ramp(30);
  const auto v1 = ramp(31);
  ASSERT_TRUE(drive(adapter, make_write(301, 30, v0, std::nullopt), 301, 301).ok());

  adapter.fail_tombstone_at(0);  // fail the first (only) tombstone of the upsert
  const auto status = drive(adapter, make_write(302, 31, v1, /*previous=*/30), 302, 302);
  EXPECT_FALSE(status.ok()) << "post-commit tombstone failure must surface an error";
  EXPECT_TRUE(adapter.is_latched());

  // Gated: search, batch_search, checkpoint, prepare, stats all refused.
  core::SegmentStats stats;
  EXPECT_FALSE(adapter.stats(stats).ok());
  core::CheckpointContext cp;
  core::CheckpointToken ct;
  EXPECT_FALSE(adapter.checkpoint(cp, ct).ok());
  core::OpaqueOperationRequest req;
  const auto pw = make_write(303, 32, ramp(32), std::nullopt);
  req.payload = &pw;
  req.payload_size = sizeof(SegmentMutationPayload);
  core::MutationContext ctx;
  ctx.transaction_id = 303;
  ctx.max_row_op_id = 303;
  core::MutationToken tok;
  EXPECT_FALSE(adapter.prepare_mutation(req, ctx, tok).ok());
  // abort still passes.
  EXPECT_TRUE(adapter.abort_mutation(tok, ctx).ok());
  std::filesystem::remove_all(dir);
}

// Ruling 11 / codex non-blocking 3: a non-wal_fsync write is rejected in prepare,
// before any pending is created (no leak) and without latching the adapter.
TEST(MutableLaserAdapter, RejectsNonDurableWriteWithoutPendingLeak) {
  const auto dir = scratch("durability");
  auto seg = open_empty(dir);
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  const auto v40 = ramp(40);
  const auto rejected =
      drive(adapter, make_write(401, 40, v40, std::nullopt), 401, 401, WriteDurability::searchable);
  EXPECT_FALSE(rejected.ok()) << "searchable-tier write must be refused";
  EXPECT_FALSE(adapter.is_latched()) << "a clean durability rejection must not latch";
  EXPECT_EQ(seg->live_count(), 0U);

  // No pending leaked: a fresh wal_fsync write with a new txid still succeeds.
  const auto v41 = ramp(41);
  ASSERT_TRUE(drive(adapter, make_write(402, 41, v41, std::nullopt), 402, 402).ok());
  EXPECT_EQ(seg->live_count(), 1U);
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::internal::collection::detail
