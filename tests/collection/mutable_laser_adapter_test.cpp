// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// Unit coverage for MutableLaserCollectionAdapter driven directly (bypassing the
// Collection) so the 2B correctness spine is validated in isolation: explicit
// physical-txid idempotency (B-01/B-02), previous-inclusive tombstoning (B-05),
// and the post-commit failure latch that gates search/checkpoint (B-04).

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <system_error>
#include <thread>
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

std::shared_ptr<::alaya::disk::MutableLaserSegment> open_empty(
    const std::filesystem::path &dir,
    std::function<void(laser::SegmentOpFailPoint)> failpoint_hook = {}) {
  std::filesystem::remove_all(dir);
  ::alaya::disk::MutableLaserSegment::create_empty(dir,
                                                   "seg_00000002",
                                                   kDim,
                                                   kDim,
                                                   kR,
                                                   core::Metric::l2);
  laser::UpdateParams params;
  params.max_points = 4096;
  params.ef_insert = 64;
  params.failpoint_hook = std::move(failpoint_hook);
  return std::make_shared<::alaya::disk::MutableLaserSegment>(dir,
                                                              params,
                                                              laser::ResidencyMode::kPagedPool,
                                                              /*allow_empty=*/true);
}

class Gate {
 public:
  void wait() {
    entered_.store(true, std::memory_order_release);
    while (!released_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
  [[nodiscard]] auto wait_until_entered() -> bool {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered_.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::yield();
    }
    return true;
  }
  void release() { released_.store(true, std::memory_order_release); }

 private:
  std::atomic_bool entered_{};
  std::atomic_bool released_{};
};

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

TEST(MutableLaserAdapter, PendingMutationRejectsMaintenanceWithoutLatch) {
  const auto dir = scratch("maintenance_pending");
  auto seg = open_empty(dir);
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);
  const auto vector = ramp(50);
  const auto payload = make_write(501, 50, vector, std::nullopt);
  WalMutationTransaction transaction;
  transaction.durability = WriteDurability::wal_fsync;
  core::OpaqueOperationRequest request;
  request.payload = &payload;
  request.payload_size = sizeof(payload);
  core::MutationContext context;
  context.transaction_token = &transaction;
  context.transaction_id = 501;
  context.max_row_op_id = 501;
  core::MutationToken token;
  ASSERT_TRUE(adapter.prepare_mutation(request, context, token).ok());

  auto conflict = adapter.consolidate(1, 0, true, false);
  ASSERT_FALSE(conflict.ok());
  EXPECT_EQ(conflict.code(), core::StatusCode::conflict);
  EXPECT_FALSE(adapter.is_latched());
  ASSERT_TRUE(adapter.abort_mutation(token, context).ok());
  EXPECT_TRUE(adapter.consolidate(1, 0, true, false).ok());
  EXPECT_FALSE(adapter.is_latched());
  std::filesystem::remove_all(dir);
}

TEST(MutableLaserAdapter, MaintenanceFlagRejectsPrepareWhileSearchContinues) {
  const auto dir = scratch("maintenance_active");
  Gate gate;
  auto seg = open_empty(dir, [&](laser::SegmentOpFailPoint point) {
    if (point == laser::SegmentOpFailPoint::after_consolidate_begin_fsync) {
      gate.wait();
    }
  });
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);
  const auto seed = ramp(60);
  ASSERT_TRUE(drive(adapter, make_write(601, 60, seed, std::nullopt), 601, 601).ok());

  auto maintenance = std::async(std::launch::async, [&] {
    return adapter.consolidate(1, 0, true, false);
  });
  ASSERT_TRUE(gate.wait_until_entered());

  const auto next = ramp(61);
  const auto payload = make_write(602, 61, next, std::nullopt);
  WalMutationTransaction transaction;
  transaction.durability = WriteDurability::wal_fsync;
  core::OpaqueOperationRequest opaque;
  opaque.payload = &payload;
  opaque.payload_size = sizeof(payload);
  core::MutationContext mutation_context;
  mutation_context.transaction_token = &transaction;
  mutation_context.transaction_id = 602;
  mutation_context.max_row_op_id = 602;
  core::MutationToken token;
  auto rejected = adapter.prepare_mutation(opaque, mutation_context, token);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.code(), core::StatusCode::conflict);
  EXPECT_FALSE(adapter.is_latched());

  std::vector<core::SearchHit> hits(4);
  std::array<core::RowCount, 2> offsets{};
  std::array<core::RowCount, 1> counts{};
  std::array<core::Status, 1> statuses{};
  std::array<core::SearchCompleteness, 1> completeness{};
  core::SearchResponse response;
  response.hits = hits;
  response.offsets = offsets;
  response.valid_counts = counts;
  response.statuses = statuses;
  response.completeness = completeness;
  core::SearchContext search_context;
  core::SearchRequest search_request;
  search_request.queries = core::TypedTensorView::contiguous(seed.data(), 1, seed.size());
  search_request.options.top_k = 4;
  search_request.context = &search_context;
  search_request.response = &response;
  EXPECT_TRUE(adapter.search(search_request).ok());
  EXPECT_GE(counts[0], 1U);

  gate.release();
  EXPECT_TRUE(maintenance.get().ok());
  std::filesystem::remove_all(dir);
}

TEST(MutableLaserAdapter, StatvfsFailureBeforeBeginClearsMaintenanceWithoutLatch) {
  const auto dir = scratch("maintenance_statvfs");
  std::atomic_bool armed{true};
  auto seg = open_empty(dir, [&](laser::SegmentOpFailPoint point) {
    if (point == laser::SegmentOpFailPoint::before_consolidate_statvfs &&
        armed.exchange(false, std::memory_order_acq_rel)) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "injected statvfs failure");
    }
  });
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  auto failed = adapter.consolidate(1, 0, true, false);
  EXPECT_FALSE(failed.ok());
  EXPECT_FALSE(adapter.is_latched());
  const auto vector = ramp(70);
  EXPECT_TRUE(drive(adapter, make_write(701, 70, vector, std::nullopt), 701, 701).ok());
  core::CheckpointContext context;
  core::CheckpointToken token;
  EXPECT_TRUE(adapter.checkpoint(context, token).ok());
  EXPECT_TRUE(adapter.consolidate(1, 0, true, false).ok());
  EXPECT_FALSE(adapter.is_latched());
  std::filesystem::remove_all(dir);
}

TEST(MutableLaserAdapter, FailureAfterDurableBeginLatchesAndGatesCheckpoint) {
  const auto dir = scratch("maintenance_after_begin");
  std::atomic_bool armed{true};
  auto seg = open_empty(dir, [&](laser::SegmentOpFailPoint point) {
    if (point == laser::SegmentOpFailPoint::after_consolidate_begin_fsync &&
        armed.exchange(false, std::memory_order_acq_rel)) {
      throw std::system_error(std::make_error_code(std::errc::no_space_on_device),
                              "injected post-BEGIN ENOSPC");
    }
  });
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  auto failed = adapter.consolidate(1, 0, true, false);
  EXPECT_FALSE(failed.ok());
  EXPECT_TRUE(adapter.is_latched());
  core::CheckpointContext context;
  core::CheckpointToken token;
  EXPECT_FALSE(adapter.checkpoint(context, token).ok());
  EXPECT_FALSE(adapter.consolidate(1, 0, true, false).ok());
  std::filesystem::remove_all(dir);
}

TEST(MutableLaserAdapter, SharedExtensionsResolveSameForActiveAndSealedDefaults) {
  ::alaya::disk::LaserSegmentSearchExtension parameters;
  parameters.effort = 37;
  parameters.beam_width = 7;
  auto extension = ::alaya::disk::make_laser_segment_search_extension(parameters);
  core::SearchOptions request_options(5);
  request_options.extensions =
      std::span<const core::AlgorithmSearchExtension>(&extension, 1);

  ::alaya::disk::DiskSearchOptions active_defaults;
  active_defaults.ef = 128;
  active_defaults.beam_width = 4;
  ::alaya::disk::DiskSearchOptions sealed_defaults;
  sealed_defaults.ef = 100;
  sealed_defaults.beam_width = 4;
  auto active = ::alaya::disk::resolve_laser_search_extensions(request_options, active_defaults);
  auto sealed = ::alaya::disk::resolve_laser_search_extensions(request_options, sealed_defaults);
  ASSERT_TRUE(active.ok());
  ASSERT_TRUE(sealed.ok());
  EXPECT_EQ(active.value().top_k, sealed.value().top_k);
  EXPECT_EQ(active.value().ef, sealed.value().ef);
  EXPECT_EQ(active.value().beam_width, sealed.value().beam_width);
  EXPECT_EQ(active.value().ef, parameters.effort);
  EXPECT_EQ(active.value().beam_width, parameters.beam_width);
  EXPECT_FALSE(active.value().exact_rerank);
  EXPECT_FALSE(sealed.value().exact_rerank);

  extension.algorithm_id = core::algorithm::qg;
  auto active_rejected =
      ::alaya::disk::resolve_laser_search_extensions(request_options, active_defaults);
  auto sealed_rejected =
      ::alaya::disk::resolve_laser_search_extensions(request_options, sealed_defaults);
  ASSERT_FALSE(active_rejected.ok());
  ASSERT_FALSE(sealed_rejected.ok());
  EXPECT_EQ(active_rejected.status().detail(), core::StatusDetail::unknown_extension);
  EXPECT_EQ(sealed_rejected.status().detail(), core::StatusDetail::unknown_extension);
}

TEST(MutableLaserAdapter, SearchExtensionChangesActiveExpansionEffort) {
  const auto dir = scratch("search_extensions");
  auto seg = open_empty(dir);
  constexpr size_t kRows = 256;
  std::vector<float> vectors(kRows * kDim);
  std::vector<std::uint64_t> labels(kRows);
  for (size_t row = 0; row < kRows; ++row) {
    labels[row] = 10'000 + row;
    for (size_t column = 0; column < kDim; ++column) {
      vectors[row * kDim + column] =
          static_cast<float>((row * 131 + column * 17 + row * column) % 1009) / 1009.0F;
    }
  }
  (void)seg->commit_physical_bundle(801, 801, vectors.data(), labels.data(), labels.size());
  MutableLaserCollectionAdapter adapter(seg, schema(), kSegId, kGen);

  const auto run = [&](std::uint32_t effort, std::uint32_t beam_width) {
    std::array<core::SearchHit, 1> hits{};
    std::array<core::RowCount, 2> offsets{};
    std::array<core::RowCount, 1> counts{};
    std::array<core::Status, 1> statuses{};
    std::array<core::SearchCompleteness, 1> completeness{};
    core::SearchResponse response;
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    ::alaya::disk::LaserSegmentSearchExtension parameters;
    parameters.effort = effort;
    parameters.beam_width = beam_width;
    auto extension = ::alaya::disk::make_laser_segment_search_extension(parameters);
    core::SearchContext context;
    core::SearchRequest request;
    request.queries = core::TypedTensorView::contiguous(vectors.data() + 173 * kDim, 1, kDim);
    request.options.top_k = 1;
    request.options.extensions =
        std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = &context;
    request.response = &response;
    const auto before = seg->search_stats().query_page_reads;
    const auto status = adapter.search(request);
    const auto after = seg->search_stats().query_page_reads;
    EXPECT_TRUE(status.ok()) << status.diagnostic();
    EXPECT_EQ(counts[0], 1U);
    return after - before;
  };

  const auto low_effort_expansions = run(/*effort=*/1, /*beam_width=*/1);
  const auto high_effort_expansions = run(/*effort=*/128, /*beam_width=*/16);
  EXPECT_GT(high_effort_expansions, low_effort_expansions)
      << "the active adapter must pass extension effort into QG traversal";
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::internal::collection::detail
