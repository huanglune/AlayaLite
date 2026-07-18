// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "core/any_segment.hpp"

namespace alaya::internal::collection {

// Gate 4 is deliberately internal. These owner-layer types give semantic
// meaning to the opaque operation payload frozen by contract v3; they are not
// a second engine boundary and are not bound into Python.

struct LogicalIdLess {
  [[nodiscard]] auto operator()(const core::LogicalId &lhs,
                                const core::LogicalId &rhs) const noexcept -> bool {
    return lhs.compare(rhs) < 0;
  }
};

using ScalarValue = std::variant<bool, std::int64_t, double, std::string>;
using Metadata = std::map<std::string, ScalarValue, std::less<>>;

class LogicalFilter {
 public:
  using Predicate =
      std::function<bool(const core::LogicalId &, const Metadata &, std::string_view)>;

  LogicalFilter() = default;
  explicit LogicalFilter(Predicate predicate, std::optional<double> selectivity_estimate = {})
      : predicate_(std::move(predicate)), selectivity_estimate_(selectivity_estimate) {}

  [[nodiscard]] static auto metadata_equals(std::string key, ScalarValue value) -> LogicalFilter {
    return LogicalFilter([key = std::move(key), value = std::move(value)](const core::LogicalId &,
                                                                          const Metadata &metadata,
                                                                          std::string_view) {
      const auto found = metadata.find(key);
      return found != metadata.end() && found->second == value;
    });
  }

  [[nodiscard]] auto active() const noexcept -> bool { return static_cast<bool>(predicate_); }

  [[nodiscard]] auto selectivity_estimate() const noexcept -> std::optional<double> {
    return selectivity_estimate_;
  }

  [[nodiscard]] auto matches(const core::LogicalId &id,
                             const Metadata &metadata,
                             std::string_view document) const -> bool {
    return !predicate_ || predicate_(id, metadata, document);
  }

 private:
  Predicate predicate_{};
  std::optional<double> selectivity_estimate_{};
};

struct CollectionSchema {
  std::uint32_t dim{};
  core::Metric metric{core::Metric::l2};
  core::ScalarType scalar_type{core::ScalarType::float32};
  std::uint64_t max_logical_id_bytes{64U * 1024U};
};

class OwnedVector {
 public:
  OwnedVector() = default;

  [[nodiscard]] static auto copy_row(const core::TypedTensorView &source, core::RowCount row)
      -> core::Result<OwnedVector> {
    if (row >= source.rows) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "vector row is outside the tensor view");
    }
    std::uint64_t row_bytes{};
    if (!core::checked_multiply(source.dim,
                                core::scalar_type_size(source.scalar_type),
                                row_bytes) ||
        row_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "vector row byte size is not representable");
    }
    OwnedVector result;
    result.scalar_type_ = source.scalar_type;
    result.dim_ = source.dim;
    result.bytes_.resize(static_cast<std::size_t>(row_bytes));
    if (row_bytes != 0) {
      const auto *source_bytes =
          static_cast<const std::byte *>(source.data) + row * source.row_stride;
      std::memcpy(result.bytes_.data(), source_bytes, result.bytes_.size());
    }
    return result;
  }

  [[nodiscard]] auto view() const noexcept -> core::TypedTensorView {
    return {bytes_.empty() ? nullptr : bytes_.data(),
            scalar_type_,
            1,
            dim_,
            static_cast<std::uint64_t>(bytes_.size())};
  }

  [[nodiscard]] auto empty() const noexcept -> bool { return bytes_.empty() && dim_ == 0; }
  [[nodiscard]] auto scalar_type() const noexcept -> core::ScalarType { return scalar_type_; }
  [[nodiscard]] auto dim() const noexcept -> std::uint32_t { return dim_; }
  [[nodiscard]] auto bytes() const noexcept -> std::span<const std::byte> { return bytes_; }

 private:
  core::ScalarType scalar_type_{core::ScalarType::float32};
  std::uint32_t dim_{};
  std::vector<std::byte> bytes_{};
};

struct RecordPayload {
  std::optional<OwnedVector> vector{};
  Metadata metadata{};
  std::string document{};
};

struct RowAddress {
  std::uint64_t segment_id{};
  std::uint64_t generation{};
  core::SegmentRowId row_id{};

  auto operator<=>(const RowAddress &) const = default;
};

enum class VersionState : std::uint8_t { live = 0, tombstone = 1 };

struct VersionEntry {
  RowAddress address{};
  std::uint64_t upsert_sequence{};
  VersionState state{VersionState::live};
  RecordPayload payload{};
};

struct ReverseEntry {
  core::LogicalId logical_id{};
  std::uint64_t upsert_sequence{};
};

enum class SegmentRole : std::uint8_t {
  sealed = 0,
  active_mutable = 1,
  legacy_readonly = 2,
};

using ExactRerank =
    std::function<core::Result<float>(const core::TypedTensorView &, core::SegmentRowId)>;

struct RegisteredRow {
  core::LogicalId logical_id{};
  core::SegmentRowId row_id{};
  std::uint64_t upsert_sequence{};
  VersionState state{VersionState::live};
  RecordPayload payload{};
};

struct SegmentMaintenanceHook {
  using ConsolidateFn = std::function<core::Status(std::size_t, std::size_t, bool, bool)>;

  ConsolidateFn consolidate{};
  std::function<bool()> recovery_required{};
};

struct SegmentRegistration {
  std::uint64_t segment_id{};
  std::uint64_t generation{1};
  SegmentRole role{SegmentRole::sealed};
  core::AnySegment segment{};
  std::vector<RegisteredRow> rows{};
  ExactRerank exact_rerank{};
  std::uint64_t next_row_id{};
  // Collection-internal capability. It does not grow the frozen AnySegment
  // table: an engine opts into receiving one opaque bundle through the
  // existing mutation token protocol.
  bool atomic_mutation_bundle{};
  SegmentMaintenanceHook maintenance{};
};

struct SegmentMaintenanceReceipt {
  std::uint64_t active_segment_id{};
  std::uint64_t active_generation{};
};

struct CollectionHit {
  core::LogicalId logical_id{};
  float score{};
  core::ScoreKind score_kind{core::ScoreKind::distance};
  core::Metric comparable_metric{core::Metric::l2};
  core::ResultFlag result_flags{core::ResultFlag::none};
  std::uint64_t upsert_sequence{};
  RowAddress source{};
};

struct CollectionQueryResult {
  std::vector<CollectionHit> hits{};
  core::Status status{};
  core::SearchCompleteness completeness{core::SearchCompleteness::eligible_exhausted};
};

struct CollectionSearchResult {
  std::uint64_t visibility_watermark{};
  std::uint64_t metadata_epoch{};
  std::vector<CollectionQueryResult> queries{};
};

struct CollectionSearchStats {
  core::VersionedStructHeader header{};
  core::FilterExecution filter_execution{core::FilterExecution::postfilter};
  bool filter_active{};
  std::uint8_t reserved_bytes[6]{};
  std::uint64_t filter_examined{};
  std::uint64_t filter_passed{};
  std::uint64_t nan_discarded{};
  // Number of segment re-queries after the first round.
  std::uint64_t overfetch_rounds{};
  std::uint64_t budget_consumed{};
  std::uint64_t lease_acquired{};
  std::uint64_t lease_released{};
  std::uint64_t lease_peak_bytes{};
  std::uint64_t io_requests_consumed{};
  std::uint64_t io_bytes_consumed{};
  std::uint64_t reserved[4]{};

  CollectionSearchStats() : header(core::current_struct_header<CollectionSearchStats>()) {}
};

struct CollectionSearchRequest {
  core::TypedTensorView queries{};
  core::SearchOptions options{};
  LogicalFilter filter{};
  core::SearchContext *context{};
  CollectionSearchStats *stats{};
  std::uint32_t maximum_overfetch_rounds{4};
};

enum class Projection : std::uint8_t {
  identity = 0,
  vector = 1U << 0U,
  metadata = 1U << 1U,
  document = 1U << 2U,
  all = (1U << 0U) | (1U << 1U) | (1U << 2U),
};

[[nodiscard]] constexpr auto projection_contains(Projection projection, Projection field) noexcept
    -> bool {
  return (static_cast<std::uint8_t>(projection) & static_cast<std::uint8_t>(field)) != 0;
}

struct CollectionRecord {
  core::LogicalId logical_id{};
  std::uint64_t upsert_sequence{};
  std::optional<OwnedVector> vector{};
  Metadata metadata{};
  std::string document{};
};

enum class WriteMode : std::uint8_t { insert_only = 0, upsert = 1, replace = 2 };

enum class WriteDurability : std::uint8_t {
  searchable = 0,
  wal_fsync = 1,
};

struct WriteOptions {
  WriteDurability durability{WriteDurability::wal_fsync};
  std::string retry_token{};
};

struct WriteRequest {
  core::LogicalId logical_id{};
  core::TypedTensorView vector{};
  Metadata metadata{};
  std::string document{};
  WriteMode mode{WriteMode::upsert};
  WriteOptions options{};
};

enum class SegmentMutationAction : std::uint8_t { write = 0, erase = 1 };

struct SegmentMutationPayload {
  core::VersionedStructHeader header{};
  SegmentMutationAction action{SegmentMutationAction::write};
  std::uint8_t reserved_bytes[7]{};
  std::uint64_t op_id{};
  std::uint64_t upsert_sequence{};
  RowAddress target{};
  std::optional<RowAddress> previous{};
  core::TypedTensorView vector{};
  std::uint64_t reserved[4]{};

  SegmentMutationPayload() : header(core::current_struct_header<SegmentMutationPayload>()) {}
};

struct SegmentMutationBundlePayload {
  core::VersionedStructHeader header{};
  std::uint64_t batch_op_id{};
  std::span<const SegmentMutationPayload> rows{};
  std::uint64_t reserved[4]{};

  SegmentMutationBundlePayload()
      : header(core::current_struct_header<SegmentMutationBundlePayload>()) {}
};

enum class RowMutationStatus : std::uint8_t {
  inserted = 0,
  updated = 1,
  replaced = 2,
  deleted = 3,
  already_exists = 4,
  not_found = 5,
  conflict = 6,
  invalid_argument = 7,
  aborted = 8,
};

enum class DurabilityState : std::uint8_t {
  memory_only = 0,
  searchable_not_durable = 1,
  wal_fsync = 2,
};

struct MutationReceipt {
  // op_id remains the row-operation compatibility spelling.
  std::uint64_t op_id{};
  std::uint64_t batch_op_id{};
  std::uint64_t row_op_id{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  bool searchable{};
  DurabilityState durability{DurabilityState::memory_only};
  RowMutationStatus row_status{RowMutationStatus::aborted};
  std::string retry_token{};
};

enum class BatchMutationMode : std::uint8_t {
  per_row_independent = 0,
  all_or_nothing = 1,
};

enum class RowMutationAction : std::uint8_t { write = 0, erase = 1 };

struct BatchRowMutation {
  RowMutationAction action{RowMutationAction::write};
  core::LogicalId logical_id{};
  core::TypedTensorView vector{};
  Metadata metadata{};
  std::string document{};
  WriteMode write_mode{WriteMode::upsert};
  std::string retry_token{};
};

struct BatchMutationRequest {
  std::span<const BatchRowMutation> rows{};
  BatchMutationMode mode{BatchMutationMode::per_row_independent};
  WriteOptions options{};
};

struct BatchMutationReceipt {
  std::uint64_t batch_op_id{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  bool searchable{};
  DurabilityState durability{DurabilityState::memory_only};
  std::string retry_token{};
  std::vector<MutationReceipt> rows{};
};

enum class MutationFailPoint : std::uint8_t {
  none = 0,
  before_prepare = 1,
  after_prepare = 2,
  after_stage = 3,
  after_commit = 4,
  after_publish = 5,
  metadata_stage_failure = 6,
  // B-03/B-11 (2B): the C6 window -- the active engine's physical publish has
  // succeeded but the Collection's logical routing snapshot has NOT been swapped.
  after_engine_publish_before_snapshot = 7,
};

struct CheckpointReceipt {
  std::uint64_t durable_watermark{};
  std::uint64_t wal_cut{};
  std::uint64_t metadata_epoch{};
  std::string checkpoint_name{};
};

struct ActiveRotationReceipt {
  std::uint64_t source_segment_id{};
  std::uint64_t source_generation{};
  std::uint64_t successor_segment_id{};
  std::uint64_t successor_generation{};
  CheckpointReceipt checkpoint{};
};

struct SegmentReplacement {
  core::LogicalId logical_id{};
  RowAddress source{};
  RowAddress target{};
  std::uint64_t upsert_sequence{};
};

struct CollectionFeatureFlags {
  bool collection_shell{true};
  bool experimental_persistence_writer{};
  bool wal_coordinator{};
  // Roll-forward gate: disabling this bit prevents new manifest-v2
  // publications but never removes the v2 reader.
  bool manifest_v2_writer{};
};

struct PersistenceOptions {
  std::filesystem::path root{};
  std::string namespace_name{"collection_shell_v1"};
};

struct WalPersistenceOptions {
  std::filesystem::path root{};
  std::string namespace_name{"collection_wal_v1"};
};

struct CollectionRecoveryOptions {
  // G7-B may preserve a source op-id range by seeding this lower bound. It is
  // accepted only as a monotonic floor; WAL/checkpoint/registered versions can
  // always advance it further.
  std::uint64_t minimum_next_op_id{1};
  // A legacy checkpoint can cover a committed delete without retaining the
  // deleted external ID. The importer uses this floor to preserve that
  // committed visibility cut without inventing a logical row/version.
  std::uint64_t minimum_visibility_watermark{};
};

struct CollectionConfig {
  CollectionFeatureFlags features{};
  PersistenceOptions persistence{};
  WalPersistenceOptions wal{};
  CollectionRecoveryOptions recovery{};
  MutationFailPoint fail_point{MutationFailPoint::none};
  std::function<void(MutationFailPoint)> failpoint_hook{};
};

enum class LifecycleState : std::uint8_t { open = 0, closing = 1, closed = 2 };

struct CollectionStats {
  core::VersionedStructHeader header{};
  core::RowCount size{};
  // Current accepted logical rows: searchable live IDs plus admitted inserts
  // still in dark stage. Historical versions and delete operations are not
  // counted; pending_count independently tracks every in-flight mutation.
  core::RowCount accepted_count{};
  core::RowCount pending_count{};
  std::uint64_t pending_bytes{};
  core::RowCount allocated_count{};
  core::RowCount tombstone_count{};
  std::uint64_t routing_generation{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{};
  LifecycleState lifecycle{LifecycleState::open};

  CollectionStats() : header(core::current_struct_header<CollectionStats>()) {}
};

}  // namespace alaya::internal::collection
