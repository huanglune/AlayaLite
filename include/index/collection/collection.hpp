// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/collection/detail/canonical_flat_segment.hpp"
#include "index/collection/detail/collection_segment_factory.hpp"
#include "index/collection/detail/collection_target_builder.hpp"
// The active (writable) LASER stack -- MutableLaserSegment/QGUpdater -- is
// Linux-only: it needs flock, O_DIRECT, libaio and sync_file_range. Sealed
// LASER segments stay available on every platform ALAYA_ENABLE_LASER covers;
// only active_engine=laser is gated to Linux builds.
#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0 && defined(__linux__)
  #define ALAYA_COLLECTION_HAS_ACTIVE_LASER 1
  #include "index/collection/detail/mutable_laser_collection_adapter.hpp"
#else
  #define ALAYA_COLLECTION_HAS_ACTIVE_LASER 0
#endif
#include "index/collection/segmented_collection.hpp"
#include "index/collection/sha256.hpp"
#include "platform/fs.hpp"

namespace alaya {

namespace internal::collection {
class CollectionTestAccess;
}  // namespace internal::collection

inline constexpr std::string_view kCollectionPublicVersion{"1.1.0"};
inline constexpr std::string_view kCollectionLegacyRemovalVersion{"1.1.0"};

enum class CollectionQuantization : std::uint8_t {
  none = 0,
  sq8 = 1,
  sq4 = 2,
  rabitq = 3,
};

using CollectionScalarValue = internal::collection::ScalarValue;
using CollectionMetadata = internal::collection::Metadata;
using CollectionWriteOptions = internal::collection::WriteOptions;
using CollectionWriteDurability = internal::collection::WriteDurability;
using CollectionMutationReceipt = internal::collection::MutationReceipt;
using CollectionBatchMutationReceipt = internal::collection::BatchMutationReceipt;
using CollectionBatchMutationMode = internal::collection::BatchMutationMode;
using CollectionRowMutationStatus = internal::collection::RowMutationStatus;
using CollectionDurabilityState = internal::collection::DurabilityState;
using CollectionCheckpointReceipt = internal::collection::CheckpointReceipt;
using CollectionProjection = internal::collection::Projection;
using CollectionRecord = internal::collection::CollectionRecord;
using CollectionFilter = internal::collection::LogicalFilter;
using CollectionSearchStatistics = internal::collection::CollectionSearchStats;

struct CollectionOptions {
  std::filesystem::path root{};
  std::uint32_t dim{};
  core::Metric metric{core::Metric::l2};
  core::ScalarType scalar_type{core::ScalarType::float32};
  core::AlgorithmId target_algorithm{core::algorithm::qg};
  // 2B: the active (writable) generation engine. flat = the in-memory exact table
  // (default, byte-compatible with pre-2B); laser = the durable on-disk mutable
  // RaBitQ graph (persisted in the facade schema; reopen restores it).
  core::AlgorithmId active_engine{core::algorithm::flat};
  CollectionQuantization quantization{CollectionQuantization::none};
  std::uint32_t build_threads{1};
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{400};
  std::uint64_t max_logical_id_bytes{64U * 1024U};
  // Zero disables automatic rotation. A positive value rotates after the
  // active generation reaches this many physical rows.
  std::uint64_t auto_seal_rows{};
};

enum class CollectionSealFailPoint : std::uint8_t {
  none = 0,
  after_cut_before_successor = 1,
  after_successor_switch = 2,
  during_export_build = 3,
  after_manifest_publish = 4,
  after_active_control_publish_before_routing_install = 5,
};

struct CollectionSealOptions {
  CollectionSealFailPoint fail_point{CollectionSealFailPoint::none};
  std::function<void(CollectionSealFailPoint)> failpoint_hook{};
};

struct CollectionConsolidateOptions {
  std::uint32_t num_threads{1};
  std::uint32_t r_target{0};
  bool reclaim_slots{true};
  bool bloom_consolidate{false};
};

struct CollectionConsolidateReceipt {
  std::uint64_t active_segment_id{};
  std::uint64_t active_generation{};
};

struct CollectionSealReceipt {
  std::uint64_t source_segment_id{};
  std::uint64_t successor_segment_id{};
  std::uint64_t sealed_segment_id{};
  std::uint64_t wal_cut{};
  core::RowCount sealed_rows{};
  std::uint64_t sealed_bytes{};
  std::uint64_t manifest_generation{};
  core::AlgorithmId built_algorithm{core::algorithm::flat};
  std::uint32_t effective_ef_construction{};
  bool flat_fallback{};
  std::string fallback_reason{};
};

// Handle returned by Collection::prepare_successor() once a successor
// segment has been built and durably published to the manifest (source
// segment(s) marked gc_pending, successor marked sealed) but not yet routed
// for queries. Pass it to Collection::rotate_to_successor() to atomically
// switch query traffic over to the successor and retire the predecessor.
//
// This is the "successor is ready" precondition from the rotate-to-successor
// design: everything upstream of a ready handle (how/when a successor gets
// built) is out of scope here. seal() composes prepare_successor() +
// rotate_to_successor() under one control_mutex_ hold, so its externally
// observable behavior is unchanged by this split.
struct CollectionRotationHandle {
  std::vector<std::uint64_t> predecessor_segment_ids{};
  std::uint64_t successor_segment_id{};
  std::uint64_t successor_generation{};

  [[nodiscard]] auto ready() const noexcept -> bool { return successor_segment_id != 0; }
};

struct CollectionCompactReceipt {
  std::vector<std::uint64_t> source_segment_ids{};
  std::uint64_t compacted_segment_id{};
  core::RowCount compacted_rows{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
  std::uint64_t manifest_generation{};
  core::AlgorithmId built_algorithm{core::algorithm::flat};
  std::uint32_t effective_ef_construction{};
  bool flat_fallback{};
  std::string fallback_reason{};
};

struct CollectionGcReceipt {
  core::RowCount pending{};
  core::RowCount reclaimed{};
  core::RowCount deferred{};
  std::uint64_t reclaimed_bytes{};
  std::uint64_t manifest_generation{};
};

struct CollectionItem {
  core::LogicalId logical_id{};
  core::TypedTensorView vector{};
  CollectionMetadata metadata{};
  std::string document{};
  std::string retry_token{};
};

enum class CollectionMutationAction : std::uint8_t {
  add = 0,
  upsert = 1,
  replace = 2,
  remove = 3,
};

struct CollectionBatchRow {
  CollectionMutationAction action{CollectionMutationAction::upsert};
  core::LogicalId logical_id{};
  core::TypedTensorView vector{};
  CollectionMetadata metadata{};
  std::string document{};
  std::string retry_token{};
};

struct CollectionSearchResponse {
  std::uint64_t visibility_watermark{};
  std::uint64_t metadata_epoch{};
  std::vector<core::LogicalId> ids{};
  std::vector<float> distances{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> valid_counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  CollectionSearchStatistics search_stats{};
};

struct CollectionStatistics {
  core::VersionedStructHeader header{};
  core::RowCount size{};
  core::RowCount accepted_count{};
  core::RowCount pending_count{};
  std::uint64_t searchable_bytes{};
  std::uint64_t accepted_bytes{};
  std::uint64_t searchable_vector_bytes{};
  std::uint64_t accepted_vector_bytes{};
  std::uint64_t pending_bytes{};
  core::RowCount allocated_count{};
  core::RowCount tombstone_count{};
  std::uint64_t routing_generation{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{};
  core::RowCount sealed_segments_count{};
  core::RowCount gc_pending_count{};
  core::AlgorithmId active_segment_algorithm{core::algorithm::flat};
  std::uint64_t compacted_bytes{};
  internal::collection::LifecycleState lifecycle{internal::collection::LifecycleState::open};

  CollectionStatistics() : header(core::current_struct_header<CollectionStatistics>()) {}
};

class Collection {
 public:
  Collection(const Collection &) = delete;
  auto operator=(const Collection &) -> Collection & = delete;
  Collection(Collection &&) = delete;
  auto operator=(Collection &&) -> Collection & = delete;

  [[nodiscard]] static auto create(CollectionOptions options)
      -> core::Result<std::shared_ptr<Collection>> {
    auto status = validate_options(options, core::OperationStage::build);
    if (!status.ok()) {
      return status;
    }
    try {
      const auto schema_path = facade_schema_path(options.root);
      if (std::filesystem::exists(schema_path) ||
          std::filesystem::exists(options.root / "schema.json") ||
          std::filesystem::exists(options.root / ".alaya_internal" /
                                  internal::collection::kCollectionWalNamespace)) {
        return error(core::StatusCode::conflict,
                     core::OperationStage::build,
                     core::StatusDetail::already_exists,
                     "canonical Collection target already contains a collection layout");
      }
      std::filesystem::create_directories(options.root);
      status = write_facade_schema(options);
      if (!status.ok()) {
        return status;
      }
      internal::collection::CollectionControlState state;
      state.auto_seal_rows = options.auto_seal_rows;
      status = internal::collection::CollectionControlStore::save(options.root, state);
      if (!status.ok()) {
        return status;
      }
      // 2B: materialize the empty active LASER segment directory before open so
      // make_active_registration can open it (create never re-creates on reopen).
      if (options.active_engine == core::algorithm::laser) {
        status = create_active_laser_segment(options, kActiveSegmentId, kActiveSegmentGeneration);
        if (!status.ok()) {
          return status;
        }
      }
      auto opened = open_segmented(options, state);
      if (!opened.ok()) {
        return opened.status();
      }
      auto result = std::shared_ptr<Collection>(
          new Collection(std::move(options), std::move(opened).value(), std::move(state)));
      core::CheckpointContext context;
      context.durability_target = core::DurabilityTarget::full_checkpoint;
      auto checkpoint = result->checkpoint(context);
      if (!checkpoint.ok()) {
        return checkpoint.status();
      }
      return result;
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] static auto open(const std::filesystem::path &root)
      -> core::Result<std::shared_ptr<Collection>> {
    if (root.empty()) {
      return error(core::StatusCode::invalid_argument,
                   core::OperationStage::open,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection root is empty");
    }
    try {
      if (std::filesystem::is_regular_file(facade_schema_path(root))) {
        auto options = read_facade_schema(root);
        if (!options.ok()) {
          return options.status();
        }
        if (internal::collection::CollectionControlStore::exists(root)) {
          auto loaded_state = internal::collection::CollectionControlStore::load(root);
          if (!loaded_state.ok()) {
            return loaded_state.status();
          }
          auto state = std::move(loaded_state).value();
          options.value().auto_seal_rows = state.auto_seal_rows;
          auto status = validate_options(options.value(), core::OperationStage::open);
          if (!status.ok()) {
            return status;
          }
          status = normalize_control_state_before_open(root, state);
          if (!status.ok()) {
            return status;
          }
          auto opened = open_segmented(options.value(), state);
          if (!opened.ok()) {
            return opened.status();
          }
          auto result = std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                                   std::move(opened).value(),
                                                                   std::move(state)));
          status = result->recover_control_state();
          if (!status.ok()) {
            return status;
          }
          return result;
        }
        internal::collection::CollectionControlState state;
        state.auto_seal_rows = options.value().auto_seal_rows;
        auto opened = open_segmented(options.value(), state);
        if (!opened.ok()) {
          return opened.status();
        }
        auto status = internal::collection::CollectionControlStore::save(root, state);
        if (!status.ok()) {
          return status;
        }
        return std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                          std::move(opened).value(),
                                                          std::move(state)));
      }

      return error(core::StatusCode::not_found,
                   core::OperationStage::open,
                   core::StatusDetail::none,
                   "no canonical Collection layout found at this path");
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] auto add(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt> {
    options.retry_token =
        item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
    return write(item, internal::collection::WriteMode::insert_only, std::move(options));
  }

  [[nodiscard]] auto upsert(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt> {
    options.retry_token =
        item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
    return write(item, internal::collection::WriteMode::upsert, std::move(options));
  }

  [[nodiscard]] auto replace(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt> {
    options.retry_token =
        item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
    return write(item, internal::collection::WriteMode::replace, std::move(options));
  }

  [[nodiscard]] auto remove(const core::LogicalId &logical_id, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt> {
    core::MutationContext context;
    return implementation_->erase(logical_id, context, std::move(options));
  }

  [[nodiscard]] auto mutate_batch(
      std::span<const CollectionBatchRow> rows,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt> {
    std::vector<internal::collection::BatchRowMutation> native_rows;
    native_rows.reserve(rows.size());
    for (const auto &row : rows) {
      internal::collection::BatchRowMutation native;
      native.logical_id = row.logical_id;
      native.vector = row.vector;
      native.metadata = row.metadata;
      native.document = row.document;
      native.retry_token = row.retry_token;
      switch (row.action) {
        case CollectionMutationAction::add:
          native.action = internal::collection::RowMutationAction::write;
          native.write_mode = internal::collection::WriteMode::insert_only;
          break;
        case CollectionMutationAction::upsert:
          native.action = internal::collection::RowMutationAction::write;
          native.write_mode = internal::collection::WriteMode::upsert;
          break;
        case CollectionMutationAction::replace:
          native.action = internal::collection::RowMutationAction::write;
          native.write_mode = internal::collection::WriteMode::replace;
          break;
        case CollectionMutationAction::remove:
          native.action = internal::collection::RowMutationAction::erase;
          native.write_mode = internal::collection::WriteMode::upsert;
          break;
      }
      native_rows.push_back(std::move(native));
    }
    internal::collection::BatchMutationRequest request;
    request.rows = native_rows;
    request.mode = mode;
    request.options = std::move(options);
    core::MutationContext context;
    auto receipt = implementation_->mutate_batch(request, context);
    if (receipt.ok()) {
      maybe_auto_seal();
    }
    return receipt;
  }

  [[nodiscard]] auto add_batch(
      std::span<const CollectionItem> items,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt> {
    std::vector<CollectionBatchRow> rows;
    rows.reserve(items.size());
    for (const auto &item : items) {
      rows.push_back(CollectionBatchRow{CollectionMutationAction::add,
                                        item.logical_id,
                                        item.vector,
                                        item.metadata,
                                        item.document,
                                        item.retry_token});
    }
    return mutate_batch(rows, mode, std::move(options));
  }

  [[nodiscard]] auto upsert_batch(
      std::span<const CollectionItem> items,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt> {
    std::vector<CollectionBatchRow> rows;
    rows.reserve(items.size());
    for (const auto &item : items) {
      rows.push_back(CollectionBatchRow{CollectionMutationAction::upsert,
                                        item.logical_id,
                                        item.vector,
                                        item.metadata,
                                        item.document,
                                        item.retry_token});
    }
    return mutate_batch(rows, mode, std::move(options));
  }

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            const core::SearchOptions &options,
                            core::SearchContext &context)
      -> core::Result<CollectionSearchResponse> {
    return search(query, options, context, CollectionFilter{});
  }

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            const core::SearchOptions &options,
                            core::SearchContext &context,
                            const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse> {
    if (query.rows != 1) {
      return error(core::StatusCode::invalid_argument,
                   core::OperationStage::validation,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection single search requires exactly one query row");
    }
    return execute_search(query, options, context, filter);
  }

  [[nodiscard]] auto search(const core::TypedTensorView &query, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    core::SearchContext context;
    return search(query, options, context);
  }

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            std::uint64_t top_k,
                            const CollectionFilter &filter,
                            core::FilterPolicy policy = core::FilterPolicy::automatic)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    options.filter_policy = policy;
    core::SearchContext context;
    return search(query, options, context, filter);
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  const core::SearchOptions &options,
                                  core::SearchContext &context)
      -> core::Result<CollectionSearchResponse> {
    return batch_search(queries, options, context, CollectionFilter{});
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  const core::SearchOptions &options,
                                  core::SearchContext &context,
                                  const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse> {
    return execute_search(queries, options, context, filter);
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    core::SearchContext context;
    return batch_search(queries, options, context);
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  std::uint64_t top_k,
                                  const CollectionFilter &filter,
                                  core::FilterPolicy policy = core::FilterPolicy::automatic)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    options.filter_policy = policy;
    core::SearchContext context;
    return batch_search(queries, options, context, filter);
  }

  [[nodiscard]] auto get_by_id(const core::LogicalId &logical_id,
                               CollectionProjection projection = CollectionProjection::all)
      -> core::Result<CollectionRecord> {
    return implementation_->get_by_id(logical_id, projection);
  }

  [[nodiscard]] auto records(CollectionProjection projection = CollectionProjection::all,
                             std::size_t limit = std::numeric_limits<std::size_t>::max())
      -> core::Result<std::vector<CollectionRecord>> {
    return implementation_->scalar_query(internal::collection::LogicalFilter{}, limit, projection);
  }

  [[nodiscard]] auto checkpoint(core::CheckpointContext &context)
      -> core::Result<CollectionCheckpointReceipt> {
    std::lock_guard lock(control_mutex_);
    return checkpoint_locked(context);
  }

  [[nodiscard]] auto checkpoint() -> core::Result<CollectionCheckpointReceipt> {
    core::CheckpointContext context;
    context.durability_target = core::DurabilityTarget::full_checkpoint;
    return checkpoint(context);
  }

  [[nodiscard]] auto consolidate(CollectionConsolidateOptions options = {})
      -> core::Result<CollectionConsolidateReceipt> {
    std::lock_guard lock(control_mutex_);
    auto receipt = implementation_->consolidate(options.num_threads,
                                                options.r_target,
                                                options.reclaim_slots,
                                                options.bloom_consolidate);
    if (!receipt.ok()) {
      return receipt.status();
    }
    return CollectionConsolidateReceipt{receipt.value().active_segment_id,
                                        receipt.value().active_generation};
  }

  [[nodiscard]] auto seal(CollectionSealOptions options = {})
      -> core::Result<CollectionSealReceipt> {
    core::SealContext context;
    return seal(context, std::move(options));
  }

  [[nodiscard]] auto seal(core::SealContext &context, CollectionSealOptions options = {})
      -> core::Result<CollectionSealReceipt> {
    std::lock_guard lock(control_mutex_);
    return seal_locked(context, options);
  }

  // Explicit two-phase rotate-to-successor primitive that seal() is built
  // from. prepare_successor() performs everything up to and including the
  // durable manifest publish (source(s) -> gc_pending, successor ->
  // sealed); the routing table still serves the predecessor at this point.
  // rotate_to_successor() then atomically switches query routing to the
  // successor and retires the predecessor (deferred reclaim via gc(), same
  // as seal()/compact()). Splitting the call in two lets a successor be
  // prepared out of band (see prepare_successor_locked()'s TODO) while
  // keeping the atomic-switch/drain/reclaim step a single, independently
  // testable operation.
  //
  // Distinct from the internal SegmentedCollection::rotate_to_successor()
  // primitive used inside prepare_successor(): that one hands off *write*
  // authority to a new empty active segment before the successor is even
  // built; this one hands off *read* routing once the successor is ready.
  [[nodiscard]] auto prepare_successor(CollectionSealOptions options = {})
      -> core::Result<CollectionRotationHandle> {
    core::SealContext context;
    return prepare_successor(context, std::move(options));
  }

  [[nodiscard]] auto prepare_successor(core::SealContext &context,
                                       CollectionSealOptions options = {})
      -> core::Result<CollectionRotationHandle> {
    std::lock_guard lock(control_mutex_);
    return prepare_successor_locked(context, options);
  }

  [[nodiscard]] auto rotate_to_successor(const CollectionRotationHandle &handle)
      -> core::Result<CollectionSealReceipt> {
    core::SealContext context;
    return rotate_to_successor(handle, context);
  }

  [[nodiscard]] auto rotate_to_successor(const CollectionRotationHandle &handle,
                                         core::SealContext &context)
      -> core::Result<CollectionSealReceipt> {
    std::lock_guard lock(control_mutex_);
    return rotate_to_successor_locked(handle, context);
  }

  [[nodiscard]] auto compact() -> core::Result<CollectionCompactReceipt> {
    core::SealContext context;
    return compact(context);
  }

  [[nodiscard]] auto compact(core::SealContext &context) -> core::Result<CollectionCompactReceipt> {
    std::lock_guard lock(control_mutex_);
    return compact_locked(context);
  }

  [[nodiscard]] auto gc() -> core::Result<CollectionGcReceipt> {
    std::lock_guard lock(control_mutex_);
    return gc_locked();
  }

  [[nodiscard]] auto stats() const -> CollectionStatistics {
    const auto native = implementation_->stats();
    CollectionStatistics result;
    result.size = native.size;
    result.accepted_count = native.accepted_count;
    result.pending_count = native.pending_count;
    result.pending_bytes = native.pending_bytes;
    result.allocated_count = native.allocated_count;
    result.tombstone_count = native.tombstone_count;
    result.routing_generation = native.routing_generation;
    result.visibility_watermark = native.visibility_watermark;
    result.durable_watermark = native.durable_watermark;
    result.metadata_epoch = native.metadata_epoch;
    result.lifecycle = native.lifecycle;
    std::uint64_t row_bytes{};
    if (core::checked_multiply(options_.dim,
                               core::scalar_type_size(options_.scalar_type),
                               row_bytes) &&
        core::checked_multiply(result.size, row_bytes, result.searchable_vector_bytes)) {
      if (!core::checked_add(result.searchable_vector_bytes,
                             result.pending_bytes,
                             result.accepted_vector_bytes)) {
        result.accepted_vector_bytes = std::numeric_limits<std::uint64_t>::max();
      }
    } else {
      result.searchable_vector_bytes = std::numeric_limits<std::uint64_t>::max();
      result.accepted_vector_bytes = std::numeric_limits<std::uint64_t>::max();
    }
    result.searchable_bytes = result.searchable_vector_bytes;
    result.accepted_bytes = result.accepted_vector_bytes;
    result.active_segment_algorithm = core::algorithm::flat;
    {
      std::lock_guard lock(control_mutex_);
      result.compacted_bytes = control_state_.compacted_bytes;
      const auto manifest = internal::collection::load_manifest_v2_if_present(options_.root);
      if (manifest.ok() && manifest.value().has_value()) {
        for (const auto &entry : manifest.value()->segments) {
          if (entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed) {
            ++result.sealed_segments_count;
          }
        }
        result.gc_pending_count = manifest.value()->gc.pending_segment_ids.size();
      }
    }
    return result;
  }

  [[nodiscard]] auto size() const -> core::RowCount { return stats().size; }
  [[nodiscard]] auto options() const -> const CollectionOptions & { return options_; }
  [[nodiscard]] auto root() const -> const std::filesystem::path & { return options_.root; }
  [[nodiscard]] auto target_algorithm() const noexcept -> core::AlgorithmId {
    return options_.target_algorithm;
  }
  [[nodiscard]] auto active_algorithm() const noexcept -> core::AlgorithmId {
    return options_.active_engine;
  }
  [[nodiscard]] auto target_implementation_key() const -> std::string_view {
    const auto *registration = internal::collection::detail::find_collection_target_registration(
        options_.target_algorithm);
    return registration == nullptr ? std::string_view{"unknown"} : registration->implementation_key;
  }
  [[nodiscard]] auto target_engine_factory_key() const -> std::string_view {
    const auto *registration = internal::collection::detail::find_collection_target_registration(
        options_.target_algorithm);
    return registration == nullptr ? std::string_view{"unknown"} : registration->factory_key;
  }

  [[nodiscard]] auto close() -> core::Status {
    auto status = implementation_->close();
    if (!status.ok()) {
      return status;
    }
    return implementation_->drain();
  }

 private:
  friend class internal::collection::CollectionTestAccess;

  inline static constexpr std::uint64_t kActiveSegmentId = 2;
  inline static constexpr std::uint64_t kActiveSegmentGeneration = 1;
  inline static constexpr std::string_view kFacadeNamespace{"collection_facade_v1"};
  inline static constexpr std::string_view kFacadeSchemaFilename{"schema.v1"};

  Collection(CollectionOptions options,
             std::shared_ptr<internal::collection::SegmentedCollection> implementation,
             internal::collection::CollectionControlState control_state)
      : options_(std::move(options)),
        implementation_(std::move(implementation)),
        control_state_(std::move(control_state)) {}

  [[nodiscard]] static auto error(core::StatusCode code,
                                  core::OperationStage stage,
                                  core::StatusDetail detail,
                                  std::string diagnostic) -> core::Status {
    return core::Status::error(code, stage, detail, std::move(diagnostic));
  }

  struct BuildAlgorithmResolution {
    core::AlgorithmId algorithm{core::algorithm::flat};
    bool flat_fallback{};
    std::string fallback_reason{};
  };

  [[nodiscard]] static auto resolve_build_algorithm(
      core::AlgorithmId requested_algorithm,
      const internal::collection::CollectionSchema &schema,
      core::RowCount live_row_count,
      const internal::collection::detail::CollectionTargetBuildParams &params)
      -> BuildAlgorithmResolution {
    const auto *registration =
        internal::collection::detail::find_collection_target_registration(requested_algorithm);
    if (registration != nullptr && registration->supports(schema, live_row_count, params) ==
                                       internal::collection::detail::TargetSupport::supported) {
      return {requested_algorithm, false, {}};
    }

    BuildAlgorithmResolution resolution;
    resolution.algorithm = core::algorithm::flat;
    resolution.flat_fallback = true;
    if (registration == nullptr) {
      resolution.fallback_reason = "requested Collection target algorithm " +
                                   std::to_string(requested_algorithm) +
                                   " has no registered sealed builder; built Flat instead";
    } else if (requested_algorithm == core::algorithm::qg && live_row_count <= 32) {
      resolution.fallback_reason = "qg requires >32 live rows; built Flat instead";
    } else if (requested_algorithm == core::algorithm::qg &&
               schema.scalar_type != core::ScalarType::float32) {
      resolution.fallback_reason = "qg requires float32 vectors; built Flat instead";
    } else if (requested_algorithm == core::algorithm::laser && live_row_count <= 32) {
      resolution.fallback_reason = "laser requires >32 live rows; built Flat instead";
    } else if (requested_algorithm == core::algorithm::laser &&
               !::alaya::disk::laser_importer_detail::dimension_supported_v1(schema.dim)) {
      resolution.fallback_reason =
          "laser requires dim in [33, 2048]; non-power-of-two dims use FHT padding; built Flat "
          "instead";
    } else {
      resolution.fallback_reason = "requested Collection target '" +
                                   std::string(registration->factory_key) +
                                   "' is unsupported for this schema or live row count; built "
                                   "Flat instead";
    }
    return resolution;
  }

  [[nodiscard]] static auto validate_options(const CollectionOptions &options,
                                             core::OperationStage stage) -> core::Status {
    if (options.root.empty() || options.dim == 0 || options.max_logical_id_bytes == 0 ||
        core::scalar_type_size(options.scalar_type) == 0 || options.build_threads == 0 ||
        options.max_neighbors == 0 || options.ef_construction == 0) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection schema/root/build parameters are invalid");
    }
    if (!checked_active_laser_capacity(options.auto_seal_rows).has_value()) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::arithmetic_overflow,
                   "canonical Collection auto_seal_rows exceeds the active LASER capacity limit");
    }
    const auto metric_valid = options.metric == core::Metric::l2 ||
                              options.metric == core::Metric::inner_product ||
                              options.metric == core::Metric::cosine;
    const auto quantization_valid = options.quantization == CollectionQuantization::none ||
                                    options.quantization == CollectionQuantization::sq8 ||
                                    options.quantization == CollectionQuantization::sq4 ||
                                    options.quantization == CollectionQuantization::rabitq;
    if (!metric_valid || !quantization_valid) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection metric/quantization schema is invalid");
    }
    const auto algorithm_valid = options.target_algorithm == core::algorithm::flat ||
                                 options.target_algorithm == core::algorithm::qg ||
                                 options.target_algorithm == core::algorithm::laser;
    if (!algorithm_valid) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::operation_slot_absent,
                   "canonical Collection target algorithm is unsupported");
    }
    // rabitq quantization is the RaBitQ-graph family's native (and only)
    // format -- both qg (in-memory) and laser (on-disk) always quantize this
    // way internally, so this cross-check covers both instead of qg alone.
    if (options.quantization == CollectionQuantization::rabitq &&
        options.target_algorithm != core::algorithm::qg &&
        options.target_algorithm != core::algorithm::laser) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection requires explicit index_type=qg or laser for rabitq");
    }
    if ((options.target_algorithm == core::algorithm::qg ||
         options.target_algorithm == core::algorithm::laser) &&
        options.quantization != CollectionQuantization::rabitq) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection qg/laser requires quantization=rabitq");
    }
    if ((options.quantization == CollectionQuantization::sq8 ||
         options.quantization == CollectionQuantization::sq4) &&
        options.scalar_type != core::ScalarType::float32) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::unsupported_scalar_type,
                   "canonical Collection quantization requires float32 vectors");
    }
    // 2B active engine (ruling 7 / B-08). Default flat is always valid. The on-disk
    // mutable LASER active engine constrains the schema hard: L2 + float32 + rabitq +
    // dim in [33,2048] (non-power-of-two dims use the same FHT padding as sealed
    // LASER) + FastScan-legal max_neighbors in {32,64} (32-slot block packing,
    // QGScanner cap 64) + target_algorithm=laser (v1 single engine).
    if (options.active_engine != core::algorithm::flat &&
        options.active_engine != core::algorithm::laser) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::operation_slot_absent,
                   "canonical Collection active engine is unsupported");
    }
#if !ALAYA_COLLECTION_HAS_ACTIVE_LASER
    // Capability admission must precede every filesystem mutation in create():
    // rejecting only inside create_active_laser_segment() would leave a persisted
    // schema/control layout that later blocks a flat retry with already_exists.
    if (options.active_engine == core::algorithm::laser) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::operation_slot_absent,
                   "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
    }
#endif
    if (options.active_engine == core::algorithm::laser) {
      const bool dimension_supported =
          ::alaya::disk::laser_importer_detail::dimension_supported_v1(options.dim);
      if (options.metric != core::Metric::l2 || options.scalar_type != core::ScalarType::float32 ||
          options.quantization != CollectionQuantization::rabitq || !dimension_supported ||
          (options.max_neighbors != 32 && options.max_neighbors != 64) ||
          options.target_algorithm != core::algorithm::laser) {
        return error(core::StatusCode::invalid_argument,
                     stage,
                     core::StatusDetail::malformed_struct,
                     "active LASER engine requires metric=l2, float32, quantization=rabitq, "
                     "dim in [33,2048] (non-power-of-two dims use FHT padding), max_neighbors "
                     "in {32,64}, target_algorithm=laser");
      }
    }
    return core::Status::success();
  }

  static auto active_laser_dir(const std::filesystem::path &root,
                               std::uint64_t segment_id,
                               std::uint64_t generation) -> std::filesystem::path {
    return root / ".alaya_internal" / "active_laser" /
           (internal::collection::detail::collection_segment_name(segment_id) + "_g" +
            std::to_string(generation));
  }

  // B-09 orphan reclamation. The durable control state is part of the reachability
  // root: successor-active/building/manifest-published recovery reopens every source
  // before completing replacement. Keep those paths until a later idle open; an
  // already-open fd cannot make unlink safe across a second process crash.
  static void sweep_orphan_active_laser_dirs(
      const std::filesystem::path &root,
      const internal::collection::CollectionControlState &control_state) {
    std::error_code error;
    const auto active_root = root / ".alaya_internal" / "active_laser";
    if (!std::filesystem::is_directory(active_root, error)) {
      return;
    }
    std::set<std::filesystem::path> keep;
    keep.insert(
        active_laser_dir(root, control_state.active_segment_id, control_state.active_generation)
            .filename());
    if (control_state.phase == internal::collection::CollectionControlPhase::successor_active ||
        control_state.phase == internal::collection::CollectionControlPhase::building ||
        control_state.phase == internal::collection::CollectionControlPhase::manifest_published) {
      for (const auto &source : control_state.sources) {
        keep.insert(active_laser_dir(root, source.segment_id, source.generation).filename());
      }
    }
    std::vector<std::filesystem::path> orphans;
    for (const auto &entry : std::filesystem::directory_iterator(active_root, error)) {
      if (entry.is_directory(error) && !keep.contains(entry.path().filename())) {
        orphans.push_back(entry.path());
      }
    }
    for (const auto &orphan : orphans) {
      std::filesystem::remove_all(orphan, error);
    }
    if (!orphans.empty()) {
      try {
        platform::sync_directory_or_throw(active_root);
      } catch (...) {  // NOLINT(bugprone-empty-catch): durability of the unlink is
        // best-effort; a crash before the parent fsync just re-sweeps on next open.
      }
    }
  }

  // Ruling 12: physical row capacity of the active LASER segment. Default 4096; when
  // auto_seal_rows is set, keep the capacity strictly above it (churn headroom) so
  // the auto-seal threshold can never exceed the physical capacity.
  [[nodiscard]] static auto checked_active_laser_capacity(std::uint64_t auto_seal_rows)
      -> std::optional<std::size_t> {
    if (auto_seal_rows == 0) {
      return 4096;
    }
    if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
      if (auto_seal_rows > std::numeric_limits<std::size_t>::max()) {
        return std::nullopt;
      }
    }
    const auto threshold = static_cast<std::size_t>(auto_seal_rows);
    std::size_t doubled{};
    std::size_t capacity{};
    if (alaya_mul_overflow(threshold, std::size_t{2}, &doubled) ||
        alaya_add_overflow(doubled, std::size_t{4096}, &capacity) || capacity <= threshold) {
      return std::nullopt;
    }
    return capacity;
  }

  static auto active_laser_capacity(const CollectionOptions &options) -> std::size_t {
    return checked_active_laser_capacity(options.auto_seal_rows).value();
  }

  // Materialize a fresh empty active LASER segment directory (create-time / rotate).
  [[nodiscard]] static auto create_active_laser_segment(const CollectionOptions &options,
                                                        std::uint64_t segment_id,
                                                        std::uint64_t generation) -> core::Status {
#if ALAYA_COLLECTION_HAS_ACTIVE_LASER
    try {
      ::alaya::disk::MutableLaserSegment::
          create_empty(active_laser_dir(options.root, segment_id, generation),
                       internal::collection::detail::collection_segment_name(segment_id),
                       options.dim,
                       options.dim,
                       options.max_neighbors,
                       options.metric);
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
#else
    (void)options;
    (void)segment_id;
    (void)generation;
    return error(core::StatusCode::not_supported,
                 core::OperationStage::build,
                 core::StatusDetail::operation_slot_absent,
                 "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
#endif
  }

  // Build the active-mutable registration for the configured active engine. flat is
  // an in-memory exact table; laser OPENS its existing on-disk directory (created at
  // Collection::create / rotate) -- a missing directory on open is corruption, never
  // a silent re-create that would drop committed rows.
  [[nodiscard]] static auto make_active_registration(
      const CollectionOptions &options,
      std::uint64_t segment_id = kActiveSegmentId,
      std::uint64_t generation = kActiveSegmentGeneration)
      -> core::Result<internal::collection::SegmentRegistration> {
    const internal::collection::CollectionSchema schema{options.dim,
                                                        options.metric,
                                                        options.scalar_type,
                                                        options.max_logical_id_bytes};
    if (options.active_engine != core::algorithm::laser) {
      return internal::collection::detail::make_canonical_flat_registration(schema,
                                                                            segment_id,
                                                                            generation);
    }
#if ALAYA_COLLECTION_HAS_ACTIVE_LASER
    const auto dir = active_laser_dir(options.root, segment_id, generation);
    if (!std::filesystem::exists(dir / "manifest.txt")) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::open,
                   core::StatusDetail::malformed_struct,
                   "active LASER segment directory is missing: " + dir.string());
    }
    try {
      laser::UpdateParams params;
      params.max_points = active_laser_capacity(options);
      params.enable_pid_reuse = true;
      auto segment =
          std::make_shared<::alaya::disk::MutableLaserSegment>(dir,
                                                               params,
                                                               laser::ResidencyMode::kPagedPool,
                                                               /*allow_empty=*/true);
      return internal::collection::detail::make_active_laser_registration(std::move(segment),
                                                                          schema,
                                                                          segment_id,
                                                                          generation);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
#else
    return error(core::StatusCode::not_supported,
                 core::OperationStage::open,
                 core::StatusDetail::operation_slot_absent,
                 "active LASER engine requires a Linux build with ALAYA_ENABLE_LASER");
#endif
  }

  [[nodiscard]] static auto numeric_segment_id(std::string_view segment_id) -> std::uint64_t {
    if (!segment_id.starts_with("seg_") || segment_id.size() != 12) {
      throw std::invalid_argument("Collection segment identity is malformed");
    }
    return parse_u64(segment_id.substr(4));
  }

  [[nodiscard]] static auto open_segmented(
      const CollectionOptions &options,
      const internal::collection::CollectionControlState &control_state)
      -> core::Result<std::shared_ptr<internal::collection::SegmentedCollection>> {
    internal::collection::CollectionSchema schema{options.dim,
                                                  options.metric,
                                                  options.scalar_type,
                                                  options.max_logical_id_bytes};
    std::vector<internal::collection::SegmentRegistration> registrations;
    auto manifest = internal::collection::load_manifest_v2_if_present(options.root);
    if (!manifest.ok()) {
      return manifest.status();
    }
    if (manifest.value().has_value()) {
      for (const auto &entry : manifest.value()->segments) {
        if (entry.lifecycle == internal::collection::SegmentLifecycleV2::retired ||
            entry.lifecycle == internal::collection::SegmentLifecycleV2::gc_pending) {
          continue;
        }
        core::OpenContext context;
        auto erased =
            internal::collection::detail::CollectionSegmentFactory::open_entry(options.root,
                                                                               entry,
                                                                               schema,
                                                                               context);
        if (!erased.ok()) {
          return erased.status();
        }
        internal::collection::SegmentRegistration registration;
        registration.segment_id = numeric_segment_id(entry.segment_id);
        registration.generation = entry.generation;
        registration.role = internal::collection::SegmentRole::sealed;
        registration.segment = std::move(erased).value();
        registrations.push_back(std::move(registration));
      }
    }

    const auto already_registered = [&](std::uint64_t segment_id, std::uint64_t generation) {
      return std::ranges::any_of(registrations, [&](const auto &registration) {
        return registration.segment_id == segment_id && registration.generation == generation;
      });
    };
    if (control_state.phase != internal::collection::CollectionControlPhase::idle &&
        control_state.phase != internal::collection::CollectionControlPhase::cut_pending) {
      for (const auto &source : control_state.sources) {
        if (already_registered(source.segment_id, source.generation)) {
          continue;
        }
        auto source_registration =
            make_active_registration(options, source.segment_id, source.generation);
        if (!source_registration.ok()) {
          return source_registration.status();
        }
        source_registration.value().role = internal::collection::SegmentRole::sealed;
        registrations.push_back(std::move(source_registration).value());
      }
    }

    if (options.active_engine == core::algorithm::laser) {
      sweep_orphan_active_laser_dirs(options.root, control_state);
    }
    auto active = make_active_registration(options,
                                           control_state.active_segment_id,
                                           control_state.active_generation);
    if (!active.ok()) {
      return active.status();
    }
    internal::collection::CollectionConfig config;
    config.features.wal_coordinator = true;
    config.features.manifest_v2_writer = true;
    config.wal.root = options.root;
    registrations.push_back(std::move(active).value());
    return internal::collection::SegmentedCollection::open(schema,
                                                           std::move(registrations),
                                                           std::move(config));
  }

  struct ReplacementBuildData {
    std::vector<internal::collection::SegmentReplacement> replacements{};
    std::vector<internal::collection::RegisteredRow> rows{};
    core::RowCount live_rows{};
    std::uint64_t snapshot_bytes{};
  };

  struct PendingGcCandidate {
    std::string manifest_segment_id{};
    std::filesystem::path artifact_directory{};
    std::weak_ptr<internal::collection::SegmentEntry> epoch_reference{};
  };

  // In-memory carrier for the piece of a prepared rotation that cannot be
  // recovered from durable state without a full reopen: the already-built
  // AnySegment handle plus the row/replacement map produced by
  // prepare_successor_locked(). Everything else rotate_to_successor_locked()
  // needs (source/target segment identities, wal_cut, mapping_file) stays in
  // control_state_, which is what crash recovery (recover_control_state())
  // independently reconstructs this same work from after a restart.
  struct PendingRotation {
    ReplacementBuildData build_data{};
    internal::collection::detail::CollectionTargetBuildResult built_target{};
  };

  static void fire_seal_failpoint(const CollectionSealOptions &options,
                                  CollectionSealFailPoint point) {
    if (options.fail_point == point && options.failpoint_hook) {
      options.failpoint_hook(point);
    }
  }

  [[nodiscard]] static auto address_is_source(
      const internal::collection::RowAddress &address,
      std::span<const internal::collection::RowAddress> sources) -> bool {
    return std::ranges::any_of(sources, [&](const auto &source) {
      return source.segment_id == address.segment_id && source.generation == address.generation;
    });
  }

  [[nodiscard]] static auto collect_replacement_rows(
      const internal::collection::RoutingSnapshot &snapshot,
      std::span<const internal::collection::RowAddress> sources,
      std::uint64_t target_segment_id,
      std::uint64_t target_generation) -> core::Result<ReplacementBuildData> {
    ReplacementBuildData result;
    std::uint64_t next_row{};
    for (const auto state : {internal::collection::VersionState::live,
                             internal::collection::VersionState::tombstone}) {
      for (const auto &[logical_id, version] : snapshot.versions) {
        if (version.state != state || !address_is_source(version.address, sources)) {
          continue;
        }
        internal::collection::RowAddress target{target_segment_id,
                                                target_generation,
                                                core::SegmentRowId(next_row++)};
        result.replacements.push_back(
            {logical_id, version.address, target, version.upsert_sequence});
        result.rows.push_back(
            {logical_id, target.row_id, version.upsert_sequence, version.state, version.payload});
        if (version.state == internal::collection::VersionState::live) {
          ++result.live_rows;
        }
        std::uint64_t row_bytes = version.payload.document.size();
        if (version.payload.vector.has_value() &&
            !core::checked_add(row_bytes, version.payload.vector->bytes().size(), row_bytes)) {
          return error(core::StatusCode::resource_exhausted,
                       core::OperationStage::freeze,
                       core::StatusDetail::arithmetic_overflow,
                       "seal snapshot accounting overflowed");
        }
        for (const auto &[key, value] : version.payload.metadata) {
          std::uint64_t scalar_bytes = key.size();
          std::visit(
              [&](const auto &item) {
                using T = std::decay_t<decltype(item)>;
                if constexpr (std::is_same_v<T, std::string>) {
                  scalar_bytes += item.size();
                } else {
                  scalar_bytes += sizeof(item);
                }
              },
              value);
          if (!core::checked_add(row_bytes, scalar_bytes, row_bytes)) {
            return error(core::StatusCode::resource_exhausted,
                         core::OperationStage::freeze,
                         core::StatusDetail::arithmetic_overflow,
                         "seal metadata accounting overflowed");
          }
        }
        if (!core::checked_add(result.snapshot_bytes, row_bytes, result.snapshot_bytes)) {
          return error(core::StatusCode::resource_exhausted,
                       core::OperationStage::freeze,
                       core::StatusDetail::arithmetic_overflow,
                       "seal snapshot total accounting overflowed");
        }
      }
    }
    return result;
  }

  [[nodiscard]] auto checkpoint_locked(core::CheckpointContext &context)
      -> core::Result<CollectionCheckpointReceipt> {
    auto receipt = implementation_->checkpoint(context);
    if (!receipt.ok()) {
      return receipt.status();
    }
    auto manifest = internal::collection::load_manifest_v2_if_present(options_.root);
    if (!manifest.ok()) {
      return manifest.status();
    }
    if (manifest.value().has_value()) {
      auto updated = std::move(*manifest.value());
      internal::collection::SegmentedCollection::apply_checkpoint_to_manifest(receipt.value(),
                                                                              updated);
      updated.publication.generation =
          std::max(updated.publication.generation + 1, control_state_.manifest_generation + 1);
      updated.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
      auto status = internal::collection::publish_manifest_v2_atomic(options_.root, updated);
      if (!status.ok()) {
        return status;
      }
      control_state_.manifest_generation = updated.publication.generation;
      status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
      if (!status.ok()) {
        return status;
      }
    }
    return receipt;
  }

  [[nodiscard]] auto patch_published_target_manifest() -> core::Status {
    auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
    if (!loaded.ok()) {
      return loaded.status();
    }
    if (!loaded.value().has_value()) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::save,
                   core::StatusDetail::malformed_struct,
                   "Collection target build did not publish manifest v2");
    }
    auto manifest = std::move(*loaded.value());
    const auto target_name =
        internal::collection::detail::collection_segment_name(control_state_.target_segment_id);
    const auto target = std::ranges::find_if(manifest.segments, [&](const auto &entry) {
      return entry.segment_id == target_name &&
             entry.generation == control_state_.target_generation;
    });
    if (target == manifest.segments.end()) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::save,
                   core::StatusDetail::malformed_struct,
                   "published manifest omits the Collection replacement target");
    }
    target->lifecycle = internal::collection::SegmentLifecycleV2::sealed;
    target->source_retention.clear();
    for (const auto &source : control_state_.sources) {
      const auto source_name =
          internal::collection::detail::collection_segment_name(source.segment_id);
      target->source_retention.push_back(source_name);
      const auto source_entry = std::ranges::find_if(manifest.segments, [&](const auto &entry) {
        return entry.segment_id == source_name && entry.generation == source.generation;
      });
      if (source_entry != manifest.segments.end()) {
        source_entry->lifecycle = internal::collection::SegmentLifecycleV2::gc_pending;
      }
      if (std::ranges::find(manifest.gc.pending_segment_ids, source_name) ==
          manifest.gc.pending_segment_ids.end()) {
        manifest.gc.pending_segment_ids.push_back(source_name);
      }
    }
    manifest.gc.phase = internal::collection::GcPhaseV2::pending;
    ++manifest.gc.generation;
    manifest.gc.retained_sources = {target_name};
    manifest.publication.generation =
        std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
    manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
    auto status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
    if (!status.ok()) {
      return status;
    }
    control_state_.manifest_generation = manifest.publication.generation;
    return core::Status::success();
  }

  [[nodiscard]] static auto normalize_control_state_before_open(
      const std::filesystem::path &root,
      internal::collection::CollectionControlState &state) -> core::Status {
    if (state.phase == internal::collection::CollectionControlPhase::cut_pending) {
      internal::collection::CollectionControlStore::remove_replacements(root, state.mapping_file);
      state.operation = internal::collection::CollectionControlOperation::idle;
      state.phase = internal::collection::CollectionControlPhase::idle;
      state.sources.clear();
      state.successor_segment_id = 0;
      state.successor_generation = 0;
      state.target_segment_id = 0;
      state.target_generation = 0;
      state.wal_cut = 0;
      state.mapping_file.clear();
      return internal::collection::CollectionControlStore::save(root, state);
    }

    auto manifest = internal::collection::load_manifest_v2_if_present(root);
    if (!manifest.ok()) {
      return manifest.status();
    }
    const auto target_name =
        state.target_segment_id == 0
            ? std::string{}
            : internal::collection::detail::collection_segment_name(state.target_segment_id);
    const auto target_published =
        manifest.value().has_value() &&
        std::ranges::any_of(manifest.value()->segments, [&](const auto &entry) {
          return entry.segment_id == target_name && entry.generation == state.target_generation;
        });
    if (state.phase == internal::collection::CollectionControlPhase::building && target_published) {
      state.phase = internal::collection::CollectionControlPhase::manifest_published;
      return internal::collection::CollectionControlStore::save(root, state);
    }
    if (state.phase == internal::collection::CollectionControlPhase::manifest_published &&
        !target_published) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::open,
                   core::StatusDetail::malformed_struct,
                   "Collection state says manifest-published but the target is absent");
    }
    if (state.phase == internal::collection::CollectionControlPhase::building &&
        !target_published) {
      auto status = internal::collection::ArtifactControlPlaneTransaction::cleanup_orphans(root);
      if (!status.ok()) {
        return status;
      }
      internal::collection::CollectionControlStore::remove_replacements(root, state.mapping_file);
      state.mapping_file.clear();
      state.phase = state.operation == internal::collection::CollectionControlOperation::seal
                        ? internal::collection::CollectionControlPhase::successor_active
                        : internal::collection::CollectionControlPhase::idle;
      if (state.phase == internal::collection::CollectionControlPhase::idle) {
        state.operation = internal::collection::CollectionControlOperation::idle;
        state.pending_compacted_bytes = 0;
        state.sources.clear();
        state.target_segment_id = 0;
        state.target_generation = 0;
      }
      return internal::collection::CollectionControlStore::save(root, state);
    }
    return core::Status::success();
  }

  [[nodiscard]] auto recover_control_state() -> core::Status {
    if (control_state_.phase != internal::collection::CollectionControlPhase::manifest_published) {
      return core::Status::success();
    }
    auto status = patch_published_target_manifest();
    if (!status.ok()) {
      return status;
    }
    auto replacements =
        internal::collection::CollectionControlStore::load_replacements(options_.root,
                                                                        control_state_
                                                                            .mapping_file);
    if (!replacements.ok()) {
      return replacements.status();
    }
    auto pinned = implementation_->pin_routing_snapshot();
    for (const auto &source : control_state_.sources) {
      if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
        const auto source_name =
            internal::collection::detail::collection_segment_name(source.segment_id);
        pending_gc_.push_back(
            {source_name,
             control_state_.operation == internal::collection::CollectionControlOperation::compact
                 ? options_.root / "segments" / source_name
                 : std::filesystem::path{},
             entry});
      }
    }
    status = implementation_->resume_segment_replacement(control_state_.sources,
                                                         control_state_.target_segment_id,
                                                         control_state_.target_generation,
                                                         replacements.value());
    if (!status.ok()) {
      return status;
    }
    pinned.reset();
    core::CheckpointContext checkpoint_context;
    checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto checkpoint = checkpoint_locked(checkpoint_context);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    const auto mapping_file = control_state_.mapping_file;
    if (control_state_.operation == internal::collection::CollectionControlOperation::compact) {
      if (!core::checked_add(control_state_.compacted_bytes,
                             control_state_.pending_compacted_bytes,
                             control_state_.compacted_bytes)) {
        control_state_.compacted_bytes = std::numeric_limits<std::uint64_t>::max();
      }
    }
    control_state_.pending_compacted_bytes = 0;
    control_state_.operation = internal::collection::CollectionControlOperation::idle;
    control_state_.phase = internal::collection::CollectionControlPhase::idle;
    control_state_.last_sealed_segment_id = control_state_.target_segment_id;
    control_state_.last_sealed_generation = control_state_.target_generation;
    control_state_.sources.clear();
    control_state_.successor_segment_id = 0;
    control_state_.successor_generation = 0;
    control_state_.target_segment_id = 0;
    control_state_.target_generation = 0;
    control_state_.wal_cut = 0;
    control_state_.mapping_file.clear();
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    internal::collection::CollectionControlStore::remove_replacements(options_.root, mapping_file);
    return core::Status::success();
  }

  [[nodiscard]] auto seal_locked(core::SealContext &context, const CollectionSealOptions &options)
      -> core::Result<CollectionSealReceipt> {
    if (auto gate = implementation_->recovery_gate(core::OperationStage::freeze); !gate.ok()) {
      return gate;
    }
    auto handle = prepare_successor_locked(context, options);
    if (!handle.ok()) {
      return handle.status();
    }
    return rotate_to_successor_locked(handle.value(), context);
  }

  // Builds a successor segment and durably publishes it (manifest: source(s)
  // -> gc_pending, successor -> sealed) without touching query routing yet.
  // See CollectionRotationHandle for the two-phase contract.
  //
  // TODO(rotate-to-successor): the only successor-construction strategy
  // implemented today is this synchronous full row-export + rebuild (the
  // same strategy the historical, single-call seal() used). A future
  // scheduler could build the successor asynchronously/incrementally (e.g.
  // from a FrozenGraphSnapshot exported off the live growing graph) and
  // call rotate_to_successor_locked() directly once it has produced an
  // equivalent handle; that scheduling policy is intentionally out of
  // scope for this change.
  [[nodiscard]] auto prepare_successor_locked(core::SealContext &context,
                                              const CollectionSealOptions &options)
      -> core::Result<CollectionRotationHandle> {
    if (auto gate = implementation_->recovery_gate(core::OperationStage::freeze); !gate.ok()) {
      return gate;
    }
    if (pending_rotation_.has_value()) {
      return error(core::StatusCode::conflict,
                   core::OperationStage::freeze,
                   core::StatusDetail::none,
                   "a successor is already prepared; call rotate_to_successor() first");
    }
    auto status = core::validate_runtime_control(context.deadline,
                                                 context.cancellation,
                                                 core::OperationStage::freeze);
    if (!status.ok()) {
      return status;
    }
    status = normalize_control_state_before_open(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    if (control_state_.phase == internal::collection::CollectionControlPhase::manifest_published) {
      status = recover_control_state();
      if (!status.ok()) {
        return status;
      }
    }
    if (control_state_.phase != internal::collection::CollectionControlPhase::idle &&
        control_state_.phase != internal::collection::CollectionControlPhase::successor_active) {
      return error(core::StatusCode::conflict,
                   core::OperationStage::freeze,
                   core::StatusDetail::none,
                   "another Collection control-plane operation is in progress");
    }

    if (control_state_.phase == internal::collection::CollectionControlPhase::idle) {
      const auto snapshot = implementation_->pin_routing_snapshot();
      const auto source = snapshot->find_active_mutable();
      if (source == nullptr || snapshot->known_rows_for(*source) == 0) {
        return error(core::StatusCode::not_found,
                     core::OperationStage::freeze,
                     core::StatusDetail::none,
                     "cannot seal an empty active segment");
      }
      if (control_state_.next_segment_id > 99'999'998) {
        return error(core::StatusCode::resource_exhausted,
                     core::OperationStage::freeze,
                     core::StatusDetail::arithmetic_overflow,
                     "Collection segment namespace is exhausted");
      }
      control_state_.operation = internal::collection::CollectionControlOperation::seal;
      control_state_.phase = internal::collection::CollectionControlPhase::cut_pending;
      control_state_.sources = {internal::collection::RowAddress{source->segment_id,
                                                                 source->generation,
                                                                 core::SegmentRowId{}}};
      control_state_.successor_segment_id = control_state_.next_segment_id++;
      control_state_.successor_generation = 1;
      control_state_.target_segment_id = control_state_.next_segment_id++;
      control_state_.target_generation = 1;
      control_state_.wal_cut = snapshot->visibility_watermark;
      control_state_.mapping_file.clear();
      status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
      if (!status.ok()) {
        return status;
      }
      fire_seal_failpoint(options, CollectionSealFailPoint::after_cut_before_successor);

      if (options_.active_engine == core::algorithm::laser) {
        auto created = create_active_laser_segment(options_,
                                                   control_state_.successor_segment_id,
                                                   control_state_.successor_generation);
        if (!created.ok()) {
          return created;
        }
      }
      auto successor = make_active_registration(options_,
                                                control_state_.successor_segment_id,
                                                control_state_.successor_generation);
      if (!successor.ok()) {
        return successor.status();
      }
      core::CheckpointContext checkpoint_context;
      checkpoint_context.deadline = context.deadline;
      checkpoint_context.cancellation = context.cancellation;
      checkpoint_context.lane = context.lane;
      checkpoint_context.dirty_page_io_credits = context.io_credits;
      checkpoint_context.wal_io_credits = context.io_credits;
      checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
      auto rotated = implementation_->rotate_to_successor(
          std::move(successor).value(),
          checkpoint_context,
          [&](const internal::collection::ActiveRotationReceipt &receipt) {
            control_state_.active_segment_id = receipt.successor_segment_id;
            control_state_.active_generation = receipt.successor_generation;
            control_state_.wal_cut = receipt.checkpoint.wal_cut;
            control_state_.phase = internal::collection::CollectionControlPhase::successor_active;
            auto saved =
                internal::collection::CollectionControlStore::save(options_.root, control_state_);
            if (!saved.ok()) {
              return saved;
            }
            fire_seal_failpoint(options,
                                CollectionSealFailPoint::
                                    after_active_control_publish_before_routing_install);
            return core::Status::success();
          });
      if (!rotated.ok()) {
        return rotated.status();
      }
      fire_seal_failpoint(options, CollectionSealFailPoint::after_successor_switch);
    }

    auto pinned = implementation_->pin_routing_snapshot();
    auto build_data = collect_replacement_rows(*pinned,
                                               control_state_.sources,
                                               control_state_.target_segment_id,
                                               control_state_.target_generation);
    if (!build_data.ok()) {
      return build_data.status();
    }
    if (build_data.value().live_rows == 0) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::build,
                   core::StatusDetail::none,
                   "active seal snapshot contains no live rows");
    }
    status = context.snapshot_reservation.ensure(build_data.value().snapshot_bytes,
                                                 core::OperationStage::freeze,
                                                 "seal snapshot reservation is too small");
    if (!status.ok()) {
      return status;
    }
    control_state_.mapping_file =
        "seal_" + std::to_string(control_state_.target_segment_id) + ".map";
    status =
        internal::collection::CollectionControlStore::save_replacements(options_.root,
                                                                        control_state_.mapping_file,
                                                                        build_data.value()
                                                                            .replacements);
    if (!status.ok()) {
      return status;
    }
    control_state_.phase = internal::collection::CollectionControlPhase::building;
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    auto base_manifest = internal::collection::load_manifest_v2_if_present(options_.root);
    if (!base_manifest.ok()) {
      return base_manifest.status();
    }
    internal::collection::detail::CollectionTargetPublication publication;
    publication.collection_root = options_.root;
    publication.segment_id =
        internal::collection::detail::collection_segment_name(control_state_.target_segment_id);
    publication.segment_generation = control_state_.target_generation;
    publication.manifest_generation =
        std::max(control_state_.manifest_generation + 1,
                 base_manifest.value().has_value()
                     ? base_manifest.value()->publication.generation + 1
                     : std::uint64_t{1});
    publication.publication_parent = std::string(internal::collection::kCollectionManifestFilename);
    publication.metadata_epoch = pinned->metadata_epoch;
    publication.metadata_checkpoint =
        "checkpoint_" + std::to_string(control_state_.wal_cut) + ".bin";
    publication.wal_cut = control_state_.wal_cut;
    publication.row_versions = {control_state_.wal_cut == 0 ? std::uint64_t{0} : std::uint64_t{1},
                                control_state_.wal_cut};
    publication.id_map_checkpoint = publication.metadata_checkpoint;
    publication.collection_features.manifest_v2_writer = true;
    publication.abort_policy =
        internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
    if (options.fail_point == CollectionSealFailPoint::during_export_build &&
        options.failpoint_hook) {
      publication.fail_point =
          internal::collection::ArtifactTransactionFailPoint::after_staging_write;
    }
    publication.base_manifest = std::move(base_manifest).value();
    core::BuildContext build_context;
    build_context.growing_reservation = context.build_reservation;
    build_context.io_credits = context.io_credits;
    build_context.deadline = context.deadline;
    build_context.cancellation = context.cancellation;
    build_context.lane = context.lane;
    internal::collection::CollectionSchema schema{options_.dim,
                                                  options_.metric,
                                                  options_.scalar_type,
                                                  options_.max_logical_id_bytes};
    internal::collection::detail::CollectionTargetBuildParams build_params;
    build_params.quantization = options_.quantization;
    build_params.max_neighbors = options_.max_neighbors;
    build_params.ef_construction = options_.ef_construction;
    build_params.thread_count = options_.build_threads;
    const auto resolution = resolve_build_algorithm(options_.target_algorithm,
                                                    schema,
                                                    build_data.value().live_rows,
                                                    build_params);
    auto built = internal::collection::detail::build_collection_target(resolution.algorithm,
                                                                       schema,
                                                                       build_data.value().rows,
                                                                       build_params,
                                                                       publication,
                                                                       build_context);
    if (options.fail_point == CollectionSealFailPoint::during_export_build &&
        options.failpoint_hook) {
      fire_seal_failpoint(options, CollectionSealFailPoint::during_export_build);
    }
    if (!built.ok()) {
      return built.status();
    }
    auto built_target = std::move(built).value();
    built_target.requested_algorithm = options_.target_algorithm;
    built_target.flat_fallback = resolution.flat_fallback;
    built_target.fallback_reason = resolution.fallback_reason;
    if (resolution.flat_fallback) {
      built_target.built_algorithm = core::algorithm::flat;
    }
    status = patch_published_target_manifest();
    if (!status.ok()) {
      return status;
    }
    control_state_.phase = internal::collection::CollectionControlPhase::manifest_published;
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    fire_seal_failpoint(options, CollectionSealFailPoint::after_manifest_publish);

    // Register the predecessor(s) for deferred GC now, in the same
    // control_mutex_ hold that just made them gc_pending in the durable
    // manifest. gc() takes control_mutex_ too, so without this an
    // interleaved gc() call between prepare_successor() and
    // rotate_to_successor() would see a segment the manifest calls
    // gc_pending but that pending_gc_ (in memory) knows nothing about yet,
    // and could reclaim it on disk while it is still the live, routed
    // predecessor. Registering it here means gc()'s weak_ptr check always
    // finds it non-expired (still routed) until rotate_to_successor_locked()
    // actually removes it from the routing table, so an interleaved gc()
    // still correctly defers instead of reclaiming a live segment.
    for (const auto &source : control_state_.sources) {
      if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
        pending_gc_.push_back(
            {internal::collection::detail::collection_segment_name(source.segment_id), {}, entry});
      }
    }

    CollectionRotationHandle handle;
    handle.successor_segment_id = control_state_.target_segment_id;
    handle.successor_generation = control_state_.target_generation;
    handle.predecessor_segment_ids.reserve(control_state_.sources.size());
    for (const auto &source : control_state_.sources) {
      handle.predecessor_segment_ids.push_back(source.segment_id);
    }
    pending_rotation_ = PendingRotation{std::move(build_data).value(), std::move(built_target)};
    return handle;
  }

  // Atomically switches query routing from the predecessor segment(s) to
  // the successor identified by `handle` (as returned by a prior
  // prepare_successor_locked() call on this same instance) and retires the
  // predecessor (deferred reclaim; see gc()). The durable WAL/manifest
  // ordering was already established by prepare_successor_locked(); this
  // step is the in-memory routing swap plus bookkeeping, and it is itself
  // self-healing: if the process dies before or during this call,
  // Collection::open()'s automatic recovery (recover_control_state())
  // redoes it from the durable manifest/mapping file on next open, with no
  // explicit rotate_to_successor() call required.
  [[nodiscard]] auto rotate_to_successor_locked(const CollectionRotationHandle &handle,
                                                core::SealContext &context)
      -> core::Result<CollectionSealReceipt> {
    if (auto gate = implementation_->recovery_gate(core::OperationStage::save); !gate.ok()) {
      return gate;
    }
    if (!pending_rotation_.has_value() ||
        control_state_.phase != internal::collection::CollectionControlPhase::manifest_published ||
        handle.successor_segment_id != control_state_.target_segment_id ||
        handle.successor_generation != control_state_.target_generation ||
        handle.predecessor_segment_ids.size() != control_state_.sources.size() ||
        !std::equal(handle.predecessor_segment_ids.begin(),
                    handle.predecessor_segment_ids.end(),
                    control_state_.sources.begin(),
                    [](std::uint64_t segment_id, const auto &source) {
                      return segment_id == source.segment_id;
                    })) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::save,
                   core::StatusDetail::none,
                   "rotate_to_successor handle does not match a prepared successor");
    }

    auto &prepared = *pending_rotation_;
    internal::collection::SegmentRegistration target;
    target.segment_id = control_state_.target_segment_id;
    target.generation = control_state_.target_generation;
    target.role = internal::collection::SegmentRole::sealed;
    target.segment = std::move(prepared.built_target.segment);
    target.rows = prepared.build_data.rows;
    auto status = implementation_->install_segment_replacement(control_state_.sources,
                                                               std::move(target),
                                                               prepared.build_data.replacements);
    if (!status.ok()) {
      return status;
    }
    core::CheckpointContext checkpoint_context;
    checkpoint_context.deadline = context.deadline;
    checkpoint_context.cancellation = context.cancellation;
    checkpoint_context.lane = context.lane;
    checkpoint_context.dirty_page_io_credits = context.io_credits;
    checkpoint_context.wal_io_credits = context.io_credits;
    checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto checkpoint = checkpoint_locked(checkpoint_context);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }

    CollectionSealReceipt receipt;
    receipt.source_segment_id = control_state_.sources.front().segment_id;
    receipt.successor_segment_id = control_state_.active_segment_id;
    receipt.sealed_segment_id = control_state_.target_segment_id;
    receipt.wal_cut = control_state_.wal_cut;
    receipt.sealed_rows = prepared.build_data.live_rows;
    receipt.sealed_bytes = prepared.built_target.artifact_bytes;
    receipt.manifest_generation = control_state_.manifest_generation;
    receipt.built_algorithm = prepared.built_target.built_algorithm;
    receipt.effective_ef_construction = prepared.built_target.effective_ef_construction;
    receipt.flat_fallback = prepared.built_target.flat_fallback;
    receipt.fallback_reason = prepared.built_target.fallback_reason;
    const auto mapping_file = control_state_.mapping_file;
    control_state_.operation = internal::collection::CollectionControlOperation::idle;
    control_state_.phase = internal::collection::CollectionControlPhase::idle;
    control_state_.last_sealed_segment_id = control_state_.target_segment_id;
    control_state_.last_sealed_generation = control_state_.target_generation;
    control_state_.sources.clear();
    control_state_.successor_segment_id = 0;
    control_state_.successor_generation = 0;
    control_state_.target_segment_id = 0;
    control_state_.target_generation = 0;
    control_state_.wal_cut = 0;
    control_state_.mapping_file.clear();
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    internal::collection::CollectionControlStore::remove_replacements(options_.root, mapping_file);
    pending_rotation_.reset();
    return receipt;
  }

  [[nodiscard]] auto verify_flat_exports(
      const internal::collection::RoutingSnapshot &snapshot,
      std::span<const internal::collection::RowAddress> sources) const -> core::Status {
    for (const auto &source : sources) {
      const auto entry = snapshot.find_segment(source.segment_id, source.generation);
      if (entry == nullptr ||
          !entry->segment.capabilities().supports(core::OperationCapability::export_rows)) {
        return error(core::StatusCode::not_supported,
                     core::OperationStage::export_rows,
                     core::StatusDetail::operation_slot_absent,
                     "Flat compact source does not expose export_rows");
      }
      std::shared_ptr<::alaya::disk::DiskFlatExportState> owner;
      ::alaya::disk::DiskFlatExportRequest typed;
      typed.batch_rows = 256;
      typed.lifetime_owner = &owner;
      core::OpaqueOperationRequest request;
      request.payload = &typed;
      request.payload_size = sizeof(typed);
      core::ExportCursor cursor;
      auto status = entry->segment.export_rows(request, cursor);
      if (!status.ok()) {
        return status;
      }
      if (owner == nullptr || cursor.state != owner.get()) {
        return error(core::StatusCode::internal,
                     core::OperationStage::export_rows,
                     core::StatusDetail::malformed_struct,
                     "Flat export cursor did not retain its source epoch");
      }
      bool done{};
      while (!done) {
        ::alaya::disk::DiskFlatExportBatch batch;
        status = owner->next(batch);
        if (!status.ok()) {
          return status;
        }
        if (batch.logical_ids.size() != batch.vectors.rows ||
            batch.vectors.scalar_type != core::ScalarType::float32 ||
            batch.vectors.dim != options_.dim) {
          return error(core::StatusCode::corruption,
                       core::OperationStage::export_rows,
                       core::StatusDetail::malformed_struct,
                       "Flat export batch has an inconsistent row schema");
        }
        for (std::size_t index = 0; index < batch.logical_ids.size(); ++index) {
          const internal::collection::RowAddress address{source.segment_id,
                                                         source.generation,
                                                         core::SegmentRowId(
                                                             batch.logical_ids[index])};
          const auto reverse = snapshot.reverse.find(address);
          if (reverse == snapshot.reverse.end()) {
            continue;
          }
          const auto version = snapshot.versions.find(reverse->second.logical_id);
          if (version == snapshot.versions.end() || version->second.address != address ||
              version->second.state != internal::collection::VersionState::live ||
              !version->second.payload.vector.has_value() ||
              options_.metric == core::Metric::cosine) {
            continue;
          }
          std::vector<float> expected;
          status = internal::collection::detail::vector_as_float(*version->second.payload.vector,
                                                                 expected);
          if (!status.ok()) {
            return status;
          }
          if (expected.size() != options_.dim ||
              std::memcmp(expected.data(),
                          batch.vectors.row<float>(index),
                          expected.size() * sizeof(float)) != 0) {
            return error(core::StatusCode::corruption,
                         core::OperationStage::export_rows,
                         core::StatusDetail::malformed_struct,
                         "Flat compact export row differs from the Collection-owned vector");
          }
        }
        done = batch.done;
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] auto compact_locked(core::SealContext &context)
      -> core::Result<CollectionCompactReceipt> {
    if (auto gate = implementation_->recovery_gate(core::OperationStage::build); !gate.ok()) {
      return gate;
    }
    auto status = core::validate_runtime_control(context.deadline,
                                                 context.cancellation,
                                                 core::OperationStage::build);
    if (!status.ok()) {
      return status;
    }
    status = normalize_control_state_before_open(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    if (control_state_.phase == internal::collection::CollectionControlPhase::manifest_published) {
      status = recover_control_state();
      if (!status.ok()) {
        return status;
      }
    }
    if (control_state_.phase != internal::collection::CollectionControlPhase::idle) {
      return error(core::StatusCode::conflict,
                   core::OperationStage::build,
                   core::StatusDetail::none,
                   "another Collection control-plane operation is in progress");
    }
    auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
    if (!loaded.ok()) {
      return loaded.status();
    }
    if (!loaded.value().has_value()) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::build,
                   core::StatusDetail::none,
                   "Flat compact requires at least two sealed manifest entries");
    }
    auto base_manifest = std::move(*loaded.value());
    std::vector<internal::collection::RowAddress> sources;
    std::uint64_t input_bytes{};
    for (const auto &entry : base_manifest.segments) {
      if (entry.lifecycle != internal::collection::SegmentLifecycleV2::sealed ||
          entry.algorithm_id != core::algorithm::flat) {
        continue;
      }
      sources.push_back(
          {numeric_segment_id(entry.segment_id), entry.generation, core::SegmentRowId{}});
      for (const auto &artifact : entry.artifacts) {
        if (!core::checked_add(input_bytes, artifact.size_bytes, input_bytes)) {
          input_bytes = std::numeric_limits<std::uint64_t>::max();
          break;
        }
      }
    }
    if (sources.size() < 2) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::build,
                   core::StatusDetail::none,
                   "Flat compact requires at least two sealed Flat segments");
    }
    if (control_state_.next_segment_id > 99'999'999) {
      return error(core::StatusCode::resource_exhausted,
                   core::OperationStage::build,
                   core::StatusDetail::arithmetic_overflow,
                   "Flat segment namespace is exhausted");
    }
    auto pinned = implementation_->pin_routing_snapshot();
    status = verify_flat_exports(*pinned, sources);
    if (!status.ok()) {
      return status;
    }
    const auto target_segment_id = control_state_.next_segment_id++;
    constexpr std::uint64_t kTargetGeneration = 1;
    auto build_data =
        collect_replacement_rows(*pinned, sources, target_segment_id, kTargetGeneration);
    if (!build_data.ok()) {
      return build_data.status();
    }
    if (build_data.value().live_rows == 0) {
      return error(core::StatusCode::not_found,
                   core::OperationStage::build,
                   core::StatusDetail::none,
                   "Flat compact sources contain no live rows");
    }
    status = context.snapshot_reservation.ensure(build_data.value().snapshot_bytes,
                                                 core::OperationStage::build,
                                                 "compact snapshot reservation is too small");
    if (!status.ok()) {
      return status;
    }
    control_state_.operation = internal::collection::CollectionControlOperation::compact;
    control_state_.phase = internal::collection::CollectionControlPhase::building;
    control_state_.sources = sources;
    control_state_.target_segment_id = target_segment_id;
    control_state_.target_generation = kTargetGeneration;
    control_state_.wal_cut = pinned->visibility_watermark;
    control_state_.pending_compacted_bytes = input_bytes;
    control_state_.mapping_file = "compact_" + std::to_string(target_segment_id) + ".map";
    status =
        internal::collection::CollectionControlStore::save_replacements(options_.root,
                                                                        control_state_.mapping_file,
                                                                        build_data.value()
                                                                            .replacements);
    if (!status.ok()) {
      return status;
    }
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }

    internal::collection::detail::CollectionTargetPublication publication;
    publication.collection_root = options_.root;
    publication.segment_id =
        internal::collection::detail::collection_segment_name(target_segment_id);
    publication.segment_generation = kTargetGeneration;
    publication.manifest_generation =
        std::max(base_manifest.publication.generation + 1, control_state_.manifest_generation + 1);
    publication.publication_parent = std::string(internal::collection::kCollectionManifestFilename);
    publication.metadata_epoch = pinned->metadata_epoch;
    publication.metadata_checkpoint =
        "checkpoint_" + std::to_string(control_state_.wal_cut) + ".bin";
    publication.wal_cut = control_state_.wal_cut;
    publication.row_versions = {control_state_.wal_cut == 0 ? std::uint64_t{0} : std::uint64_t{1},
                                control_state_.wal_cut};
    publication.id_map_checkpoint = publication.metadata_checkpoint;
    publication.collection_features.manifest_v2_writer = true;
    publication.abort_policy =
        internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
    publication.base_manifest = std::move(base_manifest);
    core::BuildContext build_context;
    build_context.growing_reservation = context.build_reservation;
    build_context.io_credits = context.io_credits;
    build_context.deadline = context.deadline;
    build_context.cancellation = context.cancellation;
    build_context.lane = context.lane;
    internal::collection::CollectionSchema schema{options_.dim,
                                                  options_.metric,
                                                  options_.scalar_type,
                                                  options_.max_logical_id_bytes};
    internal::collection::detail::CollectionTargetBuildParams build_params;
    build_params.quantization = options_.quantization;
    build_params.max_neighbors = options_.max_neighbors;
    build_params.ef_construction = options_.ef_construction;
    build_params.thread_count = options_.build_threads;
    const auto resolution = resolve_build_algorithm(options_.target_algorithm,
                                                    schema,
                                                    build_data.value().live_rows,
                                                    build_params);
    auto built = internal::collection::detail::build_collection_target(resolution.algorithm,
                                                                       schema,
                                                                       build_data.value().rows,
                                                                       build_params,
                                                                       publication,
                                                                       build_context);
    if (!built.ok()) {
      return built.status();
    }
    auto built_target = std::move(built).value();
    built_target.requested_algorithm = options_.target_algorithm;
    built_target.flat_fallback = resolution.flat_fallback;
    built_target.fallback_reason = resolution.fallback_reason;
    if (resolution.flat_fallback) {
      built_target.built_algorithm = core::algorithm::flat;
    }
    status = patch_published_target_manifest();
    if (!status.ok()) {
      return status;
    }
    control_state_.phase = internal::collection::CollectionControlPhase::manifest_published;
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }

    for (const auto &source : sources) {
      const auto source_name =
          internal::collection::detail::collection_segment_name(source.segment_id);
      if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
        pending_gc_.push_back({source_name, options_.root / "segments" / source_name, entry});
      }
    }
    internal::collection::SegmentRegistration target;
    target.segment_id = target_segment_id;
    target.generation = kTargetGeneration;
    target.role = internal::collection::SegmentRole::sealed;
    target.segment = std::move(built_target.segment);
    target.rows = build_data.value().rows;
    status = implementation_->install_segment_replacement(sources,
                                                          std::move(target),
                                                          build_data.value().replacements);
    if (!status.ok()) {
      return status;
    }
    pinned.reset();
    core::CheckpointContext checkpoint_context;
    checkpoint_context.deadline = context.deadline;
    checkpoint_context.cancellation = context.cancellation;
    checkpoint_context.lane = context.lane;
    checkpoint_context.dirty_page_io_credits = context.io_credits;
    checkpoint_context.wal_io_credits = context.io_credits;
    checkpoint_context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto checkpoint = checkpoint_locked(checkpoint_context);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }

    CollectionCompactReceipt receipt;
    for (const auto &source : sources) {
      receipt.source_segment_ids.push_back(source.segment_id);
    }
    receipt.compacted_segment_id = target_segment_id;
    receipt.compacted_rows = build_data.value().live_rows;
    receipt.input_bytes = input_bytes;
    receipt.output_bytes = built_target.artifact_bytes;
    receipt.manifest_generation = control_state_.manifest_generation;
    receipt.built_algorithm = built_target.built_algorithm;
    receipt.effective_ef_construction = built_target.effective_ef_construction;
    receipt.flat_fallback = built_target.flat_fallback;
    receipt.fallback_reason = built_target.fallback_reason;
    const auto mapping_file = control_state_.mapping_file;
    if (!core::checked_add(control_state_.compacted_bytes,
                           input_bytes,
                           control_state_.compacted_bytes)) {
      control_state_.compacted_bytes = std::numeric_limits<std::uint64_t>::max();
    }
    control_state_.pending_compacted_bytes = 0;
    control_state_.operation = internal::collection::CollectionControlOperation::idle;
    control_state_.phase = internal::collection::CollectionControlPhase::idle;
    control_state_.last_sealed_segment_id = target_segment_id;
    control_state_.last_sealed_generation = kTargetGeneration;
    control_state_.sources.clear();
    control_state_.target_segment_id = 0;
    control_state_.target_generation = 0;
    control_state_.wal_cut = 0;
    control_state_.mapping_file.clear();
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    internal::collection::CollectionControlStore::remove_replacements(options_.root, mapping_file);
    return receipt;
  }

  [[nodiscard]] auto gc_locked() -> core::Result<CollectionGcReceipt> {
    if (auto gate = implementation_->recovery_gate(core::OperationStage::save); !gate.ok()) {
      return gate;
    }
    auto loaded = internal::collection::load_manifest_v2_if_present(options_.root);
    if (!loaded.ok()) {
      return loaded.status();
    }
    CollectionGcReceipt receipt;
    if (!loaded.value().has_value()) {
      return receipt;
    }
    auto manifest = std::move(*loaded.value());
    receipt.pending = manifest.gc.pending_segment_ids.size();
    std::vector<std::string> reclaim;
    const auto retained = control_state_.last_sealed_segment_id == 0
                              ? std::string{}
                              : internal::collection::detail::collection_segment_name(
                                    control_state_.last_sealed_segment_id);
    for (const auto &segment_id : manifest.gc.pending_segment_ids) {
      if (segment_id == retained) {
        ++receipt.deferred;
        continue;
      }
      const auto candidate = std::ranges::find_if(pending_gc_, [&](const auto &item) {
        return item.manifest_segment_id == segment_id;
      });
      if (candidate != pending_gc_.end() && !candidate->epoch_reference.expired()) {
        ++receipt.deferred;
        continue;
      }
      reclaim.push_back(segment_id);
    }
    if (reclaim.empty()) {
      receipt.manifest_generation = manifest.publication.generation;
      return receipt;
    }

    manifest.gc.phase = internal::collection::GcPhaseV2::reclaimable;
    ++manifest.gc.generation;
    manifest.publication.generation =
        std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
    manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
    auto status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
    if (!status.ok()) {
      return status;
    }

    try {
      for (const auto &segment_id : reclaim) {
        const auto entry = std::ranges::find_if(manifest.segments, [&](const auto &item) {
          return item.segment_id == segment_id;
        });
        if (entry != manifest.segments.end()) {
          if (entry->lifecycle != internal::collection::SegmentLifecycleV2::gc_pending) {
            return error(core::StatusCode::conflict,
                         core::OperationStage::save,
                         core::StatusDetail::none,
                         "GC refused a segment that is not gc_pending");
          }
          for (const auto &artifact : entry->artifacts) {
            if (!core::checked_add(receipt.reclaimed_bytes,
                                   artifact.size_bytes,
                                   receipt.reclaimed_bytes)) {
              receipt.reclaimed_bytes = std::numeric_limits<std::uint64_t>::max();
              break;
            }
          }
          std::filesystem::remove_all(options_.root / "segments" / segment_id);
        }
        ++receipt.reclaimed;
      }
      const auto segments_root = options_.root / "segments";
      if (std::filesystem::is_directory(segments_root)) {
        platform::sync_directory_or_throw(segments_root);
      }
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }

    std::erase_if(manifest.segments, [&](const auto &entry) {
      return std::ranges::find(reclaim, entry.segment_id) != reclaim.end();
    });
    std::erase_if(manifest.gc.pending_segment_ids, [&](const auto &segment_id) {
      return std::ranges::find(reclaim, segment_id) != reclaim.end();
    });
    manifest.gc.phase = manifest.gc.pending_segment_ids.empty()
                            ? internal::collection::GcPhaseV2::idle
                            : internal::collection::GcPhaseV2::pending;
    ++manifest.gc.generation;
    manifest.gc.retained_sources =
        retained.empty() ? std::vector<std::string>{} : std::vector<std::string>{retained};
    manifest.publication.generation =
        std::max(manifest.publication.generation + 1, control_state_.manifest_generation + 1);
    manifest.publication.parent = std::string(internal::collection::kCollectionManifestFilename);
    status = internal::collection::publish_manifest_v2_atomic(options_.root, manifest);
    if (!status.ok()) {
      return status;
    }
    control_state_.manifest_generation = manifest.publication.generation;
    status = internal::collection::CollectionControlStore::save(options_.root, control_state_);
    if (!status.ok()) {
      return status;
    }
    std::erase_if(pending_gc_, [&](const auto &candidate) {
      return std::ranges::find(reclaim, candidate.manifest_segment_id) != reclaim.end();
    });
    receipt.manifest_generation = manifest.publication.generation;
    return receipt;
  }

  [[nodiscard]] auto write(const CollectionItem &item,
                           internal::collection::WriteMode mode,
                           CollectionWriteOptions options)
      -> core::Result<CollectionMutationReceipt> {
    internal::collection::WriteRequest request;
    request.logical_id = item.logical_id;
    request.vector = item.vector;
    request.metadata = item.metadata;
    request.document = item.document;
    request.mode = mode;
    request.options = std::move(options);
    core::MutationContext context;
    auto receipt = implementation_->write(request, context);
    if (receipt.ok()) {
      maybe_auto_seal();
    }
    return receipt;
  }

  void maybe_auto_seal() noexcept {
    if (options_.auto_seal_rows == 0) {
      return;
    }
    try {
      const auto snapshot = implementation_->pin_routing_snapshot();
      const auto active = snapshot->find_active_mutable();
      if (active == nullptr || snapshot->known_rows_for(*active) < options_.auto_seal_rows) {
        return;
      }
      (void)seal();
    } catch (...) {
      // The committed mutation remains authoritative. Auto-seal is a
      // best-effort control-plane policy and can be retried explicitly.
    }
  }

  [[nodiscard]] auto execute_search(const core::TypedTensorView &queries,
                                    const core::SearchOptions &options,
                                    core::SearchContext &context,
                                    const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse> {
    CollectionSearchStatistics search_stats;
    internal::collection::CollectionSearchRequest request;
    request.queries = queries;
    request.options = options;
    request.filter = filter;
    request.context = std::addressof(context);
    request.stats = std::addressof(search_stats);
    auto result = implementation_->search(request);
    if (!result.ok()) {
      return result.status();
    }
    CollectionSearchResponse response;
    response.search_stats = search_stats;
    response.visibility_watermark = result.value().visibility_watermark;
    response.metadata_epoch = result.value().metadata_epoch;
    response.offsets.reserve(result.value().queries.size() + 1);
    response.valid_counts.reserve(result.value().queries.size());
    response.statuses.reserve(result.value().queries.size());
    response.completeness.reserve(result.value().queries.size());
    response.offsets.push_back(0);
    for (const auto &query : result.value().queries) {
      response.valid_counts.push_back(query.hits.size());
      response.statuses.push_back(query.status);
      response.completeness.push_back(query.completeness);
      for (const auto &hit : query.hits) {
        response.ids.push_back(hit.logical_id);
        response.distances.push_back(hit.score);
      }
      response.offsets.push_back(response.ids.size());
    }
    return response;
  }

  [[nodiscard]] static auto facade_schema_path(const std::filesystem::path &root)
      -> std::filesystem::path {
    return root / ".alaya_internal" / kFacadeNamespace / kFacadeSchemaFilename;
  }

  [[nodiscard]] static auto schema_prefix(const CollectionOptions &options) -> std::string {
    std::string prefix =
        "format=1\npublic_version=" + std::string(kCollectionPublicVersion) +
        "\ndim=" + std::to_string(options.dim) +
        "\nmetric=" + std::to_string(static_cast<unsigned>(options.metric)) +
        "\nscalar_type=" + std::to_string(static_cast<unsigned>(options.scalar_type)) +
        "\ntarget_algorithm=" + std::to_string(options.target_algorithm) +
        "\nquantization=" + std::to_string(static_cast<unsigned>(options.quantization)) +
        "\nbuild_threads=" + std::to_string(options.build_threads) +
        "\nmax_neighbors=" + std::to_string(options.max_neighbors) +
        "\nef_construction=" + std::to_string(options.ef_construction) +
        "\nmax_logical_id_bytes=" + std::to_string(options.max_logical_id_bytes) +
        "\nactive_segment_id=" + std::to_string(kActiveSegmentId) +
        "\nactive_generation=" + std::to_string(kActiveSegmentGeneration) + "\n";
    // B-08: only a non-default active engine widens the schema to 15 fields, so a
    // flat collection stays byte-compatible with pre-2B readers while a laser
    // collection makes an old binary fail-closed on the strict field count.
    if (options.active_engine != core::algorithm::flat) {
      prefix += "active_engine=" + std::to_string(options.active_engine) + "\n";
    }
    return prefix;
  }

  [[nodiscard]] static auto write_facade_schema(const CollectionOptions &options) -> core::Status {
    try {
      const auto path = facade_schema_path(options.root);
      std::filesystem::create_directories(path.parent_path());
      const auto prefix = schema_prefix(options);
      const auto body = prefix + "checksum=" + internal::collection::sha256(prefix).hex() + "\n";
      const auto temporary = path.string() + ".tmp";
      platform::write_all_fsync(temporary, body.data(), body.size());
      platform::atomic_replace(temporary, path);
      platform::sync_directory_or_throw(path.parent_path());
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  [[nodiscard]] static auto parse_u64(std::string_view value) -> std::uint64_t {
    if (value.empty()) {
      throw std::invalid_argument("canonical facade schema integer is empty");
    }
    std::uint64_t result{};
    for (const auto digit : value) {
      if (digit < '0' || digit > '9' ||
          result > (std::numeric_limits<std::uint64_t>::max() -
                    static_cast<std::uint64_t>(digit - '0')) /
                       10U) {
        throw std::invalid_argument("canonical facade schema integer is invalid");
      }
      result = result * 10U + static_cast<std::uint64_t>(digit - '0');
    }
    return result;
  }

  [[nodiscard]] static auto read_facade_schema(const std::filesystem::path &root)
      -> core::Result<CollectionOptions> {
    try {
      const auto path = facade_schema_path(root);
      if (std::filesystem::file_size(path) > 64U * 1024U) {
        throw std::invalid_argument("canonical facade schema exceeds its size limit");
      }
      std::ifstream input(path, std::ios::binary);
      const std::string body{std::istreambuf_iterator<char>(input),
                             std::istreambuf_iterator<char>()};
      if (!input.eof() && !input) {
        throw std::runtime_error("cannot read canonical facade schema");
      }
      std::map<std::string, std::string, std::less<>> fields;
      std::string prefix;
      std::istringstream lines(body);
      for (std::string line; std::getline(lines, line);) {
        const auto equal = line.find('=');
        if (equal == std::string::npos || equal == 0 || equal + 1 == line.size() ||
            !fields.emplace(line.substr(0, equal), line.substr(equal + 1)).second) {
          throw std::invalid_argument("canonical facade schema contains an invalid field");
        }
        if (!line.starts_with("checksum=")) {
          prefix += line + "\n";
        }
      }
      const auto required = [&](std::string_view key) -> const std::string & {
        const auto found = fields.find(key);
        if (found == fields.end()) {
          throw std::invalid_argument("canonical facade schema is missing field " +
                                      std::string(key));
        }
        return found->second;
      };
      if ((fields.size() != 14 && fields.size() != 15) || required("format") != "1" ||
          required("public_version") != kCollectionPublicVersion ||
          required("checksum") != internal::collection::sha256(prefix).hex() ||
          parse_u64(required("active_segment_id")) != kActiveSegmentId ||
          parse_u64(required("active_generation")) != kActiveSegmentGeneration) {
        throw std::invalid_argument("canonical facade schema identity/checksum is invalid");
      }
      CollectionOptions options;
      options.root = root;
      options.dim = static_cast<std::uint32_t>(parse_u64(required("dim")));
      options.metric = static_cast<core::Metric>(parse_u64(required("metric")));
      options.scalar_type = static_cast<core::ScalarType>(parse_u64(required("scalar_type")));
      options.target_algorithm = parse_u64(required("target_algorithm"));
      options.quantization =
          static_cast<CollectionQuantization>(parse_u64(required("quantization")));
      options.build_threads = static_cast<std::uint32_t>(parse_u64(required("build_threads")));
      options.max_neighbors = static_cast<std::uint32_t>(parse_u64(required("max_neighbors")));
      options.ef_construction = static_cast<std::uint32_t>(parse_u64(required("ef_construction")));
      options.max_logical_id_bytes = parse_u64(required("max_logical_id_bytes"));
      // B-08: 14 fields = pre-2B / flat active (default flat); 15 fields carries the
      // explicit active engine. An old binary rejects the 15th field on the strict
      // count above, so a laser collection fails-closed rather than silently
      // reverting to flat.
      options.active_engine =
          fields.size() == 15 ? parse_u64(required("active_engine")) : core::algorithm::flat;
      auto status = validate_options(options, core::OperationStage::open);
      if (!status.ok()) {
        return status;
      }
      return options;
    } catch (const std::invalid_argument &exception) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::open,
                   core::StatusDetail::malformed_struct,
                   exception.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  CollectionOptions options_{};
  std::shared_ptr<internal::collection::SegmentedCollection> implementation_{};
  mutable std::mutex control_mutex_{};
  internal::collection::CollectionControlState control_state_{};
  std::vector<PendingGcCandidate> pending_gc_{};
  std::optional<PendingRotation> pending_rotation_{};
};

}  // namespace alaya
