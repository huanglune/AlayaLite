// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "index/collection/detail/canonical_flat_segment.hpp"
#include "index/collection/detail/collection_flat_target.hpp"
#include "index/collection/legacy_importer.hpp"
#include "index/collection/sha256.hpp"
#include "utils/platform_fs.hpp"

namespace alaya {

namespace internal::collection {
class CollectionTestAccess;
}  // namespace internal::collection

inline constexpr std::string_view kCollectionPublicVersion{"1.1.0"};
inline constexpr std::string_view kCollectionLegacyRemovalVersion{"1.2.0"};

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
  core::AlgorithmId target_algorithm{core::algorithm::hnsw};
  CollectionQuantization quantization{CollectionQuantization::none};
  std::uint32_t build_threads{1};
  std::uint32_t ef_construction{400};
  std::uint64_t max_logical_id_bytes{64U * 1024U};
  // Zero disables automatic rotation. A positive value rotates after the
  // active generation reaches this many physical rows.
  std::uint64_t auto_seal_rows{};
  bool imported_legacy_layout{};
};

enum class CollectionSealFailPoint : std::uint8_t {
  none = 0,
  after_cut_before_successor = 1,
  after_successor_switch = 2,
  during_export_build = 3,
  after_manifest_publish = 4,
};

struct CollectionSealOptions {
  CollectionSealFailPoint fail_point{CollectionSealFailPoint::none};
  std::function<void(CollectionSealFailPoint)> failpoint_hook{};
};

struct CollectionSealReceipt {
  std::uint64_t source_segment_id{};
  std::uint64_t successor_segment_id{};
  std::uint64_t sealed_segment_id{};
  std::uint64_t wal_cut{};
  core::RowCount sealed_rows{};
  std::uint64_t sealed_bytes{};
  std::uint64_t manifest_generation{};
};

struct CollectionCompactReceipt {
  std::vector<std::uint64_t> source_segment_ids{};
  std::uint64_t compacted_segment_id{};
  core::RowCount compacted_rows{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
  std::uint64_t manifest_generation{};
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
      auto opened = open_segmented(options, state);
      if (!opened.ok()) {
        return opened.status();
      }
      auto result = std::shared_ptr<Collection>(
          new Collection(std::move(options), std::move(opened).value(), false, std::move(state)));
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
          auto status = normalize_control_state_before_open(root, state);
          if (!status.ok()) {
            return status;
          }
          options.value().auto_seal_rows = state.auto_seal_rows;
          auto opened = open_segmented(options.value(), state);
          if (!opened.ok()) {
            return opened.status();
          }
          const auto imported_layout = options.value().imported_legacy_layout;
          auto result = std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                                   std::move(opened).value(),
                                                                   imported_layout,
                                                                   std::move(state)));
          status = result->recover_control_state();
          if (!status.ok()) {
            return status;
          }
          return result;
        }
        if (options.value().imported_legacy_layout) {
          internal::collection::LegacyImportOptions import_options;
          import_options.source_root = root;
          import_options.target_root = root;
          import_options.features.legacy_importer = true;
          import_options.active_registration_factory = [](const auto &schema) {
            return make_active_registration(schema);
          };
          auto imported = internal::collection::LegacyImporter::import(import_options);
          if (!imported.ok()) {
            return imported.status();
          }
          internal::collection::CollectionControlState state;
          state.last_sealed_segment_id = 1;
          state.last_sealed_generation = 1;
          auto status = internal::collection::CollectionControlStore::save(root, state);
          if (!status.ok()) {
            return status;
          }
          return std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                            std::move(imported).value().collection,
                                                            true,
                                                            std::move(state)));
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
                                                          false,
                                                          std::move(state)));
      }

      const auto route = internal::collection::LegacyImporter::resolve_open_route(root, root);
      if (route == internal::collection::LegacyOpenRoute::unavailable) {
        return error(core::StatusCode::not_found,
                     core::OperationStage::open,
                     core::StatusDetail::none,
                     "canonical or supported legacy Collection layout was not found");
      }
      internal::collection::LegacyImportOptions import_options;
      import_options.source_root = root;
      import_options.target_root = root;
      import_options.features.legacy_importer = true;
      import_options.active_registration_factory = [](const auto &schema) {
        return make_active_registration(schema);
      };
      auto imported = internal::collection::LegacyImporter::import(import_options);
      if (!imported.ok()) {
        return imported.status();
      }
      CollectionOptions options;
      options.root = root;
      options.dim = imported.value().audit.dim;
      options.metric = core::Metric::l2;
      options.scalar_type = imported.value().audit.source_scalar_type;
      options.target_algorithm = core::algorithm::flat;
      options.quantization = CollectionQuantization::none;
      options.build_threads = 1;
      options.imported_legacy_layout = true;
      auto status = write_facade_schema(options);
      if (!status.ok()) {
        return status;
      }
      internal::collection::CollectionControlState state;
      state.last_sealed_segment_id = 1;
      state.last_sealed_generation = 1;
      status = internal::collection::CollectionControlStore::save(root, state);
      if (!status.ok()) {
        return status;
      }
      return std::shared_ptr<Collection>(new Collection(std::move(options),
                                                        std::move(imported).value().collection,
                                                        true,
                                                        std::move(state)));
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
    return core::algorithm::flat;
  }
  [[nodiscard]] auto target_implementation_key() const -> std::string_view {
    switch (options_.target_algorithm) {
      case core::algorithm::flat:
        return "disk_flat_segment";
      case core::algorithm::hnsw:
        return "hnsw_segment";
      case core::algorithm::nsg:
        return "nsg_segment";
      case core::algorithm::fusion:
        return "fusion_segment";
      case core::algorithm::qg:
        return "qg_segment";
      default:
        return "unknown";
    }
  }
  [[nodiscard]] auto target_engine_factory_key() const -> std::string_view {
    switch (options_.target_algorithm) {
      case core::algorithm::flat:
        return "flat";
      case core::algorithm::hnsw:
        return "hnsw";
      case core::algorithm::nsg:
        return "nsg";
      case core::algorithm::fusion:
        return "fusion";
      case core::algorithm::qg:
        return "qg";
      default:
        return "unknown";
    }
  }
  [[nodiscard]] auto imported_legacy_layout() const noexcept -> bool { return imported_; }

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
             bool imported,
             internal::collection::CollectionControlState control_state)
      : options_(std::move(options)),
        implementation_(std::move(implementation)),
        imported_(imported),
        control_state_(std::move(control_state)) {}

  [[nodiscard]] static auto error(core::StatusCode code,
                                  core::OperationStage stage,
                                  core::StatusDetail detail,
                                  std::string diagnostic) -> core::Status {
    return core::Status::error(code, stage, detail, std::move(diagnostic));
  }

  [[nodiscard]] static auto validate_options(const CollectionOptions &options,
                                             core::OperationStage stage) -> core::Status {
    if (options.root.empty() || options.dim == 0 || options.max_logical_id_bytes == 0 ||
        core::scalar_type_size(options.scalar_type) == 0 || options.build_threads == 0 ||
        options.ef_construction == 0) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection schema/root/build parameters are invalid");
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
                                 options.target_algorithm == core::algorithm::hnsw ||
                                 options.target_algorithm == core::algorithm::nsg ||
                                 options.target_algorithm == core::algorithm::fusion ||
                                 options.target_algorithm == core::algorithm::qg;
    if (!algorithm_valid) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::operation_slot_absent,
                   "canonical Collection target algorithm is unsupported");
    }
    if (options.quantization == CollectionQuantization::rabitq &&
        options.target_algorithm != core::algorithm::qg) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection requires explicit index_type=qg for rabitq");
    }
    if (options.target_algorithm == core::algorithm::qg &&
        options.quantization != CollectionQuantization::rabitq) {
      return error(core::StatusCode::invalid_argument,
                   stage,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection qg requires quantization=rabitq");
    }
    if (options.quantization != CollectionQuantization::none &&
        options.scalar_type != core::ScalarType::float32) {
      return error(core::StatusCode::not_supported,
                   stage,
                   core::StatusDetail::unsupported_scalar_type,
                   "canonical Collection quantization requires float32 vectors");
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto make_active_registration(
      const internal::collection::CollectionSchema &schema,
      std::uint64_t segment_id = kActiveSegmentId,
      std::uint64_t generation = kActiveSegmentGeneration)
      -> core::Result<internal::collection::SegmentRegistration> {
    return internal::collection::detail::make_canonical_flat_registration(schema,
                                                                          segment_id,
                                                                          generation);
  }

  [[nodiscard]] static auto numeric_segment_id(std::string_view segment_id) -> std::uint64_t {
    if (!segment_id.starts_with("seg_") || segment_id.size() != 12) {
      throw std::invalid_argument("Gate-10 Flat segment identity is malformed");
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
        auto erased = internal::collection::detail::open_collection_flat_entry(options.root,
                                                                               entry,
                                                                               options.scalar_type,
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
            make_active_registration(schema, source.segment_id, source.generation);
        if (!source_registration.ok()) {
          return source_registration.status();
        }
        source_registration.value().role = internal::collection::SegmentRole::sealed;
        registrations.push_back(std::move(source_registration).value());
      }
    }

    auto active = make_active_registration(schema,
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
      std::uint64_t target_generation,
      bool preserve_source_row_ids) -> core::Result<ReplacementBuildData> {
    ReplacementBuildData result;
    std::uint64_t next_row{};
    for (const auto &[logical_id, version] : snapshot.versions) {
      if (!address_is_source(version.address, sources)) {
        continue;
      }
      const auto target_row =
          preserve_source_row_ids ? static_cast<std::uint64_t>(version.address.row_id) : next_row++;
      internal::collection::RowAddress target{target_segment_id,
                                              target_generation,
                                              core::SegmentRowId(target_row)};
      result.replacements.push_back({logical_id, version.address, target, version.upsert_sequence});
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
                   "Flat target build did not publish manifest v2");
    }
    auto manifest = std::move(*loaded.value());
    const auto target_name =
        internal::collection::detail::flat_segment_name(control_state_.target_segment_id);
    const auto target = std::ranges::find_if(manifest.segments, [&](const auto &entry) {
      return entry.segment_id == target_name &&
             entry.generation == control_state_.target_generation;
    });
    if (target == manifest.segments.end()) {
      return error(core::StatusCode::corruption,
                   core::OperationStage::save,
                   core::StatusDetail::malformed_struct,
                   "published manifest omits the Flat replacement target");
    }
    target->lifecycle = internal::collection::SegmentLifecycleV2::sealed;
    target->source_retention.clear();
    for (const auto &source : control_state_.sources) {
      const auto source_name = internal::collection::detail::flat_segment_name(source.segment_id);
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
            : internal::collection::detail::flat_segment_name(state.target_segment_id);
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
                   "Gate-10 state says manifest-published but the target is absent");
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
        const auto source_name = internal::collection::detail::flat_segment_name(source.segment_id);
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
                     "Flat segment namespace is exhausted");
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

      internal::collection::CollectionSchema schema{options_.dim,
                                                    options_.metric,
                                                    options_.scalar_type,
                                                    options_.max_logical_id_bytes};
      auto successor = make_active_registration(schema,
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
      auto rotated =
          implementation_->rotate_to_successor(std::move(successor).value(),
                                               checkpoint_context,
                                               [&](const internal::collection::ActiveRotationReceipt
                                                       &receipt) {
                                                 control_state_.active_segment_id =
                                                     receipt.successor_segment_id;
                                                 control_state_.active_generation =
                                                     receipt.successor_generation;
                                                 control_state_.wal_cut =
                                                     receipt.checkpoint.wal_cut;
                                                 control_state_.phase = internal::collection::
                                                     CollectionControlPhase::successor_active;
                                                 return internal::collection::
                                                     CollectionControlStore::save(options_.root,
                                                                                  control_state_);
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
                                               control_state_.target_generation,
                                               true);
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
    disk::DiskFlatPublicationOptions publication;
    publication.collection_root = options_.root;
    publication.segment_id =
        internal::collection::detail::flat_segment_name(control_state_.target_segment_id);
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
    auto built = internal::collection::detail::build_collection_flat_target(schema,
                                                                            build_data.value().rows,
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

    for (const auto &source : control_state_.sources) {
      if (auto entry = pinned->find_segment(source.segment_id, source.generation)) {
        pending_gc_.push_back(
            {internal::collection::detail::flat_segment_name(source.segment_id), {}, entry});
      }
    }
    internal::collection::SegmentRegistration target;
    target.segment_id = control_state_.target_segment_id;
    target.generation = control_state_.target_generation;
    target.role = internal::collection::SegmentRole::sealed;
    target.segment = std::move(built_target.segment);
    target.rows = build_data.value().rows;
    status = implementation_->install_segment_replacement(control_state_.sources,
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

    CollectionSealReceipt receipt;
    receipt.source_segment_id = control_state_.sources.front().segment_id;
    receipt.successor_segment_id = control_state_.active_segment_id;
    receipt.sealed_segment_id = control_state_.target_segment_id;
    receipt.wal_cut = control_state_.wal_cut;
    receipt.sealed_rows = build_data.value().live_rows;
    receipt.sealed_bytes = built_target.artifact_bytes;
    receipt.manifest_generation = control_state_.manifest_generation;
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
        collect_replacement_rows(*pinned, sources, target_segment_id, kTargetGeneration, false);
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

    ::alaya::disk::DiskFlatPublicationOptions publication;
    publication.collection_root = options_.root;
    publication.segment_id = internal::collection::detail::flat_segment_name(target_segment_id);
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
    auto built = internal::collection::detail::build_collection_flat_target(schema,
                                                                            build_data.value().rows,
                                                                            publication,
                                                                            build_context);
    if (!built.ok()) {
      return built.status();
    }
    auto built_target = std::move(built).value();
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
      const auto source_name = internal::collection::detail::flat_segment_name(source.segment_id);
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
                              : internal::collection::detail::flat_segment_name(
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
    return "format=1\npublic_version=" + std::string(kCollectionPublicVersion) +
           "\ndim=" + std::to_string(options.dim) +
           "\nmetric=" + std::to_string(static_cast<unsigned>(options.metric)) +
           "\nscalar_type=" + std::to_string(static_cast<unsigned>(options.scalar_type)) +
           "\ntarget_algorithm=" + std::to_string(options.target_algorithm) +
           "\nquantization=" + std::to_string(static_cast<unsigned>(options.quantization)) +
           "\nbuild_threads=" + std::to_string(options.build_threads) +
           "\nef_construction=" + std::to_string(options.ef_construction) +
           "\nmax_logical_id_bytes=" + std::to_string(options.max_logical_id_bytes) +
           "\nimported_legacy_layout=" + std::to_string(options.imported_legacy_layout ? 1 : 0) +
           "\nactive_segment_id=" + std::to_string(kActiveSegmentId) +
           "\nactive_generation=" + std::to_string(kActiveSegmentGeneration) + "\n";
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
      if (fields.size() != 14 || required("format") != "1" ||
          required("public_version") != kCollectionPublicVersion ||
          required("checksum") != internal::collection::sha256(prefix).hex() ||
          parse_u64(required("active_segment_id")) != kActiveSegmentId ||
          parse_u64(required("active_generation")) != kActiveSegmentGeneration ||
          parse_u64(required("imported_legacy_layout")) > 1) {
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
      options.ef_construction = static_cast<std::uint32_t>(parse_u64(required("ef_construction")));
      options.max_logical_id_bytes = parse_u64(required("max_logical_id_bytes"));
      options.imported_legacy_layout = parse_u64(required("imported_legacy_layout")) == 1;
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
  bool imported_{};
  mutable std::mutex control_mutex_{};
  internal::collection::CollectionControlState control_state_{};
  std::vector<PendingGcCandidate> pending_gc_{};
};

}  // namespace alaya
