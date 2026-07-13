// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "index/collection/detail/canonical_flat_segment.hpp"
#include "index/collection/legacy_importer.hpp"
#include "index/collection/sha256.hpp"
#include "utils/platform_fs.hpp"

namespace alaya {

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
  bool imported_legacy_layout{};
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
};

struct CollectionStatistics {
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
  internal::collection::LifecycleState lifecycle{internal::collection::LifecycleState::open};
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
      auto opened = open_segmented(options);
      if (!opened.ok()) {
        return opened.status();
      }
      auto result = std::shared_ptr<Collection>(
          new Collection(std::move(options), std::move(opened).value(), false));
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
          return std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                            std::move(imported).value().collection,
                                                            true));
        }
        auto opened = open_segmented(options.value());
        if (!opened.ok()) {
          return opened.status();
        }
        return std::shared_ptr<Collection>(
            new Collection(std::move(options).value(), std::move(opened).value(), false));
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
      return std::shared_ptr<Collection>(
          new Collection(std::move(options), std::move(imported).value().collection, true));
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
    return implementation_->mutate_batch(request, context);
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
    if (query.rows != 1) {
      return error(core::StatusCode::invalid_argument,
                   core::OperationStage::validation,
                   core::StatusDetail::malformed_struct,
                   "canonical Collection single search requires exactly one query row");
    }
    return execute_search(query, options, context);
  }

  [[nodiscard]] auto search(const core::TypedTensorView &query, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    core::SearchContext context;
    return search(query, options, context);
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  const core::SearchOptions &options,
                                  core::SearchContext &context)
      -> core::Result<CollectionSearchResponse> {
    return execute_search(queries, options, context);
  }

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse> {
    core::SearchOptions options(top_k);
    core::SearchContext context;
    return batch_search(queries, options, context);
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
    return implementation_->checkpoint(context);
  }

  [[nodiscard]] auto checkpoint() -> core::Result<CollectionCheckpointReceipt> {
    core::CheckpointContext context;
    context.durability_target = core::DurabilityTarget::full_checkpoint;
    return checkpoint(context);
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
  inline static constexpr std::uint64_t kActiveSegmentId = 2;
  inline static constexpr std::uint64_t kActiveSegmentGeneration = 1;
  inline static constexpr std::string_view kFacadeNamespace{"collection_facade_v1"};
  inline static constexpr std::string_view kFacadeSchemaFilename{"schema.v1"};

  Collection(CollectionOptions options,
             std::shared_ptr<internal::collection::SegmentedCollection> implementation,
             bool imported)
      : options_(std::move(options)),
        implementation_(std::move(implementation)),
        imported_(imported) {}

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
      const internal::collection::CollectionSchema &schema)
      -> core::Result<internal::collection::SegmentRegistration> {
    return internal::collection::detail::make_canonical_flat_registration(schema,
                                                                          kActiveSegmentId,
                                                                          kActiveSegmentGeneration);
  }

  [[nodiscard]] static auto open_segmented(const CollectionOptions &options)
      -> core::Result<std::shared_ptr<internal::collection::SegmentedCollection>> {
    internal::collection::CollectionSchema schema{options.dim,
                                                  options.metric,
                                                  options.scalar_type,
                                                  options.max_logical_id_bytes};
    auto active = make_active_registration(schema);
    if (!active.ok()) {
      return active.status();
    }
    internal::collection::CollectionConfig config;
    config.features.wal_coordinator = true;
    config.wal.root = options.root;
    std::vector<internal::collection::SegmentRegistration> registrations;
    registrations.push_back(std::move(active).value());
    return internal::collection::SegmentedCollection::open(schema,
                                                           std::move(registrations),
                                                           std::move(config));
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
    return implementation_->write(request, context);
  }

  [[nodiscard]] auto execute_search(const core::TypedTensorView &queries,
                                    const core::SearchOptions &options,
                                    core::SearchContext &context)
      -> core::Result<CollectionSearchResponse> {
    internal::collection::CollectionSearchRequest request;
    request.queries = queries;
    request.options = options;
    request.context = std::addressof(context);
    auto result = implementation_->search(request);
    if (!result.ok()) {
      return result.status();
    }
    CollectionSearchResponse response;
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
};

}  // namespace alaya
