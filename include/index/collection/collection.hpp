// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
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
#include "index/collection/process_lock.hpp"
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

struct CollectionOpenOptions {
  bool read_only{};
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
      -> core::Result<std::shared_ptr<Collection>>;

  [[nodiscard]] static auto open(const std::filesystem::path &root)
      -> core::Result<std::shared_ptr<Collection>>;

  [[nodiscard]] static auto open(const std::filesystem::path &root,
                                 CollectionOpenOptions open_options)
      -> core::Result<std::shared_ptr<Collection>>;

  [[nodiscard]] auto add(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt>;

  [[nodiscard]] auto upsert(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt>;

  [[nodiscard]] auto replace(const CollectionItem &item, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt>;

  [[nodiscard]] auto remove(const core::LogicalId &logical_id, CollectionWriteOptions options = {})
      -> core::Result<CollectionMutationReceipt>;

  [[nodiscard]] auto mutate_batch(
      std::span<const CollectionBatchRow> rows,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt>;

  [[nodiscard]] auto add_batch(
      std::span<const CollectionItem> items,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt>;

  [[nodiscard]] auto upsert_batch(
      std::span<const CollectionItem> items,
      CollectionBatchMutationMode mode = CollectionBatchMutationMode::per_row_independent,
      CollectionWriteOptions options = {}) -> core::Result<CollectionBatchMutationReceipt>;

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            const core::SearchOptions &options,
                            core::SearchContext &context) -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            const core::SearchOptions &options,
                            core::SearchContext &context,
                            const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto search(const core::TypedTensorView &query, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto search(const core::TypedTensorView &query,
                            std::uint64_t top_k,
                            const CollectionFilter &filter,
                            core::FilterPolicy policy = core::FilterPolicy::automatic)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  const core::SearchOptions &options,
                                  core::SearchContext &context)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  const core::SearchOptions &options,
                                  core::SearchContext &context,
                                  const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries, std::uint64_t top_k)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto batch_search(const core::TypedTensorView &queries,
                                  std::uint64_t top_k,
                                  const CollectionFilter &filter,
                                  core::FilterPolicy policy = core::FilterPolicy::automatic)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] auto get_by_id(const core::LogicalId &logical_id,
                               CollectionProjection projection = CollectionProjection::all)
      -> core::Result<CollectionRecord>;

  [[nodiscard]] auto records(CollectionProjection projection = CollectionProjection::all,
                             std::size_t limit = std::numeric_limits<std::size_t>::max())
      -> core::Result<std::vector<CollectionRecord>>;

  [[nodiscard]] auto scan(const CollectionFilter &filter,
                          std::size_t limit,
                          CollectionProjection projection = CollectionProjection::all)
      -> core::Result<std::vector<CollectionRecord>>;

  [[nodiscard]] auto checkpoint(core::CheckpointContext &context)
      -> core::Result<CollectionCheckpointReceipt>;

  [[nodiscard]] auto checkpoint() -> core::Result<CollectionCheckpointReceipt>;

  [[nodiscard]] auto consolidate(CollectionConsolidateOptions options = {})
      -> core::Result<CollectionConsolidateReceipt>;

  [[nodiscard]] auto seal(CollectionSealOptions options = {})
      -> core::Result<CollectionSealReceipt>;

  [[nodiscard]] auto seal(core::SealContext &context, CollectionSealOptions options = {})
      -> core::Result<CollectionSealReceipt>;

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
      -> core::Result<CollectionRotationHandle>;

  [[nodiscard]] auto prepare_successor(core::SealContext &context,
                                       CollectionSealOptions options = {})
      -> core::Result<CollectionRotationHandle>;

  [[nodiscard]] auto rotate_to_successor(const CollectionRotationHandle &handle)
      -> core::Result<CollectionSealReceipt>;

  [[nodiscard]] auto rotate_to_successor(const CollectionRotationHandle &handle,
                                         core::SealContext &context)
      -> core::Result<CollectionSealReceipt>;

  [[nodiscard]] auto compact() -> core::Result<CollectionCompactReceipt>;

  [[nodiscard]] auto compact(core::SealContext &context) -> core::Result<CollectionCompactReceipt>;

  [[nodiscard]] auto gc() -> core::Result<CollectionGcReceipt>;

  [[nodiscard]] auto stats() const -> CollectionStatistics;

  [[nodiscard]] auto size() const -> core::RowCount { return stats().size; }
  [[nodiscard]] auto options() const -> const CollectionOptions & { return options_; }
  [[nodiscard]] auto root() const -> const std::filesystem::path & { return options_.root; }
  [[nodiscard]] auto read_only() const noexcept -> bool { return read_only_; }
  [[nodiscard]] auto target_algorithm() const noexcept -> core::AlgorithmId {
    return options_.target_algorithm;
  }
  [[nodiscard]] auto active_algorithm() const noexcept -> core::AlgorithmId {
    return options_.active_engine;
  }
  [[nodiscard]] auto target_implementation_key() const -> std::string_view;
  [[nodiscard]] auto target_engine_factory_key() const -> std::string_view;

  [[nodiscard]] auto close() -> core::Status;

 private:
  friend class internal::collection::CollectionTestAccess;

  inline static constexpr std::uint64_t kActiveSegmentId = 2;
  inline static constexpr std::uint64_t kActiveSegmentGeneration = 1;
  inline static constexpr std::string_view kFacadeNamespace{"collection_facade_v1"};
  inline static constexpr std::string_view kFacadeSchemaFilename{"schema.v1"};

  Collection(CollectionOptions options,
             std::shared_ptr<internal::collection::SegmentedCollection> implementation,
             internal::collection::CollectionControlState control_state,
             bool read_only,
             std::unique_ptr<internal::collection::CollectionProcessLock> process_lock)
      : options_(std::move(options)),
        process_lock_(std::move(process_lock)),
        implementation_(std::move(implementation)),
        control_state_(std::move(control_state)),
        read_only_(read_only) {}

  [[nodiscard]] static auto error(core::StatusCode code,
                                  core::OperationStage stage,
                                  core::StatusDetail detail,
                                  std::string diagnostic) -> core::Status;

  [[nodiscard]] static auto readonly_open_requires_recovery(std::string diagnostic) -> core::Status;

  [[nodiscard]] auto ensure_writable(core::OperationStage stage) const -> core::Status;

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
      -> BuildAlgorithmResolution;

  [[nodiscard]] static auto validate_options(const CollectionOptions &options,
                                             core::OperationStage stage) -> core::Status;

  static auto active_laser_dir(const std::filesystem::path &root,
                               std::uint64_t segment_id,
                               std::uint64_t generation) -> std::filesystem::path;

  // B-09 orphan reclamation. The durable control state is part of the reachability
  // root: successor-active/building/manifest-published recovery reopens every source
  // before completing replacement. Keep those paths until a later idle open; an
  // already-open fd cannot make unlink safe across a second process crash.
  static void sweep_orphan_active_laser_dirs(
      const std::filesystem::path &root,
      const internal::collection::CollectionControlState &control_state);

  // Ruling 12: physical row capacity of the active LASER segment. Default 4096; when
  // auto_seal_rows is set, keep the capacity strictly above it (churn headroom) so
  // the auto-seal threshold can never exceed the physical capacity.
  [[nodiscard]] static auto checked_active_laser_capacity(std::uint64_t auto_seal_rows)
      -> std::optional<std::size_t>;

  static auto active_laser_capacity(const CollectionOptions &options) -> std::size_t {
    return checked_active_laser_capacity(options.auto_seal_rows).value();
  }

  // Materialize a fresh empty active LASER segment directory (create-time / rotate).
  [[nodiscard]] static auto create_active_laser_segment(const CollectionOptions &options,
                                                        std::uint64_t segment_id,
                                                        std::uint64_t generation) -> core::Status;

  // Build the active-mutable registration for the configured active engine. flat is
  // an in-memory exact table; laser OPENS its existing on-disk directory (created at
  // Collection::create / rotate) -- a missing directory on open is corruption, never
  // a silent re-create that would drop committed rows.
  [[nodiscard]] static auto make_active_registration(
      const CollectionOptions &options,
      std::uint64_t segment_id = kActiveSegmentId,
      std::uint64_t generation = kActiveSegmentGeneration)
      -> core::Result<internal::collection::SegmentRegistration>;

  [[nodiscard]] static auto numeric_segment_id(std::string_view segment_id) -> std::uint64_t;

  [[nodiscard]] static auto open_segmented(
      const CollectionOptions &options,
      const internal::collection::CollectionControlState &control_state,
      bool read_only = false)
      -> core::Result<std::shared_ptr<internal::collection::SegmentedCollection>>;

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
                                  CollectionSealFailPoint point);

  [[nodiscard]] static auto address_is_source(
      const internal::collection::RowAddress &address,
      std::span<const internal::collection::RowAddress> sources) -> bool;

  [[nodiscard]] static auto collect_replacement_rows(
      const internal::collection::RoutingSnapshot &snapshot,
      std::span<const internal::collection::RowAddress> sources,
      std::uint64_t target_segment_id,
      std::uint64_t target_generation) -> core::Result<ReplacementBuildData>;

  [[nodiscard]] auto checkpoint_locked(core::CheckpointContext &context)
      -> core::Result<CollectionCheckpointReceipt>;

  [[nodiscard]] auto patch_published_target_manifest() -> core::Status;

  [[nodiscard]] static auto normalize_control_state_before_open(
      const std::filesystem::path &root,
      internal::collection::CollectionControlState &state) -> core::Status;

  [[nodiscard]] auto recover_control_state() -> core::Status;

  [[nodiscard]] auto seal_locked(core::SealContext &context, const CollectionSealOptions &options)
      -> core::Result<CollectionSealReceipt>;

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
      -> core::Result<CollectionRotationHandle>;

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
      -> core::Result<CollectionSealReceipt>;

  [[nodiscard]] auto verify_flat_exports(
      const internal::collection::RoutingSnapshot &snapshot,
      std::span<const internal::collection::RowAddress> sources) const -> core::Status;

  [[nodiscard]] auto compact_locked(core::SealContext &context)
      -> core::Result<CollectionCompactReceipt>;

  [[nodiscard]] auto gc_locked() -> core::Result<CollectionGcReceipt>;

  [[nodiscard]] auto write(const CollectionItem &item,
                           internal::collection::WriteMode mode,
                           CollectionWriteOptions options)
      -> core::Result<CollectionMutationReceipt>;

  void maybe_auto_seal() noexcept;

  [[nodiscard]] auto execute_search(const core::TypedTensorView &queries,
                                    const core::SearchOptions &options,
                                    core::SearchContext &context,
                                    const CollectionFilter &filter)
      -> core::Result<CollectionSearchResponse>;

  [[nodiscard]] static auto facade_schema_path(const std::filesystem::path &root)
      -> std::filesystem::path;

  [[nodiscard]] static auto schema_prefix(const CollectionOptions &options) -> std::string;

  [[nodiscard]] static auto write_facade_schema(const CollectionOptions &options) -> core::Status;

  [[nodiscard]] static auto parse_u64(std::string_view value) -> std::uint64_t;

  [[nodiscard]] static auto read_facade_schema(const std::filesystem::path &root)
      -> core::Result<CollectionOptions>;

  CollectionOptions options_{};
  std::unique_ptr<internal::collection::CollectionProcessLock> process_lock_{};
  std::shared_ptr<internal::collection::SegmentedCollection> implementation_{};
  mutable std::mutex control_mutex_{};
  internal::collection::CollectionControlState control_state_{};
  std::vector<PendingGcCandidate> pending_gc_{};
  std::optional<PendingRotation> pending_rotation_{};
  bool read_only_{};
  std::atomic<bool> closed_{false};
};

}  // namespace alaya
