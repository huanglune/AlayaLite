// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/collection/artifact_transaction.hpp"
#include "index/collection/logical_wal.hpp"
#include "index/collection/mutation_wal_codec.hpp"
#include "index/disk/disk_engine_registry.hpp"
#include "index/graph/diskann/diskann_index.hpp"
#include "platform/detect.hpp"
#include "platform/fs.hpp"

namespace alaya::disk {

class DiskAnnSegmentLegacyFactory;
class DiskAnnMutableSegmentFactory;

// DiskANN search knobs remain algorithm-keyed. The scratch ceiling is fixed at
// open time because the retained DiskANN index owns its bounded ThreadData
// pool; requests above that ceiling are rejected instead of allocating an
// ungoverned side pool in the adapter.
struct DiskAnnSegmentSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t search_list_size{100};
  std::uint32_t rerank_count{};
  bool use_pq{true};
  bool rerank{true};
  bool deterministic{};
  std::uint8_t reserved_bytes[5]{};
  std::uint64_t reserved[3]{};

  DiskAnnSegmentSearchExtension()
      : header(core::current_struct_header<DiskAnnSegmentSearchExtension>()) {}
};

[[nodiscard]] inline auto make_diskann_segment_search_extension(
    const DiskAnnSegmentSearchExtension &options) -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::diskann;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

// Collection-internal Gate 8-B writer options.  This type is intentionally not
// routed through the SDK/Python factory surface.  Mutable operation tables are
// created only by DiskAnnMutableSegmentFactory while its independent feature
// bit is enabled.
struct DiskAnnMutableSegmentOptions {
  std::filesystem::path collection_root{};
  std::string segment_id{"seg_00000001"};
  std::uint64_t collection_segment_id{1};
  std::uint64_t segment_generation{1};
  std::uint64_t minimum_next_op_id{1};
  internal::collection::ArtifactTransactionFailPoint checkpoint_fail_point{
      internal::collection::ArtifactTransactionFailPoint::none};
};

struct DiskAnnMutableMutationStats {
  std::uint64_t prepared{};
  std::uint64_t staged{};
  std::uint64_t committed{};
  std::uint64_t applied{};
  std::uint64_t replayed{};
  std::uint64_t aborted{};
  std::uint64_t logical_dirty_bytes{};
};

// Read-only adapter over DiskANNIndex. The retained index still owns the full
// cached beam/page loop and its bounded ThreadData pool; this class validates
// contract-v3 requests and translates native label/distance rows into the
// compact SearchResponse schema.
class DiskAnnSegment {
 public:
  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::diskann;
  static constexpr std::string_view kFormatName{"diskann"};
  static constexpr std::string_view kMetaArtifactName{"meta"};
  static constexpr std::string_view kIndexArtifactName{"index"};
  static constexpr std::string_view kIdsArtifactName{"ids"};
  static constexpr std::string_view kCacheIdsArtifactName{"cache_ids"};
  static constexpr std::string_view kCacheNodesArtifactName{"cache_nodes"};
  static constexpr std::string_view kPqPivotsArtifactName{"pq_pivots"};
  static constexpr std::string_view kPqCompressedArtifactName{"pq_compressed"};
  static constexpr std::string_view kSlotsArtifactName{"slots"};
  static constexpr std::string_view kMutableStateArtifactName{"mutable_state"};
  static constexpr std::string_view kMutableStateFilename{"diskann_mutable_state.bin"};
  static constexpr std::uint32_t kSearchThreads = 4;
  static constexpr std::uint32_t kBeamWidth = 4;
  static constexpr std::uint32_t kScratchSearchListSize =
      diskann::kDefaultDiskANNScratchSearchListSize;

  DiskAnnSegment(const DiskAnnSegment &) = delete;
  auto operator=(const DiskAnnSegment &) -> DiskAnnSegment & = delete;
  DiskAnnSegment(DiskAnnSegment &&) = delete;
  auto operator=(DiskAnnSegment &&) -> DiskAnnSegment & = delete;

  ~DiskAnnSegment() {
    // Native PageReader/reactor users must die before an ephemeral writer
    // generation is reclaimed. Async erased calls pin the owning Segment until
    // their exactly-once completion, so reaching this destructor is the final
    // lifetime barrier.
    native_.reset();
    if (owns_working_directory_) {
      std::error_code error;
      std::filesystem::remove_all(directory_, error);
    }
  }

  [[nodiscard]] static auto open(core::ArtifactView artifact,
                                 const core::OpenOptions &options,
                                 core::OpenContext &context)
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    return open_impl(artifact, options, context, std::nullopt);
  }

 private:
  [[nodiscard]] static auto open_impl(
      core::ArtifactView artifact,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const std::optional<DiskAnnMutableSegmentOptions> &mutable_options)
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    if (!core::is_current_struct(artifact) || !core::is_current_struct(options) ||
        !core::is_current_struct(context)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "DiskAnnSegment open received an incompatible v3 struct");
    }
    auto status = core::validate_runtime_control(context.deadline,
                                                 context.cancellation,
                                                 core::OperationStage::open);
    if (!status.ok()) {
      return status;
    }
    const auto meta_path_view = artifact.find(kMetaArtifactName);
    if (meta_path_view.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "DiskAnnSegment open requires the meta artifact");
    }
    try {
      const auto meta_path = std::filesystem::path(meta_path_view);
      const auto source_directory = meta_path.parent_path();
      const bool mutable_mode = mutable_options.has_value();
      if (mutable_mode &&
          (mutable_options->collection_root.empty() || mutable_options->segment_id.empty() ||
           mutable_options->collection_segment_id == 0 ||
           mutable_options->segment_generation == 0 || mutable_options->minimum_next_op_id == 0)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnn mutable open options are incomplete");
      }
      const auto meta = read_meta_summary(meta_path);
      if (meta.dim > std::numeric_limits<std::uint32_t>::max() ||
          meta.max_slot_id > std::numeric_limits<std::uint32_t>::max() ||
          meta.live_count > meta.max_slot_id) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskANN metadata is outside the readonly segment limits");
      }

      std::vector<ArtifactSpec> specs{
          {kMetaArtifactName, "meta.bin", ResidentClass::resident},
          {kIndexArtifactName, "diskann.index", ResidentClass::on_disk},
          {kIdsArtifactName, "ids.bin", ResidentClass::resident},
          {kCacheIdsArtifactName, "cache_ids.bin", ResidentClass::cache},
          {kCacheNodesArtifactName, "cache_nodes.bin", ResidentClass::cache},
      };
      if (meta.has_pq) {
        specs.push_back({kPqPivotsArtifactName, "pq_pivots.bin", ResidentClass::resident});
        specs.push_back({kPqCompressedArtifactName, "pq_compressed.bin", ResidentClass::resident});
      }
      if (mutable_mode) {
        const auto slots = source_directory / "slots.bin";
        const auto state = source_directory / kMutableStateFilename;
        if (std::filesystem::exists(slots)) {
          specs.push_back({kSlotsArtifactName, "slots.bin", ResidentClass::resident});
        }
        if (std::filesystem::exists(state)) {
          specs.push_back(
              {kMutableStateArtifactName, kMutableStateFilename, ResidentClass::resident});
        }
        if (meta.live_count != meta.max_slot_id && (!std::filesystem::is_regular_file(slots) ||
                                                    !std::filesystem::is_regular_file(state))) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "tombstoned DiskANN checkpoint lacks slots/state artifacts");
        }
      }

      std::uint64_t artifact_bytes{};
      std::uint64_t resident_bytes{};
      std::uint64_t cache_bytes{};
      std::uint64_t graph_bytes{};
      for (const auto &spec : specs) {
        const auto supplied = artifact.find(spec.logical_name);
        if (supplied.empty()) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "DiskAnnSegment open is missing logical artifact '" +
                                         std::string(spec.logical_name) + "'");
        }
        const auto expected = (source_directory / spec.filename).lexically_normal();
        if (std::filesystem::path(supplied).lexically_normal() != expected) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "DiskAnnSegment ArtifactView paths disagree with the "
                                     "native DiskANN file family");
        }
        const auto size = checked_file_size(expected);
        checked_accumulate(artifact_bytes, size, "DiskANN artifact bytes overflow uint64");
        switch (spec.residency) {
          case ResidentClass::on_disk:
            graph_bytes = size;
            break;
          case ResidentClass::resident:
            checked_accumulate(resident_bytes, size, "DiskANN resident bytes overflow uint64");
            break;
          case ResidentClass::cache:
            checked_accumulate(cache_bytes, size, "DiskANN cache bytes overflow uint64");
            checked_accumulate(resident_bytes, size, "DiskANN resident bytes overflow uint64");
            break;
        }
      }

      auto scratch = estimate_scratch_bytes(meta, mutable_mode);
      if (!scratch.ok()) {
        return scratch.status();
      }
      status = core::require_lease(context.resident_lease,
                                   resident_bytes,
                                   core::OperationStage::open,
                                   "DiskAnnSegment resident lease is too small");
      if (!status.ok()) {
        return status;
      }
      status = core::require_lease(context.cache_lease,
                                   cache_bytes,
                                   core::OperationStage::open,
                                   "DiskAnnSegment cache lease is too small");
      if (!status.ok()) {
        return status;
      }
      status = core::require_lease(context.scratch_pool_lease,
                                   scratch.value(),
                                   core::OperationStage::open,
                                   "DiskAnnSegment scratch-pool lease is too small");
      if (!status.ok()) {
        return status;
      }
      std::uint64_t open_requests = specs.size();
      std::uint64_t open_bytes = artifact_bytes;
      if (mutable_mode && (!core::checked_multiply(open_requests, 2, open_requests) ||
                           !core::checked_multiply(open_bytes, 2, open_bytes))) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::open,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskAnn mutable open I/O accounting overflows uint64");
      }
      status = require_io_credits(context.io_credits,
                                  open_requests,
                                  open_bytes,
                                  core::OperationStage::open,
                                  "DiskAnnSegment open I/O credits are too small");
      if (!status.ok()) {
        return status;
      }

      auto load_directory = source_directory;
      bool owns_working_directory = false;
      WorkingDirectoryGuard working_guard;
      if (mutable_mode) {
        load_directory = make_working_copy(source_directory,
                                           mutable_options->collection_root,
                                           mutable_options->segment_id,
                                           specs);
        owns_working_directory = true;
        working_guard.path = load_directory;
      }
      auto native = make_native_index(load_directory, mutable_mode);
      if (native->dim() != meta.dim || native->size() != meta.live_count ||
          native->max_slot_id() != meta.max_slot_id) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskANN loaded state disagrees with meta.bin");
      }
      // A static readonly load cannot apply a persisted tombstone bitmap. Do
      // not return wrong rows from a mutated artifact; G8-B owns that reader.
      if (!mutable_mode && meta.live_count != meta.max_slot_id) {
        return unavailable(core::OperationStage::open,
                           "DiskAnnSegment readonly cannot open a tombstoned mutable artifact");
      }
      auto segment = std::unique_ptr<DiskAnnSegment>(new DiskAnnSegment(std::move(native),
                                                                        load_directory,
                                                                        meta,
                                                                        artifact_bytes,
                                                                        resident_bytes,
                                                                        cache_bytes,
                                                                        scratch.value(),
                                                                        graph_bytes,
                                                                        mutable_options,
                                                                        owns_working_directory));
      working_guard.release();
      if (mutable_mode) {
        status = segment->initialize_mutable_state();
        if (!status.ok()) {
          return status;
        }
      }
      return segment;
    } catch (const std::bad_alloc &error) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::open,
                                 core::StatusDetail::allocation_failure,
                                 error.what(),
                                 core::Retryability::retryable_with_backoff);
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }
  }

 public:
  [[nodiscard]] static auto open_directory(const std::filesystem::path &directory,
                                           const core::OpenOptions &options,
                                           core::OpenContext &context)
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    try {
      const auto meta = read_meta_summary(directory / "meta.bin");
      std::vector<std::string> paths;
      std::vector<core::ArtifactLocation> locations;
      const std::array<std::pair<std::string_view, std::string_view>, 7> known{{
          {kMetaArtifactName, "meta.bin"},
          {kIndexArtifactName, "diskann.index"},
          {kIdsArtifactName, "ids.bin"},
          {kCacheIdsArtifactName, "cache_ids.bin"},
          {kCacheNodesArtifactName, "cache_nodes.bin"},
          {kPqPivotsArtifactName, "pq_pivots.bin"},
          {kPqCompressedArtifactName, "pq_compressed.bin"},
      }};
      const auto count = meta.has_pq ? known.size() : known.size() - 2;
      paths.reserve(count);
      locations.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        paths.push_back((directory / known[index].second).string());
      }
      for (std::size_t index = 0; index < count; ++index) {
        locations.emplace_back(known[index].first, paths[index]);
      }
      return open(core::ArtifactView(locations), options, context);
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = meta_.format_version;
    descriptor.factory_version = 1;
    descriptor.dim = static_cast<std::uint32_t>(meta_.dim);
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::disk;
    descriptor.preprocessing = meta_.has_pq ? core::MetricPreprocessing::engine_quantized
                                            : core::MetricPreprocessing::none;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  [[nodiscard]] static auto make_search_extension(const DiskAnnSegmentSearchExtension &options)
      -> core::AlgorithmSearchExtension {
    return make_diskann_segment_search_extension(options);
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskAnnSegment single search requires exactly one query row");
    }
    return wait_native_search(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return wait_native_search(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    if (!mutable_options_.has_value()) {
      // Preserve the Gate 8-A readonly path: it neither needs the writer
      // visibility lock nor mutable lifecycle accounting.
      stats = core::SegmentStats{};
      stats.snapshot_version = 1;
      stats.live_rows = meta_.live_count;
      stats.allocated_rows = meta_.max_slot_id;
      stats.resident_bytes = resident_bytes_ + scratch_bytes_;
      stats.cache_bytes = cache_bytes_;
      stats.inflight_search = inflight_search_.load(std::memory_order_acquire);
      stats.health = core::SegmentHealth::healthy;
      return core::Status::success();
    }
    try {
      std::shared_lock native_lock(native_mutex_);
      stats = core::SegmentStats{};
      stats.snapshot_version = applied_watermark_;
      stats.live_rows = native_->live_count();
      stats.allocated_rows = native_->max_slot_id();
      stats.tombstone_rows = native_->tombstone_count();
      stats.pending_rows = pending_transactions_.load(std::memory_order_acquire);
      stats.resident_bytes = resident_bytes_ + scratch_bytes_;
      stats.cache_bytes = cache_bytes_;
      stats.dirty_bytes = logical_dirty_bytes_.load(std::memory_order_acquire);
      stats.inflight_search = inflight_search_.load(std::memory_order_acquire);
      stats.inflight_mutation = inflight_mutation_.load(std::memory_order_acquire);
      stats.health = health_.load(std::memory_order_acquire);
      stats.last_error = last_error_.load(std::memory_order_acquire);
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::stats);
    }
  }

  [[nodiscard]] auto mutable_mutation_stats() const noexcept -> DiskAnnMutableMutationStats {
    DiskAnnMutableMutationStats result;
    result.prepared = prepared_count_.load(std::memory_order_acquire);
    result.staged = staged_count_.load(std::memory_order_acquire);
    result.committed = committed_count_.load(std::memory_order_acquire);
    result.applied = applied_count_.load(std::memory_order_acquire);
    result.replayed = replayed_count_.load(std::memory_order_acquire);
    result.aborted = aborted_count_.load(std::memory_order_acquire);
    result.logical_dirty_bytes = logical_dirty_bytes_.load(std::memory_order_acquire);
    return result;
  }

  [[nodiscard]] auto minimum_next_op_id() const -> std::uint64_t {
    std::shared_lock lock(native_mutex_);
    return minimum_next_op_id_;
  }

  // Erase through the native coroutine operation table. AnySegment::search()
  // remains the stable synchronous start-then-wait wrapper; coroutine types do
  // not cross the contract-v3 boundary.
  [[nodiscard]] static auto into_any(std::unique_ptr<DiskAnnSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null DiskAnnSegment");
    }
    auto shared = std::shared_ptr<DiskAnnSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.enabled_operations = core::capability_bit(core::OperationCapability::search) |
                                core::capability_bit(core::OperationCapability::batch_search) |
                                core::capability_bit(core::OperationCapability::stats);
    config.concurrency.reentrant_search = true;
    config.concurrency.search_with_stage = false;
    config.concurrency.search_with_publish = false;
    config.concurrency.serial_mutation = true;
    config.concurrency.checkpoint_with_search = false;
    config.concurrency.native_async = true;
    config.concurrency.cooperative_cancel = true;
    config.concurrency.explicit_drain = false;
    auto descriptor = shared->descriptor();
    return core::AnySegment::from_raw(std::move(shared),
                                      std::addressof(native_operations()),
                                      std::move(descriptor),
                                      config);
  }

  [[nodiscard]] static auto into_mutable_any(std::unique_ptr<DiskAnnSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr || !segment->mutable_options_.has_value()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::readonly_instance,
                                 "DiskAnn mutable erasure requires a mutable instance");
    }
    auto shared = std::shared_ptr<DiskAnnSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = false;
    config.enabled_operations = core::capability_bit(core::OperationCapability::search) |
                                core::capability_bit(core::OperationCapability::batch_search) |
                                core::capability_bit(core::OperationCapability::mutation) |
                                core::capability_bit(core::OperationCapability::checkpoint) |
                                core::capability_bit(core::OperationCapability::stats) |
                                core::capability_bit(core::OperationCapability::close) |
                                core::capability_bit(core::OperationCapability::drain);
    config.concurrency.reentrant_search = true;
    config.concurrency.search_with_stage = true;
    // Native insert/remove is immediately visible. The Segment therefore
    // holds its visibility lock across apply and the Collection serializes
    // publish against search; graph traversal itself remains weakly
    // consistent, while the pinned logical map does the strict final filter.
    config.concurrency.search_with_publish = false;
    config.concurrency.serial_mutation = true;
    config.concurrency.checkpoint_with_search = false;
    config.concurrency.native_async = true;
    config.concurrency.cooperative_cancel = true;
    config.concurrency.explicit_drain = true;
    auto descriptor = shared->descriptor();
    descriptor.factory_version = 2;
    return core::AnySegment::from_raw(std::move(shared),
                                      std::addressof(mutable_operations()),
                                      std::move(descriptor),
                                      config);
  }

  [[nodiscard]] static auto into_roll_forward_any(std::unique_ptr<DiskAnnSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr || !segment->mutable_options_.has_value()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::readonly_instance,
                                 "DiskAnn roll-forward reader requires a checkpointed instance");
    }
    auto shared = std::shared_ptr<DiskAnnSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.enabled_operations = core::capability_bit(core::OperationCapability::search) |
                                core::capability_bit(core::OperationCapability::batch_search) |
                                core::capability_bit(core::OperationCapability::stats) |
                                core::capability_bit(core::OperationCapability::close) |
                                core::capability_bit(core::OperationCapability::drain);
    config.concurrency.reentrant_search = true;
    config.concurrency.search_with_stage = false;
    config.concurrency.search_with_publish = false;
    config.concurrency.serial_mutation = true;
    config.concurrency.checkpoint_with_search = false;
    config.concurrency.native_async = true;
    config.concurrency.cooperative_cancel = true;
    config.concurrency.explicit_drain = true;
    auto descriptor = shared->descriptor();
    descriptor.factory_version = 2;
    return core::AnySegment::from_raw(std::move(shared),
                                      std::addressof(roll_forward_operations()),
                                      std::move(descriptor),
                                      config);
  }

 private:
  friend class DiskAnnSegmentLegacyFactory;
  friend class DiskAnnMutableSegmentFactory;

  struct NativeMetaSummary {
    std::uint32_t format_version{};
    std::uint64_t num_points{};
    std::uint64_t dim{};
    std::uint32_t max_degree{};
    std::uint32_t medoid{};
    bool has_pq{};
    std::uint32_t pq_n_chunks{};
    std::uint64_t node_len{};
    std::uint64_t nodes_per_sector{};
    std::uint64_t max_slot_id{};
    std::uint64_t live_count{};
  };

  enum class ResidentClass { on_disk, resident, cache };

  struct ArtifactSpec {
    std::string_view logical_name;
    std::string_view filename;
    ResidentClass residency;
  };

  struct WorkingDirectoryGuard {
    ~WorkingDirectoryGuard() {
      if (!path.empty()) {
        std::error_code error;
        std::filesystem::remove_all(path, error);
      }
    }
    void release() noexcept { path.clear(); }
    std::filesystem::path path{};
  };

  struct ResolvedSearch {
    diskann::DiskANNSearchParams native{};
    std::uint32_t native_top_k{};
  };

  struct MutableSlotState {
    std::uint64_t label{};
    std::uint64_t op_id{};
    bool tombstone{};
  };

  struct MutableVisibility {
    std::set<std::uint64_t> live_labels{};
    std::uint64_t applied_watermark{};
  };

  struct OwnedMutationRow {
    internal::collection::SegmentMutationAction action{
        internal::collection::SegmentMutationAction::write};
    std::uint64_t op_id{};
    std::uint64_t upsert_sequence{};
    internal::collection::RowAddress target{};
    std::optional<internal::collection::RowAddress> previous{};
    std::vector<float> vector{};
  };

  struct MutableTransaction {
    std::uint64_t transaction_id{};
    std::vector<OwnedMutationRow> rows{};
    std::uint64_t payload_bytes{};
    bool bundled{};
    bool staged{};
  };

  struct NativeOperationState final : core::detail::AsyncOperationState {
    std::shared_ptr<DiskAnnSegment> segment{};
    std::shared_ptr<const MutableVisibility> visibility{};
    std::vector<std::uint64_t> labels{};
    std::vector<float> distances{};
    std::vector<std::uint32_t> counts{};
    std::vector<diskann::SearchStats> native_stats{};
    std::vector<float> contiguous_queries{};
    std::atomic<core::StatusCode> observed_control{core::StatusCode::ok};
    bool admitted{};

    [[nodiscard]] static auto probe_cancelled(const void *raw) noexcept -> bool {
      auto &state =
          *const_cast<NativeOperationState *>(static_cast<const NativeOperationState *>(raw));
      const auto control = core::validate_runtime_control(state.context.deadline,
                                                          state.context.cancellation,
                                                          core::OperationStage::search);
      if (control.ok()) {
        return false;
      }
      auto expected = core::StatusCode::ok;
      (void)state.observed_control.compare_exchange_strong(expected,
                                                           control.code(),
                                                           std::memory_order_acq_rel);
      return true;
    }

    [[nodiscard]] auto terminal_control() const -> core::Status {
      auto control = core::validate_runtime_control(context.deadline,
                                                    context.cancellation,
                                                    core::OperationStage::search);
      if (!control.ok()) {
        return control;
      }
      if (observed_control.load(std::memory_order_acquire) == core::StatusCode::deadline_exceeded) {
        return core::Status::error(core::StatusCode::deadline_exceeded,
                                   core::OperationStage::search,
                                   core::StatusDetail::deadline_reached,
                                   "operation deadline was observed at a drained DiskANN wave");
      }
      if (observed_control.load(std::memory_order_acquire) == core::StatusCode::cancelled) {
        return core::Status::error(core::StatusCode::cancelled,
                                   core::OperationStage::search,
                                   core::StatusDetail::cancellation_requested,
                                   "cancellation was observed at a drained DiskANN wave");
      }
      return core::Status::success();
    }

    void run(const std::shared_ptr<NativeOperationState> &self) noexcept {
      core::Status status;
      try {
        status = segment->execute_native_async(*this);
      } catch (...) {
        status = core::status_from_exception(core::OperationStage::search);
      }
      const auto control = terminal_control();
      if (!control.ok() && status.ok()) {
        status = segment->finish_controlled(request,
                                            control,
                                            request.queries.rows,
                                            request.response->offsets[request.queries.rows]);
      } else if (!status.ok() && request.response != nullptr &&
                 core::is_current_struct(*request.response) &&
                 request.options.partial_result_policy == core::PartialResultPolicy::discard) {
        request.response->invalidate(status);
      }
      finish_and_release(std::move(status), self);
    }

    void finish_and_release(core::Status status,
                            const std::shared_ptr<NativeOperationState> &self) noexcept {
      if (completion_started.exchange(true, std::memory_order_acq_rel)) {
        return;
      }
      auto delivery = [self, status = std::move(status)]() mutable {
        try {
          self->completion.callback(std::move(status));
        } catch (...) {
        }
        self->request.lifetime_pin.reset();
        self->completion.callback = {};
        if (self->admitted) {
          self->segment->end_search();
          self->admitted = false;
        }
      };
      context.lane.dispatch(std::move(delivery));
    }
  };

  class InflightGuard {
   public:
    explicit InflightGuard(std::atomic<std::uint64_t> &value) : value_(value) {
      value_.fetch_add(1, std::memory_order_acq_rel);
    }
    ~InflightGuard() { value_.fetch_sub(1, std::memory_order_acq_rel); }

   private:
    std::atomic<std::uint64_t> &value_;
  };

  [[nodiscard]] auto begin_search() -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    if (!admission_open_ ||
        health_.load(std::memory_order_acquire) == core::SegmentHealth::failed) {
      return core::Status::error(core::StatusCode::closed,
                                 core::OperationStage::admission,
                                 core::StatusDetail::operation_slot_absent,
                                 "DiskAnnSegment search admission is closed");
    }
    inflight_search_.fetch_add(1, std::memory_order_acq_rel);
    return core::Status::success();
  }

  void end_search() noexcept {
    inflight_search_.fetch_sub(1, std::memory_order_acq_rel);
    std::lock_guard lock(lifecycle_mutex_);
    lifecycle_changed_.notify_all();
  }

  [[nodiscard]] auto close_segment() noexcept -> core::Status {
    std::lock_guard lock(lifecycle_mutex_);
    admission_open_ = false;
    return core::Status::success();
  }

  [[nodiscard]] auto drain_segment(const core::Deadline &deadline) noexcept -> core::Status {
    try {
      std::unique_lock lock(lifecycle_mutex_);
      while (inflight_search_.load(std::memory_order_acquire) != 0 ||
             inflight_mutation_.load(std::memory_order_acquire) != 0) {
        if (deadline.expired()) {
          return core::Status::error(core::StatusCode::deadline_exceeded,
                                     core::OperationStage::drain,
                                     core::StatusDetail::deadline_reached,
                                     "DiskAnnSegment drain deadline was reached");
        }
        if (deadline.enabled) {
          lifecycle_changed_.wait_for(lock, std::chrono::milliseconds(1));
        } else {
          lifecycle_changed_.wait(lock);
        }
      }
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::drain);
    }
  }

  [[nodiscard]] static auto start_native_search(const std::shared_ptr<void> &instance,
                                                core::SearchRequest request,
                                                core::SearchCompletion completion) noexcept
      -> core::Result<core::OperationHandle> {
    try {
      auto state = std::make_shared<NativeOperationState>();
      state->segment = std::static_pointer_cast<DiskAnnSegment>(instance);
      if (state->segment->mutable_options_.has_value()) {
        auto admitted = state->segment->begin_search();
        if (!admitted.ok()) {
          return admitted;
        }
        state->admitted = true;
        state->visibility = std::atomic_load_explicit(&state->segment->mutable_visibility_,
                                                      std::memory_order_acquire);
      }
      try {
        state->request = std::move(request);
        state->completion = std::move(completion);
        state->bind_context();
        std::thread([state] {
          state->run(state);
        }).detach();
      } catch (...) {
        state->segment->end_search();
        state->admitted = false;
        throw;
      }
      return core::OperationHandle(state, [](void *raw) noexcept {
        static_cast<NativeOperationState *>(raw)->cancel();
      });
    } catch (...) {
      return core::status_from_exception(core::OperationStage::admission);
    }
  }

  [[nodiscard]] static auto native_operations() noexcept -> const core::AnySegmentOperationTable & {
    static const core::AnySegmentOperationTable operations = [] {
      core::AnySegmentOperationTable value{};
      value.table_size = sizeof(core::AnySegmentOperationTable);
      value.table_version = core::kOperationTableVersion;
      value.start_search = &start_native_search;
      value.start_batch_search = &start_native_search;
      value.stats = [](const std::shared_ptr<void> &instance,
                       core::SegmentStats &stats) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->stats(stats);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::stats);
        }
      };
      return value;
    }();
    return operations;
  }

  [[nodiscard]] static auto mutable_operations() noexcept
      -> const core::AnySegmentOperationTable & {
    static const core::AnySegmentOperationTable operations = [] {
      core::AnySegmentOperationTable value{};
      value.table_size = sizeof(core::AnySegmentOperationTable);
      value.table_version = core::kOperationTableVersion;
      value.start_search = &start_native_search;
      value.start_batch_search = &start_native_search;
      value.prepare_mutation = [](const std::shared_ptr<void> &instance,
                                  const core::OpaqueOperationRequest &request,
                                  core::MutationContext &context,
                                  core::MutationToken *token) noexcept -> core::Status {
        try {
          if (token == nullptr) {
            return malformed_mutation(core::OperationStage::mutation_prepare,
                                      "DiskAnn mutation token output is null");
          }
          return std::static_pointer_cast<DiskAnnSegment>(instance)->prepare_mutation(request,
                                                                                      context,
                                                                                      *token);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::mutation_prepare);
        }
      };
      value.stage_mutation = [](const std::shared_ptr<void> &instance,
                                core::MutationToken &token,
                                core::MutationContext &context) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->stage_mutation(token, context);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::mutation_stage);
        }
      };
      value.publish_mutation = [](const std::shared_ptr<void> &instance,
                                  core::MutationToken &token,
                                  core::MutationContext &context) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->publish_mutation(token,
                                                                                      context);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::mutation_publish);
        }
      };
      value.abort_mutation = [](const std::shared_ptr<void> &instance,
                                core::MutationToken &token,
                                core::MutationContext &context) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->abort_mutation(token, context);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::mutation_abort);
        }
      };
      value.replay_mutation = [](const std::shared_ptr<void> &instance,
                                 const core::OpaqueOperationRequest &request,
                                 core::MutationContext &context,
                                 core::MutationToken *) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->replay_mutation(request,
                                                                                     context);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::mutation_replay);
        }
      };
      value.checkpoint = [](const std::shared_ptr<void> &instance,
                            core::CheckpointContext &context,
                            core::CheckpointToken &token) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->checkpoint_mutable(context,
                                                                                        token);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::checkpoint);
        }
      };
      value.stats = [](const std::shared_ptr<void> &instance,
                       core::SegmentStats &stats) noexcept -> core::Status {
        try {
          return std::static_pointer_cast<DiskAnnSegment>(instance)->stats(stats);
        } catch (...) {
          return core::status_from_exception(core::OperationStage::stats);
        }
      };
      value.close = [](const std::shared_ptr<void> &instance) noexcept -> core::Status {
        return std::static_pointer_cast<DiskAnnSegment>(instance)->close_segment();
      };
      value.drain = [](const std::shared_ptr<void> &instance,
                       const core::Deadline &deadline) noexcept -> core::Status {
        return std::static_pointer_cast<DiskAnnSegment>(instance)->drain_segment(deadline);
      };
      return value;
    }();
    return operations;
  }

  [[nodiscard]] static auto roll_forward_operations() noexcept
      -> const core::AnySegmentOperationTable & {
    static const core::AnySegmentOperationTable operations = [] {
      core::AnySegmentOperationTable value{};
      value.table_size = sizeof(core::AnySegmentOperationTable);
      value.table_version = core::kOperationTableVersion;
      value.start_search = &start_native_search;
      value.start_batch_search = &start_native_search;
      value.stats = [](const std::shared_ptr<void> &instance,
                       core::SegmentStats &stats) noexcept -> core::Status {
        return std::static_pointer_cast<DiskAnnSegment>(instance)->stats(stats);
      };
      value.close = [](const std::shared_ptr<void> &instance) noexcept -> core::Status {
        return std::static_pointer_cast<DiskAnnSegment>(instance)->close_segment();
      };
      value.drain = [](const std::shared_ptr<void> &instance,
                       const core::Deadline &deadline) noexcept -> core::Status {
        return std::static_pointer_cast<DiskAnnSegment>(instance)->drain_segment(deadline);
      };
      return value;
    }();
    return operations;
  }

  [[nodiscard]] auto wait_native_search(const core::SearchRequest &request) const -> core::Status {
    if (request.context == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::null_data,
                                 "DiskAnnSegment SearchContext is null");
    }
    core::SearchContext wait_context = *request.context;
    wait_context.lane = core::RuntimeLane{};
    auto wait_request = request;
    wait_request.context = std::addressof(wait_context);

    struct WaitState {
      std::mutex mutex;
      std::condition_variable ready;
      bool done{};
      core::Status status{};
    };
    auto wait = std::make_shared<WaitState>();
    core::SearchCompletion completion([wait](core::Status status) {
      {
        std::lock_guard lock(wait->mutex);
        wait->status = std::move(status);
        wait->done = true;
      }
      wait->ready.notify_one();
    });

    // The method itself keeps `this` alive until the native operation's safe
    // completion. Erased async calls use the owning shared_ptr from AnySegment.
    auto self = std::shared_ptr<DiskAnnSegment>(const_cast<DiskAnnSegment *>(this),
                                                [](DiskAnnSegment
                                                       *) noexcept {  // NOLINT(readability/casting)
                                                });
    auto started = start_native_search(self, std::move(wait_request), std::move(completion));
    if (!started.ok()) {
      return started.status();
    }
    auto handle = std::move(started).value();
    std::unique_lock lock(wait->mutex);
    wait->ready.wait(lock, [&] {
      return wait->done;
    });
    return wait->status;
  }

  DiskAnnSegment(std::unique_ptr<diskann::DiskANNIndex> native,
                 std::filesystem::path directory,
                 NativeMetaSummary meta,
                 std::uint64_t artifact_bytes,
                 std::uint64_t resident_bytes,
                 std::uint64_t cache_bytes,
                 std::uint64_t scratch_bytes,
                 std::uint64_t graph_bytes,
                 std::optional<DiskAnnMutableSegmentOptions> mutable_options,
                 bool owns_working_directory)
      : native_(std::move(native)),
        directory_(std::move(directory)),
        meta_(meta),
        artifact_bytes_(artifact_bytes),
        resident_bytes_(resident_bytes),
        cache_bytes_(cache_bytes),
        scratch_bytes_(scratch_bytes),
        graph_bytes_(graph_bytes),
        mutable_options_(std::move(mutable_options)),
        owns_working_directory_(owns_working_directory) {}

  template <class T>
  static void read_exact(std::ifstream &input, T &value, const char *diagnostic) {
    input.read(reinterpret_cast<char *>(std::addressof(value)), sizeof(value));
    if (!input) {
      throw std::invalid_argument(diagnostic);
    }
  }

  [[nodiscard]] static auto safe_path_component(std::string_view value) noexcept -> bool {
    return !value.empty() && value != "." && value != ".." &&
           value.find('/') == std::string_view::npos &&
           value.find('\\') == std::string_view::npos && value.find('\0') == std::string_view::npos;
  }

  [[nodiscard]] static auto make_working_copy(const std::filesystem::path &source,
                                              const std::filesystem::path &collection_root,
                                              std::string_view segment_id,
                                              const std::vector<ArtifactSpec> &specs)
      -> std::filesystem::path {
    if (!safe_path_component(segment_id)) {
      throw std::invalid_argument("DiskAnn mutable segment id is not one safe path component");
    }
    static std::atomic_uint64_t sequence{};
    const auto work_parent = collection_root / ".alaya_internal" / "diskann_mutable_v1" / "work";
    std::filesystem::create_directories(work_parent);
    const auto work =
        work_parent / (std::string(segment_id) + "_" + std::to_string(platform::get_pid()) + "_" +
                       std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    if (!std::filesystem::create_directory(work)) {
      throw std::runtime_error("cannot create DiskAnn mutable working generation");
    }
    try {
      for (const auto &spec : specs) {
        std::filesystem::copy_file(source / spec.filename,
                                   work / spec.filename,
                                   std::filesystem::copy_options::none);
      }
      return work;
    } catch (...) {
      std::error_code error;
      std::filesystem::remove_all(work, error);
      throw;
    }
  }

  [[nodiscard]] static auto make_native_index(const std::filesystem::path &directory,
                                              bool updatable)
      -> std::unique_ptr<diskann::DiskANNIndex> {
    auto native = std::make_unique<diskann::DiskANNIndex>();
    diskann::DiskANNLoadParams load;
    load.num_threads = kSearchThreads;
    load.beam_width = kBeamWidth;
    load.scratch_search_list_size = kScratchSearchListSize;
    load.updatable = updatable;
    load.search_page_cache = updatable;
    // The public kernel already owns and drains reconnect tasklets. Keep its
    // default auto backend; no Segment-private pool or I/O path is introduced.
    native->load(directory.string(), load);
    return native;
  }

  [[nodiscard]] static auto read_native_labels(const std::filesystem::path &path,
                                               std::uint64_t expected)
      -> std::vector<std::uint64_t> {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("DiskAnnSegment cannot open " + path.string());
    }
    std::uint64_t count{};
    read_exact(input, count, "DiskANN ids.bin is truncated");
    if (count != expected || count > std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument("DiskANN ids.bin count disagrees with meta.bin");
    }
    std::vector<std::uint64_t> labels(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char *>(labels.data()),
               static_cast<std::streamsize>(labels.size() * sizeof(std::uint64_t)));
    if (!input) {
      throw std::invalid_argument("DiskANN ids.bin label payload is truncated");
    }
    return labels;
  }

  static constexpr std::uint32_t kMutableStateMagic = 0x384d4144U;    // "DAM8"
  static constexpr std::uint32_t kMutableStateTrailer = 0x38444e45U;  // "END8"
  static constexpr std::uint32_t kMutableStateVersion = 1;
  static constexpr std::size_t kMutableStateFixedBytes = 48;
  static constexpr std::size_t kMutableStateRecordBytes = 24;

  struct DecodedMutableState {
    std::uint64_t applied_watermark{};
    std::uint64_t minimum_next_op_id{1};
    std::uint64_t checkpoint_generation{};
    std::vector<MutableSlotState> slots{};
  };

  [[nodiscard]] static auto decode_mutable_state(const std::filesystem::path &path,
                                                 std::uint64_t expected_slots)
      -> DecodedMutableState {
    constexpr std::size_t kMaximumStateBytes = 256U << 20U;
    const auto raw = platform::read_regular_file_bounded(path, kMaximumStateBytes);
    const auto bytes = std::span(reinterpret_cast<const std::byte *>(raw.data()), raw.size());
    if (bytes.size() < kMutableStateFixedBytes ||
        internal::collection::logical_wal_detail::get_u32(bytes, 0) != kMutableStateMagic ||
        internal::collection::logical_wal_detail::get_u32(bytes, 4) != kMutableStateVersion) {
      throw std::invalid_argument("DiskAnn mutable state header is invalid");
    }
    DecodedMutableState decoded;
    decoded.applied_watermark = internal::collection::logical_wal_detail::get_u64(bytes, 8);
    decoded.minimum_next_op_id = internal::collection::logical_wal_detail::get_u64(bytes, 16);
    decoded.checkpoint_generation = internal::collection::logical_wal_detail::get_u64(bytes, 24);
    const auto count = internal::collection::logical_wal_detail::get_u64(bytes, 32);
    std::uint64_t records_bytes{};
    std::uint64_t expected_size{};
    if (count != expected_slots || count > std::numeric_limits<std::size_t>::max() ||
        !core::checked_multiply(count, kMutableStateRecordBytes, records_bytes) ||
        !core::checked_add(records_bytes, kMutableStateFixedBytes, expected_size) ||
        expected_size != bytes.size()) {
      throw std::invalid_argument("DiskAnn mutable state slot count/length is invalid");
    }
    const auto expected_crc =
        internal::collection::logical_wal_detail::get_u32(bytes, bytes.size() - 8);
    if (internal::collection::logical_wal_detail::get_u32(bytes, bytes.size() - 4) !=
            kMutableStateTrailer ||
        internal::collection::logical_wal_detail::crc32(bytes.first(bytes.size() - 8)) !=
            expected_crc ||
        decoded.minimum_next_op_id == 0 ||
        decoded.minimum_next_op_id <= decoded.applied_watermark) {
      throw std::invalid_argument("DiskAnn mutable state checksum/watermark is invalid");
    }
    decoded.slots.reserve(static_cast<std::size_t>(count));
    std::size_t offset = 40;
    for (std::uint64_t slot = 0; slot < count; ++slot) {
      MutableSlotState state;
      state.label = internal::collection::logical_wal_detail::get_u64(bytes, offset);
      state.op_id = internal::collection::logical_wal_detail::get_u64(bytes, offset + 8);
      const auto tombstone = std::to_integer<std::uint8_t>(bytes[offset + 16]);
      for (std::size_t reserved = 17; reserved < kMutableStateRecordBytes; ++reserved) {
        if (bytes[offset + reserved] != std::byte{0}) {
          throw std::invalid_argument("DiskAnn mutable state reserved bytes are nonzero");
        }
      }
      if (tombstone > 1 || state.op_id > decoded.applied_watermark) {
        throw std::invalid_argument("DiskAnn mutable state slot record is invalid");
      }
      state.tombstone = tombstone != 0;
      decoded.slots.push_back(state);
      offset += kMutableStateRecordBytes;
    }
    return decoded;
  }

  [[nodiscard]] auto initialize_mutable_state() -> core::Status {
    try {
      auto labels = read_native_labels(directory_ / "ids.bin", meta_.max_slot_id);
      const auto state_path = directory_ / kMutableStateFilename;
      if (std::filesystem::is_regular_file(state_path)) {
        auto decoded = decode_mutable_state(state_path, meta_.max_slot_id);
        applied_watermark_ = decoded.applied_watermark;
        minimum_next_op_id_ =
            std::max(decoded.minimum_next_op_id, mutable_options_->minimum_next_op_id);
        checkpoint_generation_ = decoded.checkpoint_generation;
        slot_states_ = std::move(decoded.slots);
      } else {
        minimum_next_op_id_ = mutable_options_->minimum_next_op_id;
        slot_states_.reserve(labels.size());
        for (std::uint32_t slot = 0; slot < labels.size(); ++slot) {
          slot_states_.push_back(MutableSlotState{labels[slot], 0, native_->is_deleted(slot)});
        }
      }
      label_to_slot_.clear();
      std::uint64_t live{};
      for (std::uint32_t slot = 0; slot < slot_states_.size(); ++slot) {
        const auto &state = slot_states_[slot];
        if (state.label != labels[slot] || state.tombstone != native_->is_deleted(slot)) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "DiskAnn mutable state disagrees with native slots/labels");
        }
        if (!state.tombstone) {
          if (!label_to_slot_.emplace(state.label, slot).second) {
            return core::Status::error(core::StatusCode::corruption,
                                       core::OperationStage::open,
                                       core::StatusDetail::already_exists,
                                       "DiskAnn mutable state has duplicate live labels");
          }
          ++live;
        }
      }
      if (live != meta_.live_count || native_->tombstone_count() != meta_.max_slot_id - live) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnn mutable state live/tombstone counts disagree");
      }
      minimum_next_op_id_ = std::max(minimum_next_op_id_, applied_watermark_ + 1);
      publish_mutable_visibility_locked();
      return core::Status::success();
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto malformed_mutation(core::OperationStage stage, std::string diagnostic)
      -> core::Status {
    return core::Status::error(core::StatusCode::invalid_argument,
                               stage,
                               core::StatusDetail::malformed_struct,
                               std::move(diagnostic));
  }

  [[nodiscard]] auto copy_mutation_row(const internal::collection::SegmentMutationPayload &payload,
                                       core::OperationStage stage) const
      -> core::Result<OwnedMutationRow> {
    if (!core::is_current_struct(payload) || payload.op_id == 0 ||
        payload.upsert_sequence != payload.op_id ||
        payload.target.segment_id != mutable_options_->collection_segment_id ||
        payload.target.generation != mutable_options_->segment_generation) {
      return malformed_mutation(stage, "DiskAnn mutation row identity is invalid");
    }
    OwnedMutationRow row;
    row.action = payload.action;
    row.op_id = payload.op_id;
    row.upsert_sequence = payload.upsert_sequence;
    row.target = payload.target;
    row.previous = payload.previous;
    if (payload.action == internal::collection::SegmentMutationAction::write) {
      auto tensor =
          core::validate_tensor(payload.vector, static_cast<std::uint32_t>(meta_.dim), stage);
      if (!tensor.ok()) {
        return tensor;
      }
      if (payload.vector.rows != 1 || payload.vector.scalar_type != core::ScalarType::float32) {
        return malformed_mutation(stage, "DiskAnn write payload must be one float32 row");
      }
      const auto *values = payload.vector.row<float>(0);
      row.vector.assign(values, values + meta_.dim);
    } else if (payload.action != internal::collection::SegmentMutationAction::erase) {
      return malformed_mutation(stage, "DiskAnn mutation action is invalid");
    }
    return row;
  }

  [[nodiscard]] auto copy_mutation_transaction(const core::OpaqueOperationRequest &request,
                                               core::OperationStage stage)
      -> core::Result<MutableTransaction> {
    if (!mutable_options_.has_value() || !core::is_current_struct(request) ||
        request.payload == nullptr) {
      return malformed_mutation(stage, "DiskAnn mutation payload is missing or incompatible");
    }
    MutableTransaction transaction;
    if (request.payload_size == sizeof(internal::collection::SegmentMutationBundlePayload)) {
      const auto &bundle =
          *static_cast<const internal::collection::SegmentMutationBundlePayload *>(request.payload);
      if (!core::is_current_struct(bundle) || bundle.batch_op_id == 0 || bundle.rows.empty()) {
        return malformed_mutation(stage, "DiskAnn mutation bundle is invalid");
      }
      transaction.transaction_id = bundle.batch_op_id;
      transaction.bundled = bundle.rows.size() > 1;
      transaction.rows.reserve(bundle.rows.size());
      std::uint64_t previous_op{};
      for (const auto &payload : bundle.rows) {
        auto row = copy_mutation_row(payload, stage);
        if (!row.ok()) {
          return row.status();
        }
        if (row.value().op_id <= previous_op) {
          return malformed_mutation(stage, "DiskAnn bundle op ids are not strictly ordered");
        }
        previous_op = row.value().op_id;
        std::uint64_t vector_bytes{};
        if (!core::checked_multiply(row.value().vector.size(), sizeof(float), vector_bytes) ||
            !core::checked_add(transaction.payload_bytes,
                               vector_bytes,
                               transaction.payload_bytes)) {
          return core::Status::error(core::StatusCode::resource_exhausted,
                                     stage,
                                     core::StatusDetail::arithmetic_overflow,
                                     "DiskAnn mutation payload size overflows uint64");
        }
        transaction.rows.push_back(std::move(row).value());
      }
    } else if (request.payload_size == sizeof(internal::collection::SegmentMutationPayload)) {
      auto row =
          copy_mutation_row(*static_cast<const internal::collection::SegmentMutationPayload *>(
                                request.payload),
                            stage);
      if (!row.ok()) {
        return row.status();
      }
      transaction.transaction_id = row.value().op_id;
      if (!core::checked_multiply(row.value().vector.size(),
                                  sizeof(float),
                                  transaction.payload_bytes)) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   stage,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskAnn mutation payload size overflows uint64");
      }
      transaction.rows.push_back(std::move(row).value());
    } else {
      return malformed_mutation(stage, "DiskAnn mutation payload size is invalid");
    }
    std::uint64_t row_bytes{};
    if (!core::checked_multiply(transaction.rows.size(), sizeof(OwnedMutationRow), row_bytes) ||
        !core::checked_add(transaction.payload_bytes, row_bytes, transaction.payload_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 stage,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnn mutation stage size overflows uint64");
    }
    return transaction;
  }

  [[nodiscard]] auto admit_mutation(MutableTransaction transaction,
                                    core::MutationContext &context,
                                    core::MutationToken &token) -> core::Status {
    if (!core::is_current_struct(context) || !core::is_current_struct(token)) {
      return malformed_mutation(core::OperationStage::mutation_prepare,
                                "DiskAnn mutation context/token is incompatible");
    }
    const auto transaction_id = transaction.transaction_id;
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::mutation_prepare);
    if (!control.ok()) {
      return control;
    }
    std::uint64_t stage_bytes = transaction.payload_bytes;
    std::uint64_t io_bytes{};
    std::uint64_t io_requests{};
    {
      std::shared_lock native_lock(native_mutex_);
      for (const auto &row : transaction.rows) {
        if (row.op_id < minimum_next_op_id_) {
          return core::Status::error(core::StatusCode::conflict,
                                     core::OperationStage::mutation_prepare,
                                     core::StatusDetail::already_exists,
                                     "DiskAnn mutation op id is below the persisted next-op floor");
        }
      }
      if (transaction.bundled) {
        std::uint64_t shadow_bytes{};
        std::uint64_t slot_bytes{};
        std::uint64_t map_bytes{};
        if (!core::checked_add(resident_bytes_, scratch_bytes_, shadow_bytes) ||
            !core::checked_multiply(slot_states_.size(), sizeof(MutableSlotState), slot_bytes) ||
            !core::checked_multiply(label_to_slot_.size(),
                                    sizeof(std::pair<const std::uint64_t, std::uint32_t>) + 32U,
                                    map_bytes) ||
            !core::checked_add(shadow_bytes, slot_bytes, shadow_bytes) ||
            !core::checked_add(shadow_bytes, map_bytes, shadow_bytes) ||
            !core::checked_add(stage_bytes, shadow_bytes, stage_bytes)) {
          return core::Status::error(core::StatusCode::resource_exhausted,
                                     core::OperationStage::mutation_prepare,
                                     core::StatusDetail::arithmetic_overflow,
                                     "DiskAnn atomic shadow reservation overflows uint64");
        }
      }
      if (!core::checked_multiply(graph_bytes_, transaction.rows.size(), io_bytes)) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::mutation_prepare,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskAnn mutation I/O estimate overflows uint64");
      }
      if (transaction.bundled) {
        std::uint64_t clone_bytes{};
        if (!core::checked_multiply(artifact_bytes_, 2, clone_bytes) ||
            !core::checked_add(io_bytes, clone_bytes, io_bytes)) {
          return core::Status::error(core::StatusCode::resource_exhausted,
                                     core::OperationStage::mutation_prepare,
                                     core::StatusDetail::arithmetic_overflow,
                                     "DiskAnn atomic clone I/O estimate overflows uint64");
        }
      }
      const auto page_size =
          diskann::DiskLayoutGeometry::compute(meta_.dim, meta_.max_degree).page_size;
      io_requests = io_bytes / page_size + (io_bytes % page_size == 0 ? 0 : 1);
    }
    auto reservation =
        context.pending_reservation.ensure(transaction.payload_bytes,
                                           core::OperationStage::mutation_prepare,
                                           "DiskAnn pending-mutation reservation is too small");
    if (!reservation.ok()) {
      return reservation;
    }
    reservation =
        context.stage_reservation.ensure(stage_bytes,
                                         core::OperationStage::mutation_prepare,
                                         "DiskAnn mutation-stage reservation is too small");
    if (!reservation.ok()) {
      return reservation;
    }
    auto io = require_io_credits(context.io_credits,
                                 io_requests,
                                 io_bytes,
                                 core::OperationStage::mutation_prepare,
                                 "DiskAnn mutation I/O credits are too small");
    if (!io.ok()) {
      return io;
    }
    {
      std::lock_guard lifecycle_lock(lifecycle_mutex_);
      if (!admission_open_ ||
          health_.load(std::memory_order_acquire) == core::SegmentHealth::failed) {
        return core::Status::error(core::StatusCode::closed,
                                   core::OperationStage::admission,
                                   core::StatusDetail::operation_slot_absent,
                                   "DiskAnn mutation admission is closed");
      }
      std::lock_guard transaction_lock(transaction_mutex_);
      if (!transactions_.emplace(transaction_id, std::move(transaction)).second) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::mutation_prepare,
                                   core::StatusDetail::already_exists,
                                   "DiskAnn mutation transaction id is already staged");
      }
      inflight_mutation_.fetch_add(1, std::memory_order_acq_rel);
      pending_transactions_.fetch_add(1, std::memory_order_acq_rel);
    }
    token.value = transaction_id;
    prepared_count_.fetch_add(1, std::memory_order_acq_rel);
    return core::Status::success();
  }

  [[nodiscard]] auto prepare_mutation(const core::OpaqueOperationRequest &request,
                                      core::MutationContext &context,
                                      core::MutationToken &token) -> core::Status {
    auto transaction = copy_mutation_transaction(request, core::OperationStage::mutation_prepare);
    if (!transaction.ok()) {
      return transaction.status();
    }
    return admit_mutation(std::move(transaction).value(), context, token);
  }

  [[nodiscard]] auto stage_mutation(core::MutationToken &token, core::MutationContext &context)
      -> core::Status {
    if (!core::is_current_struct(context) || !core::is_current_struct(token)) {
      return malformed_mutation(core::OperationStage::mutation_stage,
                                "DiskAnn mutation context/token is incompatible");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::mutation_stage);
    if (!control.ok()) {
      return control;
    }
    std::lock_guard lock(transaction_mutex_);
    const auto found = transactions_.find(token.value);
    if (found == transactions_.end()) {
      return malformed_mutation(core::OperationStage::mutation_stage,
                                "DiskAnn mutation token is unknown");
    }
    found->second.staged = true;
    staged_count_.fetch_add(1, std::memory_order_acq_rel);
    return core::Status::success();
  }

  void finish_mutation_admission() noexcept {
    inflight_mutation_.fetch_sub(1, std::memory_order_acq_rel);
    std::lock_guard lock(lifecycle_mutex_);
    lifecycle_changed_.notify_all();
  }

  [[nodiscard]] auto abort_mutation(core::MutationToken &token, core::MutationContext &)
      -> core::Status {
    bool removed{};
    {
      std::lock_guard lock(transaction_mutex_);
      removed = transactions_.erase(token.value) != 0;
    }
    if (removed) {
      pending_transactions_.fetch_sub(1, std::memory_order_acq_rel);
      aborted_count_.fetch_add(1, std::memory_order_acq_rel);
      finish_mutation_admission();
    }
    return core::Status::success();
  }

  void publish_mutable_visibility_locked() {
    auto visibility = std::make_shared<MutableVisibility>();
    visibility->applied_watermark = applied_watermark_;
    for (const auto &[label, unused] : label_to_slot_) {
      (void)unused;
      visibility->live_labels.insert(label);
    }
    std::atomic_store_explicit(&mutable_visibility_,
                               std::shared_ptr<const MutableVisibility>(std::move(visibility)),
                               std::memory_order_release);
  }

  [[nodiscard]] auto validate_apply_rows(
      const std::vector<OwnedMutationRow> &rows,
      const std::unordered_map<std::uint64_t, std::uint32_t> &labels) const -> core::Status {
    auto prospective = labels;
    for (const auto &row : rows) {
      if (row.op_id <= applied_watermark_) {
        continue;
      }
      const auto target_label = static_cast<std::uint64_t>(row.target.row_id);
      if (row.action == internal::collection::SegmentMutationAction::write &&
          prospective.contains(target_label)) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::mutation_publish,
                                   core::StatusDetail::already_exists,
                                   "DiskAnn target label is already live");
      }
      if (row.previous.has_value() &&
          row.previous->segment_id == mutable_options_->collection_segment_id &&
          row.previous->generation == mutable_options_->segment_generation) {
        const auto previous_label = static_cast<std::uint64_t>(row.previous->row_id);
        if (!prospective.contains(previous_label)) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::mutation_publish,
                                     core::StatusDetail::malformed_struct,
                                     "DiskAnn previous logical label has no live slot mapping");
        }
        prospective.erase(previous_label);
      } else if (row.action == internal::collection::SegmentMutationAction::erase) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::mutation_publish,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnn delete does not target a local previous version");
      }
      if (row.action == internal::collection::SegmentMutationAction::write) {
        prospective.emplace(target_label, 0);
      }
    }
    return core::Status::success();
  }

  static void ensure_slot_state(std::vector<MutableSlotState> &slots, std::uint32_t slot) {
    if (slots.size() <= slot) {
      slots.resize(static_cast<std::size_t>(slot) + 1, MutableSlotState{0, 0, true});
    }
  }

  void apply_rows_to_native(diskann::DiskANNIndex &native,
                            std::unordered_map<std::uint64_t, std::uint32_t> &labels,
                            std::vector<MutableSlotState> &slots,
                            const std::vector<OwnedMutationRow> &rows,
                            std::uint64_t &watermark,
                            std::uint64_t &applied_rows) const {
    for (const auto &row : rows) {
      if (row.op_id <= watermark) {
        continue;
      }
      std::optional<std::pair<std::uint64_t, std::uint32_t>> previous;
      if (row.previous.has_value() &&
          row.previous->segment_id == mutable_options_->collection_segment_id &&
          row.previous->generation == mutable_options_->segment_generation) {
        const auto label = static_cast<std::uint64_t>(row.previous->row_id);
        previous = std::pair{label, labels.at(label)};
      }
      if (row.action == internal::collection::SegmentMutationAction::write) {
        const auto label = static_cast<std::uint64_t>(row.target.row_id);
        const auto slot = native.insert(row.vector.data(), label);
        ensure_slot_state(slots, slot);
        slots[slot] = MutableSlotState{label, row.op_id, false};
        labels.emplace(label, slot);
      }
      if (previous.has_value()) {
        native.remove(previous->second);
        labels.erase(previous->first);
        ensure_slot_state(slots, previous->second);
        slots[previous->second].op_id = row.op_id;
        slots[previous->second].tombstone = true;
      }
      watermark = row.op_id;
      ++applied_rows;
    }
  }

  [[nodiscard]] auto native_artifact_specs_for_working_copy() const -> std::vector<ArtifactSpec> {
    std::vector<ArtifactSpec> specs{
        {kMetaArtifactName, "meta.bin", ResidentClass::resident},
        {kIndexArtifactName, "diskann.index", ResidentClass::on_disk},
        {kIdsArtifactName, "ids.bin", ResidentClass::resident},
        {kCacheIdsArtifactName, "cache_ids.bin", ResidentClass::cache},
        {kCacheNodesArtifactName, "cache_nodes.bin", ResidentClass::cache},
        {kSlotsArtifactName, "slots.bin", ResidentClass::resident},
    };
    if (meta_.has_pq) {
      specs.push_back({kPqPivotsArtifactName, "pq_pivots.bin", ResidentClass::resident});
      specs.push_back({kPqCompressedArtifactName, "pq_compressed.bin", ResidentClass::resident});
    }
    if (std::filesystem::is_regular_file(directory_ / kMutableStateFilename)) {
      specs.push_back({kMutableStateArtifactName, kMutableStateFilename, ResidentClass::resident});
    }
    return specs;
  }

  [[nodiscard]] auto apply_transaction_locked(const MutableTransaction &transaction)
      -> core::Status {
    auto validation = validate_apply_rows(transaction.rows, label_to_slot_);
    if (!validation.ok()) {
      return validation;
    }
    try {
      std::uint64_t newly_applied{};
      if (transaction.bundled) {
        // The public kernel has no mixed insert/remove transaction hook. For
        // the optional all-or-nothing mode, apply to an ephemeral clone and
        // swap it only after every row succeeds. This keeps partial native
        // visibility and partial failure outside the published generation.
        native_->flush();
        const auto shadow_directory = make_working_copy(directory_,
                                                        mutable_options_->collection_root,
                                                        mutable_options_->segment_id,
                                                        native_artifact_specs_for_working_copy());
        std::unique_ptr<diskann::DiskANNIndex> shadow;
        try {
          shadow = make_native_index(shadow_directory, true);
          auto shadow_labels = label_to_slot_;
          auto shadow_slots = slot_states_;
          auto shadow_watermark = applied_watermark_;
          apply_rows_to_native(*shadow,
                               shadow_labels,
                               shadow_slots,
                               transaction.rows,
                               shadow_watermark,
                               newly_applied);
          auto old_native = std::move(native_);
          const auto old_directory = directory_;
          native_ = std::move(shadow);
          directory_ = shadow_directory;
          label_to_slot_ = std::move(shadow_labels);
          slot_states_ = std::move(shadow_slots);
          applied_watermark_ = shadow_watermark;
          old_native.reset();
          std::error_code error;
          std::filesystem::remove_all(old_directory, error);
        } catch (...) {
          shadow.reset();
          std::error_code error;
          std::filesystem::remove_all(shadow_directory, error);
          throw;
        }
      } else {
        apply_rows_to_native(*native_,
                             label_to_slot_,
                             slot_states_,
                             transaction.rows,
                             applied_watermark_,
                             newly_applied);
      }
      if (newly_applied != 0) {
        meta_.max_slot_id = native_->max_slot_id();
        meta_.live_count = native_->live_count();
        minimum_next_op_id_ = std::max(minimum_next_op_id_, applied_watermark_ + 1);
        publish_mutable_visibility_locked();
        applied_count_.fetch_add(newly_applied, std::memory_order_acq_rel);
        logical_dirty_bytes_.fetch_add(transaction.payload_bytes +
                                           newly_applied * kMutableStateRecordBytes,
                                       std::memory_order_acq_rel);
      }
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_publish);
    }
  }

  void fail_closed(const core::Status &status) noexcept {
    health_.store(core::SegmentHealth::failed, std::memory_order_release);
    last_error_.store(status.detail(), std::memory_order_release);
    std::lock_guard lock(lifecycle_mutex_);
    admission_open_ = false;
    lifecycle_changed_.notify_all();
  }

  [[nodiscard]] auto publish_mutation(core::MutationToken &token, core::MutationContext &context)
      -> core::Status {
    if (!core::is_current_struct(context) || !core::is_current_struct(token)) {
      return malformed_mutation(core::OperationStage::mutation_publish,
                                "DiskAnn mutation context/token is incompatible");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::mutation_publish);
    if (!control.ok()) {
      return control;
    }
    MutableTransaction transaction;
    {
      std::lock_guard lock(transaction_mutex_);
      const auto found = transactions_.find(token.value);
      if (found == transactions_.end() || !found->second.staged) {
        return malformed_mutation(core::OperationStage::mutation_publish,
                                  "DiskAnn mutation token is not staged");
      }
      transaction = found->second;
    }
    committed_count_.fetch_add(1, std::memory_order_acq_rel);
    core::Status status;
    {
      std::unique_lock native_lock(native_mutex_);
      status = apply_transaction_locked(transaction);
    }
    {
      std::lock_guard lock(transaction_mutex_);
      transactions_.erase(token.value);
    }
    pending_transactions_.fetch_sub(1, std::memory_order_acq_rel);
    finish_mutation_admission();
    if (!status.ok()) {
      // COMMIT is authoritative before this slot is called. Stop all new
      // admission; reopening the immutable checkpoint and replaying the WAL is
      // the only legal continuation after a native apply failure.
      fail_closed(status);
    }
    return status;
  }

  [[nodiscard]] auto replay_mutation(const core::OpaqueOperationRequest &request,
                                     core::MutationContext &context) -> core::Status {
    if (!core::is_current_struct(context)) {
      return malformed_mutation(core::OperationStage::mutation_replay,
                                "DiskAnn replay context is incompatible");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::mutation_replay);
    if (!control.ok()) {
      return control;
    }
    auto decoded = copy_mutation_transaction(request, core::OperationStage::mutation_replay);
    if (!decoded.ok()) {
      return decoded.status();
    }
    inflight_mutation_.fetch_add(1, std::memory_order_acq_rel);
    core::Status status;
    {
      std::unique_lock native_lock(native_mutex_);
      status = apply_transaction_locked(decoded.value());
    }
    replayed_count_.fetch_add(1, std::memory_order_acq_rel);
    finish_mutation_admission();
    if (!status.ok()) {
      fail_closed(status);
    }
    return status;
  }

  [[nodiscard]] auto encode_mutable_state_locked(std::uint64_t checkpoint_generation) const
      -> std::vector<std::byte> {
    namespace wal = internal::collection::logical_wal_detail;
    std::vector<std::byte> output;
    output.reserve(kMutableStateFixedBytes + slot_states_.size() * kMutableStateRecordBytes);
    wal::put_u32(output, kMutableStateMagic);
    wal::put_u32(output, kMutableStateVersion);
    wal::put_u64(output, applied_watermark_);
    wal::put_u64(output, minimum_next_op_id_);
    wal::put_u64(output, checkpoint_generation);
    wal::put_u64(output, slot_states_.size());
    for (const auto &slot : slot_states_) {
      wal::put_u64(output, slot.label);
      wal::put_u64(output, slot.op_id);
      output.push_back(slot.tombstone ? std::byte{1} : std::byte{0});
      output.insert(output.end(), 7, std::byte{0});
    }
    wal::put_u32(output, wal::crc32(output));
    wal::put_u32(output, kMutableStateTrailer);
    return output;
  }

  [[nodiscard]] auto checkpoint_logical_specs() const
      -> std::vector<internal::collection::LogicalArtifactSpec> {
    using internal::collection::LogicalArtifactSpec;
    std::vector<LogicalArtifactSpec> specs{
        {std::string(kMetaArtifactName), "meta.bin", true, {}},
        {std::string(kIndexArtifactName), "diskann.index", true, {}},
        {std::string(kIdsArtifactName), "ids.bin", true, {}},
        {std::string(kCacheIdsArtifactName), "cache_ids.bin", true, {}},
        {std::string(kCacheNodesArtifactName), "cache_nodes.bin", true, {}},
        {std::string(kSlotsArtifactName), "slots.bin", true, {}},
        {std::string(kMutableStateArtifactName), std::string(kMutableStateFilename), true, {}},
    };
    if (meta_.has_pq) {
      specs.push_back({std::string(kPqPivotsArtifactName), "pq_pivots.bin", true, {}});
      specs.push_back({std::string(kPqCompressedArtifactName), "pq_compressed.bin", true, {}});
    }
    for (auto &spec : specs) {
      spec.reader_compatibility.required_features = {"diskann_mutable_segment"};
    }
    return specs;
  }

  [[nodiscard]] auto make_mutable_manifest(const std::filesystem::path &target_relative,
                                           std::uint64_t checkpoint_generation)
      -> core::Result<internal::collection::ArtifactManifestV2> {
    try {
      using namespace internal::collection;  // NOLINT(build/namespaces)
      ArtifactManifestV2 manifest;
      const auto manifest_path = mutable_options_->collection_root / kCollectionManifestFilename;
      if (std::filesystem::is_regular_file(manifest_path)) {
        manifest = ArtifactManifestV2::load(manifest_path);
        if (manifest.collection.dim != meta_.dim ||
            manifest.collection.metric != core::Metric::l2 ||
            manifest.collection.scalar_type != core::ScalarType::float32) {
          return core::Status::
              error(core::StatusCode::corruption,
                    core::OperationStage::checkpoint,
                    core::StatusDetail::malformed_struct,
                    "DiskAnn checkpoint disagrees with collection manifest schema");
        }
      }
      manifest.collection.dim = static_cast<std::uint32_t>(meta_.dim);
      manifest.collection.metric = core::Metric::l2;
      manifest.collection.scalar_type = core::ScalarType::float32;
      manifest.collection.logical_id_encoding = LogicalIdEncodingV2::canonical_kind_and_bytes;
      manifest.collection.metadata_epoch = applied_watermark_;
      manifest.collection.metadata_checkpoint =
          (target_relative / kMutableStateFilename).generic_string();
      manifest.publication.generation =
          std::max<std::uint64_t>(manifest.publication.generation + 1, checkpoint_generation + 1);
      manifest.wal_cut = applied_watermark_;
      manifest.row_versions.minimum = applied_watermark_ == 0 ? 0 : 1;
      manifest.row_versions.maximum = applied_watermark_;
      manifest.id_map_checkpoint = (target_relative / kMutableStateFilename).generic_string();

      SegmentEntryV2 entry;
      entry.segment_id = mutable_options_->segment_id;
      entry.generation = mutable_options_->segment_generation;
      entry.role = SegmentRoleV2::searchable;
      entry.algorithm_id = kAlgorithmId;
      entry.format_version = meta_.format_version;
      entry.factory_key = "diskann_mutable";
      entry.capabilities.operations =
          core::capability_bit(core::OperationCapability::search) |
          core::capability_bit(core::OperationCapability::batch_search) |
          core::capability_bit(core::OperationCapability::mutation) |
          core::capability_bit(core::OperationCapability::checkpoint) |
          core::capability_bit(core::OperationCapability::stats) |
          core::capability_bit(core::OperationCapability::close) |
          core::capability_bit(core::OperationCapability::drain);
      entry.capabilities.reentrant_search = true;
      entry.capabilities.search_with_stage = true;
      entry.capabilities.search_with_publish = false;
      entry.capabilities.serial_mutation = true;
      entry.capabilities.checkpoint_with_search = false;
      entry.capabilities.native_async = true;
      entry.capabilities.cooperative_cancel = true;
      entry.capabilities.explicit_drain = true;
      entry.lifecycle = SegmentLifecycleV2::active;
      entry.wal_cut = applied_watermark_;
      entry.row_versions = manifest.row_versions;
      entry.id_map_checkpoint = manifest.id_map_checkpoint;
      entry.reader_compatibility.required_features = {"diskann_segment",
                                                      "diskann_mutable_segment",
                                                      "collection_wal_v1"};
      entry.extensions.emplace("applied_op_id", std::to_string(applied_watermark_));
      entry.extensions.emplace("minimum_next_op_id", std::to_string(minimum_next_op_id_));
      entry.extensions.emplace("checkpoint_generation", std::to_string(checkpoint_generation));
      // prepare() owns artifact checksums/READY and replaces this entry in the
      // base manifest during publish().
      pending_checkpoint_entry_ = std::move(entry);
      return manifest;
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::checkpoint);
    }
  }

  [[nodiscard]] auto checkpoint_mutable(core::CheckpointContext &context,
                                        core::CheckpointToken &token) -> core::Status {
    if (!mutable_options_.has_value() || !core::is_current_struct(context) ||
        !core::is_current_struct(token)) {
      return malformed_mutation(core::OperationStage::checkpoint,
                                "DiskAnn mutable checkpoint structs are incompatible");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::checkpoint);
    if (!control.ok()) {
      return control;
    }
    const auto specs = checkpoint_logical_specs();
    std::uint64_t checkpoint_requests{};
    std::uint64_t checkpoint_bytes{};
    std::uint64_t state_bytes{};
    std::uint64_t slot_count{};
    std::uint64_t accounted_artifact_bytes{};
    {
      std::shared_lock native_lock(native_mutex_);
      slot_count = slot_states_.size();
      accounted_artifact_bytes = artifact_bytes_;
    }
    if (!core::checked_multiply(specs.size(), 2, checkpoint_requests) ||
        !core::checked_add(checkpoint_requests, 4, checkpoint_requests) ||
        !core::checked_multiply(slot_count, kMutableStateRecordBytes, state_bytes) ||
        !core::checked_add(state_bytes, kMutableStateFixedBytes, state_bytes) ||
        !core::checked_add(accounted_artifact_bytes,
                           logical_dirty_bytes_.load(std::memory_order_acquire),
                           checkpoint_bytes) ||
        !core::checked_add(checkpoint_bytes, state_bytes, checkpoint_bytes) ||
        !core::checked_multiply(checkpoint_bytes, 2, checkpoint_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnn checkpoint I/O accounting overflows uint64");
    }
    auto credits = require_io_credits(context.dirty_page_io_credits,
                                      checkpoint_requests,
                                      checkpoint_bytes,
                                      core::OperationStage::checkpoint,
                                      "DiskAnn checkpoint I/O credits are too small");
    if (!credits.ok()) {
      return credits;
    }
    std::lock_guard checkpoint_lock(segment_checkpoint_mutex_);
    {
      std::lock_guard transaction_lock(transaction_mutex_);
      if (!transactions_.empty() || inflight_mutation_.load(std::memory_order_acquire) != 0) {
        return core::Status::error(core::StatusCode::conflict,
                                   core::OperationStage::checkpoint,
                                   core::StatusDetail::none,
                                   "DiskAnn checkpoint requires drained mutation admission");
      }
    }
    try {
      std::unique_lock native_lock(native_mutex_);
      native_->flush();
      meta_.max_slot_id = native_->max_slot_id();
      meta_.live_count = native_->live_count();
      const auto next_generation = checkpoint_generation_ + 1;
      const auto state = encode_mutable_state_locked(next_generation);
      platform::write_all_fsync(directory_ / kMutableStateFilename, state.data(), state.size());
      platform::sync_directory_or_throw(directory_);

      const auto target_relative = std::filesystem::path("segments") /
                                   (mutable_options_->segment_id + "_g" +
                                    std::to_string(mutable_options_->segment_generation) + "_c" +
                                    std::to_string(next_generation));
      internal::collection::ArtifactTransactionOptions transaction_options;
      transaction_options.collection_root = mutable_options_->collection_root;
      transaction_options.target_relative_directory = target_relative;
      transaction_options.transaction_id = "diskann_checkpoint_" + std::to_string(next_generation) +
                                           "_" + std::to_string(platform::get_pid());
      transaction_options.manifest_v2_writer = true;
      transaction_options.abort_policy =
          internal::collection::ArtifactAbortPolicy::retain_for_restart_cleanup;
      transaction_options.fail_point = mutable_options_->checkpoint_fail_point;
      core::BuildContext build_context;
      build_context.io_credits = context.dirty_page_io_credits;
      build_context.deadline = context.deadline;
      build_context.cancellation = context.cancellation;
      build_context.lane = context.lane;
      auto begun =
          internal::collection::ArtifactControlPlaneTransaction::begin(std::move(
                                                                           transaction_options),
                                                                       build_context);
      if (!begun.ok()) {
        return begun.status();
      }
      auto transaction = std::move(begun).value();
      std::filesystem::create_directory(transaction->staging_payload_directory());
      for (const auto &spec : specs) {
        std::filesystem::copy_file(directory_ / spec.relative_path,
                                   transaction->staging_payload_directory() / spec.relative_path,
                                   std::filesystem::copy_options::none);
      }
      auto status = transaction->adopt(specs);
      if (!status.ok()) {
        return status;
      }
      auto manifest = make_mutable_manifest(target_relative, next_generation);
      if (!manifest.ok()) {
        return manifest.status();
      }
      auto prepared = transaction->prepare(pending_checkpoint_entry_);
      if (!prepared.ok()) {
        return prepared.status();
      }
      status = transaction->publish(std::move(manifest).value());
      if (!status.ok()) {
        return status;
      }
      checkpoint_generation_ = next_generation;
      artifact_bytes_ = 0;
      resident_bytes_ = 0;
      cache_bytes_ = 0;
      graph_bytes_ = 0;
      for (const auto &artifact : native_artifact_specs_for_working_copy()) {
        const auto bytes = checked_file_size(directory_ / artifact.filename);
        checked_accumulate(artifact_bytes_, bytes, "DiskANN artifact bytes overflow uint64");
        switch (artifact.residency) {
          case ResidentClass::on_disk:
            graph_bytes_ = bytes;
            break;
          case ResidentClass::resident:
            checked_accumulate(resident_bytes_, bytes, "DiskANN resident bytes overflow uint64");
            break;
          case ResidentClass::cache:
            checked_accumulate(cache_bytes_, bytes, "DiskANN cache bytes overflow uint64");
            checked_accumulate(resident_bytes_, bytes, "DiskANN resident bytes overflow uint64");
            break;
        }
      }
      auto scratch = estimate_scratch_bytes(meta_, true);
      if (scratch.ok()) {
        scratch_bytes_ = scratch.value();
      }
      logical_dirty_bytes_.store(0, std::memory_order_release);
      token.value = applied_watermark_;
      return core::Status::success();
    } catch (const std::bad_alloc &error) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::checkpoint,
                                 core::StatusDetail::allocation_failure,
                                 error.what(),
                                 core::Retryability::retryable_with_backoff);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::checkpoint);
    }
  }

  [[nodiscard]] static auto read_meta_summary(const std::filesystem::path &path)
      -> NativeMetaSummary {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("DiskAnnSegment cannot open " + path.string());
    }
    std::uint64_t magic{};
    NativeMetaSummary meta;
    std::uint8_t has_pq{};
    read_exact(input, magic, "DiskANN meta.bin is truncated");
    read_exact(input, meta.format_version, "DiskANN meta.bin is truncated");
    read_exact(input, meta.num_points, "DiskANN meta.bin is truncated");
    read_exact(input, meta.dim, "DiskANN meta.bin is truncated");
    read_exact(input, meta.max_degree, "DiskANN meta.bin is truncated");
    read_exact(input, meta.medoid, "DiskANN meta.bin is truncated");
    read_exact(input, has_pq, "DiskANN meta.bin is truncated");
    read_exact(input, meta.pq_n_chunks, "DiskANN meta.bin is truncated");
    read_exact(input, meta.node_len, "DiskANN meta.bin is truncated");
    read_exact(input, meta.nodes_per_sector, "DiskANN meta.bin is truncated");
    if (magic != diskann::DiskANNIndex::kMetaMagic ||
        (meta.format_version != 1 && meta.format_version != diskann::DiskANNIndex::kMetaVersion) ||
        meta.num_points == 0 || meta.dim == 0 || meta.max_degree == 0) {
      throw std::invalid_argument("DiskANN meta.bin has an invalid header");
    }
    meta.has_pq = has_pq != 0;
    if (meta.format_version == 1) {
      meta.max_slot_id = meta.num_points;
      meta.live_count = meta.num_points;
    } else {
      read_exact(input, meta.max_slot_id, "DiskANN v2 meta.bin is truncated");
      read_exact(input, meta.live_count, "DiskANN v2 meta.bin is truncated");
    }
    if (meta.max_slot_id == 0 || meta.live_count > meta.max_slot_id ||
        (meta.has_pq && meta.pq_n_chunks == 0)) {
      throw std::invalid_argument("DiskANN meta.bin has inconsistent counts or PQ metadata");
    }
    return meta;
  }

  [[nodiscard]] static auto checked_file_size(const std::filesystem::path &path) -> std::uint64_t {
    const auto value = std::filesystem::file_size(path);
    if (value > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("DiskANN artifact size exceeds uint64");
    }
    return static_cast<std::uint64_t>(value);
  }

  static void checked_accumulate(std::uint64_t &total,
                                 std::uint64_t value,
                                 const char *diagnostic) {
    std::uint64_t next{};
    if (!core::checked_add(total, value, next)) {
      throw std::overflow_error(diagnostic);
    }
    total = next;
  }

  [[nodiscard]] static auto estimate_scratch_bytes(const NativeMetaSummary &meta, bool updatable)
      -> core::Result<std::uint64_t> {
    try {
      const auto geometry = diskann::DiskLayoutGeometry::compute(meta.dim, meta.max_degree);
      std::uint64_t per_thread{};
      auto add_product = [&](std::uint64_t lhs, std::uint64_t rhs) {
        std::uint64_t value{};
        if (!core::checked_multiply(lhs, rhs, value)) {
          throw std::overflow_error("DiskANN scratch estimate overflows uint64");
        }
        checked_accumulate(per_thread, value, "DiskANN scratch estimate overflows uint64");
      };
      add_product(diskann::DiskANNIndex::kDefaultNoPQIoDepth, geometry.page_size);
      add_product(meta.max_slot_id, sizeof(float));
      add_product(meta.max_slot_id, sizeof(std::pair<std::uint32_t, std::uint32_t>));
      add_product((meta.max_slot_id + 63) / 64, sizeof(std::uint64_t));
      add_product(static_cast<std::uint64_t>(kScratchSearchListSize) * meta.max_degree,
                  sizeof(std::uint32_t));
      add_product(meta.pq_n_chunks, 256 * sizeof(float));
      add_product(meta.dim, sizeof(float));
      // Readonly retains one synchronous pool plus the bounded native
      // pipeline. Updatable load additionally sizes the base pool to the
      // public kernel's insert/reconnect defaults and, when io_uring is
      // available, creates its documented 4x insert-thread AsyncGate pool.
      const auto base_threads = std::max({kSearchThreads,
                                          diskann::kDefaultDiskANNUpdateInsertThreads,
                                          diskann::kDefaultDiskANNUpdateReconnectThreads});
      const std::uint64_t thread_count =
          updatable ? static_cast<std::uint64_t>(base_threads) +
                          4ULL * diskann::kDefaultDiskANNUpdateInsertThreads + kSearchThreads
                    : 2ULL * kSearchThreads;
      std::uint64_t total{};
      if (!core::checked_multiply(per_thread, thread_count, total)) {
        throw std::overflow_error("DiskANN scratch-pool estimate overflows uint64");
      }
      if (updatable) {
        std::uint64_t gate_wave{};
        if (!core::checked_multiply(meta.max_degree, geometry.page_size, gate_wave) ||
            !core::checked_multiply(gate_wave,
                                    4ULL * diskann::kDefaultDiskANNUpdateInsertThreads,
                                    gate_wave) ||
            !core::checked_add(total, gate_wave, total)) {
          throw std::overflow_error("DiskANN update-gate scratch estimate overflows uint64");
        }
      }
      return total;
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto require_io_credits(const core::IoCredits &credits,
                                               std::uint64_t requests,
                                               std::uint64_t bytes,
                                               core::OperationStage stage,
                                               const char *diagnostic) -> core::Status {
    if ((credits.available_requests != core::kUnlimitedResource &&
         requests > credits.available_requests) ||
        (credits.available_bytes != core::kUnlimitedResource && bytes > credits.available_bytes)) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 stage,
                                 core::StatusDetail::budget_denied,
                                 diagnostic,
                                 core::Retryability::retryable_with_backoff);
    }
    return core::Status::success();
  }

  [[nodiscard]] auto resolve_search(const core::SearchOptions &options) const
      -> core::Result<ResolvedSearch> {
    ResolvedSearch resolved;
    resolved.native.search_list_size = 100;
    resolved.native.use_pq = true;
    resolved.native.rerank = options.rerank_policy != core::RerankPolicy::disabled;
    resolved.native.rerank_count = 0;
    resolved.native.deterministic = false;
    for (const auto &extension : options.extensions) {
      if (!core::is_current_struct(extension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnnSegment received an incompatible search extension");
      }
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     "DiskAnnSegment received an extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr ||
          extension.payload_size < sizeof(DiskAnnSegmentSearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnnSegment search extension payload is truncated");
      }
      const auto &typed = *static_cast<const DiskAnnSegmentSearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed) || typed.search_list_size == 0) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnnSegment search extension values are invalid");
      }
      resolved.native.search_list_size = typed.search_list_size;
      resolved.native.use_pq = typed.use_pq;
      resolved.native.rerank = typed.rerank;
      resolved.native.rerank_count = typed.rerank_count;
      resolved.native.deterministic = typed.deterministic;
    }
    if (options.rerank_policy == core::RerankPolicy::disabled) {
      resolved.native.rerank = false;
    } else if (options.rerank_policy == core::RerankPolicy::exact_required) {
      resolved.native.rerank = true;
    }
    const auto bounded_live = std::min<std::uint64_t>(meta_.live_count, kScratchSearchListSize);
    resolved.native_top_k =
        static_cast<std::uint32_t>(std::min<std::uint64_t>(options.top_k, bounded_live));
    if (std::max(resolved.native.search_list_size, resolved.native_top_k) >
        kScratchSearchListSize) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::search,
                                 core::StatusDetail::budget_denied,
                                 "DiskAnnSegment request exceeds the bounded native scratch "
                                 "search-list capacity",
                                 core::Retryability::retryable_with_backoff);
    }
    return resolved;
  }

  [[nodiscard]] auto effective_deadline(const core::SearchRequest &request) const
      -> core::Deadline {
    auto deadline = request.context->deadline;
    const auto option_deadline = request.options.deadline_steady_nanoseconds;
    if (option_deadline != 0 &&
        (!deadline.enabled || option_deadline < deadline.steady_clock_nanoseconds)) {
      deadline.enabled = true;
      deadline.steady_clock_nanoseconds = option_deadline;
    }
    return deadline;
  }

  [[nodiscard]] auto runtime_control(const core::SearchRequest &request) const -> core::Status {
    return core::validate_runtime_control(effective_deadline(request),
                                          request.context->cancellation,
                                          core::OperationStage::search);
  }

  [[nodiscard]] auto validate_search_request(const core::SearchRequest &request,
                                             ResolvedSearch &resolved) const -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        !core::is_current_struct(request.filter) || request.context == nullptr ||
        request.response == nullptr || !core::is_current_struct(*request.context)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "DiskAnnSegment search request is incomplete or incompatible");
    }
    auto status = core::validate_tensor(request.queries,
                                        static_cast<std::uint32_t>(meta_.dim),
                                        core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::ScalarType::float32) {
      return unavailable(core::OperationStage::validation,
                         "DiskAnnSegment search accepts float32 tensors only");
    }
    status = core::validate_response(*request.response,
                                     request.queries.rows,
                                     request.options.top_k,
                                     core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.filter.kind != core::SegmentFilterKind::none) {
      return unavailable(core::OperationStage::validation,
                         "DiskAnnSegment has no engine-local metadata filter");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment top_k exceeds the native uint32 boundary");
    }
    if (request.queries.rows > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment query count exceeds the native uint32 boundary");
    }
    auto resolved_result = resolve_search(request.options);
    if (!resolved_result.ok()) {
      return resolved_result.status();
    }
    resolved = std::move(resolved_result).value();
    status = runtime_control(request);
    if (!status.ok()) {
      return status;
    }
    std::uint64_t output_scratch{};
    if (!core::checked_multiply(resolved.native_top_k,
                                sizeof(std::uint64_t) + sizeof(float),
                                output_scratch) ||
        !core::checked_multiply(output_scratch, request.queries.rows, output_scratch)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment query scratch size overflows uint64");
    }
    const auto concurrent_queries = std::min<std::uint64_t>(request.queries.rows, kSearchThreads);
    std::uint64_t frontier_scratch{};
    if (!core::checked_multiply(resolved.native.search_list_size,
                                sizeof(std::pair<std::uint32_t, float>),
                                frontier_scratch) ||
        !core::checked_multiply(frontier_scratch, concurrent_queries, frontier_scratch) ||
        !core::checked_add(output_scratch, frontier_scratch, output_scratch)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment query scratch size overflows uint64");
    }
    const auto contiguous_stride = static_cast<std::uint64_t>(meta_.dim * sizeof(float));
    if (request.queries.row_stride != contiguous_stride) {
      std::uint64_t query_copy{};
      if (!core::checked_multiply(request.queries.rows, contiguous_stride, query_copy) ||
          !core::checked_add(output_scratch, query_copy, output_scratch)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskAnnSegment query copy size overflows uint64");
      }
    }
    status = core::require_lease(request.context->query_scratch_lease,
                                 output_scratch,
                                 core::OperationStage::search,
                                 "DiskAnnSegment query scratch lease is too small");
    if (!status.ok()) {
      return status;
    }
    std::uint64_t io_bytes{};
    if (!core::checked_multiply(graph_bytes_, request.queries.rows, io_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment search I/O accounting overflows uint64");
    }
    return require_io_credits(request.context->io_credits,
                              request.queries.rows,
                              io_bytes,
                              core::OperationStage::search,
                              "DiskAnnSegment search I/O credits are too small");
  }

  static void update_search_stats(core::SearchStats *target,
                                  const diskann::SearchStats &source,
                                  std::uint64_t page_size) {
    if (target == nullptr) {
      return;
    }
    target->visited += source.n_nodes_processed;
    target->io_requests += source.n_ios + source.n_rerank_reads;
    std::uint64_t bytes{};
    if (core::checked_multiply(source.n_ios + source.n_rerank_reads, page_size, bytes)) {
      target->io_bytes += bytes;
    }
    target->cache_hits += source.n_cache_hits + source.n_page_cache_hits;
    target->rerank_count += source.n_rerank_reads;
  }

  [[nodiscard]] auto finish_controlled(const core::SearchRequest &request,
                                       const core::Status &control,
                                       core::RowCount completed_rows,
                                       core::RowCount cursor) const -> core::Status {
    auto &response = *request.response;
    if (request.options.partial_result_policy == core::PartialResultPolicy::discard) {
      response.invalidate(control);
      return control;
    }
    auto partial = core::Status::error(control.code(),
                                       core::OperationStage::search,
                                       control.detail(),
                                       control.diagnostic(),
                                       control.retryability(),
                                       true);
    for (core::RowCount row = 0; row < completed_rows; ++row) {
      if (response.statuses[row].ok()) {
        response.statuses[row] = partial;
        response.completeness[row] = core::SearchCompleteness::cancelled_partial;
      }
    }
    for (core::RowCount row = completed_rows; row < request.queries.rows; ++row) {
      response.offsets[row + 1] = cursor;
      response.valid_counts[row] = 0;
      response.statuses[row] = partial;
      response.completeness[row] = core::SearchCompleteness::cancelled_partial;
    }
    return partial;
  }

  [[nodiscard]] auto execute_native_async(NativeOperationState &state) const -> core::Status {
    std::shared_lock<std::shared_mutex> native_lock;
    if (mutable_options_.has_value()) {
      native_lock = std::shared_lock(native_mutex_);
    }
    auto &request = state.request;
    ResolvedSearch resolved;
    auto status = validate_search_request(request, resolved);
    if (!status.ok()) {
      if (request.response != nullptr && core::is_current_struct(*request.response) &&
          (status.code() == core::StatusCode::cancelled ||
           status.code() == core::StatusCode::deadline_exceeded)) {
        request.response->offsets[0] = 0;
        return finish_controlled(request, status, 0, 0);
      }
      if (request.response != nullptr && core::is_current_struct(*request.response) &&
          request.options.partial_result_policy == core::PartialResultPolicy::discard) {
        request.response->invalidate(status);
      }
      return status;
    }

    auto &response = *request.response;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = core::Metric::l2;
    response.result_flags = core::ResultFlag::approximate;
    if (resolved.native.rerank && resolved.native.use_pq && meta_.has_pq) {
      response.result_flags = response.result_flags | core::ResultFlag::exact_reranked;
    }
    if (resolved.native_top_k == 0 || request.queries.rows == 0) {
      core::initialize_empty_response(response,
                                      request.queries.rows,
                                      request.options.top_k == 0
                                          ? core::SearchCompleteness::complete_k
                                          : core::SearchCompleteness::eligible_exhausted);
      return core::Status::success();
    }

    std::uint64_t output_count{};
    if (!core::checked_multiply(request.queries.rows, resolved.native_top_k, output_count) ||
        output_count > std::numeric_limits<std::size_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "DiskAnnSegment native output size is not representable");
    }
    state.labels.resize(static_cast<std::size_t>(output_count));
    state.distances.resize(static_cast<std::size_t>(output_count));
    state.counts.assign(static_cast<std::size_t>(request.queries.rows), 0);
    if (request.context->stats != nullptr) {
      state.native_stats.resize(static_cast<std::size_t>(request.queries.rows));
    }

    const float *queries = request.queries.row<float>(0);
    const auto contiguous_stride = static_cast<std::uint64_t>(meta_.dim * sizeof(float));
    if (request.queries.row_stride != contiguous_stride) {
      std::uint64_t query_values{};
      if (!core::checked_multiply(request.queries.rows, meta_.dim, query_values) ||
          query_values > std::numeric_limits<std::size_t>::max()) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::arithmetic_overflow,
                                   "DiskAnnSegment contiguous query copy is not representable");
      }
      state.contiguous_queries.resize(static_cast<std::size_t>(query_values));
      for (core::RowCount row = 0; row < request.queries.rows; ++row) {
        std::copy_n(request.queries.row<float>(row),
                    static_cast<std::size_t>(meta_.dim),
                    state.contiguous_queries.data() + row * meta_.dim);
      }
      queries = state.contiguous_queries.data();
    }

    std::optional<InflightGuard> readonly_inflight;
    if (!mutable_options_.has_value()) {
      readonly_inflight.emplace(inflight_search_);
    }
    const auto query_count = static_cast<std::uint32_t>(request.queries.rows);
    const auto workers = std::max<std::uint32_t>(1, std::min(kSearchThreads, query_count));
    const diskann::BeamSearchCancelProbe probe{&state, &NativeOperationState::probe_cancelled};
    native_->search_pipelined(queries,
                              query_count,
                              resolved.native_top_k,
                              state.labels.data(),
                              state.distances.data(),
                              workers,
                              workers,
                              resolved.native,
                              state.native_stats.empty() ? nullptr : state.native_stats.data(),
                              nullptr,
                              state.counts.data(),
                              &probe);

    const auto geometry = diskann::DiskLayoutGeometry::compute(meta_.dim, meta_.max_degree);
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      const auto native_count = state.counts[static_cast<std::size_t>(row)];
      const auto native_offset = static_cast<std::size_t>(row * resolved.native_top_k);
      std::uint32_t count{};
      for (std::uint32_t hit = 0; hit < native_count; ++hit) {
        const auto label = state.labels[native_offset + hit];
        if (state.visibility != nullptr && !state.visibility->live_labels.contains(label)) {
          continue;
        }
        response.hits[static_cast<std::size_t>(cursor + count)] =
            core::SearchHit(core::SegmentRowId(label),
                            state.distances[native_offset + hit],
                            core::ScoreKind::distance,
                            core::Metric::l2,
                            response.result_flags);
        ++count;
      }
      response.valid_counts[row] = count;
      cursor += count;
      response.offsets[row + 1] = cursor;
      response.statuses[row] = core::Status::success();
      if (count == request.options.top_k) {
        response.completeness[row] = core::SearchCompleteness::complete_k;
      } else if (meta_.live_count < request.options.top_k && count == meta_.live_count) {
        response.completeness[row] = core::SearchCompleteness::eligible_exhausted;
      } else {
        response.completeness[row] = core::SearchCompleteness::strategy_incomplete;
      }
      if (!state.native_stats.empty()) {
        update_search_stats(request.context->stats,
                            state.native_stats[static_cast<std::size_t>(row)],
                            geometry.page_size);
      }
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto unavailable(core::OperationStage stage, std::string diagnostic)
      -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               stage,
                               core::StatusDetail::operation_slot_absent,
                               std::move(diagnostic));
  }

  std::unique_ptr<diskann::DiskANNIndex> native_{};
  std::filesystem::path directory_{};
  NativeMetaSummary meta_{};
  std::uint64_t artifact_bytes_{};
  std::uint64_t resident_bytes_{};
  std::uint64_t cache_bytes_{};
  std::uint64_t scratch_bytes_{};
  std::uint64_t graph_bytes_{};
  std::optional<DiskAnnMutableSegmentOptions> mutable_options_{};
  bool owns_working_directory_{};
  mutable std::shared_mutex native_mutex_{};
  std::vector<MutableSlotState> slot_states_{};
  std::unordered_map<std::uint64_t, std::uint32_t> label_to_slot_{};
  std::shared_ptr<const MutableVisibility> mutable_visibility_{};
  std::uint64_t applied_watermark_{};
  std::uint64_t minimum_next_op_id_{1};
  std::uint64_t checkpoint_generation_{};
  internal::collection::SegmentEntryV2 pending_checkpoint_entry_{};
  std::mutex transaction_mutex_{};
  std::map<std::uint64_t, MutableTransaction> transactions_{};
  std::mutex segment_checkpoint_mutex_{};
  mutable std::mutex lifecycle_mutex_{};
  std::condition_variable lifecycle_changed_{};
  bool admission_open_{true};
  mutable std::atomic<std::uint64_t> inflight_search_{};
  std::atomic<std::uint64_t> inflight_mutation_{};
  std::atomic<std::uint64_t> pending_transactions_{};
  std::atomic<std::uint64_t> prepared_count_{};
  std::atomic<std::uint64_t> staged_count_{};
  std::atomic<std::uint64_t> committed_count_{};
  std::atomic<std::uint64_t> applied_count_{};
  std::atomic<std::uint64_t> replayed_count_{};
  std::atomic<std::uint64_t> aborted_count_{};
  std::atomic<std::uint64_t> logical_dirty_bytes_{};
  std::atomic<core::SegmentHealth> health_{core::SegmentHealth::healthy};
  std::atomic<core::StatusDetail> last_error_{core::StatusDetail::none};
};

class DiskAnnSegmentFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskAnnRegistration;

  [[nodiscard]] static auto open(
      core::ArtifactView artifacts,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled("DiskAnnSegment factory is disabled; direct DiskANNIndex is unchanged");
    }
    try {
      return DiskAnnSegment::open(artifacts, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

 private:
  [[nodiscard]] static auto disabled(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               std::move(diagnostic));
  }
};

// Internal-only mutable factory. Its feature bit defaults off and is separate
// from both the readonly Segment and the retained direct DiskANN identity.
class DiskAnnMutableSegmentFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskAnnMutableRegistration;

  [[nodiscard]] static auto open(
      core::ArtifactView artifacts,
      const core::OpenOptions &options,
      core::OpenContext &context,
      DiskAnnMutableSegmentOptions mutable_options,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled("DiskAnn mutable Segment factory is disabled");
    }
    try {
      return DiskAnnSegment::open_impl(artifacts,
                                       options,
                                       context,
                                       std::optional(std::move(mutable_options)));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto open_directory(
      const std::filesystem::path &directory,
      const core::OpenOptions &options,
      core::OpenContext &context,
      DiskAnnMutableSegmentOptions mutable_options,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled("DiskAnn mutable Segment factory is disabled");
    }
    try {
      const auto meta = DiskAnnSegment::read_meta_summary(directory / "meta.bin");
      std::vector<std::pair<std::string_view, std::string_view>> known{
          {DiskAnnSegment::kMetaArtifactName, "meta.bin"},
          {DiskAnnSegment::kIndexArtifactName, "diskann.index"},
          {DiskAnnSegment::kIdsArtifactName, "ids.bin"},
          {DiskAnnSegment::kCacheIdsArtifactName, "cache_ids.bin"},
          {DiskAnnSegment::kCacheNodesArtifactName, "cache_nodes.bin"},
      };
      if (meta.has_pq) {
        known.emplace_back(DiskAnnSegment::kPqPivotsArtifactName, "pq_pivots.bin");
        known.emplace_back(DiskAnnSegment::kPqCompressedArtifactName, "pq_compressed.bin");
      }
      if (std::filesystem::is_regular_file(directory / "slots.bin")) {
        known.emplace_back(DiskAnnSegment::kSlotsArtifactName, "slots.bin");
      }
      if (std::filesystem::is_regular_file(directory / DiskAnnSegment::kMutableStateFilename)) {
        known.emplace_back(DiskAnnSegment::kMutableStateArtifactName,
                           DiskAnnSegment::kMutableStateFilename);
      }
      std::vector<std::string> paths;
      std::vector<core::ArtifactLocation> locations;
      paths.reserve(known.size());
      locations.reserve(known.size());
      for (const auto &[unused, filename] : known) {
        (void)unused;
        paths.push_back((directory / filename).string());
      }
      for (std::size_t index = 0; index < known.size(); ++index) {
        locations.emplace_back(known[index].first, paths[index]);
      }
      return open(core::ArtifactView(locations),
                  options,
                  context,
                  std::move(mutable_options),
                  features);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto open_any(
      const std::filesystem::path &directory,
      const core::OpenOptions &options,
      core::OpenContext &context,
      DiskAnnMutableSegmentOptions mutable_options,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<core::AnySegment> {
    auto opened = open_directory(directory, options, context, std::move(mutable_options), features);
    if (!opened.ok()) {
      return opened.status();
    }
    return DiskAnnSegment::into_mutable_any(std::move(opened).value());
  }

  [[nodiscard]] static auto open_checkpoint(
      DiskAnnMutableSegmentOptions mutable_options,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<DiskAnnSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled("DiskAnn mutable Segment factory is disabled");
    }
    try {
      using namespace internal::collection;  // NOLINT(build/namespaces)
      const auto cleanup =
          ArtifactControlPlaneTransaction::cleanup_orphans(mutable_options.collection_root);
      if (!cleanup.ok()) {
        return cleanup;
      }
      const auto manifest =
          ArtifactManifestV2::load(mutable_options.collection_root / kCollectionManifestFilename);
      const auto found = std::find_if(manifest.segments.begin(),
                                      manifest.segments.end(),
                                      [&](const SegmentEntryV2 &entry) {
                                        return entry.segment_id == mutable_options.segment_id &&
                                               entry.factory_key == "diskann_mutable";
                                      });
      if (found == manifest.segments.end()) {
        return disabled("collection manifest has no DiskAnn mutable checkpoint");
      }
      const auto ready_path = mutable_options.collection_root / found->ready_marker;
      if (!std::filesystem::is_regular_file(ready_path) ||
          sha256_file(ready_path) != found->ready_digest) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskAnn mutable checkpoint READY verification failed");
      }
      const auto next = found->extensions.find("minimum_next_op_id");
      if (next != found->extensions.end()) {
        mutable_options.minimum_next_op_id =
            std::max(mutable_options.minimum_next_op_id,
                     static_cast<std::uint64_t>(std::stoull(next->second)));
      }
      std::vector<std::string> paths;
      std::vector<std::string_view> logical_names;
      std::vector<core::ArtifactLocation> locations;
      paths.reserve(found->artifacts.size());
      logical_names.reserve(found->artifacts.size());
      for (const auto &artifact : found->artifacts) {
        const bool known = artifact.logical_name == DiskAnnSegment::kMetaArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kIndexArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kIdsArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kCacheIdsArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kCacheNodesArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kPqPivotsArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kPqCompressedArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kSlotsArtifactName ||
                           artifact.logical_name == DiskAnnSegment::kMutableStateArtifactName;
        if (!known) {
          continue;
        }
        const auto path = mutable_options.collection_root / artifact.relative_path;
        if (!std::filesystem::is_regular_file(path) ||
            std::filesystem::file_size(path) != artifact.size_bytes ||
            sha256_file(path) != artifact.digest) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "DiskAnn mutable checkpoint artifact verification failed");
        }
        paths.push_back(path.string());
        logical_names.push_back(artifact.logical_name);
      }
      locations.reserve(paths.size());
      for (std::size_t index = 0; index < paths.size(); ++index) {
        locations.emplace_back(logical_names[index], paths[index]);
      }
      return open(core::ArtifactView(locations),
                  options,
                  context,
                  std::move(mutable_options),
                  features);
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  // Reader/recovery is roll-forward and is intentionally not disabled with
  // the writer gate. It advertises no mutation/checkpoint slots, but loads the
  // native index in tombstone-aware mode so a fully checkpointed mutable
  // generation remains searchable after runtime rollback.
  [[nodiscard]] static auto open_checkpoint_readonly_any(
      DiskAnnMutableSegmentOptions mutable_options,
      const core::OpenOptions &options,
      core::OpenContext &context) noexcept -> core::Result<core::AnySegment> {
    const auto recovery_options = mutable_options;
    internal::disk::DiskEngineFeatureFlags reader_features;
    reader_features.diskann_mutable_segment = true;
    auto opened = open_checkpoint(std::move(mutable_options), options, context, reader_features);
    if (!opened.ok()) {
      return opened.status();
    }
    auto replayed = replay_committed_wal_tail(*opened.value(), recovery_options);
    if (!replayed.ok()) {
      return replayed;
    }
    return DiskAnnSegment::into_roll_forward_any(std::move(opened).value());
  }

 private:
  [[nodiscard]] static auto replay_committed_wal_tail(DiskAnnSegment &segment,
                                                      const DiskAnnMutableSegmentOptions &options)
      -> core::Status {
    try {
      using namespace internal::collection;  // NOLINT(build/namespaces)
      const auto wal_path = options.collection_root / ".alaya_internal" /
                            std::string(kCollectionWalNamespace) /
                            std::string(kCollectionWalFilename);
      if (!std::filesystem::is_regular_file(wal_path)) {
        return core::Status::success();
      }
      auto scanned = CollectionLogicalWal::scan_file(wal_path);
      if (!scanned.ok()) {
        return scanned.status();
      }
      std::map<std::uint64_t, WalMutationTransaction> prepared;
      std::set<std::uint64_t> committed;
      for (const auto &frame : scanned.value().frames) {
        if (frame.type == LogicalWalRecordType::prepare) {
          prepared.insert_or_assign(frame.op_id, decode_wal_transaction(frame.payload));
          continue;
        }
        if (frame.type != LogicalWalRecordType::commit || committed.contains(frame.op_id)) {
          continue;
        }
        const auto found = prepared.find(frame.op_id);
        if (found == prepared.end()) {
          continue;
        }
        committed.insert(frame.op_id);
        const auto &transaction = found->second;
        if (transaction.rows.empty()) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::mutation_replay,
                                     core::StatusDetail::malformed_struct,
                                     "committed WAL transaction has no mutation rows");
        }
        bool targets_this_segment{};
        bool targets_another_segment{};
        for (const auto &row : transaction.rows) {
          const auto targets = row.target.segment_id == options.collection_segment_id &&
                               row.target.generation == options.segment_generation;
          targets_this_segment = targets_this_segment || targets;
          targets_another_segment = targets_another_segment || !targets;
        }
        if (!targets_this_segment) {
          continue;
        }
        if (targets_another_segment) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::mutation_replay,
                                     core::StatusDetail::malformed_struct,
                                     "one committed WAL transaction targets multiple segments");
        }
        std::vector<SegmentMutationPayload> payloads;
        payloads.reserve(transaction.rows.size());
        for (const auto &row : transaction.rows) {
          SegmentMutationPayload payload;
          payload.action = row.action;
          payload.op_id = row.op_id;
          payload.upsert_sequence = row.op_id;
          payload.target = row.target;
          payload.previous = row.previous;
          if (row.action == SegmentMutationAction::write && row.payload.vector.has_value()) {
            payload.vector = row.payload.vector->view();
          }
          payloads.push_back(std::move(payload));
        }
        SegmentMutationBundlePayload bundle;
        bundle.batch_op_id = transaction.batch_op_id;
        bundle.rows = payloads;
        const auto bundled =
            transaction.batch_mode == BatchMutationMode::all_or_nothing && payloads.size() > 1;
        core::OpaqueOperationRequest opaque;
        opaque.payload = bundled ? static_cast<const void *>(&bundle)
                                 : static_cast<const void *>(&payloads.front());
        opaque.payload_size = bundled ? sizeof(bundle) : sizeof(payloads.front());
        core::MutationContext mutation_context;
        auto status = segment.replay_mutation(opaque, mutation_context);
        if (!status.ok()) {
          return status;
        }
      }
      return core::Status::success();
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::mutation_replay,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (...) {
      return core::status_from_exception(core::OperationStage::mutation_replay);
    }
  }

  [[nodiscard]] static auto disabled(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               std::move(diagnostic));
  }
};

class DiskAnnSegmentLegacyFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskAnnRegistration;

  [[nodiscard]] static auto open(const std::filesystem::path &directory) noexcept
      -> core::Result<std::unique_ptr<diskann::DiskANNIndex>> {
    try {
      auto native = std::make_unique<diskann::DiskANNIndex>();
      diskann::DiskANNLoadParams load;
      load.num_threads = DiskAnnSegment::kSearchThreads;
      load.beam_width = DiskAnnSegment::kBeamWidth;
      load.scratch_search_list_size = DiskAnnSegment::kScratchSearchListSize;
      load.updatable = false;
      load.search_page_cache = false;
      native->load(directory.string(), load);
      return native;
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  // Differential/performance tooling invokes the retained public search API
  // on the exact native instance wrapped by the Segment. No production search
  // path bypasses the v3 translation through this helper.
  [[nodiscard]] static auto search_differential(const DiskAnnSegment &segment,
                                                const float *query,
                                                std::uint32_t top_k,
                                                std::uint64_t *labels,
                                                float *distances,
                                                const diskann::DiskANNSearchParams &options)
      -> core::Result<std::uint32_t> {
    try {
      return segment.native_->search(query, top_k, labels, distances, options);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::search);
    }
  }
};

}  // namespace alaya::disk
