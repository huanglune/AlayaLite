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
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/disk/disk_engine_registry.hpp"
#include "index/graph/diskann/diskann_index.hpp"

namespace alaya::disk {

class DiskAnnSegmentLegacyFactory;

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
  static constexpr std::uint32_t kSearchThreads = 4;
  static constexpr std::uint32_t kBeamWidth = 4;
  static constexpr std::uint32_t kScratchSearchListSize =
      diskann::kDefaultDiskANNScratchSearchListSize;

  DiskAnnSegment(const DiskAnnSegment &) = delete;
  auto operator=(const DiskAnnSegment &) -> DiskAnnSegment & = delete;
  DiskAnnSegment(DiskAnnSegment &&) = delete;
  auto operator=(DiskAnnSegment &&) -> DiskAnnSegment & = delete;

  [[nodiscard]] static auto open(core::ArtifactView artifact,
                                 const core::OpenOptions &options,
                                 core::OpenContext &context)
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
      const auto directory = meta_path.parent_path();
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
        const auto expected = (directory / spec.filename).lexically_normal();
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

      auto scratch = estimate_scratch_bytes(meta);
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
      status = require_io_credits(context.io_credits,
                                  specs.size(),
                                  artifact_bytes,
                                  core::OperationStage::open,
                                  "DiskAnnSegment open I/O credits are too small");
      if (!status.ok()) {
        return status;
      }

      auto native = std::make_unique<diskann::DiskANNIndex>();
      diskann::DiskANNLoadParams load;
      load.num_threads = kSearchThreads;
      load.beam_width = kBeamWidth;
      load.scratch_search_list_size = kScratchSearchListSize;
      load.updatable = false;
      load.search_page_cache = false;
      native->load(directory.string(), load);
      if (native->dim() != meta.dim || native->size() != meta.live_count ||
          native->max_slot_id() != meta.max_slot_id) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::open,
                                   core::StatusDetail::malformed_struct,
                                   "DiskANN loaded state disagrees with meta.bin");
      }
      // A static readonly load cannot apply a persisted tombstone bitmap. Do
      // not return wrong rows from a mutated artifact; G8-B owns that reader.
      if (meta.live_count != meta.max_slot_id) {
        return unavailable(core::OperationStage::open,
                           "DiskAnnSegment readonly cannot open a tombstoned mutable artifact");
      }
      return std::unique_ptr<DiskAnnSegment>(new DiskAnnSegment(std::move(native),
                                                                directory,
                                                                meta,
                                                                artifact_bytes,
                                                                resident_bytes,
                                                                cache_bytes,
                                                                scratch.value(),
                                                                graph_bytes));
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

 private:
  friend class DiskAnnSegmentLegacyFactory;

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

  struct ResolvedSearch {
    diskann::DiskANNSearchParams native{};
    std::uint32_t native_top_k{};
  };

  struct NativeOperationState final : core::detail::AsyncOperationState {
    std::shared_ptr<DiskAnnSegment> segment{};
    std::vector<std::uint64_t> labels{};
    std::vector<float> distances{};
    std::vector<std::uint32_t> counts{};
    std::vector<diskann::SearchStats> native_stats{};
    std::vector<float> contiguous_queries{};
    std::atomic<core::StatusCode> observed_control{core::StatusCode::ok};

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
      this->finish(std::move(status),
                   std::static_pointer_cast<core::detail::AsyncOperationState>(self));
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

  [[nodiscard]] static auto start_native_search(const std::shared_ptr<void> &instance,
                                                core::SearchRequest request,
                                                core::SearchCompletion completion) noexcept
      -> core::Result<core::OperationHandle> {
    try {
      auto state = std::make_shared<NativeOperationState>();
      state->segment = std::static_pointer_cast<DiskAnnSegment>(instance);
      state->request = std::move(request);
      state->completion = std::move(completion);
      state->bind_context();
      std::thread([state] {
        state->run(state);
      }).detach();
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
                                                [](DiskAnnSegment *) noexcept {});
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
                 std::uint64_t graph_bytes)
      : native_(std::move(native)),
        directory_(std::move(directory)),
        meta_(meta),
        artifact_bytes_(artifact_bytes),
        resident_bytes_(resident_bytes),
        cache_bytes_(cache_bytes),
        scratch_bytes_(scratch_bytes),
        graph_bytes_(graph_bytes) {}

  template <class T>
  static void read_exact(std::ifstream &input, T &value, const char *diagnostic) {
    input.read(reinterpret_cast<char *>(std::addressof(value)), sizeof(value));
    if (!input) {
      throw std::invalid_argument(diagnostic);
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

  [[nodiscard]] static auto estimate_scratch_bytes(const NativeMetaSummary &meta)
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
      // The retained synchronous pool and the bounded native coroutine
      // pipeline each cache at most kSearchThreads ThreadData instances.
      std::uint64_t total{};
      if (!core::checked_multiply(per_thread, kSearchThreads * 2, total)) {
        throw std::overflow_error("DiskANN scratch-pool estimate overflows uint64");
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

    InflightGuard inflight(inflight_search_);
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
      const auto count = state.counts[static_cast<std::size_t>(row)];
      const auto native_offset = static_cast<std::size_t>(row * resolved.native_top_k);
      for (std::uint32_t hit = 0; hit < count; ++hit) {
        response.hits[static_cast<std::size_t>(cursor + hit)] =
            core::SearchHit(core::SegmentRowId(state.labels[native_offset + hit]),
                            state.distances[native_offset + hit],
                            core::ScoreKind::distance,
                            core::Metric::l2,
                            response.result_flags);
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
  mutable std::atomic<std::uint64_t> inflight_search_{};
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
