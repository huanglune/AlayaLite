// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/collection/artifact_transaction.hpp"
#include "index/collection/manifest_dual_reader.hpp"
#include "index/collection/types.hpp"
#include "index/disk/disk_engine_registry.hpp"
#include "index/disk/laser_segment_searcher.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/unified_laser_segment_searcher.hpp"
#include "index/graph/laser/qg/row_admission.hpp"

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  #define ALAYA_DISK_LASER_SEGMENT_SUPPORTED 1
#else
  #define ALAYA_DISK_LASER_SEGMENT_SUPPORTED 0
#endif

namespace alaya::disk {

namespace detail {

#if ALAYA_DISK_LASER_SEGMENT_SUPPORTED
// Residency selection for a Laser segment -- moved here from
// segment_factory.hpp (that file's load_segment_from_manifest(), the only
// call site, is dead code: nothing calls it in any production path). This
// is now the actual production decision point: LaserSegment::open() below
// consults it to pick which searcher to construct. Explicit request only,
// so the default load path stays byte-identical to the legacy searcher:
//   segment manifest x_laser_residency = paged_pool | resident_arena
//   env ALAYA_LASER_RESIDENCY overrides the manifest (same values)
// Neither present -> nullopt -> legacy LaserSegmentSearcher.
inline auto laser_residency_request(const SegmentManifest &sm)
    -> std::optional<::alaya::laser::ResidencyMode> {
  const char *env = std::getenv("ALAYA_LASER_RESIDENCY");
  if (env != nullptr && *env != '\0') {
    return ::alaya::laser::residency_mode_from_string(env);
  }
  const auto it = sm.x_extras.find("x_laser_residency");
  if (it == sm.x_extras.end() || it->second.empty()) {
    return std::nullopt;
  }
  return ::alaya::laser::residency_mode_from_string(it->second);
}
#endif

}  // namespace detail

struct LaserSegmentSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t beam_width{4};
  std::uint64_t reserved[3]{};

  LaserSegmentSearchExtension()
      : header(core::current_struct_header<LaserSegmentSearchExtension>()) {}
};

[[nodiscard]] inline auto make_laser_segment_search_extension(
    const LaserSegmentSearchExtension &options) -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::laser;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

struct LaserSegmentReferenceOptions {
  std::filesystem::path collection_root{};
  std::string segment_id{};
  std::uint64_t segment_generation{1};
  std::uint64_t manifest_generation{1};
  std::string publication_parent{};
  std::uint64_t metadata_epoch{};
  std::string metadata_checkpoint{};
  std::uint64_t wal_cut{};
  internal::collection::RowVersionRangeV2 row_versions{};
  std::string id_map_checkpoint{};
  internal::collection::CollectionFeatureFlags collection_features{};
  internal::collection::ArtifactAbortPolicy abort_policy{
      internal::collection::ArtifactAbortPolicy::eager_cleanup};
  internal::collection::ArtifactTransactionFailPoint fail_point{
      internal::collection::ArtifactTransactionFailPoint::none};
  std::optional<internal::collection::ArtifactManifestV2> base_manifest{};
};

// Immutable, open-only adapter for the retained disk LASER quantized graph.
// The searcher still owns the complete beam/page/scanner loop; this class only
// validates the frozen v3 request and translates one result vector per query.
class LaserSegment {
 public:
  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::laser;
  static constexpr std::string_view kFormatName{"disk_laser_qg"};
  static constexpr std::string_view kManifestArtifactName{"manifest"};
  static constexpr std::string_view kIdsArtifactName{"ids"};

  LaserSegment(const LaserSegment &) = delete;
  auto operator=(const LaserSegment &) -> LaserSegment & = delete;
  LaserSegment(LaserSegment &&) = delete;
  auto operator=(LaserSegment &&) -> LaserSegment & = delete;

  [[nodiscard]] static auto open(core::ArtifactView artifact,
                                 const core::OpenOptions &,
                                 core::OpenContext &context)
      -> core::Result<std::unique_ptr<LaserSegment>> {
#if !ALAYA_DISK_LASER_SEGMENT_SUPPORTED
    (void)artifact;
    (void)context;
    return unavailable(core::OperationStage::open,
                       "LaserSegment is unavailable because LASER is not compiled in");
#else
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::open);
    if (!control.ok()) {
      return control;
    }
    const auto manifest_path = artifact.find(kManifestArtifactName);
    if (manifest_path.empty()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment open requires a manifest artifact");
    }
    try {
      const auto directory = std::filesystem::path(manifest_path).parent_path();
      auto native = SegmentManifest::load(std::filesystem::path(manifest_path));
      auto status = validate_native_manifest(native, core::OperationStage::open);
      if (!status.ok()) {
        return status;
      }
      const auto specs = native_artifact_specs(native);
      std::uint64_t bytes{};
      for (const auto &spec : specs) {
        const auto supplied = artifact.find(spec.logical_name);
        if (supplied.empty()) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "LaserSegment open is missing logical artifact '" +
                                         spec.logical_name + "'");
        }
        const auto expected = (directory / spec.relative_path).lexically_normal();
        if (std::filesystem::path(supplied).lexically_normal() != expected) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "LaserSegment ArtifactView paths disagree with manifest");
        }
        const auto size = std::filesystem::file_size(expected);
        if (size > std::numeric_limits<std::uint64_t>::max() - bytes) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::arithmetic_overflow,
                                     "LaserSegment artifact bytes overflow uint64");
        }
        bytes += static_cast<std::uint64_t>(size);
      }
      status = core::require_lease(context.resident_lease,
                                   bytes,
                                   core::OperationStage::open,
                                   "LaserSegment resident lease is too small for artifacts");
      if (!status.ok()) {
        return status;
      }
      status = require_io_credits(context.io_credits,
                                  specs.size(),
                                  bytes,
                                  core::OperationStage::open,
                                  "LaserSegment open I/O credits are too small");
      if (!status.ok()) {
        return status;
      }
      const auto residency = detail::laser_residency_request(native);
      if (residency.has_value() && *residency == ::alaya::laser::ResidencyMode::kResidentArena) {
        auto unified_searcher =
            std::make_shared<UnifiedLaserSegmentSearcher>(directory, *residency);
        return std::unique_ptr<LaserSegment>(new LaserSegment(nullptr,
                                                               std::move(unified_searcher),
                                                               std::move(native),
                                                               directory,
                                                               bytes));
      }
      // Default (no residency configured) and an explicit paged_pool
      // request both land here unchanged: the legacy searcher, exactly as
      // before residency selection existed.
      auto legacy_searcher = std::make_shared<LaserSegmentSearcher>(directory);
      return std::unique_ptr<LaserSegment>(new LaserSegment(std::move(legacy_searcher),
                                                             nullptr,
                                                             std::move(native),
                                                             directory,
                                                             bytes));
    } catch (const std::invalid_argument &error) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 error.what());
    } catch (const std::bad_alloc &error) {
      return core::Status::error(core::StatusCode::resource_exhausted,
                                 core::OperationStage::open,
                                 core::StatusDetail::allocation_failure,
                                 error.what(),
                                 core::Retryability::retryable_with_backoff);
    } catch (const std::exception &error) {
      return core::Status::error(core::StatusCode::io_error,
                                 core::OperationStage::open,
                                 core::StatusDetail::engine_exception,
                                 error.what());
    }
#endif
  }

  [[nodiscard]] static auto open_directory(const std::filesystem::path &segment_directory,
                                           const core::OpenOptions &options,
                                           core::OpenContext &context)
      -> core::Result<std::unique_ptr<LaserSegment>> {
    try {
      const auto native = SegmentManifest::load(segment_directory / "manifest.txt");
      const auto specs = native_artifact_specs(native);
      std::vector<std::string> paths;
      std::vector<core::ArtifactLocation> locations;
      paths.reserve(specs.size());
      locations.reserve(specs.size());
      for (const auto &spec : specs) {
        paths.push_back((segment_directory / spec.relative_path).string());
      }
      for (std::size_t index = 0; index < specs.size(); ++index) {
        locations.emplace_back(specs[index].logical_name, paths[index]);
      }
      return open(core::ArtifactView(locations), options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto open_collection(
      const std::filesystem::path &collection_root,
      std::string_view segment_id,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::collection::ManifestReaderOptions &reader_options = {})
      -> core::Result<std::unique_ptr<LaserSegment>> {
    auto opened =
        internal::collection::CollectionManifestDualReader::open(collection_root, reader_options);
    if (!opened.ok()) {
      return opened.status();
    }
    const auto &segments = opened.value().manifest.segments;
    const auto found = std::find_if(segments.begin(), segments.end(), [&](const auto &entry) {
      return entry.segment_id == segment_id;
    });
    if (found == segments.end()) {
      return core::Status::error(core::StatusCode::not_found,
                                 core::OperationStage::open,
                                 core::StatusDetail::none,
                                 "LASER segment is absent from the collection manifest");
    }
    if (found->algorithm_id != kAlgorithmId) {
      return unavailable(core::OperationStage::open, "requested collection segment is not LASER");
    }
    const auto manifest =
        std::find_if(found->artifacts.begin(), found->artifacts.end(), [](const auto &entry) {
          return entry.logical_name == kManifestArtifactName;
        });
    if (manifest == found->artifacts.end()) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "LASER manifest entry has no native manifest artifact");
    }
    try {
      const auto native = SegmentManifest::load(collection_root / manifest->relative_path);
      const auto specs = native_artifact_specs(native);
      std::vector<std::string> paths;
      std::vector<core::ArtifactLocation> locations;
      paths.reserve(specs.size());
      locations.reserve(specs.size());
      for (const auto &spec : specs) {
        const auto artifact =
            std::find_if(found->artifacts.begin(), found->artifacts.end(), [&](const auto &entry) {
              return entry.logical_name == spec.logical_name;
            });
        if (artifact == found->artifacts.end()) {
          return core::Status::error(core::StatusCode::corruption,
                                     core::OperationStage::open,
                                     core::StatusDetail::malformed_struct,
                                     "LASER manifest entry omits a native artifact");
        }
        paths.push_back((collection_root / artifact->relative_path).string());
      }
      for (std::size_t index = 0; index < specs.size(); ++index) {
        locations.emplace_back(specs[index].logical_name, paths[index]);
      }
      return open(core::ArtifactView(locations), options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = searcher_dim();
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::disk;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  [[nodiscard]] static auto make_search_extension(const LaserSegmentSearchExtension &options)
      -> core::AlgorithmSearchExtension {
    return make_laser_segment_search_extension(options);
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment single search requires exactly one query row");
    }
    return execute_search(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = searcher_size();
    stats.allocated_rows = searcher_size();
    stats.resident_bytes = artifact_bytes_;
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  // Publish only manifest-v2 ownership metadata around an importer-created,
  // read-only native LASER directory. With the writer gate off this is a true
  // no-op: no staging path, READY marker, or collection manifest is created.
  [[nodiscard]] auto publish_reference(const LaserSegmentReferenceOptions &options,
                                       core::BuildContext &context) const -> core::Status {
    if (!options.collection_features.manifest_v2_writer) {
      return core::validate_runtime_control(context.deadline,
                                            context.cancellation,
                                            core::OperationStage::save);
    }
    if (options.collection_root.empty() || options.segment_id != native_.segment_id ||
        options.segment_generation == 0 || options.manifest_generation == 0 ||
        (options.collection_root / "segments" / options.segment_id).lexically_normal() !=
            directory_.lexically_normal()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment reference publication options do not identify its "
                                 "native directory");
    }
    internal::collection::ArtifactTransactionOptions transaction_options;
    transaction_options.collection_root = options.collection_root;
    transaction_options.target_relative_directory =
        std::filesystem::path("segments") / options.segment_id;
    transaction_options.transaction_id = "disk_laser_reference_" + options.segment_id + "_g" +
                                         std::to_string(options.segment_generation);
    transaction_options.manifest_v2_writer = true;
    transaction_options.abort_policy = options.abort_policy;
    transaction_options.fail_point = options.fail_point;
    auto begun =
        internal::collection::ArtifactControlPlaneTransaction::begin(std::move(transaction_options),
                                                                     context);
    if (!begun.ok()) {
      return begun.status();
    }
    auto transaction = std::move(begun).value();
    auto status = transaction->reference_existing(native_artifact_specs(native_));
    if (!status.ok()) {
      return status;
    }
    auto prepared = transaction->prepare(make_segment_entry(options));
    if (!prepared.ok()) {
      return prepared.status();
    }
    auto manifest = make_collection_manifest(options);
    if (!manifest.ok()) {
      return manifest.status();
    }
    return transaction->publish(std::move(manifest).value());
  }

  [[nodiscard]] static auto into_any(std::unique_ptr<LaserSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null LaserSegment");
    }
    auto shared = std::shared_ptr<LaserSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.search_with_stage = false;
    config.concurrency.search_with_publish = false;
    config.concurrency.serial_mutation = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = true;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 private:
  LaserSegment(std::shared_ptr<LaserSegmentSearcher> legacy_searcher,
               std::shared_ptr<UnifiedLaserSegmentSearcher> unified_searcher,
               SegmentManifest native,
               std::filesystem::path directory,
               std::uint64_t artifact_bytes)
      : legacy_searcher_(std::move(legacy_searcher)),
        unified_searcher_(std::move(unified_searcher)),
        native_(std::move(native)),
        directory_(std::move(directory)),
        artifact_bytes_(artifact_bytes) {}

  // Common accessors dispatching to whichever searcher residency selected
  // (see laser_residency_request() and open()'s branch below) -- exactly
  // one of legacy_searcher_/unified_searcher_ is non-null. Both
  // LaserSegmentSearcher and UnifiedLaserSegmentSearcher share dim()/size()
  // via the SegmentSearcher base, but labels() is a LASER-only "unified
  // seam" accessor (see laser_segment_searcher.hpp's graph()/labels()
  // comment) neither exposes through that base, so it needs its own
  // dispatch here too.
  [[nodiscard]] auto searcher_dim() const noexcept -> std::uint32_t {
    return unified_searcher_ ? unified_searcher_->dim() : legacy_searcher_->dim();
  }
  [[nodiscard]] auto searcher_size() const noexcept -> std::uint64_t {
    return unified_searcher_ ? unified_searcher_->size() : legacy_searcher_->size();
  }
#if ALAYA_DISK_LASER_SEGMENT_SUPPORTED
  // Guarded like the rest of the LASER-only surface (see
  // resolve_search_options()/admission_aware_search() below, its only
  // callers): the stub LaserSegmentSearcher/UnifiedLaserSegmentSearcher this
  // header falls back to when a translation unit does not opt into LASER
  // (see laser_segment_header_closure.cpp) declare only the common
  // search()/size()/dim()/type() surface, not this LASER-only "unified
  // seam" accessor.
  [[nodiscard]] auto searcher_labels() const noexcept -> const std::uint64_t * {
    return unified_searcher_ ? unified_searcher_->labels() : legacy_searcher_->labels();
  }
#endif

  [[nodiscard]] static auto adapt_open_searcher(std::shared_ptr<LaserSegmentSearcher> searcher,
                                                const std::filesystem::path &directory,
                                                core::OpenContext &context)
      -> core::Result<std::unique_ptr<LaserSegment>> {
    if (searcher == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::open,
                                 core::StatusDetail::null_data,
                                 "cannot adapt a null legacy LASER searcher");
    }
    auto control = core::validate_runtime_control(context.deadline,
                                                  context.cancellation,
                                                  core::OperationStage::open);
    if (!control.ok()) {
      return control;
    }
    try {
      auto native = SegmentManifest::load(directory / "manifest.txt");
      auto status = validate_native_manifest(native, core::OperationStage::open);
      if (!status.ok()) {
        return status;
      }
      const auto specs = native_artifact_specs(native);
      std::uint64_t bytes{};
      for (const auto &spec : specs) {
        const auto size = std::filesystem::file_size(directory / spec.relative_path);
        if (size > std::numeric_limits<std::uint64_t>::max() - bytes) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::open,
                                     core::StatusDetail::arithmetic_overflow,
                                     "LaserSegment artifact bytes overflow uint64");
        }
        bytes += static_cast<std::uint64_t>(size);
      }
      status = core::require_lease(context.resident_lease,
                                   bytes,
                                   core::OperationStage::open,
                                   "LaserSegment resident lease is too small for artifacts");
      if (!status.ok()) {
        return status;
      }
      status = require_io_credits(context.io_credits,
                                  specs.size(),
                                  bytes,
                                  core::OperationStage::open,
                                  "LaserSegment open I/O credits are too small");
      if (!status.ok()) {
        return status;
      }
      // This legacy-searcher-only adapter has no production caller today
      // (see its doc comment); it always builds the legacy path, matching
      // its pre-residency-wiring behavior exactly.
      return std::unique_ptr<LaserSegment>(
          new LaserSegment(std::move(searcher), nullptr, std::move(native), directory, bytes));
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

  [[nodiscard]] static auto unavailable(core::OperationStage stage, std::string diagnostic)
      -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               stage,
                               core::StatusDetail::operation_slot_absent,
                               std::move(diagnostic));
  }

  [[nodiscard]] static auto validate_native_manifest(const SegmentManifest &native,
                                                     core::OperationStage stage) -> core::Status {
    if (native.index_type != DiskIndexType::Laser) {
      return unavailable(stage, "LaserSegment received a non-LASER native manifest");
    }
    if (native.metric != core::Metric::l2) {
      return unavailable(stage, "LaserSegment first version supports L2 only");
    }
    if (native.version != kFormatVersion || native.dim == 0 || native.count == 0 ||
        native.dim > std::numeric_limits<std::uint32_t>::max() ||
        !detail::is_valid_segment_id(native.segment_id) ||
        !detail::is_valid_basename(native.ids_file) || !native.vectors_file.empty()) {
      return core::Status::error(core::StatusCode::corruption,
                                 stage,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment native manifest is incompatible or malformed");
    }
    return core::Status::success();
  }

  [[nodiscard]] static auto native_artifact_specs(const SegmentManifest &native)
      -> std::vector<internal::collection::LogicalArtifactSpec> {
    std::vector<internal::collection::LogicalArtifactSpec> specs;
    specs.push_back({std::string(kManifestArtifactName), "manifest.txt", true, {}});
    if (!detail::is_valid_basename(native.ids_file)) {
      throw std::invalid_argument("LaserSegment native ids artifact is not a basename");
    }
    specs.push_back({std::string(kIdsArtifactName), native.ids_file, true, {}});
    constexpr std::array required_keys{"x_laser_index_file",
                                       "x_laser_rotator_file",
                                       "x_laser_cache_ids_file",
                                       "x_laser_cache_nodes_file"};
    constexpr std::array optional_keys{"x_laser_medoids_file",
                                       "x_laser_medoids_indices_file",
                                       "x_laser_pca_file"};
    auto add = [&](std::string_view key, bool required) {
      const auto found = native.x_extras.find(std::string(key));
      if (found == native.x_extras.end()) {
        if (required) {
          throw std::invalid_argument("LaserSegment native manifest lacks " + std::string(key));
        }
        return;
      }
      if (!detail::is_valid_basename(found->second)) {
        throw std::invalid_argument("LaserSegment native artifact is not a basename: " +
                                    std::string(key));
      }
      specs.push_back({std::string(key), found->second, true, {}});
    };
    for (const auto *key : required_keys) {
      add(key, true);
    }
    for (const auto *key : optional_keys) {
      add(key, false);
    }
    return specs;
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

  // `filter` is the request's compiled view (already validated to be
  // kind=none or kind=bitmap by validate_search_request()). `pid_storage`
  // is caller-owned scratch that must outlive the DiskSearchOptions this
  // returns -- one request's worth of admission_aware_search() calls.
  [[nodiscard]] auto resolve_search_options(const core::SearchOptions &options,
                                            const core::SegmentFilterView &filter,
                                            std::vector<std::uint64_t> &pid_storage) const
      -> core::Result<DiskSearchOptions> {
    DiskSearchOptions resolved;
    if (filter.kind == core::SegmentFilterKind::bitmap) {
      if (filter.payload == nullptr) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LaserSegment bitmap filter payload is null");
      }
      if (reinterpret_cast<std::uintptr_t>(filter.payload) % alignof(std::uint64_t) != 0) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LaserSegment bitmap filter payload is not word-aligned");
      }
#if ALAYA_DISK_LASER_SEGMENT_SUPPORTED
      // PID/row space note: Collection's SegmentRowId for a LASER segment
      // is the external label, not the internal PID -- see the
      // SegmentRowId(label) hit construction in execute_search() below.
      // PIDs are a private LASER implementation detail (never exposed past
      // this file), so a bitmap arriving from Collection is indexed by
      // label. This re-derives a dense PID-indexed bitmap from it via this
      // segment's own PID->label map (searcher_labels()), at O(size())
      // -- paid once per request setup, never per candidate.
      const std::uint64_t row_capacity_bits = filter.payload_size * 8;
      const auto *label_bits = static_cast<const std::uint64_t *>(filter.payload);
      const std::uint64_t *labels = searcher_labels();
      pid_storage.assign(laser::admission_words_for_capacity(searcher_size()), std::uint64_t{0});
      for (std::uint64_t pid = 0; pid < searcher_size(); ++pid) {
        const std::uint64_t label = labels[pid];
        if (label >= row_capacity_bits) {
          continue;
        }
        if (((label_bits[label >> 6U] >> (label & 63U)) & 1ULL) != 0ULL) {
          pid_storage[pid >> 6U] |= (std::uint64_t{1} << (pid & 63U));
        }
      }
      resolved.filter.kind = core::SegmentFilterKind::bitmap;
      resolved.filter.payload = pid_storage.data();
      resolved.filter.payload_size = pid_storage.size() * sizeof(std::uint64_t);
#else
      // Unreachable: LaserSegment::open() refuses to construct a segment
      // (and therefore legacy_searcher_/unified_searcher_) without LASER
      // support, so this method never actually runs in that configuration.
      // Kept branch-compatible only so the translation above can call the
      // LASER-only searcher_labels() without an extra specialization.
      (void)pid_storage;
#endif
    }
    const auto default_ef = native_.x_extras.find("x_default_ef");
    const auto default_beam = native_.x_extras.find("x_default_beam_width");
    if (default_ef != native_.x_extras.end()) {
      const auto parsed = detail::parse_uint64(default_ef->second, "x_default_ef");
      if (parsed == 0 || parsed > std::numeric_limits<std::uint32_t>::max()) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LASER default effort is outside uint32");
      }
      resolved.ef = static_cast<std::uint32_t>(parsed);
    }
    if (default_beam != native_.x_extras.end()) {
      const auto parsed = detail::parse_uint64(default_beam->second, "x_default_beam_width");
      if (parsed == 0 || parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return core::Status::error(core::StatusCode::corruption,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LASER default beam width is outside int");
      }
      resolved.beam_width = static_cast<std::uint32_t>(parsed);
    }
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     "LaserSegment received an extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr ||
          extension.payload_size < sizeof(LaserSegmentSearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LaserSegment search extension payload is truncated");
      }
      const auto &typed = *static_cast<const LaserSegmentSearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed) || typed.effort == 0 || typed.beam_width == 0 ||
          typed.beam_width > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "LaserSegment search extension values are invalid");
      }
      resolved.ef = typed.effort;
      resolved.beam_width = typed.beam_width;
    }
    resolved.top_k = static_cast<std::uint32_t>(options.top_k);
    resolved.exact_rerank = false;
    return resolved;
  }

  [[nodiscard]] auto validate_search_request(const core::SearchRequest &request) const
      -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment search request is incomplete or incompatible");
    }
    auto status =
        core::validate_tensor(request.queries, searcher_dim(), core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::ScalarType::float32) {
      return unavailable(core::OperationStage::validation,
                         "LaserSegment search accepts float32 tensors only");
    }
    status = core::validate_response(*request.response,
                                     request.queries.rows,
                                     request.options.top_k,
                                     core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.filter.kind != core::SegmentFilterKind::none &&
        request.filter.kind != core::SegmentFilterKind::bitmap) {
      return unavailable(core::OperationStage::validation,
                         "LaserSegment only accepts a compiled bitmap filter view");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "LaserSegment top_k exceeds uint32");
    }
    std::vector<std::uint64_t> validation_pid_storage;
    auto resolved = resolve_search_options(request.options, request.filter, validation_pid_storage);
    if (!resolved.ok()) {
      return resolved.status();
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }
    const auto result_rows = std::min<std::uint64_t>(request.options.top_k, searcher_size());
    std::uint64_t scratch_per_row{};
    std::uint64_t scratch{};
    if (!core::checked_multiply(result_rows,
                                sizeof(std::uint32_t) + sizeof(DiskSearchHit),
                                scratch_per_row) ||
        !core::checked_multiply(scratch_per_row, request.queries.rows, scratch)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "LaserSegment query scratch size overflows uint64");
    }
    status = core::require_lease(request.context->query_scratch_lease,
                                 scratch,
                                 core::OperationStage::search,
                                 "LaserSegment query scratch lease is too small");
    if (!status.ok()) {
      return status;
    }
    std::uint64_t io_bytes{};
    if (!core::checked_multiply(artifact_bytes_, request.queries.rows, io_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "LaserSegment search I/O accounting overflows uint64");
    }
    return require_io_credits(request.context->io_credits,
                              request.queries.rows,
                              io_bytes,
                              core::OperationStage::search,
                              "LaserSegment search I/O credits are too small");
  }

  [[nodiscard]] auto execute_search(
      const core::SearchRequest &request,
      std::vector<std::vector<DiskSearchHit>> *direct_results = nullptr) const -> core::Status {
    auto status = validate_search_request(request);
    if (!status.ok()) {
      return status;
    }
    const bool admission_active = request.filter.kind != core::SegmentFilterKind::none;
    const auto hit_flags = admission_active
                               ? (core::ResultFlag::approximate | core::ResultFlag::filtered)
                               : core::ResultFlag::approximate;
    auto &response = *request.response;
    response.score_kind = core::ScoreKind::rank_only;
    response.comparable_metric = core::Metric::l2;
    response.result_flags = hit_flags;
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      core::initialize_empty_response(response,
                                      request.queries.rows,
                                      request.options.top_k == 0
                                          ? core::SearchCompleteness::complete_k
                                          : core::SearchCompleteness::eligible_exhausted);
      return core::Status::success();
    }
    std::vector<std::uint64_t> pid_admission_storage;
    auto resolved = resolve_search_options(request.options, request.filter, pid_admission_storage);
    if (!resolved.ok()) {
      return resolved.status();
    }
    const auto options = std::move(resolved).value();
    if (direct_results != nullptr) {
      direct_results->clear();
      direct_results->resize(static_cast<std::size_t>(request.queries.rows));
    }
    response.query_count = request.queries.rows;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      const auto control = core::validate_runtime_control(request.context->deadline,
                                                          request.context->cancellation,
                                                          core::OperationStage::search);
      if (!control.ok()) {
        for (core::RowCount remaining = row; remaining < request.queries.rows; ++remaining) {
          response.offsets[remaining + 1] = cursor;
          response.valid_counts[remaining] = 0;
          response.statuses[remaining] = control;
          response.completeness[remaining] = core::SearchCompleteness::failed;
        }
        return request.queries.rows == 1 ? control : core::Status::success();
      }
      try {
        const auto hits = admission_aware_search(request.queries.row<float>(row), options);
        for (std::size_t index = 0; index < hits.size(); ++index) {
          response.hits[static_cast<std::size_t>(cursor + index)] =
              core::SearchHit(core::SegmentRowId(hits[index].label),
                              hits[index].distance,
                              core::ScoreKind::rank_only,
                              core::Metric::l2,
                              hit_flags);
        }
        if (direct_results != nullptr) {
          (*direct_results)[static_cast<std::size_t>(row)] = hits;
        }
        const auto written = static_cast<core::RowCount>(hits.size());
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        if (written == request.options.top_k) {
          response.completeness[row] = core::SearchCompleteness::complete_k;
        } else if (request.options.top_k > searcher_size() && written == searcher_size()) {
          response.completeness[row] = core::SearchCompleteness::eligible_exhausted;
        } else {
          response.completeness[row] = core::SearchCompleteness::strategy_incomplete;
        }
      } catch (...) {
        const auto failure = core::status_from_exception(core::OperationStage::search);
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = 0;
        response.statuses[row] = failure;
        response.completeness[row] = core::SearchCompleteness::failed;
        if (request.queries.rows == 1) {
          return failure;
        }
      }
    }
    if (request.context->stats != nullptr) {
      const auto visited =
          std::min<std::uint64_t>(searcher_size(),
                                  std::max<std::uint64_t>(options.ef, options.top_k));
      request.context->stats->visited += visited * request.queries.rows;
      request.context->stats->io_requests += request.queries.rows;
      request.context->stats->io_bytes += artifact_bytes_ * request.queries.rows;
    }
    return core::Status::success();
  }

  // Unified-segment seam analog (see UnifiedLaserSegmentSearcher, decision
  // 5 of the U2-b manifest, and decision 7 of U2-c for the residency split
  // below). When unified_searcher_ is active (kResidentArena residency),
  // everything routes through its own search(), which already compiles
  // admission internally for both residencies it supports -- see the
  // unified_searcher_ branch at the top of this function. Everything below
  // this comment is legacy_searcher_-only: LaserSegmentSearcher::search()
  // itself is never modified and never reads DiskSearchOptions.filter, so a
  // non-none filter bypasses it and drives the kernel through
  // legacy_searcher_->graph() directly -- the same lock +
  // set_params-if-different discipline LaserSegmentSearcher::search() uses
  // internally, reimplemented here since that discipline lives behind a
  // private member this class cannot reach. kind=none keeps calling
  // legacy_searcher_->search() untouched (byte-identical to before the
  // admission contract landed, and to before residency selection existed:
  // legacy_searcher_ is populated in exactly the cases searcher_ used to
  // be populated unconditionally).
  //
  // Concurrency note: this uses a mutex private to LaserSegment, disjoint
  // from LaserSegmentSearcher's own search_mutex_ that the kind=none path
  // still goes through. A single LaserSegment instance receiving a genuine
  // concurrent mix of kind=none and an active-filter request, with
  // different ef/beam_width between them, has a narrow race window around
  // QuantizedGraph::set_params() -- the same shape of hazard
  // UnifiedLaserSegmentSearcher's kPagedPool/kResidentArena split already
  // carries (each mode's branch also serializes through its own mutex).
  // Workloads that hold ef/beam_width constant per collection (the common
  // case) call set_params() at most once and never hit it.
  [[nodiscard]] auto admission_aware_search(const float *query,
                                            const DiskSearchOptions &options) const
      -> std::vector<DiskSearchHit> {
    if (unified_searcher_) {
      // Decision 7 (U2-c manifest): in unified/resident-arena mode, filter
      // and admission go through UnifiedLaserSegmentSearcher's own
      // search(), which already compiles admission internally (bitmap and
      // sorted_rows, see its compile_admission()) for both residencies it
      // supports, and whose kind=none path is byte-identical to the legacy
      // one when paged. The manual bypass below exists only because
      // LaserSegmentSearcher::search() itself never reads
      // DiskSearchOptions.filter -- that limitation does not apply to
      // UnifiedLaserSegmentSearcher, so it never needs the bypass.
      return unified_searcher_->search(query, options);
    }
    if (options.filter.kind == core::SegmentFilterKind::none) {
      return legacy_searcher_->search(query, options);
    }
#if ALAYA_DISK_LASER_SEGMENT_SUPPORTED
    const std::lock_guard<std::mutex> lock(admission_search_mutex_);
    auto &graph = legacy_searcher_->graph();
    const auto effective_top_k = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(options.top_k), searcher_size()));
    const AdmissionLastSetParams requested{
        static_cast<std::size_t>(std::max(options.ef, effective_top_k)),
        1,
        static_cast<int>(options.beam_width),
    };
    if (requested != admission_last_set_params_) {
      graph.set_params(requested.ef_search, requested.num_threads, requested.beam_width);
      admission_last_set_params_ = requested;
    }

    laser::RowAdmission admission_value{};
    const laser::RowAdmission *admission = nullptr;
    if (options.filter.kind == core::SegmentFilterKind::bitmap) {
      // Already PID-indexed: resolve_search_options() performed the
      // label->PID translation before storing this filter.
      admission_value = laser::admission_from_bitmap_payload(options.filter.payload,
                                                             options.filter.payload_size,
                                                             searcher_size());
      admission = &admission_value;
    }

    std::vector<std::uint32_t> pid_buf(effective_top_k);
    graph.search(query, effective_top_k, pid_buf.data(), admission);

    const std::uint64_t *labels = searcher_labels();
    std::vector<DiskSearchHit> out;
    out.reserve(effective_top_k);
    for (std::uint32_t pid : pid_buf) {
      if (pid >= searcher_size()) {
        throw std::runtime_error("LaserSegment: QuantizedGraph returned PID " +
                                 std::to_string(pid) + " outside segment count " +
                                 std::to_string(searcher_size()));
      }
      out.push_back(DiskSearchHit{labels[pid], std::numeric_limits<float>::quiet_NaN()});
    }
    return out;
#else
    // Unreachable: LaserSegment::open() refuses to construct a segment (and
    // therefore legacy_searcher_) without LASER support, so this branch
    // never actually runs in that configuration. legacy_searcher_->graph()
    // is a LASER-only member the stub SegmentSearcher does not declare.
    return legacy_searcher_->search(query, options);
#endif
  }

  [[nodiscard]] static auto make_segment_entry(const LaserSegmentReferenceOptions &options)
      -> internal::collection::SegmentEntryV2 {
    internal::collection::SegmentEntryV2 entry;
    entry.segment_id = options.segment_id;
    entry.generation = options.segment_generation;
    entry.role = internal::collection::SegmentRoleV2::searchable;
    entry.algorithm_id = kAlgorithmId;
    entry.format_version = kFormatVersion;
    entry.factory_key = "laser";
    entry.capabilities.operations = core::capability_bit(core::OperationCapability::search) |
                                    core::capability_bit(core::OperationCapability::batch_search) |
                                    core::capability_bit(core::OperationCapability::stats);
    entry.capabilities.reentrant_search = true;
    entry.capabilities.cooperative_cancel = true;
    entry.capabilities.explicit_drain = false;
    entry.lifecycle = internal::collection::SegmentLifecycleV2::sealed;
    entry.wal_cut = options.wal_cut;
    entry.row_versions = options.row_versions;
    entry.id_map_checkpoint = options.id_map_checkpoint;
    entry.reader_compatibility.required_features = {"disk_laser_segment"};
    entry.extensions.emplace("format_name", std::string(kFormatName));
    entry.extensions.emplace("score_kind", "rank_only");
    entry.extensions.emplace("numeric_score_comparable", "false");
    entry.extensions.emplace("native_payload", "read_only_reference");
    return entry;
  }

  [[nodiscard]] auto make_collection_manifest(const LaserSegmentReferenceOptions &options) const
      -> core::Result<internal::collection::ArtifactManifestV2> {
    internal::collection::ArtifactManifestV2 manifest;
    if (options.base_manifest.has_value()) {
      manifest = *options.base_manifest;
    } else if (std::filesystem::is_regular_file(
                   options.collection_root / internal::collection::kCollectionManifestFilename)) {
      internal::collection::ManifestReaderOptions reader_options;
      reader_options.verify_artifacts = false;
      auto opened =
          internal::collection::CollectionManifestDualReader::open(options.collection_root,
                                                                   reader_options);
      if (!opened.ok()) {
        return opened.status();
      }
      manifest = std::move(opened).value().manifest;
    }
    if (manifest.collection.dim != 0 &&
        (manifest.collection.dim != searcher_dim() ||
         manifest.collection.metric != core::Metric::l2 ||
         manifest.collection.scalar_type != core::ScalarType::float32)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::save,
                                 core::StatusDetail::malformed_struct,
                                 "LaserSegment publication disagrees with collection schema");
    }
    manifest.collection.dim = searcher_dim();
    manifest.collection.metric = core::Metric::l2;
    manifest.collection.scalar_type = core::ScalarType::float32;
    manifest.collection.logical_id_encoding =
        internal::collection::LogicalIdEncodingV2::legacy_u64_le;
    manifest.collection.metadata_epoch = options.metadata_epoch;
    manifest.collection.metadata_checkpoint = options.metadata_checkpoint;
    manifest.publication.generation = options.manifest_generation;
    manifest.publication.parent = options.publication_parent;
    manifest.wal_cut = options.wal_cut;
    manifest.row_versions = options.row_versions;
    manifest.id_map_checkpoint = options.id_map_checkpoint;
    const auto numeric_id = static_cast<std::uint64_t>(std::stoull(options.segment_id.substr(4)));
    manifest.next_segment_id_hint =
        std::max(manifest.next_segment_id_hint, numeric_id + std::uint64_t{1});
    return manifest;
  }

  struct AdmissionLastSetParams {
    std::size_t ef_search{};
    std::size_t num_threads{};
    int beam_width{};

    friend auto operator==(const AdmissionLastSetParams &, const AdmissionLastSetParams &)
        -> bool = default;
  };

  // Exactly one of these is non-null, chosen once at construction time by
  // the residency this segment opened with. legacy_searcher_ is the
  // default/paged-pool path (byte-identical to pre-residency-wiring
  // behavior when no residency is configured); unified_searcher_ is
  // populated only when a manifest/env residency request resolves to
  // kResidentArena.
  std::shared_ptr<LaserSegmentSearcher> legacy_searcher_{};
  std::shared_ptr<UnifiedLaserSegmentSearcher> unified_searcher_{};
  SegmentManifest native_{};
  std::filesystem::path directory_{};
  std::uint64_t artifact_bytes_{};
  mutable std::mutex admission_search_mutex_{};
  mutable AdmissionLastSetParams admission_last_set_params_{};
};

class LaserSegmentFactory {
 public:
  static constexpr auto registration = internal::disk::kDiskLaserRegistration;

  [[nodiscard]] static auto open(
      core::ArtifactView artifacts,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::disk::DiskEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<LaserSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled("LaserSegment factory is disabled; DiskCollection v1 is unchanged");
    }
#if !ALAYA_DISK_LASER_SEGMENT_SUPPORTED
    return disabled("LaserSegment factory is unavailable because LASER is not compiled in");
#else
    try {
      return LaserSegment::open(artifacts, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
#endif
  }

 private:
  [[nodiscard]] static auto disabled(std::string diagnostic) -> core::Status {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               std::move(diagnostic));
  }
};
}  // namespace alaya::disk

#undef ALAYA_DISK_LASER_SEGMENT_SUPPORTED
