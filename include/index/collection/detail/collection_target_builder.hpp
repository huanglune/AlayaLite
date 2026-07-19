// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/collection/detail/collection_flat_target.hpp"
#include "index/collection/detail/collection_memory_target.hpp"
#include "index/collection/detail/collection_normalized_segment.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/laser_segment_importer.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "platform/fs.hpp"

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  #include "index/graph/laser/qg/qg_builder.hpp"
  #include "index/graph/vamana/vamana_builder.hpp"
  #include "index/graph/vamana/vamana_writer.hpp"
#endif

namespace alaya {
enum class CollectionQuantization : std::uint8_t;
}

namespace alaya::internal::collection::detail {

inline constexpr std::string_view kQgLaserImplementationKey{"qg_laser_segment"};

struct CollectionTargetBuildParams {
  CollectionQuantization quantization{};
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{400};
  std::uint32_t thread_count{1};
  float alpha{1.2F};
  std::uint64_t seed{1234};
};

struct CollectionTargetBuildResult {
  core::AnySegment segment{};
  core::AlgorithmId requested_algorithm{};
  core::AlgorithmId built_algorithm{};
  std::string implementation_key{};
  std::string factory_key{};
  std::uint64_t artifact_bytes{};
  std::uint32_t effective_ef_construction{};
  bool flat_fallback{};
  std::string fallback_reason{};
};

enum class TargetSupport : std::uint8_t {
  unsupported = 0,
  supported = 1,
};

using BuildTargetFn =
    core::Result<CollectionTargetBuildResult> (*)(const CollectionSchema &,
                                                  std::span<const RegisteredRow>,
                                                  const CollectionTargetBuildParams &,
                                                  const CollectionTargetPublication &,
                                                  core::BuildContext &);

using OpenTargetFn = core::Result<core::AnySegment> (*)(const std::filesystem::path &,
                                                        const SegmentEntryV2 &,
                                                        const CollectionSchema &,
                                                        core::OpenContext &);

struct CollectionTargetRegistration {
  core::AlgorithmId algorithm_id{};
  std::string_view implementation_key{};
  std::string_view factory_key{};
  TargetSupport (*supports)(const CollectionSchema &,
                            core::RowCount,
                            const CollectionTargetBuildParams &) = nullptr;
  BuildTargetFn build{};
  OpenTargetFn open{};
};

[[nodiscard]] inline auto flat_target_support(const CollectionSchema &,
                                              core::RowCount,
                                              const CollectionTargetBuildParams &) -> TargetSupport;

[[nodiscard]] inline auto unsupported_target_support(const CollectionSchema &,
                                                     core::RowCount,
                                                     const CollectionTargetBuildParams &)
    -> TargetSupport;

[[nodiscard]] inline auto qg_target_support(const CollectionSchema &schema,
                                            core::RowCount row_count,
                                            const CollectionTargetBuildParams &params)
    -> TargetSupport;

[[nodiscard]] inline auto laser_target_support(const CollectionSchema &schema,
                                               core::RowCount row_count,
                                               const CollectionTargetBuildParams &params)
    -> TargetSupport;

[[nodiscard]] inline auto build_flat_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult>;

[[nodiscard]] inline auto build_unsupported_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult>;

[[nodiscard]] inline auto build_qg_collection_target(const CollectionSchema &schema,
                                                     std::span<const RegisteredRow> rows,
                                                     const CollectionTargetBuildParams &params,
                                                     const CollectionTargetPublication &publication,
                                                     core::BuildContext &context)
    -> core::Result<CollectionTargetBuildResult>;

[[nodiscard]] inline auto build_qg_laser_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult>;

[[nodiscard]] inline auto build_laser_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult>;

[[nodiscard]] inline auto open_flat_collection_target(const std::filesystem::path &root,
                                                      const SegmentEntryV2 &entry,
                                                      const CollectionSchema &schema,
                                                      core::OpenContext &context)
    -> core::Result<core::AnySegment>;

[[nodiscard]] inline auto open_unsupported_collection_target(const std::filesystem::path &root,
                                                             const SegmentEntryV2 &entry,
                                                             const CollectionSchema &schema,
                                                             core::OpenContext &context)
    -> core::Result<core::AnySegment>;

[[nodiscard]] inline auto open_qg_collection_target(const std::filesystem::path &root,
                                                    const SegmentEntryV2 &entry,
                                                    const CollectionSchema &schema,
                                                    core::OpenContext &context)
    -> core::Result<core::AnySegment>;

[[nodiscard]] inline auto open_laser_collection_target(const std::filesystem::path &root,
                                                       const SegmentEntryV2 &entry,
                                                       const CollectionSchema &schema,
                                                       core::OpenContext &context)
    -> core::Result<core::AnySegment>;

inline constexpr std::array<CollectionTargetRegistration, 3> kCollectionTargetRegistrations{
    CollectionTargetRegistration{core::algorithm::flat,
                                 "disk_flat_segment",
                                 "flat",
                                 flat_target_support,
                                 build_flat_collection_target,
                                 open_flat_collection_target},
    CollectionTargetRegistration{core::algorithm::qg,
                                 kQgLaserImplementationKey,
                                 "qg",
                                 qg_target_support,
                                 build_qg_laser_collection_target,
                                 open_qg_collection_target},
    CollectionTargetRegistration{core::algorithm::laser,
                                 "disk_laser_segment",
                                 "laser",
                                 laser_target_support,
                                 build_laser_collection_target,
                                 open_laser_collection_target},
};

[[nodiscard]] inline auto collection_target_registrations()
    -> std::span<const CollectionTargetRegistration> {
  return kCollectionTargetRegistrations;
}

[[nodiscard]] inline auto find_collection_target_registration(core::AlgorithmId algorithm_id)
    -> const CollectionTargetRegistration * {
  const auto found = std::ranges::find_if(kCollectionTargetRegistrations, [&](const auto &entry) {
    return entry.algorithm_id == algorithm_id;
  });
  return found == kCollectionTargetRegistrations.end() ? nullptr : &*found;
}

[[nodiscard]] inline auto find_collection_target_registration(std::string_view factory_key)
    -> const CollectionTargetRegistration * {
  const auto found = std::ranges::find_if(kCollectionTargetRegistrations, [&](const auto &entry) {
    return entry.factory_key == factory_key;
  });
  return found == kCollectionTargetRegistrations.end() ? nullptr : &*found;
}

[[nodiscard]] inline auto flat_target_support(const CollectionSchema &,
                                              core::RowCount,
                                              const CollectionTargetBuildParams &)
    -> TargetSupport {
  return TargetSupport::supported;
}

[[nodiscard]] inline auto unsupported_target_support(const CollectionSchema &,
                                                     core::RowCount,
                                                     const CollectionTargetBuildParams &)
    -> TargetSupport {
  return TargetSupport::unsupported;
}

[[nodiscard]] inline auto qg_target_support(const CollectionSchema &schema,
                                            core::RowCount row_count,
                                            const CollectionTargetBuildParams &params)
    -> TargetSupport {
  // The public qg id now resolves to the sealed LASER implementation, so its
  // schema gate is intentionally identical to the direct LASER target. This
  // function does not consult the compile-time LASER flag: on a platform where
  // LASER is absent, every qg request must reach the registered builder and
  // fail explicitly with not_supported. Returning unsupported for even a
  // small/otherwise-ineligible generation would make Collection silently build
  // Flat and hide the platform limitation.
#if !(defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0)
  (void)schema;
  (void)row_count;
  (void)params;
  return TargetSupport::supported;
#else
  return laser_target_support(schema, row_count, params);
#endif
}

// Mirrors qg_target_support's shape (same quantization/scalar/row-count
// gauge -- RaBitQSpace<>::kDegreeBound -- since laser is the on-disk sibling
// of the same RaBitQ-quantized-graph family). L2 is native; inner_product
// dispatches the LASER IP factor/exact kernels; cosine stores normalized rows
// and wraps query normalization at the Collection boundary.
//   - dim: LaserSegmentImporter admits [33, 2048]. LASER's single-round
//     FHTRotator pads a non-power-of-two dimension to 2^ceil(log2(dim)); the
//     RaBitQ codebook and FastScan consume that padded width while raw/exact
//     terms retain schema.dim. Gating on the same range here makes the
//     "unsupported -> fall back to flat" contract hold; otherwise an
//     incompatible dim would only be discovered inside
//     build_laser_collection_target() as a build failure.
// Any schema/row-count combination that fails this returns `unsupported`,
// so resolve_build_algorithm() (collection.hpp) silently falls back to
// flat -- this function's return value is the only signal that decides
// that, it never partially commits.
[[nodiscard]] inline auto laser_target_support(const CollectionSchema &schema,
                                               core::RowCount row_count,
                                               const CollectionTargetBuildParams &params)
    -> TargetSupport {
  constexpr std::uint8_t kRaBitQQuantization = 3;
  const auto dim_supported =
      ::alaya::disk::laser_importer_detail::dimension_supported_v1(schema.dim);
  const auto metric_supported = schema.metric == core::Metric::l2 ||
                                schema.metric == core::Metric::inner_product ||
                                schema.metric == core::Metric::cosine;
  return static_cast<std::uint8_t>(params.quantization) == kRaBitQQuantization &&
                 schema.scalar_type == core::ScalarType::float32 && metric_supported &&
                 dim_supported && row_count > ::alaya::RaBitQSpace<>::kDegreeBound
             ? TargetSupport::supported
             : TargetSupport::unsupported;
}

template <typename Scalar>
[[nodiscard]] inline auto harvest_memory_graph_vectors(const CollectionSchema &schema,
                                                       std::span<const RegisteredRow> rows,
                                                       std::string_view engine_name)
    -> core::Result<std::vector<Scalar>> {
  try {
    std::vector<Scalar> vectors;
    std::uint64_t live_count{};
    for (const auto &row : rows) {
      if (row.state != VersionState::live) {
        continue;
      }
      if (!row.payload.vector.has_value()) {
        return core::Status::error(core::StatusCode::resource_exhausted,
                                   core::OperationStage::build,
                                   core::StatusDetail::budget_denied,
                                   std::string(engine_name) +
                                       " seal target requires every live source vector");
      }
      if (static_cast<std::uint64_t>(row.row_id) != live_count) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   std::string(engine_name) +
                                       " seal target requires dense live row IDs in vector order");
      }
      const auto view = row.payload.vector->view();
      if (view.scalar_type != core::scalar_type_for<Scalar> || view.dim != schema.dim) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::build,
                                   core::StatusDetail::malformed_struct,
                                   std::string(engine_name) +
                                       " source vector disagrees with the Collection schema");
      }
      const auto *source = view.template row<Scalar>(0);
      vectors.insert(vectors.end(), source, source + schema.dim);
      ++live_count;
    }
    if (live_count == 0) {
      return core::Status::error(core::StatusCode::not_found,
                                 core::OperationStage::build,
                                 core::StatusDetail::none,
                                 "cannot build an empty " + std::string(engine_name) +
                                     " seal target");
    }
    if (live_count > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::arithmetic_overflow,
                                 std::string(engine_name) +
                                     " seal target row count exceeds uint32");
    }
    return vectors;
  } catch (...) {
    return core::status_from_exception(core::OperationStage::build);
  }
}

[[nodiscard]] inline auto disk_flat_publication_from_collection(
    const CollectionTargetPublication &publication) -> ::alaya::disk::DiskFlatPublicationOptions {
  ::alaya::disk::DiskFlatPublicationOptions translated;
  translated.collection_root = publication.collection_root;
  translated.segment_id = publication.segment_id;
  translated.segment_generation = publication.segment_generation;
  translated.manifest_generation = publication.manifest_generation;
  translated.publication_parent = publication.publication_parent;
  translated.metadata_epoch = publication.metadata_epoch;
  translated.metadata_checkpoint = publication.metadata_checkpoint;
  translated.wal_cut = publication.wal_cut;
  translated.row_versions = publication.row_versions;
  translated.id_map_checkpoint = publication.id_map_checkpoint;
  translated.collection_features = publication.collection_features;
  translated.abort_policy = publication.abort_policy;
  translated.fail_point = publication.fail_point;
  translated.base_manifest = publication.base_manifest;
  return translated;
}

[[nodiscard]] inline auto build_flat_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult> {
  auto translated = disk_flat_publication_from_collection(publication);
  auto built = build_collection_flat_target(schema, rows, translated, context);
  if (!built.ok()) {
    return built.status();
  }
  auto flat_target = std::move(built).value();
  const auto *registration = find_collection_target_registration(core::algorithm::flat);
  CollectionTargetBuildResult result;
  result.segment = std::move(flat_target.segment);
  result.requested_algorithm = core::algorithm::flat;
  result.built_algorithm = core::algorithm::flat;
  result.implementation_key = registration->implementation_key;
  result.factory_key = registration->factory_key;
  result.artifact_bytes = flat_target.artifact_bytes;
  return result;
}

[[nodiscard]] inline auto build_qg_collection_target(const CollectionSchema &schema,
                                                     std::span<const RegisteredRow> rows,
                                                     const CollectionTargetBuildParams &params,
                                                     const CollectionTargetPublication &publication,
                                                     core::BuildContext &context)
    -> core::Result<CollectionTargetBuildResult> {
  // Retained only as the topology/compatibility producer while memqg remains
  // builder-only. kCollectionTargetRegistrations no longer routes new qg
  // targets here; keeping the legacy feature string is what lets tests mint an
  // old artifact and prove the new reader still accepts it.
  constexpr std::string_view kLegacyImplementationKey{"qg_segment"};
  using Space = ::alaya::RaBitQSpace<>;
  using Segment = ::alaya::QgSegment<Space>;

  auto harvested = harvest_memory_graph_vectors<float>(schema, rows, "QG");
  if (!harvested.ok()) {
    return harvested.status();
  }
  auto vectors = std::move(harvested).value();
  if (schema.metric == core::Metric::cosine) {
    auto status = l2_normalize_float_rows(vectors, schema.dim, core::OperationStage::build);
    if (!status.ok()) {
      return status;
    }
  }
  core::AnySegment erased;
  try {
    const auto count = static_cast<std::uint32_t>(vectors.size() / schema.dim);
    auto space = std::make_shared<Space>(count, schema.dim, schema.metric);
    space->fit(vectors.data(), count);
    typename Segment::BuildInput input(core::TypedTensorView::contiguous(vectors.data(),
                                                                         count,
                                                                         schema.dim),
                                       std::move(space));
    ::alaya::QgBuildOptions options;
    options.ef_build = params.ef_construction;
    options.thread_count = params.thread_count;
    auto built = Segment::build(std::move(input), options, context);
    auto erased_result = Segment::into_any(std::move(built));
    if (!erased_result.ok()) {
      return erased_result.status();
    }
    erased = std::move(erased_result).value();
  } catch (...) {
    return core::status_from_exception(core::OperationStage::build);
  }
  if (schema.metric == core::Metric::cosine) {
    // QgSegment::descriptor() always reports engine_quantized (RaBitQ is the
    // only quantization QG has), so the wrap below relies on
    // make_l2_normalized_query_segment() accepting engine_quantized inner
    // segments, not just none -- see collection_normalized_segment.hpp.
    auto normalized = make_l2_normalized_query_segment(std::move(erased));
    if (!normalized.ok()) {
      return normalized.status();
    }
    erased = std::move(normalized).value();
  }

  const auto *registration = find_collection_target_registration(core::algorithm::qg);
  MemoryCollectionTargetDefinition definition;
  definition.algorithm_id = core::algorithm::qg;
  definition.implementation_key = kLegacyImplementationKey;
  definition.factory_key = registration->factory_key;
  definition.format_version = Segment::kFormatVersion;
  definition.preprocessing = schema.metric == core::Metric::cosine
                                 ? core::MetricPreprocessing::l2_normalized
                                 : core::MetricPreprocessing::engine_quantized;
  definition.entry_extensions.emplace("quantization", "rabitq");
  definition.entry_extensions.emplace("ef_build", std::to_string(params.ef_construction));
  definition.entry_extensions.emplace("thread_count", std::to_string(params.thread_count));
  definition.artifacts = {
      {std::string(Segment::kArtifactName), "qg.bin", true, {}},
  };
  auto published =
      publish_memory_collection_target(erased, schema, publication, std::move(definition), context);
  if (!published.ok()) {
    return published.status();
  }

  CollectionTargetBuildResult result;
  result.segment = std::move(erased);
  result.requested_algorithm = core::algorithm::qg;
  result.built_algorithm = core::algorithm::qg;
  result.implementation_key = kLegacyImplementationKey;
  result.factory_key = registration->factory_key;
  result.artifact_bytes = published.value();
  result.effective_ef_construction = params.ef_construction;
  return result;
}

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
namespace laser_target_detail {

[[nodiscard]] inline auto validate_finite_vectors(std::span<const float> vectors) -> core::Status {
  for (const float value : vectors) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::build,
                                 core::StatusDetail::malformed_struct,
                                 "Collection LASER target rejects non-finite source vectors");
    }
  }
  return core::Status::success();
}

// Streams `vectors` (row-major, `row_count * dim` floats) to
// "{prefix}_pca_base.fbin" in the two-int32-header layout
// QGBuilder::build() requires as input (see qg_builder.hpp's doc comment on
// build(): "The input vector file must be at path:
// {filename}_pca_base.fbin"). "pca_base" is QGBuilder's historical name for
// this slot; the Collection target builder runs no actual PCA step here, it
// just supplies the schema's raw float32 vectors directly (residual_dim=0,
// since the QuantizedGraph below is constructed with main_dim == dim).
inline auto write_pca_base_fbin(const std::string &prefix,
                                std::span<const float> vectors,
                                std::uint32_t row_count,
                                std::uint32_t dim) -> void {
  std::ofstream out(prefix + "_pca_base.fbin", std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("Collection LASER target: cannot open pca_base scratch file: " +
                             prefix);
  }
  const auto n = static_cast<std::int32_t>(row_count);
  const auto d = static_cast<std::int32_t>(dim);
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(&d), sizeof(d));
  out.write(reinterpret_cast<const char *>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  if (!out) {
    throw std::runtime_error("Collection LASER target: failed writing pca_base scratch file: " +
                             prefix);
  }
}

[[nodiscard]] inline auto build_memory_qg_topology(std::span<const float> vectors,
                                                   std::uint32_t count,
                                                   std::uint32_t dim,
                                                   core::Metric metric,
                                                   std::uint32_t max_degree,
                                                   const CollectionTargetBuildParams &params,
                                                   core::BuildContext &context)
    -> ::alaya::FrozenGraphSnapshot {
  using Space = ::alaya::RaBitQSpace<>;
  using Segment = ::alaya::QgSegment<Space>;

  auto space = std::make_shared<Space>(count, dim, metric);
  space->fit(vectors.data(), count);
  typename Segment::BuildInput input(core::TypedTensorView::contiguous(vectors.data(), count, dim),
                                     std::move(space));
  ::alaya::QgBuildOptions options;
  options.ef_build = params.ef_construction;
  options.thread_count = params.thread_count;
  auto segment = Segment::build(std::move(input), options, context);
  auto source = segment->export_graph_snapshot();
  if (source.max_degree() == max_degree) {
    return source;
  }
  ::alaya::FrozenGraphSnapshot::Adjacency adjacency(source.adjacency());
  for (auto &neighbors : adjacency) {
    if (neighbors.size() > max_degree) {
      neighbors.resize(max_degree);
    }
  }
  return ::alaya::FrozenGraphSnapshot(std::move(adjacency), source.entry_point(), max_degree);
}

// RAII scratch directory for the raw native LASER files (Vamana graph +
// QGBuilder's out-of-core .index/_rotator/_cache_ids/_cache_nodes output --
// the LASER packer remains file-oriented, see QGBuilder::build_from_graph()'s
// doc comment) before LaserSegmentImporter copies/links the pieces it wants
// into the final Collection-owned segment directory. Removed on scope exit
// regardless of success or failure -- only the importer's copies survive.
struct ScratchDir {
  std::filesystem::path path;
  explicit ScratchDir(std::filesystem::path scratch_path) : path(std::move(scratch_path)) {
    std::filesystem::create_directories(path);
  }
  ScratchDir(const ScratchDir &) = delete;
  auto operator=(const ScratchDir &) -> ScratchDir & = delete;
  ScratchDir(ScratchDir &&) = delete;
  auto operator=(ScratchDir &&) -> ScratchDir & = delete;
  ~ScratchDir() {
    std::error_code error_code;
    std::filesystem::remove_all(path, error_code);
  }
};

}  // namespace laser_target_detail
#endif

[[nodiscard]] inline auto laser_publication_from_collection(
    const CollectionTargetPublication &publication,
    core::AlgorithmId algorithm_id = core::algorithm::laser,
    std::string_view factory_key = "laser",
    std::string_view implementation_key = "disk_laser_segment",
    bool numeric_score_comparable = false) -> ::alaya::disk::LaserSegmentReferenceOptions {
  ::alaya::disk::LaserSegmentReferenceOptions translated;
  translated.collection_root = publication.collection_root;
  translated.segment_id = publication.segment_id;
  translated.algorithm_id = algorithm_id;
  translated.factory_key = factory_key;
  translated.implementation_key = implementation_key;
  translated.numeric_score_comparable = numeric_score_comparable;
  translated.segment_generation = publication.segment_generation;
  translated.manifest_generation = publication.manifest_generation;
  translated.publication_parent = publication.publication_parent;
  translated.metadata_epoch = publication.metadata_epoch;
  translated.metadata_checkpoint = publication.metadata_checkpoint;
  translated.wal_cut = publication.wal_cut;
  translated.row_versions = publication.row_versions;
  translated.id_map_checkpoint = publication.id_map_checkpoint;
  translated.collection_features = publication.collection_features;
  translated.abort_policy = publication.abort_policy;
  translated.fail_point = publication.fail_point;
  translated.base_manifest = publication.base_manifest;
  return translated;
}

// Builds a native on-disk LASER segment from this Collection's currently
// live rows and registers it into the Collection's manifest-v2 control
// plane. The legacy qg memory builder above remains only as the topology
// producer and old-artifact compatibility fixture; both public qg and direct
// LASER targets arrive here. LASER's on-disk packer (VamanaBuilder ->
// QGBuilder -> LaserSegmentImporter) is inherently file-oriented, so this
// function's shape is: harvest vectors -> build raw native files in a scratch
// dir -> import them into
// "<collection_root>/segments/<segment_id>" (the exact directory
// LaserSegment::publish_reference() requires its open handle to already sit
// at) -> open that directory -> publish_reference() to mint the manifest-v2
// SegmentEntryV2 -> erase to AnySegment.
[[nodiscard]] inline auto build_laser_collection_target_impl(
    core::AlgorithmId exposed_algorithm,
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult> {
#if !(defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0)
  (void)schema;
  (void)rows;
  (void)params;
  (void)publication;
  (void)context;
  return core::Status::error(core::StatusCode::not_supported,
                             core::OperationStage::build,
                             core::StatusDetail::operation_slot_absent,
                             exposed_algorithm == core::algorithm::qg
                                 ? "Collection qg target requires the LASER implementation, which "
                                   "is not supported on "
                                   "this platform/build; Flat fallback is disabled"
                                 : "Collection LASER target builder requires ALAYA_ENABLE_LASER");
#else
  if (exposed_algorithm != core::algorithm::qg && exposed_algorithm != core::algorithm::laser) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::build,
                               core::StatusDetail::malformed_struct,
                               "Collection LASER builder received an invalid exposed algorithm id");
  }
  const bool qg_same_id_swap = exposed_algorithm == core::algorithm::qg;
  // L2 keeps the established Vamana topology path byte-for-byte. IP/cosine
  // build a metric-aware memory-QG topology below and pass its frozen snapshot
  // to the same LASER packer; a future rotate optimization can still thread a
  // predecessor snapshot here to avoid rebuilding either topology.

  auto harvested = harvest_memory_graph_vectors<float>(schema, rows, "LASER");
  if (!harvested.ok()) {
    return harvested.status();
  }
  auto vectors = std::move(harvested).value();
  const auto count = static_cast<std::uint32_t>(vectors.size() / schema.dim);

  auto finite_status = laser_target_detail::validate_finite_vectors(vectors);
  if (!finite_status.ok()) {
    return finite_status;
  }
  if (schema.metric == core::Metric::cosine) {
    auto normalize_status =
        l2_normalize_float_rows(vectors, schema.dim, core::OperationStage::build);
    if (!normalize_status.ok()) {
      return normalize_status;
    }
  }

  try {
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    laser_target_detail::ScratchDir raw_dir(std::filesystem::temp_directory_path() /
                                            ("alayalite-laser-collection-build-" +
                                             std::to_string(::alaya::platform::get_pid()) + "-" +
                                             std::to_string(tick) + "-" + publication.segment_id));
    const std::string raw_prefix = (raw_dir.path / ("dsqg_" + publication.segment_id)).string();

    laser_target_detail::write_pca_base_fbin(raw_prefix, vectors, count, schema.dim);

    alaya::vamana::VamanaBuildParams vamana_params;
    // Vamana is native L2 and honors the requested R. IP/cosine temporarily
    // source topology from memqg, whose physical degree is fixed at 32, so
    // that bridge cannot honestly publish a larger R.
    vamana_params.R =
        schema.metric == core::Metric::l2
            ? params.max_neighbors
            : std::min<std::uint32_t>(params.max_neighbors, ::alaya::RaBitQSpace<>::kDegreeBound);
    vamana_params.L = params.ef_construction;
    vamana_params.alpha = params.alpha;
    vamana_params.num_threads = params.thread_count;
    vamana_params.seed = params.seed;
    const std::string vamana_path = raw_prefix + "_vamana.index";
    std::optional<::alaya::FrozenGraphSnapshot> metric_topology;
    if (schema.metric == core::Metric::l2) {
      alaya::vamana::VamanaBuilder vamana_builder(vectors.data(), count, schema.dim, vamana_params);
      vamana_builder.build();
      alaya::vamana::save_graph(vamana_builder.graph(),
                                vamana_path,
                                vamana_params.R,
                                vamana_builder.medoid());
    } else {
      // VamanaBuilder is intentionally L2-only. Reuse the existing memqg
      // metric-aware topology and hand its finalized graph to the LASER
      // packer instead of feeding negative-IP distances into L2 pruning.
      metric_topology.emplace(laser_target_detail::build_memory_qg_topology(vectors,
                                                                            count,
                                                                            schema.dim,
                                                                            schema.metric,
                                                                            vamana_params.R,
                                                                            params,
                                                                            context));
    }

    const auto native_preprocessing = ::alaya::laser::qg_expected_preprocessing(schema.metric);
    alaya::laser::QuantizedGraph quantized_graph(count,
                                                 vamana_params.R,
                                                 schema.dim,
                                                 schema.dim,
                                                 /*rotator_seed=*/params.seed,
                                                 /*rotator_dump_path=*/{},
                                                 schema.metric,
                                                 native_preprocessing);
    alaya::laser::QGBuilder qg_builder(quantized_graph,
                                       /*ef_build=*/params.ef_construction,
                                       /*num_threads=*/params.thread_count);
    if (metric_topology.has_value()) {
      qg_builder.build_from_graph(*metric_topology, raw_prefix.c_str());
    } else {
      qg_builder.build(vamana_path.c_str(), raw_prefix.c_str());
    }

    // Post-seal row IDs are already dense 0..N-1 in vector order
    // (harvest_memory_graph_vectors() verified this above), and that dense
    // row ID *is* what Collection treats as this segment's SegmentRowId --
    // see LaserSegment::execute_search()'s SegmentRowId(hits[index].label)
    // hit construction. So the identity mapping is the correct labels
    // vector, not an arbitrary choice; it still must be physically written,
    // since the reader side (LaserSegmentSearcher's ids_mmap_) hard-checks
    // the ids file is exactly count*8 bytes.
    std::vector<std::uint64_t> labels(count);
    std::iota(labels.begin(), labels.end(), std::uint64_t{0});

    ::alaya::disk::LaserSegmentImportParams import_params;
    import_params.R = vamana_params.R;
    // Same env var and "empty = legacy" convention as
    // segment_factory.hpp's (now laser_segment.hpp's, see decision 6)
    // laser_residency_request() load-side resolution -- this is the
    // build-side half: what residency hint to persist into the manifest for
    // future opens to read back.
    if (qg_same_id_swap) {
      // qg remains the memory-resident product surface even though its sealed
      // artifact is now LASER. Persist the resident arena choice so reopen is
      // deterministic and does not silently regress to paged service.
      import_params.residency = "resident_arena";
    } else {
      const char *residency_env = std::getenv("ALAYA_LASER_RESIDENCY");
      if (residency_env != nullptr && *residency_env != '\0') {
        import_params.residency = residency_env;
      }
    }

    ::alaya::disk::LaserSegmentImporter importer(schema.dim, schema.metric, import_params);
    const auto seg_dir = publication.collection_root / "segments" / publication.segment_id;
    // LaserSegmentImporter::import_from() requires seg_dir's parent to
    // already exist (it deliberately does not create ancestor directories,
    // only its own atomic tmp-dir + rename at that level) -- unlike the
    // memory targets above, which go through ArtifactControlPlaneTransaction
    // and get "segments/" created for them. This is the first LASER build
    // for a fresh Collection root, so "segments/" may not exist yet.
    std::filesystem::create_directories(seg_dir.parent_path());
    (void)importer.import_from(raw_dir.path, labels.data(), labels.size(), seg_dir);

    core::OpenContext open_context;
    open_context.deadline = context.deadline;
    open_context.cancellation = context.cancellation;
    auto opened =
        ::alaya::disk::LaserSegment::open_directory(seg_dir, core::OpenOptions{}, open_context);
    if (!opened.ok()) {
      return opened.status();
    }
    auto laser_segment = std::move(opened).value();

    core::SegmentStats stats{};
    auto stats_status = laser_segment->stats(stats);
    if (!stats_status.ok()) {
      return stats_status;
    }

    const auto reference =
        qg_same_id_swap ? laser_publication_from_collection(publication,
                                                            core::algorithm::qg,
                                                            "qg",
                                                            kQgLaserImplementationKey,
                                                            /*numeric_score_comparable=*/true)
                        : laser_publication_from_collection(publication);
    auto publish_status = laser_segment->publish_reference(reference, context);
    if (!publish_status.ok()) {
      return publish_status;
    }

    auto erased_result =
        ::alaya::disk::LaserSegment::into_any(std::move(laser_segment), exposed_algorithm);
    if (!erased_result.ok()) {
      return erased_result.status();
    }
    auto erased = std::move(erased_result).value();
    if (schema.metric == core::Metric::cosine) {
      auto normalized = make_l2_normalized_query_segment(std::move(erased));
      if (!normalized.ok()) {
        return normalized.status();
      }
      erased = std::move(normalized).value();
    }

    const auto *registration = find_collection_target_registration(exposed_algorithm);
    CollectionTargetBuildResult result;
    result.segment = std::move(erased);
    result.requested_algorithm = exposed_algorithm;
    result.built_algorithm = exposed_algorithm;
    result.implementation_key = registration->implementation_key;
    result.factory_key = registration->factory_key;
    result.artifact_bytes = stats.resident_bytes;
    result.effective_ef_construction = params.ef_construction;
    return result;
  } catch (...) {
    return core::status_from_exception(core::OperationStage::build);
  }
#endif
}

[[nodiscard]] inline auto build_qg_laser_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult> {
  return build_laser_collection_target_impl(core::algorithm::qg,
                                            schema,
                                            rows,
                                            params,
                                            publication,
                                            context);
}

[[nodiscard]] inline auto build_laser_collection_target(
    const CollectionSchema &schema,
    std::span<const RegisteredRow> rows,
    const CollectionTargetBuildParams &params,
    const CollectionTargetPublication &publication,
    core::BuildContext &context) -> core::Result<CollectionTargetBuildResult> {
  return build_laser_collection_target_impl(core::algorithm::laser,
                                            schema,
                                            rows,
                                            params,
                                            publication,
                                            context);
}

[[nodiscard]] inline auto build_unsupported_collection_target(const CollectionSchema &,
                                                              std::span<const RegisteredRow>,
                                                              const CollectionTargetBuildParams &,
                                                              const CollectionTargetPublication &,
                                                              core::BuildContext &)
    -> core::Result<CollectionTargetBuildResult> {
  return core::Status::error(core::StatusCode::not_supported,
                             core::OperationStage::build,
                             core::StatusDetail::operation_slot_absent,
                             "Collection target builder is not enabled in this rollout phase");
}

[[nodiscard]] inline auto open_flat_collection_target(const std::filesystem::path &root,
                                                      const SegmentEntryV2 &entry,
                                                      const CollectionSchema &schema,
                                                      core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  return open_collection_flat_entry(root, entry, schema.scalar_type, context);
}

[[nodiscard]] inline auto open_unsupported_collection_target(const std::filesystem::path &,
                                                             const SegmentEntryV2 &,
                                                             const CollectionSchema &,
                                                             core::OpenContext &)
    -> core::Result<core::AnySegment> {
  return core::Status::error(core::StatusCode::not_supported,
                             core::OperationStage::open,
                             core::StatusDetail::operation_slot_absent,
                             "Collection target opener is not enabled in this rollout phase");
}

struct PersistedMemoryGraphTarget {
  core::ScalarType stored_scalar_type{core::ScalarType::float32};
  core::MetricPreprocessing preprocessing{core::MetricPreprocessing::none};
  std::string_view quantization{};
};

[[nodiscard]] inline auto persisted_memory_graph_target(const SegmentEntryV2 &entry)
    -> core::Result<PersistedMemoryGraphTarget> {
  const auto scalar = entry.extensions.find("stored_scalar_type");
  const auto preprocessing = entry.extensions.find("preprocessing");
  const auto quantization = entry.extensions.find("quantization");
  if (scalar == entry.extensions.end() || preprocessing == entry.extensions.end() ||
      quantization == entry.extensions.end()) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "memory graph replacement manifest is missing persisted type "
                               "metadata");
  }

  PersistedMemoryGraphTarget target;
  if (scalar->second == "float32") {
    target.stored_scalar_type = core::ScalarType::float32;
  } else if (scalar->second == "int8") {
    target.stored_scalar_type = core::ScalarType::int8;
  } else if (scalar->second == "uint8") {
    target.stored_scalar_type = core::ScalarType::uint8;
  } else {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "memory graph replacement manifest has an unknown stored scalar "
                               "type");
  }
  if (preprocessing->second == "none") {
    target.preprocessing = core::MetricPreprocessing::none;
  } else if (preprocessing->second == "l2_normalized") {
    target.preprocessing = core::MetricPreprocessing::l2_normalized;
  } else if (preprocessing->second == "engine_quantized") {
    target.preprocessing = core::MetricPreprocessing::engine_quantized;
  } else {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "memory graph replacement manifest has unknown preprocessing");
  }
  if (quantization->second != "none" && quantization->second != "sq8" &&
      quantization->second != "sq4") {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "memory graph replacement manifest has unknown quantization");
  }
  target.quantization = quantization->second;
  return target;
}

template <typename Segment>
[[nodiscard]] inline auto open_typed_memory_graph_segment(const std::filesystem::path &root,
                                                          const SegmentEntryV2 &entry,
                                                          core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  constexpr bool kQuantized = !std::is_same_v<typename Segment::SearchSpaceTypeAlias,
                                              typename Segment::BuildSpaceTypeAlias>;
  std::array<std::string, 3> paths{};
  for (const auto &artifact : entry.artifacts) {
    if (artifact.logical_name == Segment::kGraphArtifactName) {
      paths[0] = (root / artifact.relative_path).string();
    } else if (artifact.logical_name == Segment::kDataArtifactName) {
      paths[1] = (root / artifact.relative_path).string();
    } else if (artifact.logical_name == Segment::kQuantArtifactName) {
      paths[2] = (root / artifact.relative_path).string();
    }
  }
  if (paths[0].empty() || paths[1].empty() || (kQuantized && paths[2].empty()) ||
      (!kQuantized && !paths[2].empty())) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "memory graph replacement manifest has incompatible native "
                               "artifacts");
  }
  const std::array<core::ArtifactLocation, 3> locations{
      core::ArtifactLocation(Segment::kGraphArtifactName, paths[0]),
      core::ArtifactLocation(Segment::kDataArtifactName, paths[1]),
      core::ArtifactLocation(Segment::kQuantArtifactName, paths[2]),
  };
  const auto artifact_count = kQuantized ? std::size_t{3} : std::size_t{2};
  try {
    auto opened =
        Segment::open(core::ArtifactView(std::span<const core::ArtifactLocation>(locations.data(),
                                                                                 artifact_count)),
                      core::OpenOptions{},
                      context);
    return Segment::into_any(std::move(opened));
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}

[[nodiscard]] inline auto open_memory_qg_collection_target(const std::filesystem::path &root,
                                                           const SegmentEntryV2 &entry,
                                                           const CollectionSchema &schema,
                                                           core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  using Space = ::alaya::RaBitQSpace<>;
  using Segment = ::alaya::QgSegment<Space>;

  if (entry.algorithm_id != core::algorithm::qg ||
      entry.format_version != Segment::kFormatVersion ||
      entry.lifecycle == SegmentLifecycleV2::retired ||
      schema.scalar_type != core::ScalarType::float32 ||
      (schema.metric != core::Metric::l2 && schema.metric != core::Metric::inner_product &&
       schema.metric != core::Metric::cosine)) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               "Collection QG opener received an incompatible segment entry");
  }

  const auto scalar = entry.extensions.find("stored_scalar_type");
  const auto preprocessing = entry.extensions.find("preprocessing");
  const auto quantization = entry.extensions.find("quantization");
  // QG always RaBitQ-quantizes internally, so only cosine (external l2
  // normalization on top of that internal quantization) needs to diverge
  // from the plain engine_quantized tag -- mirrors
  // build_qg_collection_target()'s write side.
  const auto expected_preprocessing =
      schema.metric == core::Metric::cosine ? "l2_normalized" : "engine_quantized";
  if (scalar == entry.extensions.end() || scalar->second != "float32" ||
      preprocessing == entry.extensions.end() || preprocessing->second != expected_preprocessing ||
      quantization == entry.extensions.end() || quantization->second != "rabitq") {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "QG replacement manifest has incompatible persisted type metadata");
  }

  std::string artifact_path;
  for (const auto &artifact : entry.artifacts) {
    if (artifact.logical_name == Segment::kArtifactName) {
      artifact_path = (root / artifact.relative_path).string();
    }
  }
  if (artifact_path.empty()) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "QG replacement manifest is missing its native artifact");
  }

  const std::array<core::ArtifactLocation, 1> locations{
      core::ArtifactLocation(Segment::kArtifactName, artifact_path),
  };
  try {
    auto opened = Segment::open(core::ArtifactView(locations), core::OpenOptions{}, context);
    const auto descriptor = opened->descriptor();
    if (descriptor.dim != schema.dim || descriptor.metric != schema.metric ||
        descriptor.stored_scalar_type != schema.scalar_type ||
        descriptor.preprocessing != core::MetricPreprocessing::engine_quantized) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "QG replacement descriptor disagrees with the Collection schema");
    }
    auto erased_result = Segment::into_any(std::move(opened));
    if (!erased_result.ok() || schema.metric != core::Metric::cosine) {
      return erased_result;
    }
    return make_l2_normalized_query_segment(std::move(erased_result).value());
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}

// Manifest-driven reopen: CollectionSegmentFactory::open_entry() (see
// collection_segment_factory.hpp) already resolved `entry.factory_key ==
// "laser"` to this registration and confirmed `entry.algorithm_id ==
// registration.algorithm_id` before calling here, so this function only
// needs the LASER-specific schema/format checks plus the actual open call.
// LaserSegment::open_collection() re-derives the native artifact set from
// the *collection*-level manifest-v2 entry (not a second manifest.txt read
// off some cached path), matching how open_qg_collection_target() above
// always re-resolves from `entry` too.
[[nodiscard]] inline auto open_laser_collection_target_impl(core::AlgorithmId exposed_algorithm,
                                                            std::string_view required_feature,
                                                            const std::filesystem::path &root,
                                                            const SegmentEntryV2 &entry,
                                                            const CollectionSchema &schema,
                                                            core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  const auto feature =
      std::ranges::find(entry.reader_compatibility.required_features, required_feature);
  if (entry.algorithm_id != exposed_algorithm ||
      entry.format_version != ::alaya::disk::LaserSegment::kFormatVersion ||
      feature == entry.reader_compatibility.required_features.end() ||
      entry.lifecycle == SegmentLifecycleV2::retired ||
      schema.scalar_type != core::ScalarType::float32 ||
      (schema.metric != core::Metric::l2 && schema.metric != core::Metric::inner_product &&
       schema.metric != core::Metric::cosine)) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::open,
                               core::StatusDetail::operation_slot_absent,
                               "Collection LASER opener received an incompatible segment entry");
  }
  auto opened = ::alaya::disk::LaserSegment::open_collection(root,
                                                             entry.segment_id,
                                                             core::OpenOptions{},
                                                             context,
                                                             {},
                                                             exposed_algorithm);
  if (!opened.ok()) {
    return opened.status();
  }
  const auto descriptor = opened.value()->descriptor();
  if (descriptor.dim != schema.dim || descriptor.metric != schema.metric ||
      descriptor.stored_scalar_type != schema.scalar_type ||
      descriptor.preprocessing != core::MetricPreprocessing::none) {
    return core::Status::error(core::StatusCode::corruption,
                               core::OperationStage::open,
                               core::StatusDetail::malformed_struct,
                               "LASER replacement descriptor disagrees with the Collection schema");
  }
  if (schema.metric != core::Metric::l2) {
    const auto preprocessing = entry.extensions.find("preprocessing");
    const auto expected = schema.metric == core::Metric::cosine ? "l2_normalized" : "none";
    if (preprocessing == entry.extensions.end() || preprocessing->second != expected) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "LASER replacement manifest lacks its preprocessing proof");
    }
  }
  auto erased = ::alaya::disk::LaserSegment::into_any(std::move(opened).value(), exposed_algorithm);
  if (!erased.ok() || schema.metric != core::Metric::cosine) {
    return erased;
  }
  return make_l2_normalized_query_segment(std::move(erased).value());
}

[[nodiscard]] inline auto open_laser_collection_target(const std::filesystem::path &root,
                                                       const SegmentEntryV2 &entry,
                                                       const CollectionSchema &schema,
                                                       core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  return open_laser_collection_target_impl(core::algorithm::laser,
                                           "disk_laser_segment",
                                           root,
                                           entry,
                                           schema,
                                           context);
}

[[nodiscard]] inline auto open_qg_collection_target(const std::filesystem::path &root,
                                                    const SegmentEntryV2 &entry,
                                                    const CollectionSchema &schema,
                                                    core::OpenContext &context)
    -> core::Result<core::AnySegment> {
  const auto has_feature = [&](std::string_view feature) {
    return std::ranges::find(entry.reader_compatibility.required_features, feature) !=
           entry.reader_compatibility.required_features.end();
  };
  if (has_feature(kQgLaserImplementationKey)) {
    return open_laser_collection_target_impl(core::algorithm::qg,
                                             kQgLaserImplementationKey,
                                             root,
                                             entry,
                                             schema,
                                             context);
  }
  if (has_feature("qg_segment")) {
    return open_memory_qg_collection_target(root, entry, schema, context);
  }
  return core::Status::error(core::StatusCode::not_supported,
                             core::OperationStage::open,
                             core::StatusDetail::operation_slot_absent,
                             "Collection qg artifact requires an unknown implementation feature");
}

[[nodiscard]] inline auto build_collection_target(core::AlgorithmId requested_algorithm,
                                                  const CollectionSchema &schema,
                                                  std::span<const RegisteredRow> rows,
                                                  const CollectionTargetBuildParams &params,
                                                  const CollectionTargetPublication &publication,
                                                  core::BuildContext &context)
    -> core::Result<CollectionTargetBuildResult> {
  const auto *registration = find_collection_target_registration(requested_algorithm);
  if (registration == nullptr) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::build,
                               core::StatusDetail::operation_slot_absent,
                               "Collection target algorithm has no registered builder");
  }
  return registration->build(schema, rows, params, publication, context);
}

}  // namespace alaya::internal::collection::detail
