// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/graph/frozen_graph_snapshot.hpp"
#include "index/graph/qg/detail/qg_builder_kernel.hpp"
#include "index/graph/qg/detail/qg_graph.hpp"
#include "index/graph/qg/qg_search_extension.hpp"
#include "space/rabitq_space.hpp"
#include "space/space_concepts.hpp"

namespace alaya {

namespace detail {
template <typename SpaceType>
struct QgSegmentBridge;
}

struct QgBuildOptions {
  core::VersionedStructHeader header{56, core::kContractAbiVersion};
  std::uint32_t ef_build{400};
  std::uint32_t thread_count{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t reserved_options[2]{};
  std::uint64_t reserved[4]{};
};

static_assert(sizeof(QgBuildOptions) == 56,
              "same-toolchain layout regression canary for QG build options");

template <typename SpaceType>
  requires Space<SpaceType> && is_rabitq_space_v<SpaceType>
class QgSegment {
 public:
  using SearchSpaceTypeAlias = SpaceType;
  using BuildSpaceTypeAlias = SpaceType;
  using DataType = typename SpaceType::DataTypeAlias;
  using DistanceType = typename SpaceType::DistanceTypeAlias;
  using IDType = typename SpaceType::IDTypeAlias;
  using BuildOptions = QgBuildOptions;
  using SearchExtension = QgSearchExtension;
  using GraphType = detail::QgGraph<SpaceType>;
  using SpaceViewType = detail::QgSegmentSpaceView<SpaceType>;
  using SearchJobType = GraphSearchJob<SpaceViewType>;
  using Gate5MemoryGraphSegmentTag = void;
  using QgSegmentTag = void;

  static_assert(std::is_same_v<DataType, float>, "memory QG supports float32 vectors only");
  static_assert(std::is_same_v<DistanceType, float>, "memory QG uses float32 distances");
  static_assert(std::is_same_v<IDType, std::uint32_t>, "memory QG uses uint32 node IDs");

  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::qg;
  static constexpr std::string_view kArtifactName = "qg";
  static constexpr std::string_view kLegacyArtifactName = "quant";

  struct BuildInput {
    core::VersionedStructHeader header{};
    core::TypedTensorView vectors{};
    std::shared_ptr<SpaceType> space;
    std::uint64_t reserved[3]{};

    BuildInput(core::TypedTensorView vector_view, std::shared_ptr<SpaceType> distance_space)
        : header(core::current_struct_header<BuildInput>()),
          vectors(vector_view),
          space(std::move(distance_space)) {}
  };

  static auto build(BuildInput input, const BuildOptions &options, core::BuildContext &context)
      -> std::unique_ptr<QgSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::build);
    validate_build_input(input);
    validate_build_options(options);
    validate_build_budget(input, context);

    auto graph = std::make_shared<GraphType>(input.space);
    auto space_view = std::make_shared<SpaceViewType>(*input.space, *graph);
    detail::QgBuilderKernel<SpaceViewType> builder(space_view, options.thread_count);
    builder.set_ef_build(options.ef_build);
    builder.build_graph();
    validate_loaded_graph(*input.space, *graph);
    return std::unique_ptr<QgSegment>(
        new QgSegment(std::move(input.space), std::move(graph), std::move(space_view)));
  }

  static auto open(core::ArtifactView artifact,
                   const core::OpenOptions &,
                   core::OpenContext &context) -> std::unique_ptr<QgSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::open);
    const auto artifact_path = required_input_artifact(artifact);
    validate_open_budget(artifact_path, context);
    auto space = std::make_shared<SpaceType>();
    space->load(artifact_path);
    auto graph = std::make_shared<GraphType>(space, space->get_ep());
    auto space_view = std::make_shared<SpaceViewType>(*space, *graph);
    validate_loaded_graph(*space, *graph);
    return std::unique_ptr<QgSegment>(
        new QgSegment(std::move(space), std::move(graph), std::move(space_view)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = space_->get_dim();
    descriptor.metric = space_->metric();
    descriptor.stored_scalar_type = core::scalar_type_for<DataType>;
    descriptor.medium = core::Medium::memory;
    descriptor.preprocessing = core::MetricPreprocessing::engine_quantized;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  // Export the finalized Segment-owned topology. QG stores exactly
  // kDegreeBound edge slots per node; sanitize those slots at the format
  // boundary so every downstream seal consumer receives a simple graph.
  [[nodiscard]] auto export_graph_snapshot() const -> FrozenGraphSnapshot {
    const auto num_points = static_cast<std::size_t>(space_->get_data_num());
    FrozenGraphSnapshot::Adjacency adjacency(num_points);
    for (std::size_t node = 0; node < num_points; ++node) {
      auto &neighbors = adjacency[node];
      neighbors.reserve(GraphType::kDegreeBound);
      const auto *edges = graph_->get_edges(static_cast<IDType>(node));
      for (std::size_t edge = 0; edge < GraphType::kDegreeBound; ++edge) {
        const IDType neighbor = edges[edge];
        if (static_cast<std::size_t>(neighbor) == node ||
            std::find(neighbors.begin(), neighbors.end(), neighbor) != neighbors.end()) {
          continue;
        }
        neighbors.push_back(neighbor);
      }
    }

    FrozenGraphSnapshot snapshot(std::move(adjacency),
                                 graph_->get_ep(),
                                 static_cast<std::uint32_t>(GraphType::kDegreeBound));
    snapshot.validate();
    return snapshot;
  }

  [[nodiscard]] static auto make_search_extension(const SearchExtension &options)
      -> core::AlgorithmSearchExtension {
    return make_qg_search_extension(options);
  }

  auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "QG single search requires exactly one query row");
    }
    return execute_search(request);
  }

  auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  auto save(core::ArtifactWriter &writer,
            const core::SaveOptions &,
            core::ArtifactManifest &manifest) const -> core::Status {
    const auto artifact_path = required_output_artifact(writer);
    // QgSegment owns the authoritative entry point. Refresh the retained v1
    // codec mirror before serialization in case an internal legacy view was
    // used while the Segment was alive.
    space_->set_ep(graph_->get_ep());
    space_->save(artifact_path);
    artifact_ = core::Artifact(kArtifactName,
                               static_cast<std::uint64_t>(std::filesystem::file_size(
                                   std::filesystem::path(artifact_path))),
                               0);
    manifest = core::ArtifactManifest{};
    manifest.schema_version = 1;
    manifest.format_version = kFormatVersion;
    manifest.algorithm_id = kAlgorithmId;
    manifest.artifacts = std::span<const core::Artifact>(std::addressof(artifact_), 1);
    return core::Status::success();
  }

  auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = space_->get_data_num();
    stats.allocated_rows = space_->get_capacity();
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  [[nodiscard]] static auto into_any(std::unique_ptr<QgSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null QG segment");
    }
    auto shared = std::shared_ptr<QgSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = false;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 private:
  friend struct detail::QgSegmentBridge<SpaceType>;

  struct SearchScratch {
    std::vector<IDType> ids;
  };

  explicit QgSegment(std::shared_ptr<SpaceType> space,
                     std::shared_ptr<GraphType> graph,
                     std::shared_ptr<SpaceViewType> space_view)
      : space_(std::move(space)),
        graph_(std::move(graph)),
        space_view_(std::move(space_view)),
        search_job_(std::make_shared<SearchJobType>(space_view_, nullptr, nullptr, space_view_)) {}

  static void validate_build_input(const BuildInput &input) {
    if (!core::is_current_struct(input)) {
      throw std::invalid_argument("QG build input has an incompatible size or ABI version");
    }
    if (input.space == nullptr) {
      throw std::invalid_argument("QG build requires a RaBitQ space");
    }
    const auto tensor_status =
        core::validate_tensor(input.vectors, input.space->get_dim(), core::OperationStage::build);
    if (!tensor_status.ok()) {
      throw std::invalid_argument(tensor_status.diagnostic());
    }
    if (input.vectors.scalar_type != core::scalar_type_for<DataType>) {
      throw std::invalid_argument("QG build tensor scalar type does not match the space");
    }
    if (input.vectors.rows != input.space->get_data_num()) {
      throw std::invalid_argument("QG build tensor row count does not match the space");
    }
    if (input.space->get_data_num() <= SpaceType::kDegreeBound) {
      throw std::invalid_argument("QG build requires more vectors than its fixed degree bound");
    }
  }

  static void validate_build_options(const BuildOptions &options) {
    if (!core::is_current_struct(options)) {
      throw std::invalid_argument("QG build options have an incompatible size or ABI version");
    }
    if (options.ef_build == 0) {
      throw std::invalid_argument("QG ef_build must be at least 1");
    }
    if (options.thread_count == 0) {
      throw std::invalid_argument("QG thread_count must be at least 1");
    }
  }

  static void validate_loaded_graph(SpaceType &space, const GraphType &graph) {
    const auto rows = static_cast<core::RowCount>(space.get_data_num());
    if (rows <= SpaceType::kDegreeBound || space.get_dim() == 0 ||
        static_cast<core::RowCount>(graph.get_ep()) >= rows) {
      throw std::runtime_error("Invalid QG artifact");
    }
    for (core::RowCount row = 0; row < rows; ++row) {
      const auto *edges = graph.get_edges(static_cast<IDType>(row));
      for (std::size_t edge = 0; edge < SpaceType::kDegreeBound; ++edge) {
        if (static_cast<core::RowCount>(edges[edge]) >= rows) {
          throw std::runtime_error("Invalid QG artifact neighbor ID");
        }
      }
    }
  }

  static void validate_context(const core::Deadline &deadline,
                               const core::CancellationToken &cancellation,
                               core::OperationStage stage) {
    const auto status = core::validate_runtime_control(deadline, cancellation, stage);
    if (!status.ok()) {
      throw std::runtime_error(status.diagnostic());
    }
  }

  static void validate_build_budget(const BuildInput &input, core::BuildContext &context) {
    std::uint64_t elements{};
    std::uint64_t bytes{};
    if (!core::checked_multiply(input.space->get_data_num(), input.space->get_dim(), elements) ||
        !core::checked_multiply(elements, sizeof(DataType), bytes)) {
      throw std::invalid_argument("QG build input byte size overflows uint64");
    }
    const auto budget = context.growing_reservation.ensure(bytes,
                                                           core::OperationStage::build,
                                                           "QG build reservation is too small");
    if (!budget.ok()) {
      throw std::runtime_error("QG resource_exhausted: build reservation is too small");
    }
  }

  static void validate_open_budget(std::string_view path, const core::OpenContext &context) {
    const auto bytes = std::filesystem::file_size(std::filesystem::path(path));
    const auto budget = core::require_lease(context.resident_lease,
                                            bytes,
                                            core::OperationStage::open,
                                            "QG resident lease is too small");
    if (!budget.ok()) {
      throw std::runtime_error("QG resource_exhausted: resident lease is too small");
    }
  }

  static auto required_input_artifact(const core::ArtifactView &artifact) -> std::string_view {
    auto path = artifact.find(kArtifactName);
    if (path.empty()) {
      path = artifact.find(kLegacyArtifactName);
    }
    if (path.empty()) {
      throw std::invalid_argument("QG required artifact is missing: qg");
    }
    return path;
  }

  static auto required_output_artifact(const core::ArtifactWriter &writer) -> std::string_view {
    auto path = writer.find(kArtifactName);
    if (path.empty()) {
      path = writer.find(kLegacyArtifactName);
    }
    if (path.empty()) {
      throw std::invalid_argument("QG output artifact is missing: qg");
    }
    return path;
  }

  auto validate_search_request(const core::SearchRequest &request) const -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "QG search request is incomplete or incompatible");
    }
    auto status =
        core::validate_tensor(request.queries, space_->get_dim(), core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::scalar_type_for<DataType>) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "QG does not implicitly convert query tensor scalar types");
    }
    status = core::validate_response(*request.response,
                                     request.queries.rows,
                                     request.options.top_k,
                                     core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.filter.kind != core::SegmentFilterKind::none) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::operation_slot_absent,
                                 "QG does not support a compiled filter view");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "QG top_k exceeds uint32");
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }
    std::uint64_t scratch_bytes{};
    if (!core::checked_multiply(request.options.top_k, sizeof(IDType), scratch_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "QG query scratch size overflows uint64");
    }
    return core::require_lease(request.context->query_scratch_lease,
                               scratch_bytes,
                               core::OperationStage::search,
                               "QG query scratch lease is too small");
  }

  static auto resolve_effort(const core::SearchOptions &options) -> core::Result<std::uint32_t> {
    std::uint64_t effort = 100;
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     "QG received an extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr || extension.payload_size < sizeof(SearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "QG search extension payload is truncated");
      }
      const auto &typed = *static_cast<const SearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "QG search extension has an incompatible version");
      }
      effort = typed.effort;
    }
    if (effort < options.top_k || effort > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "QG effort must be in [top_k, UINT32_MAX]");
    }
    return static_cast<std::uint32_t>(effort);
  }

  auto execute_search(const core::SearchRequest &request) const -> core::Status {
    auto status = validate_search_request(request);
    if (!status.ok()) {
      return status;
    }
    auto &response = *request.response;
    response.score_kind = core::ScoreKind::distance;
    response.comparable_metric = space_->metric();
    response.result_flags = core::ResultFlag::approximate;
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      core::initialize_empty_response(response,
                                      request.queries.rows,
                                      request.options.top_k == 0
                                          ? core::SearchCompleteness::complete_k
                                          : core::SearchCompleteness::eligible_exhausted);
      return core::Status::success();
    }
    auto effort_result = resolve_effort(request.options);
    if (!effort_result.ok()) {
      return effort_result.status();
    }
    const auto effort = std::move(effort_result).value();
    const auto top_k = static_cast<std::uint32_t>(request.options.top_k);
    auto &scratch = search_scratch();
    core::RowCount cursor = 0;
    response.query_count = request.queries.rows;
    response.offsets[0] = 0;
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      try {
        scratch.ids.assign(top_k, std::numeric_limits<IDType>::max());
        const auto *query = request.queries.template row<DataType>(row);
        search_job_->rabitq_search_solo(query, top_k, scratch.ids.data(), effort);
        core::RowCount written = 0;
        for (; written < top_k && scratch.ids[written] != std::numeric_limits<IDType>::max();
             ++written) {
          const auto id = scratch.ids[written];
          const auto score = static_cast<float>(
              space_->get_dist_func()(query, space_->get_data_by_id(id), space_->get_dim()));
          if (std::isnan(score)) {
            throw std::runtime_error("QG produced a NaN numeric score");
          }
          response.hits[cursor + written] =
              core::SearchHit(core::SegmentRowId(static_cast<std::uint64_t>(id)),
                              score,
                              core::ScoreKind::distance,
                              space_->metric(),
                              core::ResultFlag::approximate);
        }
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        if (written == top_k) {
          response.completeness[row] = core::SearchCompleteness::complete_k;
        } else if (space_->get_data_num() < top_k && written == space_->get_data_num()) {
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
    return core::Status::success();
  }

  static auto search_scratch() -> SearchScratch & {
    static thread_local SearchScratch scratch;
    return scratch;
  }

  std::shared_ptr<SpaceType> space_;
  std::shared_ptr<GraphType> graph_;
  std::shared_ptr<SpaceViewType> space_view_;
  std::shared_ptr<SearchJobType> search_job_;
  mutable core::Artifact artifact_{};
};

static_assert(core::Searchable<QgSegment<RaBitQSpace<>>>);
static_assert(core::BatchSearchable<QgSegment<RaBitQSpace<>>>);
static_assert(core::Saveable<QgSegment<RaBitQSpace<>>>);
static_assert(core::StatsProvider<QgSegment<RaBitQSpace<>>>);
static_assert(!core::Mutable<QgSegment<RaBitQSpace<>>>);

}  // namespace alaya
