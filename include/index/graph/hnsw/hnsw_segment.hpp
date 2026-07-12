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
#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/hnsw/detail/hnsw_builder_kernel.hpp"
#include "space/raw_space.hpp"
#include "space/space_concepts.hpp"

namespace alaya {

namespace detail {
template <typename SearchSpaceType, typename BuildSpaceType>
struct HnswSegmentBridge;
}

struct HnswBuildOptions {
  core::VersionedStructHeader header{56, core::kContractAbiVersion};
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{200};
  std::uint32_t thread_count{1};
  std::uint32_t reserved_options{};
  std::uint64_t reserved[4]{};
};

static_assert(sizeof(HnswBuildOptions) == 56,
              "same-toolchain layout regression canary for HNSW build options");

struct HnswSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  HnswSearchExtension() : header(core::current_struct_header<HnswSearchExtension>()) {}
};

[[nodiscard]] inline auto make_hnsw_search_extension(const HnswSearchExtension &options)
    -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::hnsw;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

template <typename SearchSpaceType, typename BuildSpaceType = SearchSpaceType>
  requires Space<SearchSpaceType> && Space<BuildSpaceType>
class HnswSegment {
 public:
  using SearchSpaceTypeAlias = SearchSpaceType;
  using BuildSpaceTypeAlias = BuildSpaceType;
  using DataType = typename SearchSpaceType::DataTypeAlias;
  using DistanceType = typename SearchSpaceType::DistanceTypeAlias;
  using IDType = typename SearchSpaceType::IDTypeAlias;
  using GraphType =
      Graph<typename BuildSpaceType::DataTypeAlias, typename BuildSpaceType::IDTypeAlias>;

  static_assert(std::is_same_v<DataType, typename BuildSpaceType::DataTypeAlias>,
                "HNSW search/build spaces must use the same vector scalar type");
  static_assert(std::is_same_v<IDType, typename BuildSpaceType::IDTypeAlias>,
                "HNSW search/build spaces must use the same internal node ID type");
  static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, std::int8_t> ||
                    std::is_same_v<DataType, std::uint8_t>,
                "contract v3 supports float32, int8, and uint8 tensors");

  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr std::string_view kGraphArtifactName = "graph";
  static constexpr std::string_view kDataArtifactName = "data";
  static constexpr std::string_view kQuantArtifactName = "quant";

  struct BuildInput {
    core::VersionedStructHeader header{};
    core::TypedTensorView vectors{};
    std::shared_ptr<SearchSpaceType> search_space;
    std::shared_ptr<BuildSpaceType> build_space;
    std::uint64_t reserved[3]{};

    BuildInput(core::TypedTensorView vector_view,
               std::shared_ptr<SearchSpaceType> search,
               std::shared_ptr<BuildSpaceType> build)
        : header(core::current_struct_header<BuildInput>()),
          vectors(vector_view),
          search_space(std::move(search)),
          build_space(std::move(build)) {}
  };

  static auto build(BuildInput input, const HnswBuildOptions &options, core::BuildContext &context)
      -> std::unique_ptr<HnswSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::build);
    validate_spaces(input);
    validate_build_options(options);
    validate_build_budget(input, context);
    detail::HnswBuilderKernel<BuildSpaceType> builder(input.build_space,
                                                      options.max_neighbors,
                                                      options.ef_construction);
    auto graph = std::shared_ptr<GraphType>(builder.build_graph(options.thread_count).release());
    return std::unique_ptr<HnswSegment>(new HnswSegment(std::move(input.search_space),
                                                        std::move(input.build_space),
                                                        std::move(graph)));
  }

  static auto open(core::ArtifactView artifact,
                   const core::OpenOptions &,
                   core::OpenContext &context) -> std::unique_ptr<HnswSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::open);
    const auto graph_path = required_artifact(artifact, kGraphArtifactName);
    const auto data_path = required_artifact(artifact, kDataArtifactName);
    validate_open_budget(graph_path, data_path, context);
    auto build_space = std::make_shared<BuildSpaceType>();
    build_space->load(data_path);
    build_space->set_metric_function();
    std::shared_ptr<SearchSpaceType> search_space;
    if constexpr (std::is_same_v<SearchSpaceType, BuildSpaceType>) {
      search_space = build_space;
    } else {
      const auto quant_path = required_artifact(artifact, kQuantArtifactName);
      search_space = std::make_shared<SearchSpaceType>();
      search_space->load(quant_path);
      search_space->set_metric_function();
    }
    validate_loaded_spaces(search_space, build_space);
    auto graph = std::make_shared<GraphType>();
    graph->load(graph_path);
    validate_graph(*graph, build_space->get_data_num());
    return std::unique_ptr<HnswSegment>(
        new HnswSegment(std::move(search_space), std::move(build_space), std::move(graph)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = core::algorithm::hnsw;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = search_space_->get_dim();
    descriptor.metric = search_space_->metric();
    descriptor.stored_scalar_type = core::scalar_type_for<DataType>;
    descriptor.medium = core::Medium::memory;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = core::algorithm::hnsw;
    return descriptor;
  }

  auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "HNSW single search requires exactly one query row");
    }
    return execute_search(request);
  }

  auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  auto save(core::ArtifactWriter &writer,
            const core::SaveOptions &,
            core::ArtifactManifest &manifest) const -> core::Status {
    const auto graph_path = required_artifact(writer, kGraphArtifactName);
    const auto data_path = required_artifact(writer, kDataArtifactName);
    graph_->save(graph_path);
    build_space_->save(data_path);
    std::string_view quant_path;
    if constexpr (!std::is_same_v<SearchSpaceType, BuildSpaceType>) {
      quant_path = required_artifact(writer, kQuantArtifactName);
      search_space_->save(quant_path);
    }
    refresh_artifacts(graph_path, data_path, quant_path);
    manifest = core::ArtifactManifest{};
    manifest.schema_version = 1;
    manifest.format_version = kFormatVersion;
    manifest.algorithm_id = core::algorithm::hnsw;
    manifest.artifacts = std::span<const core::Artifact>(artifacts_.data(), artifact_count_);
    return core::Status::success();
  }

  auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = search_space_->get_data_num();
    stats.allocated_rows = search_space_->get_capacity();
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  [[nodiscard]] static auto into_any(std::unique_ptr<HnswSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null HNSW segment");
    }
    auto shared = std::shared_ptr<HnswSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = false;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 private:
  friend struct detail::HnswSegmentBridge<SearchSpaceType, BuildSpaceType>;

  struct SearchScratch {
    std::vector<IDType> ids;
    std::vector<DistanceType> distances;
  };

  HnswSegment(std::shared_ptr<SearchSpaceType> search_space,
              std::shared_ptr<BuildSpaceType> build_space,
              std::shared_ptr<GraphType> graph)
      : search_space_(std::move(search_space)),
        build_space_(std::move(build_space)),
        graph_(std::move(graph)),
        search_job_(
            std::make_shared<GraphSearchJob<SearchSpaceType, BuildSpaceType>>(search_space_,
                                                                              graph_,
                                                                              nullptr,
                                                                              build_space_)) {}

  static void validate_spaces(const BuildInput &input) {
    if (!core::is_current_struct(input)) {
      throw std::invalid_argument("HNSW build input has an incompatible size or ABI version");
    }
    if (input.search_space == nullptr || input.build_space == nullptr) {
      throw std::invalid_argument("HNSW build requires search and build spaces");
    }
    const auto tensor_status = core::validate_tensor(input.vectors,
                                                     input.build_space->get_dim(),
                                                     core::OperationStage::build);
    if (!tensor_status.ok()) {
      throw std::invalid_argument(tensor_status.diagnostic());
    }
    if (input.vectors.scalar_type != core::scalar_type_for<DataType>) {
      throw std::invalid_argument("HNSW build tensor scalar type does not match the spaces");
    }
    if (input.vectors.rows != input.build_space->get_data_num()) {
      throw std::invalid_argument("HNSW build tensor row count does not match the spaces");
    }
    validate_loaded_spaces(input.search_space, input.build_space);
  }

  static void validate_loaded_spaces(const std::shared_ptr<SearchSpaceType> &search_space,
                                     const std::shared_ptr<BuildSpaceType> &build_space) {
    if (search_space == nullptr || build_space == nullptr) {
      throw std::invalid_argument("HNSW build requires search and build spaces");
    }
    if (search_space->get_dim() != build_space->get_dim()) {
      throw std::invalid_argument("HNSW search/build space dimension mismatch");
    }
    if (search_space->get_data_num() != build_space->get_data_num()) {
      throw std::invalid_argument("HNSW search/build space row-count mismatch");
    }
    if (build_space->get_data_num() == 0) {
      throw std::invalid_argument("HNSW requires at least one vector");
    }
  }

  static void validate_build_options(const HnswBuildOptions &options) {
    if (!core::is_current_struct(options)) {
      throw std::invalid_argument("HNSW build options have an incompatible size or ABI version");
    }
    if (options.max_neighbors < 2) {
      throw std::invalid_argument("HNSW max_neighbors must be at least 2");
    }
    if (options.ef_construction == 0 ||
        options.ef_construction > std::numeric_limits<std::uint16_t>::max()) {
      throw std::invalid_argument("HNSW ef_construction must be in [1, 65535]");
    }
  }

  static void validate_graph(const GraphType &graph, core::RowCount rows) {
    if (graph.max_nodes_ < rows || graph.max_nbrs_ == 0 || graph.overlay_graph_ == nullptr ||
        graph.overlay_graph_->node_num_ != graph.max_nodes_ ||
        graph.overlay_graph_->max_nbrs_ != graph.max_nbrs_) {
      throw std::runtime_error("Invalid HNSW graph artifact");
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
    if (!core::checked_multiply(input.build_space->get_data_num(),
                                input.build_space->get_dim(),
                                elements) ||
        !core::checked_multiply(elements, sizeof(DataType), bytes)) {
      throw std::invalid_argument("HNSW build input byte size overflows uint64");
    }
    const auto budget = context.growing_reservation.ensure(bytes,
                                                           core::OperationStage::build,
                                                           "HNSW build reservation is too small");
    if (!budget.ok()) {
      throw std::runtime_error("resource_exhausted: HNSW build reservation is too small");
    }
  }

  static void validate_open_budget(std::string_view graph_path,
                                   std::string_view data_path,
                                   const core::OpenContext &context) {
    const auto graph_bytes = std::filesystem::file_size(std::filesystem::path(graph_path));
    const auto data_bytes = std::filesystem::file_size(std::filesystem::path(data_path));
    std::uint64_t bytes{};
    if (!core::checked_add(graph_bytes, data_bytes, bytes)) {
      throw std::invalid_argument("HNSW artifact byte size overflows uint64");
    }
    const auto budget = core::require_lease(context.resident_lease,
                                            bytes,
                                            core::OperationStage::open,
                                            "HNSW resident lease is too small");
    if (!budget.ok()) {
      throw std::runtime_error("resource_exhausted: HNSW resident lease is too small");
    }
  }

  static auto required_artifact(const core::ArtifactView &artifact, std::string_view name)
      -> std::string_view {
    const auto path = artifact.find(name);
    if (path.empty()) {
      throw std::invalid_argument("HNSW required artifact is missing: " + std::string(name));
    }
    return path;
  }

  auto validate_search_request(const core::SearchRequest &request) const -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "HNSW search request is incomplete or incompatible");
    }
    auto status = core::validate_tensor(request.queries,
                                        search_space_->get_dim(),
                                        core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::scalar_type_for<DataType>) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::validation,
                                 core::StatusDetail::unsupported_scalar_type,
                                 "HNSW does not implicitly convert query tensor scalar types");
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
                                 "HNSW does not support a compiled filter view");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "HNSW top_k exceeds uint32");
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }
    std::uint64_t scratch_per_hit{};
    std::uint64_t scratch_bytes{};
    if (!core::checked_add(sizeof(IDType), sizeof(DistanceType), scratch_per_hit) ||
        !core::checked_multiply(request.options.top_k, scratch_per_hit, scratch_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "HNSW query scratch size overflows uint64");
    }
    return core::require_lease(request.context->query_scratch_lease,
                               scratch_bytes,
                               core::OperationStage::search,
                               "HNSW query scratch lease is too small");
  }

  static auto resolve_effort(const core::SearchOptions &options) -> core::Result<std::uint32_t> {
    std::uint64_t effort = 100;
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != core::algorithm::hnsw) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     "HNSW received an extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr || extension.payload_size < sizeof(HnswSearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "HNSW search extension payload is truncated");
      }
      const auto &hnsw = *static_cast<const HnswSearchExtension *>(extension.payload);
      if (!core::is_current_struct(hnsw)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "HNSW search extension has an incompatible version");
      }
      effort = hnsw.effort;
    }
    if (effort < options.top_k || effort > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "HNSW effort must be in [top_k, UINT32_MAX]");
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
    response.comparable_metric = search_space_->metric();
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
    scratch.ids.resize(top_k);
    scratch.distances.resize(top_k);
    core::RowCount cursor = 0;
    response.query_count = request.queries.rows;
    response.offsets[0] = 0;
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      try {
        auto *query = const_cast<DataType *>(request.queries.template row<DataType>(row));
        search_job_->search_solo(query,
                                 scratch.ids.data(),
                                 scratch.distances.data(),
                                 top_k,
                                 effort);
        core::RowCount written = 0;
        for (; written < top_k && scratch.ids[written] != std::numeric_limits<IDType>::max();
             ++written) {
          const auto score = static_cast<float>(scratch.distances[written]);
          if (std::isnan(score)) {
            throw std::runtime_error("HNSW produced a NaN numeric score");
          }
          response.hits[cursor + written] =
              core::SearchHit(core::SegmentRowId(static_cast<std::uint64_t>(scratch.ids[written])),
                              score,
                              core::ScoreKind::distance,
                              search_space_->metric(),
                              core::ResultFlag::approximate);
        }
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        if (written == top_k) {
          response.completeness[row] = core::SearchCompleteness::complete_k;
        } else if (search_space_->get_data_num() < top_k &&
                   written == search_space_->get_data_num()) {
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

  void refresh_artifacts(std::string_view graph_path,
                         std::string_view data_path,
                         std::string_view quant_path) const {
    artifact_count_ = 0;
    auto add = [this](std::string_view name, std::string_view path) {
      if (!path.empty()) {
        artifacts_[artifact_count_++] =
            core::Artifact(name,
                           static_cast<std::uint64_t>(
                               std::filesystem::file_size(std::filesystem::path(path))),
                           0);
      }
    };
    add(kGraphArtifactName, graph_path);
    add(kDataArtifactName, data_path);
    if constexpr (!std::is_same_v<SearchSpaceType, BuildSpaceType>) {
      add(kQuantArtifactName, quant_path);
    }
  }

  std::shared_ptr<SearchSpaceType> search_space_;
  std::shared_ptr<BuildSpaceType> build_space_;
  std::shared_ptr<GraphType> graph_;
  std::shared_ptr<GraphSearchJob<SearchSpaceType, BuildSpaceType>> search_job_;
  mutable std::array<core::Artifact, 3> artifacts_{};
  mutable std::size_t artifact_count_{0};
};

static_assert(core::Searchable<HnswSegment<RawSpace<>>>);
static_assert(core::BatchSearchable<HnswSegment<RawSpace<>>>);
static_assert(core::Saveable<HnswSegment<RawSpace<>>>);
static_assert(core::StatsProvider<HnswSegment<RawSpace<>>>);
static_assert(!core::Mutable<HnswSegment<RawSpace<>>>);

}  // namespace alaya
