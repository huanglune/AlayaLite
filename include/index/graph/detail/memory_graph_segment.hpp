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
#include "space/space_concepts.hpp"

namespace alaya::detail {

struct MemoryGraphSegmentBridge;

// Shared contract-v3 orchestration for immutable in-memory graphs whose
// existing wire format is Graph + build Space + optional quantized Space.
// Algorithm policy and the legacy build kernel remain in the per-engine
// traits; this class deliberately does not introduce another graph codec.
template <typename Derived, typename SearchSpaceType, typename BuildSpaceType, typename Traits>
  requires Space<SearchSpaceType> && Space<BuildSpaceType>
class MemoryGraphSegmentBase {
 public:
  using SearchSpaceTypeAlias = SearchSpaceType;
  using BuildSpaceTypeAlias = BuildSpaceType;
  using DataType = typename SearchSpaceType::DataTypeAlias;
  using DistanceType = typename SearchSpaceType::DistanceTypeAlias;
  using IDType = typename SearchSpaceType::IDTypeAlias;
  using GraphType =
      Graph<typename BuildSpaceType::DataTypeAlias, typename BuildSpaceType::IDTypeAlias>;
  using BuildOptions = typename Traits::BuildOptions;
  using SearchExtension = typename Traits::SearchExtension;
  using Gate5MemoryGraphSegmentTag = void;

  static_assert(std::is_same_v<DataType, typename BuildSpaceType::DataTypeAlias>,
                "memory graph search/build spaces must use the same vector scalar type");
  static_assert(std::is_same_v<IDType, typename BuildSpaceType::IDTypeAlias>,
                "memory graph search/build spaces must use the same internal node ID type");
  static_assert(std::is_same_v<DataType, float> || std::is_same_v<DataType, std::int8_t> ||
                    std::is_same_v<DataType, std::uint8_t>,
                "contract v3 supports float32, int8, and uint8 tensors");

  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr core::AlgorithmId kAlgorithmId = Traits::kAlgorithmId;
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

  static auto build(BuildInput input, const BuildOptions &options, core::BuildContext &context)
      -> std::unique_ptr<Derived> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::build);
    validate_spaces(input);
    validate_build_options(options, input.build_space->get_data_num());
    validate_build_budget(input, context);
    auto graph =
        std::shared_ptr<GraphType>(Traits::build_graph(input.build_space, options).release());
    Traits::validate_graph(*graph, input.build_space->get_data_num());
    return std::unique_ptr<Derived>(
        new Derived(std::move(input.search_space), std::move(input.build_space), std::move(graph)));
  }

  static auto open(core::ArtifactView artifact,
                   const core::OpenOptions &,
                   core::OpenContext &context) -> std::unique_ptr<Derived> {
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
    Traits::prepare_loaded_graph(*graph);
    Traits::validate_graph(*graph, build_space->get_data_num());
    return std::unique_ptr<Derived>(
        new Derived(std::move(search_space), std::move(build_space), std::move(graph)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = search_space_->get_dim();
    descriptor.metric = search_space_->metric();
    descriptor.stored_scalar_type = core::scalar_type_for<DataType>;
    descriptor.medium = core::Medium::memory;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  [[nodiscard]] static auto make_search_extension(const SearchExtension &options)
      -> core::AlgorithmSearchExtension {
    core::AlgorithmSearchExtension extension;
    extension.algorithm_id = kAlgorithmId;
    extension.payload = std::addressof(options);
    extension.payload_size = sizeof(options);
    return extension;
  }

  auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 diagnostic("single search requires exactly one query row"));
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
    manifest.algorithm_id = kAlgorithmId;
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

  [[nodiscard]] static auto into_any(std::unique_ptr<Derived> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 diagnostic("cannot erase a null segment"));
    }
    auto shared = std::shared_ptr<Derived>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = false;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 protected:
  MemoryGraphSegmentBase(std::shared_ptr<SearchSpaceType> search_space,
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

 private:
  friend struct MemoryGraphSegmentBridge;

  struct SearchScratch {
    std::vector<IDType> ids;
    std::vector<DistanceType> distances;
  };

  [[nodiscard]] static auto diagnostic(std::string_view message) -> std::string {
    return std::string(Traits::kName) + " " + std::string(message);
  }

  static void validate_spaces(const BuildInput &input) {
    if (!core::is_current_struct(input)) {
      throw std::invalid_argument(
          diagnostic("build input has an incompatible size or ABI version"));
    }
    if (input.search_space == nullptr || input.build_space == nullptr) {
      throw std::invalid_argument(diagnostic("build requires search and build spaces"));
    }
    const auto tensor_status = core::validate_tensor(input.vectors,
                                                     input.build_space->get_dim(),
                                                     core::OperationStage::build);
    if (!tensor_status.ok()) {
      throw std::invalid_argument(tensor_status.diagnostic());
    }
    if (input.vectors.scalar_type != core::scalar_type_for<DataType>) {
      throw std::invalid_argument(diagnostic("build tensor scalar type does not match the spaces"));
    }
    if (input.vectors.rows != input.build_space->get_data_num()) {
      throw std::invalid_argument(diagnostic("build tensor row count does not match the spaces"));
    }
    validate_loaded_spaces(input.search_space, input.build_space);
  }

  static void validate_loaded_spaces(const std::shared_ptr<SearchSpaceType> &search_space,
                                     const std::shared_ptr<BuildSpaceType> &build_space) {
    if (search_space == nullptr || build_space == nullptr) {
      throw std::invalid_argument(diagnostic("build requires search and build spaces"));
    }
    if (search_space->get_dim() != build_space->get_dim()) {
      throw std::invalid_argument(diagnostic("search/build space dimension mismatch"));
    }
    if (search_space->get_data_num() != build_space->get_data_num()) {
      throw std::invalid_argument(diagnostic("search/build space row-count mismatch"));
    }
    if (build_space->get_data_num() == 0) {
      throw std::invalid_argument(diagnostic("requires at least one vector"));
    }
  }

  static void validate_build_options(const BuildOptions &options, core::RowCount rows) {
    if (!core::is_current_struct(options)) {
      throw std::invalid_argument(
          diagnostic("build options have an incompatible size or ABI version"));
    }
    Traits::validate_build_options(options, rows);
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
      throw std::invalid_argument(diagnostic("build input byte size overflows uint64"));
    }
    const auto budget =
        context.growing_reservation.ensure(bytes,
                                           core::OperationStage::build,
                                           "memory graph build reservation is too small");
    if (!budget.ok()) {
      throw std::runtime_error(diagnostic("resource_exhausted: build reservation is too small"));
    }
  }

  static void validate_open_budget(std::string_view graph_path,
                                   std::string_view data_path,
                                   const core::OpenContext &context) {
    const auto graph_bytes = std::filesystem::file_size(std::filesystem::path(graph_path));
    const auto data_bytes = std::filesystem::file_size(std::filesystem::path(data_path));
    std::uint64_t bytes{};
    if (!core::checked_add(graph_bytes, data_bytes, bytes)) {
      throw std::invalid_argument(diagnostic("artifact byte size overflows uint64"));
    }
    const auto budget = core::require_lease(context.resident_lease,
                                            bytes,
                                            core::OperationStage::open,
                                            "memory graph resident lease is too small");
    if (!budget.ok()) {
      throw std::runtime_error(diagnostic("resource_exhausted: resident lease is too small"));
    }
  }

  static auto required_artifact(const core::ArtifactView &artifact, std::string_view name)
      -> std::string_view {
    const auto path = artifact.find(name);
    if (path.empty()) {
      throw std::invalid_argument(diagnostic("required artifact is missing: " + std::string(name)));
    }
    return path;
  }

  auto validate_search_request(const core::SearchRequest &request) const -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 diagnostic("search request is incomplete or incompatible"));
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
                                 diagnostic(
                                     "does not implicitly convert query tensor scalar types"));
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
                                 diagnostic("does not support a compiled filter view"));
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 diagnostic("top_k exceeds uint32"));
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
                                 diagnostic("query scratch size overflows uint64"));
    }
    return core::require_lease(request.context->query_scratch_lease,
                               scratch_bytes,
                               core::OperationStage::search,
                               "memory graph query scratch lease is too small");
  }

  static auto resolve_effort(const core::SearchOptions &options) -> core::Result<std::uint32_t> {
    std::uint64_t effort = 100;
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     diagnostic("received an extension for another algorithm"));
        }
        continue;
      }
      if (extension.payload == nullptr || extension.payload_size < sizeof(SearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   diagnostic("search extension payload is truncated"));
      }
      const auto &typed = *static_cast<const SearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   diagnostic("search extension has an incompatible version"));
      }
      effort = typed.effort;
    }
    if (effort < options.top_k || effort > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 diagnostic("effort must be in [top_k, UINT32_MAX]"));
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
            throw std::runtime_error(diagnostic("produced a NaN numeric score"));
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

// One detail-only bridge serves NSG and Fusion consumers that still require
// GraphSearchJob/materialized-view access until their later abstraction gate.
struct MemoryGraphSegmentBridge {
  template <typename Segment>
  static auto graph(const Segment &segment) {
    return segment.graph_;
  }

  template <typename Segment>
  static auto search_space(const Segment &segment) {
    return segment.search_space_;
  }

  template <typename Segment>
  static auto build_space(const Segment &segment) {
    return segment.build_space_;
  }
};

}  // namespace alaya::detail
