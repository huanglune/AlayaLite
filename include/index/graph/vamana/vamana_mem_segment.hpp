// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/algorithm_registry.hpp"
#include "core/any_segment.hpp"
#include "core/platform_fs.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_greedy_search.hpp"
#include "index/graph/vamana/vamana_reader.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "index/memory_engine_registry.hpp"

namespace alaya {

// Vamana's retained in-memory kernels are float32/L2-only.  The single-thread
// build requirement makes the existing builder's fixed-seed artifact contract
// explicit: its seed is wired through, while parallel inter-insert scheduling
// is intentionally not presented as deterministic.
struct VamanaMemBuildOptions {
  core::VersionedStructHeader header{64, core::kContractAbiVersion};
  std::uint32_t max_neighbors{64};
  std::uint32_t construction_effort{200};
  float alpha{1.2F};
  std::uint32_t thread_count{1};
  std::uint32_t max_candidates{750};
  std::uint32_t reserved_options{};
  std::uint64_t seed{1234};
  std::uint64_t reserved[3]{};
};

static_assert(sizeof(VamanaMemBuildOptions) == 64,
              "same-toolchain layout regression canary for Vamana-memory build options");

struct VamanaMemSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  VamanaMemSearchExtension() : header(core::current_struct_header<VamanaMemSearchExtension>()) {}
};

[[nodiscard]] inline auto make_vamana_mem_search_extension(const VamanaMemSearchExtension &options)
    -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::vamana;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

class VamanaMemSegment {
 public:
  using DataType = float;
  using DistanceType = float;
  using IDType = std::uint32_t;
  using BuildOptions = VamanaMemBuildOptions;
  using SearchExtension = VamanaMemSearchExtension;

  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::vamana;
  static constexpr std::string_view kGraphArtifactName = "graph";
  static constexpr std::string_view kDataArtifactName = "data";

  struct BuildInput {
    core::VersionedStructHeader header{};
    core::TypedTensorView vectors{};
    std::uint64_t reserved[3]{};

    explicit BuildInput(core::TypedTensorView vector_view)
        : header(core::current_struct_header<BuildInput>()), vectors(vector_view) {}
  };

  static auto build(BuildInput input, const BuildOptions &options, core::BuildContext &context)
      -> std::unique_ptr<VamanaMemSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::build);
    validate_build_input(input);
    validate_build_options(options, input.vectors.rows);
    validate_build_budget(input, options, context);

    auto vectors = copy_vectors(input.vectors);
    vamana::VamanaBuildParams params;
    params.R = options.max_neighbors;
    params.L = options.construction_effort;
    params.alpha = options.alpha;
    params.num_threads = options.thread_count;
    params.maxc = options.max_candidates;
    params.seed = options.seed;
    vamana::VamanaBuilder builder(vectors.data(),
                                  static_cast<std::size_t>(input.vectors.rows),
                                  input.vectors.dim,
                                  params);
    builder.build();

    // VamanaReader intentionally owns and validates the searchable adjacency,
    // while VamanaBuilder returns a different in-memory shape.  Compose the
    // retained writer and reader once at build time rather than copying either
    // codec or the greedy-search loop into the Segment.
    auto reader = reader_from_built_graph(builder.graph(), options.max_neighbors, builder.medoid());
    return std::unique_ptr<VamanaMemSegment>(
        new VamanaMemSegment(std::move(vectors), input.vectors.dim, std::move(reader)));
  }

  static auto open(core::ArtifactView artifact,
                   const core::OpenOptions &,
                   core::OpenContext &context) -> std::unique_ptr<VamanaMemSegment> {
    validate_context(context.deadline, context.cancellation, core::OperationStage::open);
    const auto graph_path = required_artifact(artifact, kGraphArtifactName);
    const auto data_path = required_artifact(artifact, kDataArtifactName);
    validate_open_budget(graph_path, data_path, context);
    auto [vectors, shape] = load_fbin(data_path);
    auto reader = std::make_unique<vamana::VamanaReader>(std::filesystem::path(graph_path));
    if (reader->num_nodes() != shape.rows) {
      throw std::runtime_error("Vamana-memory graph/data row-count mismatch");
    }
    return std::unique_ptr<VamanaMemSegment>(
        new VamanaMemSegment(std::move(vectors), shape.dim, std::move(reader)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = kAlgorithmId;
    descriptor.format_version = kFormatVersion;
    descriptor.factory_version = 1;
    descriptor.dim = dim_;
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::memory;
    descriptor.preprocessing = core::MetricPreprocessing::none;
    descriptor.engine_factory_id = kAlgorithmId;
    return descriptor;
  }

  [[nodiscard]] static auto make_search_extension(const SearchExtension &options)
      -> core::AlgorithmSearchExtension {
    return make_vamana_mem_search_extension(options);
  }

  auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "Vamana-memory single search requires exactly one query row");
    }
    return execute_search(request);
  }

  auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute_search(request);
  }

  auto save(core::ArtifactWriter &writer,
            const core::SaveOptions &,
            core::ArtifactManifest &manifest) const -> core::Status {
    try {
      // Resolve the complete logical transaction before creating either file.
      const auto graph_path = required_artifact(writer, kGraphArtifactName);
      const auto data_path = required_artifact(writer, kDataArtifactName);
      vamana::save_graph(reader_->graph(),
                         std::filesystem::path(graph_path),
                         reader_->max_degree(),
                         reader_->start(),
                         reader_->frozen_pts());
      save_fbin(data_path, vectors_, static_cast<std::uint32_t>(reader_->num_nodes()), dim_);
      artifacts_[0] = core::Artifact(kGraphArtifactName,
                                     static_cast<std::uint64_t>(std::filesystem::file_size(
                                         std::filesystem::path(graph_path))),
                                     0);
      artifacts_[1] = core::Artifact(kDataArtifactName,
                                     static_cast<std::uint64_t>(std::filesystem::file_size(
                                         std::filesystem::path(data_path))),
                                     0);
      manifest = core::ArtifactManifest{};
      manifest.schema_version = 1;
      manifest.format_version = kFormatVersion;
      manifest.algorithm_id = kAlgorithmId;
      manifest.artifacts = std::span<const core::Artifact>(artifacts_);
      return core::Status::success();
    } catch (...) {
      return core::status_from_exception(core::OperationStage::save);
    }
  }

  auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.snapshot_version = 1;
    stats.live_rows = reader_->num_nodes();
    stats.allocated_rows = reader_->num_nodes();
    stats.resident_bytes = resident_bytes_;
    stats.health = core::SegmentHealth::healthy;
    return core::Status::success();
  }

  [[nodiscard]] static auto into_any(std::unique_ptr<VamanaMemSegment> segment)
      -> core::Result<core::AnySegment> {
    if (segment == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::admission,
                                 core::StatusDetail::null_data,
                                 "cannot erase a null Vamana-memory segment");
    }
    auto shared = std::shared_ptr<VamanaMemSegment>(std::move(segment));
    core::SegmentInstanceConfig config;
    config.readonly = true;
    config.concurrency.reentrant_search = true;
    config.concurrency.native_async = false;
    config.concurrency.cooperative_cancel = false;
    config.concurrency.explicit_drain = false;
    return core::AnySegment::from_sync(std::move(shared), std::move(config));
  }

 private:
  struct FbinShape {
    std::uint32_t rows{};
    std::uint32_t dim{};
  };

  struct ScopedTemporaryFile {
    std::filesystem::path path;
    ~ScopedTemporaryFile() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  };

  VamanaMemSegment(std::vector<float> vectors,
                   std::uint32_t dim,
                   std::unique_ptr<vamana::VamanaReader> reader)
      : vectors_(std::move(vectors)), dim_(dim), reader_(std::move(reader)) {
    greedy_search_ = std::make_unique<vamana::VamanaGreedySearch>(*reader_, vectors_.data(), dim_);
    resident_bytes_ = estimate_resident_bytes();
  }

  static void validate_context(const core::Deadline &deadline,
                               const core::CancellationToken &cancellation,
                               core::OperationStage stage) {
    const auto status = core::validate_runtime_control(deadline, cancellation, stage);
    if (!status.ok()) {
      throw std::runtime_error(status.diagnostic());
    }
  }

  static void validate_build_input(const BuildInput &input) {
    if (!core::is_current_struct(input)) {
      throw std::invalid_argument(
          "Vamana-memory build input has an incompatible size or ABI version");
    }
    const auto tensor_status =
        core::validate_tensor(input.vectors, input.vectors.dim, core::OperationStage::build);
    if (!tensor_status.ok()) {
      throw std::invalid_argument(tensor_status.diagnostic());
    }
    if (input.vectors.scalar_type != core::ScalarType::float32) {
      throw std::invalid_argument("Vamana-memory build requires a float32 tensor");
    }
    if (input.vectors.dim == 0) {
      throw std::invalid_argument("Vamana-memory build dimension must be at least 1");
    }
    if (input.vectors.rows < 2 || input.vectors.rows > std::numeric_limits<std::uint32_t>::max()) {
      throw std::invalid_argument("Vamana-memory build row count must be in [2, UINT32_MAX]");
    }
  }

  static void validate_build_options(const BuildOptions &options, core::RowCount rows) {
    if (!core::is_current_struct(options)) {
      throw std::invalid_argument(
          "Vamana-memory build options have an incompatible size or ABI version");
    }
    if (options.max_neighbors == 0 || options.max_neighbors >= rows) {
      throw std::invalid_argument("Vamana-memory max_neighbors must be in [1, row_count)");
    }
    if (options.construction_effort < options.max_neighbors) {
      throw std::invalid_argument(
          "Vamana-memory construction_effort must be at least max_neighbors");
    }
    if (!std::isfinite(options.alpha) || options.alpha < 1.0F) {
      throw std::invalid_argument("Vamana-memory alpha must be finite and at least 1.0");
    }
    if (options.thread_count != 1) {
      throw std::invalid_argument(
          "Vamana-memory format v1 requires thread_count=1 for deterministic fixed-seed builds");
    }
    if (options.max_candidates < options.max_neighbors) {
      throw std::invalid_argument("Vamana-memory max_candidates must be at least max_neighbors");
    }
  }

  static void validate_build_budget(const BuildInput &input,
                                    const BuildOptions &options,
                                    core::BuildContext &context) {
    std::uint64_t elements{};
    std::uint64_t vector_bytes{};
    std::uint64_t edges{};
    std::uint64_t graph_bytes{};
    std::uint64_t total{};
    if (!core::checked_multiply(input.vectors.rows, input.vectors.dim, elements) ||
        !core::checked_multiply(elements, sizeof(float), vector_bytes) ||
        !core::checked_multiply(input.vectors.rows, options.max_neighbors, edges) ||
        !core::checked_multiply(edges, sizeof(std::uint32_t), graph_bytes) ||
        !core::checked_add(vector_bytes, graph_bytes, total)) {
      throw std::invalid_argument("Vamana-memory build byte size overflows uint64");
    }
    const auto budget =
        context.growing_reservation.ensure(total,
                                           core::OperationStage::build,
                                           "Vamana-memory build reservation is too small");
    if (!budget.ok()) {
      throw std::runtime_error("Vamana-memory resource_exhausted: build reservation is too small");
    }
  }

  static void validate_open_budget(std::string_view graph_path,
                                   std::string_view data_path,
                                   const core::OpenContext &context) {
    const auto graph_bytes = std::filesystem::file_size(std::filesystem::path(graph_path));
    const auto data_bytes = std::filesystem::file_size(std::filesystem::path(data_path));
    std::uint64_t total{};
    if (!core::checked_add(graph_bytes, data_bytes, total)) {
      throw std::invalid_argument("Vamana-memory artifact byte size overflows uint64");
    }
    const auto budget = core::require_lease(context.resident_lease,
                                            total,
                                            core::OperationStage::open,
                                            "Vamana-memory resident lease is too small");
    if (!budget.ok()) {
      throw std::runtime_error("Vamana-memory resource_exhausted: resident lease is too small");
    }
  }

  static auto required_artifact(const core::ArtifactView &artifact, std::string_view name)
      -> std::string_view {
    const auto path = artifact.find(name);
    if (path.empty()) {
      throw std::invalid_argument("Vamana-memory required artifact is missing: " +
                                  std::string(name));
    }
    return path;
  }

  static auto copy_vectors(const core::TypedTensorView &tensor) -> std::vector<float> {
    const auto rows = static_cast<std::size_t>(tensor.rows);
    const auto dim = static_cast<std::size_t>(tensor.dim);
    std::vector<float> vectors(rows * dim);
    for (std::size_t row = 0; row < rows; ++row) {
      const auto *source = tensor.row<float>(row);
      for (std::size_t column = 0; column < dim; ++column) {
        if (!std::isfinite(source[column])) {
          throw std::invalid_argument("Vamana-memory build tensor contains a non-finite component");
        }
      }
      std::copy_n(source, dim, vectors.data() + row * dim);
    }
    return vectors;
  }

  static auto temporary_graph_path() -> std::filesystem::path {
    static std::atomic<std::uint64_t> sequence{0};
    const auto serial = sequence.fetch_add(1, std::memory_order_relaxed);
    const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("alayalite-vamana-mem-" + std::to_string(::alaya::platform::get_pid()) + "-" +
            std::to_string(tick) + "-" + std::to_string(serial) + ".index");
  }

  static auto reader_from_built_graph(const std::vector<std::vector<std::uint32_t>> &graph,
                                      std::uint32_t max_degree,
                                      std::uint32_t start)
      -> std::unique_ptr<vamana::VamanaReader> {
    ScopedTemporaryFile temporary{temporary_graph_path()};
    vamana::save_graph(graph, temporary.path, max_degree, start);
    return std::make_unique<vamana::VamanaReader>(temporary.path);
  }

  static void save_fbin(std::string_view path,
                        const std::vector<float> &vectors,
                        std::uint32_t rows,
                        std::uint32_t dim) {
    const auto output_path = std::filesystem::path(path);
    const auto parent = output_path.parent_path();
    if (!parent.empty()) {
      std::filesystem::create_directories(parent);
    }
    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Vamana-memory cannot open data artifact for writing: " +
                               output_path.string());
    }
    output.write(reinterpret_cast<const char *>(std::addressof(rows)), sizeof(rows));
    output.write(reinterpret_cast<const char *>(std::addressof(dim)), sizeof(dim));
    output.write(reinterpret_cast<const char *>(vectors.data()),
                 static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (!output.good()) {
      throw std::runtime_error("Vamana-memory data artifact write failed: " + output_path.string());
    }
  }

  static auto load_fbin(std::string_view path) -> std::pair<std::vector<float>, FbinShape> {
    const auto input_path = std::filesystem::path(path);
    if (!std::filesystem::is_regular_file(input_path)) {
      throw std::runtime_error("Vamana-memory data artifact is not a regular file: " +
                               input_path.string());
    }
    const auto actual_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(input_path));
    if (actual_bytes < 2 * sizeof(std::uint32_t)) {
      throw std::runtime_error("Vamana-memory data artifact has a truncated fbin header");
    }
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("Vamana-memory cannot open data artifact: " + input_path.string());
    }
    FbinShape shape;
    input.read(reinterpret_cast<char *>(std::addressof(shape.rows)), sizeof(shape.rows));
    input.read(reinterpret_cast<char *>(std::addressof(shape.dim)), sizeof(shape.dim));
    if (!input.good() || shape.rows < 2 || shape.dim == 0) {
      throw std::runtime_error("Vamana-memory data artifact has an invalid fbin shape");
    }
    std::uint64_t elements{};
    std::uint64_t payload_bytes{};
    std::uint64_t expected_bytes{};
    if (!core::checked_multiply(shape.rows, shape.dim, elements) ||
        !core::checked_multiply(elements, sizeof(float), payload_bytes) ||
        !core::checked_add(payload_bytes, 2 * sizeof(std::uint32_t), expected_bytes) ||
        expected_bytes != actual_bytes || elements > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("Vamana-memory data artifact fbin size mismatch");
    }
    std::vector<float> vectors(static_cast<std::size_t>(elements));
    input.read(reinterpret_cast<char *>(vectors.data()),
               static_cast<std::streamsize>(payload_bytes));
    if (!input.good()) {
      throw std::runtime_error("Vamana-memory data artifact has a truncated fbin payload");
    }
    if (std::any_of(vectors.begin(), vectors.end(), [](float value) {
          return !std::isfinite(value);
        })) {
      throw std::runtime_error("Vamana-memory data artifact contains a non-finite component");
    }
    return {std::move(vectors), shape};
  }

  auto validate_search_request(const core::SearchRequest &request) const -> core::Status {
    if (!core::is_current_struct(request) || !core::is_current_struct(request.options) ||
        request.context == nullptr || request.response == nullptr) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "Vamana-memory search request is incomplete or incompatible");
    }
    auto status = core::validate_tensor(request.queries, dim_, core::OperationStage::validation);
    if (!status.ok()) {
      return status;
    }
    if (request.queries.scalar_type != core::ScalarType::float32) {
      return core::Status::
          error(core::StatusCode::not_supported,
                core::OperationStage::validation,
                core::StatusDetail::unsupported_scalar_type,
                "Vamana-memory does not implicitly convert query tensor scalar types");
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
                                 "Vamana-memory does not support a compiled filter view");
    }
    if (request.options.top_k > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "Vamana-memory top_k exceeds uint32");
    }
    status = core::validate_runtime_control(request.context->deadline,
                                            request.context->cancellation,
                                            core::OperationStage::search);
    if (!status.ok()) {
      return status;
    }
    std::uint64_t visited_bytes{};
    std::uint64_t pool_bytes{};
    std::uint64_t scratch_bytes{};
    if (!core::checked_multiply(reader_->num_nodes(),
                                sizeof(std::uint8_t) + sizeof(std::uint32_t),
                                visited_bytes) ||
        !core::checked_multiply(reader_->num_nodes(), sizeof(vamana::GreedyHit), pool_bytes) ||
        !core::checked_add(visited_bytes, pool_bytes, scratch_bytes)) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "Vamana-memory query scratch size overflows uint64");
    }
    return core::require_lease(request.context->query_scratch_lease,
                               scratch_bytes,
                               core::OperationStage::search,
                               "Vamana-memory query scratch lease is too small");
  }

  static auto resolve_effort(const core::SearchOptions &options) -> core::Result<std::uint32_t> {
    std::uint64_t effort = 100;
    for (const auto &extension : options.extensions) {
      if (extension.algorithm_id != kAlgorithmId) {
        if (extension.unknown_policy == core::UnknownExtensionPolicy::reject) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::validation,
                                     core::StatusDetail::unknown_extension,
                                     "Vamana-memory received an extension for another algorithm");
        }
        continue;
      }
      if (extension.payload == nullptr || extension.payload_size < sizeof(SearchExtension)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "Vamana-memory search extension payload is truncated");
      }
      const auto &typed = *static_cast<const SearchExtension *>(extension.payload);
      if (!core::is_current_struct(typed)) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::validation,
                                   core::StatusDetail::malformed_struct,
                                   "Vamana-memory search extension has an incompatible version");
      }
      effort = typed.effort;
    }
    if (effort == 0 || effort > std::numeric_limits<std::uint32_t>::max()) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::arithmetic_overflow,
                                 "Vamana-memory effort must be in [1, UINT32_MAX]");
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
    response.comparable_metric = core::Metric::l2;
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
    const auto requested_top_k = static_cast<std::uint32_t>(request.options.top_k);
    const auto row_count = static_cast<std::uint32_t>(reader_->num_nodes());
    const auto effective_top_k = std::min(requested_top_k, row_count);
    const auto effective_effort =
        std::min(row_count, std::max(std::move(effort_result).value(), effective_top_k));

    core::RowCount cursor = 0;
    response.query_count = request.queries.rows;
    response.offsets[0] = 0;
    for (core::RowCount row = 0; row < request.queries.rows; ++row) {
      try {
        const auto *query = request.queries.row<float>(row);
        for (std::uint32_t column = 0; column < dim_; ++column) {
          if (!std::isfinite(query[column])) {
            throw std::invalid_argument("Vamana-memory query contains a non-finite component");
          }
        }
        const auto hits = greedy_search_->search(query, effective_top_k, effective_effort);
        for (std::size_t hit = 0; hit < hits.size(); ++hit) {
          if (std::isnan(hits[hit].distance)) {
            throw std::runtime_error("Vamana-memory produced a NaN numeric score");
          }
          response.hits[cursor + hit] = core::SearchHit(core::SegmentRowId(hits[hit].id),
                                                        hits[hit].distance,
                                                        core::ScoreKind::distance,
                                                        core::Metric::l2,
                                                        core::ResultFlag::approximate);
        }
        const auto written = static_cast<core::RowCount>(hits.size());
        cursor += written;
        response.offsets[row + 1] = cursor;
        response.valid_counts[row] = written;
        response.statuses[row] = core::Status::success();
        if (written == requested_top_k) {
          response.completeness[row] = core::SearchCompleteness::complete_k;
        } else if (requested_top_k > row_count && written == row_count) {
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

  auto estimate_resident_bytes() const noexcept -> std::uint64_t {
    std::uint64_t result = static_cast<std::uint64_t>(vectors_.size()) * sizeof(float);
    for (const auto &neighbors : reader_->graph()) {
      const auto edge_bytes = static_cast<std::uint64_t>(neighbors.size()) * sizeof(std::uint32_t);
      if (result > std::numeric_limits<std::uint64_t>::max() - edge_bytes) {
        return std::numeric_limits<std::uint64_t>::max();
      }
      result += edge_bytes;
    }
    return result;
  }

  std::vector<float> vectors_;
  std::uint32_t dim_{};
  std::unique_ptr<vamana::VamanaReader> reader_;
  std::unique_ptr<vamana::VamanaGreedySearch> greedy_search_;
  std::uint64_t resident_bytes_{};
  mutable std::array<core::Artifact, 2> artifacts_{};
};

// Standalone factory: Vamana-memory has no Python dispatch row and no legacy
// memory model.  The pre-provisioned feature bit either constructs the real
// Segment or returns not_supported.
class VamanaMemSegmentFactory {
 public:
  static constexpr auto registration = internal::memory::kVamanaMemoryRegistration;

  [[nodiscard]] static auto build(
      VamanaMemSegment::BuildInput input,
      const VamanaMemBuildOptions &options,
      core::BuildContext &context,
      const internal::memory::MemoryEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<VamanaMemSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled_status(core::OperationStage::build);
    }
    try {
      return VamanaMemSegment::build(std::move(input), options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::build);
    }
  }

  [[nodiscard]] static auto open(
      core::ArtifactView artifact,
      const core::OpenOptions &options,
      core::OpenContext &context,
      const internal::memory::MemoryEngineFeatureFlags &features = {}) noexcept
      -> core::Result<std::unique_ptr<VamanaMemSegment>> {
    if (!features.enabled(registration.feature)) {
      return disabled_status(core::OperationStage::open);
    }
    try {
      return VamanaMemSegment::open(artifact, options, context);
    } catch (...) {
      return core::status_from_exception(core::OperationStage::open);
    }
  }

 private:
  static auto disabled_status(core::OperationStage stage) -> core::Status {
    return core::Status::
        error(core::StatusCode::not_supported,
              stage,
              core::StatusDetail::operation_slot_absent,
              "Vamana-memory factory is disabled and has no legacy memory fallback");
  }
};

static_assert(core::Searchable<VamanaMemSegment>);
static_assert(core::BatchSearchable<VamanaMemSegment>);
static_assert(core::Saveable<VamanaMemSegment>);
static_assert(core::StatsProvider<VamanaMemSegment>);
static_assert(!core::Mutable<VamanaMemSegment>);

}  // namespace alaya
