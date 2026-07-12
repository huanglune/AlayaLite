// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
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

#include "core/capabilities.hpp"
#include "core/compat.hpp"
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
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{200};
  std::uint32_t thread_count{1};
};

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

  static constexpr std::uint32_t kFormatVersion = 1;
  static constexpr std::string_view kGraphArtifactName = "graph";
  static constexpr std::string_view kDataArtifactName = "data";
  static constexpr std::string_view kQuantArtifactName = "quant";

  struct BuildInput {
    std::shared_ptr<SearchSpaceType> search_space;
    std::shared_ptr<BuildSpaceType> build_space;
  };

  static auto build(BuildInput input, const HnswBuildOptions &options, core::BuildContext &)
      -> std::unique_ptr<HnswSegment> {
    validate_spaces(input);
    validate_build_options(options);
    detail::HnswBuilderKernel<BuildSpaceType> builder(input.build_space,
                                                      options.max_neighbors,
                                                      options.ef_construction);
    auto graph = std::shared_ptr<GraphType>(builder.build_graph(options.thread_count).release());
    return std::unique_ptr<HnswSegment>(new HnswSegment(std::move(input.search_space),
                                                        std::move(input.build_space),
                                                        std::move(graph)));
  }

  static auto open(core::ArtifactView artifact, const core::OpenOptions &, core::OpenContext &)
      -> std::unique_ptr<HnswSegment> {
    const auto graph_path = required_artifact(artifact, kGraphArtifactName);
    const auto data_path = required_artifact(artifact, kDataArtifactName);
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
    validate_spaces({search_space, build_space});
    auto graph = std::make_shared<GraphType>();
    graph->load(graph_path);
    validate_graph(*graph, build_space->get_data_num());
    return std::unique_ptr<HnswSegment>(
        new HnswSegment(std::move(search_space), std::move(build_space), std::move(graph)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    return {.id = 0,
            .algorithm_id = core::compat::kAlgorithmHnsw,
            .format_version = kFormatVersion,
            .dim = search_space_->get_dim(),
            .rows = search_space_->get_data_num(),
            .metric = search_space_->metric(),
            .medium = core::Medium::memory,
            .state = core::SegmentState::sealed,
            .reserved = 0};
  }

  auto search(core::QueryView query,
              const core::SearchOptions &options,
              core::SearchSink output) const -> core::SearchResult {
    validate_search(query, options, output.size());
    if (options.top_k == 0) {
      return {};
    }

    auto &scratch = search_scratch();
    scratch.ids.resize(options.top_k);
    scratch.distances.resize(options.top_k);
    auto *typed_query = prepare_query(query, scratch);
    search_job_->search_solo(typed_query,
                             scratch.ids.data(),
                             scratch.distances.data(),
                             options.top_k,
                             options.effort);
    std::size_t written = 0;
    for (; written < options.top_k && scratch.ids[written] != std::numeric_limits<IDType>::max();
         ++written) {
      output[written] = {static_cast<core::ExternalId>(scratch.ids[written]),
                         static_cast<float>(scratch.distances[written])};
    }
    return {.count = written, .visited = 0};
  }

  auto batch_search(core::QueryBatchView queries,
                    const core::SearchOptions &options,
                    core::SearchSink output) const -> core::BatchSearchResult {
    if (queries.dim != search_space_->get_dim()) {
      throw std::invalid_argument("HNSW query dimension mismatch");
    }
    if (queries.rows != 0 && queries.data == nullptr) {
      throw std::invalid_argument("HNSW query batch data is null");
    }
    if (options.filter != nullptr) {
      throw std::invalid_argument("HNSW does not support filtered search");
    }
    if (options.effort < options.top_k) {
      throw std::invalid_argument("HNSW search requires effort >= top_k");
    }
    if (options.top_k != 0 &&
        queries.rows > static_cast<core::RowCount>(output.size() / options.top_k)) {
      throw std::invalid_argument("HNSW batch search sink is smaller than rows * top_k");
    }
    const auto per_query = static_cast<std::size_t>(options.top_k);
    core::RowCount hits = 0;
    for (core::RowCount row = 0; row < queries.rows; ++row) {
      auto sink = output.subspan(static_cast<std::size_t>(row) * per_query, per_query);
      const auto offset = static_cast<std::size_t>(row) * queries.dim;
      hits += search({queries.data + offset, queries.dim}, options, sink).count;
    }
    return {.query_count = queries.rows, .hit_count = hits};
  }

  auto save(core::ArtifactWriter &writer, const core::SaveOptions &) const
      -> core::ArtifactManifest {
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
    return {.schema_version = 1,
            .format_version = kFormatVersion,
            .algorithm_id = core::compat::kAlgorithmHnsw,
            .artifacts = std::span<const core::Artifact>(artifacts_.data(), artifact_count_)};
  }

 private:
  friend struct detail::HnswSegmentBridge<SearchSpaceType, BuildSpaceType>;

  struct SearchScratch {
    std::vector<IDType> ids;
    std::vector<DistanceType> distances;
    std::vector<DataType> converted_query;
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
    if (input.search_space == nullptr || input.build_space == nullptr) {
      throw std::invalid_argument("HNSW build requires search and build spaces");
    }
    if (input.search_space->get_dim() != input.build_space->get_dim()) {
      throw std::invalid_argument("HNSW search/build space dimension mismatch");
    }
    if (input.search_space->get_data_num() != input.build_space->get_data_num()) {
      throw std::invalid_argument("HNSW search/build space row-count mismatch");
    }
    if (input.build_space->get_data_num() == 0) {
      throw std::invalid_argument("HNSW requires at least one vector");
    }
  }

  static void validate_build_options(const HnswBuildOptions &options) {
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

  static auto required_artifact(const core::ArtifactView &artifact, std::string_view name)
      -> std::string_view {
    const auto path = artifact.find(name);
    if (path.empty()) {
      throw std::invalid_argument("HNSW required artifact is missing: " + std::string(name));
    }
    return path;
  }

  void validate_search(core::QueryView query,
                       const core::SearchOptions &options,
                       std::size_t output_size) const {
    if (query.dim != search_space_->get_dim()) {
      throw std::invalid_argument("HNSW query dimension mismatch");
    }
    if (query.dim != 0 && query.data == nullptr) {
      throw std::invalid_argument("HNSW query data is null");
    }
    if (options.filter != nullptr) {
      throw std::invalid_argument("HNSW does not support filtered search");
    }
    if (options.effort < options.top_k || output_size < options.top_k) {
      throw std::invalid_argument("HNSW search requires effort and sink >= top_k");
    }
  }

  static auto search_scratch() -> SearchScratch & {
    static thread_local SearchScratch scratch;
    return scratch;
  }

  static auto prepare_query(core::QueryView query, SearchScratch &scratch) -> DataType * {
    if constexpr (std::is_same_v<DataType, float>) {
      return const_cast<float *>(query.data);
    } else {
      scratch.converted_query.resize(query.dim);
      std::transform(query.data,
                     query.data + query.dim,
                     scratch.converted_query.begin(),
                     [](float value) {
                       return static_cast<DataType>(value);
                     });
      return scratch.converted_query.data();
    }
  }

  void refresh_artifacts(std::string_view graph_path,
                         std::string_view data_path,
                         std::string_view quant_path) const {
    artifact_count_ = 0;
    auto add = [this](std::string_view name, std::string_view path) {
      if (!path.empty()) {
        artifacts_[artifact_count_++] = {name,
                                         static_cast<std::uint64_t>(std::filesystem::file_size(
                                             std::filesystem::path(path))),
                                         0};
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
static_assert(core::Persistable<HnswSegment<RawSpace<>>>);
static_assert(!core::Mutable<HnswSegment<RawSpace<>>>);

}  // namespace alaya
