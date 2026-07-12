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
#include <type_traits>
#include <vector>

#include "core/capabilities.hpp"
#include "core/compat.hpp"
#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
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
  using DistanceSpaceTypeAlias = BuildSpaceType;
  using DataType = typename SearchSpaceType::DataTypeAlias;
  using DistanceType = typename SearchSpaceType::DistanceTypeAlias;
  using IDType = typename SearchSpaceType::IDTypeAlias;
  using GraphType =
      Graph<typename BuildSpaceType::DataTypeAlias, typename BuildSpaceType::IDTypeAlias>;

  struct BuildInput {
    std::shared_ptr<SearchSpaceType> search_space;
    std::shared_ptr<BuildSpaceType> build_space;
  };

  static auto build(BuildInput input, const HnswBuildOptions &options, core::BuildContext &)
      -> std::unique_ptr<HnswSegment> {
    validate_spaces(input);
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
    if (artifact.graph_path.empty()) {
      throw std::invalid_argument("HNSW graph artifact path is empty");
    }
    if (artifact.data_path.empty()) {
      throw std::invalid_argument("HNSW data artifact path is empty");
    }
    auto build_space = std::make_shared<BuildSpaceType>();
    build_space->load(artifact.data_path);
    build_space->set_metric_function();
    std::shared_ptr<SearchSpaceType> search_space;
    if constexpr (std::is_same_v<SearchSpaceType, BuildSpaceType>) {
      search_space = build_space;
    } else {
      if (artifact.quant_path.empty()) {
        throw std::invalid_argument("HNSW quant artifact path is empty");
      }
      search_space = std::make_shared<SearchSpaceType>();
      search_space->load(artifact.quant_path);
      search_space->set_metric_function();
    }
    validate_spaces({search_space, build_space});
    auto graph = std::make_shared<GraphType>();
    graph->load(artifact.graph_path);
    return std::unique_ptr<HnswSegment>(
        new HnswSegment(std::move(search_space), std::move(build_space), std::move(graph)));
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    return {.id = 0,
            .algorithm_id = core::compat::kAlgorithmHnsw,
            .format_version = kFormatVersion,
            .dim = search_space_->get_dim(),
            .rows = search_space_->get_data_num(),
            .metric = metric(),
            .medium = core::Medium::memory,
            .state = core::SegmentState::sealed,
            .reserved = 0};
  }

  auto search(core::QueryView query,
              const core::SearchOptions &options,
              core::SearchSink output) const -> core::SearchResult {
    validate_search(query.dim, options, output.size());
    const auto count = std::min<std::size_t>(options.top_k, output.size());
    std::vector<IDType> ids(count);
    std::vector<DistanceType> distances(count);
    search_job_->search_solo(const_cast<DataType *>(query.data),
                             ids.data(),
                             distances.data(),
                             static_cast<std::uint32_t>(count),
                             options.effort);
    std::size_t written = 0;
    for (; written < count && ids[written] != std::numeric_limits<IDType>::max(); ++written) {
      output[written] = {static_cast<core::ExternalId>(ids[written]),
                         static_cast<float>(distances[written])};
    }
    return {.count = written, .visited = 0};
  }

  auto batch_search(core::QueryBatchView queries,
                    const core::SearchOptions &options,
                    core::SearchSink output) const -> core::BatchSearchResult {
    if (queries.rows != 0 && queries.data == nullptr) {
      throw std::invalid_argument("HNSW query batch data is null");
    }
    const auto per_query =
        std::min<std::size_t>(options.top_k, queries.rows == 0 ? 0 : output.size() / queries.rows);
    core::RowCount hits = 0;
    for (core::RowCount row = 0; row < queries.rows; ++row) {
      auto sink = output.subspan(static_cast<std::size_t>(row) * per_query, per_query);
      hits += search({queries.data + row * queries.dim, queries.dim}, options, sink).count;
    }
    return {.query_count = queries.rows, .hit_count = hits};
  }

  auto save(core::ArtifactWriter &writer, const core::SaveOptions &) const
      -> core::ArtifactManifest {
    if (writer.graph_path.empty()) {
      throw std::invalid_argument("HNSW graph artifact path is empty");
    }
    graph_->save(writer.graph_path);
    if (!writer.data_path.empty()) {
      build_space_->save(writer.data_path);
    }
    if constexpr (!std::is_same_v<SearchSpaceType, BuildSpaceType>) {
      if (!writer.quant_path.empty()) {
        search_space_->save(writer.quant_path);
      }
    }
    refresh_artifacts(writer);
    return {.schema_version = 1,
            .format_version = kFormatVersion,
            .algorithm_id = core::compat::kAlgorithmHnsw,
            .artifacts = std::span<const core::Artifact>(artifacts_.data(), artifact_count_)};
  }

 private:
  friend struct detail::HnswSegmentBridge<SearchSpaceType, BuildSpaceType>;
  static constexpr std::uint32_t kFormatVersion = 1;

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
  }

  void validate_search(std::uint32_t dim,
                       const core::SearchOptions &options,
                       std::size_t output_size) const {
    if (dim != search_space_->get_dim()) {
      throw std::invalid_argument("HNSW query dimension mismatch");
    }
    if (options.effort < options.top_k || output_size < options.top_k) {
      throw std::invalid_argument("HNSW search requires effort and sink >= top_k");
    }
  }

  [[nodiscard]] auto metric() const noexcept -> core::Metric {
    switch (search_space_->metric_) {
      case MetricType::IP:
        return core::Metric::inner_product;
      case MetricType::COS:
        return core::Metric::cosine;
      default:
        return core::Metric::l2;
    }
  }

  void refresh_artifacts(const core::ArtifactWriter &writer) const {
    artifact_count_ = 0;
    auto add = [this](std::string_view name) {
      if (!name.empty()) {
        artifacts_[artifact_count_++] = {name,
                                         static_cast<std::uint64_t>(std::filesystem::file_size(
                                             std::filesystem::path(name))),
                                         0};
      }
    };
    add(writer.graph_path);
    add(writer.data_path);
    if constexpr (!std::is_same_v<SearchSpaceType, BuildSpaceType>) add(writer.quant_path);
  }

  std::shared_ptr<SearchSpaceType> search_space_;
  std::shared_ptr<BuildSpaceType> build_space_;
  std::shared_ptr<GraphType> graph_;
  std::shared_ptr<GraphSearchJob<SearchSpaceType, BuildSpaceType>> search_job_;
  mutable std::array<core::Artifact, 3> artifacts_{};
  mutable std::size_t artifact_count_{0};
};

namespace detail {
// Temporary owner bridge for consumers whose hybrid-search/update plumbing is
// migrated in later abstraction steps. It is intentionally absent from the
// public HnswSegment API and can be deleted with those jobs.
template <typename SearchSpaceType, typename BuildSpaceType>
struct HnswSegmentBridge {
  using Segment = HnswSegment<SearchSpaceType, BuildSpaceType>;
  static auto graph(const Segment &segment) { return segment.graph_; }
  static auto search_job(const Segment &segment) { return segment.search_job_; }
};
}  // namespace detail

static_assert(core::Searchable<HnswSegment<RawSpace<>>>);
static_assert(core::BatchSearchable<HnswSegment<RawSpace<>>>);
static_assert(core::Persistable<HnswSegment<RawSpace<>>>);
static_assert(!core::Mutable<HnswSegment<RawSpace<>>>);

}  // namespace alaya
