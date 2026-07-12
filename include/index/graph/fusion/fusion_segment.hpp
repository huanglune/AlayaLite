// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "core/algorithm_registry.hpp"
#include "index/graph/detail/memory_graph_segment.hpp"
#include "index/graph/fusion/detail/fusion_builder_kernel.hpp"
#include "index/graph/hnsw/detail/hnsw_builder_kernel.hpp"
#include "index/graph/nsg/detail/nsg_builder_kernel.hpp"
#include "space/raw_space.hpp"

namespace alaya {

struct FusionBuildOptions {
  core::VersionedStructHeader header{56, core::kContractAbiVersion};
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{200};
  std::uint32_t thread_count{1};
  std::uint32_t reserved_options{};
  std::uint64_t reserved[4]{};
};

static_assert(sizeof(FusionBuildOptions) == 56,
              "same-toolchain layout regression canary for Fusion build options");

struct FusionSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  FusionSearchExtension() : header(core::current_struct_header<FusionSearchExtension>()) {}
};

[[nodiscard]] inline auto make_fusion_search_extension(const FusionSearchExtension &options)
    -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::fusion;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

namespace detail {

template <typename BuildSpaceType>
struct FusionSegmentTraits {
  using BuildOptions = FusionBuildOptions;
  using SearchExtension = FusionSearchExtension;

  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::fusion;
  static constexpr std::string_view kName = "Fusion";

  static auto build_graph(std::shared_ptr<BuildSpaceType> &space, const BuildOptions &options)
      -> std::unique_ptr<
          Graph<typename BuildSpaceType::DataTypeAlias, typename BuildSpaceType::IDTypeAlias>> {
    using Kernel = FusionBuilderKernel<BuildSpaceType,
                                       HnswBuilderKernel<BuildSpaceType>,
                                       NsgBuilderKernel<BuildSpaceType>>;
    Kernel builder(space, options.max_neighbors, options.ef_construction);
    return builder.build_graph(options.thread_count);
  }

  static void validate_build_options(const BuildOptions &options, core::RowCount rows) {
    if (options.max_neighbors < 2) {
      throw std::invalid_argument("Fusion max_neighbors must be at least 2");
    }
    if (options.ef_construction == 0 || options.ef_construction > rows ||
        options.ef_construction > std::numeric_limits<std::uint16_t>::max()) {
      throw std::invalid_argument("Fusion ef_construction must be in [1, min(row_count, 65535)]");
    }
    if (rows <= 64) {
      throw std::invalid_argument("Fusion NSG kernel requires at least 65 vectors");
    }
  }

  template <typename GraphType>
  static void prepare_loaded_graph(GraphType &graph) {
    if constexpr (sizeof(typename GraphType::NodeIDTypeAlias) > sizeof(std::uint32_t)) {
      if (graph.overlay_graph_ != nullptr) {
        // OverlayGraph's retained codec writes these header fields as four
        // bytes even when NodeIDType is uint64_t. Its legacy load leaves the
        // upper bytes of ep_ indeterminate; zero-extension restores the value
        // that was actually persisted without changing the wire format.
        graph.overlay_graph_->node_num_ =
            static_cast<std::uint32_t>(graph.overlay_graph_->node_num_);
        graph.overlay_graph_->max_nbrs_ =
            static_cast<std::uint32_t>(graph.overlay_graph_->max_nbrs_);
        graph.overlay_graph_->ep_ = static_cast<std::uint32_t>(graph.overlay_graph_->ep_);
      }
    }
  }

  template <typename GraphType>
  static void validate_graph(const GraphType &graph, core::RowCount rows) {
    if (graph.max_nodes_ < rows || graph.max_nbrs_ == 0 || graph.overlay_graph_ == nullptr ||
        graph.overlay_graph_->node_num_ != graph.max_nodes_ ||
        graph.overlay_graph_->max_nbrs_ == 0 || graph.overlay_graph_->max_nbrs_ > graph.max_nbrs_ ||
        static_cast<core::RowCount>(graph.overlay_graph_->ep_) >= rows) {
      throw std::runtime_error("Invalid Fusion graph artifact");
    }
  }
};

}  // namespace detail

template <typename SearchSpaceType, typename BuildSpaceType = SearchSpaceType>
  requires Space<SearchSpaceType> && Space<BuildSpaceType>
class FusionSegment
    : public detail::MemoryGraphSegmentBase<FusionSegment<SearchSpaceType, BuildSpaceType>,
                                            SearchSpaceType,
                                            BuildSpaceType,
                                            detail::FusionSegmentTraits<BuildSpaceType>> {
 private:
  using Base = detail::MemoryGraphSegmentBase<FusionSegment<SearchSpaceType, BuildSpaceType>,
                                              SearchSpaceType,
                                              BuildSpaceType,
                                              detail::FusionSegmentTraits<BuildSpaceType>>;
  friend Base;

  FusionSegment(std::shared_ptr<SearchSpaceType> search_space,
                std::shared_ptr<BuildSpaceType> build_space,
                std::shared_ptr<typename Base::GraphType> graph)
      : Base(std::move(search_space), std::move(build_space), std::move(graph)) {}
};

static_assert(core::Searchable<FusionSegment<RawSpace<>>>);
static_assert(core::BatchSearchable<FusionSegment<RawSpace<>>>);
static_assert(core::Saveable<FusionSegment<RawSpace<>>>);
static_assert(core::StatsProvider<FusionSegment<RawSpace<>>>);
static_assert(!core::Mutable<FusionSegment<RawSpace<>>>);

}  // namespace alaya
