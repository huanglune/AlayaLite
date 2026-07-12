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
#include "index/graph/nsg/detail/nsg_builder_kernel.hpp"
#include "space/raw_space.hpp"

namespace alaya {

struct NsgBuildOptions {
  core::VersionedStructHeader header{56, core::kContractAbiVersion};
  std::uint32_t max_neighbors{32};
  std::uint32_t ef_construction{200};
  std::uint32_t thread_count{1};
  std::uint32_t reserved_options{};
  std::uint64_t reserved[4]{};
};

static_assert(sizeof(NsgBuildOptions) == 56,
              "same-toolchain layout regression canary for NSG build options");

struct NsgSearchExtension {
  core::VersionedStructHeader header{};
  std::uint32_t effort{100};
  std::uint32_t reserved_effort{};
  std::uint64_t reserved[3]{};

  NsgSearchExtension() : header(core::current_struct_header<NsgSearchExtension>()) {}
};

[[nodiscard]] inline auto make_nsg_search_extension(const NsgSearchExtension &options)
    -> core::AlgorithmSearchExtension {
  core::AlgorithmSearchExtension extension;
  extension.algorithm_id = core::algorithm::nsg;
  extension.payload = std::addressof(options);
  extension.payload_size = sizeof(options);
  return extension;
}

namespace detail {

template <typename BuildSpaceType>
struct NsgSegmentTraits {
  using BuildOptions = NsgBuildOptions;
  using SearchExtension = NsgSearchExtension;

  static constexpr core::AlgorithmId kAlgorithmId = core::algorithm::nsg;
  static constexpr std::string_view kName = "NSG";

  static auto build_graph(std::shared_ptr<BuildSpaceType> &space, const BuildOptions &options)
      -> std::unique_ptr<
          Graph<typename BuildSpaceType::DataTypeAlias, typename BuildSpaceType::IDTypeAlias>> {
    NsgBuilderKernel<BuildSpaceType> builder(space, options.max_neighbors, options.ef_construction);
    return builder.build_graph(options.thread_count);
  }

  static void validate_build_options(const BuildOptions &options, core::RowCount rows) {
    if (options.max_neighbors == 0) {
      throw std::invalid_argument("NSG max_neighbors must be at least 1");
    }
    if (options.ef_construction == 0 || options.ef_construction > rows) {
      throw std::invalid_argument("NSG ef_construction must be in [1, row_count]");
    }
    // The legacy NSG kernel seeds a fixed 64-neighbor NN-Descent graph.
    // Reject undersized inputs instead of entering its out-of-range path.
    if (rows <= 64) {
      throw std::invalid_argument("NSG legacy build kernel requires at least 65 vectors");
    }
  }

  template <typename GraphType>
  static void prepare_loaded_graph(GraphType &) {}

  template <typename GraphType>
  static void validate_graph(const GraphType &graph, core::RowCount rows) {
    if (graph.max_nodes_ < rows || graph.max_nbrs_ == 0 || graph.overlay_graph_ != nullptr ||
        graph.eps_.empty()) {
      throw std::runtime_error("Invalid NSG graph artifact");
    }
    for (const auto entry : graph.eps_) {
      if (static_cast<core::RowCount>(entry) >= rows) {
        throw std::runtime_error("Invalid NSG graph entry point");
      }
    }
  }
};

}  // namespace detail

template <typename SearchSpaceType, typename BuildSpaceType = SearchSpaceType>
  requires Space<SearchSpaceType> && Space<BuildSpaceType>
class NsgSegment
    : public detail::MemoryGraphSegmentBase<NsgSegment<SearchSpaceType, BuildSpaceType>,
                                            SearchSpaceType,
                                            BuildSpaceType,
                                            detail::NsgSegmentTraits<BuildSpaceType>> {
 private:
  using Base = detail::MemoryGraphSegmentBase<NsgSegment<SearchSpaceType, BuildSpaceType>,
                                              SearchSpaceType,
                                              BuildSpaceType,
                                              detail::NsgSegmentTraits<BuildSpaceType>>;
  friend Base;

  NsgSegment(std::shared_ptr<SearchSpaceType> search_space,
             std::shared_ptr<BuildSpaceType> build_space,
             std::shared_ptr<typename Base::GraphType> graph)
      : Base(std::move(search_space), std::move(build_space), std::move(graph)) {}
};

static_assert(core::Searchable<NsgSegment<RawSpace<>>>);
static_assert(core::BatchSearchable<NsgSegment<RawSpace<>>>);
static_assert(core::Saveable<NsgSegment<RawSpace<>>>);
static_assert(core::StatsProvider<NsgSegment<RawSpace<>>>);
static_assert(!core::Mutable<NsgSegment<RawSpace<>>>);

}  // namespace alaya
