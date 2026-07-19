// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "core/resource_contexts.hpp"
#include "core/value_types.hpp"
#include "index/graph/frozen_graph_snapshot.hpp"
#include "index/graph/seal_topology/detail/qg_builder_kernel.hpp"
#include "index/graph/seal_topology/detail/qg_graph.hpp"
#include "space/rabitq_space.hpp"
#include "space/space_concepts.hpp"

namespace alaya::memory_qg {

struct BuildOptions {
  core::VersionedStructHeader header{56, core::kContractAbiVersion};
  std::uint32_t ef_build{400};
  std::uint32_t thread_count{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t reserved_options[2]{};
  std::uint64_t reserved[4]{};
};

static_assert(sizeof(BuildOptions) == 56,
              "same-toolchain layout regression canary for memory QG build options");

// Topology-only memory QG builder. It intentionally has no segment lifecycle,
// artifact codec, search operation, descriptor, or AnySegment registration.
// The Collection IP/cosine seal bridge is its production consumer.
template <typename SpaceType>
  requires Space<SpaceType> && is_rabitq_space_v<SpaceType>
class Builder {
 public:
  using DataType = typename SpaceType::DataTypeAlias;
  using DistanceType = typename SpaceType::DistanceTypeAlias;
  using IDType = typename SpaceType::IDTypeAlias;

  static_assert(std::is_same_v<DataType, float>, "memory QG builds float32 vectors only");
  static_assert(std::is_same_v<DistanceType, float>, "memory QG uses float32 distances");
  static_assert(std::is_same_v<IDType, std::uint32_t>, "memory QG uses uint32 node IDs");

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

  [[nodiscard]] static auto build(BuildInput input,
                                  const BuildOptions &options,
                                  core::BuildContext &context) -> FrozenGraphSnapshot {
    validate_context(context);
    validate_input(input);
    validate_options(options);
    validate_budget(input, context);

    auto graph = std::make_shared<detail::QgBuildGraph<SpaceType>>(input.space);
    auto space_view = std::make_shared<detail::QgBuilderSpaceView<SpaceType>>(*input.space, *graph);
    detail::QgBuilderKernel<detail::QgBuilderSpaceView<SpaceType>> builder(space_view,
                                                                           options.thread_count);
    builder.set_ef_build(options.ef_build);
    builder.build_graph();
    validate_graph(*input.space, *graph);
    return export_snapshot(*input.space, *graph);
  }

 private:
  static void validate_context(const core::BuildContext &context) {
    const auto status = core::validate_runtime_control(context.deadline,
                                                       context.cancellation,
                                                       core::OperationStage::build);
    if (!status.ok()) {
      throw std::runtime_error(status.diagnostic());
    }
  }

  static void validate_input(const BuildInput &input) {
    if (!core::is_current_struct(input)) {
      throw std::invalid_argument("memory QG build input has an incompatible size or ABI version");
    }
    if (input.space == nullptr) {
      throw std::invalid_argument("memory QG build requires a RaBitQ space");
    }
    const auto tensor_status =
        core::validate_tensor(input.vectors, input.space->get_dim(), core::OperationStage::build);
    if (!tensor_status.ok()) {
      throw std::invalid_argument(tensor_status.diagnostic());
    }
    if (input.vectors.scalar_type != core::scalar_type_for<DataType>) {
      throw std::invalid_argument("memory QG build tensor scalar type does not match the space");
    }
    if (input.vectors.rows != input.space->get_data_num()) {
      throw std::invalid_argument("memory QG build tensor row count does not match the space");
    }
    if (input.space->get_data_num() <= SpaceType::kDegreeBound) {
      throw std::invalid_argument(
          "memory QG build requires more vectors than its fixed degree bound");
    }
  }

  static void validate_options(const BuildOptions &options) {
    if (!core::is_current_struct(options)) {
      throw std::invalid_argument(
          "memory QG build options have an incompatible size or ABI version");
    }
    if (options.ef_build == 0) {
      throw std::invalid_argument("memory QG ef_build must be at least 1");
    }
    if (options.thread_count == 0) {
      throw std::invalid_argument("memory QG thread_count must be at least 1");
    }
  }

  static void validate_budget(const BuildInput &input, core::BuildContext &context) {
    std::uint64_t elements{};
    std::uint64_t bytes{};
    if (!core::checked_multiply(input.space->get_data_num(), input.space->get_dim(), elements) ||
        !core::checked_multiply(elements, sizeof(DataType), bytes)) {
      throw std::invalid_argument("memory QG build input byte size overflows uint64");
    }
    const auto budget =
        context.growing_reservation.ensure(bytes,
                                           core::OperationStage::build,
                                           "memory QG build reservation is too small");
    if (!budget.ok()) {
      throw std::runtime_error("memory QG resource_exhausted: build reservation is too small");
    }
  }

  static void validate_graph(SpaceType &space, const detail::QgBuildGraph<SpaceType> &graph) {
    const auto rows = static_cast<core::RowCount>(space.get_data_num());
    if (rows <= SpaceType::kDegreeBound || space.get_dim() == 0 ||
        static_cast<core::RowCount>(graph.get_ep()) >= rows) {
      throw std::runtime_error("invalid memory QG build result");
    }
    for (core::RowCount row = 0; row < rows; ++row) {
      const auto *edges = graph.get_edges(static_cast<IDType>(row));
      for (std::size_t edge = 0; edge < SpaceType::kDegreeBound; ++edge) {
        if (static_cast<core::RowCount>(edges[edge]) >= rows) {
          throw std::runtime_error("invalid memory QG build neighbor ID");
        }
      }
    }
  }

  [[nodiscard]] static auto export_snapshot(const SpaceType &space,
                                            const detail::QgBuildGraph<SpaceType> &graph)
      -> FrozenGraphSnapshot {
    const auto num_points = static_cast<std::size_t>(space.get_data_num());
    FrozenGraphSnapshot::Adjacency adjacency(num_points);
    for (std::size_t node = 0; node < num_points; ++node) {
      auto &neighbors = adjacency[node];
      neighbors.reserve(detail::QgBuildGraph<SpaceType>::kDegreeBound);
      const auto *edges = graph.get_edges(static_cast<IDType>(node));
      for (std::size_t edge = 0; edge < detail::QgBuildGraph<SpaceType>::kDegreeBound; ++edge) {
        const IDType neighbor = edges[edge];
        if (static_cast<std::size_t>(neighbor) == node ||
            std::find(neighbors.begin(), neighbors.end(), neighbor) != neighbors.end()) {
          continue;
        }
        neighbors.push_back(neighbor);
      }
    }

    FrozenGraphSnapshot snapshot(std::move(adjacency),
                                 graph.get_ep(),
                                 static_cast<std::uint32_t>(
                                     detail::QgBuildGraph<SpaceType>::kDegreeBound));
    snapshot.validate();
    return snapshot;
  }
};

template <typename DataType>
using Quantizer = ::alaya::RaBitQQuantizer<DataType>;

}  // namespace alaya::memory_qg
