// SPDX-License-Identifier: AGPL-3.0-only
#include "index/graph/graph.hpp"

#include <cstdint>
#include <type_traits>

using GoldenGraph = alaya::Graph<float, uint32_t>;
static_assert(std::is_constructible_v<GoldenGraph, uint32_t, uint32_t>);

auto golden_graph_compile() -> int {
  GoldenGraph graph(4, 2);
  graph.at(0, 0) = 1;
  return static_cast<int>(graph.at(0, 0));
}

auto main() -> int { return golden_graph_compile() == 1 ? 0 : 1; }
