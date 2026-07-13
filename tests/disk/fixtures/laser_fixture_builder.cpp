// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/vamana/build_dispatch.hpp"

namespace {

auto build_vamana(int argc, char **argv) -> int {
  if (argc != 10) {
    std::cerr << "usage: laser_fixture_builder vamana DATA GRAPH R L ALPHA SEED THREADS DRAM_GB\n";
    return 2;
  }
  const std::string data_path(argv[2]);
  const std::string graph_path(argv[3]);
  auto options = alaya::vamana::kDefaultVamanaBuildParams;
  options.data_path = data_path;
  options.output_path = graph_path;
  options.R = static_cast<std::uint32_t>(std::stoul(argv[4]));
  options.L = static_cast<std::uint32_t>(std::stoul(argv[5]));
  options.alpha = std::stof(argv[6]);
  options.seed = std::stoull(argv[7]);
  options.num_threads = static_cast<std::uint32_t>(std::stoul(argv[8]));
  options.build_dram_budget_gb = std::stof(argv[9]);
  alaya::vamana::build_vamana(options);
  return 0;
}

auto build_laser(int argc, char **argv) -> int {
  if (argc != 11) {
    std::cerr << "usage: laser_fixture_builder laser GRAPH OUTPUT_PREFIX COUNT DIM MAIN_DIM R "
                 "SEED EF THREADS\n";
    return 2;
  }
  const std::string graph_path(argv[2]);
  const std::string output_prefix(argv[3]);
  const auto count = static_cast<std::size_t>(std::stoull(argv[4]));
  const auto dim = static_cast<std::size_t>(std::stoull(argv[5]));
  const auto main_dim = static_cast<std::size_t>(std::stoull(argv[6]));
  const auto degree = static_cast<std::size_t>(std::stoull(argv[7]));
  const auto seed = std::stoull(argv[8]);
  const auto ef_indexing = static_cast<std::size_t>(std::stoull(argv[9]));
  const auto threads = static_cast<std::size_t>(std::stoull(argv[10]));

  alaya::laser::QuantizedGraph graph(count, degree, main_dim, dim, seed, "");
  alaya::laser::QGBuilder builder(graph, ef_indexing, threads);
  builder.build(graph_path.c_str(), output_prefix.c_str());
  return 0;
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    if (argc < 2) {
      std::cerr << "usage: laser_fixture_builder {vamana|laser} ...\n";
      return 2;
    }
    const std::string mode(argv[1]);
    if (mode == "vamana") {
      return build_vamana(argc, argv);
    }
    if (mode == "laser") {
      return build_laser(argc, argv);
    }
    std::cerr << "unknown mode: " << mode << '\n';
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "laser_fixture_builder: " << error.what() << '\n';
    return 1;
  }
}
