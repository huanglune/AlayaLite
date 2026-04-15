/**
 * @file build_vamana_graph.cpp
 * @brief Standalone tool to build a Vamana graph using AlayaLite's ShardVamanaBuilder
 *        and export in the official .vamana format for use with laser_build_bench.
 *
 * The core build+export logic lives in alaya::bench::build_vamana_from_raw(); this
 * program is a thin CLI wrapper that is also shared with laser_build_bench as its
 * internal vamana step.
 *
 * Usage:
 *   ./build_vamana_graph <base_fvecs> <output.vamana>
 *       [max_degree=64] [ef_construction=128] [alpha=1.2]
 *       [num_threads=0] [max_memory_mb=8192]
 */

// NOLINTBEGIN

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

#include "laser_bench_common.hpp"
#include "utils/timer.hpp"

auto main(int argc, char* argv[]) -> int {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <base_fvecs> <output.vamana>"
              << " [max_degree=64] [ef_construction=128] [alpha=1.2]"
              << " [num_threads=0] [max_memory_mb=8192]\n";
    return 1;
  }

  std::string base_path = argv[1];
  std::string output_path = argv[2];

  alaya::bench::VamanaBuildOptions opts;
  try {
    if (argc > 3) opts.max_degree_ = static_cast<uint32_t>(std::stoul(argv[3]));
    if (argc > 4) opts.ef_construction_ = static_cast<uint32_t>(std::stoul(argv[4]));
    if (argc > 5) opts.alpha_ = std::stof(argv[5]);
    if (argc > 6) opts.num_threads_ = static_cast<uint32_t>(std::stoul(argv[6]));
    if (argc > 7) opts.max_memory_mb_ = static_cast<size_t>(std::stoull(argv[7]));
  } catch (const std::exception& e) {
    std::cerr << "Invalid numeric argument: " << e.what() << '\n';
    return 1;
  }

  std::cout << "=== AlayaLite Vamana Graph Builder ===\n";

  alaya::Timer t_total;
  auto stats = alaya::bench::build_vamana_from_raw(base_path, output_path, opts);
  auto total_s = t_total.elapsed_s();

  double vec_per_s =
      static_cast<double>(stats.num_vectors_) / std::max(stats.build_time_s_, 1e-9);

  std::cout << "\n=== Graph Stats ===\n"
            << "  Avg degree : " << std::fixed << std::setprecision(2) << stats.avg_degree_
            << "\n"
            << "  Min degree : " << stats.min_observed_degree_ << "\n"
            << "  Max degree : " << stats.max_observed_degree_ << "\n"
            << "  Load time  : " << std::setprecision(1) << stats.load_time_s_ << " s\n"
            << "  Build time : " << stats.build_time_s_ << " s (" << std::setprecision(0)
            << vec_per_s << " vec/s)\n"
            << "  Export time: " << std::setprecision(1) << stats.export_time_s_ << " s\n"
            << "  Total time : " << total_s << " s\n"
            << "  Output     : " << output_path << " ("
            << (stats.output_bytes_ / (1024UL * 1024UL)) << " MiB)\n";

  return 0;
}

// NOLINTEND
