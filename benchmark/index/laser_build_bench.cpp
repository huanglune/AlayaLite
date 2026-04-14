/**
 * @file laser_build_bench.cpp
 * @brief End-to-end LASER build + search benchmark with memory budget control.
 *
 * Runs PCA → Medoid → QG using io_uring disk-based access, then benchmarks
 * search QPS/recall/latency. Peak RSS stays within the max_memory budget.
 *
 * Usage:
 *   ./laser_build_bench <base_fvecs> <query_fvecs> <gt_ivecs>
 *       <vamana_index> <output_dir>
 *       [max_memory_mib=1024] [max_degree=64] [main_dim=256]
 *       [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]
 */

// NOLINTBEGIN

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "bench_utils.hpp"
#include "index/laser/laser_builder.hpp"
#include "index/laser/laser_index.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"
#include "utils/vector_file_reader.hpp"

// ============================================================================
// RSS measurement
// ============================================================================

static auto get_rss_mib() -> double {
  std::ifstream ifs("/proc/self/status");
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.starts_with("VmRSS:")) {
      size_t pos = line.find_first_of("0123456789");
      if (pos != std::string::npos) return std::stod(line.substr(pos)) / 1024.0;
    }
  }
  return 0.0;
}

static void print_rss(const char *label) {
  std::cout << "  [RSS] " << std::left << std::setw(35) << label << std::fixed
            << std::setprecision(1) << get_rss_mib() << " MiB\n"
            << std::flush;
}

// ============================================================================
// Main
// ============================================================================

auto main(int argc, char *argv[]) -> int {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " <base_fvecs> <query_fvecs> <gt_ivecs>"
              << " <vamana_index> <output_dir>"
              << " [max_memory_mib=1024] [max_degree=64] [main_dim=256]"
              << " [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]\n";
    return 1;
  }

  std::string base_path = argv[1];
  std::string query_path = argv[2];
  std::string gt_path = argv[3];
  std::string vamana_path = argv[4];
  std::string output_dir = argv[5];
  uint32_t max_memory_mib = (argc > 6) ? static_cast<uint32_t>(std::stoul(argv[6])) : 1024;
  uint32_t max_degree = (argc > 7) ? static_cast<uint32_t>(std::stoul(argv[7])) : 64;
  uint32_t main_dim = (argc > 8) ? static_cast<uint32_t>(std::stoul(argv[8])) : 256;
  uint32_t num_medoids = (argc > 9) ? static_cast<uint32_t>(std::stoul(argv[9])) : 300;
  uint32_t num_threads = (argc > 10) ? static_cast<uint32_t>(std::stoul(argv[10])) : 0;
  float dram_budget = (argc > 11) ? std::stof(argv[11]) : 1.0F;
  if (num_threads == 0) num_threads = std::thread::hardware_concurrency();

  constexpr uint32_t kTopK = 10;
  constexpr size_t kWarmup = 2;
  constexpr size_t kRuns = 5;
  std::vector<size_t> ef_values = {80, 100, 150, 200, 300, 500};

  std::filesystem::create_directories(output_dir);
  auto prefix = std::filesystem::path(output_dir) / "dsqg_gist";

  print_rss("initial");
  std::cout << "\n=== Configuration ===" << '\n';
  std::cout << "  max_memory: " << max_memory_mib << " MiB, threads: " << num_threads << '\n';
  std::cout << "  max_degree: " << max_degree << ", main_dim: " << main_dim
            << ", medoids: " << num_medoids << '\n';

  std::cout << "\n=== Build ===" << '\n' << std::flush;
  alaya::Timer t_build;

  alaya::LaserBuildParams build_params;
  build_params.main_dim_ = main_dim;
  build_params.max_degree_ = max_degree;
  build_params.num_medoids_ = num_medoids;
  build_params.max_memory_mb_ = max_memory_mib;
  build_params.num_threads_ = num_threads;
  build_params.external_vamana_ = vamana_path;
  build_params.keep_intermediates_ = true;

  alaya::LaserBuilder builder(build_params);
  builder.build(base_path, prefix.string());

  // Read dimensions from the base file for search
  uint32_t num_base = 0;
  uint32_t full_dim = 0;
  alaya::FloatVectorFileReader base_reader;
  base_reader.open(base_path);
  num_base = base_reader.num_vectors();
  full_dim = base_reader.dim();

  double build_s = t_build.elapsed_s();
  std::cout << "\n=== Build Summary ===" << '\n';
  std::cout << "  Total: " << std::fixed << std::setprecision(1) << build_s << " s\n";
  std::cout << "  Budget: " << max_memory_mib << " MiB\n";
  print_rss("after build");

  // ================================================================
  // Search Phase
  // ================================================================
  std::cout << "\n=== Loading Queries & Ground Truth ===" << '\n' << std::flush;
  auto qvecs = alaya::load_float_vectors(query_path);
  auto gtvecs = alaya::load_int_vectors(gt_path);
  auto nq = qvecs.num_;
  auto qdim = qvecs.dim_;
  auto gt_k = gtvecs.dim_;
  std::cout << "  Queries: " << nq << " x " << qdim << ", GT: " << gtvecs.num_ << " x " << gt_k
            << '\n';

  std::cout << "\n=== Loading Index ===" << '\n' << std::flush;
  alaya::LaserIndex index;
  symqg::LaserSearchParams sp;
  sp.ef_search = 200;
  sp.num_threads = 1;
  sp.beam_width = 16;
  sp.search_dram_budget_gb = dram_budget;
  alaya::Timer t_load;
  index.load(prefix.string(), num_base, max_degree, main_dim, full_dim, sp);
  std::cout << "  Load: " << std::setprecision(1) << t_load.elapsed_ms() << " ms, "
            << "cache: " << index.cached_node_count() << " nodes ("
            << (index.cache_size_bytes() / (1024 * 1024)) << " MB)\n";
  print_rss("after index load");

  std::cout << "\n" << std::string(90, '=') << '\n';
  std::cout << std::left << std::setw(10) << "EF" << std::setw(15) << "QPS" << std::setw(15)
            << "Recall(%)" << std::setw(18) << "Mean(us)" << std::setw(18) << "P99.9(us)" << '\n';
  std::cout << std::string(90, '-') << '\n' << std::flush;

  for (size_t ef : ef_values) {
    sp.ef_search = ef;
    sp.num_threads = 1;
    sp.beam_width = 16;
    index.set_search_params(sp);

    std::vector<uint32_t> results(static_cast<size_t>(nq) * kTopK);

    // Warmup
    for (size_t w = 0; w < std::min(kWarmup, static_cast<size_t>(nq)); ++w) {
      uint32_t tmp[kTopK];
      index.search(qvecs.data_.data() + w * qdim, kTopK, tmp);
    }

    double sum_qps = 0, sum_lat = 0, sum_p99 = 0;
    double last_recall = 0;
    for (size_t run = 0; run < kRuns; ++run) {
      std::vector<double> lats;
      lats.reserve(nq);
      double total = 0;
      for (uint32_t i = 0; i < nq; ++i) {
        alaya::Timer qt;
        index.search(qvecs.data_.data() + static_cast<size_t>(i) * qdim,
                     kTopK,
                     results.data() + static_cast<size_t>(i) * kTopK);
        double us = qt.elapsed_us();
        lats.push_back(us);
        total += us;
      }
      total /= 1e6;
      sum_qps += nq / total;
      sum_lat += std::accumulate(lats.begin(), lats.end(), 0.0) / nq;
      sum_p99 += alaya::bench::percentile(lats, 99.9);
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
      auto *gt_u32 = reinterpret_cast<const uint32_t *>(gtvecs.data_.data());
      last_recall = alaya::calc_recall(results.data(), gt_u32, nq, gt_k, kTopK) * 100.0;
    }

    std::cout << std::left << std::setw(10) << ef << std::setw(15) << std::fixed
              << std::setprecision(1) << (sum_qps / kRuns) << std::setw(15) << std::setprecision(2)
              << last_recall << std::setw(18) << std::setprecision(1) << (sum_lat / kRuns)
              << std::setw(18) << (sum_p99 / kRuns) << '\n'
              << std::flush;
  }
  std::cout << std::string(90, '=') << '\n';
  print_rss("final");
  return 0;
}

// NOLINTEND
