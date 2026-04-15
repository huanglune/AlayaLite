/**
 * @file laser_build_bench.cpp
 * @brief End-to-end LASER build + search benchmark with memory budget control.
 *
 * Runs Vamana → PCA → Medoid → QG internally using the shared in-process Vamana
 * builder, then benchmarks search QPS/recall/latency. Peak RSS stays within the
 * LASER-phase max_memory budget (the Vamana build has its own separate budget).
 *
 * Usage:
 *   ./laser_build_bench <base_fvecs> <query_fvecs> <gt_ivecs> <output_dir>
 *       [max_memory_mib=1024] [max_degree=64] [main_dim=256]
 *       [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]
 *       [--force-build] [--search-only]
 *       [--vamana-mem-mib=8192] [--vamana-ef=200] [--vamana-alpha=1.2]
 */

// NOLINTBEGIN

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "index/laser/laser_builder.hpp"
#include "index/laser/laser_index.hpp"
#include "laser_bench_common.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"
#include "utils/vector_file_reader.hpp"

// ============================================================================
// Helpers
// ============================================================================

namespace {

auto get_index_file_path(const std::filesystem::path &prefix,
                         uint32_t max_degree,
                         uint32_t main_dim) -> std::filesystem::path {
  return prefix.string() + "_R" + std::to_string(max_degree) + "_MD" + std::to_string(main_dim) +
         ".index";
}

auto get_required_index_files(const std::filesystem::path &prefix,
                              uint32_t max_degree,
                              uint32_t main_dim) -> std::vector<std::filesystem::path> {
  auto index_file = get_index_file_path(prefix, max_degree, main_dim);
  return {
      index_file,
      index_file.string() + "_rotator",
      prefix.string() + "_medoids",
      prefix.string() + "_medoids_indices",
      prefix.string() + "_pca.bin",
  };
}

auto has_reusable_index(const std::filesystem::path &prefix,
                        uint32_t max_degree,
                        uint32_t main_dim,
                        std::vector<std::filesystem::path> &missing_files) -> bool {
  missing_files.clear();
  for (const auto &path : get_required_index_files(prefix, max_degree, main_dim)) {
    if (!std::filesystem::exists(path)) {
      missing_files.push_back(path);
    }
  }
  return missing_files.empty();
}

}  // namespace

// ============================================================================
// Main
// ============================================================================

auto main(int argc, char *argv[]) -> int {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " <base_fvecs> <query_fvecs> <gt_ivecs> <output_dir>"
              << " [max_memory_mib=1024] [max_degree=64] [main_dim=256]"
              << " [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]"
              << " [--force-build] [--search-only]"
              << " [--vamana-mem-mib=8192] [--vamana-ef=200] [--vamana-alpha=1.2]\n";
    return 1;
  }

  enum class BuildMode : uint8_t {
    kAuto,
    kForceBuild,
    kSearchOnly,
  };

  BuildMode build_mode = BuildMode::kAuto;
  std::string base_path = argv[1];
  std::string query_path = argv[2];
  std::string gt_path = argv[3];
  std::string output_dir = argv[4];

  size_t vamana_mem_mib = 8192;
  uint32_t vamana_ef = 200;
  float vamana_alpha = 1.2F;
  std::vector<std::string> numeric_args;
  numeric_args.reserve(static_cast<size_t>(argc));
  for (int i = 5; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--force-build") {
      if (build_mode == BuildMode::kSearchOnly) {
        std::cerr << "Invalid arguments: --force-build and --search-only cannot be used together\n";
        return 1;
      }
      build_mode = BuildMode::kForceBuild;
      continue;
    }
    if (arg == "--search-only") {
      if (build_mode == BuildMode::kForceBuild) {
        std::cerr << "Invalid arguments: --force-build and --search-only cannot be used together\n";
        return 1;
      }
      build_mode = BuildMode::kSearchOnly;
      continue;
    }
    constexpr std::string_view kVamanaMemFlag = "--vamana-mem-mib=";
    if (arg.starts_with(kVamanaMemFlag)) {
      try {
        vamana_mem_mib = static_cast<size_t>(std::stoull(arg.substr(kVamanaMemFlag.size())));
      } catch (const std::exception &e) {
        std::cerr << "Invalid --vamana-mem-mib value: " << arg << " (" << e.what() << ")\n";
        return 1;
      }
      continue;
    }
    constexpr std::string_view kVamanaEfFlag = "--vamana-ef=";
    if (arg.starts_with(kVamanaEfFlag)) {
      try {
        vamana_ef = static_cast<uint32_t>(std::stoul(arg.substr(kVamanaEfFlag.size())));
      } catch (const std::exception &e) {
        std::cerr << "Invalid --vamana-ef value: " << arg << " (" << e.what() << ")\n";
        return 1;
      }
      continue;
    }
    constexpr std::string_view kVamanaAlphaFlag = "--vamana-alpha=";
    if (arg.starts_with(kVamanaAlphaFlag)) {
      try {
        vamana_alpha = std::stof(arg.substr(kVamanaAlphaFlag.size()));
      } catch (const std::exception &e) {
        std::cerr << "Invalid --vamana-alpha value: " << arg << " (" << e.what() << ")\n";
        return 1;
      }
      continue;
    }
    numeric_args.push_back(std::move(arg));
  }

  uint32_t max_memory_mib =
      (numeric_args.size() > 0) ? static_cast<uint32_t>(std::stoul(numeric_args[0])) : 1024;
  uint32_t max_degree =
      (numeric_args.size() > 1) ? static_cast<uint32_t>(std::stoul(numeric_args[1])) : 64;
  uint32_t main_dim =
      (numeric_args.size() > 2) ? static_cast<uint32_t>(std::stoul(numeric_args[2])) : 256;
  uint32_t num_medoids =
      (numeric_args.size() > 3) ? static_cast<uint32_t>(std::stoul(numeric_args[3])) : 300;
  uint32_t num_threads =
      (numeric_args.size() > 4) ? static_cast<uint32_t>(std::stoul(numeric_args[4])) : 0;
  float dram_budget = (numeric_args.size() > 5) ? std::stof(numeric_args[5]) : 1.0F;
  if (num_threads == 0) num_threads = std::thread::hardware_concurrency();

  constexpr uint32_t kTopK = 10;
  constexpr size_t kWarmup = 2;
  constexpr size_t kRuns = 5;
  const std::vector<size_t> ef_values = {80, 100, 150, 200, 300, 500};

  std::filesystem::create_directories(output_dir);
  auto prefix = std::filesystem::path(output_dir) / "dsqg_gist";
  std::filesystem::path internal_vamana_path(prefix.string() + ".vamana.index");

  auto run_build = [&](bool force_vamana_rebuild) {
    std::cout << "\n=== Build ===" << '\n' << std::flush;
    alaya::Timer build_timer;

    bool need_vamana = force_vamana_rebuild || !std::filesystem::exists(internal_vamana_path);
    if (need_vamana) {
      std::cout << "\n--- Vamana build ---\n" << std::flush;
      alaya::Timer vamana_timer;
      alaya::bench::VamanaBuildOptions vamana_opts;
      vamana_opts.max_degree_ = max_degree;
      vamana_opts.ef_construction_ = vamana_ef;
      vamana_opts.alpha_ = vamana_alpha;
      vamana_opts.num_threads_ = num_threads;
      vamana_opts.max_memory_mb_ = vamana_mem_mib;
      alaya::bench::build_vamana_from_raw(base_path, internal_vamana_path, vamana_opts);
      std::cout << "  Vamana total: " << std::fixed << std::setprecision(1)
                << vamana_timer.elapsed_s() << " s\n";
      alaya::bench::print_rss("after vamana build");
    } else {
      std::cout << "  Reuse existing Vamana: " << internal_vamana_path.string() << '\n';
    }

    std::cout << "\n--- LASER build ---\n" << std::flush;
    alaya::LaserBuildParams build_params;
    build_params.main_dim_ = main_dim;
    build_params.max_degree_ = max_degree;
    build_params.num_medoids_ = num_medoids;
    build_params.max_memory_mb_ = max_memory_mib;
    build_params.num_threads_ = num_threads;
    build_params.external_vamana_ = internal_vamana_path.string();
    build_params.keep_intermediates_ = true;

    alaya::LaserBuilder builder(build_params);
    builder.build(base_path, prefix.string());

    std::cout << "\n=== Build Summary ===" << '\n';
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << build_timer.elapsed_s()
              << " s\n";
    std::cout << "  LASER budget: " << max_memory_mib << " MiB\n";
    std::cout << "  Vamana budget: " << vamana_mem_mib << " MiB\n";
    alaya::bench::print_rss("after build");
  };

  alaya::bench::print_rss("initial");
  std::cout << "\n=== Configuration ===" << '\n';
  std::cout << "  max_memory_mib : " << max_memory_mib << '\n';
  std::cout << "  vamana_mem_mib : " << vamana_mem_mib << '\n';
  std::cout << "  vamana_ef      : " << vamana_ef << '\n';
  std::cout << "  vamana_alpha   : " << std::fixed << std::setprecision(2) << vamana_alpha << '\n';
  std::cout << "  build_threads  : " << num_threads << '\n';
  std::cout << "  max_degree     : " << max_degree << '\n';
  std::cout << "  main_dim       : " << main_dim << '\n';
  std::cout << "  num_medoids    : " << num_medoids << '\n';
  std::cout << "  vamana_path    : " << internal_vamana_path.string() << '\n';
  std::cout << "  build_mode     : "
            << (build_mode == BuildMode::kForceBuild
                    ? "force-build"
                    : (build_mode == BuildMode::kSearchOnly ? "search-only" : "auto"))
            << '\n';

  std::vector<std::filesystem::path> missing_files;
  bool has_index = has_reusable_index(prefix, max_degree, main_dim, missing_files);
  bool reused_existing_index = false;
  if (build_mode == BuildMode::kForceBuild) {
    std::cout << "\n=== Build ===\n";
    std::cout << "  Force build mode enabled.\n";
    run_build(/*force_vamana_rebuild=*/true);
  } else if (has_index) {
    reused_existing_index = true;
    std::cout << "\n=== Build ===\n";
    std::cout << "  Reuse existing index artifacts. Skip build.\n";
    alaya::bench::print_rss("reuse existing index");
  } else if (build_mode == BuildMode::kSearchOnly) {
    std::cout << "\n=== Build ===\n";
    std::cout << "  Search-only mode requires a ready index, but files are missing:\n";
    for (const auto &path : missing_files) {
      std::cout << "    - " << path.string() << '\n';
    }
    return 1;
  } else {
    std::cout << "\n=== Build ===\n";
    std::cout << "  Existing index not complete, missing files:\n";
    for (const auto &path : missing_files) {
      std::cout << "    - " << path.string() << '\n';
    }
    run_build(/*force_vamana_rebuild=*/false);
  }

  // Read dimensions from the base file for search
  uint32_t num_base = 0;
  uint32_t full_dim = 0;
  alaya::FloatVectorFileReader base_reader;
  base_reader.open(base_path);
  num_base = base_reader.num_vectors();
  full_dim = base_reader.dim();

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
  auto gt_u32 = alaya::bench::to_u32_ground_truth(gtvecs);

  std::cout << "\n=== Loading Index ===" << '\n' << std::flush;
  alaya::LaserIndex index;
  symqg::LaserSearchParams search_params;
  search_params.ef_search = 200;
  search_params.num_threads = 1;
  search_params.beam_width = 16;
  search_params.search_dram_budget_gb = dram_budget;
  auto load_index = [&]() {
    alaya::Timer load_timer;
    index.load(prefix.string(), num_base, max_degree, main_dim, full_dim, search_params);
    std::cout << "  Load: " << std::setprecision(1) << load_timer.elapsed_ms() << " ms, "
              << "cache: " << index.cached_node_count() << " nodes ("
              << (index.cache_size_bytes() / (1024 * 1024)) << " MB)\n";
  };

  try {
    load_index();
  } catch (const std::exception &error) {
    if (build_mode == BuildMode::kSearchOnly) {
      std::cerr << "  Search-only mode failed to load index: " << error.what() << '\n';
      return 1;
    }
    if (!reused_existing_index) {
      throw;
    }
    std::cout << "  Reused index load failed: " << error.what() << '\n';
    std::cout << "  Fallback to rebuild, then retry load.\n";
    run_build(/*force_vamana_rebuild=*/true);
    load_index();
  }
  alaya::bench::print_rss("after index load");

  alaya::bench::SearchSweepOptions sweep_options;
  sweep_options.top_k = kTopK;
  sweep_options.warmup_queries = kWarmup;
  sweep_options.runs = kRuns;
  sweep_options.ef_values = ef_values;
  sweep_options.num_threads = 1;
  sweep_options.beam_width = 16;
  sweep_options.allow_batch_search = false;

  alaya::bench::print_search_table_header();
  auto rows = alaya::bench::run_search_sweep(
      index, qvecs, gt_u32, gt_k, search_params, sweep_options,
      [](const alaya::bench::SearchBenchRow &row) { alaya::bench::print_search_table_row(row); });
  alaya::bench::print_search_table_footer();

  alaya::bench::print_rss("final");
  return 0;
}

// NOLINTEND
