/**
 * @file laser_build_bench.cpp
 * @brief End-to-end LASER build + search benchmark with memory budget control.
 *
 * Runs PCA → Vamana → Medoid → QG internally. By default the Vamana graph is
 * built on the PCA-transformed base (matches the Laser official pipeline).
 * Use --external-vamana to fall back to building on raw vectors instead.
 *
 * Usage:
 *   ./laser_build_bench <base_fvecs> <query_fvecs> <gt_ivecs> <output_dir>
 *       [max_memory_mib=1024] [max_degree=64] [main_dim=256]
 *       [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]
 *       [--force-build] [--search-only]
 *       [--vamana-mem-mib=8192] [--vamana-ef=200] [--vamana-alpha=1.2]
 *       [--search-threads=1] [--search-batch]
 *       [--builtin-vamana | --external-vamana]
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
  // _pca.bin is treated as optional at load time (load_optional_pca_from_disk),
  // so keep it out of the required-files check so pre-PCA indices also load.
  return {
      index_file,
      index_file.string() + "_rotator",
      prefix.string() + "_medoids",
      prefix.string() + "_medoids_indices",
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
              << " [--vamana-mem-mib=8192] [--vamana-ef=200] [--vamana-alpha=1.2]"
              << " [--search-threads=1] [--search-batch]"
              << " [--builtin-vamana|--external-vamana]\n";
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
  uint32_t search_threads = 1;
  bool search_batch = false;
  // Default to PCA-space Vamana (matches Laser official pipeline). Use
  // --external-vamana to fall back to building on raw vectors.
  bool builtin_vamana = true;
  std::string external_vamana_path;  // --vamana-index=<path>: skip build, reuse
  std::string external_pca_base_path;    // --pca-base=<path>: Laser's _pca_base.fbin
  std::string external_pca_matrix_path;  // --pca-matrix=<path>: Laser's _pca.bin
  std::string external_rotator_path;     // --rotator=<path>: reuse prebuilt QG rotator
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
    constexpr std::string_view kSearchThreadsFlag = "--search-threads=";
    if (arg.starts_with(kSearchThreadsFlag)) {
      try {
        auto value = static_cast<uint32_t>(std::stoul(arg.substr(kSearchThreadsFlag.size())));
        if (value == 0) {
          std::cerr << "Invalid --search-threads value: " << arg
                    << " (must be >= 1, explicit thread count required)\n";
          return 1;
        }
        search_threads = value;
      } catch (const std::exception &e) {
        std::cerr << "Invalid --search-threads value: " << arg << " (" << e.what() << ")\n";
        return 1;
      }
      continue;
    }
    if (arg == "--search-batch") {
      search_batch = true;
      continue;
    }
    if (arg == "--builtin-vamana") {
      builtin_vamana = true;
      continue;
    }
    if (arg == "--external-vamana") {
      builtin_vamana = false;
      continue;
    }
    constexpr std::string_view kVamanaIndexFlag = "--vamana-index=";
    if (arg.starts_with(kVamanaIndexFlag)) {
      external_vamana_path = arg.substr(kVamanaIndexFlag.size());
      builtin_vamana = false;  // force external path
      continue;
    }
    constexpr std::string_view kPcaBaseFlag = "--pca-base=";
    if (arg.starts_with(kPcaBaseFlag)) {
      external_pca_base_path = arg.substr(kPcaBaseFlag.size());
      continue;
    }
    constexpr std::string_view kPcaMatrixFlag = "--pca-matrix=";
    if (arg.starts_with(kPcaMatrixFlag)) {
      external_pca_matrix_path = arg.substr(kPcaMatrixFlag.size());
      continue;
    }
    constexpr std::string_view kRotatorFlag = "--rotator=";
    if (arg.starts_with(kRotatorFlag)) {
      external_rotator_path = arg.substr(kRotatorFlag.size());
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
  // Match Laser baseline methodology (see /tmp/dataset_sweep_20260415/laser_baselines.json):
  // 10 warmup iterations + 30 measurement runs -> tighter QPS estimate, fair compare.
  constexpr size_t kWarmup = 10;
  constexpr size_t kWarmupBatches = 10;
  constexpr size_t kRuns = 30;
  const std::vector<size_t> ef_values = {80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500};

  std::filesystem::create_directories(output_dir);
  auto prefix = std::filesystem::path(output_dir) / "dsqg_gist";
  std::filesystem::path internal_vamana_path(prefix.string() + ".vamana.index");

  auto run_build = [&](bool force_vamana_rebuild) {
    std::cout << "\n=== Build ===" << '\n' << std::flush;
    alaya::Timer build_timer;

    // Optionally stage externally-provided PCA artifacts into output_dir so
    // LaserBuilder.run_pca() detects them and skips retraining. Both files
    // must be present (matrix + transformed base) for the skip to trigger.
    if (!external_pca_base_path.empty()) {
      std::filesystem::path target_pca_base(prefix.string() + "_pca_base.fbin");
      std::cout << "\n--- PCA (external) ---\n"
                << "  Base:   " << external_pca_base_path << " -> "
                << target_pca_base.string() << '\n';
      std::filesystem::copy_file(external_pca_base_path, target_pca_base,
                                 std::filesystem::copy_options::overwrite_existing);
      if (!external_pca_matrix_path.empty()) {
        std::filesystem::path target_pca_matrix(prefix.string() + "_pca.bin");
        std::cout << "  Matrix: " << external_pca_matrix_path << " -> "
                  << target_pca_matrix.string() << '\n';
        std::filesystem::copy_file(external_pca_matrix_path, target_pca_matrix,
                                   std::filesystem::copy_options::overwrite_existing);
      } else {
        std::cout << "  [WARN] no --pca-matrix provided; queries must be PCA-transformed\n";
      }
    }

    alaya::LaserBuildParams build_params;
    build_params.main_dim_ = main_dim;
    build_params.max_degree_ = max_degree;
    build_params.ef_construction_ = vamana_ef;
    build_params.alpha_ = vamana_alpha;
    build_params.num_medoids_ = num_medoids;
    build_params.max_memory_mb_ = max_memory_mib;
    build_params.num_threads_ = num_threads;
    build_params.keep_intermediates_ = true;

    if (!external_rotator_path.empty()) {
      std::cout << "\n--- QG rotator (external) ---\n"
                << "  Source: " << external_rotator_path << '\n';
      if (!std::filesystem::exists(external_rotator_path)) {
        throw std::runtime_error("--rotator path not found: " + external_rotator_path);
      }
      build_params.external_rotator_ = external_rotator_path;
    }

    if (builtin_vamana) {
      build_params.vamana_max_memory_mb_ = vamana_mem_mib;
    } else if (!external_vamana_path.empty()) {
      std::cout << "\n--- Vamana (preconstructed) ---\n"
                << "  Source: " << external_vamana_path << "\n" << std::flush;
      if (!std::filesystem::exists(external_vamana_path)) {
        throw std::runtime_error("--vamana-index path not found: " + external_vamana_path);
      }
      // Let LaserBuilder's copy_external_vamana handle the copy into output_dir
      build_params.external_vamana_ = external_vamana_path;
    } else {
      bool need_vamana = force_vamana_rebuild || !std::filesystem::exists(internal_vamana_path);
      if (need_vamana) {
        std::cout << "\n--- Vamana build (external) ---\n" << std::flush;
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
      build_params.external_vamana_ = internal_vamana_path.string();
    }

    std::cout << "\n--- LASER build ---\n" << std::flush;
    alaya::LaserBuilder builder(build_params);
    builder.build(base_path, prefix.string());

    std::cout << "\n=== Build Summary ===" << '\n';
    std::cout << "  Total: " << std::fixed << std::setprecision(1) << build_timer.elapsed_s()
              << " s\n";
    std::cout << "  LASER budget: " << max_memory_mib << " MiB\n";
    std::cout << "  Vamana budget: " << vamana_mem_mib << " MiB\n";
    std::cout << "  Builtin vamana: " << (builtin_vamana ? "true" : "false") << '\n';
    alaya::bench::print_rss("after build");
  };

  alaya::bench::print_rss("initial");
  std::cout << "\n=== Configuration ===" << '\n';
  std::cout << "  max_memory_mib : " << max_memory_mib << '\n';
  std::cout << "  vamana_mem_mib : " << vamana_mem_mib << '\n';
  std::cout << "  vamana_ef      : " << vamana_ef << '\n';
  std::cout << "  vamana_alpha   : " << std::fixed << std::setprecision(2) << vamana_alpha << '\n';
  std::cout << "  build_threads  : " << num_threads << '\n';
  std::cout << "  search_threads : " << search_threads << '\n';
  std::cout << "  search_batch   : " << (search_batch ? "true" : "false") << '\n';
  std::cout << "  max_degree     : " << max_degree << '\n';
  std::cout << "  main_dim       : " << main_dim << '\n';
  std::cout << "  num_medoids    : " << num_medoids << '\n';
  std::cout << "  vamana_path    : " << internal_vamana_path.string() << '\n';
  std::cout << "  builtin_vamana : " << (builtin_vamana ? "true" : "false") << '\n';
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
  search_params.num_threads = search_threads;
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
  sweep_options.top_k_ = kTopK;
  sweep_options.warmup_queries_ = kWarmup;
  sweep_options.warmup_batches_ = kWarmupBatches;
  sweep_options.runs_ = kRuns;
  sweep_options.ef_values_ = ef_values;
  sweep_options.num_threads_ = search_threads;
  sweep_options.beam_width_ = 16;
  sweep_options.allow_batch_search_ = search_batch;

  alaya::bench::print_search_table_header();
  auto rows = alaya::bench::run_search_sweep(
      index, qvecs, gt_u32, gt_k, search_params, sweep_options,
      [](const alaya::bench::SearchBenchRow &row) { alaya::bench::print_search_table_row(row); });
  alaya::bench::print_search_table_footer();

  alaya::bench::print_rss("final");
  return 0;
}

// NOLINTEND
