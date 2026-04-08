/**
 * @file laser_build_bench.cpp
 * @brief End-to-end benchmark: build LASER index from raw vectors, then
 *        measure search QPS, recall, and latency.
 *
 * Usage:
 *   ./laser_build_bench <base_fvecs> <query_fvecs> <gt_ivecs> <output_dir> \
 *       [max_degree=64] [main_dim=256] [ef_construction=200] [ef_build=128] \
 *       [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]
 *
 * Example (GIST-1M):
 *   ./laser_build_bench \
 *       /path/to/gist_base.fvecs \
 *       /path/to/gist_query.fvecs \
 *       /path/to/gist_groundtruth.ivecs \
 *       /tmp/laser_gist_build
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "index/laser/laser_build_params.hpp"
#include "index/laser/laser_builder.hpp"
#include "index/laser/laser_index.hpp"
#include "space/raw_space.hpp"
#include "utils/timer.hpp"
#include "utils/types.hpp"

// ============================================================================
// Standard fvecs / ivecs I/O
// ============================================================================

static auto read_fvecs(const std::string& path) -> std::pair<std::vector<float>, std::pair<uint32_t, uint32_t>> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }

    // Read first vector's dimension to determine dim
    int32_t dim = 0;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (dim <= 0) {
        throw std::runtime_error("Invalid dimension in: " + path);
    }

    // Compute number of vectors from file size
    fin.seekg(0, std::ios::end);
    auto file_size = static_cast<size_t>(fin.tellg());
    size_t vec_size = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(float);
    if (file_size % vec_size != 0) {
        throw std::runtime_error("File size mismatch in: " + path);
    }
    auto num = static_cast<uint32_t>(file_size / vec_size);

    // Read all vectors
    std::vector<float> data(static_cast<size_t>(num) * dim);
    fin.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < num; ++i) {
        int32_t d = 0;
        fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(data.data() + static_cast<size_t>(i) * dim),
                 static_cast<std::streamsize>(dim * sizeof(float)));
    }
    return {data, {num, static_cast<uint32_t>(dim)}};
}

static auto read_ivecs(const std::string& path) -> std::pair<std::vector<int32_t>, std::pair<uint32_t, uint32_t>> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }

    int32_t dim = 0;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (dim <= 0) {
        throw std::runtime_error("Invalid dimension in: " + path);
    }

    fin.seekg(0, std::ios::end);
    auto file_size = static_cast<size_t>(fin.tellg());
    size_t vec_size = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(int32_t);
    if (file_size % vec_size != 0) {
        throw std::runtime_error("File size mismatch in: " + path);
    }
    auto num = static_cast<uint32_t>(file_size / vec_size);

    std::vector<int32_t> data(static_cast<size_t>(num) * dim);
    fin.seekg(0, std::ios::beg);
    for (uint32_t i = 0; i < num; ++i) {
        int32_t d = 0;
        fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(data.data() + static_cast<size_t>(i) * dim),
                 static_cast<std::streamsize>(dim * sizeof(int32_t)));
    }
    return {data, {num, static_cast<uint32_t>(dim)}};
}

// ============================================================================
// Recall computation
// ============================================================================

static auto compute_recall(const uint32_t* results, const int32_t* gt,
                           size_t num_queries, size_t k, size_t gt_k) -> double {
    size_t total_correct = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t j = 0; j < k; ++j) {
            auto res_id = static_cast<int32_t>(results[i * k + j]);
            for (size_t g = 0; g < std::min(k, gt_k); ++g) {
                if (res_id == gt[i * gt_k + g]) {
                    ++total_correct;
                    break;
                }
            }
        }
    }
    return static_cast<double>(total_correct) / static_cast<double>(num_queries * k) * 100.0;
}

// ============================================================================
// Percentile
// ============================================================================

static auto percentile(std::vector<double>& data, double p) -> double {
    std::sort(data.begin(), data.end());
    double idx = p / 100.0 * static_cast<double>(data.size() - 1);
    auto lo = static_cast<size_t>(idx);
    auto hi = lo + 1;
    if (hi >= data.size()) {
        return data.back();
    }
    double frac = idx - static_cast<double>(lo);
    return data[lo] * (1.0 - frac) + data[hi] * frac;
}

// ============================================================================
// Main
// ============================================================================

auto main(int argc, char* argv[]) -> int {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <base_fvecs> <query_fvecs> <gt_ivecs> <output_dir>"
                  << " [max_degree=64] [main_dim=256] [ef_construction=200]"
                  << " [ef_build=128] [num_medoids=300] [num_threads=0]"
                  << " [dram_budget_gb=1.0]\n";
        return 1;
    }

    std::string base_path = argv[1];
    std::string query_path = argv[2];
    std::string gt_path = argv[3];
    std::string output_dir = argv[4];

    // Check for flags anywhere in argv
    bool single_shard = false;
    std::string external_vamana;
    std::vector<std::string> positional_args;
    for (int i = 5; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--single-shard") {
            single_shard = true;
        } else if (arg.starts_with("--vamana=")) {
            external_vamana = arg.substr(9);
        } else {
            positional_args.emplace_back(arg);
        }
    }
    auto pos_arg = [&](size_t idx, const char* fallback) -> std::string {
        return idx < positional_args.size() ? positional_args[idx] : fallback;
    };

    uint32_t max_degree = static_cast<uint32_t>(std::stoul(pos_arg(0, "64")));
    uint32_t main_dim = static_cast<uint32_t>(std::stoul(pos_arg(1, "256")));
    uint32_t ef_construction = static_cast<uint32_t>(std::stoul(pos_arg(2, "200")));
    uint32_t ef_build = static_cast<uint32_t>(std::stoul(pos_arg(3, "128")));
    uint32_t num_medoids = static_cast<uint32_t>(std::stoul(pos_arg(4, "300")));
    uint32_t num_threads = static_cast<uint32_t>(std::stoul(pos_arg(5, "0")));
    float dram_budget = std::stof(pos_arg(6, "1.0"));

    constexpr uint32_t kTopK = 10;
    constexpr size_t kWarmupQueries = 2;
    constexpr size_t kRuns = 5;

    std::vector<size_t> ef_values = {80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500};

    // ================================================================
    // Phase 1: Load base vectors
    // ================================================================
    std::cout << "\n=== Phase 1: Loading Base Vectors ===" << '\n';
    auto [base_data, base_shape] = read_fvecs(base_path);
    auto [num_base, full_dim] = base_shape;
    std::cout << "  Base: " << num_base << " x " << full_dim << '\n';

    // ================================================================
    // Phase 2: Build LASER index
    // ================================================================
    std::cout << "\n=== Phase 2: Building LASER Index ===" << '\n';
    std::cout << "  max_degree:      " << max_degree << '\n';
    std::cout << "  main_dim:        " << main_dim << '\n';
    std::cout << "  ef_construction:  " << ef_construction << '\n';
    std::cout << "  ef_build:        " << ef_build << '\n';
    std::cout << "  num_medoids:     " << num_medoids << '\n';
    std::cout << "  num_threads:     " << (num_threads == 0 ? std::thread::hardware_concurrency() : num_threads)
              << (num_threads == 0 ? " (auto)" : "") << '\n';
    std::cout << "  single_shard:    " << (single_shard ? "true" : "false") << '\n';

    std::filesystem::create_directories(output_dir);
    auto output_prefix = std::filesystem::path(output_dir) / "dsqg_gist";

    auto space = std::make_shared<alaya::RawSpace<float, float, uint32_t>>(
        num_base, full_dim, alaya::MetricType::L2);
    space->fit(base_data.data(), num_base);

    alaya::LaserBuildParams build_params;
    build_params.max_degree_ = max_degree;
    build_params.main_dim_ = main_dim;
    build_params.ef_construction_ = ef_construction;
    build_params.ef_build_ = ef_build;
    build_params.num_medoids_ = num_medoids;
    build_params.num_threads_ = num_threads;
    build_params.single_shard_ = single_shard;
    build_params.external_vamana_ = external_vamana;
    if (!external_vamana.empty()) {
        std::cout << "  external_vamana: " << external_vamana << '\n';
    }
    if (single_shard) {
        // Full dataset needs more memory: vectors + neighbor table (with graph slack factor 1.3)
        constexpr float kGraphSlack = 1.3F;
        auto slack_degree = static_cast<size_t>(static_cast<float>(max_degree) * kGraphSlack);
        size_t needed_mb = (static_cast<size_t>(num_base) * full_dim * sizeof(float)
                            + static_cast<size_t>(num_base) * slack_degree * sizeof(uint32_t)
                            + static_cast<size_t>(num_base) * slack_degree * sizeof(float))
                           / (1024 * 1024) + 512;  // +512MB headroom
        // Account for 0.9 budget fraction in ShardVamanaBuilder
        needed_mb = static_cast<size_t>(static_cast<double>(needed_mb) / 0.9) + 1;
        build_params.max_memory_mb_ = std::max(build_params.max_memory_mb_, needed_mb);
    }

    alaya::LaserBuilder<alaya::RawSpace<float, float, uint32_t>> builder(space, build_params);

    alaya::Timer build_timer;
    builder.build(output_prefix);
    double build_sec = build_timer.elapsed_s();

    std::cout << "\n  Build time: " << std::fixed << std::setprecision(1) << build_sec << " s\n";

    // Free base data memory before search phase
    base_data.clear();
    base_data.shrink_to_fit();
    space.reset();

    // ================================================================
    // Phase 3: Load queries and ground truth
    // ================================================================
    std::cout << "\n=== Phase 3: Loading Queries & Ground Truth ===" << '\n';
    auto [query_data, query_shape] = read_fvecs(query_path);
    auto [num_queries, query_dim] = query_shape;
    std::cout << "  Queries: " << num_queries << " x " << query_dim << '\n';

    auto [gt_data, gt_shape] = read_ivecs(gt_path);
    auto [gt_n, gt_k] = gt_shape;
    std::cout << "  Ground truth: " << gt_n << " x " << gt_k << '\n';

    // ================================================================
    // Phase 4: Load built index and search
    // ================================================================
    std::cout << "\n=== Phase 4: Loading Built Index ===" << '\n';

    alaya::LaserIndex index;
    symqg::LaserSearchParams search_params;
    search_params.ef_search = 200;
    search_params.num_threads = 1;
    search_params.beam_width = 16;
    search_params.search_dram_budget_gb = dram_budget;

    alaya::Timer load_timer;
    index.load(output_prefix.string(), num_base, max_degree, main_dim, full_dim, search_params);
    double load_ms = load_timer.elapsed_ms();

    std::cout << "  Load time:  " << std::fixed << std::setprecision(1) << load_ms << " ms\n";
    std::cout << "  Cached:     " << index.cached_node_count() << " nodes\n";
    std::cout << "  Cache size: " << index.cache_size_bytes() / (1024 * 1024) << " MB\n";
    std::cout << std::flush;

    // ================================================================
    // Phase 5: Search benchmark
    // ================================================================
    std::cout << "\n" << std::string(90, '=') << '\n';
    std::cout << std::left
              << std::setw(10) << "EF"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall(%)"
              << std::setw(18) << "Mean Lat(us)"
              << std::setw(18) << "P99.9 Lat(us)"
              << '\n';
    std::cout << std::string(90, '-') << '\n' << std::flush;

    for (size_t ef_idx = 0; ef_idx < ef_values.size(); ++ef_idx) {
        size_t ef = ef_values[ef_idx];
        std::cout << "[Search] Measuring EF=" << ef
                  << " (" << (ef_idx + 1) << "/" << ef_values.size() << ")"
                  << ", runs=" << kRuns
                  << ", queries=" << num_queries << '\n'
                  << std::flush;

        search_params.ef_search = ef;
        search_params.num_threads = 1;
        search_params.beam_width = 16;
        index.set_search_params(search_params);

        std::vector<uint32_t> results(static_cast<size_t>(num_queries) * kTopK);

        // Warmup
        size_t warmup_count = std::min(kWarmupQueries, static_cast<size_t>(num_queries));
        for (size_t w = 0; w < warmup_count; ++w) {
            uint32_t tmp[kTopK];
            index.search(query_data.data() + w * query_dim, kTopK, tmp);
        }

        // Multi-run
        double sum_qps = 0;
        double sum_mean_lat = 0;
        double sum_p99_9_lat = 0;
        double last_recall = 0;

        for (size_t run = 0; run < kRuns; ++run) {
            std::vector<double> latencies;
            latencies.reserve(num_queries);
            double total_time = 0;

            for (uint32_t i = 0; i < num_queries; ++i) {
                alaya::Timer query_timer;
                index.search(query_data.data() + static_cast<size_t>(i) * query_dim,
                             kTopK,
                             results.data() + static_cast<size_t>(i) * kTopK);
                double us = query_timer.elapsed_us();
                latencies.push_back(us);
                total_time += us;
            }
            total_time /= 1e6;
            sum_qps += num_queries / total_time;
            sum_mean_lat += std::accumulate(latencies.begin(), latencies.end(), 0.0) / num_queries;
            sum_p99_9_lat += percentile(latencies, 99.9);

            last_recall = compute_recall(results.data(), gt_data.data(),
                                         num_queries, kTopK, gt_k);
        }

        std::cout << std::left
                  << std::setw(10) << ef
                  << std::setw(15) << std::fixed << std::setprecision(1) << (sum_qps / kRuns)
                  << std::setw(15) << std::setprecision(2) << last_recall
                  << std::setw(18) << std::setprecision(1) << (sum_mean_lat / kRuns)
                  << std::setw(18) << std::setprecision(1) << (sum_p99_9_lat / kRuns)
                  << '\n'
                  << std::flush;
    }

    std::cout << std::string(90, '=') << '\n' << std::flush;
    return 0;
}
