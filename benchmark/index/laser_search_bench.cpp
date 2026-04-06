/**
 * @file laser_search_bench.cpp
 * @brief Benchmark for Laser index search performance.
 *
 * Loads a pre-built Laser index and measures QPS, recall, and latency
 * across a range of ef_search values, following the same benchmark
 * methodology as the standalone Laser project.
 *
 * Usage:
 *   ./laser_search_bench <index_prefix> <query_file> <gt_file> \
 *       <num_points> <degree> <main_dim> <full_dim> \
 *       [num_threads] [beam_width] [dram_budget_gb]
 *
 * Example (GIST):
 *   ./laser_search_bench \
 *       /path/to/data/gist/dsqg_gist \
 *       /path/to/gist/gist_query.fbin \
 *       /path/to/gist/gist_gt.ibin \
 *       1000000 64 256 960
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "index/laser/laser_index.hpp"

// ============================================================================
// Binary file I/O (compatible with Laser's .fbin/.ibin format)
// ============================================================================

struct FbinHeader {
    int32_t num_vectors_;
    int32_t dimension_;
};

static auto read_fbin(const std::string& path, int32_t& n, int32_t& d) -> std::vector<float> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    fin.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<float> data(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(sizeof(float) * n * d));
    return data;
}

static auto read_ibin(const std::string& path, int32_t& n, int32_t& d) -> std::vector<int32_t> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    fin.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    std::vector<int32_t> data(static_cast<size_t>(n) * d);
    fin.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(sizeof(int32_t) * n * d));
    return data;
}

// ============================================================================
// Recall computation
// ============================================================================

static auto compute_recall(
    const uint32_t* results, const int32_t* gt,
    size_t num_queries, size_t k, size_t gt_k
)-> double {
    size_t total_correct = 0;
    for (size_t i = 0; i < num_queries; ++i) {
        for (size_t j = 0; j < k; ++j) {
            uint32_t res_id = results[i * k + j];
            for (size_t g = 0; g < std::min(k, gt_k); ++g) {
                if (static_cast<int32_t>(res_id) == gt[i * gt_k + g]) {
                    ++total_correct;
                    break;
                }
            }
        }
    }
    return static_cast<double>(total_correct) / (num_queries * k) * 100.0;
}

// ============================================================================
// Percentile computation
// ============================================================================

static auto percentile(std::vector<double>& data, double p) -> double {
    std::sort(data.begin(), data.end());
    double idx = p / 100.0 * (data.size() - 1);
    auto lo = static_cast<size_t>(idx);
    auto hi = lo + 1;
    if (hi >= data.size()) {
        return data.back();
    }
    double frac = idx - lo;
    return data[lo] * (1.0 - frac) + data[hi] * frac;
}

// ============================================================================
// Main benchmark
// ============================================================================

auto main(int argc, char* argv[]) -> int {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <index_prefix> <query_file> <gt_file>"
                  << " <num_points> <degree> <main_dim> <full_dim>"
                  << " [num_threads=1] [beam_width=16] [dram_budget_gb=1.0]\n";
        return 1;
    }

    // Parse arguments
    std::string index_prefix = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];
    size_t num_points = std::stoull(argv[4]);
    size_t degree = std::stoull(argv[5]);
    size_t main_dim = std::stoull(argv[6]);
    size_t full_dim = std::stoull(argv[7]);
    size_t num_threads = argc > 8 ? std::stoull(argv[8]) : 1;
    size_t beam_width = argc > 9 ? std::stoull(argv[9]) : 16;
    float dram_budget = argc > 10 ? std::stof(argv[10]) : 1.0F;

    constexpr uint32_t kTopK = 10;
    constexpr size_t kWarmupQueries = 100;
    constexpr size_t kRuns = 3;

    // ef_search sweep (matching Laser's settings.py)
    std::vector<size_t> ef_values = {80, 90, 100, 110, 130, 150, 200, 250, 300, 400, 500};

    // Load query vectors
    int32_t nq = 0;
    int32_t qd = 0;
    std::cout << "Loading queries: " << query_file << '\n';
    auto queries = read_fbin(query_file, nq, qd);
    std::cout << "  Queries: " << nq << " x " << qd << '\n';

    // Load ground truth
    int32_t gt_n = 0;
    int32_t gt_k = 0;
    std::cout << "Loading ground truth: " << gt_file << '\n';
    auto gt = read_ibin(gt_file, gt_n, gt_k);
    std::cout << "  Ground truth: " << gt_n << " x " << gt_k << '\n';

    // Load Laser index
    std::cout << "\n=== Loading Laser Index ===" << '\n';
    std::cout << "  Prefix:     " << index_prefix << '\n';
    std::cout << "  Points:     " << num_points << '\n';
    std::cout << "  Degree:     " << degree << '\n';
    std::cout << "  Main dim:   " << main_dim << '\n';
    std::cout << "  Full dim:   " << full_dim << '\n';
    std::cout << "  DRAM budget: " << dram_budget << " GB" << '\n';

    alaya::LaserIndex index;
    symqg::LaserSearchParams params;
    params.ef_search = 200;
    params.num_threads = num_threads;
    params.beam_width = beam_width;
    params.search_dram_budget_gb = dram_budget;

    auto load_start = std::chrono::high_resolution_clock::now();
    index.load(index_prefix, num_points, degree, main_dim, full_dim, params);
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(load_end - load_start).count();

    std::cout << "  Load time:  " << std::fixed << std::setprecision(1) << load_ms << " ms" << '\n';
    std::cout << "  Cached:     " << index.cached_node_count() << " nodes" << '\n';
    std::cout << "  Cache size: " << index.cache_size_bytes() / (1024 * 1024) << " MB" << '\n';

    bool single_search = (num_threads == 1);

    // Print header
    std::cout << "\n" << std::string(90, '=') << '\n';
    std::cout << std::left
              << std::setw(10) << "EF"
              << std::setw(15) << "QPS"
              << std::setw(15) << "Recall(%)"
              << std::setw(18) << "Mean Lat(us)"
              << std::setw(18) << "P99.9 Lat(us)"
              << '\n';
    std::cout << std::string(90, '-') << '\n';

    // Benchmark loop with warmup + multi-run averaging
    for (size_t ef : ef_values) {
        params.ef_search = ef;
        params.num_threads = num_threads;
        params.beam_width = beam_width;
        index.set_search_params(params);

        std::vector<uint32_t> results(static_cast<size_t>(nq) * kTopK);

        // Warmup: run a subset of queries to warm caches and page tables
        size_t warmup_count = std::min(kWarmupQueries, static_cast<size_t>(nq));
        for (size_t w = 0; w < warmup_count; ++w) {
            uint32_t tmp[kTopK];
            index.search(queries.data() + w * qd, kTopK, tmp);
        }

        // Multi-run: collect kRuns measurements and average
        double sum_qps = 0;
        double sum_mean_lat = 0;
        double sum_p99_9_lat = 0;
        double last_recall = 0;

        for (size_t run = 0; run < kRuns; ++run) {
            std::vector<double> latencies;
            double total_time = 0;

            if (single_search) {
                latencies.reserve(nq);
                for (int32_t i = 0; i < nq; ++i) {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    index.search(
                        queries.data() + i * qd, kTopK,
                        results.data() + i * kTopK
                    );
                    auto t2 = std::chrono::high_resolution_clock::now();
                    double us = std::chrono::duration<double, std::micro>(t2 - t1).count();
                    latencies.push_back(us);
                    total_time += us;
                }
                total_time /= 1e6;
            } else {
                auto t1 = std::chrono::high_resolution_clock::now();
                index.batch_search(
                    queries.data(), static_cast<size_t>(nq), kTopK, results.data()
                );
                auto t2 = std::chrono::high_resolution_clock::now();
                total_time = std::chrono::duration<double>(t2 - t1).count();
            }

            sum_qps += nq / total_time;
            if (single_search) {
                sum_mean_lat += std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;
                sum_p99_9_lat += percentile(latencies, 99.9);
            }

            // Recall from last run (deterministic, same across runs)
            last_recall = compute_recall(
                results.data(), gt.data(),
                static_cast<size_t>(nq), kTopK, static_cast<size_t>(gt_k)
            );
        }

        double avg_qps = sum_qps / kRuns;
        double avg_mean_lat = sum_mean_lat / kRuns;
        double avg_p99_9_lat = sum_p99_9_lat / kRuns;

        std::cout << std::left
                  << std::setw(10) << ef
                  << std::setw(15) << std::fixed << std::setprecision(1) << avg_qps
                  << std::setw(15) << std::setprecision(2) << last_recall
                  << std::setw(18) << std::setprecision(1) << avg_mean_lat
                  << std::setw(18) << std::setprecision(1) << avg_p99_9_lat
                  << '\n';
    }

    std::cout << std::string(90, '=') << '\n';
    return 0;
}
