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
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "index/laser/laser_index.hpp"
#include "utils/timer.hpp"

// ============================================================================
// RSS monitor — background thread that polls /proc/self/status
// ============================================================================

static auto read_current_rss_kb() -> size_t {
    std::ifstream fin("/proc/self/status");
    if (!fin.is_open()) {
        return 0;
    }
    std::string line;
    while (std::getline(fin, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            return std::stoull(line.substr(6));
        }
    }
    return 0;
}

class RssMonitor {
  public:
    explicit RssMonitor(std::chrono::milliseconds interval)
        : interval_(interval) {}

    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread([this] { sample_loop(); });
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    /// Mark a named phase boundary (e.g. "load", "search ef=80").
    void mark_phase(const std::string& name) {
        size_t current = read_current_rss_kb();
        std::lock_guard<std::mutex> lock(mu_);
        phases_.push_back({name, current, peak_kb_.load(std::memory_order_acquire)});
    }

    void print_report() const {
        size_t peak = peak_kb_.load(std::memory_order_acquire);
        size_t samples = sample_count_.load(std::memory_order_acquire);

        std::cout << "\n=== RSS Monitor Report ===" << '\n';
        std::cout << "  Sampling interval: " << interval_.count() << " ms"
                  << "  |  Total samples: " << samples << '\n';

        {
            std::lock_guard<std::mutex> lock(mu_);
            if (!phases_.empty()) {
                std::cout << "\n  Phase snapshots:\n";
                std::cout << "  " << std::left << std::setw(28) << "Phase"
                          << std::setw(16) << "RSS (MB)"
                          << "Peak so far (MB)" << '\n';
                std::cout << "  " << std::string(60, '-') << '\n';
                for (const auto& p : phases_) {
                    std::cout << "  " << std::left << std::setw(28) << p.name_
                              << std::setw(16) << p.rss_kb_ / 1024
                              << p.peak_so_far_kb_ / 1024 << '\n';
                }
            }
        }

        std::cout << "\n  >>> Peak RSS: " << peak / 1024
                  << " MB (" << peak << " kB) <<<" << '\n';
        std::cout << std::string(28, '=') << '\n';
    }

    ~RssMonitor() { stop(); }

    RssMonitor(const RssMonitor&) = delete;
    auto operator=(const RssMonitor&) -> RssMonitor& = delete;
    RssMonitor(RssMonitor&&) = delete;
    auto operator=(RssMonitor&&) -> RssMonitor& = delete;

  private:
    struct PhaseSnapshot {
        std::string name_;
        size_t rss_kb_;
        size_t peak_so_far_kb_;
    };

    void sample_loop() {
        while (running_.load(std::memory_order_acquire)) {
            size_t rss = read_current_rss_kb();
            // Update peak via CAS loop
            size_t prev = peak_kb_.load(std::memory_order_relaxed);
            while (rss > prev &&
                   !peak_kb_.compare_exchange_weak(prev, rss, std::memory_order_release,
                                                   std::memory_order_relaxed)) {
            }
            sample_count_.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(interval_);
        }
    }

    std::chrono::milliseconds interval_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> peak_kb_{0};
    std::atomic<size_t> sample_count_{0};
    mutable std::mutex mu_;
    std::vector<PhaseSnapshot> phases_;
    std::thread thread_;
};

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
    constexpr size_t kWarmupQueries = 3;
    constexpr size_t kRuns = 5;

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

    // Start RSS monitor (samples every 10ms)
    constexpr auto kRssSampleInterval = std::chrono::milliseconds(10);
    RssMonitor rss_monitor(kRssSampleInterval);
    rss_monitor.start();
    rss_monitor.mark_phase("before load");

    alaya::LaserIndex index;
    symqg::LaserSearchParams params;
    params.ef_search = 200;
    params.num_threads = num_threads;
    params.beam_width = beam_width;
    params.search_dram_budget_gb = dram_budget;

    alaya::Timer load_timer;
    index.load(index_prefix, num_points, degree, main_dim, full_dim, params);
    double load_ms = load_timer.elapsed_ms();

    std::cout << "  Load time:  " << std::fixed << std::setprecision(1) << load_ms << " ms" << '\n';
    std::cout << "  Cached:     " << index.cached_node_count() << " nodes" << '\n';
    std::cout << "  Cache size: " << index.cache_size_bytes() / (1024 * 1024) << " MB" << '\n';
    std::cout << std::flush;
    rss_monitor.mark_phase("after load");

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
    std::cout << std::string(90, '-') << '\n' << std::flush;

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
                    alaya::Timer query_timer;
                    index.search(
                        queries.data() + i * qd, kTopK,
                        results.data() + i * kTopK
                    );
                    double us = query_timer.elapsed_us();
                    latencies.push_back(us);
                    total_time += us;
                }
                total_time /= 1e6;
            } else {
                alaya::Timer batch_timer;
                index.batch_search(
                    queries.data(), static_cast<size_t>(nq), kTopK, results.data()
                );
                total_time = batch_timer.elapsed_s();
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
                  << '\n'
                  << std::flush;

        rss_monitor.mark_phase("after search ef=" + std::to_string(ef));
    }

    std::cout << std::string(90, '=') << '\n' << std::flush;

    rss_monitor.mark_phase("after all searches");
    rss_monitor.stop();
    rss_monitor.print_report();

    return 0;
}
