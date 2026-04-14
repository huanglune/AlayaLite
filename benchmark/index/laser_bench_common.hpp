/**
 * @file laser_bench_common.hpp
 * @brief Shared utilities for LASER build/search benchmark programs.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "bench_utils.hpp"
#include "index/laser/laser_index.hpp"
#include "utils/evaluate.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"

namespace alaya::bench {

inline auto read_current_rss_kb() -> size_t {
  std::ifstream status("/proc/self/status");
  std::string key;
  size_t value_kb = 0;
  std::string unit;
  while (status >> key >> value_kb >> unit) {
    if (key == "VmRSS:") return value_kb;
  }
  return 0;
}

inline auto read_current_rss_mib() -> double {
  return static_cast<double>(read_current_rss_kb()) / 1024.0;
}

inline void print_rss(const std::string &label) {
  std::cout << "  [RSS] " << std::left << std::setw(32) << label << std::fixed
            << std::setprecision(1) << read_current_rss_mib() << " MiB\n"
            << std::flush;
}

class RssMonitor {
 public:
  explicit RssMonitor(std::chrono::milliseconds interval) : interval_(interval) {}

  void start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] {
      sample_loop();
    });
  }

  void stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  void mark_phase(const std::string &name) {
    std::lock_guard<std::mutex> lock(mu_);
    phases_.push_back({name, read_current_rss_kb(), peak_kb_.load(std::memory_order_acquire)});
  }

  void print_report() const {
    std::cout << "\n=== RSS Monitor Report ===\n";
    std::cout << "  Sampling interval: " << interval_.count() << " ms"
              << "  |  Total samples: " << sample_count_.load(std::memory_order_acquire) << '\n';

    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!phases_.empty()) {
        std::cout << "\n  Phase snapshots:\n";
        std::cout << "  " << std::left << std::setw(30) << "Phase" << std::setw(14) << "RSS (MiB)"
                  << "Peak so far (MiB)\n";
        std::cout << "  " << std::string(62, '-') << '\n';
        for (const auto &phase : phases_) {
          std::cout << "  " << std::left << std::setw(30) << phase.name_ << std::setw(14)
                    << (phase.rss_kb_ / 1024) << (phase.peak_so_far_kb_ / 1024) << '\n';
        }
      }
    }

    auto peak_kb = peak_kb_.load(std::memory_order_acquire);
    std::cout << "\n  >>> Peak RSS: " << peak_kb / 1024 << " MiB (" << peak_kb << " kB) <<<\n";
    std::cout << std::string(30, '=') << '\n';
  }

  ~RssMonitor() { stop(); }

  RssMonitor(const RssMonitor &) = delete;
  auto operator=(const RssMonitor &) -> RssMonitor & = delete;
  RssMonitor(RssMonitor &&) = delete;
  auto operator=(RssMonitor &&) -> RssMonitor & = delete;

 private:
  struct PhaseSnapshot {
    std::string name_;
    size_t rss_kb_;
    size_t peak_so_far_kb_;
  };

  void sample_loop() {
    while (running_.load(std::memory_order_acquire)) {
      size_t rss_kb = read_current_rss_kb();
      size_t prev = peak_kb_.load(std::memory_order_relaxed);
      while (rss_kb > prev && !peak_kb_.compare_exchange_weak(prev,
                                                              rss_kb,
                                                              std::memory_order_release,
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

struct SearchSweepOptions {
  uint32_t top_k = 10;
  size_t warmup_queries = 3;
  size_t runs = 5;
  std::vector<size_t> ef_values;
  size_t num_threads = 1;
  size_t beam_width = 16;
  bool allow_batch_search = true;
};

struct SearchBenchRow {
  size_t ef = 0;
  double qps = 0;
  double recall_pct = 0;
  bool has_latency = false;
  double mean_latency_us = 0;
  double p99_9_latency_us = 0;
};

inline auto to_u32_ground_truth(const VecFileData<int32_t> &gtvecs) -> std::vector<uint32_t> {
  std::vector<uint32_t> gt_u32(gtvecs.data_.size());
  std::transform(gtvecs.data_.begin(), gtvecs.data_.end(), gt_u32.begin(), [](int32_t value) {
    return static_cast<uint32_t>(value);
  });
  return gt_u32;
}

inline void print_search_table_header() {
  std::cout << "\n" << std::string(84, '=') << '\n';
  std::cout << std::left << std::setw(8) << "EF" << std::setw(14) << "QPS" << std::setw(14)
            << "Recall(%)" << std::setw(16) << "Mean(us)" << std::setw(16) << "P99.9(us)" << '\n';
  std::cout << std::string(84, '-') << '\n' << std::flush;
}

inline void print_search_table_row(const SearchBenchRow &row) {
  std::cout << std::left << std::setw(8) << row.ef << std::setw(14) << std::fixed
            << std::setprecision(1) << row.qps << std::setw(14) << std::setprecision(2)
            << row.recall_pct;

  if (row.has_latency) {
    std::cout << std::setw(16) << std::setprecision(1) << row.mean_latency_us << std::setw(16)
              << std::setprecision(1) << row.p99_9_latency_us;
  } else {
    std::cout << std::setw(16) << "N/A" << std::setw(16) << "N/A";
  }
  std::cout << '\n' << std::flush;
}

inline void print_search_table_footer() { std::cout << std::string(84, '=') << '\n' << std::flush; }

inline auto run_search_sweep(LaserIndex &index,
                             const VecFileData<float> &qvecs,
                             const std::vector<uint32_t> &gt_u32,
                             uint32_t gt_dim,
                             symqg::LaserSearchParams base_params,
                             const SearchSweepOptions &options,
                             const std::function<void(size_t)> &on_ef_complete = {})
    -> std::vector<SearchBenchRow> {
  std::vector<SearchBenchRow> rows;
  rows.reserve(options.ef_values.size());

  auto nq = qvecs.num_;
  auto qdim = qvecs.dim_;
  if (nq == 0 || qdim == 0 || gt_dim == 0 || options.top_k == 0 || options.runs == 0 ||
      options.ef_values.empty()) {
    return rows;
  }
  if (gt_u32.size() < static_cast<size_t>(nq) * gt_dim) {
    return rows;
  }

  const bool single_query_mode = (options.num_threads == 1) || !options.allow_batch_search;
  std::vector<uint32_t> results(static_cast<size_t>(nq) * options.top_k);
  std::vector<uint32_t> warmup_result(options.top_k);

  for (size_t ef : options.ef_values) {
    base_params.ef_search = ef;
    base_params.num_threads = options.num_threads;
    base_params.beam_width = options.beam_width;
    index.set_search_params(base_params);

    auto warmup_count = std::min(options.warmup_queries, static_cast<size_t>(nq));
    for (size_t warmup = 0; warmup < warmup_count; ++warmup) {
      index.search(qvecs.data_.data() + warmup * qdim, options.top_k, warmup_result.data());
    }

    double sum_qps = 0;
    double sum_mean_latency = 0;
    double sum_p99_9_latency = 0;
    double last_recall = 0;

    for (size_t run = 0; run < options.runs; ++run) {
      double total_time_s = 0;

      if (single_query_mode) {
        std::vector<double> latencies;
        latencies.reserve(nq);
        double total_latency_us = 0;
        for (uint32_t i = 0; i < nq; ++i) {
          alaya::Timer query_timer;
          index.search(qvecs.data_.data() + static_cast<size_t>(i) * qdim,
                       options.top_k,
                       results.data() + static_cast<size_t>(i) * options.top_k);
          double latency_us = query_timer.elapsed_us();
          latencies.push_back(latency_us);
          total_latency_us += latency_us;
        }
        total_time_s = total_latency_us / 1e6;
        sum_mean_latency += std::accumulate(latencies.begin(), latencies.end(), 0.0) / nq;
        sum_p99_9_latency += percentile(latencies, 99.9);
      } else {
        alaya::Timer batch_timer;
        index.batch_search(qvecs.data_.data(),
                           static_cast<size_t>(nq),
                           options.top_k,
                           results.data());
        total_time_s = batch_timer.elapsed_s();
      }

      if (total_time_s <= 0) {
        continue;
      }
      sum_qps += static_cast<double>(nq) / total_time_s;
      last_recall =
          alaya::calc_recall(results.data(), gt_u32.data(), nq, gt_dim, options.top_k) * 100.0;
    }

    rows.push_back(
        {.ef = ef,
         .qps = sum_qps / options.runs,
         .recall_pct = last_recall,
         .has_latency = single_query_mode,
         .mean_latency_us = single_query_mode ? (sum_mean_latency / options.runs) : 0.0,
         .p99_9_latency_us = single_query_mode ? (sum_p99_9_latency / options.runs) : 0.0});

    if (on_ef_complete) {
      on_ef_complete(ef);
    }
  }

  return rows;
}

}  // namespace alaya::bench
