/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "index/graph/diskann/diskann_builder.hpp"
#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "index/graph/diskann/diskann_searcher.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"

namespace alaya {

// =============================================================================
// Test configuration
// =============================================================================

static constexpr uint32_t kDataNum = 10000;
static constexpr uint32_t kQueryNum = 100;
static constexpr uint32_t kDim = 128;
static constexpr uint32_t kGtTopk = 100;
static constexpr uint32_t kPQSubspaces = 8;

struct BenchResources {
  Dataset ds_;
  std::shared_ptr<RawSpace<>> space_;
  bool loaded_{false};

  void load() {
    if (loaded_) {
      return;
    }
    ds_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk));
    space_ = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), ds_.data_num_);
    loaded_ = true;
    LOG_INFO("BenchResources: {} vectors, dim={}, {} queries", ds_.data_num_, ds_.dim_, kQueryNum);
  }
};

static BenchResources g_bench;

static void remove_index_files(const std::string &base_path) {
  std::filesystem::remove(base_path + ".meta");
  std::filesystem::remove(base_path + ".data");
  std::filesystem::remove(base_path + ".pq");
}

// =============================================================================
// Benchmark fixture
// =============================================================================

class DiskANNBenchmark : public ::testing::Test {
 protected:
  static inline std::string index_path = "/tmp/diskann_bench.index";
  static inline bool index_built = false;

  static void SetUpTestSuite() {
    g_bench.load();

    if (!index_built) {
      remove_index_files(index_path);

      auto params = DiskANNBuildParams()
                        .set_max_degree(64)
                        .set_ef_construction(128)
                        .set_num_threads(std::thread::hardware_concurrency())
                        .set_pq_params(kPQSubspaces);

      Timer timer;
      DiskANNIndex<float, uint32_t>::build(g_bench.space_, index_path, params);
      LOG_INFO("Index built in {:.2f}s (PQ enabled, M={})", timer.elapsed_s(), kPQSubspaces);
      index_built = true;
    }
  }

  static void TearDownTestSuite() { remove_index_files(index_path); }
};

// =============================================================================
// Helper: run benchmark for given parameters, return {recall, qps}
// =============================================================================

static auto run_bench(DiskANNIndex<float, uint32_t> &index,
                      const Dataset &ds,
                      uint32_t topk,
                      const DiskANNSearchParams &params,
                      uint32_t warmup_rounds,
                      uint32_t bench_rounds) -> std::pair<float, double> {
  std::vector<uint32_t> results(kQueryNum * topk);
  std::vector<double> latencies;
  latencies.reserve(kQueryNum * bench_rounds);

  // Warmup: populate buffer pool cache and stabilize performance
  for (uint32_t r = 0; r < warmup_rounds; ++r) {
    for (uint32_t q = 0; q < kQueryNum; ++q) {
      const float *query = ds.queries_.data() + q * kDim;
      index.search(query, topk, results.data() + q * topk, params);
    }
  }

  // Timed benchmark
  Timer total_timer;
  for (uint32_t r = 0; r < bench_rounds; ++r) {
    for (uint32_t q = 0; q < kQueryNum; ++q) {
      const float *query = ds.queries_.data() + q * kDim;
      Timer query_timer;
      index.search(query, topk, results.data() + q * topk, params);
      latencies.push_back(query_timer.elapsed_us());
    }
  }
  double total_us = total_timer.elapsed_us();

  float recall =
      calc_recall(results.data(), ds.ground_truth_.data(), kQueryNum, ds.gt_dim_, topk);

  std::ranges::sort(latencies);
  auto n = latencies.size();
  double avg_us = total_us / static_cast<double>(n);
  double p50_us = latencies[n / 2];
  double p99_us = latencies[static_cast<size_t>(static_cast<double>(n) * 0.99)];
  double qps = static_cast<double>(n) / (total_us / 1e6);

  LOG_INFO("{:>6}  {:>10.4f}  {:>10.1f}  {:>10.1f}  {:>10.1f}  {:>10.1f}",
           params.ef_search_, recall, qps, avg_us, p50_us, p99_us);

  return {recall, qps};
}

// =============================================================================
// PQ two-phase search benchmark (main path)
// =============================================================================

TEST_F(DiskANNBenchmark, SearchPQ) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 3;
  constexpr uint32_t kBenchRounds = 5;

  const auto &ds = g_bench.ds_;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN PQ Search Benchmark (Two-Phase: PQ Nav + Rerank)");
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64, M={}",
           kDataNum, kDim, kQueryNum, kTopk, kPQSubspaces);
  LOG_INFO("============================================================");
  LOG_INFO("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
           "ef", "recall@10", "QPS", "avg(us)", "p50(us)", "p99(us)");
  LOG_INFO("------------------------------------------------------------");

  std::vector<uint32_t> ef_values = {32, 64, 128, 256};
  for (auto ef : ef_values) {
    DiskANNSearchParams params;
    params.set_ef_search(ef).set_pq_rerank(true, 4);

    auto [recall, qps] = run_bench(index, ds, kTopk, params, kWarmupRounds, kBenchRounds);
    EXPECT_GT(qps, 0.0);
  }

  LOG_INFO("============================================================");
}

// =============================================================================
// Disk-only search benchmark (fallback path, for comparison)
// =============================================================================

TEST_F(DiskANNBenchmark, SearchDisk) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 1;
  constexpr uint32_t kBenchRounds = 1;

  const auto &ds = g_bench.ds_;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN Disk-Only Search Benchmark (In-Memory Graph + Disk Vec)");
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64",
           kDataNum, kDim, kQueryNum, kTopk);
  LOG_INFO("============================================================");
  LOG_INFO("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
           "ef", "recall@10", "QPS", "avg(us)", "p50(us)", "p99(us)");
  LOG_INFO("------------------------------------------------------------");

  DiskANNSearchParams params;
  params.set_ef_search(64).set_pq_rerank(false);

  auto [recall, qps] = run_bench(index, ds, kTopk, params, kWarmupRounds, kBenchRounds);
  EXPECT_GT(qps, 0.0);
  LOG_INFO("============================================================");
}

// =============================================================================
// Memory-constrained benchmarks: index exceeds buffer pool capacity
// =============================================================================

// Helper: run benchmark with a specific buffer pool size
static void run_constrained_bench(const std::string &index_path,
                                  const Dataset &ds,
                                  size_t cache_capacity,
                                  uint32_t pct_over,
                                  bool use_pq) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path, cache_capacity);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 3;
  constexpr uint32_t kBenchRounds = 3;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN {} Benchmark — Data exceeds memory by {}%",
           use_pq ? "PQ" : "Disk-Only", pct_over);
  LOG_INFO("  cache_capacity={} pages ({}KB)", cache_capacity, cache_capacity * 4);
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64",
           kDataNum, kDim, kQueryNum, kTopk);
  LOG_INFO("============================================================");
  LOG_INFO("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
           "ef", "recall@10", "QPS", "avg(us)", "p50(us)", "p99(us)");
  LOG_INFO("------------------------------------------------------------");

  std::vector<uint32_t> ef_values = {32, 64, 128, 256};
  for (auto ef : ef_values) {
    DiskANNSearchParams params;
    params.set_ef_search(ef).set_pq_rerank(use_pq, 4);

    auto [recall, qps] = run_bench(index, ds, kTopk, params, kWarmupRounds, kBenchRounds);
    EXPECT_GT(qps, 0.0);
  }

  LOG_INFO("============================================================");
}

// Total blocks = ceil(10048 / 5) = 2010.
// 20% over: cache = 2010 / 1.2 = 1675
// 40% over: cache = 2010 / 1.4 = 1436

TEST_F(DiskANNBenchmark, SearchDiskOver20Pct) {
  run_constrained_bench(index_path, g_bench.ds_, 1675, 20, false);
}

TEST_F(DiskANNBenchmark, SearchDiskOver40Pct) {
  run_constrained_bench(index_path, g_bench.ds_, 1436, 40, false);
}

TEST_F(DiskANNBenchmark, SearchPQOver20Pct) {
  run_constrained_bench(index_path, g_bench.ds_, 1675, 20, true);
}

TEST_F(DiskANNBenchmark, SearchPQOver40Pct) {
  run_constrained_bench(index_path, g_bench.ds_, 1436, 40, true);
}

// =============================================================================
// Multi-threaded benchmark helper
// =============================================================================

static auto run_mt_bench(DiskANNIndex<float, uint32_t> &index,
                         const Dataset &ds,
                         uint32_t topk,
                         DiskANNSearchParams params,
                         uint32_t num_threads,
                         uint32_t warmup_rounds,
                         uint32_t bench_rounds) -> std::pair<float, double> {
  params.set_num_threads(num_threads);
  std::vector<uint32_t> results(kQueryNum * topk);

  // Warmup
  for (uint32_t r = 0; r < warmup_rounds; ++r) {
    index.batch_search(ds.queries_.data(), kQueryNum, topk, results.data(), params);
  }

  // Timed benchmark
  double total_us = 0;
  for (uint32_t r = 0; r < bench_rounds; ++r) {
    Timer round_timer;
    index.batch_search(ds.queries_.data(), kQueryNum, topk, results.data(), params);
    total_us += round_timer.elapsed_us();
  }

  float recall =
      calc_recall(results.data(), ds.ground_truth_.data(), kQueryNum, ds.gt_dim_, topk);
  uint64_t total_queries = static_cast<uint64_t>(kQueryNum) * bench_rounds;
  double qps = static_cast<double>(total_queries) / (total_us / 1e6);
  double avg_us = total_us / static_cast<double>(total_queries);

  LOG_INFO("{:>4}T  {:>6}  {:>10.4f}  {:>10.1f}  {:>10.1f}",
           num_threads, params.ef_search_, recall, qps, avg_us);

  return {recall, qps};
}

// =============================================================================
// Multi-threaded Disk-Only benchmarks
// =============================================================================

static void run_mt_disk_bench(const std::string &index_path,
                              const Dataset &ds,
                              size_t cache_capacity,
                              uint32_t pct_over) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path, cache_capacity);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 3;
  constexpr uint32_t kBenchRounds = 3;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN Disk-Only Multi-Thread — Data exceeds memory by {}%", pct_over);
  LOG_INFO("  cache_capacity={} pages ({}KB)", cache_capacity, cache_capacity * 4);
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64",
           kDataNum, kDim, kQueryNum, kTopk);
  LOG_INFO("============================================================");
  LOG_INFO("{:>4}   {:>6}  {:>10}  {:>10}  {:>10}",
           "ntr", "ef", "recall@10", "QPS", "avg(us)");
  LOG_INFO("------------------------------------------------------------");

  std::vector<uint32_t> thread_counts = {1, 2, 4, 8};

  for (auto nthreads : thread_counts) {
    DiskANNSearchParams params;
    params.set_ef_search(64).set_pq_rerank(false);

    auto [recall, qps] =
        run_mt_bench(index, ds, kTopk, params, nthreads, kWarmupRounds, kBenchRounds);
    EXPECT_GT(qps, 0.0);
  }

  LOG_INFO("============================================================");
}

TEST_F(DiskANNBenchmark, SearchDiskMT_Over20Pct) {
  run_mt_disk_bench(index_path, g_bench.ds_, 1675, 20);
}

TEST_F(DiskANNBenchmark, SearchDiskMT_Over40Pct) {
  run_mt_disk_bench(index_path, g_bench.ds_, 1436, 40);
}

// Full-cache multi-threaded disk-only (baseline)
TEST_F(DiskANNBenchmark, SearchDiskMT_FullCache) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 3;
  constexpr uint32_t kBenchRounds = 5;

  const auto &ds = g_bench.ds_;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN Disk-Only Multi-Thread — Full Cache (baseline)");
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64",
           kDataNum, kDim, kQueryNum, kTopk);
  LOG_INFO("============================================================");
  LOG_INFO("{:>4}   {:>6}  {:>10}  {:>10}  {:>10}",
           "ntr", "ef", "recall@10", "QPS", "avg(us)");
  LOG_INFO("------------------------------------------------------------");

  std::vector<uint32_t> thread_counts = {1, 2, 4, 8};

  for (auto nthreads : thread_counts) {
    DiskANNSearchParams params;
    params.set_ef_search(64).set_pq_rerank(false);

    auto [recall, qps] =
        run_mt_bench(index, ds, kTopk, params, nthreads, kWarmupRounds, kBenchRounds);
    EXPECT_GT(qps, 0.0);
  }

  LOG_INFO("============================================================");
}

// =============================================================================
// Inner Product (IP) metric benchmarks
// =============================================================================

struct BenchResourcesIP {
  Dataset ds_;
  std::shared_ptr<RawSpace<>> space_;
  bool loaded_{false};

  void load() {
    if (loaded_) {
      return;
    }
    ds_ = load_dataset(random_config(kDataNum, kQueryNum, kDim, kGtTopk, 42, MetricType::IP));
    space_ = std::make_shared<RawSpace<>>(ds_.data_num_, ds_.dim_, MetricType::IP);
    space_->fit(ds_.data_.data(), ds_.data_num_);
    loaded_ = true;
    LOG_INFO("BenchResourcesIP: {} vectors, dim={}, {} queries (IP metric)",
             ds_.data_num_, ds_.dim_, kQueryNum);
  }
};

static BenchResourcesIP g_bench_ip;

class DiskANNBenchmarkIP : public ::testing::Test {
 protected:
  static inline std::string index_path = "/tmp/diskann_bench_ip.index";
  static inline bool index_built = false;

  static void SetUpTestSuite() {
    g_bench_ip.load();

    if (!index_built) {
      remove_index_files(index_path);

      auto params = DiskANNBuildParams()
                        .set_max_degree(64)
                        .set_ef_construction(128)
                        .set_num_threads(std::thread::hardware_concurrency())
                        .set_pq_params(kPQSubspaces);

      Timer timer;
      DiskANNIndex<float, uint32_t>::build(g_bench_ip.space_, index_path, params);
      LOG_INFO("IP Index built in {:.2f}s (PQ enabled, M={})", timer.elapsed_s(), kPQSubspaces);
      index_built = true;
    }
  }

  static void TearDownTestSuite() { remove_index_files(index_path); }
};

/// Helper: run IP benchmark (uses Dataset ground truth computed with IP metric)
static auto run_bench_ip(DiskANNIndex<float, uint32_t> &index,
                         const Dataset &ds,
                         uint32_t topk,
                         const DiskANNSearchParams &params,
                         uint32_t warmup_rounds,
                         uint32_t bench_rounds) -> std::pair<float, double> {
  std::vector<uint32_t> results(kQueryNum * topk);
  std::vector<double> latencies;
  latencies.reserve(kQueryNum * bench_rounds);

  for (uint32_t r = 0; r < warmup_rounds; ++r) {
    for (uint32_t q = 0; q < kQueryNum; ++q) {
      const float *query = ds.queries_.data() + q * kDim;
      index.search(query, topk, results.data() + q * topk, params);
    }
  }

  Timer total_timer;
  for (uint32_t r = 0; r < bench_rounds; ++r) {
    for (uint32_t q = 0; q < kQueryNum; ++q) {
      const float *query = ds.queries_.data() + q * kDim;
      Timer query_timer;
      index.search(query, topk, results.data() + q * topk, params);
      latencies.push_back(query_timer.elapsed_us());
    }
  }
  double total_us = total_timer.elapsed_us();

  float recall = calc_recall(results.data(), ds.ground_truth_.data(), kQueryNum, kGtTopk, topk);

  std::ranges::sort(latencies);
  auto n = latencies.size();
  double avg_us = total_us / static_cast<double>(n);
  double p50_us = latencies[n / 2];
  double p99_us = latencies[static_cast<size_t>(static_cast<double>(n) * 0.99)];
  double qps = static_cast<double>(n) / (total_us / 1e6);

  LOG_INFO("{:>6}  {:>10.4f}  {:>10.1f}  {:>10.1f}  {:>10.1f}  {:>10.1f}",
           params.ef_search_, recall, qps, avg_us, p50_us, p99_us);

  return {recall, qps};
}

TEST_F(DiskANNBenchmarkIP, SearchPQ_IP) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 3;
  constexpr uint32_t kBenchRounds = 5;

  const auto &ds = g_bench_ip.ds_;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN PQ Search Benchmark — IP Metric");
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64, M={}",
           kDataNum, kDim, kQueryNum, kTopk, kPQSubspaces);
  LOG_INFO("============================================================");
  LOG_INFO("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
           "ef", "recall@10", "QPS", "avg(us)", "p50(us)", "p99(us)");
  LOG_INFO("------------------------------------------------------------");

  std::vector<uint32_t> ef_values = {32, 64, 128, 256};
  for (auto ef : ef_values) {
    DiskANNSearchParams params;
    params.set_ef_search(ef).set_pq_rerank(true, 4);

    auto [recall, qps] =
        run_bench_ip(index, ds, kTopk, params, kWarmupRounds, kBenchRounds);
    EXPECT_GT(qps, 0.0);
  }

  LOG_INFO("============================================================");
}

TEST_F(DiskANNBenchmarkIP, SearchDisk_IP) {
  DiskANNIndex<float, uint32_t> index;
  index.load(index_path);

  constexpr uint32_t kTopk = 10;
  constexpr uint32_t kWarmupRounds = 1;
  constexpr uint32_t kBenchRounds = 1;

  const auto &ds = g_bench_ip.ds_;

  LOG_INFO("");
  LOG_INFO("============================================================");
  LOG_INFO("  DiskANN Disk-Only Search Benchmark — IP Metric");
  LOG_INFO("  data_num={}, dim={}, query_num={}, topk={}, R=64",
           kDataNum, kDim, kQueryNum, kTopk);
  LOG_INFO("============================================================");
  LOG_INFO("{:>6}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
           "ef", "recall@10", "QPS", "avg(us)", "p50(us)", "p99(us)");
  LOG_INFO("------------------------------------------------------------");

  DiskANNSearchParams params;
  params.set_ef_search(64).set_pq_rerank(false);

  auto [recall, qps] =
      run_bench_ip(index, ds, kTopk, params, kWarmupRounds, kBenchRounds);
  EXPECT_GT(qps, 0.0);
  LOG_INFO("============================================================");
}

}  // namespace alaya
