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

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {

struct SearcherBenchmarkResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  uint32_t max_threads_{std::min(std::thread::hardware_concurrency(), 60U)};

  std::vector<uint32_t> gt_adjusted_;
  bool gt_is_one_based_{false};

  // file in ROOT_DIR/build/benchmark/index/diskann_searcher_benchmark
  // get the data from ROOT_DIR/data/deep1M
  std::filesystem::path data_dir_ = std::filesystem::current_path().parent_path().parent_path() / "data";
  std::filesystem::path tmp_dir_;
  std::filesystem::path index_path_;
  bool index_built_{false};
  size_t total_cache_blocks_{0};

  void load() {
    ds_ = load_dataset(deep1m(data_dir_));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = ds_.query_num_;
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::L2);
    space_->fit(ds_.data_.data(), data_num_);

    if (data_num_ == 0 || query_num_ == 0 || dim_ == 0 || ds_.gt_dim_ == 0) {
      throw std::runtime_error("Deep1M dataset is not loaded correctly. Set ALAYA_DATA_DIR to the data directory.");
    }


    tmp_dir_ = std::filesystem::temp_directory_path() / "diskann_searcher_benchmark";
    if (std::filesystem::exists(tmp_dir_)) {
      std::filesystem::remove_all(tmp_dir_);
    }
    std::filesystem::create_directories(tmp_dir_);

    index_path_ = tmp_dir_ / "bench_idx";
    LOG_INFO("SearcherBenchmarkResources: Loaded {} vectors, dim={}", data_num_, dim_);
  }

  void build_index() {
    if (index_built_) {
      return;
    }
    auto params = DiskANNBuildParams()
                      .set_max_degree(kDefaultR)
                      .set_ef_construction(kDefaultEf)
                      .set_num_iterations(2)
                      .set_num_threads(max_threads_);

    DiskANNIndex<>::build_graph(space_, index_path_.string(), params);
    index_built_ = true;

    // Compute total data blocks needed to hold all vectors in memory.
    // Without PQ, disk search reads full vectors for every neighbor, so the
    // buffer pool must be large enough to avoid thrashing.
    size_t row_size = sizeof(uint32_t) + static_cast<size_t>(kDefaultR) * sizeof(uint32_t)
                      + static_cast<size_t>(dim_) * sizeof(float);
    uint32_t nodes_per_block = kDataBlockSize / row_size;
    total_cache_blocks_ = (static_cast<size_t>(data_num_) + nodes_per_block - 1) / nodes_per_block + 64;
    LOG_INFO("SearcherBenchmarkResources: total_cache_blocks = {}", total_cache_blocks_);

    LOG_INFO("SearcherBenchmarkResources: Index built at {}", index_path_.string());
  }

  /// Compute buffer pool capacity for a given cache percentage (1-100).
  [[nodiscard]] auto cache_cap_for_pct(int pct) const -> size_t {
    double ratio = std::clamp(pct, 1, 100) / 100.0;
    return std::max(static_cast<size_t>(1),
                    static_cast<size_t>(static_cast<double>(total_cache_blocks_) * ratio));
  }

 private:
  static constexpr uint32_t kDefaultR = 32;
  static constexpr uint32_t kDefaultEf = 64;
};

static SearcherBenchmarkResources g_bench_res;  // NOLINT(cert-err58-cpp)
static void ensure_benchmark_ready() {
  static std::once_flag flag;
  std::call_once(flag, []() -> void {
    g_bench_res.load();
    g_bench_res.build_index();
  });
}

static auto get_thread_local_index(size_t cache_cap) -> DiskANNIndex<> & {
  thread_local DiskANNIndex<> index;
  thread_local size_t current_cap = 0;
  if (current_cap != cache_cap) {
    index = DiskANNIndex<>();
    index.load(g_bench_res.index_path_.string(), cache_cap);
    current_cap = cache_cap;
    LOG_INFO("Thread index loaded with cache_cap = {}", cache_cap);
  }
  return index;
}

static void BM_DiskANNSearch(benchmark::State &state) { // NOLINT
  ensure_benchmark_ready();

  auto topk = static_cast<uint32_t>(state.range(0)); // NOLINT
  auto ef = static_cast<uint32_t>(state.range(1)); // NOLINT
  int cache_pct = static_cast<int>(state.range(2));
  size_t cache_cap = g_bench_res.cache_cap_for_pct(cache_pct);

  auto &index = get_thread_local_index(cache_cap);

  DiskANNSearchParams params;
  params.set_ef_search(ef).set_beam_width(4).set_num_threads(1);

  const float *queries = g_bench_res.ds_.queries_.data();
  auto dim = g_bench_res.dim_;
  auto query_num = g_bench_res.query_num_;
  std::vector<uint32_t> results(query_num * topk);

  // Timing loop, to ensure that the measured average time is stable and accurate
  for (auto _ : state) {
    for (size_t i = 0; i < query_num; ++i) {
      const float *query = queries + i * dim;
      index.search(query, topk, results.data() + i * topk, params);
      benchmark::DoNotOptimize(results.data() + i * topk);
    }
  }

  state.counters["QPS"] = benchmark::Counter(static_cast<double>(state.iterations()) * query_num,
                                             benchmark::Counter::kIsRate);

  state.PauseTiming();
  std::vector<uint32_t> recall_results(query_num * topk);
  index.batch_search(g_bench_res.ds_.queries_.data(),
                     query_num,
                     topk,
                     recall_results.data(),
                     params);
  const auto *gt_ptr = g_bench_res.gt_is_one_based_ ? g_bench_res.gt_adjusted_.data()
                                                    : g_bench_res.ds_.ground_truth_.data();
  auto recall = calc_recall(recall_results.data(),
                            gt_ptr,
                            query_num,
                            g_bench_res.ds_.gt_dim_,
                            topk);
  state.counters["Recall@K"] = recall;
  state.ResumeTiming();
}

static void BM_DiskANNSearchBatch(benchmark::State &state) { // NOLINT
  ensure_benchmark_ready();

  auto topk = static_cast<uint32_t>(state.range(0));
  auto ef = static_cast<uint32_t>(state.range(1));
  int cache_pct = static_cast<int>(state.range(2));
  int num_threads = static_cast<int>(state.range(3));

  size_t cache_cap = g_bench_res.cache_cap_for_pct(cache_pct);

  auto &index = get_thread_local_index(cache_cap);

  DiskANNSearchParams params;
  params.set_ef_search(ef).set_beam_width(4).set_num_threads(num_threads);

  auto query_num = g_bench_res.query_num_;
  std::vector<uint32_t> results(query_num * topk);

  for (auto _ : state) {
    index.batch_search(g_bench_res.ds_.queries_.data(), query_num, topk, results.data(), params);
    benchmark::DoNotOptimize(results.data());
  }

  auto queries_processed = static_cast<double>(state.iterations()) * query_num;
  state.counters["QPS"] = benchmark::Counter(queries_processed,
                                             benchmark::Counter::kIsRate);

  state.PauseTiming();
  const auto *gt_ptr = g_bench_res.gt_is_one_based_ ? g_bench_res.gt_adjusted_.data()
                                                    : g_bench_res.ds_.ground_truth_.data();
  auto recall = calc_recall(results.data(),
                            gt_ptr,
                            query_num,
                            g_bench_res.ds_.gt_dim_,
                            topk);
  state.counters["Recall@K"] = recall;
  state.ResumeTiming();
}

// BENCHMARK(BM_DiskANNSearch)
//     ->Args({10, 32, 100})
//     ->Args({10, 64, 100})
//     ->Args({10, 128, 100})
//     ->Args({10, 64, 50})
//     ->Args({10, 64, 25})
//     ->Args({10, 64, 10});
BENCHMARK(BM_DiskANNSearchBatch)
    ->Args({10, 32, 100, 16})
    ->Args({10, 64, 100, 16})
    ->Args({10, 128, 100, 16})
    ->Args({10, 64, 40, 16})
    ->Args({10, 32, 40, 16})
    ->Args({10, 64, 20, 16})
    ->Args({10, 32, 20, 16});
}  // namespace alaya

BENCHMARK_MAIN();
