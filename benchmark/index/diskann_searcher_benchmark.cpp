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
#include <cstdlib>
#include <cstring>
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

namespace {
constexpr const char *kSkipBuildEnv = "ALAYA_BENCH_SKIP_BUILD";

auto is_skip_build_env_enabled() -> bool {
  const char *value = std::getenv(kSkipBuildEnv);
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
}
}  // namespace

struct SearcherBenchmarkResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  uint32_t max_threads_{std::min(std::thread::hardware_concurrency(), 60U)};

  std::vector<uint32_t> gt_adjusted_;
  bool gt_is_one_based_{false};

  // Resolve data directory: env ALAYA_DATA_DIR > compile-time project root / data
  std::filesystem::path data_dir_ = []() -> std::filesystem::path {
    const char *env = std::getenv("ALAYA_DATA_DIR");
    if (env != nullptr) {
      return {env};
    }
#ifdef ALAYA_PROJECT_ROOT
    return std::filesystem::path{ALAYA_PROJECT_ROOT} / "data";
#else
    // when exec path is project_root/build
    return std::filesystem::current_path().parent_path().parent_path() / "data";
#endif
  }();
  std::filesystem::path tmp_dir_;
  std::filesystem::path index_prefix_;
  bool index_built_{false};
  size_t total_cache_blocks_{0};
  bool skip_build_{false};

  void load() {
    skip_build_ = is_skip_build_env_enabled();
    ds_ = load_dataset(deep10m(data_dir_));
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = std::min(ds_.query_num_, 1000U); // Limit to 1000 queries for benchmarking
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::IP);
    space_->fit(ds_.data_.data(), data_num_);

    if (data_num_ == 0 || query_num_ == 0 || dim_ == 0 || ds_.gt_dim_ == 0) {
      throw std::runtime_error("Deep1M dataset is not loaded correctly. Set ALAYA_DATA_DIR to the data directory.");
    }

    tmp_dir_ = std::filesystem::temp_directory_path() / "diskann_searcher_benchmark";
    if (!skip_build_) {
      if (std::filesystem::exists(tmp_dir_)) {
        std::filesystem::remove_all(tmp_dir_);
      }
      std::filesystem::create_directories(tmp_dir_);
    } else {
      if (!std::filesystem::exists(tmp_dir_)) {
        std::filesystem::create_directories(tmp_dir_);
      }
    }

    index_prefix_ = tmp_dir_ / "bench_idx";
    LOG_INFO("SearcherBenchmarkResources: Loaded {} vectors, dim={}", data_num_, dim_);
  }

  void build_index() {
    if (index_built_) {
      return;
    }
    // Compute total data blocks needed to hold all vectors in memory.
    // Without PQ, disk search reads full vectors for every neighbor, so the
    // buffer pool must be large enough to avoid thrashing.
    size_t row_size = sizeof(uint32_t) + static_cast<size_t>(kDefaultR) * sizeof(uint32_t)
                      + static_cast<size_t>(dim_) * sizeof(float);
    uint32_t nodes_per_block = kDataBlockSize / row_size;
    total_cache_blocks_ = (static_cast<size_t>(data_num_) + nodes_per_block - 1) / nodes_per_block + 64;
    LOG_INFO("SearcherBenchmarkResources: total_cache_blocks = {}", total_cache_blocks_);

    if (skip_build_) {
      auto data_path = index_prefix_.string() + ".data";
      if (std::filesystem::exists(tmp_dir_) && std::filesystem::exists(data_path)) {
        LOG_INFO("SearcherBenchmarkResources: Skip build enabled, reuse existing index at {}",
                 index_prefix_.string());
        index_built_ = true;
        return;
      }
      LOG_WARN("SearcherBenchmarkResources: Skip build enabled but index not found, building now.");
    }
    auto params = DiskANNBuildParams()
                      .set_max_degree(kDefaultR)
                      .set_ef_construction(kDefaultEf)
                      .set_num_iterations(2)
                      .set_num_threads(max_threads_);

    DiskANNIndex<>::build_graph(space_, index_prefix_.string(), params);
    index_built_ = true;
    LOG_INFO("SearcherBenchmarkResources: Index built at {}", index_prefix_.string());

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
    index.load(g_bench_res.index_prefix_.string(), cache_cap);
    current_cap = cache_cap;
    LOG_INFO("Thread index loaded with cache_cap = {}", cache_cap);
  }
  return index;
}

static void BM_DiskANNSearchBatch(benchmark::State &state) { // NOLINT
  ensure_benchmark_ready();

  auto topk = static_cast<uint32_t>(state.range(0));
  auto ef = static_cast<uint32_t>(state.range(1));
  int cache_pct = static_cast<int>(state.range(2));
  int num_threads = static_cast<int>(state.range(3));

  size_t cache_cap = g_bench_res.cache_cap_for_pct(cache_pct);

  auto &index = get_thread_local_index(cache_cap);
  index.get_searcher().buffer_pool().clear();

  DiskANNSearchParams params;
  params.set_ef_search(ef).set_beam_width(4).set_num_threads(num_threads);

  auto query_num = g_bench_res.query_num_;
  std::vector<uint32_t> results(query_num * topk);

  const auto &stats = index.get_searcher().buffer_pool().stats();
  double cold_hit_rate = 0.0;
  double warm_hit_rate = 0.0;
  int iter = 0;

  for (auto _ : state) {
    auto hits_before = stats.hits_.load(std::memory_order_relaxed);
    auto misses_before = stats.misses_.load(std::memory_order_relaxed);

    index.batch_search(g_bench_res.ds_.queries_.data(), query_num, topk, results.data(), params);
    benchmark::DoNotOptimize(results.data());

    state.PauseTiming();
    auto delta_hits = stats.hits_.load(std::memory_order_relaxed) - hits_before;
    auto delta_total = delta_hits + (stats.misses_.load(std::memory_order_relaxed) - misses_before);
    double hit_rate = delta_total > 0 ? static_cast<double>(delta_hits) / static_cast<double>(delta_total) : 0.0;
    if (iter == 0) {
      cold_hit_rate = hit_rate;
    }
    warm_hit_rate = hit_rate;
    ++iter;
    state.ResumeTiming();
  }

  auto queries_processed = static_cast<double>(state.iterations()) * query_num;
  state.counters["QPS"] = benchmark::Counter(queries_processed,
                                             benchmark::Counter::kIsRate);

  const auto *gt_ptr = g_bench_res.gt_is_one_based_ ? g_bench_res.gt_adjusted_.data()
                                                    : g_bench_res.ds_.ground_truth_.data();
  auto recall = calc_recall(results.data(),
                            gt_ptr,
                            query_num,
                            g_bench_res.ds_.gt_dim_,
                            topk);
  state.counters["Recall@K"] = recall;
  state.counters["ColdHitRate"] = cold_hit_rate;
  state.counters["WarmHitRate"] = warm_hit_rate;
}

// BENCHMARK(BM_DiskANNSearch)
//     ->Args({10, 32, 100})
//     ->Args({10, 64, 100})
//     ->Args({10, 128, 100})
//     ->Args({10, 64, 50})
//     ->Args({10, 64, 25})
//     ->Args({10, 64, 10});
BENCHMARK(BM_DiskANNSearchBatch)
    ->Args({10, 32,100, 16})
    ->Args({10, 16,100, 16})
    ->Args({10, 16,100, 1})
    ->Args({10, 32, 40, 16})
    ->Args({10, 16, 40, 16})
    ->Args({10, 16, 40, 1})
    ->Args({10, 32, 20,16})
    ->Args({10, 16, 20,16})
    ->Args({10, 16, 20,1})
    ->Iterations(2)
    ->UseRealTime();
}  // namespace alaya

BENCHMARK_MAIN();
