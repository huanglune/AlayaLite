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
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/diskann_params.hpp"
#include "space/raw_space.hpp"
#include "storage/diskann/meta_file.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

namespace alaya {

namespace {
constexpr const char *kSkipBuildEnv = "ALAYA_BENCH_SKIP_BUILD";
constexpr const char *kIndexPrefixEnv = "ALAYA_BENCH_INDEX_PREFIX";
constexpr const char *kEfSearchValuesEnv = "ALAYA_BENCH_EF_SEARCH_VALUES";
constexpr const char *kCachePctValuesEnv = "ALAYA_BENCH_CACHE_PCT_VALUES";
constexpr const char *kThreadValuesEnv = "ALAYA_BENCH_THREAD_VALUES";
constexpr double kBytesPerMiB = 1024.0 * 1024.0;
constexpr uint32_t kQueryLimit = 1000;
constexpr uint32_t kMaxBenchmarkThreads = 60;
constexpr uint32_t kCachePadBlocks = 64;
constexpr uint32_t kDefaultTopK = 10;
constexpr uint32_t kTopKArgIndex = 0;
constexpr uint32_t kEfSearchArgIndex = 1;
constexpr uint32_t kCachePctArgIndex = 2;
constexpr uint32_t kNumThreadsArgIndex = 3;
constexpr uint32_t kSearchBeamWidth = 4;
constexpr uint32_t kWarmupSplitDivisor = 2;
constexpr uint32_t kWarmupEvalMinQueryCount = 2;
constexpr uint32_t kQueryShuffleSeed = 42;
constexpr auto kRssSampleInterval = std::chrono::milliseconds(1);

struct BenchmarkCaseArgs {
  uint32_t topk_;
  uint32_t ef_search_;
  int cache_pct_;
  int num_threads_;
};

struct WarmupMetrics {
  uint64_t ready_to_search_rss_bytes_;
  double thread_amplified_rss_mb_;
};

struct EvalSearchMetrics {
  double eval_reuse_hit_rate_;
  double timed_rss_delta_mb_;
};

constexpr std::array<uint32_t, 2> kDefaultEfSearchValues{{16, 32}};
constexpr std::array<int, 3> kDefaultCachePctValues{{20, 40, 100}};
constexpr std::array<int, 2> kDefaultThreadValues{{1, 16}};

auto is_skip_build_env_enabled() -> bool {
  const char *value = std::getenv(kSkipBuildEnv);
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") != 0;
}

auto resolve_index_prefix() -> std::filesystem::path {
  const char *value = std::getenv(kIndexPrefixEnv);
  if (value != nullptr && value[0] != '\0') {
    return {value};
  }
  return std::filesystem::temp_directory_path() / "diskann_searcher_benchmark" / "bench_idx";
}

auto trim_copy(std::string value) -> std::string {
  auto begin = std::ranges::find_if_not(
      value,
      [](unsigned char ch) -> bool { return static_cast<bool>(std::isspace(ch)); });
  auto end = std::find_if_not(
                 value.rbegin(),
                 value.rend(),
                 [](unsigned char ch) -> bool { return static_cast<bool>(std::isspace(ch)); })
                 .base();
  if (begin >= end) {
    return "";
  }
  return {begin, end};
}

template <typename ValueType>
auto parse_positive_value(const std::string &env_name, const std::string &token) -> ValueType {
  std::string trimmed = trim_copy(token);
  if (trimmed.empty()) {
    throw std::runtime_error("Environment variable " + env_name + " contains an empty value.");
  }
  size_t parsed_chars = 0;
  int value = std::stoi(trimmed, &parsed_chars);
  if (parsed_chars != trimmed.size()) {
    throw std::runtime_error("Environment variable " + env_name + " contains an invalid integer: " + trimmed);
  }
  if (value <= 0) {
    throw std::runtime_error("Environment variable " + env_name + " must contain positive integers.");
  }
  return static_cast<ValueType>(value);
}

template <typename ValueType, size_t N>
auto parse_benchmark_values_env(const char *env_name, const std::array<ValueType, N> &defaults) -> std::vector<ValueType> {
  const char *raw = std::getenv(env_name);
  if (raw == nullptr || raw[0] == '\0') {
    return {defaults.begin(), defaults.end()};
  }

  std::vector<ValueType> values;
  std::stringstream stream(raw);
  std::string token;
  while (std::getline(stream, token, ',')) {
    values.push_back(parse_positive_value<ValueType>(env_name, token));
  }
  if (values.empty()) {
    throw std::runtime_error("Environment variable " + std::string(env_name) + " did not provide any values.");
  }
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

auto benchmark_ef_search_values() -> const std::vector<uint32_t> & {
  static const std::vector<uint32_t> kValues = parse_benchmark_values_env(kEfSearchValuesEnv, kDefaultEfSearchValues);
  return kValues;
}

auto benchmark_cache_pct_values() -> const std::vector<int> & {
  static const std::vector<int> kValues = []() -> std::vector<int> {
    std::vector<int> parsed = parse_benchmark_values_env(kCachePctValuesEnv, kDefaultCachePctValues);
    for (int value : parsed) {
      if (value > 100) {
        throw std::runtime_error("Environment variable " + std::string(kCachePctValuesEnv) +
                                 " must stay within 1..100.");
      }
    }
    return parsed;
  }();
  return kValues;
}

auto benchmark_thread_values() -> const std::vector<int> & {
  static const std::vector<int> kValues = []() -> std::vector<int> {
    std::vector<int> parsed = parse_benchmark_values_env(kThreadValuesEnv, kDefaultThreadValues);
    for (int value : parsed) {
      if (value > static_cast<int>(kMaxBenchmarkThreads)) {
        throw std::runtime_error("Environment variable " + std::string(kThreadValuesEnv) + " must stay within 1.." +
                                 std::to_string(kMaxBenchmarkThreads) + ".");
      }
    }
    return parsed;
  }();
  return kValues;
}

auto bytes_to_mib(uint64_t bytes) -> double { return static_cast<double>(bytes) / kBytesPerMiB; }

auto read_status_kib(const char *field_name) -> uint64_t {
#ifdef __linux__
  std::ifstream input("/proc/self/status");
  std::string line;
  while (std::getline(input, line)) {
    if (!line.starts_with(field_name)) {
      continue;
    }
    std::istringstream iss(line.substr(std::strlen(field_name)));
    uint64_t value = 0;
    iss >> value;
    return value;
  }
#else
  (void)field_name;
#endif
  return 0;
}

auto current_rss_bytes() -> uint64_t { return read_status_kib("VmRSS:") * 1024ULL; }

class RssSampler {
 public:
  RssSampler() = default;
  ~RssSampler() { stop(); }

  void start(uint64_t baseline_bytes) {
    stop();
    peak_bytes_.store(baseline_bytes, std::memory_order_relaxed);
    running_.store(true, std::memory_order_release);
    worker_ = std::thread([this]() -> void {
      while (running_.load(std::memory_order_acquire)) {
        sample_once();
        std::this_thread::sleep_for(kRssSampleInterval);
      }
      sample_once();
    });
  }

  void stop() {
    running_.store(false, std::memory_order_release);
    if (worker_.joinable()) {
      worker_.join();
    }
  }

  [[nodiscard]] auto peak_bytes() const -> uint64_t {
    return peak_bytes_.load(std::memory_order_relaxed);
  }

 private:
  void sample_once() {
    uint64_t current = current_rss_bytes();
    uint64_t peak = peak_bytes_.load(std::memory_order_relaxed);
    while (current > peak &&
           !peak_bytes_.compare_exchange_weak(peak, current, std::memory_order_relaxed)) {
    }
  }

  std::atomic<bool> running_{false};
  std::atomic<uint64_t> peak_bytes_{0};
  std::thread worker_;
};

template <typename SearcherType>
auto resolve_bypass_page_cache(const SearcherType &searcher) -> double {
  if constexpr (requires { searcher.bypasses_page_cache(); }) {
    return searcher.bypasses_page_cache() ? 1.0 : 0.0;
  }
  return -1.0;
}
}  // namespace

struct SearcherBenchmarkResources {
  Dataset ds_;
  uint32_t dim_{0};
  uint32_t data_num_{0};
  uint32_t query_num_{0};
  uint32_t warmup_query_num_{0};
  uint32_t eval_query_num_{0};
  std::shared_ptr<RawSpace<>> space_;
  std::vector<float> warmup_queries_;
  std::vector<float> eval_queries_;
  std::vector<uint32_t> eval_ground_truth_;
  uint32_t max_threads_{std::max(1U, std::min(std::thread::hardware_concurrency(), kMaxBenchmarkThreads))};
  std::filesystem::path data_dir_ = []() -> std::filesystem::path {
    const char *env = std::getenv("ALAYA_DATA_DIR");
    if (env != nullptr) {
      return {env};
    }
#ifdef ALAYA_PROJECT_ROOT
    return std::filesystem::path{ALAYA_PROJECT_ROOT} / "data";
#else
    return std::filesystem::current_path().parent_path().parent_path() / "data";
#endif
  }();
  std::filesystem::path index_prefix_{resolve_index_prefix()};
  bool index_built_{false};
  size_t total_cache_blocks_{0};
  bool skip_build_{false};

  void load() {
    skip_build_ = is_skip_build_env_enabled();
    if (skip_build_ && has_existing_index()) {
      load_queries_and_ground_truth();
      load_index_metadata();
    } else {
      load_full_dataset();
    }

    if (data_num_ == 0 || query_num_ == 0 || dim_ == 0 || ds_.gt_dim_ == 0) {
      throw std::runtime_error("Deep1M dataset is not loaded correctly. Set ALAYA_DATA_DIR to the data directory.");
    }
    split_benchmark_queries();

    auto index_dir = index_prefix_.parent_path();
    if (!index_dir.empty()) {
      std::filesystem::create_directories(index_dir);
    }
    if (!skip_build_) {
      std::filesystem::remove(index_prefix_.string() + ".data");
      std::filesystem::remove(index_prefix_.string() + ".meta");
      std::filesystem::remove(index_prefix_.string() + ".pq");
    }

    LOG_INFO("SearcherBenchmarkResources: Loaded {} vectors, dim={}, index_prefix={}",
             data_num_,
             dim_,
             index_prefix_.string());
  }

  void build_index() {
    if (index_built_) {
      return;
    }

    size_t row_size = sizeof(uint32_t) + static_cast<size_t>(kDefaultR) * sizeof(uint32_t)
                      + static_cast<size_t>(dim_) * sizeof(float);
    uint32_t nodes_per_block = kDataBlockSize / row_size;
    total_cache_blocks_ =
        (static_cast<size_t>(data_num_) + nodes_per_block - 1) / nodes_per_block + kCachePadBlocks;
    LOG_INFO("SearcherBenchmarkResources: total_cache_blocks = {}", total_cache_blocks_);

    if (skip_build_ && has_existing_index()) {
      LOG_INFO("SearcherBenchmarkResources: Skip build enabled, reuse existing index at {}",
               index_prefix_.string());
      index_built_ = true;
      return;
    }

    if (skip_build_) {
      LOG_WARN("SearcherBenchmarkResources: Skip build enabled but index not found, building now.");
    }

    auto params = DiskANNBuildParams()
                      .set_max_degree(kDefaultR)
                      .set_ef_construction(kDefaultEf)
                      .set_num_iterations(2)
                      .set_num_threads(max_threads_);

    DiskANNIndex<>::build_graph(space_, index_prefix_.string(), params);
    release_build_only_memory();
    index_built_ = true;
    LOG_INFO("SearcherBenchmarkResources: Index built at {}", index_prefix_.string());
  }

  [[nodiscard]] auto cache_cap_for_pct(int pct) const -> size_t {
    double ratio = std::clamp(pct, 1, 100) / 100.0;
    return std::max(static_cast<size_t>(1),
                    static_cast<size_t>(static_cast<double>(total_cache_blocks_) * ratio));
  }

 private:
  [[nodiscard]] auto config() const -> DatasetConfig { return deep10m(data_dir_); }

  [[nodiscard]] auto has_existing_index() const -> bool {
    return std::filesystem::exists(index_prefix_.string() + ".data") &&
           std::filesystem::exists(index_prefix_.string() + ".meta");
  }

  void load_full_dataset() {
    ds_ = load_dataset(config());
    dim_ = ds_.dim_;
    data_num_ = ds_.data_num_;
    query_num_ = std::min(ds_.query_num_, kQueryLimit);
    trim_queries_to_limit();
    space_ = std::make_shared<RawSpace<>>(data_num_, dim_, MetricType::IP);
    space_->fit(ds_.data_.data(), data_num_);
  }

  void load_queries_and_ground_truth() {
    ds_.name_ = config().name_;
    uint32_t query_dim = 0;
    load_fvecs(config().query_file_, ds_.queries_, ds_.query_num_, query_dim);
    load_ivecs(config().gt_file_, ds_.ground_truth_, ds_.query_num_, ds_.gt_dim_);
    dim_ = query_dim;
    query_num_ = std::min(ds_.query_num_, kQueryLimit);
    trim_queries_to_limit();
  }

  void load_index_metadata() {
    MetaFile meta;
    meta.open(index_prefix_.string() + ".meta");
    data_num_ = static_cast<uint32_t>(meta.num_active_points());
    if (dim_ == 0) {
      dim_ = meta.dimension();
    }
  }

  void trim_queries_to_limit() {
    if (query_num_ >= ds_.query_num_) {
      return;
    }
    ds_.queries_.resize(static_cast<size_t>(query_num_) * dim_);
    ds_.ground_truth_.resize(static_cast<size_t>(query_num_) * ds_.gt_dim_);
    ds_.query_num_ = query_num_;
  }

  void split_benchmark_queries() {
    if (query_num_ < kWarmupEvalMinQueryCount) {
      throw std::runtime_error("DiskANN benchmark needs at least two queries to split warmup and eval sets.");
    }

    std::vector<uint32_t> query_order(query_num_);
    std::iota(query_order.begin(), query_order.end(), 0U);
    std::mt19937 rng(kQueryShuffleSeed);
    std::shuffle(query_order.begin(), query_order.end(), rng);

    warmup_query_num_ = query_num_ / kWarmupSplitDivisor;
    eval_query_num_ = query_num_ - warmup_query_num_;
    if (warmup_query_num_ == 0 || eval_query_num_ == 0) {
      throw std::runtime_error("DiskANN benchmark query split produced an empty warmup or eval set.");
    }

    warmup_queries_.resize(static_cast<size_t>(warmup_query_num_) * dim_);
    eval_queries_.resize(static_cast<size_t>(eval_query_num_) * dim_);
    eval_ground_truth_.resize(static_cast<size_t>(eval_query_num_) * ds_.gt_dim_);

    for (uint32_t i = 0; i < warmup_query_num_; ++i) {
      uint32_t src_query = query_order[i];
      std::copy_n(ds_.queries_.data() + static_cast<size_t>(src_query) * dim_,
                  dim_,
                  warmup_queries_.data() + static_cast<size_t>(i) * dim_);
    }

    for (uint32_t i = 0; i < eval_query_num_; ++i) {
      uint32_t src_query = query_order[warmup_query_num_ + i];
      std::copy_n(ds_.queries_.data() + static_cast<size_t>(src_query) * dim_,
                  dim_,
                  eval_queries_.data() + static_cast<size_t>(i) * dim_);
      std::copy_n(ds_.ground_truth_.data() + static_cast<size_t>(src_query) * ds_.gt_dim_,
                  ds_.gt_dim_,
                  eval_ground_truth_.data() + static_cast<size_t>(i) * ds_.gt_dim_);
    }

    LOG_INFO("SearcherBenchmarkResources: Split {} queries into {} warmup and {} eval (seed={})",
             query_num_,
             warmup_query_num_,
             eval_query_num_,
             kQueryShuffleSeed);

    ds_.queries_.clear();
    ds_.queries_.shrink_to_fit();
    ds_.ground_truth_.clear();
    ds_.ground_truth_.shrink_to_fit();
  }

  void release_build_only_memory() {
    space_.reset();
    ds_.data_.clear();
    ds_.data_.shrink_to_fit();
  }

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

static auto benchmark_args_from_state(const benchmark::State &state) -> BenchmarkCaseArgs {
  return {
      .topk_ = static_cast<uint32_t>(state.range(kTopKArgIndex)),
      .ef_search_ = static_cast<uint32_t>(state.range(kEfSearchArgIndex)),
      .cache_pct_ = static_cast<int>(state.range(kCachePctArgIndex)),
      .num_threads_ = static_cast<int>(state.range(kNumThreadsArgIndex)),
  };
}

static auto make_search_params(uint32_t ef_search, int num_threads) -> DiskANNSearchParams {
  DiskANNSearchParams params;
  params.set_ef_search(ef_search).set_beam_width(kSearchBeamWidth).set_num_threads(num_threads);
  return params;
}

static void execute_batch_search(DiskANNIndex<> &index,
                                 const float *queries,
                                 uint32_t query_num,
                                 uint32_t topk,
                                 std::vector<uint32_t> &results,
                                 const DiskANNSearchParams &params) {
  index.batch_search(queries,
                     query_num,
                     topk,
                     results.data(),
                     params);
}

static auto warm_up_search(DiskANNIndex<> &index,
                           uint32_t topk,
                           std::vector<uint32_t> &results,
                           const DiskANNSearchParams &params) -> WarmupMetrics {
  index.get_searcher().buffer_pool().clear();
  uint64_t steady_rss_bytes = current_rss_bytes();
  execute_batch_search(index, g_bench_res.warmup_queries_.data(), g_bench_res.warmup_query_num_, topk, results, params);
  uint64_t ready_to_search_rss_bytes = current_rss_bytes();
  return {
      .ready_to_search_rss_bytes_ = ready_to_search_rss_bytes,
      .thread_amplified_rss_mb_ =
          bytes_to_mib(ready_to_search_rss_bytes > steady_rss_bytes ? ready_to_search_rss_bytes - steady_rss_bytes
                                                                    : 0),
  };
}

template <typename StatsType>
static auto measure_eval_search(benchmark::State &state,
                                DiskANNIndex<> &index,
                                uint32_t topk,
                                std::vector<uint32_t> &results,
                                const DiskANNSearchParams &params,
                                const StatsType &stats,
                                uint64_t ready_to_search_rss_bytes) -> EvalSearchMetrics {
  index.get_searcher().buffer_pool().reset_stats();
  RssSampler rss_sampler;
  rss_sampler.start(ready_to_search_rss_bytes);

  for (auto _ : state) {
    (void)_;
    execute_batch_search(index, g_bench_res.eval_queries_.data(), g_bench_res.eval_query_num_, topk, results, params);
    benchmark::DoNotOptimize(results.data());
  }

  state.PauseTiming();
  rss_sampler.stop();
  uint64_t peak_timed_rss_bytes = rss_sampler.peak_bytes();
  double eval_reuse_hit_rate = stats.reuse_hit_rate();
  double timed_rss_delta_mb =
      bytes_to_mib(peak_timed_rss_bytes > ready_to_search_rss_bytes ? peak_timed_rss_bytes - ready_to_search_rss_bytes
                                                                    : 0);
  state.ResumeTiming();

  return {
      .eval_reuse_hit_rate_ = eval_reuse_hit_rate,
      .timed_rss_delta_mb_ = timed_rss_delta_mb,
  };
}

static void publish_counters(benchmark::State &state,
                             uint32_t topk,
                             const std::vector<uint32_t> &results,
                             double cache_footprint_mb,
                             double bypass_page_cache,
                             const WarmupMetrics &warmup_metrics,
                             const EvalSearchMetrics &eval_metrics) {
  auto queries_processed = static_cast<double>(state.iterations()) * g_bench_res.eval_query_num_;
  state.counters["QPS"] = benchmark::Counter(queries_processed, benchmark::Counter::kIsRate);
  state.counters["Recall@K"] = calc_recall(results.data(),
                                           g_bench_res.eval_ground_truth_.data(),
                                           g_bench_res.eval_query_num_,
                                           g_bench_res.ds_.gt_dim_,
                                           topk);
  state.counters["EvalReuseHitRate"] = eval_metrics.eval_reuse_hit_rate_;
  state.counters["CacheFootprintMB"] = cache_footprint_mb;
  state.counters["BypassPageCache"] = bypass_page_cache;
  state.counters["ReadyToSearchRSSMB"] = bytes_to_mib(warmup_metrics.ready_to_search_rss_bytes_);
  state.counters["ThreadAmplifiedRSSMB"] = warmup_metrics.thread_amplified_rss_mb_;
  state.counters["TimedRSSDeltaMB"] = eval_metrics.timed_rss_delta_mb_;
}

static void BM_DiskANNSearchBatch(benchmark::State &state) {  // NOLINT
  ensure_benchmark_ready();

  BenchmarkCaseArgs args = benchmark_args_from_state(state);
  size_t cache_cap = g_bench_res.cache_cap_for_pct(args.cache_pct_);
  double cache_footprint_mb = bytes_to_mib(static_cast<uint64_t>(cache_cap) * kDataBlockSize);

  auto &index = get_thread_local_index(cache_cap);
  double bypass_page_cache = resolve_bypass_page_cache(index.get_searcher());
  auto params = make_search_params(args.ef_search_, args.num_threads_);
  std::vector<uint32_t> results(g_bench_res.eval_query_num_ * args.topk_);

  WarmupMetrics warmup_metrics = warm_up_search(index, args.topk_, results, params);
  const auto &stats = index.get_searcher().buffer_pool().stats();
  EvalSearchMetrics eval_metrics =
      measure_eval_search(state, index, args.topk_, results, params, stats, warmup_metrics.ready_to_search_rss_bytes_);
  publish_counters(state,
                   args.topk_,
                   results,
                   cache_footprint_mb,
                   bypass_page_cache,
                   warmup_metrics,
                   eval_metrics);
}

static void register_benchmark_cases() {
  auto *benchmark_case = benchmark::RegisterBenchmark("BM_DiskANNSearchBatch", &BM_DiskANNSearchBatch);
  for (uint32_t ef_search : benchmark_ef_search_values()) {
    for (int cache_pct : benchmark_cache_pct_values()) {
      for (int num_threads : benchmark_thread_values()) {
        benchmark_case->Args({kDefaultTopK, static_cast<int64_t>(ef_search), cache_pct, num_threads});
      }
    }
  }
  benchmark_case->Iterations(1)->UseRealTime();
}

const bool kBenchmarkRegistered = []() -> bool {
  register_benchmark_cases();
  return true;
}();

}  // namespace alaya

BENCHMARK_MAIN();
