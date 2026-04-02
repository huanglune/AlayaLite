/*
 * HNSW Build & Search benchmark.
 *
 * Usage:
 *   ./hnsw_bench <config.toml>
 *
 * Output format matches hnswlib benchmark for easy comparison:
 *   ef, Recall@1, Recall@10, QPS, Latency(us)
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <toml++/toml.hpp>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"

using namespace alaya;

// =============================================================================
// Config
// =============================================================================

struct Config {
  // [dataset]
  std::string data_path;
  std::string query_path;
  std::string gt_path;
  std::string metric = "L2";

  // Random dataset (used when data_path is empty)
  uint32_t random_data_num = 10000;
  uint32_t random_query_num = 100;
  uint32_t random_dim = 128;
  uint32_t random_gt_topk = 100;

  // [index]
  std::vector<uint32_t> R = {16, 32, 64};
  std::vector<uint32_t> ef_construction = {200};
  uint32_t num_threads = 0;  // 0 = hardware_concurrency

  // [search]
  std::vector<uint32_t> ef_search = {16, 32, 64, 128, 256};
  uint32_t topk = 10;
  uint32_t num_queries = 0;  // 0 = all queries in dataset

  // [benchmark]
  std::string graph_dir = "/tmp/hnsw_bench";
  uint32_t warmup_rounds = 3;
  uint32_t measure_rounds = 5;

  void print() const {
    printf("[dataset]\n");
    if (!data_path.empty()) {
      printf("  data_path  = %s\n", data_path.c_str());
      printf("  query_path = %s\n", query_path.c_str());
      printf("  gt_path    = %s\n", gt_path.c_str());
    } else {
      printf("  random: %u vectors, %u queries, dim=%u\n",
             random_data_num, random_query_num, random_dim);
    }
    printf("  metric     = %s\n", metric.c_str());
    printf("[index]\n");
    printf("  R          = [");
    for (size_t i = 0; i < R.size(); ++i) printf("%s%u", i ? ", " : "", R[i]);
    printf("]\n");
    printf("  ef_construction = [");
    for (size_t i = 0; i < ef_construction.size(); ++i)
      printf("%s%u", i ? ", " : "", ef_construction[i]);
    printf("]\n");
    printf("  num_threads= %u\n", num_threads);
    printf("[search]\n");
    printf("  ef_search  = [");
    for (size_t i = 0; i < ef_search.size(); ++i) printf("%s%u", i ? ", " : "", ef_search[i]);
    printf("]\n");
    printf("  topk       = %u\n", topk);
    printf("  num_queries= %u\n", num_queries);
    printf("[benchmark]\n");
    printf("  graph_dir  = %s\n", graph_dir.c_str());
    printf("  warmup     = %u\n", warmup_rounds);
    printf("  measure    = %u\n", measure_rounds);
  }
};

template <typename T>
static auto toml_array_to_vec(const toml::array &arr) -> std::vector<T> {
  std::vector<T> v;
  for (auto &el : arr) v.push_back(static_cast<T>(el.value_or(T{0})));
  return v;
}

static auto load_config(const char *path) -> Config {
  auto tbl = toml::parse_file(path);
  Config c;

  if (auto ds = tbl["dataset"].as_table()) {
    if (auto *v = ds->get("data_path")) c.data_path = v->value_or(std::string{});
    if (auto *v = ds->get("query_path")) c.query_path = v->value_or(std::string{});
    if (auto *v = ds->get("gt_path")) c.gt_path = v->value_or(std::string{});
    if (auto *v = ds->get("metric")) c.metric = v->value_or(std::string{"L2"});
    if (auto *v = ds->get("random_data_num")) c.random_data_num = v->value_or(10000);
    if (auto *v = ds->get("random_query_num")) c.random_query_num = v->value_or(100);
    if (auto *v = ds->get("random_dim")) c.random_dim = v->value_or(128);
    if (auto *v = ds->get("random_gt_topk")) c.random_gt_topk = v->value_or(100);
  }
  if (auto idx = tbl["index"].as_table()) {
    if (auto *a = idx->get("R"); a && a->is_array())
      c.R = toml_array_to_vec<uint32_t>(*a->as_array());
    if (auto *a = idx->get("ef_construction"); a && a->is_array())
      c.ef_construction = toml_array_to_vec<uint32_t>(*a->as_array());
    if (auto *v = idx->get("num_threads")) c.num_threads = v->value_or(0U);
  }
  if (auto s = tbl["search"].as_table()) {
    if (auto *a = s->get("ef_search"); a && a->is_array())
      c.ef_search = toml_array_to_vec<uint32_t>(*a->as_array());
    if (auto *v = s->get("topk")) c.topk = v->value_or(10);
    if (auto *v = s->get("num_queries")) c.num_queries = v->value_or(0U);
  }
  if (auto b = tbl["benchmark"].as_table()) {
    if (auto *v = b->get("graph_dir")) c.graph_dir = v->value_or(std::string{"/tmp/hnsw_bench"});
    if (auto *v = b->get("warmup_rounds")) c.warmup_rounds = v->value_or(3);
    if (auto *v = b->get("measure_rounds")) c.measure_rounds = v->value_or(5);
  }

  return c;
}

// =============================================================================
// Helpers
// =============================================================================

struct Stopwatch {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  auto sec() const -> double {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
};

static auto get_rss_mb() -> double {
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      long kb = 0;
      std::sscanf(line.c_str(), "VmRSS: %ld", &kb);
      return static_cast<double>(kb) / 1024.0;
    }
  }
#endif
  return -1.0;
}

// =============================================================================
// Benchmark for one (R, ef_construction) combination
// =============================================================================

static void run(const Config &cfg, Dataset &ds, uint32_t R, uint32_t ef_con) {
  uint32_t num_threads = cfg.num_threads;
  if (num_threads == 0) {
    num_threads = std::min(static_cast<uint32_t>(std::thread::hardware_concurrency()), 60U);
  }

  uint32_t nq = cfg.num_queries > 0
                    ? std::min(cfg.num_queries, ds.query_num_)
                    : ds.query_num_;

  auto metric = MetricType::L2;
  if (cfg.metric == "IP") metric = MetricType::IP;
  if (cfg.metric == "COS") metric = MetricType::COS;

  printf("\n=== AlayaLite HNSW Benchmark ===\n");
  printf("Base vectors: %u x %u\n", ds.data_num_, ds.dim_);
  printf("Query vectors: %u\n", nq);
  printf("Ground truth k: %u\n", ds.gt_dim_);
  printf("M=%u, ef_construction=%u, threads=%u\n", R, ef_con, num_threads);

  // --- Build ---
  auto space = std::make_shared<RawSpace<>>(ds.data_num_, ds.dim_, metric);
  space->fit(ds.data_.data(), ds.data_num_);

  printf("\nBuilding index...\n");
  Stopwatch build_timer;
  auto builder = std::make_unique<HNSWBuilder<RawSpace<>>>(space, R, ef_con);
  auto graph = builder->build_graph(num_threads);
  double build_sec = build_timer.sec();
  printf("Index built in %.3f seconds\n", build_sec);
  printf("Build throughput: %.0f vectors/sec\n", ds.data_num_ / std::max(build_sec, 1e-9));

  // Save graph
  auto graph_path = cfg.graph_dir + "/hnsw_r" + std::to_string(R) +
                    "_l" + std::to_string(ef_con) + ".graph";
  std::filesystem::create_directories(cfg.graph_dir);
  graph->save(graph_path);
  double rss = get_rss_mb();
  if (rss > 0) printf("RSS: %.0f MB\n", rss);

  // Wrap in shared_ptr for GraphSearchJob
  auto graph_shared = std::shared_ptr<Graph<float, uint32_t>>(std::move(graph));

  // --- Search ---
  printf("%s\n", std::string(60, '=').c_str());
  printf("%-8s %-12s %-12s %-15s %-12s\n", "ef", "Recall@1", "Recall@10", "QPS", "Latency(us)");
  printf("%s\n", std::string(60, '-').c_str());

  GraphSearchJob<RawSpace<>> search_job(space, graph_shared);

  for (uint32_t ef : cfg.ef_search) {
    std::vector<uint32_t> all_ids(static_cast<size_t>(nq) * cfg.topk);

    // Warmup
    for (uint32_t w = 0; w < cfg.warmup_rounds; ++w) {
      for (uint32_t q = 0; q < std::min(nq, 10U); ++q) {
        auto *query = const_cast<float *>(ds.queries_.data() + q * ds.dim_);
        search_job.search_solo(query, cfg.topk, all_ids.data() + q * cfg.topk, ef);
      }
    }

    // Measure: run multiple rounds and average
    double total_sec = 0.0;
    for (uint32_t r = 0; r < cfg.measure_rounds; ++r) {
      Stopwatch t;
      for (uint32_t q = 0; q < nq; ++q) {
        auto *query = const_cast<float *>(ds.queries_.data() +
                                          (q % ds.query_num_) * ds.dim_);
        search_job.search_solo(query, cfg.topk, all_ids.data() + q * cfg.topk, ef);
      }
      total_sec += t.sec();
    }
    double avg_sec = total_sec / cfg.measure_rounds;
    double qps = nq / std::max(avg_sec, 1e-9);
    double latency_us = (avg_sec / nq) * 1e6;

    // Recall@1: is the top-1 result the true nearest neighbor?
    uint32_t correct_1 = 0;
    for (uint32_t q = 0; q < nq; ++q) {
      if (all_ids[q * cfg.topk] == ds.ground_truth_[q * ds.gt_dim_]) {
        correct_1++;
      }
    }
    double recall_1 = static_cast<double>(correct_1) / nq;

    // Recall@10: how many of top-10 gt are in top-10 results?
    uint32_t correct_10 = 0;
    uint32_t k10 = std::min(cfg.topk, 10U);
    uint32_t gt10 = std::min(ds.gt_dim_, 10U);
    for (uint32_t q = 0; q < nq; ++q) {
      std::unordered_set<uint32_t> gt_set;
      for (uint32_t j = 0; j < gt10; ++j) {
        gt_set.insert(ds.ground_truth_[q * ds.gt_dim_ + j]);
      }
      for (uint32_t j = 0; j < k10; ++j) {
        if (gt_set.count(all_ids[q * cfg.topk + j]) != 0U) {
          correct_10++;
        }
      }
    }
    double recall_10 = static_cast<double>(correct_10) / (nq * k10);

    printf("%-8u %-12.4f %-12.4f %-15.0f %-12.1f\n", ef, recall_1, recall_10, qps, latency_us);
  }

  printf("%s\n", std::string(60, '=').c_str());
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <config.toml>\n", argv[0]);
    return 1;
  }

  auto cfg = load_config(argv[1]);
  printf("Config:\n");
  cfg.print();
  printf("\n");

  // Load or generate dataset
  Dataset ds;
  if (!cfg.data_path.empty()) {
    DatasetConfig dc;
    dc.name_ = "bench";
    dc.dir_ = std::filesystem::path(cfg.data_path).parent_path();
    dc.data_file_ = cfg.data_path;
    dc.query_file_ = cfg.query_path;
    dc.gt_file_ = cfg.gt_path;
    ds = load_dataset(dc);
  } else {
    auto metric = MetricType::L2;
    if (cfg.metric == "IP") metric = MetricType::IP;
    if (cfg.metric == "COS") metric = MetricType::COS;
    ds = load_dataset(random_config(cfg.random_data_num, cfg.random_query_num,
                                    cfg.random_dim, cfg.random_gt_topk, 42, metric));
  }

  if (cfg.num_queries > 0 && cfg.num_queries < ds.query_num_) {
    ds.query_num_ = cfg.num_queries;
    ds.queries_.resize(static_cast<size_t>(ds.query_num_) * ds.dim_);
    ds.ground_truth_.resize(static_cast<size_t>(ds.query_num_) * ds.gt_dim_);
  }

  printf("Dataset: %u vectors, dim=%u, %u queries, gt_dim=%u\n\n",
         ds.data_num_, ds.dim_, ds.query_num_, ds.gt_dim_);

  for (uint32_t r : cfg.R) {
    for (uint32_t ef_con : cfg.ef_construction) {
      run(cfg, ds, r, ef_con);
    }
  }

  printf("Done.\n");
  return 0;
}
