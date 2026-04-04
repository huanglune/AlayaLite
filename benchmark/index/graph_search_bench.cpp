/*
 * Graph Build & Search benchmark.
 *
 * Usage:
 *   ./graph_search_bench <config.toml>
 *
 * Output format matches hnswlib benchmark for easy comparison:
 *   ef, Recall@1, Recall@10, QPS, Latency(us)
 */

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "bench_config.hpp"

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/hnsw/hnsw_builder.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"
#include "utils/types.hpp"

using alaya::Dataset;
using alaya::DatasetConfig;
using alaya::Graph;
using alaya::GraphSearchJob;
using alaya::HNSWBuilder;
using alaya::RawSpace;
using alaya::Timer;
using alaya::kMetricMap;
using alaya::load_dataset;
using alaya::random_config;

// =============================================================================
// Config
// =============================================================================

struct HnswBenchSection {
  std::string graph_dir_ = "/tmp/hnsw_bench";
  uint32_t warmup_rounds_ = 3;
  uint32_t measure_rounds_ = 5;
};

struct Config {
  bench::DatasetSection dataset_;
  bench::IndexSection index_;
  bench::SearchSection search_;
  HnswBenchSection benchmark_;

  void print() const {
    dataset_.print();
    index_.print();
    search_.print();
    printf("[benchmark]\n");
    printf("  graph_dir  = %s\n", benchmark_.graph_dir_.c_str());
    printf("  warmup     = %u\n", benchmark_.warmup_rounds_);
    printf("  measure    = %u\n", benchmark_.measure_rounds_);
  }
};

static auto load_config(const char *path) -> Config {
  auto parsed = bench::parse_common(path);
  Config c;
  c.dataset_ = std::move(parsed.dataset_);
  c.index_ = std::move(parsed.index_);
  c.search_ = std::move(parsed.search_);
  if (auto *b = parsed.root_["benchmark"].as_table()) {
    c.benchmark_.graph_dir_ =
        bench::toml_get<std::string>(*b, "graph_dir", "/tmp/hnsw_bench");
    c.benchmark_.warmup_rounds_ =
        bench::toml_get<uint32_t>(*b, "warmup_rounds", 3);
    c.benchmark_.measure_rounds_ =
        bench::toml_get<uint32_t>(*b, "measure_rounds", 5);
  }
  return c;
}

// =============================================================================
// Benchmark for one (R, ef_construction) combination
// =============================================================================

static void run(const Config &cfg, Dataset &ds, uint32_t R, uint32_t ef_con) {
  uint32_t num_threads = cfg.index_.num_threads_;
  if (num_threads == 0) {
    num_threads = std::min(static_cast<uint32_t>(std::thread::hardware_concurrency()), 60U);
  }

  uint32_t nq = cfg.search_.num_queries_ > 0
                    ? std::min(cfg.search_.num_queries_, ds.query_num_)
                    : ds.query_num_;

  auto metric = kMetricMap[cfg.dataset_.metric_];

  printf("\n=== AlayaLite HNSW Benchmark ===\n");
  printf("Base vectors: %u x %u\n", ds.data_num_, ds.dim_);
  printf("Query vectors: %u\n", nq);
  printf("Ground truth k: %u\n", ds.gt_dim_);
  printf("R=%u (eqM=%u), ef_construction=%u, threads=%u\n", R, R / 2, ef_con, num_threads);

  // --- Build ---
  auto space = std::make_shared<RawSpace<>>(ds.data_num_, ds.dim_, metric);
  space->fit(ds.data_.data(), ds.data_num_);

  printf("\nBuilding index...\n");
  Timer build_timer;
  auto builder = std::make_unique<HNSWBuilder<RawSpace<>>>(space, R, ef_con);
  auto graph = builder->build_graph(num_threads);
  double build_sec = build_timer.elapsed_s();
  printf("Index built in %.3f seconds\n", build_sec);
  printf("Build throughput: %.0f vectors/sec\n", ds.data_num_ / std::max(build_sec, 1e-9));

  // Save graph
  auto graph_path = cfg.benchmark_.graph_dir_ + "/hnsw_r" + std::to_string(R) +
                    "_l" + std::to_string(ef_con) + ".graph";
  std::filesystem::create_directories(cfg.benchmark_.graph_dir_);
  graph->save(graph_path);
  double rss = bench::get_rss_mb();
  if (rss > 0) { printf("RSS: %.0f MB\n", rss);
}

  // Wrap in shared_ptr for GraphSearchJob
  auto graph_shared = std::shared_ptr<Graph<float, uint32_t>>(std::move(graph));

  // --- Search ---
  printf("%s\n", std::string(60, '=').c_str());
  printf("%-8s %-12s %-12s %-15s %-12s\n", "ef", "Recall@1", "Recall@10", "QPS", "Latency(us)");
  printf("%s\n", std::string(60, '-').c_str());

  GraphSearchJob<RawSpace<>> search_job(space, graph_shared);

  for (uint32_t ef : cfg.search_.ef_search_) {
    std::vector<uint32_t> all_ids(static_cast<size_t>(nq) * cfg.search_.topk_);

    // Warmup
    for (uint32_t w = 0; w < cfg.benchmark_.warmup_rounds_; ++w) {
      for (uint32_t q = 0; q < std::min(nq, 10U); ++q) {
        auto *query = const_cast<float *>(ds.queries_.data() + q * ds.dim_);
        search_job.search_solo(query, cfg.search_.topk_,
                               all_ids.data() + q * cfg.search_.topk_, ef);
      }
    }

    // Measure: run multiple rounds and average
    double total_sec = 0.0;
    for (uint32_t r = 0; r < cfg.benchmark_.measure_rounds_; ++r) {
      Timer t;
      for (uint32_t q = 0; q < nq; ++q) {
        auto *query = const_cast<float *>(ds.queries_.data() +
                                          (q % ds.query_num_) * ds.dim_);
        search_job.search_solo(query, cfg.search_.topk_,
                               all_ids.data() + q * cfg.search_.topk_, ef);
      }
      total_sec += t.elapsed_s();
    }
    double avg_sec = total_sec / cfg.benchmark_.measure_rounds_;
    double qps = nq / std::max(avg_sec, 1e-9);
    double latency_us = (avg_sec / nq) * 1e6;

    // Recall@1: is the top-1 result the true nearest neighbor?
    uint32_t correct_1 = 0;
    for (uint32_t q = 0; q < nq; ++q) {
      if (all_ids[q * cfg.search_.topk_] == ds.ground_truth_[q * ds.gt_dim_]) {
        correct_1++;
      }
    }
    double recall_1 = static_cast<double>(correct_1) / nq;

    // Recall@10: how many of top-10 gt are in top-10 results?
    uint32_t correct_10 = 0;
    uint32_t k10 = std::min(cfg.search_.topk_, 10U);
    uint32_t gt10 = std::min(ds.gt_dim_, 10U);
    for (uint32_t q = 0; q < nq; ++q) {
      std::unordered_set<uint32_t> gt_set;
      for (uint32_t j = 0; j < gt10; ++j) {
        gt_set.insert(ds.ground_truth_[q * ds.gt_dim_ + j]);
      }
      for (uint32_t j = 0; j < k10; ++j) {
        if (gt_set.contains(all_ids[q * cfg.search_.topk_ + j])) {
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

auto main(int argc, char **argv) -> int {
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
  if (!cfg.dataset_.data_path_.empty()) {
    DatasetConfig dc;
    dc.name_ = "bench";
    dc.dir_ = std::filesystem::path(cfg.dataset_.data_path_).parent_path();
    dc.data_file_ = cfg.dataset_.data_path_;
    dc.query_file_ = cfg.dataset_.query_path_;
    dc.gt_file_ = cfg.dataset_.gt_path_;
    ds = load_dataset(dc);
  } else {
    auto metric = kMetricMap[cfg.dataset_.metric_];
    ds = load_dataset(random_config(cfg.dataset_.random_data_num_, cfg.dataset_.random_query_num_,
                                    cfg.dataset_.random_dim_, cfg.dataset_.random_gt_topk_, 42,
                                    metric));
  }

  if (cfg.search_.num_queries_ > 0 && cfg.search_.num_queries_ < ds.query_num_) {
    ds.query_num_ = cfg.search_.num_queries_;
    ds.queries_.resize(static_cast<size_t>(ds.query_num_) * ds.dim_);
    ds.ground_truth_.resize(static_cast<size_t>(ds.query_num_) * ds.gt_dim_);
  }

  printf("Dataset: %u vectors, dim=%u, %u queries, gt_dim=%u\n\n",
         ds.data_num_, ds.dim_, ds.query_num_, ds.gt_dim_);

  for (uint32_t r : cfg.index_.r_) {
    for (uint32_t ef_con : cfg.index_.ef_construction_) {
      run(cfg, ds, r, ef_con);
    }
  }

  printf("Done.\n");
  return 0;
}
