/*
 * DiskANN Search benchmark — measures QPS, Recall, Latency across
 * different R, ef_search (L), and beam_width (W) combinations.
 *
 * Designed for direct comparison with official Microsoft DiskANN results.
 *
 * Usage:
 *   ./diskann_search_bench <config.toml>
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "bench_config.hpp"

#include "index/diskann/diskann_index.hpp"
#include "index/diskann/diskann_params.hpp"
#include "space/raw_space.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"
#include "utils/types.hpp"

using alaya::DiskANNBuildParams;
using alaya::DiskANNIndex;
using alaya::DiskANNSearchParams;
using alaya::Dataset;
using alaya::DatasetConfig;
using alaya::RawSpace;
using alaya::Timer;
using alaya::calc_recall;
using alaya::kDataBlockSize;
using alaya::kMetricMap;
using alaya::load_dataset;
using alaya::load_fvecs;

// =============================================================================
// Config
// =============================================================================

struct SearchBenchSection {
  std::string index_dir_ = "/tmp/diskann_search_bench";
  float cache_ratio_ = 0.20F;
  uint32_t warmup_queries_ = 100;
  uint32_t measure_rounds_ = 3;
  std::vector<uint32_t> beam_widths_ = {4};
};

struct Config {
  bench::DatasetSection dataset_;
  bench::IndexSection index_;
  bench::SearchSection search_;
  SearchBenchSection benchmark_;

  void print() const {
    dataset_.print();
    index_.print();
    search_.print();
    printf("[benchmark]\n");
    printf("  index_dir      = %s\n", benchmark_.index_dir_.c_str());
    printf("  cache_ratio    = %.2f\n", benchmark_.cache_ratio_);
    printf("  warmup_queries = %u\n", benchmark_.warmup_queries_);
    printf("  measure_rounds = %u\n", benchmark_.measure_rounds_);
    printf("  beam_widths    = [");
    for (size_t i = 0; i < benchmark_.beam_widths_.size(); ++i) {
      printf("%s%u", i != 0U ? ", " : "", benchmark_.beam_widths_[i]);
    }
    printf("]\n");
  }
};

static auto load_config(const char *path) -> Config {
  auto parsed = bench::parse_common(path);
  Config c;
  c.dataset_ = std::move(parsed.dataset_);
  c.index_ = std::move(parsed.index_);
  c.search_ = std::move(parsed.search_);
  if (auto *b = parsed.root_["benchmark"].as_table()) {
    c.benchmark_.index_dir_ =
        bench::toml_get<std::string>(*b, "index_dir", "/tmp/diskann_search_bench");
    c.benchmark_.cache_ratio_ =
        bench::toml_get<float>(*b, "cache_ratio", 0.20F);
    c.benchmark_.warmup_queries_ =
        bench::toml_get<uint32_t>(*b, "warmup_queries", 100);
    c.benchmark_.measure_rounds_ =
        bench::toml_get<uint32_t>(*b, "measure_rounds", 3);
    c.benchmark_.beam_widths_ =
        bench::toml_get_vec<uint32_t>(*b, "beam_widths", {4});
  }

  if (c.dataset_.data_path_.empty() || c.dataset_.query_path_.empty() ||
      c.dataset_.gt_path_.empty()) {
    fprintf(stderr, "Config must specify [dataset] data_path, query_path, gt_path\n");
    exit(1);
  }
  return c;
}

// =============================================================================
// Per-query latency measurement
// =============================================================================

struct SearchStats {
  double mean_latency_us_ = 0;
  double p999_latency_us_ = 0;
  double qps_ = 0;
  float recall_ = 0;
};

static auto run_search(DiskANNIndex<> &idx,
                       const Dataset &ds,
                       uint32_t nq,
                       uint32_t topk,
                       uint32_t ef,
                       uint32_t bw,
                       uint32_t warmup_queries,
                       uint32_t measure_rounds) -> SearchStats {
  DiskANNSearchParams params;
  params.set_ef_search(ef).set_beam_width(bw).set_num_threads(1);

  std::vector<uint32_t> ids(static_cast<size_t>(nq) * topk);

  // Warmup
  uint32_t nw = std::min(warmup_queries, nq);
  for (uint32_t q = 0; q < nw; ++q) {
    idx.search(ds.queries_.data() + (q % ds.query_num_) * ds.dim_,
               topk, ids.data() + q * topk, params);
  }

  // Measure per-query latencies
  std::vector<double> latencies(nq);
  double best_total = 1e18;

  for (uint32_t round = 0; round < measure_rounds; ++round) {
    double round_total = 0;
    for (uint32_t q = 0; q < nq; ++q) {
      Timer query_timer;
      idx.search(ds.queries_.data() + (q % ds.query_num_) * ds.dim_,
                 topk, ids.data() + q * topk, params);
      double us = query_timer.elapsed_us();
      latencies[q] = us;
      round_total += us;
    }
    best_total = std::min(best_total, round_total);
  }

  // Compute stats from last round's latencies
  std::sort(latencies.begin(), latencies.end());
  double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  double mean_us = sum / nq;
  auto p999_idx = static_cast<size_t>(nq * 0.999);
  double p999_us = latencies[std::min(p999_idx, latencies.size() - 1)];
  double qps = static_cast<double>(nq) / (best_total * 1e-6);

  // Recall
  float recall = calc_recall(ids.data(), ds.ground_truth_.data(),
                             nq, ds.gt_dim_, topk);

  return {mean_us, p999_us, qps, recall};
}

// =============================================================================
// Benchmark for one R value
// =============================================================================

static void run(const Config &cfg, Dataset &ds, uint32_t R) {
  // Compute cache pages
  std::size_t row =
      4 + static_cast<std::size_t>(R) * 4 + static_cast<std::size_t>(ds.dim_) * 4;
  uint32_t npb = std::max(1U, static_cast<uint32_t>(kDataBlockSize / row));
  uint32_t blocks = (ds.data_num_ + npb - 1) / npb;
  auto cache = static_cast<std::size_t>(
      static_cast<float>(blocks) * cfg.benchmark_.cache_ratio_);

  uint32_t ef_con = std::max(R * 2, 64U);
  uint32_t nq = cfg.search_.num_queries_ > 0
                    ? std::min(cfg.search_.num_queries_, ds.query_num_)
                    : ds.query_num_;

  auto idx_path = cfg.benchmark_.index_dir_ + "/idx_r" + std::to_string(R);

  printf("\n");
  printf("============================================================\n");
  printf("  R=%u  L_build=%u  threads=%u\n", R, ef_con, cfg.index_.num_threads_);
  printf("  cache=%zu pages (%.0f MB, %.1f%%)\n",
         cache, cache * kDataBlockSize / 1048576.0,
         cfg.benchmark_.cache_ratio_ * 100);
  printf("  queries=%u  topk=%u  measure_rounds=%u\n",
         nq, cfg.search_.topk_, cfg.benchmark_.measure_rounds_);
  printf("============================================================\n\n");

  // --- Build (skip if index already exists) ---
  bool index_exists = std::filesystem::exists(idx_path + ".data") &&
                      std::filesystem::exists(idx_path + ".meta");
  if (index_exists) {
    printf("Index already exists at %s, skipping build.\n\n", idx_path.c_str());
    fflush(stdout);
  } else {
    std::filesystem::create_directories(cfg.benchmark_.index_dir_);

    auto metric = kMetricMap[cfg.dataset_.metric_];
    std::vector<float> raw;
    uint32_t raw_num = 0;
    uint32_t raw_dim = 0;
    load_fvecs(cfg.dataset_.data_path_, raw, raw_num, raw_dim);

    auto sp = std::make_shared<RawSpace<>>(ds.data_num_, ds.dim_, metric);
    sp->fit(raw.data(), ds.data_num_);
    raw.clear();
    raw.shrink_to_fit();

    auto params = DiskANNBuildParams()
                      .set_max_degree(R)
                      .set_ef_construction(ef_con)
                      .set_num_iterations(2)
                      .set_num_threads(cfg.index_.num_threads_);

    printf("Building index (R=%u, ef_con=%u, threads=%u)...\n", R, ef_con, cfg.index_.num_threads_);
    fflush(stdout);
    Timer build_timer;
    DiskANNIndex<>::build(sp, idx_path, params);
    double build_sec = build_timer.elapsed_s();
    printf("Build completed in %.1f seconds (%.0f vectors/sec)\n\n",
           build_sec, ds.data_num_ / std::max(build_sec, 1e-9));
    fflush(stdout);
  }

  // --- Search sweep (load index once, reuse across beam widths) ---
  printf("%-6s %-10s %-12s %-16s %-16s %-12s\n",
         "L", "Beamwidth", "Recall@10", "QPS", "Mean Lat(us)", "P99.9 Lat(us)");
  printf("========================================================================\n");
  fflush(stdout);

  DiskANNIndex<> idx;
  idx.load(idx_path, cache);

  for (uint32_t bw : cfg.benchmark_.beam_widths_) {
    for (uint32_t ef : cfg.search_.ef_search_) {
      if (ef < cfg.search_.topk_) {
        continue;
      }

      auto stats = run_search(idx, ds, nq, cfg.search_.topk_, ef, bw,
                              cfg.benchmark_.warmup_queries_,
                              cfg.benchmark_.measure_rounds_);

      printf("%-6u %-10u %-12.2f %-16.2f %-16.2f %-12.2f\n",
             ef, bw, stats.recall_ * 100, stats.qps_,
             stats.mean_latency_us_, stats.p999_latency_us_);
      fflush(stdout);
    }
  }

  idx.close();
  printf("========================================================================\n\n");
  fflush(stdout);
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

  DatasetConfig dc;
  dc.name_ = "bench";
  dc.dir_ = std::filesystem::path(cfg.dataset_.data_path_).parent_path();
  dc.data_file_ = cfg.dataset_.data_path_;
  dc.query_file_ = cfg.dataset_.query_path_;
  dc.gt_file_ = cfg.dataset_.gt_path_;
  auto ds = load_dataset(dc);
  ds.data_.clear();
  ds.data_.shrink_to_fit();

  if (cfg.search_.num_queries_ > 0 && cfg.search_.num_queries_ < ds.query_num_) {
    ds.query_num_ = cfg.search_.num_queries_;
    ds.queries_.resize(static_cast<size_t>(ds.query_num_) * ds.dim_);
    ds.ground_truth_.resize(static_cast<size_t>(ds.query_num_) * ds.gt_dim_);
  }

  printf("Loaded: %u vectors, dim=%u, %u queries, gt_dim=%u\n\n",
         ds.data_num_, ds.dim_, ds.query_num_, ds.gt_dim_);

  for (uint32_t r : cfg.index_.r_) {
    run(cfg, ds, r);
  }

  printf("Done.\n");
  return 0;
}
