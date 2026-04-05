/*
 * DiskANN Delete/Insert/Search benchmark.
 *
 * Usage:
 *   ./diskann_update_bench <config.toml>
 */

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "bench_config.hpp"

#include "index/diskann/diskann_index.hpp"
#include "index/diskann/diskann_params.hpp"
#include "index/diskann/diskann_searcher.hpp"
#include "space/raw_space.hpp"
#include "storage/buffer/replacer/clock.hpp"
#include "storage/diskann/data_file.hpp"
#include "utils/dataset_utils.hpp"
#include "utils/evaluate.hpp"
#include "utils/log.hpp"
#include "utils/timer.hpp"
#include "utils/types.hpp"

using alaya::BufferPool;
using alaya::ClockReplacer;
using alaya::DataFile;
using alaya::Dataset;
using alaya::DatasetConfig;
using alaya::DiskANNBuildParams;
using alaya::DiskANNIndex;
using alaya::DiskANNInsertParams;
using alaya::DiskANNSearchParams;
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

struct DiskannBenchSection {
  std::string index_dir_ = "/tmp/diskann_bench";
  float delete_ratio_ = 0.01F;
  uint32_t num_rounds_ = 3;
  float cache_ratio_ = 0.20F;
};

struct Config {
  bench::DatasetSection dataset_;
  bench::IndexSection index_;
  bench::SearchSection search_;
  DiskannBenchSection benchmark_;

  void print() const {
    dataset_.print();
    index_.print();
    search_.print();
    printf("[benchmark]\n");
    printf("  index_dir  = %s\n", benchmark_.index_dir_.c_str());
    printf("  delete_ratio = %.2f\n", benchmark_.delete_ratio_);
    printf("  num_rounds = %u\n", benchmark_.num_rounds_);
    printf("  cache_ratio= %.2f\n", benchmark_.cache_ratio_);
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
        bench::toml_get<std::string>(*b, "index_dir", "/tmp/diskann_bench");
    c.benchmark_.delete_ratio_ =
        bench::toml_get<float>(*b, "delete_ratio", 0.01F);
    c.benchmark_.num_rounds_ =
        bench::toml_get<uint32_t>(*b, "num_rounds", 3);
    c.benchmark_.cache_ratio_ =
        bench::toml_get<float>(*b, "cache_ratio", 0.20F);
  }

  if (c.dataset_.data_path_.empty() || c.dataset_.query_path_.empty() ||
      c.dataset_.gt_path_.empty()) {
    fprintf(stderr, "Config must specify [dataset] data_path, query_path, gt_path\n");
    exit(1);
  }
  return c;
}

// =============================================================================
// Helpers
// =============================================================================

/// Tracks peak RSS in a background thread (polls every 5 ms).
class PeakRSSTracker {
 public:
  void start() {
    peak_kb_.store(rss_kb(), std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] {
      while (running_.load(std::memory_order_relaxed)) {
        std::int64_t cur = rss_kb();
        std::int64_t prev = peak_kb_.load(std::memory_order_relaxed);
        while (cur > prev &&
               !peak_kb_.compare_exchange_weak(prev, cur, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    });
  }

  /// Stop the tracker and return the peak RSS in MB.
  auto stop_mb() -> double {
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) {
      thread_.join();
    }
    return static_cast<double>(peak_kb_.load(std::memory_order_relaxed)) / 1024.0;
  }

 private:
  static auto rss_kb() -> std::int64_t {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.starts_with("VmRSS:")) {
        std::int64_t kb = 0;
        kb = static_cast<std::int64_t>(std::stoll(line.substr(std::string("VmRSS:").size())));
        return kb;
      }
    }
#endif
    return 0;
  }

  std::atomic<std::int64_t> peak_kb_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
};

/// Print buffer pool stats + Direct I/O status + RSS (and optional peak RSS).
static void print_io_stats(DiskANNIndex<> &idx,
                           const char *label,
                           double peak_rss = -1.0,
                           double prev_rss = -1.0) {
  auto *srch = idx.searcher();
  if (srch == nullptr) {
    return;
  }
  auto &stats = srch->buffer_pool().stats();
  bool direct_io = srch->bypasses_page_cache();
  double rss = bench::get_rss_mb();
  double delta = (prev_rss > 0.0 && rss > 0.0) ? rss - prev_rss : 0.0;
  if (peak_rss > 0.0) {
    printf("  [%s] Direct-I/O=%s  RSS=%.0fMB (peak=%.0fMB delta=%+.0fMB)  "
           "BP(hits=%lu misses=%lu hit=%.1f%% evict=%lu)  "
           "cache(ins=%zu del=%zu freelist=%zu)\n",
           label, direct_io ? "YES" : "NO", rss, peak_rss, delta,
           stats.hits_.load(std::memory_order_relaxed),
           stats.misses_.load(std::memory_order_relaxed),
           stats.hit_rate() * 100,
           stats.evictions_.load(std::memory_order_relaxed),
           srch->inserted_edge_cache_size(),
           srch->deleted_neighbor_cache_size(),
           srch->freelist_depth());
  } else {
    printf("  [%s] Direct-I/O=%s  RSS=%.0fMB  "
           "BP(hits=%lu misses=%lu hit=%.1f%% evict=%lu)  "
           "cache(ins=%zu del=%zu freelist=%zu)\n",
           label, direct_io ? "YES" : "NO", rss,
           stats.hits_.load(std::memory_order_relaxed),
           stats.misses_.load(std::memory_order_relaxed),
           stats.hit_rate() * 100,
           stats.evictions_.load(std::memory_order_relaxed),
           srch->inserted_edge_cache_size(),
           srch->deleted_neighbor_cache_size(),
           srch->freelist_depth());
  }
}

static void copy_index(const std::string &src, const std::string &dst) {
  for (const char *ext : {".data", ".meta"}) {
    auto s = src + ext;
    if (std::filesystem::exists(s)) {
      std::filesystem::copy_file(s, dst + ext, std::filesystem::copy_options::overwrite_existing);
    }
  }
}

static auto batch_read_vectors(const std::string &data_path,
                               const std::vector<uint32_t> &ids,
                               uint32_t capacity, uint32_t dim, uint32_t R)
    -> std::vector<std::vector<float>> {
  BufferPool<uint32_t, ClockReplacer> bp(256, kDataBlockSize);
  DataFile<float, uint32_t, ClockReplacer> df(&bp);
  df.open(data_path, capacity, dim, R, false);
  std::vector<std::vector<float>> vecs;
  vecs.reserve(ids.size());
  for (uint32_t id : ids) {
    auto node = df.get_node(id);
    auto v = node.vector();
    vecs.emplace_back(v.begin(), v.end());
  }
  return vecs;
}

// =============================================================================
// Measurements
// =============================================================================

static auto measure_recall(DiskANNIndex<> &idx, const Dataset &ds,
                           uint32_t nq, uint32_t topk, uint32_t ef, uint32_t bw,
                           uint32_t search_threads) -> float {
  DiskANNSearchParams p;
  p.set_ef_search(ef).set_beam_width(bw).set_num_threads(search_threads);
  std::vector<uint32_t> ids(static_cast<size_t>(nq) * topk);
  idx.batch_search(ds.queries_.data(), nq, topk, ids.data(), p);
  return calc_recall(ids.data(), ds.ground_truth_.data(), nq, ds.gt_dim_, topk);
}

static auto measure_qps(DiskANNIndex<> &idx,
                        const Dataset &ds,
                        uint32_t nq,
                        uint32_t topk,
                        uint32_t ef,
                        uint32_t bw,
                        uint32_t search_threads) -> double {
  DiskANNSearchParams p;
  p.set_ef_search(ef).set_beam_width(bw).set_num_threads(search_threads);
  std::vector<uint32_t> ids(static_cast<size_t>(nq) * topk);
  for (uint32_t q = 0; q < std::min(10U, nq); ++q) {
    idx.search(ds.queries_.data() + (q % ds.query_num_) * ds.dim_, topk,
               ids.data() + q * topk, p);
  }
  Timer t;
  idx.batch_search(ds.queries_.data(), nq, topk, ids.data(), p);
  double elapsed = t.elapsed_s();
  return nq / std::max(elapsed, 1e-9);
}

// =============================================================================
// Benchmark for one R value
// =============================================================================

static void run(const Config &cfg, Dataset &ds, uint32_t R) {
  // Compute cache
  std::size_t row =
      4 + static_cast<std::size_t>(R) * 4 + static_cast<std::size_t>(ds.dim_) * 4;
  uint32_t npb = std::max(1U, static_cast<uint32_t>(4096 / row));
  uint32_t blocks = (ds.data_num_ + npb - 1) / npb;
  auto cache = static_cast<std::size_t>(static_cast<float>(blocks) * cfg.benchmark_.cache_ratio_);
  uint32_t ef_con = std::max(R * 2, 64U);
  auto num_del = static_cast<uint32_t>(ds.data_num_ * cfg.benchmark_.delete_ratio_);
  uint32_t nq = cfg.search_.num_queries_ > 0
                    ? std::min(cfg.search_.num_queries_, ds.query_num_)
                    : ds.query_num_;

  auto idx_path = cfg.benchmark_.index_dir_ + "/idx_r" + std::to_string(R);

  printf("\n==========================================================\n");
  printf("R=%u  L=%u  cache=%zu pages (%.0f MB, %.0f%%)\n",
         R, ef_con, cache, cache * 4096.0 / 1048576, cfg.benchmark_.cache_ratio_ * 100);
  printf("==========================================================\n\n");

  // Build
  for (const char *ext : {".data", ".meta"}) {
    std::filesystem::remove(idx_path + ext);
  }
  std::filesystem::create_directories(cfg.benchmark_.index_dir_);
  {
    auto metric = kMetricMap[cfg.dataset_.metric_];
    // Load raw vectors locally so they are freed after fit(), before any search
    // benchmarks run. This keeps RSS clean (buffer pool only, not raw dataset).
    std::vector<float> raw;
    [[maybe_unused]] uint32_t raw_num = 0;
    [[maybe_unused]] uint32_t raw_dim = 0;
    load_fvecs(cfg.dataset_.data_path_, raw, raw_num, raw_dim);
    auto sp = std::make_shared<RawSpace<>>(ds.data_num_, ds.dim_, metric);
    sp->fit(raw.data(), ds.data_num_);
    raw.clear();
    raw.shrink_to_fit();  // RawSpace owns a copy; free ~1 GB now
    auto p = DiskANNBuildParams()
                 .set_max_degree(R)
                 .set_ef_construction(ef_con)
                 .set_num_iterations(2)
                 .set_num_threads(cfg.index_.num_threads_);
    DiskANNIndex<>::build(sp, idx_path, p);
    printf("\n");
  }

  // --- 1. Baseline ---
  printf("[Baseline]\n");
  {
    DiskANNIndex<> idx;
    idx.load(idx_path, cache);
    print_io_stats(idx, "load");
    PeakRSSTracker srch_tracker;
    srch_tracker.start();
    for (uint32_t ef : cfg.search_.ef_search_) {
      float rc = measure_recall(idx, ds, nq, cfg.search_.topk_, ef,
                                cfg.index_.beam_width_, cfg.search_.search_threads_);
      double qps = measure_qps(idx, ds, nq, cfg.search_.topk_, ef,
                               cfg.index_.beam_width_, cfg.search_.search_threads_);
      printf("  ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.search_.topk_, rc, qps);
    }
    print_io_stats(idx, "after search", srch_tracker.stop_mb());
  }

  // --- 2. Delete-only ---
  printf("\n[Delete-only %.0f%% (%u)]\n", cfg.benchmark_.delete_ratio_ * 100, num_del);
  {
    auto tmp = std::filesystem::temp_directory_path() / "bench_del";
    std::filesystem::create_directories(tmp);
    auto tp = (tmp / "idx").string();
    copy_index(idx_path, tp);

    DiskANNIndex<> idx;
    idx.load(tp, cache, true);

    std::vector<uint32_t> pool(ds.data_num_);
    std::iota(pool.begin(), pool.end(), 0U);
    std::mt19937 rng(42);
    std::shuffle(pool.begin(), pool.end(), rng);

    uint32_t n = 0;
    PeakRSSTracker del_tracker;
    del_tracker.start();
    Timer t;
    for (uint32_t i = 0; i < pool.size() && n < num_del; ++i) {
      try { idx.delete_vector(pool[i]); ++n; }
      catch (const std::logic_error &e) { (void)e; /* already deleted */ }
    }
    double del_elapsed = t.elapsed_s();
    printf("  %u deleted in %.2fs  QPS=%.1f\n", n, del_elapsed, n / std::max(del_elapsed, 1e-9));
    for (uint32_t ef : cfg.search_.ef_search_) {
      float rc = measure_recall(idx, ds, nq, cfg.search_.topk_, ef,
                                cfg.index_.beam_width_, cfg.search_.search_threads_);
      printf("  ef=%u  recall@%u=%.4f\n", ef, cfg.search_.topk_, rc);
    }
    print_io_stats(idx, "after delete+search", del_tracker.stop_mb());
    idx.close();
    std::filesystem::remove_all(tmp);
  }

  // --- 3. Delete + Insert rounds ---
  printf("\n[Delete %.0f%% + Insert %.0f%%, %u rounds]\n",
         cfg.benchmark_.delete_ratio_ * 100, cfg.benchmark_.delete_ratio_ * 100,
         cfg.benchmark_.num_rounds_);
  {
    auto tmp = std::filesystem::temp_directory_path() / "bench_di";
    std::filesystem::create_directories(tmp);
    auto tp = (tmp / "idx").string();
    copy_index(idx_path, tp);

    DiskANNIndex<> idx;
    idx.load(tp, cache, true);
    auto &srch = idx.get_searcher();
    print_io_stats(idx, "load");

    // Baseline
    for (uint32_t ef : cfg.search_.ef_search_) {
      float rc = measure_recall(idx, ds, nq, cfg.search_.topk_, ef,
                                cfg.index_.beam_width_, cfg.search_.search_threads_);
      double qps = measure_qps(idx, ds, nq, cfg.search_.topk_, ef,
                               cfg.index_.beam_width_, cfg.search_.search_threads_);
      printf("  baseline  ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.search_.topk_, rc, qps);
    }

    // All live IDs — re-shuffled each round so deletion targets the full live set,
    // including nodes that were previously deleted and re-inserted.
    std::vector<uint32_t> pool(ds.data_num_);
    std::iota(pool.begin(), pool.end(), 0U);
    std::mt19937 rng(42);

    DiskANNInsertParams ip;
    ip.set_ef_construction(ef_con).set_alpha(1.2F).set_beam_width(cfg.index_.beam_width_);

    double prev_rss = bench::get_rss_mb();
    for (uint32_t rd = 1; rd <= cfg.benchmark_.num_rounds_; ++rd) {
      // Re-shuffle each round to sample randomly from the full live set.
      std::shuffle(pool.begin(), pool.end(), rng);
      std::vector<uint32_t> ids(pool.begin(), pool.begin() + num_del);

      // Read vectors from disk + run delete/insert/search under peak tracker
      PeakRSSTracker round_tracker;
      round_tracker.start();

      auto vecs = batch_read_vectors(tp + ".data", ids, srch.capacity(), ds.dim_, R);

      // Delete
      std::vector<uint32_t> del_ids;
      std::vector<std::vector<float>> del_vecs;
      del_ids.reserve(ids.size());
      del_vecs.reserve(ids.size());
      Timer td;
      for (size_t i = 0; i < ids.size(); ++i) {
        try {
          idx.delete_vector(ids[i]);
          del_ids.push_back(ids[i]);
          del_vecs.push_back(std::move(vecs[i]));
        } catch (const std::logic_error &e) { (void)e; /* already deleted */ }
      }
      double d_sec = td.elapsed_s();

      // Insert
      Timer ti;
      for (size_t i = 0; i < del_ids.size(); ++i) {
        idx.insert(del_vecs[i].data(), del_ids[i], ip);
      }
      double i_sec = ti.elapsed_s();

      auto dn = static_cast<double>(del_ids.size());
      printf("  round %u  del=%.2fs(%.0f QPS)  ins=%.2fs(%.1f QPS)\n",
             rd, d_sec, dn / std::max(d_sec, 1e-9), i_sec, dn / std::max(i_sec, 1e-9));
      for (uint32_t ef : cfg.search_.ef_search_) {
        float rc = measure_recall(idx, ds, nq, cfg.search_.topk_, ef,
                                  cfg.index_.beam_width_, cfg.search_.search_threads_);
        double qps = measure_qps(idx, ds, nq, cfg.search_.topk_, ef,
                                 cfg.index_.beam_width_, cfg.search_.search_threads_);
        printf("           ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.search_.topk_, rc, qps);
      }
      print_io_stats(idx, ("round " + std::to_string(rd)).c_str(), round_tracker.stop_mb(),
                     prev_rss);
      prev_rss = bench::get_rss_mb();
    }
    idx.close();
    std::filesystem::remove_all(tmp);
  }
  printf("\n");
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

  // Load dataset via existing infrastructure
  DatasetConfig dc;
  dc.name_ = "bench";
  dc.dir_ = std::filesystem::path(cfg.dataset_.data_path_).parent_path();
  dc.data_file_ = cfg.dataset_.data_path_;
  dc.query_file_ = cfg.dataset_.query_path_;
  dc.gt_file_ = cfg.dataset_.gt_path_;
  auto ds = load_dataset(dc);
  // Raw vectors are only needed inside run() during the build phase.
  // Free them here so the ~1 GB allocation does not inflate RSS measurements.
  ds.data_.clear();
  ds.data_.shrink_to_fit();

  if (cfg.search_.num_queries_ > 0 && cfg.search_.num_queries_ < ds.query_num_) {
    ds.query_num_ = cfg.search_.num_queries_;
    ds.queries_.resize(static_cast<std::size_t>(ds.query_num_) * ds.dim_);
    ds.ground_truth_.resize(static_cast<std::size_t>(ds.query_num_) * ds.gt_dim_);
  }

  printf("Loaded: %u vectors, dim=%u, %u queries, gt_dim=%u\n\n",
         ds.data_num_, ds.dim_, ds.query_num_, ds.gt_dim_);

  for (uint32_t r : cfg.index_.r_) {
    run(cfg, ds, r);
  }

  printf("Done.\n");
  return 0;
}
