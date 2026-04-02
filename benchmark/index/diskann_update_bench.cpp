/*
 * DiskANN Delete/Insert/Search benchmark.
 *
 * Usage:
 *   ./diskann_bench <config.toml>
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

#include <toml++/toml.hpp>

#include "index/diskann/diskann_index.hpp"
#include "index/diskann/diskann_params.hpp"
#include "index/diskann/diskann_searcher.hpp"
#include "space/raw_space.hpp"
#include "storage/buffer/replacer/clock.hpp"
#include "storage/diskann/data_file.hpp"
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

  // [index]
  std::vector<uint32_t> R = {16, 32};
  uint32_t beam_width = 4;
  uint32_t num_threads = 0;  // 0 = min(hardware_concurrency, 60)

  // [search]
  std::vector<uint32_t> ef_search = {4, 8};
  uint32_t topk = 10;
  uint32_t num_queries = 200;
  uint32_t search_threads = 1;

  // [benchmark]
  std::string index_dir = "/tmp/diskann_bench";
  float delete_ratio = 0.01F;
  uint32_t num_rounds = 3;
  float cache_ratio = 0.20F;

  void print() const {
    printf("[dataset]\n");
    printf("  data_path  = %s\n", data_path.c_str());
    printf("  query_path = %s\n", query_path.c_str());
    printf("  gt_path    = %s\n", gt_path.c_str());
    printf("  metric     = %s\n", metric.c_str());
    printf("[index]\n");
    printf("  R          = [");
    for (size_t i = 0; i < R.size(); ++i) printf("%s%u", i ? ", " : "", R[i]);
    printf("]\n");
    printf("  beam_width = %u\n", beam_width);
    printf("  num_threads= %u\n", num_threads);
    printf("[search]\n");
    printf("  ef_search  = [");
    for (size_t i = 0; i < ef_search.size(); ++i) printf("%s%u", i ? ", " : "", ef_search[i]);
    printf("]\n");
    printf("  topk       = %u\n", topk);
    printf("  num_queries= %u\n", num_queries);
    printf("  search_threads= %u\n", search_threads);
    printf("[benchmark]\n");
    printf("  index_dir  = %s\n", index_dir.c_str());
    printf("  delete_ratio = %.2f\n", delete_ratio);
    printf("  num_rounds = %u\n", num_rounds);
    printf("  cache_ratio= %.2f\n", cache_ratio);
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
    c.data_path  = ds->get("data_path")->value_or(std::string{});
    c.query_path = ds->get("query_path")->value_or(std::string{});
    c.gt_path    = ds->get("gt_path")->value_or(std::string{});
    c.metric     = ds->get("metric")->value_or(std::string{"L2"});
  }
  if (auto idx = tbl["index"].as_table()) {
    if (auto *a = idx->get("R"); a && a->is_array())
      c.R = toml_array_to_vec<uint32_t>(*a->as_array());
    c.beam_width = idx->get("beam_width")->value_or(4);
    if (auto *v = idx->get("num_threads")) c.num_threads = v->value_or(0);
  }
  if (auto s = tbl["search"].as_table()) {
    if (auto *a = s->get("ef_search"); a && a->is_array())
      c.ef_search = toml_array_to_vec<uint32_t>(*a->as_array());
    c.topk        = s->get("topk")->value_or(10);
    c.num_queries = s->get("num_queries")->value_or(200);
    if (auto *v = s->get("search_threads")) c.search_threads = v->value_or(1);
  }
  if (auto b = tbl["benchmark"].as_table()) {
    c.index_dir    = b->get("index_dir")->value_or(std::string{"/tmp/diskann_bench"});
    c.delete_ratio = static_cast<float>(b->get("delete_ratio")->value_or(0.01));
    c.num_rounds   = b->get("num_rounds")->value_or(3);
    c.cache_ratio  = static_cast<float>(b->get("cache_ratio")->value_or(0.20));
  }

  if (c.data_path.empty() || c.query_path.empty() || c.gt_path.empty()) {
    fprintf(stderr, "Config must specify [dataset] data_path, query_path, gt_path\n");
    exit(1);
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

/// Read current process RSS from /proc/self/status (Linux only).
static auto get_rss_mb() -> double {
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      // Format: "VmRSS:    12345 kB"
      long kb = 0;
      std::sscanf(line.c_str(), "VmRSS: %ld", &kb);
      return static_cast<double>(kb) / 1024.0;
    }
  }
#endif
  return -1.0;
}

/// Tracks peak RSS in a background thread (polls every 5 ms).
class PeakRSSTracker {
 public:
  void start() {
    peak_kb_.store(rss_kb(), std::memory_order_relaxed);
    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread([this] {
      while (running_.load(std::memory_order_relaxed)) {
        long cur = rss_kb();
        long prev = peak_kb_.load(std::memory_order_relaxed);
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
  static auto rss_kb() -> long {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        long kb = 0;
        std::sscanf(line.c_str(), "VmRSS: %ld", &kb);
        return kb;
      }
    }
#endif
    return 0;
  }

  std::atomic<long> peak_kb_{0};
  std::atomic<bool> running_{false};
  std::thread thread_;
};

/// Print buffer pool stats + Direct I/O status + RSS (and optional peak RSS).
static void print_io_stats(DiskANNIndex<> &idx,
                           const char *label,
                           double peak_rss = -1.0,
                           double prev_rss = -1.0) {
  auto *srch = idx.searcher();
  if (srch == nullptr) return;
  auto &stats = srch->buffer_pool().stats();
  bool direct_io = srch->bypasses_page_cache();
  double rss = get_rss_mb();
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
    if (std::filesystem::exists(s))
      std::filesystem::copy_file(s, dst + ext, std::filesystem::copy_options::overwrite_existing);
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

static auto measure_qps(DiskANNIndex<> &idx, const Dataset &ds,
                         uint32_t nq, uint32_t topk, uint32_t ef, uint32_t bw,
                         uint32_t search_threads) -> double {
  DiskANNSearchParams p;
  p.set_ef_search(ef).set_beam_width(bw).set_num_threads(search_threads);
  std::vector<uint32_t> ids(static_cast<size_t>(nq) * topk);
  for (uint32_t q = 0; q < std::min(10U, nq); ++q) {
    idx.search(ds.queries_.data() + (q % ds.query_num_) * ds.dim_, topk,
               ids.data() + q * topk, p);
  }
  Stopwatch t;
  idx.batch_search(ds.queries_.data(), nq, topk, ids.data(), p);
  double elapsed = t.sec();
  return nq / std::max(elapsed, 1e-9);
}

// =============================================================================
// Benchmark for one R value
// =============================================================================

static void run(const Config &cfg, Dataset &ds, uint32_t R) {
  // Compute cache
  size_t row = 4 + size_t(R) * 4 + size_t(ds.dim_) * 4;
  uint32_t npb = std::max(1U, uint32_t(4096 / row));
  uint32_t blocks = (ds.data_num_ + npb - 1) / npb;
  auto cache = static_cast<size_t>(float(blocks) * cfg.cache_ratio);
  uint32_t ef_con = std::max(R * 2, 64U);
  uint32_t num_del = static_cast<uint32_t>(ds.data_num_ * cfg.delete_ratio);

  auto idx_path = cfg.index_dir + "/idx_r" + std::to_string(R);

  printf("\n==========================================================\n");
  printf("R=%u  L=%u  cache=%zu pages (%.0f MB, %.0f%%)\n",
         R, ef_con, cache, cache * 4096.0 / 1048576, cfg.cache_ratio * 100);
  printf("==========================================================\n\n");

  // Build
  for (const char *ext : {".data", ".meta"})
    std::filesystem::remove(idx_path + ext);
  std::filesystem::create_directories(cfg.index_dir);
  {
    auto metric = (cfg.metric == "IP") ? MetricType::IP : MetricType::L2;
    // Load raw vectors locally so they are freed after fit(), before any search
    // benchmarks run. This keeps RSS clean (buffer pool only, not raw dataset).
    std::vector<float> raw;
    [[maybe_unused]] uint32_t raw_num = 0;
    [[maybe_unused]] uint32_t raw_dim = 0;
    load_fvecs(cfg.data_path, raw, raw_num, raw_dim);
    auto sp = std::make_shared<RawSpace<>>(ds.data_num_, ds.dim_, metric);
    sp->fit(raw.data(), ds.data_num_);
    raw.clear();
    raw.shrink_to_fit();  // RawSpace owns a copy; free ~1 GB now
    auto p = DiskANNBuildParams()
                 .set_max_degree(R)
                 .set_ef_construction(ef_con)
                 .set_num_iterations(2)
                 .set_num_threads(cfg.num_threads);
    printf("Building... ");
    fflush(stdout);
    Stopwatch t;
    DiskANNIndex<>::build(sp, idx_path, p);
    printf("%.1fs\n\n", t.sec());
  }

  // --- 1. Baseline ---
  printf("[Baseline]\n");
  {
    DiskANNIndex<> idx;
    idx.load(idx_path, cache);
    print_io_stats(idx, "load");
    PeakRSSTracker srch_tracker;
    srch_tracker.start();
    for (uint32_t ef : cfg.ef_search) {
      float rc = measure_recall(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
      double qps = measure_qps(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
      printf("  ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.topk, rc, qps);
    }
    print_io_stats(idx, "after search", srch_tracker.stop_mb());
  }

  // --- 2. Delete-only ---
  printf("\n[Delete-only %.0f%% (%u)]\n", cfg.delete_ratio * 100, num_del);
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
    Stopwatch t;
    for (uint32_t i = 0; i < pool.size() && n < num_del; ++i) {
      try { idx.delete_vector(pool[i]); ++n; }
      catch (const std::logic_error &) {}
    }
    double del_elapsed = t.sec();
    printf("  %u deleted in %.2fs  QPS=%.1f\n", n, del_elapsed, n / std::max(del_elapsed, 1e-9));
    for (uint32_t ef : cfg.ef_search) {
      float rc = measure_recall(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
      printf("  ef=%u  recall@%u=%.4f\n", ef, cfg.topk, rc);
    }
    print_io_stats(idx, "after delete+search", del_tracker.stop_mb());
    idx.close();
    std::filesystem::remove_all(tmp);
  }

  // --- 3. Delete + Insert rounds ---
  printf("\n[Delete %.0f%% + Insert %.0f%%, %u rounds]\n",
         cfg.delete_ratio * 100, cfg.delete_ratio * 100, cfg.num_rounds);
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
    for (uint32_t ef : cfg.ef_search) {
      float rc = measure_recall(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
      double qps = measure_qps(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
      printf("  baseline  ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.topk, rc, qps);
    }

    // All live IDs — re-shuffled each round so deletion targets the full live set,
    // including nodes that were previously deleted and re-inserted.
    std::vector<uint32_t> pool(ds.data_num_);
    std::iota(pool.begin(), pool.end(), 0U);
    std::mt19937 rng(42);

    DiskANNInsertParams ip;
    ip.set_ef_construction(ef_con).set_alpha(1.2F).set_beam_width(cfg.beam_width);

    double prev_rss = get_rss_mb();
    for (uint32_t rd = 1; rd <= cfg.num_rounds; ++rd) {
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
      Stopwatch td;
      for (size_t i = 0; i < ids.size(); ++i) {
        try {
          idx.delete_vector(ids[i]);
          del_ids.push_back(ids[i]);
          del_vecs.push_back(std::move(vecs[i]));
        } catch (const std::logic_error &) {}
      }
      double d_sec = td.sec();

      // Insert
      Stopwatch ti;
      for (size_t i = 0; i < del_ids.size(); ++i)
        idx.insert(del_vecs[i].data(), del_ids[i], ip);
      double i_sec = ti.sec();

      auto dn = static_cast<double>(del_ids.size());
      printf("  round %u  del=%.2fs(%.0f QPS)  ins=%.2fs(%.1f QPS)\n",
             rd, d_sec, dn / std::max(d_sec, 1e-9), i_sec, dn / std::max(i_sec, 1e-9));
      for (uint32_t ef : cfg.ef_search) {
        float rc = measure_recall(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
        double qps = measure_qps(idx, ds, cfg.num_queries, cfg.topk, ef, cfg.beam_width, cfg.search_threads);
        printf("           ef=%u  recall@%u=%.4f  QPS=%.1f\n", ef, cfg.topk, rc, qps);
      }
      print_io_stats(idx, ("round " + std::to_string(rd)).c_str(), round_tracker.stop_mb(),
                     prev_rss);
      prev_rss = get_rss_mb();
    }
    idx.close();
    std::filesystem::remove_all(tmp);
  }
  printf("\n");
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <config.conf>\n", argv[0]);
    return 1;
  }

  auto cfg = load_config(argv[1]);
  printf("Config:\n");
  cfg.print();
  printf("\n");

  // Load dataset via existing infrastructure
  DatasetConfig dc;
  dc.name_ = "bench";
  dc.dir_ = std::filesystem::path(cfg.data_path).parent_path();
  dc.data_file_ = cfg.data_path;
  dc.query_file_ = cfg.query_path;
  dc.gt_file_ = cfg.gt_path;
  auto ds = load_dataset(dc);
  // Raw vectors are only needed inside run() during the build phase.
  // Free them here so the ~1 GB allocation does not inflate RSS measurements.
  ds.data_.clear();
  ds.data_.shrink_to_fit();

  if (cfg.num_queries < ds.query_num_) {
    ds.query_num_ = cfg.num_queries;
    ds.queries_.resize(size_t(ds.query_num_) * ds.dim_);
    ds.ground_truth_.resize(size_t(ds.query_num_) * ds.gt_dim_);
  }

  printf("Loaded: %u vectors, dim=%u, %u queries, gt_dim=%u\n\n",
         ds.data_num_, ds.dim_, ds.query_num_, ds.gt_dim_);

  for (uint32_t r : cfg.R)
    run(cfg, ds, r);

  printf("Done.\n");
  return 0;
}
