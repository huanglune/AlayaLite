// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// SIFT-scale experiment harness for the LASER streaming-update prototype.
//
//   build   — Vamana + QGBuilder static build on the first N vectors
//   insert  — stream vectors [from, from+count) into an existing index
//   eval    — recall@K / QPS sweep with masked ground truth (live set =
//             id < live_max minus an optional tombstone range)
//
// Examples:
//   bench_laser_update_sift build --base sift_base.fbin --n 900000
//       --prefix /data/laser/sift900k --R 64 --L 200 --threads 128
//   bench_laser_update_sift insert --prefix /data/laser/sift900k --n 900000
//       --base sift_base.fbin --from 900000 --count 100000 --arm alpha
//   bench_laser_update_sift eval --prefix /data/laser/sift900k --n 1000000
//       --query sift_query.fbin --gt sift_gt.ibin --efs 60,80,100,150,200
//
// The insert mode mutates the index files in place — clone the artifact
// directory first when comparing arms.

#include <malloc.h>
#include <omp.h>
#include <future>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "bench_laser_oracles.hpp"
#include "bench_laser_update_sift_support.hpp"
#include "index/graph/laser/qg/detail/qg_updater_core.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/utils/pca_transform.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace {

using alaya::laser::bench::apply_cache_cap_pages;
using alaya::laser::bench::MappedFloatMatrix;
using alaya::laser::bench::parse_cache_cap_pages;

struct FloatMatrix {
  std::vector<float> data;
  uint32_t n = 0;
  uint32_t dim = 0;
  const float *row(size_t i) const { return data.data() + i * dim; }
};

struct IntMatrix {
  std::vector<uint32_t> data;
  uint32_t n = 0;
  uint32_t dim = 0;
  const uint32_t *row(size_t i) const { return data.data() + i * dim; }
};

FloatMatrix read_fbin(const std::string &path, int64_t max_rows = -1) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&dim), 4);
  if (n <= 0 || dim <= 0) {
    throw std::runtime_error("bad fbin header: " + path);
  }
  if (max_rows > 0 && max_rows < n) {
    n = static_cast<int32_t>(max_rows);
  }
  FloatMatrix m;
  m.n = static_cast<uint32_t>(n);
  m.dim = static_cast<uint32_t>(dim);
  m.data.resize(static_cast<size_t>(n) * m.dim);
  f.read(reinterpret_cast<char *>(m.data.data()),
         static_cast<std::streamsize>(m.data.size() * sizeof(float)));
  if (!f) {
    throw std::runtime_error("short read: " + path);
  }
  return m;
}

IntMatrix read_ibin(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("cannot open " + path);
  }
  int32_t n = 0;
  int32_t dim = 0;
  f.read(reinterpret_cast<char *>(&n), 4);
  f.read(reinterpret_cast<char *>(&dim), 4);
  if (n <= 0 || dim <= 0) {
    throw std::runtime_error("bad ibin header: " + path);
  }
  IntMatrix m;
  m.n = static_cast<uint32_t>(n);
  m.dim = static_cast<uint32_t>(dim);
  m.data.resize(static_cast<size_t>(n) * m.dim);
  f.read(reinterpret_cast<char *>(m.data.data()),
         static_cast<std::streamsize>(m.data.size() * sizeof(uint32_t)));
  if (!f) {
    throw std::runtime_error("short read: " + path);
  }
  return m;
}

std::vector<int> read_query_groups(const std::string &path, size_t expected) {
  if (path.empty()) return {};
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<int> groups;
  std::string line;
  while (std::getline(in, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    std::istringstream row(line);
    int group = -1;
    double frac = 0;
    if (!(row >> group >> frac) || group < 0 || group > 2) {
      throw std::runtime_error("bad query group row: " + line);
    }
    groups.push_back(group);
  }
  if (groups.size() != expected) {
    throw std::runtime_error("query_groups/query count mismatch");
  }
  return groups;
}

double telemetry_p50(const alaya::laser::UpdateStats &s) {
  if (s.evict_tel_samples == 0) return -1;
  const uint64_t target = (s.evict_tel_samples + 1) / 2;
  uint64_t cumulative = 0;
  for (size_t rank = 0; rank < s.evict_tel_regret.size(); ++rank) {
    cumulative += s.evict_tel_regret[rank];
    if (cumulative >= target) return static_cast<double>(rank);
  }
  return 3;
}

void print_telemetry(const alaya::laser::UpdateStats &s) {
  const double agree =
      s.evict_tel_samples == 0 ? -1 : static_cast<double>(s.evict_tel_agree) / s.evict_tel_samples;
  const double relerr =
      s.evict_tel_samples == 0 ? -1 : s.evict_tel_relerr_sum / s.evict_tel_samples;
  std::cout << ",tel_agree," << agree << ",tel_regret_p50," << telemetry_p50(s) << ",tel_relerr,"
            << relerr;
}

void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cannot write " + path);
  }
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * static_cast<int64_t>(n) * dim));
}

struct Args {
  std::string mode;
  std::string base, query, gt, prefix;
  std::string query_groups;
  uint64_t n = 0;
  uint64_t from = 0, count = 0;
  uint64_t live_max = 0;
  uint64_t new_id_split = 0;  // eval: also report recall restricted to GT ids >= split
  uint64_t tombstone_from = 0, tombstone_n = 0;
  uint32_t R = 64, L = 200, ef_indexing = 200, threads = 64, beam = 16, topk = 10, runs = 3;
  uint32_t vamana_R = 0;     // build: Vamana degree < degree_bound leaves free slots ("headroom")
  uint32_t reuse_graph = 0;  // build: skip VamanaBuilder, pack an existing <prefix>_vamana.index
  uint32_t main_dim = 0;     // 0 -> dim
  uint32_t pca_preprocessed = 0;  // build: reuse <prefix>_pca_{base.fbin,bin}
  size_t ef_insert = 100, prune_cap = 300, alpha_check_max = 16;
  size_t batch = 4096, insert_threads = 32;
  uint32_t consolidate = 0, r_target = 0;
  size_t consolidate_every = 1;    // churn: consolidate every N rounds; 0 = never
  uint32_t bloom_consolidate = 0;  // 1 = read-only Bloom prefilter before consolidate_row
  uint32_t reuse = 0;              // churn: 1 = allocate from consolidated free-list first
  uint32_t checkpoint_every = 1;   // churn: superblock cadence in rounds
  uint32_t garden = 0;
  double garden_frac = 0.05;
  size_t ef_maintenance = 200, pump_budget = 4;
  uint32_t garden_pump_only = 0;
  std::string garden_policy = "lowdeg";
  float alpha = 1.2F;
  uint64_t seed = 42;
  std::string arm = "alpha";  // none|evict|alpha|full
  double evict_telemetry = 0;
  double evict_margin = 0;
  uint32_t splice = 1;
  uint64_t shuffle_seed = 0;
  uint32_t direct = 0;       // 1 = O_DIRECT write fd (P0.1 attribution arm), 0 = buffered (default)
  uint32_t write_cache = 1;  // 0 = immediate per-patch writes (P0.1-era control)
  size_t cache_cap_pages = 0;              // 0 = retain UpdateParams default
  float dram_gb = 0.0F;                    // eval mode: static node-cache DRAM budget (GB)
  bool arena = false;                      // eval mode: resident-arena search (no beam/AIO)
  size_t maintenance_evict_stride = 4096;  // 0 = phase-boundary-only legacy behavior
  double garden_churn_threshold =
      0.0;                // 0 = garden every call; >0 = skip until churn >= threshold*N
  uint32_t pipeline = 0;  // staged mode only: overlap drain(i) with batch i+1 (2-batch isolation)
  uint32_t stage = 0;     // 1 = stage backlinks + barrier drain; 0 = inline patches into pool
  uint32_t qpatch = 0;    // 1 = lock-free RaBitQ preencode + generation-validated apply
  size_t flush_threads = 0;       // threads for the overlapped flush; 0 = insert_threads
  uint32_t query_threads = 16;    // mixed: closed-loop query clients
  uint32_t mixed_ef = 100;        // mixed: updater pool-search ef
  double mixed_seconds = 10.0;    // mixed: duration of each insert/query window
  uint32_t mixed_rate_pct = 100;  // mixed: 0|25|50|100; 100 is unlimited
  double mixed_full_rate = 0.0;   // token-bucket 100% reference QPS for 25/50
  size_t samples = 1000;          // oracle modes: accepted samples
  std::string csv;                // mixed: append one row per phase
  std::vector<uint32_t> efs = {60, 80, 100, 150, 200, 300};
};

// Keep the historical bare-bench measurement shape: the OpenMP dynamic
// fanout moved out of QuantizedGraph, while every lane invokes the same
// per-call kernel with explicit ef/beam values.
void run_query_lanes(alaya::laser::QuantizedGraph &qg,
                     const float *queries,
                     uint32_t knn,
                     uint32_t *results,
                     size_t num_queries,
                     size_t ef_search,
                     size_t beam_width,
                     uint32_t threads,
                     bool arena = false) {
  const auto query_stride = qg.dimension() + qg.residual_dimension();
  const auto num_queries_signed = static_cast<int64_t>(num_queries);
  const auto threads_signed = static_cast<int>(threads);
#pragma omp parallel for schedule(dynamic) num_threads(threads_signed)
  for (int64_t ii = 0; ii < num_queries_signed; ++ii) {
    const auto i = static_cast<size_t>(ii);
    const float *query = queries + i * query_stride;
    uint32_t *result = results + i * knn;
    if (arena) {
      qg.arena_search_qg(query,
                         knn,
                         result,
                         ef_search,
                         beam_width,
                         /*admission=*/nullptr,
                         /*distances=*/nullptr);
    } else {
      qg.search(query,
                knn,
                result,
                ef_search,
                beam_width,
                /*admission=*/nullptr,
                /*distances=*/nullptr);
    }
  }
}

Args parse(int argc, char **argv) {
  Args a;
  if (argc < 2) {
    throw std::runtime_error(
        "usage: bench_laser_update_sift "
        "<build|insert|churn|eval|mixed|fastscan_oracle|twohop_oracle> [--k v ...]; "
        "churn --efs E0,E1,... reports E0 as round/round_age and later values as round_ef");
  }
  a.mode = argv[1];
  if (a.mode == "--mixed") a.mode = "mixed";
  for (int i = 2; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    const std::string v = argv[i + 1];
    if (k == "--base")
      a.base = v;
    else if (k == "--query")
      a.query = v;
    else if (k == "--gt")
      a.gt = v;
    else if (k == "--prefix")
      a.prefix = v;
    else if (k == "--query_groups")
      a.query_groups = v;
    else if (k == "--n")
      a.n = std::stoull(v);
    else if (k == "--from")
      a.from = std::stoull(v);
    else if (k == "--count")
      a.count = std::stoull(v);
    else if (k == "--live_max")
      a.live_max = std::stoull(v);
    else if (k == "--new_id_split")
      a.new_id_split = std::stoull(v);
    else if (k == "--tombstone_from")
      a.tombstone_from = std::stoull(v);
    else if (k == "--tombstone_n")
      a.tombstone_n = std::stoull(v);
    else if (k == "--R")
      a.R = std::stoul(v);
    else if (k == "--vamana_R")
      a.vamana_R = std::stoul(v);
    else if (k == "--reuse_graph")
      a.reuse_graph = std::stoul(v);
    else if (k == "--L")
      a.L = std::stoul(v);
    else if (k == "--ef_indexing")
      a.ef_indexing = std::stoul(v);
    else if (k == "--threads")
      a.threads = std::stoul(v);
    else if (k == "--beam")
      a.beam = std::stoul(v);
    else if (k == "--topk")
      a.topk = std::stoul(v);
    else if (k == "--runs")
      a.runs = std::stoul(v);
    else if (k == "--main_dim")
      a.main_dim = std::stoul(v);
    else if (k == "--pca_preprocessed")
      a.pca_preprocessed = std::stoul(v);
    else if (k == "--ef_insert")
      a.ef_insert = std::stoull(v);
    else if (k == "--prune_cap")
      a.prune_cap = std::stoull(v);
    else if (k == "--alpha_check_max")
      a.alpha_check_max = std::stoull(v);
    else if (k == "--evict_telemetry")
      a.evict_telemetry = std::stod(v);
    else if (k == "--evict_margin")
      a.evict_margin = std::stod(v);
    else if (k == "--splice")
      a.splice = std::stoul(v);
    else if (k == "--shuffle_seed")
      a.shuffle_seed = std::stoull(v);
    else if (k == "--batch")
      a.batch = std::stoull(v);
    else if (k == "--insert_threads")
      a.insert_threads = std::stoull(v);
    else if (k == "--consolidate")
      a.consolidate = std::stoul(v);
    else if (k == "--consolidate_every")
      a.consolidate_every = std::stoull(v);
    else if (k == "--bloom_consolidate")
      a.bloom_consolidate = std::stoul(v);
    else if (k == "--r_target")
      a.r_target = std::stoul(v);
    else if (k == "--reuse")
      a.reuse = std::stoul(v);
    else if (k == "--checkpoint_every")
      a.checkpoint_every = std::stoul(v);
    else if (k == "--garden")
      a.garden = std::stoul(v);
    else if (k == "--garden_frac")
      a.garden_frac = std::stod(v);
    else if (k == "--ef_maintenance")
      a.ef_maintenance = std::stoull(v);
    else if (k == "--pump_budget")
      a.pump_budget = std::stoull(v);
    else if (k == "--garden_pump_only")
      a.garden_pump_only = std::stoul(v);
    else if (k == "--garden_policy")
      a.garden_policy = v;
    else if (k == "--direct")
      a.direct = std::stoul(v);
    else if (k == "--write_cache")
      a.write_cache = std::stoul(v);
    else if (k == "--cache_cap_pages")
      a.cache_cap_pages = parse_cache_cap_pages(v);
    else if (k == "--dram_gb")
      a.dram_gb = std::stof(v);
    else if (k == "--arena")
      a.arena = std::stoul(v) != 0;
    else if (k == "--maintenance_evict_stride")
      a.maintenance_evict_stride = std::stoull(v);
    else if (k == "--garden_churn_threshold")
      a.garden_churn_threshold = std::stod(v);
    else if (k == "--pipeline")
      a.pipeline = std::stoul(v);
    else if (k == "--stage")
      a.stage = std::stoul(v);
    else if (k == "--qpatch")
      a.qpatch = std::stoul(v);
    else if (k == "--flush_threads")
      a.flush_threads = std::stoull(v);
    else if (k == "--query_threads")
      a.query_threads = std::stoul(v);
    else if (k == "--mixed_ef")
      a.mixed_ef = std::stoul(v);
    else if (k == "--mixed_seconds")
      a.mixed_seconds = std::stod(v);
    else if (k == "--mixed_rate_pct")
      a.mixed_rate_pct = std::stoul(v);
    else if (k == "--mixed_full_rate")
      a.mixed_full_rate = std::stod(v);
    else if (k == "--samples")
      a.samples = std::stoull(v);
    else if (k == "--csv")
      a.csv = v;
    else if (k == "--alpha")
      a.alpha = std::stof(v);
    else if (k == "--seed")
      a.seed = std::stoull(v);
    else if (k == "--arm" || k == "--backlink")
      a.arm = v;
    else if (k == "--efs") {
      a.efs.clear();
      std::stringstream ss(v);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        a.efs.push_back(std::stoul(tok));
      }
    } else {
      throw std::runtime_error("unknown arg " + k);
    }
  }
  if (a.checkpoint_every == 0) {
    throw std::runtime_error("--checkpoint_every must be >= 1");
  }
  if (a.reuse > 1) {
    throw std::runtime_error("--reuse must be 0 or 1");
  }
  if (a.bloom_consolidate > 1) {
    throw std::runtime_error("--bloom_consolidate must be 0 or 1");
  }
  if (a.splice > 1) {
    throw std::runtime_error("--splice must be 0 or 1");
  }
  if (a.qpatch > 1) {
    throw std::runtime_error("--qpatch must be 0 or 1");
  }
  if (!std::isfinite(a.evict_margin) || a.evict_margin < 0) {
    throw std::runtime_error("--evict_margin must be finite and nonnegative");
  }
  if (a.pca_preprocessed > 1) {
    throw std::runtime_error("--pca_preprocessed must be 0 or 1");
  }
  if (a.query_threads == 0 || a.mixed_ef == 0 || !(a.mixed_seconds > 0)) {
    throw std::runtime_error("mixed query_threads/mixed_ef/mixed_seconds must be positive");
  }
  if (a.mixed_rate_pct != 0 && a.mixed_rate_pct != 25 && a.mixed_rate_pct != 50 &&
      a.mixed_rate_pct != 100) {
    throw std::runtime_error("--mixed_rate_pct must be one of 0,25,50,100");
  }
  if ((a.mixed_rate_pct == 25 || a.mixed_rate_pct == 50) && !(a.mixed_full_rate > 0)) {
    throw std::runtime_error("--mixed_full_rate is required for the 25/50 token-bucket legs");
  }
  return a;
}

void flush_update_pool(alaya::laser::QGUpdater &upd, size_t num_threads, bool explicit_cache_cap) {
  upd.flush(num_threads);
  // The default bench allocator tuning deliberately retains large arenas for
  // throughput. Under an explicit RAM-scan cap, return pages released by the
  // pool's high->low watermark eviction to the OS at each batch barrier.
  if (explicit_cache_cap) ::malloc_trim(0);
}

int do_build(const Args &a) {
  FloatMatrix base = read_fbin(a.base, static_cast<int64_t>(a.n));
  const uint32_t dim = base.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;
  std::cout << "[build] n=" << base.n << " dim=" << dim << " R=" << a.R << " L=" << a.L
            << " alpha=" << a.alpha << " threads=" << a.threads << "\n";

  auto t0 = std::chrono::steady_clock::now();
  const std::string vamana_path = a.prefix + "_vamana.index";
  if (a.reuse_graph != 0) {
    // Topology-seal path: pack a pre-existing graph from any producer that
    // wrote the Vamana format.
    if (!std::filesystem::exists(vamana_path)) {
      throw std::runtime_error("--reuse_graph 1 but missing " + vamana_path);
    }
    std::cout << "[build] reusing graph " << vamana_path << "\n";
  } else {
    alaya::vamana::VamanaBuildParams vp;
    // Headroom build: grow the Vamana graph at a smaller degree than
    // degree_bound so every row keeps free (ghost) slots for future backlinks.
    vp.R = a.vamana_R == 0 ? a.R : a.vamana_R;
    vp.L = a.L;
    vp.alpha = a.alpha;
    vp.num_threads = a.threads;
    alaya::vamana::VamanaBuilder vb(base.data.data(), base.n, dim, vp);
    vb.build();
    // Header max_degree is written as degree_bound (a.R): QGBuilder asserts
    // header == degree_bound; per-node degrees may be smaller.
    alaya::vamana::save_graph(vb.graph(), vamana_path, a.R, vb.medoid());
    std::cout << "[build] vamana medoid=" << vb.medoid() << "\n";
  }
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "[build] graph ready in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";

  const std::string pca_base_path = a.prefix + "_pca_base.fbin";
  const std::string pca_param_path = a.prefix + "_pca.bin";
  if (a.pca_preprocessed == 0) {
    if (main_dim < dim) {
      std::cout << "[build] training PCA " << dim << " -> " << main_dim << " on " << base.n
                << " vectors...\n";
      alaya::laser::PCATransform pca(dim);
      pca.train(base.data.data(), base.n);
      auto tp = std::chrono::steady_clock::now();
      std::cout << "[build] PCA trained in " << std::chrono::duration<double>(tp - t1).count()
                << "s, projecting...\n";
      std::vector<float> projected(static_cast<size_t>(base.n) * dim);
      for (size_t i = 0; i < base.n; ++i) {
        pca.transform(base.row(i), projected.data() + i * dim);
      }
      write_fbin(pca_base_path,
                 projected.data(),
                 static_cast<int32_t>(base.n),
                 static_cast<int32_t>(dim));
      pca.save(pca_param_path);
      auto tw = std::chrono::steady_clock::now();
      std::cout << "[build] PCA projected+saved in "
                << std::chrono::duration<double>(tw - tp).count() << "s\n";
    } else {
      write_fbin(pca_base_path,
                 base.data.data(),
                 static_cast<int32_t>(base.n),
                 static_cast<int32_t>(dim));
    }
  } else {
    std::ifstream pca_base(pca_base_path, std::ios::binary);
    int32_t pca_n = 0;
    int32_t pca_dim = 0;
    pca_base.read(reinterpret_cast<char *>(&pca_n), sizeof(pca_n));
    pca_base.read(reinterpret_cast<char *>(&pca_dim), sizeof(pca_dim));
    const uintmax_t expected_size = 8 + static_cast<uintmax_t>(base.n) * dim * sizeof(float);
    if (!pca_base || pca_n != static_cast<int32_t>(base.n) ||
        pca_dim != static_cast<int32_t>(dim) ||
        std::filesystem::file_size(pca_base_path) != expected_size) {
      throw std::runtime_error("invalid preprocessed PCA base: " + pca_base_path);
    }
    if (main_dim < dim && !std::filesystem::exists(pca_param_path)) {
      throw std::runtime_error("missing preprocessed PCA params: " + pca_param_path);
    }
    std::cout << "[build] using preprocessed PCA base " << pca_base_path << "\n";
  }

  alaya::laser::QuantizedGraph qg(base.n, a.R, main_dim, dim, a.seed);
  alaya::laser::QGBuilder builder(qg, a.ef_indexing, a.threads);
  builder.build(vamana_path.c_str(), a.prefix.c_str());
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "[build] laser build done in " << std::chrono::duration<double>(t2 - t1).count()
            << "s (total " << std::chrono::duration<double>(t2 - t0).count() << "s)\n";
  return 0;
}

int do_insert(const Args &a) {
  MappedFloatMatrix base(a.base);
  if (a.from + a.count > base.n) {
    throw std::runtime_error("insert range exceeds base file");
  }
  const uint32_t dim = base.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;

  alaya::laser::QuantizedGraph qg(a.n, a.R, main_dim, dim);
  qg.load_disk_index(a.prefix.c_str(), 0.0F);

  alaya::laser::UpdateParams p;
  p.ef_insert = a.ef_insert;
  p.alpha = a.alpha;
  p.prune_pool_cap = a.prune_cap;
  p.alpha_check_max = a.alpha_check_max;
  p.evict_telemetry = a.evict_telemetry;
  p.evict_margin = a.evict_margin;
  p.splice_enabled = a.splice != 0;
  if (a.arm == "none")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kNone;
  else if (a.arm == "evict")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kEvict;
  else if (a.arm == "exact")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kExactEvict;
  else if (a.arm == "alpha")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kAlphaEvict;
  else if (a.arm == "full")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kFullPrune;
  else
    throw std::runtime_error("bad --arm " + a.arm);

  std::cout << "[insert] arm=" << a.arm << " ef_insert=" << p.ef_insert << " alpha=" << p.alpha
            << " range=[" << a.from << "," << a.from + a.count << ")\n";

  p.max_points = a.n + a.count + 1024;
  p.direct_io = a.direct != 0;
  p.write_cache = a.write_cache != 0;
  p.stage_backlinks = a.stage != 0;
  p.preencode_patch_intents = a.qpatch != 0;
  p.maintenance_evict_stride = a.maintenance_evict_stride;
  p.garden_churn_threshold = a.garden_churn_threshold;
  apply_cache_cap_pages(a.cache_cap_pages, p);
  alaya::laser::QGUpdater upd(qg, p);
  const int ins_threads = static_cast<int>(std::max<size_t>(1, a.insert_threads));
  std::cout << "[insert] batch=" << a.batch << " insert_threads=" << ins_threads
            << " direct_io=" << (upd.direct_io() ? 1 : 0)
            << " cache_cap_pages=" << upd.cache_cap_pages() << " qpatch=" << a.qpatch << "\n";
  const size_t fl_threads = a.flush_threads == 0 ? a.insert_threads : a.flush_threads;
  const bool explicit_cache_cap = a.cache_cap_pages != 0;
  auto t0 = std::chrono::steady_clock::now();
  // Two-batch pipeline: while batch k's searches run, batch k-1's staged
  // backlinks drain on a helper thread. publish(k-1) happens only after that
  // flush joins, so visibility semantics are unchanged; batch k simply
  // searches the snapshot from before batch k-1 (2-batch isolation, the
  // quality-equivalent of doubling --batch). Edges batch k stages while the
  // drain is swapping stripes may drain one cycle early — harmless, searches
  // filter ids >= snapshot until publish.
  std::future<void> pending;
  uint64_t pending_publish = 0;
  std::vector<uint64_t> insert_order(a.count);
  std::iota(insert_order.begin(), insert_order.end(), uint64_t{0});
  if (a.shuffle_seed != 0) {
    std::mt19937_64 rng(a.shuffle_seed);
    std::shuffle(insert_order.begin(), insert_order.end(), rng);
  }
  for (uint64_t start = 0; start < a.count; start += a.batch) {
    const uint64_t end = std::min(a.count, start + a.batch);
    const int64_t batch_size = static_cast<int64_t>(end - start);
#pragma omp parallel for num_threads(ins_threads) schedule(dynamic)
    for (int64_t oi = 0; oi < batch_size; ++oi) {
      const uint64_t i = insert_order[start + static_cast<size_t>(oi)];
      upd.insert_with_id(base.row(a.from + static_cast<uint64_t>(i)),
                         static_cast<alaya::laser::PID>(a.n + static_cast<uint64_t>(i)));
    }
    if (a.shuffle_seed == 0) base.discard_rows(a.from + start, a.from + end);
    if (pending.valid()) {
      pending.get();
      if (a.shuffle_seed == 0) upd.publish(pending_publish);
    }
    if (a.pipeline != 0) {
      pending = std::async(std::launch::async, [&upd, fl_threads, explicit_cache_cap] {
        flush_update_pool(upd, fl_threads, explicit_cache_cap);
      });
      pending_publish = a.n + end;
    } else {
      flush_update_pool(upd, fl_threads, explicit_cache_cap);
      if (a.shuffle_seed == 0) upd.publish(a.n + end);
    }
    if (end / 10000 != start / 10000) {
      auto now = std::chrono::steady_clock::now();
      const double s = std::chrono::duration<double>(now - t0).count();
      std::cout << "[insert] " << end << "/" << a.count << "  " << end / s << " inserts/s\n";
    }
  }
  if (pending.valid()) {
    pending.get();
    if (a.shuffle_seed == 0) upd.publish(pending_publish);
  }
  if (a.shuffle_seed != 0) upd.publish(a.n + a.count);
  upd.finalize();
  auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const auto &s = upd.stats();
  std::cout << "[insert] done: " << a.count << " inserts in " << secs << "s = " << a.count / secs
            << " inserts/s (single writer)\n";
  std::cout << "[insert] phases: drain=" << s.drain_us / 1e6 << "s flush=" << s.flush_us / 1e6
            << "s\n";
  std::cout << "[insert] io: search_reads=" << s.search_page_reads
            << " patch_reads=" << s.patch_page_reads << " writes=" << s.page_writes
            << " logical_writes=" << s.logical_row_writes
            << " flush_unique_pages=" << s.flush_unique_pages
            << "  per-insert: search_r=" << static_cast<double>(s.search_page_reads) / a.count
            << " patch_r=" << static_cast<double>(s.patch_page_reads) / a.count
            << " w=" << static_cast<double>(s.page_writes) / a.count << "\n";
  std::cout << "[insert] backlinks: free_fills=" << s.free_slot_fills
            << " evictions=" << s.evictions << " est_skips=" << s.est_skips
            << " alpha_skips=" << s.alpha_skips << " degenerate=" << s.degenerate_skips
            << " full_recomputes=" << s.full_recomputes << " forced_links=" << s.forced_links
            << " intents_prepared=" << s.patch_intents_prepared
            << " intents_applied=" << s.patch_intents_applied
            << " intent_stale=" << s.patch_intent_stale_fallbacks << "\n";
  std::cout << "round,1,recall,-1";
  print_telemetry(s);
  std::cout << "\n";
  return 0;
}

// Sliding-window churn: tombstone the oldest `count` live ids and insert the
// next `count` unseen vectors, `runs` rounds; masked recall after each round.
// Live set after round r: ids [r*count, n + r*count) minus nothing else.
int do_churn(const Args &a) {
  MappedFloatMatrix base(a.base);
  FloatMatrix query = read_fbin(a.query);
  IntMatrix gt = read_ibin(a.gt);
  const uint32_t dim = base.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;
  if (a.n + a.runs * a.count > base.n) {
    throw std::runtime_error("churn: not enough vectors for rounds");
  }
  if (a.reuse != 0 && a.pipeline != 0) {
    throw std::runtime_error("churn: --reuse 1 requires --pipeline 0 (publish each reuse batch)");
  }

  alaya::laser::QuantizedGraph qg(a.n, a.R, main_dim, dim);
  qg.load_disk_index(a.prefix.c_str(), 0.0F);

  alaya::laser::UpdateParams p;
  p.ef_insert = a.ef_insert;
  p.alpha = a.alpha;
  p.prune_pool_cap = a.prune_cap;
  p.alpha_check_max = a.alpha_check_max;
  p.evict_telemetry = a.evict_telemetry;
  p.evict_margin = a.evict_margin;
  p.splice_enabled = a.splice != 0;
  if (a.arm == "none")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kNone;
  else if (a.arm == "evict")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kEvict;
  else if (a.arm == "exact")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kExactEvict;
  else if (a.arm == "alpha")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kAlphaEvict;
  else if (a.arm == "full")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kFullPrune;
  else
    throw std::runtime_error("bad --arm " + a.arm);

  p.max_points = a.n + a.runs * a.count + 1024;
  p.direct_io = a.direct != 0;
  p.write_cache = a.write_cache != 0;
  p.stage_backlinks = a.stage != 0;
  p.preencode_patch_intents = a.qpatch != 0;
  p.maintain_indegree = a.garden != 0;
  p.maintain_turnover = a.garden != 0 && a.garden_policy == "turnover";
  p.maintenance_evict_stride = a.maintenance_evict_stride;
  p.consolidate_every = a.consolidate_every;
  p.garden_churn_threshold = a.garden_churn_threshold;
  apply_cache_cap_pages(a.cache_cap_pages, p);
  alaya::laser::QGUpdater upd(qg, p);
  if (a.garden != 0) upd.init_indegree(a.insert_threads);
  if (p.maintain_turnover) upd.init_turnover();
  std::cout << "[churn] direct_io=" << (upd.direct_io() ? 1 : 0)
            << " cache_cap_pages=" << upd.cache_cap_pages()
            << " maintenance_evict_stride=" << upd.maintenance_evict_stride()
            << " cache_low_pages=" << upd.cache_cap_pages() / 2 << " splice=" << a.splice
            << " evict_margin=" << a.evict_margin << " qpatch=" << a.qpatch << "\n";
  std::vector<uint32_t> results(static_cast<size_t>(query.n) * a.topk);
  const std::vector<int> query_groups = read_query_groups(a.query_groups, query.n);
  const uint32_t ef_eval = a.efs.empty() ? 100 : a.efs[0];
  const int ins_threads = static_cast<int>(std::max<size_t>(1, a.insert_threads));
  const size_t fl_threads = a.flush_threads == 0 ? a.insert_threads : a.flush_threads;
  const bool explicit_cache_cap = a.cache_cap_pages != 0;
  std::vector<alaya::laser::PID> source_to_pid(base.n, alaya::laser::kPidMax);
  std::vector<uint32_t> source_insert_round(base.n, a.runs + 1);
  for (size_t source = 0; source < a.n; ++source) {
    source_to_pid[source] = static_cast<alaya::laser::PID>(source);
    source_insert_round[source] = 0;
  }
  alaya::laser::UpdateStats last_round_stats;

  auto eval_round = [&](uint64_t round) {
    qg.set_result_filter(&upd.deleted());
    qg.set_params(ef_eval, a.threads, static_cast<int>(a.beam));
    auto t0 = std::chrono::steady_clock::now();
    run_query_lanes(qg,
                    query.data.data(),
                    a.topk,
                    results.data(),
                    query.n,
                    ef_eval,
                    a.beam,
                    a.threads);
    auto t1 = std::chrono::steady_clock::now();
    const double qps = query.n / std::chrono::duration<double>(t1 - t0).count();
    uint64_t hits = 0;
    uint64_t total = 0;
    std::array<uint64_t, 3> group_hits{};
    std::array<uint64_t, 3> group_total{};
    std::array<uint64_t, 4> age_hits{};
    std::array<uint64_t, 4> age_total{};
    for (size_t qi = 0; qi < query.n; ++qi) {
      const uint32_t *truth_row = gt.row(qi);
      std::unordered_set<uint32_t> truth;
      std::array<std::unordered_set<uint32_t>, 4> age_truth;
      for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
        const uint32_t source = truth_row[j];
        if (source < source_to_pid.size()) {
          const alaya::laser::PID pid = source_to_pid[source];
          if (pid != alaya::laser::kPidMax && upd.deleted().find(pid) == upd.deleted().end()) {
            const auto inserted = truth.insert(pid);
            if (inserted.second) {
              const uint64_t age = round - source_insert_round[source];
              size_t bucket = 0;
              if (source_insert_round[source] != 0) {
                bucket = age == 0 ? 1 : age <= 3 ? 2 : 3;
              }
              age_truth[bucket].insert(pid);
              ++age_total[bucket];
            }
          }
        }
      }
      total += truth.size();
      if (!query_groups.empty()) group_total[query_groups[qi]] += truth.size();
      const uint32_t *res_row = results.data() + qi * a.topk;
      for (uint32_t j = 0; j < a.topk; ++j) {
        if (truth.count(res_row[j]) != 0) {
          ++hits;
          for (size_t bucket = 0; bucket < age_truth.size(); ++bucket) {
            if (age_truth[bucket].count(res_row[j]) != 0) {
              ++age_hits[bucket];
              break;
            }
          }
          if (!query_groups.empty()) ++group_hits[query_groups[qi]];
        }
      }
    }
    const auto &s = upd.stats();
    std::vector<int32_t> indegrees;
    if (a.garden != 0) {
      indegrees.reserve(upd.num_points() - upd.deleted().size());
      for (size_t id = 0; id < upd.num_points(); ++id) {
        if (upd.deleted().count(static_cast<alaya::laser::PID>(id)) == 0) {
          indegrees.push_back(upd.indegree(static_cast<alaya::laser::PID>(id)));
        }
      }
      std::sort(indegrees.begin(), indegrees.end());
    }
    auto pct = [&](double pctl) {
      if (indegrees.empty()) return int32_t{0};
      return indegrees[static_cast<size_t>(pctl * static_cast<double>(indegrees.size() - 1))];
    };
    std::cout << "round," << round << ",recall,"
              << (total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total))
              << ",qps," << qps << ",live," << upd.live_count() << ",free_fills,"
              << s.free_slot_fills << ",evictions," << s.evictions << ",est_skips," << s.est_skips
              << ",forced," << s.forced_links << ",indeg_p1," << pct(0.01) << ",indeg_p5,"
              << pct(0.05) << ",indeg_p50," << pct(0.50) << ",file_pages," << upd.file_pages()
              << ",pool_pages," << upd.pool_pages() << ",maintenance_peak_pool_pages,"
              << s.maintenance_peak_pool_pages << ",garden_skipped," << s.garden_skipped
              << ",freed," << s.freed_slots - last_round_stats.freed_slots << ",reused,"
              << s.reused_slots - last_round_stats.reused_slots << ",consolidated,"
              << s.consolidated_rows - last_round_stats.consolidated_rows << ",bloom_candidates,"
              << s.bloom_candidate_rows - last_round_stats.bloom_candidate_rows << ",bloom_scan_ms,"
              << static_cast<double>(s.bloom_scan_us - last_round_stats.bloom_scan_us) / 1000.0
              << ",bloom_row_ms,"
              << static_cast<double>(s.bloom_row_us - last_round_stats.bloom_row_us) / 1000.0
              << ",bloom_finalize_ms,"
              << static_cast<double>(s.bloom_finalize_us - last_round_stats.bloom_finalize_us) /
                     1000.0
              << ",intents_prepared," << s.patch_intents_prepared << ",intents_applied,"
              << s.patch_intents_applied << ",intent_stale," << s.patch_intent_stale_fallbacks
              << ",live_frac,"
              << (upd.allocated_points() == 0
                      ? 0.0
                      : static_cast<double>(upd.live_count()) / upd.allocated_points());
    if (!query_groups.empty()) {
      for (size_t g = 0; g < 3; ++g) {
        const double value =
            group_total[g] == 0 ? -1 : static_cast<double>(group_hits[g]) / group_total[g];
        std::cout << ",recall_g" << g << "," << value;
      }
    }
    print_telemetry(s);
    std::cout << "\n" << std::flush;
    if (p.maintain_turnover) {
      const auto turnover = upd.turnover_summary();
      std::cout << "turnover," << round << ",sum," << turnover.sum << ",p50," << turnover.p50
                << ",p99," << turnover.p99 << ",rows," << turnover.rows << "\n"
                << std::flush;
    }
    const std::array<const char *, 4> age_names = {"base", "fresh", "mid", "old"};
    std::cout << "round_age," << round;
    for (size_t bucket = 0; bucket < age_names.size(); ++bucket) {
      const double value =
          age_total[bucket] == 0 ? -1 : static_cast<double>(age_hits[bucket]) / age_total[bucket];
      std::cout << "," << age_names[bucket] << "," << value << "," << age_total[bucket];
    }
    std::cout << "\n" << std::flush;
    last_round_stats = s;

    for (size_t ef_index = 1; ef_index < a.efs.size(); ++ef_index) {
      const uint32_t ef = a.efs[ef_index];
      qg.set_params(ef, a.threads, static_cast<int>(a.beam));
      auto ef_t0 = std::chrono::steady_clock::now();
      run_query_lanes(qg,
                      query.data.data(),
                      a.topk,
                      results.data(),
                      query.n,
                      ef,
                      a.beam,
                      a.threads);
      auto ef_t1 = std::chrono::steady_clock::now();
      const double ef_qps = query.n / std::chrono::duration<double>(ef_t1 - ef_t0).count();
      uint64_t ef_hits = 0;
      uint64_t ef_total = 0;
      std::array<uint64_t, 3> ef_group_hits{};
      std::array<uint64_t, 3> ef_group_total{};
      for (size_t qi = 0; qi < query.n; ++qi) {
        const uint32_t *truth_row = gt.row(qi);
        std::unordered_set<uint32_t> truth;
        for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
          const uint32_t source = truth_row[j];
          if (source < source_to_pid.size()) {
            const alaya::laser::PID pid = source_to_pid[source];
            if (pid != alaya::laser::kPidMax && upd.deleted().find(pid) == upd.deleted().end()) {
              truth.insert(pid);
            }
          }
        }
        ef_total += truth.size();
        if (!query_groups.empty()) ef_group_total[query_groups[qi]] += truth.size();
        const uint32_t *res_row = results.data() + qi * a.topk;
        for (uint32_t j = 0; j < a.topk; ++j) {
          if (truth.count(res_row[j]) != 0) {
            ++ef_hits;
            if (!query_groups.empty()) ++ef_group_hits[query_groups[qi]];
          }
        }
      }
      std::cout << "round_ef," << round << "," << ef << ",recall,"
                << (ef_total == 0 ? 0.0
                                  : static_cast<double>(ef_hits) / static_cast<double>(ef_total))
                << ",qps," << ef_qps;
      if (!query_groups.empty()) {
        for (size_t g = 0; g < 3; ++g) {
          const double value = ef_group_total[g] == 0
                                   ? -1
                                   : static_cast<double>(ef_group_hits[g]) / ef_group_total[g];
          std::cout << ",recall_g" << g << "," << value;
        }
      }
      std::cout << "\n" << std::flush;
    }
    qg.set_params(ef_eval, a.threads, static_cast<int>(a.beam));
  };

  std::cout << "[churn] base_n=" << a.n << " rounds=" << a.runs << " count=" << a.count
            << " arm=" << a.arm << " ef_eval=" << ef_eval << " insert_threads=" << ins_threads
            << " consolidate=" << a.consolidate << " r_target=" << a.r_target
            << " consolidate_every=" << a.consolidate_every
            << " bloom_consolidate=" << a.bloom_consolidate << " reuse=" << a.reuse
            << " checkpoint_every=" << a.checkpoint_every << "\n";
  eval_round(0);
  for (uint64_t r = 0; r < a.runs; ++r) {
    for (uint64_t i = 0; i < a.count; ++i) {
      const uint64_t source = r * a.count + i;
      const alaya::laser::PID pid = source_to_pid[source];
      if (pid == alaya::laser::kPidMax) {
        throw std::runtime_error("churn: source-to-PID map lost a live source");
      }
      upd.tombstone(pid);
      source_to_pid[source] = alaya::laser::kPidMax;
    }
    auto tc0 = std::chrono::steady_clock::now();
    if (a.consolidate != 0 && a.consolidate_every != 0 && (r + 1) % a.consolidate_every == 0) {
      upd.consolidate(a.insert_threads, a.r_target, a.reuse != 0, a.bloom_consolidate != 0);
      if (explicit_cache_cap) flush_update_pool(upd, fl_threads, true);
    }
    auto tc1 = std::chrono::steady_clock::now();
    const auto garden_before = upd.stats();
    auto tg0 = std::chrono::steady_clock::now();
    if (a.garden != 0) {
      alaya::laser::GardenParams gp;
      gp.frac = a.garden_frac;
      gp.ef_maintenance = a.ef_maintenance;
      gp.pump_budget = a.pump_budget;
      gp.r_target = a.r_target;
      gp.pump_only = a.garden_pump_only != 0;
      if (a.garden_policy == "lowdeg")
        gp.policy = alaya::laser::GardenParams::Policy::kLowIndegree;
      else if (a.garden_policy == "random")
        gp.policy = alaya::laser::GardenParams::Policy::kRandom;
      else if (a.garden_policy == "turnover")
        gp.policy = alaya::laser::GardenParams::Policy::kTurnover;
      else
        throw std::runtime_error("bad --garden_policy " + a.garden_policy);
      upd.garden(a.insert_threads, gp);
      if (explicit_cache_cap) flush_update_pool(upd, fl_threads, true);
    }
    auto tg1 = std::chrono::steady_clock::now();
    const auto garden_after = upd.stats();
    const uint64_t ins_base = a.n + r * a.count;
    auto t0 = std::chrono::steady_clock::now();
    std::future<void> pending;
    uint64_t pending_publish = 0;
    std::vector<uint64_t> insert_order(a.count);
    std::iota(insert_order.begin(), insert_order.end(), uint64_t{0});
    if (a.shuffle_seed != 0) {
      std::mt19937_64 rng(a.shuffle_seed + r);
      std::shuffle(insert_order.begin(), insert_order.end(), rng);
    }
    for (uint64_t start = 0; start < a.count; start += a.batch) {
      const uint64_t end = std::min(a.count, start + a.batch);
      const int64_t batch_size = static_cast<int64_t>(end - start);
#pragma omp parallel for num_threads(ins_threads) schedule(dynamic)
      for (int64_t oi = 0; oi < batch_size; ++oi) {
        const uint64_t i = insert_order[start + static_cast<size_t>(oi)];
        const uint64_t source = ins_base + i;
        alaya::laser::PID pid;
        if (a.reuse != 0) {
          pid = upd.allocate_and_insert(base.row(source));
        } else {
          pid = static_cast<alaya::laser::PID>(source);
          upd.insert_with_id(base.row(source), pid);
        }
        source_to_pid[source] = pid;
        source_insert_round[source] = static_cast<uint32_t>(r + 1);
      }
      if (a.shuffle_seed == 0) base.discard_rows(ins_base + start, ins_base + end);
      if (a.reuse != 0) {
        flush_update_pool(upd, fl_threads, explicit_cache_cap);
        upd.publish(upd.allocated_points());
        continue;
      }
      if (pending.valid()) {
        pending.get();
        if (a.shuffle_seed == 0) upd.publish(pending_publish);
      }
      if (a.pipeline != 0) {
        pending = std::async(std::launch::async, [&upd, fl_threads, explicit_cache_cap] {
          flush_update_pool(upd, fl_threads, explicit_cache_cap);
        });
        pending_publish = ins_base + end;
      } else {
        flush_update_pool(upd, fl_threads, explicit_cache_cap);
        if (a.shuffle_seed == 0) upd.publish(ins_base + end);
      }
    }
    if (pending.valid()) {
      pending.get();
      if (a.shuffle_seed == 0) upd.publish(pending_publish);
    }
    if (a.reuse == 0 && a.shuffle_seed != 0) upd.publish(ins_base + a.count);
    auto t1 = std::chrono::steady_clock::now();
    std::cout << "[churn] round " << r + 1
              << " insert_qps=" << a.count / std::chrono::duration<double>(t1 - t0).count()
              << " consolidate_s=" << std::chrono::duration<double>(tc1 - tc0).count()
              << " garden_s=" << std::chrono::duration<double>(tg1 - tg0).count()
              << " gardened_rows=" << garden_after.gardened_rows - garden_before.gardened_rows
              << "\n";
    if (p.maintain_turnover) {
      const uint64_t selected_sum =
          garden_after.garden_selected_turnover_sum - garden_before.garden_selected_turnover_sum;
      const uint64_t selected_rows =
          garden_after.garden_selected_turnover_rows - garden_before.garden_selected_turnover_rows;
      const uint64_t all_sum =
          garden_after.garden_all_turnover_sum - garden_before.garden_all_turnover_sum;
      const uint64_t all_rows =
          garden_after.garden_all_turnover_rows - garden_before.garden_all_turnover_rows;
      const double selected_avg =
          selected_rows == 0 ? 0.0 : static_cast<double>(selected_sum) / selected_rows;
      const double all_avg = all_rows == 0 ? 0.0 : static_cast<double>(all_sum) / all_rows;
      std::cout << "turnover_garden," << r + 1 << ",selected_avg," << selected_avg << ",all_avg,"
                << all_avg << ",selected_sum," << selected_sum << ",selected_rows," << selected_rows
                << ",all_sum," << all_sum << ",all_rows," << all_rows << "\n";
    }
    if ((r + 1) % a.checkpoint_every == 0) {
      upd.checkpoint();
    } else {
      upd.writeback(a.insert_threads);
    }
    eval_round(r + 1);
  }
  if (a.runs % a.checkpoint_every != 0) upd.checkpoint();
  return 0;
}

class TokenBucket {
 public:
  explicit TokenBucket(double rate_per_second)
      : rate_(rate_per_second),
        capacity_(std::max(1.0, rate_per_second * 0.02)),
        tokens_(capacity_),
        last_(std::chrono::steady_clock::now()) {}

  void consume() {
    if (!(rate_ > 0)) return;
    for (;;) {
      std::unique_lock<std::mutex> lock(mutex_);
      const auto now = std::chrono::steady_clock::now();
      tokens_ =
          std::min(capacity_, tokens_ + std::chrono::duration<double>(now - last_).count() * rate_);
      last_ = now;
      if (tokens_ >= 1.0) {
        tokens_ -= 1.0;
        return;
      }
      const auto wait = std::chrono::duration<double>((1.0 - tokens_) / rate_);
      lock.unlock();
      std::this_thread::sleep_for(wait);
    }
  }

 private:
  double rate_;
  double capacity_;
  double tokens_;
  std::chrono::steady_clock::time_point last_;
  std::mutex mutex_;
};

struct MixedWorkerMetrics {
  uint64_t queries = 0;
  std::vector<uint64_t> latency_ns;
  std::exception_ptr error;
};

struct MixedPhaseMetrics {
  std::string phase;
  bool maintenance = false;
  uint64_t eval_watermark = 0;
  uint64_t query_count = 0;
  uint64_t hits = 0;
  uint64_t total = 0;
  uint64_t inserted = 0;
  uint64_t seqlock_calls = 0;
  uint64_t seqlock_retries = 0;
  uint64_t query_page_reads = 0;
  double elapsed_s = 0;
  double query_qps = 0;
  double p50_us = 0;
  double p99_us = 0;
  double recall = 0;
  double insert_qps = 0;
};

double percentile_us(std::vector<uint64_t> &latencies, double percentile) {
  if (latencies.empty()) return 0;
  std::sort(latencies.begin(), latencies.end());
  const size_t rank =
      static_cast<size_t>(std::ceil(percentile * static_cast<double>(latencies.size())));
  const size_t index = std::min(latencies.size() - 1, std::max<size_t>(1, rank) - 1);
  return static_cast<double>(latencies[index]) / 1000.0;
}

std::pair<uint64_t, uint64_t> evaluate_boundary_recall(
    alaya::laser::QGUpdater &upd,
    const FloatMatrix &query,
    const IntMatrix &gt,
    const Args &a,
    uint64_t eval_watermark,
    const std::unordered_set<alaya::laser::PID> &eval_dead) {
  uint64_t hits = 0;
  uint64_t total = 0;
  const int threads = static_cast<int>(a.query_threads);
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16) reduction(+ : hits, total)
  for (int64_t raw_qi = 0; raw_qi < static_cast<int64_t>(query.n); ++raw_qi) {
    const size_t qi = static_cast<size_t>(raw_qi);
    const std::vector<alaya::laser::PID> result = upd.search(query.row(qi), a.topk, a.mixed_ef);
    std::vector<alaya::laser::PID> truth;
    truth.reserve(a.topk);
    for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
      const auto id = static_cast<alaya::laser::PID>(gt.row(qi)[j]);
      if (id >= eval_watermark || eval_dead.count(id) != 0 ||
          std::find(truth.begin(), truth.end(), id) != truth.end()) {
        continue;
      }
      truth.push_back(id);
    }
    total += truth.size();
    for (alaya::laser::PID id : truth) {
      hits += static_cast<uint64_t>(std::find(result.begin(), result.end(), id) != result.end());
    }
  }
  return {hits, total};
}

MixedPhaseMetrics measure_mixed_phase(const std::string &phase,
                                      bool maintenance,
                                      alaya::laser::QGUpdater &upd,
                                      const FloatMatrix &query,
                                      const IntMatrix &gt,
                                      const Args &a,
                                      const std::unordered_set<alaya::laser::PID> &eval_dead,
                                      const std::function<uint64_t()> &work) {
  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::atomic<uint32_t> ready{0};
  std::vector<MixedWorkerMetrics> worker(a.query_threads);
  std::vector<std::thread> threads;
  threads.reserve(a.query_threads);
  for (uint32_t tid = 0; tid < a.query_threads; ++tid) {
    threads.emplace_back([&, tid] {
      try {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        uint64_t iteration = 0;
        auto &local = worker[tid];
        local.latency_ns.reserve(4096);
        while (!stop.load(std::memory_order_acquire)) {
          const size_t qi = (static_cast<size_t>(tid) + iteration * a.query_threads) % query.n;
          ++iteration;
          const auto t0 = std::chrono::steady_clock::now();
          const std::vector<alaya::laser::PID> result =
              upd.search(query.row(qi), a.topk, a.mixed_ef);
          (void)result;
          const auto t1 = std::chrono::steady_clock::now();
          local.latency_ns.push_back(static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
          ++local.queries;
        }
      } catch (...) {
        worker[tid].error = std::current_exception();
        stop.store(true, std::memory_order_release);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != a.query_threads) std::this_thread::yield();

  const alaya::laser::UpdateStats before = upd.stats();
  const auto phase_start = std::chrono::steady_clock::now();
  start.store(true, std::memory_order_release);
  uint64_t inserted = 0;
  std::exception_ptr work_error;
  try {
    inserted = work();
  } catch (...) {
    work_error = std::current_exception();
  }
  stop.store(true, std::memory_order_release);
  for (auto &thread : threads) thread.join();
  const auto phase_end = std::chrono::steady_clock::now();
  const alaya::laser::UpdateStats after = upd.stats();
  if (work_error) std::rethrow_exception(work_error);
  for (const auto &local : worker) {
    if (local.error) std::rethrow_exception(local.error);
  }

  // Recall is deliberately not accumulated from the latency window: that
  // window spans multiple publish watermarks, so either a start or end GT
  // would score some queries against the wrong live set. Close the workload,
  // freeze the phase-end published watermark/live mask, then run the same
  // pool search as a separate boundary evaluation. Its work is excluded from
  // QPS, latency, and phase seqlock deltas captured above.
  const uint64_t boundary_watermark = upd.num_points();
  const auto [boundary_hits, boundary_total] =
      evaluate_boundary_recall(upd, query, gt, a, boundary_watermark, eval_dead);

  MixedPhaseMetrics out;
  out.phase = phase;
  out.maintenance = maintenance;
  out.eval_watermark = boundary_watermark;
  out.inserted = inserted;
  out.elapsed_s = std::chrono::duration<double>(phase_end - phase_start).count();
  std::vector<uint64_t> latencies;
  for (auto &local : worker) {
    out.query_count += local.queries;
    latencies.insert(latencies.end(), local.latency_ns.begin(), local.latency_ns.end());
  }
  out.hits = boundary_hits;
  out.total = boundary_total;
  out.query_qps = out.elapsed_s == 0 ? 0 : out.query_count / out.elapsed_s;
  out.insert_qps = out.elapsed_s == 0 ? 0 : out.inserted / out.elapsed_s;
  out.recall = out.total == 0 ? 0 : static_cast<double>(out.hits) / out.total;
  out.p50_us = percentile_us(latencies, 0.50);
  out.p99_us = percentile_us(latencies, 0.99);
  out.seqlock_calls = after.query_seqlock_read_calls - before.query_seqlock_read_calls;
  out.seqlock_retries = after.query_seqlock_read_retries - before.query_seqlock_read_retries;
  out.query_page_reads = after.query_page_reads - before.query_page_reads;
  return out;
}

void write_mixed_csv(const Args &a, const std::vector<MixedPhaseMetrics> &phases) {
  static constexpr const char *header =
      "phase,insert_rate_pct,query_threads,insert_threads,eval_watermark,query_count,"
      "query_qps,query_p50_us,query_p99_us,recall,insert_count,insert_qps,"
      "maintenance_query_p50_us,maintenance_query_p99_us,seqlock_read_calls,"
      "seqlock_read_retries,seqlock_retry_rate,query_page_reads,elapsed_s\n";
  std::ofstream file;
  std::ostream *out = &std::cout;
  bool need_header = true;
  if (!a.csv.empty()) {
    std::error_code ec;
    need_header = !std::filesystem::exists(a.csv, ec) || std::filesystem::file_size(a.csv, ec) == 0;
    file.open(a.csv, std::ios::app);
    if (!file) throw std::runtime_error("cannot append mixed csv " + a.csv);
    out = &file;
  }
  if (need_header) *out << header;
  for (const auto &phase : phases) {
    const double retry_rate =
        phase.seqlock_calls + phase.seqlock_retries == 0
            ? 0
            : static_cast<double>(phase.seqlock_retries) /
                  static_cast<double>(phase.seqlock_calls + phase.seqlock_retries);
    *out << phase.phase << ',' << a.mixed_rate_pct << ',' << a.query_threads << ','
         << a.insert_threads << ',' << phase.eval_watermark << ',' << phase.query_count << ','
         << phase.query_qps << ',' << phase.p50_us << ',' << phase.p99_us << ',' << phase.recall
         << ',' << phase.inserted << ',' << phase.insert_qps << ','
         << (phase.maintenance ? phase.p50_us : -1.0) << ','
         << (phase.maintenance ? phase.p99_us : -1.0) << ',' << phase.seqlock_calls << ','
         << phase.seqlock_retries << ',' << retry_rate << ',' << phase.query_page_reads << ','
         << phase.elapsed_s << '\n';
  }
  out->flush();
}

int do_mixed(const Args &a) {
  MappedFloatMatrix base(a.base);
  FloatMatrix query = read_fbin(a.query);
  IntMatrix gt = read_ibin(a.gt);
  if (query.n == 0 || gt.n != query.n || base.dim != query.dim) {
    throw std::runtime_error("mixed: base/query/gt shape mismatch");
  }
  if (a.from != a.n || a.from + a.count > base.n) {
    throw std::runtime_error(
        "mixed: append-only GT mapping requires --from == --n and a valid --count range");
  }
  const uint32_t dim = base.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;
  alaya::laser::QuantizedGraph qg(a.n, a.R, main_dim, dim);
  qg.load_disk_index(a.prefix.c_str(), 0.0F);

  alaya::laser::UpdateParams p;
  p.ef_insert = a.ef_insert;
  p.alpha = a.alpha;
  p.prune_pool_cap = a.prune_cap;
  p.alpha_check_max = a.alpha_check_max;
  p.evict_telemetry = a.evict_telemetry;
  p.evict_margin = a.evict_margin;
  p.splice_enabled = a.splice != 0;
  if (a.arm == "none")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kNone;
  else if (a.arm == "evict")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kEvict;
  else if (a.arm == "exact")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kExactEvict;
  else if (a.arm == "alpha")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kAlphaEvict;
  else if (a.arm == "full")
    p.backlink_mode = alaya::laser::UpdateParams::Backlink::kFullPrune;
  else
    throw std::runtime_error("bad --arm " + a.arm);
  p.max_points = a.n + a.count + 1024;
  p.direct_io = a.direct != 0;
  p.write_cache = a.write_cache != 0;
  p.stage_backlinks = a.stage != 0;
  p.preencode_patch_intents = a.qpatch != 0;
  p.maintain_indegree = true;
  p.maintain_turnover = a.garden_policy == "turnover";
  p.maintenance_evict_stride = a.maintenance_evict_stride;
  p.garden_churn_threshold = a.garden_churn_threshold;
  apply_cache_cap_pages(a.cache_cap_pages, p);
  alaya::laser::QGUpdater upd(qg, p);
  upd.init_indegree(a.insert_threads);
  if (p.maintain_turnover) upd.init_turnover();

  const double target_rate = a.mixed_rate_pct == 25 || a.mixed_rate_pct == 50
                                 ? a.mixed_full_rate * static_cast<double>(a.mixed_rate_pct) / 100.0
                                 : 0.0;
  uint64_t source_cursor = a.from;
  const uint64_t source_end = a.from + a.count;
  const int insert_threads = static_cast<int>(std::max<size_t>(1, a.insert_threads));
  const size_t flush_threads = a.flush_threads == 0 ? a.insert_threads : a.flush_threads;
  const bool explicit_cache_cap = a.cache_cap_pages != 0;

  auto insert_window = [&]() -> uint64_t {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(a.mixed_seconds));
    if (a.mixed_rate_pct == 0) {
      std::this_thread::sleep_until(deadline);
      return 0;
    }
    TokenBucket bucket(target_rate);
    uint64_t inserted = 0;
    while (std::chrono::steady_clock::now() < deadline && source_cursor < source_end) {
      const size_t batch_cap = a.mixed_rate_pct == 100 ? a.batch : std::min<size_t>(a.batch, 512);
      const uint64_t batch_n = std::min<uint64_t>(batch_cap, source_end - source_cursor);
      const uint64_t pid_base = upd.allocated_points();
      if (pid_base != source_cursor) {
        throw std::runtime_error("mixed: source id/PID mapping diverged");
      }
#pragma omp parallel for num_threads(insert_threads) schedule(dynamic)
      for (int64_t i = 0; i < static_cast<int64_t>(batch_n); ++i) {
        if (a.mixed_rate_pct != 100) bucket.consume();
        const uint64_t source = source_cursor + static_cast<uint64_t>(i);
        upd.insert_with_id(base.row(source),
                           static_cast<alaya::laser::PID>(pid_base + static_cast<uint64_t>(i)));
      }
      flush_update_pool(upd, flush_threads, explicit_cache_cap);
      upd.publish(pid_base + batch_n);
      base.discard_rows(source_cursor, source_cursor + batch_n);
      source_cursor += batch_n;
      inserted += batch_n;
    }
    if (std::chrono::steady_clock::now() < deadline) std::this_thread::sleep_until(deadline);
    return inserted;
  };

  std::unordered_set<alaya::laser::PID> dead(upd.deleted().begin(), upd.deleted().end());
  std::vector<MixedPhaseMetrics> phases;
  phases.push_back(
      measure_mixed_phase("pure_insert", false, upd, query, gt, a, dead, insert_window));

  std::vector<alaya::laser::PID> scheduled_delete;
  std::unordered_set<alaya::laser::PID> delete_boundary = dead;
  for (uint64_t i = 0; i < a.tombstone_n; ++i) {
    const uint64_t raw = a.tombstone_from + i;
    if (raw >= upd.num_points()) break;
    const auto id = static_cast<alaya::laser::PID>(raw);
    if (delete_boundary.insert(id).second) scheduled_delete.push_back(id);
  }
  phases.push_back(measure_mixed_phase("insert_delete",
                                       false,
                                       upd,
                                       query,
                                       gt,
                                       a,
                                       delete_boundary,
                                       [&]() -> uint64_t {
                                         for (alaya::laser::PID id : scheduled_delete)
                                           upd.tombstone(id);
                                         dead = delete_boundary;
                                         return insert_window();
                                       }));

  phases.push_back(
      measure_mixed_phase("consolidate", true, upd, query, gt, a, dead, [&]() -> uint64_t {
        upd.consolidate(a.insert_threads, a.r_target, a.reuse != 0, a.bloom_consolidate != 0);
        if (explicit_cache_cap) flush_update_pool(upd, flush_threads, true);
        return 0;
      }));

  alaya::laser::GardenParams gp;
  gp.frac = a.garden_frac;
  gp.ef_maintenance = a.ef_maintenance;
  gp.pump_budget = a.pump_budget;
  gp.r_target = a.r_target;
  if (a.garden_policy == "lowdeg")
    gp.policy = alaya::laser::GardenParams::Policy::kLowIndegree;
  else if (a.garden_policy == "random")
    gp.policy = alaya::laser::GardenParams::Policy::kRandom;
  else if (a.garden_policy == "turnover")
    gp.policy = alaya::laser::GardenParams::Policy::kTurnover;
  else
    throw std::runtime_error("bad --garden_policy " + a.garden_policy);
  phases.push_back(measure_mixed_phase("garden", true, upd, query, gt, a, dead, [&]() -> uint64_t {
    upd.garden(a.insert_threads, gp);
    if (explicit_cache_cap) flush_update_pool(upd, flush_threads, true);
    return 0;
  }));

  upd.checkpoint();
  write_mixed_csv(a, phases);
  std::cout << "[mixed] rate_pct=" << a.mixed_rate_pct << " query_threads=" << a.query_threads
            << " inserted=" << source_cursor - a.from << " final_watermark=" << upd.num_points()
            << " direct_io=" << (upd.direct_io() ? 1 : 0)
            << " cache_cap_pages=" << upd.cache_cap_pages() << " pool_pages=" << upd.pool_pages()
            << "\n";
  return 0;
}

int do_eval(const Args &a) {
  FloatMatrix query = read_fbin(a.query);
  IntMatrix gt = read_ibin(a.gt);
  if (gt.n != query.n) {
    throw std::runtime_error("gt/query count mismatch");
  }
  const uint32_t dim = query.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;
  const uint64_t live_max = a.live_max == 0 ? a.n : a.live_max;
  const std::vector<int> query_groups = read_query_groups(a.query_groups, query.n);

  std::unordered_set<alaya::laser::PID> dead;
  for (uint64_t i = 0; i < a.tombstone_n; ++i) {
    dead.insert(static_cast<alaya::laser::PID>(a.tombstone_from + i));
  }

  alaya::laser::QuantizedGraph qg(a.n, a.R, main_dim, dim);
  qg.load_disk_index(a.prefix.c_str(), a.dram_gb);
  if (!dead.empty()) {
    qg.set_result_filter(&dead);
  }

  auto is_live = [&](uint32_t id) {
    return id < live_max && (dead.empty() || dead.find(id) == dead.end());
  };

  std::cout << "[eval] n=" << a.n << " live_max=" << live_max << " tombstones=" << dead.size()
            << " queries=" << query.n << " topk=" << a.topk << " threads=" << a.threads
            << " beam=" << a.beam << "\n";
  std::cout << "ef,recall,qps,mean_us";
  if (!query_groups.empty()) std::cout << ",recall_g0,recall_g1,recall_g2";
  std::cout << "\n";

  std::vector<uint32_t> results(static_cast<size_t>(query.n) * a.topk);
  for (uint32_t ef : a.efs) {
    qg.set_params(ef, a.threads, static_cast<int>(a.beam));
    double best_qps = 0;
    // Async I/O completion order makes each run's traversal (and its recall at
    // low ef) nondeterministic — docs §14. Score every timed run and report
    // the mean in the CSV row; the trailing "# recall_runs" line exposes the
    // per-run spread. Denominators are identical across runs (fixed mask), so
    // summing hits/totals over runs yields the mean exactly.
    uint64_t hits = 0;
    uint64_t total = 0;
    uint64_t new_hits = 0;
    uint64_t new_total = 0;
    std::array<uint64_t, 3> group_hits{};
    std::array<uint64_t, 3> group_total{};
    std::vector<double> run_recalls;
    const auto score_run = [&]() {
      uint64_t run_hits = 0;
      uint64_t run_total = 0;
      for (size_t qi = 0; qi < query.n; ++qi) {
        const uint32_t *truth_row = gt.row(qi);
        std::unordered_set<uint32_t> truth;
        for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
          if (is_live(truth_row[j])) {
            truth.insert(truth_row[j]);
          }
        }
        run_total += truth.size();
        if (!query_groups.empty()) group_total[query_groups[qi]] += truth.size();
        const uint32_t *res_row = results.data() + qi * a.topk;
        std::unordered_set<uint32_t> res_set(res_row, res_row + a.topk);
        for (uint32_t t : truth) {
          const bool hit = res_set.count(t) != 0;
          run_hits += static_cast<uint64_t>(hit);
          if (!query_groups.empty()) group_hits[query_groups[qi]] += static_cast<uint64_t>(hit);
          if (a.new_id_split != 0 && t >= a.new_id_split) {
            ++new_total;
            new_hits += static_cast<uint64_t>(hit);
          }
        }
      }
      hits += run_hits;
      total += run_total;
      run_recalls.push_back(
          run_total == 0 ? 0.0 : static_cast<double>(run_hits) / static_cast<double>(run_total));
    };
    for (uint32_t r = 0; r < a.runs + 1; ++r) {  // first run = warmup
      auto t0 = std::chrono::steady_clock::now();
      run_query_lanes(qg,
                      query.data.data(),
                      a.topk,
                      results.data(),
                      query.n,
                      ef,
                      a.beam,
                      a.threads,
                      a.arena);
      auto t1 = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(t1 - t0).count();
      if (r > 0) {
        best_qps = std::max(best_qps, query.n / secs);
        score_run();
      }
    }
    if (run_recalls.empty()) score_run();  // --runs 0: score the only pass
    const double recall = total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    const double mean_us = 1e6 * static_cast<double>(a.threads) / best_qps;
    std::cout << ef << "," << recall << "," << best_qps << "," << mean_us;
    if (a.new_id_split != 0) {
      const double new_recall =
          new_total == 0 ? 0.0 : static_cast<double>(new_hits) / static_cast<double>(new_total);
      // new_total stays in per-run units for cross-config sanity checks.
      std::cout << ",new_recall=" << new_recall << ",new_total=" << new_total / run_recalls.size();
    }
    if (!query_groups.empty()) {
      for (size_t g = 0; g < 3; ++g) {
        const double value =
            group_total[g] == 0 ? -1 : static_cast<double>(group_hits[g]) / group_total[g];
        std::cout << "," << value;
      }
    }
    std::cout << "\n";
    if (run_recalls.size() > 1) {
      std::cout << "# recall_runs," << ef;
      for (const double v : run_recalls) std::cout << "," << v;
      std::cout << "\n";
    }
    std::cout.flush();
    ALAYA_KSP_REPORT("laser");
  }
  qg.set_result_filter(nullptr);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Args a = parse(argc, argv);
    // The updater's staged-backlink drain makes millions of short-lived
    // (often aligned) allocations per batch; glibc's default heap trim/grow
    // cycling serializes all threads on mprotect/mmap_sem (measured: drain
    // 10.9s -> 5.1s at 64T with these settings). See docs §9.2. An explicit
    // pool cap selects the RAM-scan regime instead: small arena padding and
    // prompt trimming make evicted pool pages leave anonymous RSS.
    mallopt(M_TRIM_THRESHOLD, a.cache_cap_pages == 0 ? 1 << 30 : 1 << 20);
    mallopt(M_TOP_PAD, a.cache_cap_pages == 0 ? 1 << 28 : 1 << 20);
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    if (a.mode == "build") {
      return do_build(a);
    }
    if (a.mode == "insert") {
      return do_insert(a);
    }
    if (a.mode == "eval") {
      return do_eval(a);
    }
    if (a.mode == "churn") {
      return do_churn(a);
    }
    if (a.mode == "mixed") {
      return do_mixed(a);
    }
    if (a.mode == "fastscan_oracle" || a.mode == "twohop_oracle") {
      alaya::laser::bench::OracleConfig config;
      config.prefix = a.prefix;
      config.base = a.base;
      config.degree = a.R;
      config.main_dim = a.main_dim;
      config.samples = a.samples;
      config.ef_maintenance = a.ef_maintenance;
      config.prune_pool_cap = a.prune_cap;
      config.r_target = a.r_target;
      config.alpha = a.alpha;
      config.seed = a.seed;
      return a.mode == "fastscan_oracle" ? alaya::laser::bench::run_fastscan_oracle(config)
                                         : alaya::laser::bench::run_twohop_oracle(config);
    }
    throw std::runtime_error("unknown mode " + a.mode);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
