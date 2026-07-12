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
#include <future>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace {

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
  const double agree = s.evict_tel_samples == 0
                           ? -1
                           : static_cast<double>(s.evict_tel_agree) / s.evict_tel_samples;
  const double relerr = s.evict_tel_samples == 0
                            ? -1
                            : s.evict_tel_relerr_sum / s.evict_tel_samples;
  std::cout << ",tel_agree," << agree << ",tel_regret_p50," << telemetry_p50(s)
            << ",tel_relerr," << relerr;
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
  uint32_t vamana_R = 0;  // build: Vamana degree < degree_bound leaves free slots ("headroom")
  uint32_t main_dim = 0;  // 0 -> dim
  size_t ef_insert = 100, prune_cap = 300, alpha_check_max = 16;
  size_t batch = 4096, insert_threads = 32;
  uint32_t consolidate = 0, r_target = 0;
  uint32_t reuse = 0;             // churn: 1 = allocate from consolidated free-list first
  uint32_t checkpoint_every = 1;  // churn: superblock cadence in rounds
  uint32_t garden = 0;
  double garden_frac = 0.05;
  size_t ef_maintenance = 200, pump_budget = 4;
  std::string garden_policy = "lowdeg";
  float alpha = 1.2F;
  uint64_t seed = 42;
  std::string arm = "alpha";  // none|evict|alpha|full
  double evict_telemetry = 0;
  uint64_t shuffle_seed = 0;
  uint32_t direct = 0;  // 1 = O_DIRECT write fd (P0.1 attribution arm), 0 = buffered (default)
  uint32_t write_cache = 1;  // 0 = immediate per-patch writes (P0.1-era control)
  uint32_t pipeline = 0;     // staged mode only: overlap drain(i) with batch i+1 (2-batch isolation)
  uint32_t stage = 0;        // 1 = stage backlinks + barrier drain; 0 = inline patches into pool
  size_t flush_threads = 0;  // threads for the overlapped flush; 0 = insert_threads
  std::vector<uint32_t> efs = {60, 80, 100, 150, 200, 300};
};

Args parse(int argc, char **argv) {
  Args a;
  if (argc < 2) {
    throw std::runtime_error("usage: bench_laser_update_sift <build|insert|eval> [--k v ...]");
  }
  a.mode = argv[1];
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
    else if (k == "--ef_insert")
      a.ef_insert = std::stoull(v);
    else if (k == "--prune_cap")
      a.prune_cap = std::stoull(v);
    else if (k == "--alpha_check_max")
      a.alpha_check_max = std::stoull(v);
    else if (k == "--evict_telemetry")
      a.evict_telemetry = std::stod(v);
    else if (k == "--shuffle_seed")
      a.shuffle_seed = std::stoull(v);
    else if (k == "--batch")
      a.batch = std::stoull(v);
    else if (k == "--insert_threads")
      a.insert_threads = std::stoull(v);
    else if (k == "--consolidate")
      a.consolidate = std::stoul(v);
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
    else if (k == "--garden_policy")
      a.garden_policy = v;
    else if (k == "--direct")
      a.direct = std::stoul(v);
    else if (k == "--write_cache")
      a.write_cache = std::stoul(v);
    else if (k == "--pipeline")
      a.pipeline = std::stoul(v);
    else if (k == "--stage")
      a.stage = std::stoul(v);
    else if (k == "--flush_threads")
      a.flush_threads = std::stoull(v);
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
  return a;
}

int do_build(const Args &a) {
  FloatMatrix base = read_fbin(a.base, static_cast<int64_t>(a.n));
  const uint32_t dim = base.dim;
  const uint32_t main_dim = a.main_dim == 0 ? dim : a.main_dim;
  std::cout << "[build] n=" << base.n << " dim=" << dim << " R=" << a.R << " L=" << a.L
            << " alpha=" << a.alpha << " threads=" << a.threads << "\n";

  auto t0 = std::chrono::steady_clock::now();
  alaya::vamana::VamanaBuildParams vp;
  // Headroom build: grow the Vamana graph at a smaller degree than
  // degree_bound so every row keeps free (ghost) slots for future backlinks.
  vp.R = a.vamana_R == 0 ? a.R : a.vamana_R;
  vp.L = a.L;
  vp.alpha = a.alpha;
  vp.num_threads = a.threads;
  alaya::vamana::VamanaBuilder vb(base.data.data(), base.n, dim, vp);
  vb.build();
  const std::string vamana_path = a.prefix + "_vamana.index";
  // Header max_degree is written as degree_bound (a.R): QGBuilder asserts
  // header == degree_bound; per-node degrees may be smaller.
  alaya::vamana::save_graph(vb.graph(), vamana_path, a.R, vb.medoid());
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "[build] vamana done in " << std::chrono::duration<double>(t1 - t0).count()
            << "s, medoid=" << vb.medoid() << "\n";

  write_fbin(a.prefix + "_pca_base.fbin",
             base.data.data(),
             static_cast<int32_t>(base.n),
             static_cast<int32_t>(dim));

  alaya::laser::QuantizedGraph qg(base.n, a.R, main_dim, dim, a.seed);
  alaya::laser::QGBuilder builder(qg, a.ef_indexing, a.threads);
  builder.build(vamana_path.c_str(), a.prefix.c_str());
  auto t2 = std::chrono::steady_clock::now();
  std::cout << "[build] laser build done in " << std::chrono::duration<double>(t2 - t1).count()
            << "s (total " << std::chrono::duration<double>(t2 - t0).count() << "s)\n";
  return 0;
}

int do_insert(const Args &a) {
  FloatMatrix base = read_fbin(a.base);
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
  alaya::laser::QGUpdater upd(qg, p);
  const int ins_threads = static_cast<int>(std::max<size_t>(1, a.insert_threads));
  std::cout << "[insert] batch=" << a.batch << " insert_threads=" << ins_threads
            << " direct_io=" << (upd.direct_io() ? 1 : 0) << "\n";
  const size_t fl_threads = a.flush_threads == 0 ? a.insert_threads : a.flush_threads;
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
    if (pending.valid()) {
      pending.get();
      if (a.shuffle_seed == 0) upd.publish(pending_publish);
    }
    if (a.pipeline != 0) {
      pending = std::async(std::launch::async, [&upd, fl_threads] { upd.flush(fl_threads); });
      pending_publish = a.n + end;
    } else {
      upd.flush(fl_threads);
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
            << "\n";
  std::cout << "round,1,recall,-1";
  print_telemetry(s);
  std::cout << "\n";
  return 0;
}

// Sliding-window churn: tombstone the oldest `count` live ids and insert the
// next `count` unseen vectors, `runs` rounds; masked recall after each round.
// Live set after round r: ids [r*count, n + r*count) minus nothing else.
int do_churn(const Args &a) {
  FloatMatrix base = read_fbin(a.base);
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
  p.maintain_indegree = a.garden != 0;
  alaya::laser::QGUpdater upd(qg, p);
  if (a.garden != 0) upd.init_indegree(a.insert_threads);
  std::cout << "[churn] direct_io=" << (upd.direct_io() ? 1 : 0) << "\n";
  std::vector<uint32_t> results(static_cast<size_t>(query.n) * a.topk);
  const std::vector<int> query_groups = read_query_groups(a.query_groups, query.n);
  const uint32_t ef_eval = a.efs.empty() ? 100 : a.efs[0];
  const int ins_threads = static_cast<int>(std::max<size_t>(1, a.insert_threads));
  std::vector<alaya::laser::PID> source_to_pid(base.n, alaya::laser::kPidMax);
  for (size_t source = 0; source < a.n; ++source) {
    source_to_pid[source] = static_cast<alaya::laser::PID>(source);
  }
  alaya::laser::UpdateStats last_round_stats;

  auto eval_round = [&](uint64_t round) {
    qg.set_result_filter(&upd.deleted());
    qg.set_params(ef_eval, a.threads, static_cast<int>(a.beam));
    auto t0 = std::chrono::steady_clock::now();
    qg.batch_search(query.data.data(), a.topk, results.data(), query.n);
    auto t1 = std::chrono::steady_clock::now();
    const double qps = query.n / std::chrono::duration<double>(t1 - t0).count();
    uint64_t hits = 0;
    uint64_t total = 0;
    std::array<uint64_t, 3> group_hits{};
    std::array<uint64_t, 3> group_total{};
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
      total += truth.size();
      if (!query_groups.empty()) group_total[query_groups[qi]] += truth.size();
      const uint32_t *res_row = results.data() + qi * a.topk;
      for (uint32_t j = 0; j < a.topk; ++j) {
        if (truth.count(res_row[j]) != 0) {
          ++hits;
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
              << ",qps," << qps << ",live," << upd.live_count()
              << ",free_fills," << s.free_slot_fills << ",evictions," << s.evictions
              << ",est_skips," << s.est_skips << ",forced," << s.forced_links
              << ",indeg_p1," << pct(0.01) << ",indeg_p5," << pct(0.05)
              << ",indeg_p50," << pct(0.50) << ",file_pages," << upd.file_pages()
              << ",freed," << s.freed_slots - last_round_stats.freed_slots
              << ",reused," << s.reused_slots - last_round_stats.reused_slots
              << ",live_frac,"
              << (upd.allocated_points() == 0
                      ? 0.0
                      : static_cast<double>(upd.live_count()) / upd.allocated_points());
    if (!query_groups.empty()) {
      for (size_t g = 0; g < 3; ++g) {
        const double value = group_total[g] == 0
                                 ? -1
                                 : static_cast<double>(group_hits[g]) / group_total[g];
        std::cout << ",recall_g" << g << "," << value;
      }
    }
    print_telemetry(s);
    std::cout << "\n" << std::flush;
    last_round_stats = s;
  };

  std::cout << "[churn] base_n=" << a.n << " rounds=" << a.runs << " count=" << a.count
            << " arm=" << a.arm << " ef_eval=" << ef_eval << " insert_threads=" << ins_threads
            << " consolidate=" << a.consolidate << " r_target=" << a.r_target
            << " reuse=" << a.reuse << " checkpoint_every=" << a.checkpoint_every << "\n";
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
    if (a.consolidate != 0 || a.reuse != 0) {
      upd.consolidate(a.insert_threads, a.r_target, a.reuse != 0);
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
      if (a.garden_policy == "lowdeg")
        gp.policy = alaya::laser::GardenParams::Policy::kLowIndegree;
      else if (a.garden_policy == "random")
        gp.policy = alaya::laser::GardenParams::Policy::kRandom;
      else
        throw std::runtime_error("bad --garden_policy " + a.garden_policy);
      upd.garden(a.insert_threads, gp);
    }
    auto tg1 = std::chrono::steady_clock::now();
    const auto garden_after = upd.stats();
    const uint64_t ins_base = a.n + r * a.count;
    auto t0 = std::chrono::steady_clock::now();
    const size_t fl_threads = a.flush_threads == 0 ? a.insert_threads : a.flush_threads;
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
      }
      if (a.reuse != 0) {
        upd.flush(fl_threads);
        upd.publish(upd.allocated_points());
        continue;
      }
      if (pending.valid()) {
        pending.get();
        if (a.shuffle_seed == 0) upd.publish(pending_publish);
      }
      if (a.pipeline != 0) {
        pending = std::async(std::launch::async, [&upd, fl_threads] { upd.flush(fl_threads); });
        pending_publish = ins_base + end;
      } else {
        upd.flush(fl_threads);
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
              << " gardened_rows="
              << garden_after.gardened_rows - garden_before.gardened_rows << "\n";
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
  qg.load_disk_index(a.prefix.c_str(), 0.0F);
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
    for (uint32_t r = 0; r < a.runs + 1; ++r) {  // first run = warmup
      auto t0 = std::chrono::steady_clock::now();
      qg.batch_search(query.data.data(), a.topk, results.data(), query.n);
      auto t1 = std::chrono::steady_clock::now();
      const double secs = std::chrono::duration<double>(t1 - t0).count();
      if (r > 0) {
        best_qps = std::max(best_qps, query.n / secs);
      }
    }
    // masked recall on the final run's results
    uint64_t hits = 0;
    uint64_t total = 0;
    uint64_t new_hits = 0;
    uint64_t new_total = 0;
    std::array<uint64_t, 3> group_hits{};
    std::array<uint64_t, 3> group_total{};
    for (size_t qi = 0; qi < query.n; ++qi) {
      const uint32_t *truth_row = gt.row(qi);
      std::unordered_set<uint32_t> truth;
      for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
        if (is_live(truth_row[j])) {
          truth.insert(truth_row[j]);
        }
      }
      total += truth.size();
      if (!query_groups.empty()) group_total[query_groups[qi]] += truth.size();
      const uint32_t *res_row = results.data() + qi * a.topk;
      std::unordered_set<uint32_t> res_set(res_row, res_row + a.topk);
      for (uint32_t t : truth) {
        const bool hit = res_set.count(t) != 0;
        hits += static_cast<uint64_t>(hit);
        if (!query_groups.empty()) group_hits[query_groups[qi]] += static_cast<uint64_t>(hit);
        if (a.new_id_split != 0 && t >= a.new_id_split) {
          ++new_total;
          new_hits += static_cast<uint64_t>(hit);
        }
      }
    }
    const double recall = total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
    const double mean_us = 1e6 * static_cast<double>(a.threads) / best_qps;
    std::cout << ef << "," << recall << "," << best_qps << "," << mean_us;
    if (a.new_id_split != 0) {
      const double new_recall =
          new_total == 0 ? 0.0 : static_cast<double>(new_hits) / static_cast<double>(new_total);
      std::cout << ",new_recall=" << new_recall << ",new_total=" << new_total;
    }
    if (!query_groups.empty()) {
      for (size_t g = 0; g < 3; ++g) {
        const double value = group_total[g] == 0
                                 ? -1
                                 : static_cast<double>(group_hits[g]) / group_total[g];
        std::cout << "," << value;
      }
    }
    std::cout << "\n";
  }
  qg.set_result_filter(nullptr);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  // The updater's staged-backlink drain makes millions of short-lived
  // (often aligned) allocations per batch; glibc's default heap trim/grow
  // cycling serializes all threads on mprotect/mmap_sem (measured: drain
  // 10.9s -> 5.1s at 64T with these settings). See docs §9.2.
  mallopt(M_TRIM_THRESHOLD, 1 << 30);
  mallopt(M_TOP_PAD, 1 << 28);
  mallopt(M_MMAP_THRESHOLD, 1 << 30);

  try {
    const Args a = parse(argc, argv);
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
    throw std::runtime_error("unknown mode " + a.mode);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
