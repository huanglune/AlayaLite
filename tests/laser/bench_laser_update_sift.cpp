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

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
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
  uint64_t n = 0;
  uint64_t from = 0, count = 0;
  uint64_t live_max = 0;
  uint64_t new_id_split = 0;  // eval: also report recall restricted to GT ids >= split
  uint64_t tombstone_from = 0, tombstone_n = 0;
  uint32_t R = 64, L = 200, ef_indexing = 200, threads = 64, beam = 16, topk = 10, runs = 3;
  uint32_t main_dim = 0;  // 0 -> dim
  size_t ef_insert = 100, prune_cap = 300, alpha_check_max = 16;
  float alpha = 1.2F;
  uint64_t seed = 42;
  std::string arm = "alpha";  // none|evict|alpha|full
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
    if (k == "--base") a.base = v;
    else if (k == "--query") a.query = v;
    else if (k == "--gt") a.gt = v;
    else if (k == "--prefix") a.prefix = v;
    else if (k == "--n") a.n = std::stoull(v);
    else if (k == "--from") a.from = std::stoull(v);
    else if (k == "--count") a.count = std::stoull(v);
    else if (k == "--live_max") a.live_max = std::stoull(v);
    else if (k == "--new_id_split") a.new_id_split = std::stoull(v);
    else if (k == "--tombstone_from") a.tombstone_from = std::stoull(v);
    else if (k == "--tombstone_n") a.tombstone_n = std::stoull(v);
    else if (k == "--R") a.R = std::stoul(v);
    else if (k == "--L") a.L = std::stoul(v);
    else if (k == "--ef_indexing") a.ef_indexing = std::stoul(v);
    else if (k == "--threads") a.threads = std::stoul(v);
    else if (k == "--beam") a.beam = std::stoul(v);
    else if (k == "--topk") a.topk = std::stoul(v);
    else if (k == "--runs") a.runs = std::stoul(v);
    else if (k == "--main_dim") a.main_dim = std::stoul(v);
    else if (k == "--ef_insert") a.ef_insert = std::stoull(v);
    else if (k == "--prune_cap") a.prune_cap = std::stoull(v);
    else if (k == "--alpha_check_max") a.alpha_check_max = std::stoull(v);
    else if (k == "--alpha") a.alpha = std::stof(v);
    else if (k == "--seed") a.seed = std::stoull(v);
    else if (k == "--arm") a.arm = v;
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
  vp.R = a.R;
  vp.L = a.L;
  vp.alpha = a.alpha;
  vp.num_threads = a.threads;
  alaya::vamana::VamanaBuilder vb(base.data.data(), base.n, dim, vp);
  vb.build();
  const std::string vamana_path = a.prefix + "_vamana.index";
  alaya::vamana::save_graph(vb.graph(), vamana_path, a.R, vb.medoid());
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "[build] vamana done in " << std::chrono::duration<double>(t1 - t0).count()
            << "s, medoid=" << vb.medoid() << "\n";

  write_fbin(a.prefix + "_pca_base.fbin", base.data.data(), static_cast<int32_t>(base.n),
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
  if (a.arm == "none") p.backlink_mode = alaya::laser::UpdateParams::Backlink::kNone;
  else if (a.arm == "evict") p.backlink_mode = alaya::laser::UpdateParams::Backlink::kEvict;
  else if (a.arm == "alpha") p.backlink_mode = alaya::laser::UpdateParams::Backlink::kAlphaEvict;
  else if (a.arm == "full") p.backlink_mode = alaya::laser::UpdateParams::Backlink::kFullPrune;
  else throw std::runtime_error("bad --arm " + a.arm);

  std::cout << "[insert] arm=" << a.arm << " ef_insert=" << p.ef_insert << " alpha=" << p.alpha
            << " range=[" << a.from << "," << a.from + a.count << ")\n";

  alaya::laser::QGUpdater upd(qg, p);
  auto t0 = std::chrono::steady_clock::now();
  for (uint64_t i = 0; i < a.count; ++i) {
    upd.insert(base.row(a.from + i));
    if ((i + 1) % 10000 == 0) {
      auto now = std::chrono::steady_clock::now();
      const double s = std::chrono::duration<double>(now - t0).count();
      std::cout << "[insert] " << (i + 1) << "/" << a.count << "  " << (i + 1) / s
                << " inserts/s\n";
    }
  }
  upd.finalize();
  auto t1 = std::chrono::steady_clock::now();
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const auto &s = upd.stats();
  std::cout << "[insert] done: " << a.count << " inserts in " << secs << "s = " << a.count / secs
            << " inserts/s (single writer)\n";
  std::cout << "[insert] io: search_reads=" << s.search_page_reads
            << " patch_reads=" << s.patch_page_reads << " writes=" << s.page_writes
            << "  per-insert: search_r=" << static_cast<double>(s.search_page_reads) / a.count
            << " patch_r=" << static_cast<double>(s.patch_page_reads) / a.count
            << " w=" << static_cast<double>(s.page_writes) / a.count << "\n";
  std::cout << "[insert] backlinks: free_fills=" << s.free_slot_fills
            << " evictions=" << s.evictions << " est_skips=" << s.est_skips
            << " alpha_skips=" << s.alpha_skips << " degenerate=" << s.degenerate_skips
            << " full_recomputes=" << s.full_recomputes << "\n";
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
  std::cout << "ef,recall,qps,mean_us\n";

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
    for (size_t qi = 0; qi < query.n; ++qi) {
      const uint32_t *truth_row = gt.row(qi);
      std::unordered_set<uint32_t> truth;
      for (uint32_t j = 0; j < gt.dim && truth.size() < a.topk; ++j) {
        if (is_live(truth_row[j])) {
          truth.insert(truth_row[j]);
        }
      }
      total += truth.size();
      const uint32_t *res_row = results.data() + qi * a.topk;
      std::unordered_set<uint32_t> res_set(res_row, res_row + a.topk);
      for (uint32_t t : truth) {
        const bool hit = res_set.count(t) != 0;
        hits += static_cast<uint64_t>(hit);
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
    std::cout << "\n";
  }
  qg.set_result_filter(nullptr);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
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
    throw std::runtime_error("unknown mode " + a.mode);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
