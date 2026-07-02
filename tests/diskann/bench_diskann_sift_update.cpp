// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only
//
// PQ update benchmark for DiskANNIndex on SIFT1M/GIST1M-style fbin datasets.
// The update trace format matches Yi's update runner: each round file contains
// uint32 update_size, followed by update_size delete ids and update_size insert ids.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "index/graph/diskann/diskann_index.hpp"

namespace {
using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;

constexpr uint32_t kMissingSlot = std::numeric_limits<uint32_t>::max();
constexpr uint32_t kDefaultBenchmarkGraphR = 64;
constexpr uint32_t kDefaultBenchmarkBuildL = 100;
constexpr uint32_t kDefaultBenchmarkUpdateSearchL = 30;

struct FloatMatrix {
  std::vector<float> data;
  uint32_t n = 0;
  uint32_t dim = 0;
};

struct IntMatrix {
  std::vector<uint32_t> data;
  uint32_t n = 0;
  uint32_t dim = 0;
  const uint32_t *row(uint32_t i) const { return data.data() + static_cast<size_t>(i) * dim; }
};

struct TraceManifest {
  std::string prefix = "round_";
  uint32_t initial_count = 0;
  uint32_t total_count = 0;
  uint32_t rounds = 0;
  uint32_t update_size = 0;
};

struct TraceRound {
  std::vector<uint32_t> deletes;
  std::vector<uint32_t> inserts;
};

struct DatasetFiles {
  std::string name;
  std::filesystem::path base;
  std::filesystem::path queries;
  std::filesystem::path groundtruth;
};

struct Options {
  std::string data_dir = "./sift1m";
  std::string trace_dir = "/tmp/diskann_sift1m_update_trace_20260701";
  std::string index_dir = "/tmp/diskann_sift1m_alaya_update_pq";
  std::string out_csv = "/tmp/diskann_sift1m_alaya_update_pq.csv";
  bool rebuild = false;
  bool deterministic = false;
  bool flush_rounds = false;
  bool build_only = false;
  bool eval_only = false;
  bool single_updates = false;
  uint32_t max_rounds = 0;
  uint32_t nq = 1000;
  uint32_t build_r = kDefaultBenchmarkGraphR;
  uint32_t build_l = kDefaultBenchmarkBuildL;
  uint32_t search_l = 100;
  uint32_t rerank_count = 0;
  uint32_t update_l = kDefaultBenchmarkUpdateSearchL;
  uint32_t beam = 4;
  uint32_t search_threads = 1;
  uint32_t insert_batch = 32;
  uint32_t update_insert_threads = 32;
  uint32_t update_reconnect_threads = 4;
  uint32_t warmup_searches = 0;
  uint32_t updates_per_round = 0;
  uint64_t page_cache_capacity = 0;
};

DatasetFiles resolve_dataset_files(const std::filesystem::path &data_dir) {
  const std::vector<DatasetFiles> candidates = {
      {"sift",
       data_dir / "sift_base.fbin",
       data_dir / "sift_query.fbin",
       data_dir / "sift_gt.ibin"},
      {"gist",
       data_dir / "gist_base.fbin",
       data_dir / "gist_query.fbin",
       data_dir / "gist_gt.ibin"},
  };
  for (const DatasetFiles &files : candidates) {
    if (!std::filesystem::exists(files.base)) {
      continue;
    }
    if (!std::filesystem::exists(files.queries) || !std::filesystem::exists(files.groundtruth)) {
      throw std::runtime_error("incomplete " + files.name + " dataset under " +
                               data_dir.string());
    }
    return files;
  }
  throw std::runtime_error("cannot find sift_base.fbin or gist_base.fbin under " +
                           data_dir.string());
}

FloatMatrix read_fbin(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  int32_t n = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!in || n <= 0 || dim <= 0) {
    throw std::runtime_error("bad fbin header: " + path.string());
  }
  FloatMatrix matrix;
  matrix.n = static_cast<uint32_t>(n);
  matrix.dim = static_cast<uint32_t>(dim);
  matrix.data.resize(static_cast<size_t>(matrix.n) * matrix.dim);
  in.read(reinterpret_cast<char *>(matrix.data.data()),
          static_cast<std::streamsize>(matrix.data.size() * sizeof(float)));
  if (!in) {
    throw std::runtime_error("short fbin read: " + path.string());
  }
  return matrix;
}

IntMatrix read_ibin(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  int32_t n = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!in || n <= 0 || dim <= 0) {
    throw std::runtime_error("bad ibin header: " + path.string());
  }
  IntMatrix matrix;
  matrix.n = static_cast<uint32_t>(n);
  matrix.dim = static_cast<uint32_t>(dim);
  matrix.data.resize(static_cast<size_t>(matrix.n) * matrix.dim);
  in.read(reinterpret_cast<char *>(matrix.data.data()),
          static_cast<std::streamsize>(matrix.data.size() * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error("short ibin read: " + path.string());
  }
  return matrix;
}

uint32_t parse_u32(const std::string &value, const char *name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " overflows uint32");
  }
  return static_cast<uint32_t>(parsed);
}

uint64_t parse_u64(const std::string &value, const char *name) {
  return static_cast<uint64_t>(std::stoull(value));
}

TraceManifest read_manifest(const std::filesystem::path &path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open trace manifest: " + path.string());
  }
  std::unordered_map<std::string, std::string> kv;
  for (std::string line; std::getline(in, line);) {
    const size_t eq = line.find('=');
    if (eq != std::string::npos) {
      kv[line.substr(0, eq)] = line.substr(eq + 1);
    }
  }
  TraceManifest m;
  m.prefix = kv.at("file_prefix");
  m.initial_count = parse_u32(kv.at("initial_count"), "initial_count");
  m.total_count = parse_u32(kv.at("total_count"), "total_count");
  m.rounds = parse_u32(kv.at("rounds"), "rounds");
  m.update_size = parse_u32(kv.at("update_size"), "update_size");
  return m;
}

TraceRound read_round(const std::filesystem::path &path, uint32_t expected_size) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open trace round: " + path.string());
  }
  uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  if (!in || n != expected_size) {
    throw std::runtime_error("bad trace round header: " + path.string());
  }
  TraceRound round;
  round.deletes.resize(n);
  round.inserts.resize(n);
  in.read(reinterpret_cast<char *>(round.deletes.data()),
          static_cast<std::streamsize>(round.deletes.size() * sizeof(uint32_t)));
  in.read(reinterpret_cast<char *>(round.inserts.data()),
          static_cast<std::streamsize>(round.inserts.size() * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error("short trace round read: " + path.string());
  }
  return round;
}

Options parse_args(int argc, char **argv) {
  Options opt;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rebuild") {
      opt.rebuild = true;
    } else if (arg == "--deterministic") {
      opt.deterministic = true;
    } else if (arg == "--flush_rounds") {
      opt.flush_rounds = true;
    } else if (arg == "--build_only") {
      opt.build_only = true;
    } else if (arg == "--eval_only") {
      opt.eval_only = true;
    } else if (arg == "--single_updates") {
      opt.single_updates = true;
    } else if (arg == "--rounds" && i + 1 < argc) {
      opt.max_rounds = parse_u32(argv[++i], "--rounds");
    } else if (arg == "--nq" && i + 1 < argc) {
      opt.nq = parse_u32(argv[++i], "--nq");
    } else if (arg == "--R" && i + 1 < argc) {
      opt.build_r = std::max<uint32_t>(1, parse_u32(argv[++i], "--R"));
    } else if (arg == "--build_L" && i + 1 < argc) {
      opt.build_l = std::max<uint32_t>(1, parse_u32(argv[++i], "--build_L"));
    } else if (arg == "--L" && i + 1 < argc) {
      opt.search_l = parse_u32(argv[++i], "--L");
    } else if (arg == "--rerank_count" && i + 1 < argc) {
      opt.rerank_count = parse_u32(argv[++i], "--rerank_count");
    } else if (arg == "--update_L" && i + 1 < argc) {
      opt.update_l = std::max<uint32_t>(1, parse_u32(argv[++i], "--update_L"));
    } else if (arg == "--beam" && i + 1 < argc) {
      opt.beam = parse_u32(argv[++i], "--beam");
    } else if (arg == "--threads" && i + 1 < argc) {
      opt.search_threads = std::max<uint32_t>(1, parse_u32(argv[++i], "--threads"));
    } else if (arg == "--insert_batch" && i + 1 < argc) {
      opt.insert_batch = std::max<uint32_t>(1, parse_u32(argv[++i], "--insert_batch"));
    } else if (arg == "--update_insert_threads" && i + 1 < argc) {
      opt.update_insert_threads =
          std::max<uint32_t>(1, parse_u32(argv[++i], "--update_insert_threads"));
    } else if (arg == "--update_reconnect_threads" && i + 1 < argc) {
      opt.update_reconnect_threads =
          std::max<uint32_t>(1, parse_u32(argv[++i], "--update_reconnect_threads"));
    } else if (arg == "--warmup_searches" && i + 1 < argc) {
      opt.warmup_searches = parse_u32(argv[++i], "--warmup_searches");
    } else if (arg == "--updates_per_round" && i + 1 < argc) {
      opt.updates_per_round = parse_u32(argv[++i], "--updates_per_round");
    } else if (arg == "--page_cache" && i + 1 < argc) {
      opt.page_cache_capacity = parse_u64(argv[++i], "--page_cache");
    } else {
      if (arg.rfind("--", 0) == 0) {
        throw std::invalid_argument("unknown option: " + arg);
      }
      pos.push_back(arg);
    }
  }
  if (pos.size() > 0) {
    opt.data_dir = pos[0];
  }
  if (pos.size() > 1) {
    opt.trace_dir = pos[1];
  }
  if (pos.size() > 2) {
    opt.index_dir = pos[2];
  }
  if (pos.size() > 3) {
    opt.out_csv = pos[3];
  }
  return opt;
}

TraceRound limit_round_updates(TraceRound round, const Options &opt) {
  if (opt.updates_per_round == 0) {
    return round;
  }
  const size_t count = static_cast<size_t>(opt.updates_per_round);
  if (count > round.deletes.size() || count > round.inserts.size()) {
    throw std::invalid_argument("--updates_per_round exceeds trace round size");
  }
  round.deletes.resize(count);
  round.inserts.resize(count);
  return round;
}

void build_initial_index(const Options &opt,
                         const FloatMatrix &base,
                         const TraceManifest &manifest) {
  namespace fs = std::filesystem;
  if (opt.rebuild && fs::exists(opt.index_dir)) {
    fs::remove_all(opt.index_dir);
  }
  if (fs::exists(opt.index_dir)) {
    std::cout << "[update_bench] reusing index " << opt.index_dir << "\n";
    return;
  }
  std::vector<uint64_t> labels(manifest.initial_count);
  for (uint32_t id = 0; id < manifest.initial_count; ++id) {
    labels[id] = id;
  }
  DiskANNBuildParams bp;
  bp.R = opt.build_r;
  bp.L = opt.build_l;
  bp.alpha = 1.2f;
  bp.pq_n_chunks = 32;
  bp.cache_ratio = 0.01;
  bp.num_threads = 96;
  bp.seed = 1234;
  bp.verbose = true;
  std::cout << "[update_bench] building initial PQ index, n=" << manifest.initial_count
            << " dir=" << opt.index_dir << "\n";
  DiskANNIndex::build(opt.index_dir,
                      base.data.data(),
                      labels.data(),
                      manifest.initial_count,
                      base.dim,
                      bp);
}

struct RecallResult {
  uint64_t hits = 0;
  uint64_t total = 0;
  double mean_us = 0.0;
  double qps = 0.0;
  double recall() const { return total == 0 ? 0.0 : static_cast<double>(hits) / total; }
};

RecallResult evaluate_search(const DiskANNIndex &idx,
                             const FloatMatrix &queries,
                             const IntMatrix &gt,
                             const std::vector<uint8_t> &live,
                             const Options &opt) {
  const uint32_t nq = opt.nq == 0 ? queries.n : std::min<uint32_t>(opt.nq, queries.n);
  constexpr uint32_t kTopK = 10;
  DiskANNSearchParams sp;
  sp.search_list_size = opt.search_l;
  sp.use_pq = true;
  sp.rerank = true;
  sp.rerank_count = opt.rerank_count == 0 ? opt.search_l : opt.rerank_count;
  sp.deterministic = opt.deterministic;

  std::vector<uint64_t> labels(static_cast<size_t>(nq) * kTopK);
  std::vector<float> distances(static_cast<size_t>(nq) * kTopK);
  std::vector<double> lat_us(nq, 0.0);
  std::atomic<uint32_t> next{0};
  auto worker = [&]() {
    for (;;) {
      const uint32_t qi = next.fetch_add(1, std::memory_order_relaxed);
      if (qi >= nq) {
        break;
      }
      const float *query = queries.data.data() + static_cast<size_t>(qi) * queries.dim;
      auto t0 = std::chrono::steady_clock::now();
      idx.search(query,
                 kTopK,
                 labels.data() + static_cast<size_t>(qi) * kTopK,
                 distances.data() + static_cast<size_t>(qi) * kTopK,
                 sp);
      auto t1 = std::chrono::steady_clock::now();
      lat_us[qi] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
  };
  const auto wall0 = std::chrono::steady_clock::now();
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < opt.search_threads; ++i) {
    threads.emplace_back(worker);
  }
  for (auto &thread : threads) {
    thread.join();
  }
  const auto wall1 = std::chrono::steady_clock::now();

  RecallResult result;
  for (uint32_t qi = 0; qi < nq; ++qi) {
    std::unordered_set<uint64_t> got;
    const uint64_t *row = labels.data() + static_cast<size_t>(qi) * kTopK;
    for (uint32_t i = 0; i < kTopK; ++i) {
      if (row[i] != DiskANNIndex::kNoLabel) {
        got.insert(row[i]);
      }
    }
    const uint32_t *truth = gt.row(qi);
    for (uint32_t i = 0; i < kTopK && i < gt.dim; ++i) {
      if (truth[i] < live.size() && live[truth[i]] != 0) {
        ++result.total;
        if (got.count(truth[i]) != 0) {
          ++result.hits;
        }
      }
    }
    result.mean_us += lat_us[qi];
  }
  result.mean_us /= nq;
  result.qps = static_cast<double>(nq) / std::chrono::duration<double>(wall1 - wall0).count();
  return result;
}

void apply_deletes(DiskANNIndex &idx,
                   const TraceRound &round,
                   const Options &opt,
                   std::vector<uint8_t> &live,
                   std::vector<uint32_t> &label_to_slot) {
  if (opt.single_updates) {
    for (const uint32_t label : round.deletes) {
      if (label >= live.size() || live[label] == 0 || label_to_slot[label] == kMissingSlot) {
        throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
      }
      idx.remove(label_to_slot[label]);
      live[label] = 0;
      label_to_slot[label] = kMissingSlot;
    }
    return;
  }

  std::vector<uint32_t> slots;
  slots.reserve(round.deletes.size());
  for (const uint32_t label : round.deletes) {
    if (label >= live.size() || live[label] == 0 || label_to_slot[label] == kMissingSlot) {
      throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
    }
    slots.push_back(label_to_slot[label]);
  }
  idx.batch_remove(slots.data(), static_cast<uint32_t>(slots.size()));
  for (const uint32_t label : round.deletes) {
    live[label] = 0;
    label_to_slot[label] = kMissingSlot;
  }
}

void apply_inserts(DiskANNIndex &idx,
                   const FloatMatrix &base,
                   const TraceRound &round,
                   const Options &opt,
                   std::vector<uint8_t> &live,
                   std::vector<uint32_t> &label_to_slot) {
  if (opt.single_updates) {
    for (const uint32_t label : round.inserts) {
      if (label >= live.size() || live[label] != 0) {
        throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
      }
      const float *src = base.data.data() + static_cast<size_t>(label) * base.dim;
      label_to_slot[label] = idx.insert(src, label);
      live[label] = 1;
    }
    return;
  }

  std::vector<float> batch_vectors(static_cast<size_t>(round.inserts.size()) * base.dim);
  std::vector<uint64_t> batch_labels(round.inserts.size());
  for (size_t i = 0; i < round.inserts.size(); ++i) {
    const uint32_t label = round.inserts[i];
    if (label >= live.size() || live[label] != 0) {
      throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
    }
    const float *src = base.data.data() + static_cast<size_t>(label) * base.dim;
    std::copy(src, src + base.dim, batch_vectors.data() + i * base.dim);
    batch_labels[i] = label;
  }
  const std::vector<uint32_t> slots = idx.batch_insert(batch_vectors.data(),
                                                       batch_labels.data(),
                                                       static_cast<uint32_t>(batch_labels.size()),
                                                       opt.insert_batch);
  for (size_t i = 0; i < batch_labels.size(); ++i) {
    const uint32_t label = static_cast<uint32_t>(batch_labels[i]);
    live[label] = 1;
    label_to_slot[label] = slots[i];
  }
}

void apply_live_mask_only(const TraceRound &round, std::vector<uint8_t> &live) {
  for (const uint32_t label : round.deletes) {
    if (label >= live.size() || live[label] == 0) {
      throw std::runtime_error("trace delete id is not live: " + std::to_string(label));
    }
    live[label] = 0;
  }
  for (const uint32_t label : round.inserts) {
    if (label >= live.size() || live[label] != 0) {
      throw std::runtime_error("trace insert id is already live: " + std::to_string(label));
    }
    live[label] = 1;
  }
}

void run_warmup_searches(const DiskANNIndex &idx,
                         const FloatMatrix &queries,
                         const IntMatrix &gt,
                         const std::vector<uint8_t> &live,
                         const Options &opt) {
  for (uint32_t i = 0; i < opt.warmup_searches; ++i) {
    const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
    std::cout << "[warmup " << i << "] masked_recall@10=" << recall.recall()
              << " search_mean_us=" << recall.mean_us << " search_qps=" << recall.qps << "\n";
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options opt = parse_args(argc, argv);
    const std::filesystem::path data_dir = opt.data_dir;
    const std::filesystem::path trace_dir = opt.trace_dir;
    const TraceManifest manifest = read_manifest(trace_dir / "manifest.txt");
    if (opt.updates_per_round > manifest.update_size) {
      throw std::invalid_argument("--updates_per_round exceeds trace update_size");
    }

    const DatasetFiles dataset = resolve_dataset_files(data_dir);
    FloatMatrix base = read_fbin(dataset.base);
    FloatMatrix queries = read_fbin(dataset.queries);
    IntMatrix gt = read_ibin(dataset.groundtruth);
    if (base.n != manifest.total_count || queries.dim != base.dim || gt.n != queries.n) {
      throw std::runtime_error(dataset.name + " data and trace manifest are inconsistent");
    }

    build_initial_index(opt, base, manifest);
    if (opt.build_only) {
      std::cout << "[update_bench] build_only complete: " << opt.index_dir << "\n";
      return 0;
    }
    DiskANNLoadParams lp;
    lp.num_threads = std::max<uint32_t>(opt.search_threads, 1);
    lp.beam_width = opt.beam;
    lp.updatable = true;
    lp.update_search_l = opt.update_l;
    lp.update_insert_threads = opt.update_insert_threads;
    lp.update_reconnect_threads = opt.update_reconnect_threads;
    lp.page_cache_capacity = opt.page_cache_capacity;
    DiskANNIndex idx;
    idx.load(opt.index_dir, lp);

    std::vector<uint8_t> live(manifest.total_count, 0);
    std::vector<uint32_t> label_to_slot(manifest.total_count, kMissingSlot);
    for (uint32_t id = 0; id < manifest.initial_count; ++id) {
      live[id] = 1;
      label_to_slot[id] = id;
    }

    const uint32_t rounds =
        opt.max_rounds == 0 ? manifest.rounds : std::min<uint32_t>(opt.max_rounds, manifest.rounds);
    if (opt.eval_only) {
      for (uint32_t round_id = 0; round_id < rounds; ++round_id) {
        const auto path = trace_dir / (manifest.prefix + std::to_string(round_id));
        apply_live_mask_only(limit_round_updates(read_round(path, manifest.update_size), opt), live);
      }
      run_warmup_searches(idx, queries, gt, live, opt);
      const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
      std::cout << "[eval_only] rounds=" << rounds << " masked_recall@10=" << recall.recall()
                << " search_mean_us=" << recall.mean_us << " search_qps=" << recall.qps << "\n";
      return 0;
    }

    run_warmup_searches(idx, queries, gt, live, opt);

    std::ofstream csv(opt.out_csv);
    csv << "round,deletes,inserts,update_ms,delete_ms,insert_ms,update_qps,"
           "masked_recall_at_10,recall_hits,recall_total,search_mean_us,search_qps,"
           "live_count,tombstones,free_slots\n";

    for (uint32_t round_id = 0; round_id < rounds; ++round_id) {
      const auto path = trace_dir / (manifest.prefix + std::to_string(round_id));
      const TraceRound round = limit_round_updates(read_round(path, manifest.update_size), opt);
      const auto t0 = std::chrono::steady_clock::now();
      apply_deletes(idx, round, opt, live, label_to_slot);
      const auto t_delete = std::chrono::steady_clock::now();
      apply_inserts(idx, base, round, opt, live, label_to_slot);
      const auto t_insert = std::chrono::steady_clock::now();
      if (opt.flush_rounds) {
        idx.flush();
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double update_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      const double delete_ms = std::chrono::duration<double, std::milli>(t_delete - t0).count();
      const double insert_ms =
          std::chrono::duration<double, std::milli>(t_insert - t_delete).count();
      const double update_qps =
          static_cast<double>(round.deletes.size() + round.inserts.size()) / (update_ms / 1000.0);
      const RecallResult recall = evaluate_search(idx, queries, gt, live, opt);
      std::cout << "[round " << round_id << "] update_qps=" << update_qps
                << " delete_ms=" << delete_ms << " insert_ms=" << insert_ms
                << " masked_recall@10=" << recall.recall() << " live=" << idx.live_count()
                << " tombstones=" << idx.tombstone_count() << "\n";
      csv << round_id << "," << round.deletes.size() << "," << round.inserts.size() << ","
          << update_ms << "," << delete_ms << "," << insert_ms << "," << update_qps << ","
          << recall.recall() << "," << recall.hits << "," << recall.total << ","
          << recall.mean_us << "," << recall.qps << "," << idx.live_count() << ","
          << idx.tombstone_count() << "," << idx.free_slot_count() << "\n";
    }
    idx.flush();
    std::cout << "[update_bench] wrote " << opt.out_csv << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[update_bench] ERROR: " << e.what() << "\n";
    return 1;
  }
}
