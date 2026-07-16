// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Memory QG (RaBitQ) native-ISA reference arm for the full-cache adjudication
// probe. Mirrors rabitq_benchmark.cpp but takes fbin/ibin inputs so it can run
// on the exact same base/query/GT as the LASER arms, and reports both serial
// and multi-threaded QPS. Build with -DALAYA_NATIVE_ARCH=ON -DBUILD_PYTHON=OFF
// to measure the engine ceiling (AVX-512 fastscan) instead of the shipped
// portable-wheel baseline.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "executor/jobs/graph_search_job.hpp"
#include "index/graph/qg/qg_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "space/rabitq_space.hpp"

namespace {

struct Matrix {
  uint32_t n = 0;
  uint32_t dim = 0;
  std::vector<char> raw;
  template <typename T>
  auto row(size_t i) const -> const T * {
    return reinterpret_cast<const T *>(raw.data()) + i * dim;
  }
};

auto read_bin(const std::string &path) -> Matrix {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  int32_t header[2];
  in.read(reinterpret_cast<char *>(header), sizeof(header));
  Matrix m;
  m.n = static_cast<uint32_t>(header[0]);
  m.dim = static_cast<uint32_t>(header[1]);
  m.raw.resize(static_cast<size_t>(m.n) * m.dim * 4);
  in.read(m.raw.data(), static_cast<std::streamsize>(m.raw.size()));
  if (!in) {
    throw std::runtime_error("short read " + path);
  }
  return m;
}

}  // namespace

auto main(int argc, char **argv) -> int {
  if (argc < 4) {
    std::cerr << "usage: bench_memqg_native <base.fbin> <query.fbin> <gt.ibin> "
                 "[ef_build=100] [rounds=3] [dump_graph_path]\n";
    return 1;
  }
  const Matrix base = read_bin(argv[1]);
  const Matrix query = read_bin(argv[2]);
  const Matrix gt = read_bin(argv[3]);
  const size_t ef_build = argc > 4 ? std::stoul(argv[4]) : 100;
  const size_t rounds = argc > 5 ? std::stoul(argv[5]) : 3;
  const std::string dump_graph = argc > 6 ? argv[6] : "";
  if (gt.n != query.n) {
    throw std::runtime_error("gt/query count mismatch");
  }
  std::cout << "base n=" << base.n << " dim=" << base.dim << " queries=" << query.n
            << " gt_dim=" << gt.dim << " ef_build=" << ef_build << "\n";

  using Space = alaya::RaBitQSpace<>;
  auto space = std::make_shared<Space>(base.n, base.dim, alaya::MetricType::L2);
  auto t0 = std::chrono::steady_clock::now();
  space->fit(reinterpret_cast<const float *>(base.raw.data()), base.n);
  auto builder = alaya::QGBuilder<Space>(space);
  builder.set_ef_build(ef_build);
  builder.build_graph();
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "build_seconds," << std::chrono::duration<double>(t1 - t0).count() << "\n";

  if (!dump_graph.empty()) {
    // Export the QG topology in vamana .index format so the LASER packer can
    // seal it into its row format (fullcache probe: graph-quality isolation).
    std::vector<std::vector<uint32_t>> adj(base.n);
    size_t edges = 0;
    for (size_t i = 0; i < base.n; ++i) {
      std::unordered_set<uint32_t> seen;
      adj[i].reserve(32);
      for (const auto &nb : builder.final_neighbors(i)) {
        const auto id = static_cast<uint32_t>(nb.id_);
        if (id != i && id < base.n && seen.insert(id).second) {
          adj[i].push_back(id);
        }
      }
      edges += adj[i].size();
    }
    alaya::vamana::save_graph(adj, dump_graph, 32,
                              static_cast<uint32_t>(builder.entry_point()));
    std::cout << "dumped_graph," << dump_graph << ",edges," << edges << ",ep,"
              << builder.entry_point() << "\n";
  }

  // Profile mode (kernel-gap probe): PROFILE_EF=100 pins the sweep to a
  // single {ef, topk=10, threads=1} config so perf can attach to a long
  // steady-state search window (crank rounds via argv[5]).
  std::vector<size_t> efs = {40, 60, 100, 200};
  std::vector<size_t> topks = {10, 100};
  std::vector<int> thread_arms = {1, 16};
  if (const char *pe = std::getenv("PROFILE_EF")) {
    efs = {std::stoul(pe)};
    topks = {10};
    thread_arms = {1};
  }

  ALAYA_KSP_RESET();  // drop build-phase (find_candidates) accumulation
  std::cout << "arm,topk,ef,threads,recall,qps,mean_us\n";
  for (const size_t topk : topks) {
    for (const int threads : thread_arms) {
      std::vector<std::unique_ptr<alaya::GraphSearchJob<Space>>> jobs;
      jobs.reserve(threads);
      for (int t = 0; t < threads; ++t) {
        jobs.emplace_back(std::make_unique<alaya::GraphSearchJob<Space>>(space, nullptr));
      }
      for (const size_t ef : efs) {
        const size_t eff_ef = std::max(ef, topk);
        std::vector<uint32_t> results(static_cast<size_t>(query.n) * topk);
        // warmup
        jobs[0]->rabitq_search_solo(query.row<float>(0), topk, results.data(), eff_ef);
        double best_qps = 0.0;
        for (size_t r = 0; r < rounds; ++r) {
          auto s0 = std::chrono::steady_clock::now();
#pragma omp parallel for num_threads(threads) schedule(dynamic, 64)
          for (uint32_t q = 0; q < query.n; ++q) {
            const int tid = omp_get_thread_num();
            jobs[tid]->rabitq_search_solo(query.row<float>(q), topk,
                                          results.data() + static_cast<size_t>(q) * topk, eff_ef);
          }
          auto s1 = std::chrono::steady_clock::now();
          const double secs = std::chrono::duration<double>(s1 - s0).count();
          best_qps = std::max(best_qps, query.n / secs);
        }
        uint64_t hits = 0;
        for (uint32_t q = 0; q < query.n; ++q) {
          std::unordered_set<uint32_t> truth(gt.row<uint32_t>(q), gt.row<uint32_t>(q) + topk);
          for (size_t k = 0; k < topk; ++k) {
            hits += truth.count(results[static_cast<size_t>(q) * topk + k]);
          }
        }
        const double recall = static_cast<double>(hits) / (static_cast<double>(query.n) * topk);
        std::printf("memqg_native,%zu,%zu,%d,%.4f,%.1f,%.1f\n", topk, ef, threads, recall,
                    best_qps, 1e6 * threads / best_qps);
        std::fflush(stdout);
        ALAYA_KSP_REPORT("memqg");
      }
    }
  }
  return 0;
}
