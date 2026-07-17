// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Regression test for a data-dependent heap overflow in QGBuilder's
// out-of-core build path.
//
// Root cause (see qg_builder.hpp, init_from_vamana() and build()): per-node
// out-neighbour counts read from a Vamana graph file were never bounds-checked
// against QuantizedGraph::degree_bound_. VamanaBuilder's in-flight degrees can
// temporarily exceed R by up to 1.3x (GRAPH_SLACK_FACTOR, see
// index/graph/vamana/vamana_builder.hpp) before the final cleanup/prune pass;
// when that pass does not bring every single node back down to <= R (data
// dependent -- happens for some random seeds, not others), vamana_writer.hpp's
// save_graph() writes the node's *actual* (still-oversized) neighbour count as
// its per-node `k`, while unconditionally writing the *requested* R into the
// header's max_observed_degree field regardless of the true data (see
// save_graph()'s comment: "The writer truncates the header's
// max_observed_degree to R on output") -- so the file header always looks
// consistent even when individual node records are not.
//
// QGBuilder::init_from_vamana() used to trust both the header field and the
// per-node `k` unconditionally (guarded only by assert(), which every LASER
// test target compiles out via -DNDEBUG -- see tests/laser/CMakeLists.txt's
// _laser_test_opts -- and which Release builds also strip). A node with
// k > degree_bound_ then made QuantizedGraph::update_qg_out_of_memory() write
// `k` neighbour-vector pages into thread_data.neighbor_vector_scratch_, a
// buffer sized for exactly degree_bound_ pages -- an out-of-bounds write whose
// extent is controlled by file contents.
//
// tests/laser/qg/test_admission_contract.cpp's TinyIndex comment documents the
// original discovery: building several small indices back-to-back in one
// process crashed roughly 2 times out of 5 (SIGSEGV inside a memmove reached
// from the read-completion path, confirmed via gdb, single- and
// multi-threaded alike -- so data-dependent, not a race). All builder-facing
// test fixtures as of 2026-07 sidestep it by building exactly once per
// process. This test does the opposite on purpose, to prove the fix (a
// defensive per-node clamp to degree_bound_ in init_from_vamana(), plus
// hard-failing on a genuinely incompatible vector-file dimension instead of
// relying on assert()) actually holds under repeated same-process builds.
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kDeg = 64;
constexpr size_t kN = 2000;
// The manifest asks for >=10 stable rounds post-fix; a couple of extra rounds
// give some margin above that floor without materially slowing the suite.
constexpr int kRounds = 12;

std::vector<float> make_data(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (auto &v : data) {
    v = dist(gen);
  }
  return data;
}

// QGBuilder::build() requires "{filename}_pca_base.fbin" to already exist
// (its own doc comment says so) -- the raw float32 vectors it streams through
// PHASE 1. Writes the standard two-int32-header fbin layout used by every
// other LASER test fixture in this tree (e.g.
// tests/laser/qg/qg_wal_test_support.hpp's write_fbin()).
void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * static_cast<size_t>(n) *
                                         static_cast<size_t>(dim)));
}

// Builds one tiny QG index end-to-end (Vamana graph -> QGBuilder::build()) in
// its own scratch directory, then removes the directory. Every call exercises
// the full out-of-core path: init_from_vamana() followed by the
// omp-parallel-for update_qg_out_of_memory() loop.
void build_and_discard_one_index(uint32_t seed) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() /
      ("qg_builder_oom_regression_" + std::to_string(::getpid()) + "_" + std::to_string(seed));
  std::filesystem::create_directories(dir);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(path, ec);
    }
  } cleanup{dir};

  const std::string prefix = (dir / "tiny").string();
  const std::vector<float> data = make_data(kN, kDim, seed);
  write_fbin(prefix + "_pca_base.fbin", data.data(), static_cast<int32_t>(kN),
             static_cast<int32_t>(kDim));

  alaya::vamana::VamanaBuildParams vp;
  vp.R = kDeg;
  vp.L = 64;
  vp.alpha = 1.2F;
  vp.num_threads = 4;
  alaya::vamana::VamanaBuilder vb(data.data(), kN, kDim, vp);
  vb.build();
  const std::string vamana_path = prefix + "_vamana.index";
  alaya::vamana::save_graph(vb.graph(), vamana_path, kDeg, vb.medoid());

  QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/static_cast<uint64_t>(seed));
  QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/4);
  builder.build(vamana_path.c_str(), prefix.c_str());
}

}  // namespace

TEST(QGBuilderOomRegressionTest, RepeatedBuildsInOneProcessDoNotCorruptTheHeap) {
  for (int round = 0; round < kRounds; ++round) {
    SCOPED_TRACE(::testing::Message() << "round " << round);
    // A fresh seed per round: the trigger is data-dependent (which nodes
    // VamanaBuilder's cleanup pass leaves above R), not merely "build twice".
    build_and_discard_one_index(/*seed=*/9000U + static_cast<uint32_t>(round));
  }
  SUCCEED() << kRounds << " back-to-back builds completed without heap corruption";
}

}  // namespace alaya::laser
