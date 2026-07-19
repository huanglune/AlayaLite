// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Unit tests for the residency seams behind UnifiedLaserSegment (U-line plan C):
//   1. arena_prefetch_default: env override wins; otherwise min(row_lines, 20).
//   2. ResidencyProvider dispatch is a pass-through over the two existing
//      kernels (paged == QuantizedGraph::search, arena == arena_search_qg).
//   3. ensure_resident_arena materializes a byte-faithful arena straight from
//      the index file when the cache sidecar was budget-truncated — residency
//      is a load-time policy, not a build-time family choice.
//   4. QGUpdater's write_at mirror keeps resident searches fresh across
//      insert + writeback (append row and reverse-edge patches both land).

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/laser/qg/residency.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kDeg = 64;  // leaves format-v2 trailer slack (see updater test)
constexpr size_t kN = 2000;

#if defined(__SANITIZE_THREAD__)
constexpr bool kRunningTsan = true;
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
constexpr bool kRunningTsan = true;
  #else
constexpr bool kRunningTsan = false;
  #endif
#else
constexpr bool kRunningTsan = false;
#endif

std::vector<float> make_data(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (auto &v : data) {
    v = dist(gen);
  }
  return data;
}

void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * n * dim));
}

struct TinyIndex {
  std::filesystem::path dir;
  std::string prefix;
  std::vector<float> data;

  static TinyIndex build(uint32_t seed) {
    TinyIndex t;
    t.dir = std::filesystem::temp_directory_path() /
            ("unified_residency_test_" + std::to_string(::getpid()) + "_" + std::to_string(seed));
    std::filesystem::create_directories(t.dir);
    t.prefix = (t.dir / "tiny").string();
    t.data = make_data(kN, kDim, seed);

    alaya::vamana::VamanaBuildParams vp;
    vp.R = kDeg;
    vp.L = 64;
    vp.alpha = 1.2F;
    vp.num_threads = kRunningTsan ? 1 : 4;
    alaya::vamana::VamanaBuilder vb(t.data.data(), kN, kDim, vp);
    vb.build();
    const std::string vamana_path = t.prefix + "_vamana.index";
    alaya::vamana::save_graph(vb.graph(), vamana_path, kDeg, vb.medoid());

    write_fbin(t.prefix + "_pca_base.fbin", t.data.data(), kN, kDim);

    QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
    QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/kRunningTsan ? 1 : 4);
    builder.build(vamana_path.c_str(), t.prefix.c_str());
    return t;
  }

  ~TinyIndex() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

std::vector<uint32_t> run_search(ResidencyProvider &provider,
                                 QuantizedGraph &qg,
                                 const float *query,
                                 uint32_t knn) {
  std::vector<uint32_t> out(knn);
  provider.search(qg,
                  query,
                  knn,
                  out.data(),
                  /*ef_search=*/96,
                  /*beam_width=*/4);
  return out;
}

TEST(UnifiedResidency, PrefetchDefaultBudget) {
  // env override wins verbatim (0 = disable is a valid override).
  EXPECT_EQ(arena_prefetch_default(/*row_lines=*/120, /*env_lines=*/0), 0U);
  EXPECT_EQ(arena_prefetch_default(120, 10), 10U);
  EXPECT_EQ(arena_prefetch_default(120, 48), 48U);
  // No env: min(row_lines, 20) — short rows prefetch whole rows, long rows
  // stop at the ~1.3KB in-flight budget (E1).
  EXPECT_EQ(arena_prefetch_default(9, -1), 9U);
  EXPECT_EQ(arena_prefetch_default(20, -1), 20U);
  EXPECT_EQ(arena_prefetch_default(120, -1), 20U);
}

TEST(UnifiedResidency, ResidencyModeStrings) {
  EXPECT_EQ(residency_mode_from_string("paged_pool"), ResidencyMode::kPagedPool);
  EXPECT_EQ(residency_mode_from_string("resident_arena"), ResidencyMode::kResidentArena);
  EXPECT_EQ(residency_mode_to_string(ResidencyMode::kPagedPool), "paged_pool");
  EXPECT_EQ(residency_mode_to_string(ResidencyMode::kResidentArena), "resident_arena");
  EXPECT_THROW((void)residency_mode_from_string("mmap"), std::invalid_argument);
}

TEST(UnifiedResidency, ProviderDispatchMatchesKernels) {
  const TinyIndex tiny = TinyIndex::build(/*seed=*/11);

  QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
  qg.load_disk_index(tiny.prefix.c_str(), /*search_DRAM_budget=*/4.0F);
  // Builder sidecars order cache ids by in-degree (hot rows first), so even a
  // 100% sidecar is NOT an identity arena at load time; prepare() must
  // materialize it (this is the ResidencyProvider contract).
  ASSERT_FALSE(qg.arena_resident()) << "in-degree-sorted sidecar must not count as an arena";
  qg.set_params(/*ef_search=*/96, /*num_threads=*/1, /*beam_width=*/4);

  auto paged = make_residency_provider(ResidencyMode::kPagedPool);
  auto arena = make_residency_provider(ResidencyMode::kResidentArena);
  EXPECT_EQ(paged->mode(), ResidencyMode::kPagedPool);
  EXPECT_EQ(arena->mode(), ResidencyMode::kResidentArena);
  paged->prepare(qg);
  arena->prepare(qg);
  ASSERT_TRUE(qg.arena_resident()) << "prepare() must materialize the resident arena";
  arena->prepare(qg);  // idempotent on an already-resident graph

  constexpr uint32_t kK = 10;
  for (uint32_t qi = 0; qi < 8; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi) * kDim;

    const auto via_provider_paged = run_search(*paged, qg, query, kK);
    std::vector<uint32_t> direct_paged(kK);
    qg.search(query, kK, direct_paged.data());
    EXPECT_EQ(via_provider_paged, direct_paged) << "paged provider must be pass-through";

    const auto via_provider_arena = run_search(*arena, qg, query, kK);
    std::vector<uint32_t> direct_arena(kK);
    qg.arena_search_qg(query, kK, direct_arena.data());
    EXPECT_EQ(via_provider_arena, direct_arena) << "arena provider must be pass-through";

    // Self-query: the point itself is at distance zero.
    EXPECT_EQ(via_provider_arena.front(), qi);
  }

  ResidentArenaProvider interleave{NumaPolicy{NumaPolicy::Kind::kInterleave}};
  EXPECT_THROW(interleave.prepare(qg), std::logic_error)
      << "NUMA interleave is an interface hook only in v1";
}

TEST(UnifiedResidency, EnsureResidentArenaFromFile) {
  const TinyIndex tiny = TinyIndex::build(/*seed=*/23);

  // Reference: full-budget load (full in-degree-sorted sidecar), then
  // materialize the identity arena from the file.
  QuantizedGraph full(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
  full.load_disk_index(tiny.prefix.c_str(), /*search_DRAM_budget=*/4.0F);
  full.ensure_resident_arena();
  ASSERT_TRUE(full.arena_resident());
  full.set_params(96, 1, 4);

  // Probe: budget-truncated sidecar (0 cached rows) — materialization must be
  // independent of whatever partial cache state the load left behind.
  QuantizedGraph probe(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
  probe.load_disk_index(tiny.prefix.c_str(), /*search_DRAM_budget=*/0.000001F);
  ASSERT_FALSE(probe.arena_resident()) << "a ~800B budget must not cache 2000 rows";

  probe.ensure_resident_arena();
  ASSERT_TRUE(probe.arena_resident());
  probe.ensure_resident_arena();  // idempotent
  probe.set_params(96, 1, 4);

  constexpr uint32_t kK = 10;
  for (uint32_t qi = 0; qi < 16; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi * 7 % kN) * kDim;
    std::vector<uint32_t> want(kK);
    std::vector<uint32_t> got(kK);
    full.arena_search_qg(query, kK, want.data());
    probe.arena_search_qg(query, kK, got.data());
    EXPECT_EQ(got, want) << "file-materialized arena must be byte-faithful (query " << qi << ")";
  }
}

TEST(UnifiedResidency, UpdaterMirrorKeepsArenaFresh) {
  const TinyIndex tiny = TinyIndex::build(/*seed=*/37);

  QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
  qg.load_disk_index(tiny.prefix.c_str(), /*search_DRAM_budget=*/4.0F);
  qg.ensure_resident_arena();
  ASSERT_TRUE(qg.arena_resident());
  qg.set_params(96, 1, 4);

  UpdateParams params;  // defaults: write cache on -> writeback() drives write_at
  QGUpdater updater(qg, params);

  const auto extra = make_data(1, kDim, /*seed=*/91);
  const PID appended = updater.insert(extra.data());
  EXPECT_GE(static_cast<size_t>(appended), kN);
  updater.writeback(/*num_threads=*/1);

  // The resident kernel (not the updater's read-through path) must see both
  // the appended row and the reverse-edge patches that make it reachable.
  constexpr uint32_t kK = 10;
  std::vector<uint32_t> got(kK);
  qg.arena_search_qg(extra.data(), kK, got.data());
  EXPECT_EQ(got.front(), appended)
      << "freshly appended vector must be its own nearest neighbor in the arena";
}

}  // namespace
}  // namespace alaya::laser
