// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Kernel-level tests for the segment admission contract
// (docs/design/segment-admission-contract.md):
//   - acceptance #2: tombstone parity between the legacy result_filter_
//     exclude-set path and the RowAdmission bitmap path built from the
//     same exclude set (admission_from_exclude_set), for both the paged
//     and arena admit points.
//   - a bitmap-filter correctness sanity check (every returned row must
//     satisfy the filter) as a precursor to the disk-layer recall test.
//   - acceptance #4: admission overhead vs. the exclude-set hash probe at
//     an equal live ratio (simple wall-clock timing, not a CI assertion --
//     the contract only asks that the numbers be reported).
//
// All tests share one small on-disk index (built once in
// SetUpTestSuite()), each loading its own QuantizedGraph instance(s) from
// it. QGBuilder::build()'s out-of-memory patch path
// (QuantizedGraph::update_qg_out_of_memory, reached from the omp-parallel
// clone of QGBuilder::build()) used to have a data-dependent heap overflow,
// independent of this change, when several tiny indices were built
// back-to-back in one process (confirmed via gdb: SIGSEGV inside a
// memmove). Fixed in qg_builder.hpp (init_from_vamana()'s per-node degree
// clamp against degree_bound_, plus build()'s hard vector-file dimension
// check) -- see tests/laser/qg/test_qg_builder_oom_regression.cpp for the
// regression test that rebuilds several indices per process. This file
// still builds its shared index exactly once per process (a plain,
// deliberate fixture-sharing choice, not a workaround, now that the
// constraint is lifted) because SetUpTestSuite() sharing it across TEST_Fs
// remains simpler and faster than rebuilding per test.
//
// Separately: this fixture used to call QGBuilder::build() without ever
// writing "{prefix}_pca_base.fbin" first, even though build() documents
// that file as a required input -- the missing file left build()'s vector
// file dimension probe (`int n, d`) reading from a closed/never-opened
// ifstream, which silently leaves `d` as indeterminate stack garbage
// instead of an error. That is almost certainly the true trigger behind
// the historical "~2/5 of runs crash" observation this comment used to
// describe: build()'s hardened dimension check now turns that into a
// deterministic, immediate, diagnosable failure instead of undefined
// behavior, which is what surfaced the missing write_fbin() call below.

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/row_admission.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kDeg = 64;
constexpr size_t kN = 2000;

std::vector<float> make_data(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (auto &v : data) {
    v = dist(gen);
  }
  return data;
}

// QGBuilder::build() requires "{filename}_pca_base.fbin" to already exist --
// its own doc comment says so, and it is the raw float32 vector data PHASE 1
// streams through. Matches the two-int32-header fbin layout every other
// LASER test fixture in this tree writes (e.g.
// tests/laser/qg/qg_wal_test_support.hpp's write_fbin()).
void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * static_cast<size_t>(n) *
                                         static_cast<size_t>(dim)));
}

// Move-only: TinyIndex::build() returns by value, and SetUpTestSuite() below
// wraps that in std::make_unique<TinyIndex>(...), which binds the returned
// prvalue as a constructor argument rather than eliding into the final
// storage. A defaulted/implicit copy would leave two TinyIndex objects
// pointing at the same `dir`; the temporary's destructor would then delete
// the directory out from under the surviving one (this was caught the hard
// way: every test failed with "file not found" immediately after switching
// to a shared, once-built index). Deleting copy and hand-rolling move
// (clearing the source's `dir`) makes the destructor safe either way.
struct TinyIndex {
  std::filesystem::path dir;
  std::string prefix;
  std::vector<float> data;

  TinyIndex() = default;
  TinyIndex(const TinyIndex &) = delete;
  auto operator=(const TinyIndex &) -> TinyIndex & = delete;
  TinyIndex(TinyIndex &&other) noexcept
      : dir(std::move(other.dir)), prefix(std::move(other.prefix)), data(std::move(other.data)) {
    other.dir.clear();
  }
  auto operator=(TinyIndex &&other) noexcept -> TinyIndex & {
    if (this != &other) {
      std::error_code ec;
      if (!dir.empty()) {
        std::filesystem::remove_all(dir, ec);
      }
      dir = std::move(other.dir);
      prefix = std::move(other.prefix);
      data = std::move(other.data);
      other.dir.clear();
    }
    return *this;
  }

  static TinyIndex build(uint32_t seed) {
    TinyIndex t;
    t.dir = std::filesystem::temp_directory_path() /
            ("admission_contract_test_" + std::to_string(::getpid()) + "_" + std::to_string(seed));
    std::filesystem::create_directories(t.dir);
    t.prefix = (t.dir / "tiny").string();
    t.data = make_data(kN, kDim, seed);
    write_fbin(t.prefix + "_pca_base.fbin", t.data.data(), static_cast<int32_t>(kN),
               static_cast<int32_t>(kDim));

    alaya::vamana::VamanaBuildParams vp;
    vp.R = kDeg;
    vp.L = 64;
    vp.alpha = 1.2F;
    vp.num_threads = 4;
    alaya::vamana::VamanaBuilder vb(t.data.data(), kN, kDim, vp);
    vb.build();
    const std::string vamana_path = t.prefix + "_vamana.index";
    alaya::vamana::save_graph(vb.graph(), vamana_path, kDeg, vb.medoid());

    QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
    QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/4);
    builder.build(vamana_path.c_str(), t.prefix.c_str());
    return t;
  }

  ~TinyIndex() {
    if (dir.empty()) {
      return;
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

auto load(const TinyIndex &tiny) -> std::unique_ptr<QuantizedGraph> {
  auto qg = std::make_unique<QuantizedGraph>(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
  qg->load_disk_index(tiny.prefix.c_str(), /*search_DRAM_budget=*/4.0F);
  qg->set_params(/*ef_search=*/96, /*num_threads=*/1, /*beam_width=*/4);
  return qg;
}

class AdmissionContractTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { shared_index_ = std::make_unique<TinyIndex>(TinyIndex::build(1001)); }

  static void TearDownTestSuite() { shared_index_.reset(); }

  static std::unique_ptr<TinyIndex> shared_index_;
};

std::unique_ptr<TinyIndex> AdmissionContractTest::shared_index_;

// ---------------------------------------------------------------------------
// Acceptance #2: tombstone parity.
// ---------------------------------------------------------------------------

// The paged kernel consumes completed reads in logical frontier order, so two
// equivalent admission mechanisms must now produce byte-identical top-k
// results even when the backend completes reads out of order.
TEST_F(AdmissionContractTest, TombstoneParityPagedKernel) {
  const TinyIndex &tiny = *shared_index_;
  auto qg_legacy = load(tiny);
  auto qg_admission = load(tiny);

  const std::unordered_set<PID> dead = {3, 17, 42, 99, 250, 777, 1024, 1999};
  qg_legacy->set_result_filter(&dead);

  std::vector<uint64_t> storage;
  const RowAdmission admission = admission_from_exclude_set(dead, kN, storage);

  constexpr uint32_t kK = 10;
  for (uint32_t qi = 0; qi < 25; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi) * kDim;
    std::vector<uint32_t> legacy_out(kK);
    std::vector<uint32_t> admission_out(kK);
    qg_legacy->search(query, kK, legacy_out.data());
    qg_admission->search(query, kK, admission_out.data(), &admission);
    EXPECT_EQ(admission_out, legacy_out) << "paged admission parity, query " << qi;
    for (auto r : legacy_out) {
      EXPECT_EQ(dead.count(r), 0U) << "tombstoned id leaked into legacy path results, query " << qi;
    }
    for (auto r : admission_out) {
      EXPECT_EQ(dead.count(r), 0U)
          << "tombstoned id leaked into admission path results, query " << qi;
    }
  }
  qg_legacy->set_result_filter(nullptr);
}

TEST_F(AdmissionContractTest, TombstoneParityArenaKernel) {
  const TinyIndex &tiny = *shared_index_;
  auto qg_legacy = load(tiny);
  auto qg_admission = load(tiny);
  qg_legacy->ensure_resident_arena();
  qg_admission->ensure_resident_arena();

  const std::unordered_set<PID> dead = {5, 40, 88, 401, 900, 1500, 1888};
  qg_legacy->set_result_filter(&dead);

  std::vector<uint64_t> storage;
  const RowAdmission admission = admission_from_exclude_set(dead, kN, storage);

  constexpr uint32_t kK = 10;
  for (uint32_t qi = 0; qi < 25; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi) * kDim;
    std::vector<uint32_t> legacy_out(kK);
    std::vector<uint32_t> admission_out(kK);
    qg_legacy->arena_search_qg(query, kK, legacy_out.data());
    qg_admission->arena_search_qg(query, kK, admission_out.data(), &admission);
    EXPECT_EQ(legacy_out, admission_out) << "query " << qi;
    for (auto r : legacy_out) {
      EXPECT_EQ(dead.count(r), 0U) << "tombstoned id leaked into legacy arena path results";
    }
  }
  qg_legacy->set_result_filter(nullptr);
}

// ---------------------------------------------------------------------------
// Bitmap-filter correctness sanity (precursor to the disk-layer recall
// test): every row the kernel returns under an active admission must
// satisfy the bitmap.
// ---------------------------------------------------------------------------

TEST_F(AdmissionContractTest, BitmapFilterOnlyReturnsAdmissibleRowsPagedKernel) {
  const TinyIndex &tiny = *shared_index_;
  auto qg = load(tiny);

  std::vector<uint64_t> rows;
  for (uint64_t i = 0; i < kN; ++i) {
    if (i % 10 < 3) {  // ~30% selectivity
      rows.push_back(i);
    }
  }
  std::vector<uint64_t> storage;
  const RowAdmission admission = admission_from_sorted_rows(rows.data(), rows.size(), kN, storage);
  const std::unordered_set<uint64_t> allowed(rows.begin(), rows.end());
  EXPECT_NEAR(admission.popcount, kN * 3 / 10, 5U);

  constexpr uint32_t kK = 10;
  size_t total_hits = 0;
  for (uint32_t qi = 0; qi < 30; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi) * kDim;
    std::vector<uint32_t> out(kK);
    qg->search(query, kK, out.data(), &admission);
    for (auto pid : out) {
      ++total_hits;
      EXPECT_TRUE(allowed.count(pid) > 0) << "pid " << pid << " fails the bitmap filter";
    }
  }
  EXPECT_EQ(total_hits, 30U * kK);
}

TEST_F(AdmissionContractTest, BitmapFilterOnlyReturnsAdmissibleRowsArenaKernel) {
  const TinyIndex &tiny = *shared_index_;
  auto qg = load(tiny);
  qg->ensure_resident_arena();

  std::vector<uint64_t> rows;
  for (uint64_t i = 0; i < kN; ++i) {
    if (i % 10 < 3) {
      rows.push_back(i);
    }
  }
  std::vector<uint64_t> storage;
  const RowAdmission admission = admission_from_sorted_rows(rows.data(), rows.size(), kN, storage);
  const std::unordered_set<uint64_t> allowed(rows.begin(), rows.end());

  constexpr uint32_t kK = 10;
  size_t total_hits = 0;
  for (uint32_t qi = 0; qi < 30; ++qi) {
    const float *query = tiny.data.data() + static_cast<size_t>(qi) * kDim;
    std::vector<uint32_t> out(kK);
    qg->arena_search_qg(query, kK, out.data(), &admission);
    for (auto pid : out) {
      ++total_hits;
      EXPECT_TRUE(allowed.count(pid) > 0) << "pid " << pid << " fails the bitmap filter";
    }
  }
  EXPECT_EQ(total_hits, 30U * kK);
}

// ---------------------------------------------------------------------------
// Acceptance #4: admission overhead vs. the exclude-set hash probe at an
// equal live ratio. Simple wall-clock timing; the contract wants the
// numbers reported (section-profile-style evidence), not a CI bound, since
// microbenchmarks on a shared/virtualized host are noisy.
// ---------------------------------------------------------------------------

TEST_F(AdmissionContractTest, PerformanceAdmissionVsExcludeSetEqualLiveRatioArenaKernel) {
  const TinyIndex &tiny = *shared_index_;
  auto qg_exclude = load(tiny);
  auto qg_admission = load(tiny);
  qg_exclude->ensure_resident_arena();
  qg_admission->ensure_resident_arena();

  // 10% dead / 90% live -- a representative tombstone churn ratio.
  std::unordered_set<PID> dead;
  for (PID p = 0; p < kN; p += 10) {
    dead.insert(p);
  }
  qg_exclude->set_result_filter(&dead);

  std::vector<uint64_t> storage;
  const RowAdmission admission = admission_from_exclude_set(dead, kN, storage);

  constexpr uint32_t kK = 10;
  constexpr int kWarmup = 20;
  constexpr int kIterations = 400;
  std::vector<uint32_t> out(kK);

  for (int i = 0; i < kWarmup; ++i) {
    const float *query = tiny.data.data() + static_cast<size_t>(i % kN) * kDim;
    qg_exclude->arena_search_qg(query, kK, out.data());
    qg_admission->arena_search_qg(query, kK, out.data(), &admission);
  }

  const auto exclude_start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const float *query = tiny.data.data() + static_cast<size_t>(i % kN) * kDim;
    qg_exclude->arena_search_qg(query, kK, out.data());
  }
  const auto exclude_end = std::chrono::steady_clock::now();

  const auto admission_start = std::chrono::steady_clock::now();
  for (int i = 0; i < kIterations; ++i) {
    const float *query = tiny.data.data() + static_cast<size_t>(i % kN) * kDim;
    qg_admission->arena_search_qg(query, kK, out.data(), &admission);
  }
  const auto admission_end = std::chrono::steady_clock::now();

  const double exclude_us =
      std::chrono::duration<double, std::micro>(exclude_end - exclude_start).count() /
      kIterations;
  const double admission_us =
      std::chrono::duration<double, std::micro>(admission_end - admission_start).count() /
      kIterations;
  std::cout << "admission_vs_excludeset,live_ratio=0.9,excludeset_us_per_query=" << exclude_us
            << ",admission_us_per_query=" << admission_us
            << ",ratio_admission_over_excludeset=" << (admission_us / exclude_us) << "\n";

  qg_exclude->set_result_filter(nullptr);
  // No hard assertion: this is evidence for the report (contract acceptance
  // #4), not a CI gate -- shared/virtualized hosts are too noisy for a
  // tight bound. A generous sanity bound still catches a gross regression
  // (e.g. an accidental O(N) admission scan per candidate).
  EXPECT_LT(admission_us, exclude_us * 5.0)
      << "admission path is more than 5x the exclude-set path; investigate before trusting this "
         "as evidence";
}

}  // namespace
}  // namespace alaya::laser
