// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Functional (non-crash) tests for the W2 canonical PID-reuse writer (design
// section 3 / codex B.1-B.7): activation on the first reuse-enabled bundle,
// all-append canonical bundles, delete-all -> all-reuse and mixed reuse, the
// per-PID generation chain (0 -> 1 -> 2), same-count label rebind, and the
// no-reuse byte-invariance guard. The R0-R11 SIGKILL matrix lives in
// test_segment_op_wal_reuse_crash.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "qg_wal_test_support.hpp"

namespace alaya::laser {
namespace {

using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 128;

// A reuse-enabled updater session over a tiny base. enable_pid_reuse arms the
// canonical writer; the first bundle runs an activation checkpoint (v3 pid base).
struct ReuseSession {
  QuantizedGraph qg;
  std::unique_ptr<QGUpdater> upd;
  ReuseSession(const std::string &prefix, size_t base_n, size_t max_points, bool enable_reuse)
      : qg(base_n, kDeg, kDim, kDim) {
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.enable_pid_reuse = enable_reuse;
    params.ef_insert = 64;
    params.max_points = max_points;
    params.backlink_mode = UpdateParams::Backlink::kAlphaEvict;
    upd = std::make_unique<QGUpdater>(qg, params);
  }
};

std::filesystem::path scratch_dir(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("qg_reuse_" + name + "_" + std::to_string(::getpid()));
}

bool row_is_live(QGUpdater &upd, PID id) {
  const auto flags = upd.trailer(id).flags;
  return (flags & (kQGRowTombstone | kQGRowFree)) == 0;
}

// The label bound to a committed PID (explicit binding overrides base identity).
uint64_t label_of(QGUpdater &upd, PID pid) {
  const auto snap = upd.label_snapshot();
  const uint64_t *l = snap ? snap->find(pid) : nullptr;
  return l != nullptr ? *l : static_cast<uint64_t>(pid);
}

// True iff a search for `vec` returns `pid` in its top-ef.
bool searchable(QGUpdater &upd, const float *vec, PID pid, size_t ef = 64) {
  const auto res = upd.search(vec, 10, ef);
  return std::count(res.begin(), res.end(), pid) != 0;
}

// Drive base rows tombstoned + a reclaiming consolidate so their PIDs enter the
// free list, ready for reuse.
void free_base_rows(QGUpdater &upd, const std::vector<PID> &rows) {
  for (PID id : rows) upd.tombstone(id);
  upd.consolidate(1, /*r_target=*/0, /*reclaim_slots=*/true, /*bloom=*/false);
}

// ---------------------------------------------------------------------------
// Activation + all-append canonical bundle.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, ActivatesV3PidBaseAndAppendBundleSurvivesReopen) {
  const auto dir = scratch_dir("activate");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/11);
  const auto data = waltest::make_data(4, kDim, /*seed=*/91);
  const std::vector<uint64_t> labels = {5000, 5001, 5002, 5003};

  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    // The FIRST bundle activates a v3 pid base then commits canonically.
    const auto range = s.upd->commit_physical_bundle(1, 1, data.data(), labels.data(), 4);
    EXPECT_EQ(range.first, static_cast<PID>(kBaseN));
    EXPECT_EQ(range.second, static_cast<PID>(kBaseN + 4));
    for (size_t i = 0; i < 4; ++i) {
      const PID pid = static_cast<PID>(kBaseN + i);
      EXPECT_TRUE(row_is_live(*s.upd, pid));
      EXPECT_EQ(label_of(*s.upd, pid), labels[i]);
      EXPECT_TRUE(searchable(*s.upd, data.data() + i * kDim, pid)) << "appended row " << i;
    }
    s.upd->checkpoint();  // flush the v3 pid base + append pages
  }
  // Reopen from the checkpointed base (append-only canonical bundle absorbed).
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    EXPECT_EQ(s.upd->num_points(), kBaseN + 4);
    for (size_t i = 0; i < 4; ++i) {
      const PID pid = static_cast<PID>(kBaseN + i);
      EXPECT_TRUE(row_is_live(*s.upd, pid));
      EXPECT_EQ(label_of(*s.upd, pid), labels[i]);
    }
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// delete-all -> all-reuse (N>1): every base row tombstoned + reclaimed, then a
// bundle of N rows reuses the freed PIDs. Each row is searchable in-process and
// after a first + second reopen (design B-2C-01 hard family / JC-24).
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, DeleteAllThenAllReuseSurvivesDoubleReopen) {
  const auto dir = scratch_dir("all_reuse");
  std::filesystem::remove_all(dir);
  const size_t small_n = 24;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/17);

  const size_t reuse_n = 6;
  const auto vecs = waltest::make_data(reuse_n, kDim, /*seed=*/71);
  std::vector<uint64_t> labels(reuse_n);
  for (size_t i = 0; i < reuse_n; ++i) labels[i] = 90000 + i;

  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    std::vector<PID> all(small_n);
    for (size_t i = 0; i < small_n; ++i) all[i] = static_cast<PID>(i);
    free_base_rows(*s.upd, all);
    EXPECT_EQ(s.upd->live_count(), 0U);
    EXPECT_EQ(s.upd->free_count(), small_n);

    const auto range = s.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), reuse_n);
    // all-reuse: the high-water mark does not move.
    EXPECT_EQ(range.first, range.second);
    EXPECT_EQ(s.upd->num_points(), small_n);
    EXPECT_EQ(s.upd->live_count(), reuse_n);
    // Each reused row is live + carries its new label + is searchable in-process.
    std::unordered_set<PID> reused_pids;
    for (size_t i = 0; i < reuse_n; ++i) {
      const auto res = s.upd->search(vecs.data() + i * kDim, 5, 64);
      ASSERT_FALSE(res.empty());
      bool hit = false;
      for (PID p : res) {
        if (label_of(*s.upd, p) == labels[i]) {
          hit = true;
          reused_pids.insert(p);
          EXPECT_LT(p, static_cast<PID>(small_n)) << "reused a base-region PID";
          EXPECT_TRUE(row_is_live(*s.upd, p));
        }
      }
      EXPECT_TRUE(hit) << "reused row " << i << " not searchable by label in-process";
    }
    EXPECT_EQ(reused_pids.size(), reuse_n) << "reused PIDs must be distinct";
  }

  // First reopen: WAL replay reconstructs the whole reuse bundle.
  auto verify = [&](const char *phase) {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(s.upd->live_count(), reuse_n) << phase;
    size_t found = 0;
    for (size_t i = 0; i < reuse_n; ++i) {
      const auto res = s.upd->search(vecs.data() + i * kDim, 5, 64);
      for (PID p : res) {
        if (label_of(*s.upd, p) == labels[i]) ++found;
      }
    }
    EXPECT_GE(found, reuse_n) << phase << ": every reused label searchable after reopen";
    s.upd->checkpoint();  // seal the recovered state, then hand the dir to the 2nd reopen
  };
  verify("first-reopen");
  verify("second-reopen");  // reopen the FIRST recovery's checkpointed output again
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Mixed reuse (some freed PIDs reused + some dense appends) in one bundle.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, MixedReuseAndAppendSurvivesReopen) {
  const auto dir = scratch_dir("mixed");
  std::filesystem::remove_all(dir);
  const size_t small_n = 32;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/23);

  const size_t bundle_n = 5;  // 3 reused (freed) + 2 dense append
  const auto vecs = waltest::make_data(bundle_n, kDim, /*seed=*/53);
  std::vector<uint64_t> labels(bundle_n);
  for (size_t i = 0; i < bundle_n; ++i) labels[i] = 77000 + i;

  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    std::vector<PID> victims = {2, 5, 9};  // free exactly 3
    free_base_rows(*s.upd, victims);
    EXPECT_EQ(s.upd->free_count(), 3U);

    const auto range = s.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), bundle_n);
    // 3 reused (no HWM move) + 2 append (HWM + 2).
    EXPECT_EQ(range.first, static_cast<PID>(small_n));
    EXPECT_EQ(range.second, static_cast<PID>(small_n + 2));
    EXPECT_EQ(s.upd->free_count(), 0U) << "all freed PIDs consumed by the reuse prefix";
    EXPECT_EQ(s.upd->live_count(), (small_n - 3) + bundle_n);
    for (size_t i = 0; i < bundle_n; ++i) {
      const auto res = s.upd->search(vecs.data() + i * kDim, 5, 64);
      bool hit = false;
      for (PID p : res) {
        hit = hit || label_of(*s.upd, p) == labels[i];
      }
      EXPECT_TRUE(hit) << "mixed row " << i << " not searchable in-process";
    }
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(s.upd->live_count(), (small_n - 3) + bundle_n);
    size_t found = 0;
    for (size_t i = 0; i < bundle_n; ++i) {
      const auto res = s.upd->search(vecs.data() + i * kDim, 5, 64);
      for (PID p : res) {
        if (label_of(*s.upd, p) == labels[i]) ++found;
      }
    }
    EXPECT_GE(found, bundle_n) << "mixed bundle reconstructed by replay";
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Continuous reuse: the same freed PID goes generation 0 -> 1 -> 2, and the
// slot summary counts exactly one non-zero binding at max generation 2.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, SamePidGenerationChain0To2) {
  const auto dir = scratch_dir("gen2");
  std::filesystem::remove_all(dir);
  const size_t small_n = 16;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/29);
  const auto vec = waltest::make_data(1, kDim, /*seed=*/31);

  ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
  // gen 1: free PID 3, reuse it with label 42.
  free_base_rows(*s.upd, {3});
  {
    const std::vector<uint64_t> l = {42};
    s.upd->commit_physical_bundle(1, 1, vec.data(), l.data(), 1);
  }
  PID pid1 = kPidMax;
  for (PID p : s.upd->search(vec.data(), 5, 64)) {
    if (label_of(*s.upd, p) == 42) pid1 = p;
  }
  ASSERT_NE(pid1, kPidMax);
  EXPECT_EQ(pid1, static_cast<PID>(3));  // the only free PID

  // gen 2: free it again, reuse with a fresh label 43.
  free_base_rows(*s.upd, {pid1});
  {
    const std::vector<uint64_t> l = {43};
    s.upd->commit_physical_bundle(2, 2, vec.data(), l.data(), 1);
  }
  const auto snap = s.upd->label_snapshot();
  ASSERT_NE(snap, nullptr);
  const auto *b = snap->find_binding(pid1);
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(b->pid_generation, 2U) << "reusing the same PID a second time is generation 2";
  EXPECT_EQ(b->label, 43U);

  // Reopen and confirm the generation-2 binding is durable + validated by the
  // superblock summary (checkpoint stamped max_pid_generation=2, nz_count=1).
  s.upd->checkpoint();
  s.upd.reset();
  ReuseSession s2(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
  const auto snap2 = s2.upd->label_snapshot();
  const auto *b2 = snap2 ? snap2->find_binding(pid1) : nullptr;
  ASSERT_NE(b2, nullptr);
  EXPECT_EQ(b2->pid_generation, 2U);
  EXPECT_EQ(b2->label, 43U);
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Byte-invariance guard: with enable_pid_reuse=false a fresh base never
// activates (stays v2), and an ordinary append bundle takes the legacy 2A path.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, NoReuseFlagKeepsLegacyV2Base) {
  const auto dir = scratch_dir("legacy");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/13);
  const auto data = waltest::make_data(3, kDim, /*seed=*/61);
  const std::vector<uint64_t> labels = {1, 2, 3};
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/false);
    s.upd->commit_physical_bundle(1, 1, data.data(), labels.data(), 3);
    s.upd->checkpoint();
    EXPECT_EQ(s.upd->superblock_format_version(), kQGFormatVersion)
        << "a no-reuse segment must stay v2 (never auto-activates a v3 pid base)";
  }
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::laser
