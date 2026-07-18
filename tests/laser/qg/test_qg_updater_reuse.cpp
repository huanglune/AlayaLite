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
#include <cstring>
#include <filesystem>
#include <memory>
#include <queue>
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
// BLOCKER-1/2 regression: delete-all -> all-reuse N>1 under kFullPrune with a tiny
// prune pool (a later row's full-recompute would otherwise rewrite an earlier spine
// edge, and the persisted dead entry is reused). The FINAL bidirectional spine + the
// repair_routing_roots entry rule must keep EVERY reused row reachable both in-process
// and after a replay reopen AND a checkpoint reopen -- the two must agree.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, DeleteAllAllReuseFullPruneStaysReachableCleanAndReplay) {
  const auto dir = scratch_dir("fullprune");
  std::filesystem::remove_all(dir);
  const size_t small_n = 20;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/37);
  const size_t reuse_n = 6;
  const auto vecs = waltest::make_data(reuse_n, kDim, /*seed=*/44);
  std::vector<uint64_t> labels(reuse_n);
  for (size_t i = 0; i < reuse_n; ++i) labels[i] = 61000 + i;

  // Holder: the QuantizedGraph must outlive the QGUpdater (which keeps a reference).
  struct Holder {
    QuantizedGraph qg;
    std::unique_ptr<QGUpdater> upd;
    Holder(const std::string &prefix, size_t base_n) : qg(base_n, kDeg, kDim, kDim) {
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.enable_pid_reuse = true;
      params.ef_insert = 64;
      params.max_points = 4 * base_n;
      params.backlink_mode = UpdateParams::Backlink::kFullPrune;  // the clobber-prone mode
      params.prune_pool_cap = 1;                                  // maximal spine pressure
      upd = std::make_unique<QGUpdater>(qg, params);
    }
  };
  auto open = [&](bool) { return std::make_unique<Holder>(base.prefix, small_n); };
  const auto all_labels_found = [&](QGUpdater &upd) {
    size_t found = 0;
    for (size_t i = 0; i < reuse_n; ++i) {
      const auto res = upd.search(vecs.data() + i * kDim, 8, 64);
      for (PID p : res) {
        if (label_of(upd, p) == labels[i]) {
          ++found;
          break;
        }
      }
    }
    return found;
  };

  {
    auto h = open(true);
    for (size_t i = 0; i < small_n; ++i) h->upd->tombstone(static_cast<PID>(i));
    h->upd->consolidate(1, 0, true, false);
    EXPECT_EQ(h->upd->live_count(), 0U);
    h->upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), reuse_n);
    EXPECT_EQ(all_labels_found(*h->upd), reuse_n) << "in-process: a bundle row is unreachable";
    // Do NOT checkpoint: the next reopen is a pure WAL replay.
  }
  {
    auto h = open(true);
    EXPECT_EQ(all_labels_found(*h->upd), reuse_n) << "replay reopen: a bundle row is unreachable";
    h->upd->checkpoint();  // seal, hand to the checkpoint-reopen below
  }
  {
    auto h = open(true);
    EXPECT_EQ(all_labels_found(*h->upd), reuse_n) << "checkpoint reopen: a bundle row is unreachable";
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// canonical prebind recovery integrity (B-2C-02): a crafted WAL with kind=7 binds
// and a kind=8 tx_publish but NO row_patch (no final page for the bound PID) must
// poison on reopen -- recovery never promotes a binding whose page is missing.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, CanonicalBindWithoutRowPatchPoisonsOnReopen) {
  const auto dir = scratch_dir("poison_no_rowpatch");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/83);
  uint64_t seg_uid = 0;
  uint64_t gen = 0;
  uint64_t old_hwm = 0;
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    const auto v = waltest::make_data(1, kDim, 1);
    const uint64_t l = 5;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);  // activate v3 pid
    s.upd->checkpoint();
    seg_uid = s.upd->segment_uid();
    gen = s.upd->generation();
    old_hwm = s.upd->num_points();
  }
  // Append a canonical bundle (kind=7 append bind + kind=8) with no row_patch.
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  {
    alaya::wal::WalFile wal(wal_path);  // scans the flip frame, then appends
    const uint64_t txid = 2;
    wal.append(kSegmentOpRecordType, 0, 1, txid,
               encode_label_bind(seg_uid, gen, txid, /*row_op=*/0,
                                 static_cast<uint32_t>(old_hwm), /*gen=*/0, /*label=*/6),
               alaya::wal::WalFile::Sync::buffered);
    wal.append(kSegmentOpRecordType, 0, 2, txid,
               encode_tx_publish(seg_uid, gen, txid, /*new_hwm=*/old_hwm + 1,
                                 /*binding_count=*/1, /*applied=*/1),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW(
      { ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); },
      std::exception)
      << "a canonical bundle with no final page for its bound PID must poison";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// leg-7 fail-closed replay family: BLOCKER-3 crafted flips (rejected BEFORE any pwrite,
// index byte-for-byte unchanged) and NEW-BLOCKER-1 classifier-downgrade poison.
// ---------------------------------------------------------------------------
std::vector<char> read_all_bytes(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// The active on-disk base superblock (highest-generation structurally-valid A/B copy).
QGSuperblockV2 read_active_superblock(const std::string &index_path) {
  std::ifstream in(index_path, std::ios::binary);
  std::vector<char> header(2 * kQGSuperblockSize);
  in.read(header.data(), static_cast<std::streamsize>(header.size()));
  QGSuperblockV2 sb{};
  EXPECT_GE(select_qg_superblock(header.data(), sb), 0);
  return sb;
}

// Build a durable pid-activated v3 base (1-row activation bundle + checkpoint) and return its
// uid + active slot; the caller reads the persisted superblock separately.
struct ActivatedInfo {
  uint64_t seg_uid = 0;
  int active_slot = 0;
};
ActivatedInfo activate_v3_pid_base(const std::string &prefix, size_t base_n) {
  ActivatedInfo info;
  ReuseSession s(prefix, base_n, 4 * base_n, /*enable_reuse=*/true);
  const auto v = waltest::make_data(1, kDim, 7);
  const uint64_t l = 5;
  s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);  // activate v3 pid
  s.upd->checkpoint();                                   // durable v3 pid base, WAL == one flip
  EXPECT_TRUE(s.upd->pid_generation_activated());
  info.seg_uid = s.upd->segment_uid();
  info.active_slot = s.upd->active_superblock_slot();
  return info;
}

// BLOCKER-3 (leg-7): craft a superblock_flip carrying a v3 image whose defect must be caught in
// the PURE validation phase (selector) or the pre-pwrite label-slot validation. The reopen must
// poison AND leave the .index byte-for-byte unchanged (the flip is rejected before write_superblock).
template <typename Corrupt>
void assert_crafted_flip_fails_closed(const char *name, Corrupt corrupt) {
  const auto dir = scratch_dir(name);
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/47);
  const auto info = activate_v3_pid_base(base.prefix, kBaseN);
  const std::string index_path = base.prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  const QGSuperblockV2 base_sb = read_active_superblock(index_path);
  const auto index_before = read_all_bytes(index_path);

  QGSuperblockV2 image = base_sb;
  image.generation = base_sb.generation + 1;  // a legal next-generation flip target
  corrupt(image);                             // inject the specific defect
  image.checksum = qg_superblock_checksum(image);
  const int target_slot = info.active_slot == 0 ? 1 : 0;
  std::vector<std::byte> img_bytes(sizeof(image));
  std::memcpy(img_bytes.data(), &image, sizeof(image));
  {
    alaya::wal::WalFile wal(wal_path);  // scans the base flip, then appends the crafted flip
    wal.append(kSegmentOpRecordType, 0, 1, 0,
               encode_superblock_flip(info.seg_uid, image.generation,
                                      static_cast<uint8_t>(target_slot),
                                      std::span<const std::byte>(img_bytes.data(), img_bytes.size())),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW({ ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); },
               std::exception)
      << name << ": crafted flip must poison on replay";
  EXPECT_EQ(read_all_bytes(index_path), index_before)
      << name << ": index must be byte-for-byte unchanged (flip rejected before any pwrite)";
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterReuse, CraftedFlipMaintenanceGenZeroFailsClosedBeforeWrite) {
  // maintenance is required for every v3, so a maint-activation-gen of 0 is an impossible v3.
  // The selector (qg_superblock_supported) now rejects it in the pure validation phase, before
  // replay_flip's write_superblock.
  assert_crafted_flip_fails_closed("flip_maint_gen_zero", [](QGSuperblockV2 &image) {
    const uint64_t zero = 0;
    std::memcpy(image.reserved.data() + kWal2cReservedOffset + 24, &zero, 8);
  });
}

TEST(QgUpdaterReuse, CraftedFlipActivationGenFutureFailsClosedBeforeWrite) {
  // A pid-reuse activation generation NEWER than the image's own generation is forged; the
  // selector's (0, sb.generation] bound rejects it before any pwrite.
  assert_crafted_flip_fails_closed("flip_activation_future", [](QGSuperblockV2 &image) {
    const uint64_t future = image.generation + 1;  // > image.generation
    std::memcpy(image.reserved.data() + kWal2cReservedOffset + 32, &future, 8);
  });
}

TEST(QgUpdaterReuse, CraftedFlipBadLabelSlotFailsClosedBeforeWrite) {
  // A structurally/activation-valid image whose label tuple references a slot file that cannot
  // exist (count*16 != file size) must be rejected by replay_flip's pre-pwrite
  // validate_flip_label_state, not by adopt_label_state AFTER the superblock is installed.
  assert_crafted_flip_fails_closed("flip_bad_label_slot", [](QGSuperblockV2 &image) {
    const uint64_t slot = 0;
    const uint64_t gen = 1;
    const uint64_t count = 99;                 // no real slot file has 99*16 bytes here
    const uint64_t checksum = 0xDEADBEEFULL;
    auto *b = image.reserved.data() + 8;       // label tuple: slot@0 gen@8 count@16 checksum@24
    std::memcpy(b + 0, &slot, 8);
    std::memcpy(b + 8, &gen, 8);
    std::memcpy(b + 16, &count, 8);
    std::memcpy(b + 24, &checksum, 8);
  });
}

// BLOCKER-3 (leg-8, r2 section 1): the write-before validate_flip_transition family. Each crafted
// flip is CRC-legal and passes the selector + label-slot validation, yet would corrupt the index (or
// only fail closed on the NEXT open) if replay did not validate the full transition BEFORE its
// pwrite/ftruncate. All must poison on replay AND leave the .index byte-for-byte unchanged.
TEST(QgUpdaterReuse, CraftedFlipGeometryMismatchFailsClosedBeforeWrite) {
  // An immutable-geometry drift (dimension + 1) would install a higher-generation base whose next
  // open fails the load_v2_state identity check -- validate_flip_transition rejects it up front.
  assert_crafted_flip_fails_closed("flip_geometry",
                                   [](QGSuperblockV2 &image) { image.dimension += 1; });
}

TEST(QgUpdaterReuse, CraftedFlipShortFileSizeFailsClosedBeforeWrite) {
  // file_size == kSectorLen is the truncation bomb: after the superblock install, replay's
  // ftruncate(image.file_size) would delete every data page. The exact-length check rejects it.
  assert_crafted_flip_fails_closed("flip_short_file",
                                   [](QGSuperblockV2 &image) { image.file_size = kSectorLen; });
}

TEST(QgUpdaterReuse, CraftedFlipTargetsActiveSlotFailsClosedBeforeWrite) {
  // A flip targeting the ACTIVE A/B slot could, on a torn 512-byte pwrite, leave only the G-1 copy
  // while the durable WAL still demands G+1 -- the next replay would face a two-generation jump.
  // validate_flip_transition requires the inactive slot (the image is otherwise valid, so this
  // isolates the target-slot check; assert_crafted_flip_fails_closed hardcodes the inactive slot).
  const auto dir = scratch_dir("flip_active_slot");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/46);
  const auto info = activate_v3_pid_base(base.prefix, kBaseN);
  const std::string index_path = base.prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  const QGSuperblockV2 base_sb = read_active_superblock(index_path);
  const auto index_before = read_all_bytes(index_path);

  QGSuperblockV2 image = base_sb;
  image.generation = base_sb.generation + 1;
  image.checksum = qg_superblock_checksum(image);
  std::vector<std::byte> img_bytes(sizeof(image));
  std::memcpy(img_bytes.data(), &image, sizeof(image));
  {
    alaya::wal::WalFile wal(wal_path);
    wal.append(kSegmentOpRecordType, 0, 1, 0,
               encode_superblock_flip(info.seg_uid, image.generation,
                                      static_cast<uint8_t>(info.active_slot),  // ACTIVE slot: illegal
                                      std::span<const std::byte>(img_bytes.data(), img_bytes.size())),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW({ ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); },
               std::exception)
      << "a flip targeting the active A/B slot must poison on replay";
  EXPECT_EQ(read_all_bytes(index_path), index_before)
      << "the index must be byte-for-byte unchanged (flip rejected before any pwrite)";
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterReuse, ActivatedBaseLegacyCrossGenNewTxidPoisonsIndexUnchanged) {
  // NEW-BLOCKER-1 (leg-7): on a pid-ACTIVATED base, a legacy-lane label transaction (segment
  // generation < activation, so is_canonical_generation() routes it to the legacy path) that
  // carries a NEW txid (> the base's committed txid) is a cross-generation classifier downgrade.
  // Replay must poison at the kind=7 -- before any row_patch apply -- with the index unchanged.
  const auto dir = scratch_dir("newb1_downgrade");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/48);
  uint64_t seg_uid = 0;
  uint64_t activation_gen = 0;
  uint64_t base_txid = 0;
  uint64_t old_hwm = 0;
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    const auto v = waltest::make_data(1, kDim, 1);
    const uint64_t l = 5;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);  // activate v3 pid
    s.upd->checkpoint();
    ASSERT_TRUE(s.upd->pid_generation_activated());
    seg_uid = s.upd->segment_uid();
    activation_gen = s.upd->pid_reuse_activation_gen();
    base_txid = s.upd->last_committed_txid();
    old_hwm = s.upd->num_points();
  }
  ASSERT_GT(activation_gen, 0U);
  const std::string index_path = base.prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  const auto index_before = read_all_bytes(index_path);
  const uint64_t forged_gen = activation_gen - 1;  // < activation -> legacy lane
  const uint64_t forged_txid = base_txid + 1;      // > base committed -> non-absorbed
  {
    alaya::wal::WalFile wal(wal_path);  // scans the base flip, then appends the forged bundle
    wal.append(kSegmentOpRecordType, 0, 1, forged_txid,
               encode_label_bind(seg_uid, forged_gen, forged_txid, /*row_op=*/0,
                                 static_cast<uint32_t>(old_hwm), /*gen=*/0, /*label=*/6),
               alaya::wal::WalFile::Sync::buffered);
    wal.append(kSegmentOpRecordType, 0, 2, forged_txid,
               encode_tx_publish(seg_uid, forged_gen, forged_txid, /*new_hwm=*/old_hwm + 1,
                                 /*binding_count=*/1, /*applied=*/1),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW({ ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); },
               std::exception)
      << "a post-activation legacy cross-generation new-txid transaction must poison";
  EXPECT_EQ(read_all_bytes(index_path), index_before)
      << "the index must be byte-for-byte unchanged (poison at kind=7, before any row_patch apply)";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// garden() stays gated under reuse (it is still not a WAL maintenance transaction).
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, GardenStillThrowsUnderReuse) {
  const auto dir = scratch_dir("garden");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/71);
  ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
  EXPECT_THROW(s.upd->garden(1, GardenParams{}), std::logic_error);
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

// The full recovery-relevant fingerprint (routing entry, medoid set, per-PID trailer +
// hidden bit + row bytes, walked free chain, label bindings). Clean and replayed states
// with the SAME fingerprint are identical.
std::string full_fp(QGUpdater &upd) {
  std::string fp;
  auto put = [&](uint64_t v) { fp.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
  const auto n = static_cast<PID>(upd.num_points());
  put(upd.num_points());
  put(upd.live_count());
  put(upd.free_count());
  put(static_cast<uint64_t>(upd.entry_point()));
  put(upd.medoids().size());
  for (PID m : upd.medoids()) put(static_cast<uint64_t>(m));
  for (PID id = 0; id < n; ++id) {
    const auto tr = upd.trailer(id);
    put(tr.flags);
    put(tr.valid_degree);
    put(upd.row_hidden(id) ? 1U : 0U);
    const auto row = upd.debug_read_row(id);
    fp.append(row.data(), row.size());
  }
  PID cur = upd.free_list_head();
  size_t guard = 0;
  while (cur != kPidMax && guard <= upd.num_points() + 1) {
    put(cur);
    const auto row = upd.debug_read_row(cur);
    uint64_t nx = 0;
    std::memcpy(&nx, row.data(), sizeof(nx));
    cur = static_cast<PID>(nx);
    ++guard;
  }
  put(cur);
  const auto snap = upd.label_snapshot();
  if (snap != nullptr) {
    for (const auto &[pid, binding] : snap->bindings) {
      put(pid);
      put(binding.pid_generation);
      put(binding.label);
    }
  }
  return fp;
}

std::vector<PID> walk_free_chain(QGUpdater &upd) {
  std::vector<PID> chain;
  PID cur = upd.free_list_head();
  size_t guard = 0;
  while (cur != kPidMax && guard <= upd.num_points() + 1) {
    chain.push_back(cur);
    const auto row = upd.debug_read_row(cur);
    uint64_t nx = 0;
    std::memcpy(&nx, row.data(), sizeof(nx));
    cur = static_cast<PID>(nx);
    ++guard;
  }
  return chain;
}

// ---------------------------------------------------------------------------
// BLOCKER-1 (all-append case) + MAJOR-2 (medoid): delete-all -> all-APPEND (no reclaim)
// must converge the routing entry + medoid set + full state clean vs a WAL replay reopen
// AND a checkpoint reopen. The old repair scanned committed_ == old_hwm and left the entry
// in the dead component (clean query empty) while recovery relocated over the new watermark.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, DeleteAllThenAllAppendConvergesCleanVsReplay) {
  const auto dir = scratch_dir("del_all_append");
  std::filesystem::remove_all(dir);
  const size_t small_n = 24;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/91);
  const size_t append_n = 7;
  const auto vecs = waltest::make_data(append_n, kDim, /*seed=*/48);
  std::vector<uint64_t> labels(append_n);
  for (size_t i = 0; i < append_n; ++i) labels[i] = 51000 + i;

  std::string clean_fp;
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    for (size_t i = 0; i < small_n; ++i) s.upd->tombstone(static_cast<PID>(i));  // delete all
    EXPECT_EQ(s.upd->live_count(), 0U);
    // all-APPEND (nothing reclaimed): the bundle activates + appends new PIDs.
    const auto range = s.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), append_n);
    EXPECT_EQ(range.first, static_cast<PID>(small_n));
    EXPECT_EQ(range.second, static_cast<PID>(small_n + append_n));
    // Every appended row is searchable in-process (entry relocated over the NEW watermark).
    for (size_t i = 0; i < append_n; ++i) {
      EXPECT_TRUE(searchable(*s.upd, vecs.data() + i * kDim, static_cast<PID>(small_n + i)))
          << "clean: appended row " << i << " unreachable (dead entry not relocated)";
    }
    clean_fp = full_fp(*s.upd);
    // Do NOT checkpoint: the next reopen is a pure WAL replay.
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(full_fp(*s.upd), clean_fp) << "WAL replay diverged from the clean state (entry/medoid)";
    for (size_t i = 0; i < append_n; ++i) {
      EXPECT_TRUE(searchable(*s.upd, vecs.data() + i * kDim, static_cast<PID>(small_n + i)))
          << "replay: appended row " << i << " unreachable";
    }
    s.upd->checkpoint();
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(full_fp(*s.upd), clean_fp) << "checkpoint reopen diverged from the clean state";
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// MAJOR-2 (medoid) + BLOCKER-1 (entry): a mixed reuse bundle must converge the FULL state
// (entry + medoid set + bindings + free chain) clean vs WAL replay vs checkpoint reopen.
// A reused base row whose PID was a medoid drops from the medoid set on BOTH sides.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, MixedReuseFullStateConvergesCleanVsReplay) {
  const auto dir = scratch_dir("mixed_converge");
  std::filesystem::remove_all(dir);
  const size_t small_n = 40;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/57);
  const size_t bundle_n = 5;  // 3 reused + 2 append
  const auto vecs = waltest::make_data(bundle_n, kDim, /*seed=*/66);
  std::vector<uint64_t> labels(bundle_n);
  for (size_t i = 0; i < bundle_n; ++i) labels[i] = 44000 + i;

  std::string clean_fp;
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    free_base_rows(*s.upd, {1, 4, 12});  // free 3 base rows (incl. possible medoid PIDs)
    s.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), bundle_n);
    clean_fp = full_fp(*s.upd);
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(full_fp(*s.upd), clean_fp) << "WAL replay diverged (medoid/entry/free-chain)";
    s.upd->checkpoint();
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(full_fp(*s.upd), clean_fp) << "checkpoint reopen diverged";
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// MAJOR-3: two reclaim rounds (round 1 frees LOW PIDs, round 2 frees HIGH PIDs) must leave
// a GLOBALLY ascending free chain -- byte-identical to the recovery rebuild. The old code
// prepended the new set onto the old chain (high -> low), so the runtime order (10,11,3,4)
// contradicted the reopen order (3,4,10,11) and the next bundle reused a different PID.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, TwoRoundReclaimFreeChainIsAscendingAndMatchesReopen) {
  const auto dir = scratch_dir("free_chain");
  std::filesystem::remove_all(dir);
  const size_t small_n = 32;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/61);

  std::vector<PID> clean_chain;
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    // Activate first so both rounds run on a v3 pid base.
    const auto v = waltest::make_data(1, kDim, 5);
    const uint64_t l = 1;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);
    free_base_rows(*s.upd, {3, 4});    // round 1: LOW PIDs
    free_base_rows(*s.upd, {10, 11});  // round 2: HIGH PIDs (merged onto the round-1 chain)
    clean_chain = walk_free_chain(*s.upd);
    ASSERT_EQ(clean_chain.size(), 4U);
    // Globally ascending: head = smallest, strictly increasing.
    EXPECT_TRUE(std::is_sorted(clean_chain.begin(), clean_chain.end()))
        << "runtime free chain is not globally ascending after a second reclaim round";
    EXPECT_EQ(clean_chain.front(), static_cast<PID>(3));
    s.upd->checkpoint();
  }
  {
    ReuseSession s(base.prefix, small_n, 4 * small_n, /*enable_reuse=*/true);
    EXPECT_EQ(walk_free_chain(*s.upd), clean_chain)
        << "reopened free chain differs from the runtime chain (M3)";
    // The next reuse pops the same (smallest) PID clean vs after reopen.
    const auto v = waltest::make_data(1, kDim, 6);
    const uint64_t l = 2;
    s.upd->commit_physical_bundle(2, 2, v.data(), &l, 1);
    bool reused_three = false;
    for (PID p : s.upd->search(v.data(), 5, 64)) {
      if (label_of(*s.upd, p) == 2) reused_three = (p == static_cast<PID>(3));
    }
    EXPECT_TRUE(reused_three) << "the next reuse must pop the canonical smallest free PID (3)";
  }
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// MAJOR-1: a saturated bundle (N > degree) must keep the FINAL directed spine cycle so
// every bundle row reaches every other via graph edges. Under Backlink::kNone the spine is
// the ONLY reverse connectivity. Verify (a) each row's neighbor list holds the next cycle
// edge (the protected edge survived a forced eviction) and (b) BFS from the entry reaches all.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, SaturatedBundleSpineCycleReachesEveryRow) {
  const auto dir = scratch_dir("spine");
  std::filesystem::remove_all(dir);
  const size_t small_n = 8;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/73);
  const size_t append_n = kDeg + 6;  // N > degree: later rows saturate, forcing eviction
  const auto vecs = waltest::make_data(append_n, kDim, /*seed=*/74);
  std::vector<uint64_t> labels(append_n);
  for (size_t i = 0; i < append_n; ++i) labels[i] = 31000 + i;

  struct Holder {
    QuantizedGraph qg;
    std::unique_ptr<QGUpdater> upd;
    Holder(const std::string &prefix, size_t base_n, size_t max_points) : qg(base_n, kDeg, kDim, kDim) {
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.enable_pid_reuse = true;
      params.ef_insert = 64;
      params.max_points = max_points;
      params.backlink_mode = UpdateParams::Backlink::kNone;  // spine is the ONLY reverse edge
      upd = std::make_unique<QGUpdater>(qg, params);
    }
  };
  Holder h(base.prefix, small_n, 4 * (small_n + append_n));
  for (size_t i = 0; i < small_n; ++i) h.upd->tombstone(static_cast<PID>(i));  // delete all base
  const auto range = h.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), append_n);
  const PID first = static_cast<PID>(range.first);
  ASSERT_EQ(range.second - range.first, static_cast<PID>(append_n));

  // (a) the directed cycle edge rows[i] -> rows[(i+1)%n] survived on every row.
  for (size_t i = 0; i < append_n; ++i) {
    const PID from = first + static_cast<PID>(i);
    const PID want = first + static_cast<PID>((i + 1) % append_n);
    const auto nbrs = h.upd->debug_row_neighbors(from);
    EXPECT_NE(std::find(nbrs.begin(), nbrs.end(), want), nbrs.end())
        << "row " << i << " lost its protected spine edge to " << (i + 1) % append_n;
  }
  // (b) BFS from the entry reaches every bundle row over the committed adjacency.
  std::unordered_set<PID> seen;
  std::queue<PID> q;
  q.push(h.upd->entry_point());
  seen.insert(h.upd->entry_point());
  while (!q.empty()) {
    const PID u = q.front();
    q.pop();
    for (PID v : h.upd->debug_row_neighbors(u)) {
      if (v < static_cast<PID>(h.upd->num_points()) && seen.insert(v).second) q.push(v);
    }
  }
  size_t reached = 0;
  for (size_t i = 0; i < append_n; ++i) reached += seen.count(first + static_cast<PID>(i));
  EXPECT_EQ(reached, append_n) << "BFS from the entry did not reach every saturated bundle row";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// B-2C-02 poison 3: a canonical bundle whose kind=8 binds an appended PID but whose FINAL
// page after-image still carries a FREE trailer (row never made live) must poison on reopen.
// A whole page with the correct geometry is written, differing from poison 1 (no page at all).
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, CanonicalFinalTrailerFreePoisonsOnReopen) {
  const auto dir = scratch_dir("poison_free_trailer");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/85);
  uint64_t seg_uid = 0;
  uint64_t gen = 0;
  uint64_t old_hwm = 0;
  size_t page_size = 0;
  size_t npp = 0;
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    const auto v = waltest::make_data(1, kDim, 1);
    const uint64_t l = 5;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);  // activate v3 pid
    s.upd->checkpoint();
    seg_uid = s.upd->segment_uid();
    gen = s.upd->generation();
    old_hwm = s.upd->num_points();
    page_size = s.upd->debug_page_size();
    npp = s.upd->debug_npp();
  }
  // One appended bind for PID old_hwm, plus a whole-page final row_patch whose slot trailer is
  // left FREE (the "not made live" bug). The append algebra passes; the B-2C-02 final-live
  // trailer check must reject it.
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  {
    alaya::wal::WalFile wal(wal_path);
    const uint64_t txid = 2;
    wal.append(kSegmentOpRecordType, 0, 1, txid,
               encode_label_bind(seg_uid, gen, txid, /*row_op=*/0, static_cast<uint32_t>(old_hwm),
                                 /*gen=*/0, /*label=*/6),
               alaya::wal::WalFile::Sync::buffered);
    const size_t page = old_hwm / npp;
    const size_t slot = old_hwm % npp;
    std::vector<char> bytes(page_size, 0);
    QGRowTrailer tr{};
    tr.valid_degree = 0;
    tr.flags = static_cast<uint16_t>(kQGRowTombstone | kQGRowFree);  // NOT live: the injected bug
    qg_write_page_trailer(bytes.data(), page_size, npp, slot, tr);
    wal.append(kSegmentOpRecordType, 0, 2, 0,
               encode_row_patch(seg_uid, gen, /*first_pid=*/page * npp,
                                /*offset=*/512 + page * page_size,
                                std::span<const std::byte>(
                                    reinterpret_cast<const std::byte *>(bytes.data()), page_size)),
               alaya::wal::WalFile::Sync::buffered);
    wal.append(kSegmentOpRecordType, 0, 3, txid,
               encode_tx_publish(seg_uid, gen, txid, /*new_hwm=*/old_hwm + 1,
                                 /*binding_count=*/1, /*applied=*/1),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW(
      { ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); }, std::exception)
      << "a bound PID whose final page trailer is FREE must poison (B-2C-02)";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// BLOCKER-5: a NEW (non-absorbed) canonical bundle written at a generation OTHER than the
// replay cursor (here base_generation + 1) must poison -- a CRC-legal cross-generation
// bundle can no longer be applied at the wrong generation.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, CrossGenerationCanonicalBundlePoisonsOnReopen) {
  const auto dir = scratch_dir("poison_xgen");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/88);
  uint64_t seg_uid = 0;
  uint64_t base_gen = 0;
  uint64_t old_hwm = 0;
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    const auto v = waltest::make_data(1, kDim, 1);
    const uint64_t l = 5;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);  // activate v3 pid
    s.upd->checkpoint();
    seg_uid = s.upd->segment_uid();
    base_gen = s.upd->generation();
    old_hwm = s.upd->num_points();
  }
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  {
    alaya::wal::WalFile wal(wal_path);
    const uint64_t txid = 2;
    const uint64_t wrong_gen = base_gen + 1;  // a generation the base was never checkpointed at
    wal.append(kSegmentOpRecordType, 0, 1, txid,
               encode_label_bind(seg_uid, wrong_gen, txid, 0, static_cast<uint32_t>(old_hwm),
                                 /*gen=*/0, /*label=*/6),
               alaya::wal::WalFile::Sync::buffered);
    wal.append(kSegmentOpRecordType, 0, 2, txid,
               encode_tx_publish(seg_uid, wrong_gen, txid, old_hwm + 1, 1, 1),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW(
      { ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); }, std::exception)
      << "a canonical bundle at a non-cursor generation must poison (BLOCKER-5)";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// BLOCKER-5: a canonical bind carrying the reserved sentinel PID (kPidMax) is range-rejected
// BEFORE the narrowing cast, so it can never slip past the reuse/append bound checks.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, CanonicalSentinelPidPoisonsOnReopen) {
  const auto dir = scratch_dir("poison_pidmax");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/89);
  uint64_t seg_uid = 0;
  uint64_t base_gen = 0;
  {
    ReuseSession s(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true);
    const auto v = waltest::make_data(1, kDim, 1);
    const uint64_t l = 5;
    s.upd->commit_physical_bundle(1, 1, v.data(), &l, 1);
    s.upd->checkpoint();
    seg_uid = s.upd->segment_uid();
    base_gen = s.upd->generation();
  }
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  {
    alaya::wal::WalFile wal(wal_path);
    const uint64_t txid = 2;
    wal.append(kSegmentOpRecordType, 0, 1, txid,
               encode_label_bind(seg_uid, base_gen, txid, 0, static_cast<uint32_t>(kPidMax),
                                 /*gen=*/0, /*label=*/6),
               alaya::wal::WalFile::Sync::fsync);
  }
  EXPECT_THROW(
      { ReuseSession s2(base.prefix, kBaseN, 4 * kBaseN, /*enable_reuse=*/true); }, std::exception)
      << "a canonical bind with the sentinel PID must poison (BLOCKER-5 range check)";
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// BLOCKER-2: any exception AFTER the reservation is published must fail the handle closed,
// even a NON-std::exception thrown by a failpoint hook (throw 7) and an OOM in the reason
// path. The atomic poison latch -- not the reason string -- is authoritative.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, ReservationNonStdExceptionPoisonsHandle) {
  const auto dir = scratch_dir("poison_reserve");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/95);
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.enable_pid_reuse = true;
  params.ef_insert = 64;
  params.max_points = 4 * kBaseN;
  // Throw a NON-std::exception right after the reservation is published (R0).
  params.failpoint_hook = [](SegmentOpFailPoint fp) {
    if (fp == SegmentOpFailPoint::after_reuse_reserve_before_binds) throw 7;  // NOLINT
  };
  QGUpdater upd(qg, params);
  const auto v = waltest::make_data(1, kDim, 1);
  const uint64_t l = 9;
  bool threw = false;
  try {
    upd.commit_physical_bundle(1, 1, v.data(), &l, 1);
  } catch (...) {  // the int 7 escapes as the original exception
    threw = true;
  }
  ASSERT_TRUE(threw);
  // The handle is now latched: every subsequent write op fails closed with std::exception.
  EXPECT_THROW(upd.commit_physical_bundle(2, 2, v.data(), &l, 1), std::exception)
      << "a non-std::exception past the reservation must still poison the handle";
  EXPECT_THROW(upd.tombstone(0), std::exception);
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// BLOCKER-4: once the checkpoint flip is durable, an in-process exception (here a post-flip
// failpoint throw) must poison the handle so it can never write a second, differing flip.
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, DurableFlipThenExceptionPoisonsHandle) {
  const auto dir = scratch_dir("poison_flip");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, /*seed=*/96);
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.enable_pid_reuse = true;
  params.ef_insert = 64;
  params.max_points = 4 * kBaseN;
  // Throw AFTER the flip frame is appended + fsynced (durable), before the superblock write --
  // but ONLY once armed, so the constructor's fresh-enable checkpoint flip is unaffected.
  auto armed = std::make_shared<bool>(false);
  params.failpoint_hook = [armed](SegmentOpFailPoint fp) {
    if (*armed && fp == SegmentOpFailPoint::after_flip_append_before_superblock_write) {
      throw std::runtime_error("post-flip failpoint");
    }
  };
  QGUpdater upd(qg, params);  // fresh-enable checkpoint runs here (not yet armed)
  const auto v = waltest::make_data(1, kDim, 1);
  const uint64_t l = 9;
  *armed = true;
  // The checkpoint flip becomes durable, then the failpoint throws -> the handle must poison
  // (BLOCKER-4) so it can never write a second, differing G+1 flip.
  EXPECT_THROW(upd.checkpoint(), std::exception);
  // A retried checkpoint / bundle must fail closed.
  EXPECT_THROW(upd.checkpoint(), std::exception)
      << "a durable-flip exception must latch the handle (no second flip)";
  EXPECT_THROW(upd.commit_physical_bundle(1, 1, v.data(), &l, 1), std::exception);
  std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// BLOCKER-6: a no-reuse (enable_pid_reuse=false) FullPrune bundle stays on the legacy 2A
// path with the writer-visible floor INERT (bundle_ctx_ == null => full_reverse_recompute
// uses committed_ exactly, byte-for-byte the pre-2C behavior). Verify the segment never
// activates (stays v2) and the produced graph is fully deterministic clean vs a replay/
// checkpoint reopen (the floor leaking would change the FullPrune candidate set + payload).
// ---------------------------------------------------------------------------
TEST(QgUpdaterReuse, LegacyFullPruneBundleStaysV2AndByteStable) {
  const auto dir = scratch_dir("legacy_fullprune");
  std::filesystem::remove_all(dir);
  const size_t small_n = 40;
  auto base = WalTinyIndex::build(dir, small_n, /*seed=*/33);
  const size_t append_n = 6;
  const auto vecs = waltest::make_data(append_n, kDim, /*seed=*/34);
  std::vector<uint64_t> labels(append_n);
  for (size_t i = 0; i < append_n; ++i) labels[i] = 22000 + i;

  struct Holder {
    QuantizedGraph qg;
    std::unique_ptr<QGUpdater> upd;
    Holder(const std::string &prefix, size_t base_n) : qg(base_n, kDeg, kDim, kDim) {
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.enable_pid_reuse = false;  // legacy 2A path
      params.ef_insert = 64;
      params.max_points = 4 * base_n;
      params.backlink_mode = UpdateParams::Backlink::kFullPrune;  // the floor-sensitive mode
      upd = std::make_unique<QGUpdater>(qg, params);
    }
  };
  std::string clean_fp;
  {
    Holder h(base.prefix, small_n);
    h.upd->commit_physical_bundle(1, 1, vecs.data(), labels.data(), append_n);
    EXPECT_EQ(h.upd->superblock_format_version(), kQGFormatVersion)
        << "a no-reuse FullPrune bundle must NOT activate a v3 base";
    clean_fp = full_fp(*h.upd);
  }
  {
    Holder h(base.prefix, small_n);  // WAL replay
    EXPECT_EQ(full_fp(*h.upd), clean_fp) << "legacy FullPrune replay diverged (floor leaked?)";
    h.upd->checkpoint();
  }
  {
    Holder h(base.prefix, small_n);  // checkpoint reopen
    EXPECT_EQ(full_fp(*h.upd), clean_fp) << "legacy FullPrune checkpoint reopen diverged";
    EXPECT_EQ(h.upd->superblock_format_version(), kQGFormatVersion);
  }
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::laser
