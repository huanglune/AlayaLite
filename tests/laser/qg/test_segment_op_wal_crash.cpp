// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// The G1 crash matrix for the QGUpdater op-WAL, in two layers:
//
//  1. SIGKILL / failpoint layer: a real forked child self-SIGKILLs at each
//     labelled lifecycle cut; the parent reopens (recovery), asserts a
//     prefix-reachable state, and reopens again for double-replay stability.
//
//  2. Persistence-model (power-loss) layer: a SegmentIoObserver snapshots the
//     forced content of the index and the op-WAL at every fsync. At the crash
//     point the harness materializes the possible power-loss disk states
//     (retain/drop the unforced tail of each stream independently) and recovers
//     each — SIGKILL alone cannot drop the kernel page cache, so it cannot
//     expose a cross-fd ordering hole. Assertions compare the full recovered
//     state tuple + search observables and require byte-stable double replay.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "qg_wal_test_support.hpp"

namespace alaya::laser {
namespace {

using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 128;
constexpr size_t kInsert = 12;

std::filesystem::path battery_root(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("qg_wal_crash_" + name + "_" + std::to_string(::getpid()));
}

void copy_tree(const std::filesystem::path &from, const std::filesystem::path &to) {
  std::filesystem::remove_all(to);
  std::filesystem::copy(from, to,
                        std::filesystem::copy_options::recursive |
                            std::filesystem::copy_options::overwrite_existing);
}

std::vector<char> read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Stamp the lineage uid + create the .opwal once, with no failpoint/observer, so
// the fresh-enable checkpoint is setup rather than the operation under test.
void prepare_wal_base(const std::string &prefix, size_t max_points) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.max_points = max_points;
  QGUpdater upd(qg, params);
}

void write_file(const std::filesystem::path &path, const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// A recovery open + the full state tuple used for prefix / stability assertions.
struct Recovered {
  bool poisoned = false;
  size_t num_points = 0;
  uint64_t live_count = 0;
  uint64_t free_count = 0;
  uint64_t last_txid = 0;
  uint64_t applied_op = 0;
  std::optional<uint64_t> appended_label;  // label of pid == kBaseN, when committed
  std::vector<char> index_bytes;           // whole .index (both superblocks + all pages)
  std::vector<char> wal_bytes;             // whole .opwal
  uint64_t epoch = 0;                       // wal-2c BLOCKER-2: last committed maint epoch
  std::string graph_fp;                     // wal-2c BLOCKER-2: full S_old/S_new graph fingerprint
};

// wal-2c BLOCKER-2: a generation-independent fingerprint of the recovered GRAPH -- per-pid
// trailer (flags+valid_degree), the page bytes of every LIVE row, the walked canonical free
// chain, the maintenance epoch, and a routing probe. Two recovered states are the SAME graph
// iff their fingerprints match. This lifts the consolidate crash matrix from "counts + within-
// case byte stability" to "the recovery landed on exactly S_old or S_new" (page content +
// hidden/free PID sets + free chain + routing + epoch, per codex BLOCKER-2).
std::string capture_graph_fp(QGUpdater &upd) {
  std::string fp;
  auto put = [&](uint64_t v) { fp.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
  const auto n = static_cast<PID>(upd.num_points());
  put(upd.num_points());
  put(upd.live_count());
  put(upd.free_count());
  put(upd.last_completed_consolidate_epoch());
  for (PID id = 0; id < n; ++id) {
    const auto tr = upd.trailer(id);
    put(tr.flags);
    put(tr.valid_degree);
    if ((tr.flags & (kQGRowTombstone | kQGRowFree)) == 0) {
      const auto row = upd.debug_read_row(id);
      fp.append(row.data(), row.size());  // live-row page content
    }
  }
  PID cur = upd.free_list_head();
  size_t guard = 0;
  while (cur != kPidMax && guard <= upd.num_points()) {
    put(cur);
    const auto row = upd.debug_read_row(cur);
    uint64_t nx = 0;
    std::memcpy(&nx, row.data(), sizeof(nx));
    cur = static_cast<PID>(nx);
    ++guard;
  }
  put(cur);  // terminal (kPidMax on a well-formed chain)
  const auto q = waltest::make_data(1, kDim, 0xC0FFEEU);
  auto hits = upd.search(q.data(), 10, 64);
  std::sort(hits.begin(), hits.end());
  for (PID h : hits) put(h);
  return fp;
}

Recovered recover(const std::string &prefix, size_t max_points) {
  Recovered r;
  const std::string index_path = prefix + waltest::index_suffix();
  try {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    QGUpdater upd(qg, params);
    r.num_points = upd.num_points();
    r.live_count = upd.live_count();
    r.free_count = upd.free_count();
    r.last_txid = upd.last_committed_txid();
    r.applied_op = upd.applied_collection_op_id();
    if (upd.num_points() > kBaseN) {
      const auto snap = upd.label_snapshot();
      if (snap != nullptr) {
        const auto *found = snap->find(static_cast<PID>(kBaseN));
        if (found != nullptr) r.appended_label = *found;
      }
    }
    r.epoch = upd.last_completed_consolidate_epoch();
    r.graph_fp = capture_graph_fp(upd);
  } catch (const std::exception &) {
    r.poisoned = true;
  }
  r.index_bytes = read_file(index_path);
  r.wal_bytes = read_file(index_path + ".opwal");
  return r;
}

// ---------------------------------------------------------------------------
// Layer 1: SIGKILL / failpoint.
// ---------------------------------------------------------------------------

struct KillCase {
  const char *name;
  SegmentOpFailPoint point;
  bool checkpoint;   // op = insert+publish+checkpoint (else insert+publish)
  bool expect_durable;  // after recovery the inserts must be committed
};

class SegmentOpWalSigkill : public ::testing::TestWithParam<KillCase> {};

TEST_P(SegmentOpWalSigkill, ReopenIsPrefixReachableAndDoubleReplayStable) {
  const auto param = GetParam();
  const auto root = battery_root(param.name);
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 4242);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);  // fresh-enable is setup, not the tested op
  const auto inserted = waltest::make_data(kInsert, kDim, 314);

  const auto case_dir = root / "case";
  copy_tree(template_dir, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      const SegmentOpFailPoint target = param.point;
      params.failpoint_hook = [target](SegmentOpFailPoint fp) {
        if (fp == target) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      QGUpdater upd(qg, params);
      for (size_t i = 0; i < kInsert; ++i) upd.allocate_and_insert(inserted.data() + i * kDim);
      upd.publish(upd.allocated_points());
      if (param.checkpoint) upd.checkpoint();
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);  // failpoint not reached (still a valid — fully durable — outcome)
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  // wal-2c BLOCKER-2: reject a child that died from a CAUGHT exception before the cut (a
  // _exit(70) could let a stale-count recovery pass spuriously).
  ASSERT_FALSE(WIFEXITED(status) && WEXITSTATUS(status) == 70)
      << param.name << " child threw before reaching the cut";

  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned) << "recovery of " << param.name << " must not poison";
  // Prefix reachability: the committed count is one of the two log prefixes.
  EXPECT_TRUE(first.num_points == kBaseN || first.num_points == kBaseN + kInsert)
      << param.name << " num_points=" << first.num_points;
  if (param.expect_durable) {
    EXPECT_EQ(first.num_points, kBaseN + kInsert);
    EXPECT_EQ(first.live_count, kBaseN + kInsert);
  }
  // Double replay is byte + state stable.
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.num_points, second.num_points);
  EXPECT_EQ(first.live_count, second.live_count);
  EXPECT_EQ(first.index_bytes, second.index_bytes) << param.name << " index not byte-stable";
  EXPECT_EQ(first.wal_bytes, second.wal_bytes) << param.name << " wal not byte-stable";

  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    KillPoints, SegmentOpWalSigkill,
    ::testing::Values(
        KillCase{"append_before_apply", SegmentOpFailPoint::after_wal_append_before_apply, false,
                 false},
        KillCase{"before_publish_fsync", SegmentOpFailPoint::after_apply_before_publish_fsync,
                 false, false},
        KillCase{"after_publish_fsync", SegmentOpFailPoint::after_publish_fsync, false, true},
        KillCase{"flip_before_superblock",
                 SegmentOpFailPoint::after_flip_append_before_superblock_write, true, true},
        KillCase{"superblock_before_reset",
                 SegmentOpFailPoint::after_superblock_write_before_wal_reset, true, true},
        KillCase{"after_wal_reset", SegmentOpFailPoint::after_wal_reset, true, true}),
    [](const ::testing::TestParamInfo<KillCase> &info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Layer 1 (consolidate transaction, W1): SIGKILL at the maintenance-epoch cuts.
// The op is tombstone(kTomb rows) + consolidate(reclaim). The activation
// checkpoint (which makes the tombstones + the v3 base durable) runs BEFORE any
// consolidate failpoint, so live_count is always kBaseN-kTomb after recovery; the
// distinguishing field is free_count: 0 until the END fsync commits the epoch,
// then kTomb (roll-forward). No cut may poison; double replay is byte-stable.
// ---------------------------------------------------------------------------
constexpr size_t kConsolTomb = 4;

struct ConsolidateKillCase {
  const char *name;
  SegmentOpFailPoint point;
  bool expect_reclaimed;  // after recovery the free-list holds the kConsolTomb rows
  size_t cache_cap = 0;   // 0 = default (no spill); a small cap forces overlay spill (C4/C5)
};

class SegmentOpWalConsolidateSigkill : public ::testing::TestWithParam<ConsolidateKillCase> {};

// wal-2c BLOCKER-2: the two definitive graph fingerprints this matrix must land on, built
// once from the SAME base (seed 4343 + tombstone 0..kConsolTomb) as every cut. S_old drives
// the REAL activation checkpoint then aborts before the BEGIN append (identical disk state to
// a C0 SIGKILL); S_new runs the full clean consolidate to its committed END.
struct ConsolRefs {
  std::string s_old;
  std::string s_new;
};
const ConsolRefs &consol_refs() {
  static const ConsolRefs refs = [] {
    const auto root = battery_root("consol_refs");
    std::filesystem::remove_all(root);
    const auto tmpl = root / "template";
    auto base = WalTinyIndex::build(tmpl, kBaseN, 4343);
    const size_t max_points = kBaseN + kInsert + 16;
    prepare_wal_base(base.prefix, max_points);
    const auto old_dir = root / "old";
    copy_tree(tmpl, old_dir);
    {
      const std::string prefix = (old_dir / "wal_base").string();
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      params.failpoint_hook = [](SegmentOpFailPoint fp) {
        if (fp == SegmentOpFailPoint::before_consolidate_begin_append) {
          throw std::runtime_error("abort-after-activation");
        }
      };
      QGUpdater upd(qg, params);
      for (PID id = 0; id < static_cast<PID>(kConsolTomb); ++id) upd.tombstone(id);
      try {
        upd.consolidate(1, 0, /*reclaim_slots=*/true, /*bloom_consolidate=*/false);
      } catch (const std::exception &) {
      }  // handle poisoned; disk = [flip(activation)] only
    }
    const auto new_dir = root / "new";
    copy_tree(tmpl, new_dir);
    {
      const std::string prefix = (new_dir / "wal_base").string();
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      QGUpdater upd(qg, params);
      for (PID id = 0; id < static_cast<PID>(kConsolTomb); ++id) upd.tombstone(id);
      upd.consolidate(1, 0, /*reclaim_slots=*/true, /*bloom_consolidate=*/false);
    }
    ConsolRefs r;
    r.s_old = recover((old_dir / "wal_base").string(), max_points).graph_fp;
    r.s_new = recover((new_dir / "wal_base").string(), max_points).graph_fp;
    std::filesystem::remove_all(root);
    return r;
  }();
  return refs;
}

TEST_P(SegmentOpWalConsolidateSigkill, ReopenIsPrefixReachableAndDoubleReplayStable) {
  const auto param = GetParam();
  const auto root = battery_root(std::string("consol_") + param.name);
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 4343);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);

  const auto case_dir = root / "case";
  copy_tree(template_dir, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      if (param.cache_cap != 0) params.cache_cap_pages = param.cache_cap;  // force overlay spill
      const SegmentOpFailPoint target = param.point;
      params.failpoint_hook = [target](SegmentOpFailPoint fp) {
        if (fp == target) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      QGUpdater upd(qg, params);
      for (PID id = 0; id < static_cast<PID>(kConsolTomb); ++id) upd.tombstone(id);
      upd.consolidate(1, /*r_target=*/0, /*reclaim_slots=*/true, /*bloom_consolidate=*/false);
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  // wal-2c BLOCKER-2: the child must have died AT the cut (SIGKILL), not from a caught
  // exception before it -- a BUILD steal-guard poison that _exit(70)s could otherwise let a
  // stale-count recovery pass spuriously.
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << param.name << " child did not SIGKILL at the cut (status=" << status << ")";

  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned) << "recovery of " << param.name << " must not poison";
  EXPECT_EQ(first.num_points, kBaseN) << param.name << " HWM is unchanged by consolidate";
  EXPECT_EQ(first.live_count, kBaseN - kConsolTomb)
      << param.name << " tombstones durable via the activation checkpoint";
  EXPECT_EQ(first.free_count, param.expect_reclaimed ? kConsolTomb : 0U)
      << param.name << " free-list reflects the epoch commit point";
  // wal-2c BLOCKER-2: counts alone cannot distinguish a partially-repaired graph (e.g. a live
  // row's pre-END repair page installed while free flags stayed 0) from clean S_old. Require
  // the FULL recovered fingerprint (page content + PID sets + free chain + routing + epoch) to
  // equal exactly S_old or S_new.
  const auto &refs = consol_refs();
  EXPECT_TRUE(first.graph_fp == refs.s_old || first.graph_fp == refs.s_new)
      << param.name << " recovered graph matches neither reference state";
  EXPECT_EQ(first.graph_fp, param.expect_reclaimed ? refs.s_new : refs.s_old)
      << param.name << " recovered graph is not the expected "
      << (param.expect_reclaimed ? "S_new" : "S_old");
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.num_points, second.num_points);
  EXPECT_EQ(first.live_count, second.live_count);
  EXPECT_EQ(first.free_count, second.free_count);
  EXPECT_EQ(first.graph_fp, second.graph_fp) << param.name << " graph fingerprint not stable";
  EXPECT_EQ(first.index_bytes, second.index_bytes) << param.name << " index not byte-stable";
  EXPECT_EQ(first.wal_bytes, second.wal_bytes) << param.name << " wal not byte-stable";

  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    ConsolidateKillPoints, SegmentOpWalConsolidateSigkill,
    ::testing::Values(
        // C0-C7: every cut before the END *force* loses the buffered END (SIGKILL keeps
        // the OS page cache but not this process's userspace stream buffer), so recovery
        // sees an unmatched BEGIN, truncates the epoch, and converges to S_old (free 0).
        ConsolidateKillCase{"c0_before_begin_append",
                            SegmentOpFailPoint::before_consolidate_begin_append, false},
        ConsolidateKillCase{"c1_after_begin_append",
                            SegmentOpFailPoint::after_consolidate_begin_append, false},
        ConsolidateKillCase{"c2_before_begin_fsync",
                            SegmentOpFailPoint::before_consolidate_begin_fsync, false},
        ConsolidateKillCase{"c3_after_begin_fsync",
                            SegmentOpFailPoint::after_consolidate_begin_fsync, false},
        ConsolidateKillCase{"c4_overlay_before_spill",
                            SegmentOpFailPoint::after_consolidate_overlay_modify_before_spill, false,
                            /*cache_cap=*/2},
        ConsolidateKillCase{"c5_after_spill_flush",
                            SegmentOpFailPoint::after_consolidate_spill_flush, false,
                            /*cache_cap=*/2},
        ConsolidateKillCase{"c6_live_repair_before_free",
                            SegmentOpFailPoint::after_consolidate_live_repair_before_free_image,
                            false},
        ConsolidateKillCase{"before_end_append",
                            SegmentOpFailPoint::before_consolidate_end_append, false},
        ConsolidateKillCase{"c7_end_append_before_fsync",
                            SegmentOpFailPoint::after_consolidate_end_append_before_fsync, false},
        // C8-C11: the END force is the commit point; every later cut rolls forward to
        // S_new (free-list holds the kConsolTomb reclaimed rows) via pure WAL redo.
        ConsolidateKillCase{"c8_end_fsync", SegmentOpFailPoint::after_consolidate_end_fsync, true},
        ConsolidateKillCase{"c9_install_page",
                            SegmentOpFailPoint::after_consolidate_install_page, true},
        ConsolidateKillCase{"c10_install_before_publish",
                            SegmentOpFailPoint::after_consolidate_install_before_publish, true},
        ConsolidateKillCase{"c11_after_publish",
                            SegmentOpFailPoint::after_consolidate_publish, true}),
    [](const ::testing::TestParamInfo<ConsolidateKillCase> &info) { return info.param.name; });

// Layer 1 (label transaction, B-07): SIGKILL at the four new bundle/checkpoint
// cuts. A torn bundle (kill before the kind=8 fsync) loses its whole buffered
// prefix and stays uncommitted; a kill after the fsync (or at the checkpoint slot
// cut, over an already-durable bundle) recovers the bundle with its labels.
struct BundleKillCase {
  const char *name;
  SegmentOpFailPoint point;
  bool checkpoint;
  bool expect_durable;
};
class SegmentOpWalBundleSigkill : public ::testing::TestWithParam<BundleKillCase> {};

TEST_P(SegmentOpWalBundleSigkill, BundleReopenIsPrefixReachableAndDoubleReplayStable) {
  const auto param = GetParam();
  const auto root = battery_root(std::string("bundle_") + param.name);
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 5150);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const auto vecs = waltest::make_data(kInsert, kDim, 271);
  std::vector<uint64_t> labels;
  for (size_t i = 0; i < kInsert; ++i) labels.push_back(70000 + i);

  const auto case_dir = root / "case";
  copy_tree(template_dir, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      const SegmentOpFailPoint target = param.point;
      params.failpoint_hook = [target](SegmentOpFailPoint fp) {
        if (fp == target) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      QGUpdater upd(qg, params);
      (void)upd.commit_physical_bundle(7, 3, vecs.data(), labels.data(), kInsert);
      if (param.checkpoint) upd.checkpoint();
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  // wal-2c BLOCKER-2: reject a child that died from a CAUGHT exception before the cut.
  ASSERT_FALSE(WIFEXITED(status) && WEXITSTATUS(status) == 70)
      << "bundle_" << param.name << " child threw before reaching the cut";

  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned) << "recovery of bundle_" << param.name << " must not poison";
  EXPECT_TRUE(first.num_points == kBaseN || first.num_points == kBaseN + kInsert)
      << param.name << " num_points=" << first.num_points;
  if (param.expect_durable) {
    EXPECT_EQ(first.num_points, kBaseN + kInsert);
    EXPECT_EQ(first.live_count, kBaseN + kInsert);
    ASSERT_TRUE(first.appended_label.has_value());
    EXPECT_EQ(*first.appended_label, 70000U) << "bundle label recovered";
    EXPECT_EQ(first.last_txid, 7U);
  } else {
    EXPECT_EQ(first.num_points, kBaseN) << "torn bundle stays uncommitted";
    EXPECT_EQ(first.last_txid, 0U) << "torn bundle does not consume its txid";
  }
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.num_points, second.num_points);
  EXPECT_EQ(first.live_count, second.live_count);
  EXPECT_EQ(first.appended_label, second.appended_label);
  EXPECT_EQ(first.index_bytes, second.index_bytes) << param.name << " index not byte-stable";
  EXPECT_EQ(first.wal_bytes, second.wal_bytes) << param.name << " wal not byte-stable";

  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    BundleKillPoints, SegmentOpWalBundleSigkill,
    ::testing::Values(
        BundleKillCase{"after_label_bind", SegmentOpFailPoint::after_label_bind_append, false,
                       false},
        BundleKillCase{"before_tx_publish", SegmentOpFailPoint::before_tx_publish_append, false,
                       false},
        BundleKillCase{"after_tx_publish_fsync", SegmentOpFailPoint::after_tx_publish_fsync, false,
                       true},
        BundleKillCase{"slot_before_flip", SegmentOpFailPoint::label_slot_written_before_flip, true,
                       true}),
    [](const ::testing::TestParamInfo<BundleKillCase> &info) { return info.param.name; });

// ---------------------------------------------------------------------------
// Layer 2: persistence model (power loss).
// ---------------------------------------------------------------------------

// Captures the forced (fsynced) content of both files, updated on every fsync,
// so the crash point can drop the unforced tail of either stream.
struct ForcedSnapshots {
  std::filesystem::path index_path;
  std::filesystem::path wal_path;
  std::vector<char> forced_index;
  std::vector<char> forced_wal;
};

// Materialize one power-loss disk state into `dst` (a copy of the base tree) by
// overwriting the .index / .opwal with either their forced or current content.
Recovered recover_power_loss_state(const std::filesystem::path &template_dir,
                                   const std::filesystem::path &dst,
                                   const std::vector<char> &index_bytes,
                                   const std::vector<char> &wal_bytes,
                                   size_t max_points) {
  copy_tree(template_dir, dst);
  const std::string prefix = (dst / "wal_base").string();
  write_file(prefix + waltest::index_suffix(), index_bytes);
  write_file(prefix + waltest::index_suffix() + ".opwal", wal_bytes);
  return recover(prefix, max_points);
}

// Run `scenario` under an IoObserver, then materialize the four retain/drop
// combinations and assert every one is prefix-reachable, non-poisoned, and
// byte-stable under a second replay. `expected` is the committed count that a
// state which retains all forced writes must converge to.
void assert_power_loss_matrix(const std::string &name,
                              const std::function<void(QGUpdater &)> &scenario,
                              size_t expected_when_forced_retained,
                              std::optional<uint64_t> label_when_committed = std::nullopt) {
  const auto root = battery_root(name);
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 9090);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);  // fresh-enable stamp before the run
  const auto run_dir = root / "run";
  copy_tree(template_dir, run_dir);
  const std::string run_prefix = (run_dir / "wal_base").string();
  const std::string index_path = run_prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";

  ForcedSnapshots snap;
  snap.index_path = index_path;
  snap.wal_path = wal_path;
  snap.forced_index = read_file(index_path);  // durable base (post fresh-enable checkpoint)
  snap.forced_wal = read_file(wal_path);

  SegmentIoObserver observer;
  observer.on_index_fsync = [&] { snap.forced_index = read_file(index_path); };
  observer.on_wal_fsync = [&] { snap.forced_wal = read_file(wal_path); };
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(run_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    params.io_observer = &observer;
    QGUpdater upd(qg, params);
    scenario(upd);
  }
  const auto current_index = read_file(index_path);
  const auto current_wal = read_file(wal_path);

  struct State {
    const char *label;
    const std::vector<char> &index;
    const std::vector<char> &wal;
  };
  const std::vector<State> states = {
      {"retain-index retain-wal", current_index, current_wal},
      {"drop-index   retain-wal", snap.forced_index, current_wal},
      {"retain-index drop-wal", current_index, snap.forced_wal},
      {"drop-index   drop-wal", snap.forced_index, snap.forced_wal},
  };
  for (const auto &state : states) {
    SCOPED_TRACE(std::string(name) + ": " + state.label);
    const auto dst = root / "state";
    const auto first = recover_power_loss_state(template_dir, dst, state.index, state.wal,
                                                max_points);
    ASSERT_FALSE(first.poisoned) << "a power-loss state must recover, not poison";
    // Prefix reachability: every state converges to a committed prefix.
    EXPECT_TRUE(first.num_points == kBaseN || first.num_points == kBaseN + kInsert)
        << "num_points=" << first.num_points;
    // Label atomicity: when the bundle is committed its label is present too
    // (all-or-nothing); when not committed no appended label leaked.
    if (label_when_committed.has_value()) {
      if (first.num_points == kBaseN + kInsert) {
        ASSERT_TRUE(first.appended_label.has_value());
        EXPECT_EQ(*first.appended_label, *label_when_committed);
      } else {
        EXPECT_FALSE(first.appended_label.has_value());
      }
    }
    // Double replay is byte + state stable. wal-2c BLOCKER-2: the second replay must reopen the
    // FIRST recovery's OUTPUT (dst), not re-materialize the original crash image into a fresh
    // dir -- a non-idempotent second recovery would otherwise be missed.
    const auto second = recover((dst / "wal_base").string(), max_points);
    EXPECT_EQ(first.num_points, second.num_points);
    EXPECT_EQ(first.live_count, second.live_count);
    EXPECT_EQ(first.index_bytes, second.index_bytes) << "index not byte-stable on double replay";
    EXPECT_EQ(first.wal_bytes, second.wal_bytes) << "wal not byte-stable on double replay";
  }

  // The all-forced-retained state (current index + current wal) reaches the
  // fully durable prefix.
  const auto dst = root / "forced";
  const auto forced = recover_power_loss_state(template_dir, dst, current_index, current_wal,
                                               max_points);
  EXPECT_EQ(forced.num_points, expected_when_forced_retained);

  std::filesystem::remove_all(root);
}

// Published inserts survive every power-loss combination via the WAL, even when
// the unforced index pages are dropped.
TEST(SegmentOpWalPowerLoss, PublishedInsertsAreDurableAcrossUnforcedIndexLoss) {
  auto inserted = waltest::make_data(kInsert, kDim, 2718);
  assert_power_loss_matrix(
      "published",
      [inserted](QGUpdater &upd) {
        for (size_t i = 0; i < kInsert; ++i) upd.allocate_and_insert(inserted.data() + i * kDim);
        upd.publish(upd.allocated_points());
        upd.writeback(1);  // pwrite pages (unforced) after forcing the WAL
      },
      kBaseN + kInsert);
}

// Unpublished inserts are never visible after any power-loss state, even though
// writeback forced their after-images into the WAL.
TEST(SegmentOpWalPowerLoss, UnpublishedInsertsAreNeverVisible) {
  auto inserted = waltest::make_data(kInsert, kDim, 1618);
  assert_power_loss_matrix(
      "unpublished",
      [inserted](QGUpdater &upd) {
        for (size_t i = 0; i < kInsert; ++i) upd.allocate_and_insert(inserted.data() + i * kDim);
        upd.writeback(1);  // forces the row after-images, but there is no publish record
      },
      kBaseN);  // committed stays at the base — the batch was never published
}

// A checkpoint is atomic across power loss: the flip frame rolls it forward or
// the old base rolls it back, never a torn superblock.
TEST(SegmentOpWalPowerLoss, CheckpointIsAtomicAcrossPowerLoss) {
  auto inserted = waltest::make_data(kInsert, kDim, 4224);
  assert_power_loss_matrix(
      "checkpoint",
      [inserted](QGUpdater &upd) {
        for (size_t i = 0; i < kInsert; ++i) upd.allocate_and_insert(inserted.data() + i * kDim);
        upd.publish(upd.allocated_points());
        upd.checkpoint();
      },
      kBaseN + kInsert);
}

// A label bundle is atomic across power loss (B-07): the kind=8 fsync forces the
// whole bundle (row after-images + binds + publish) durable, so every retain/drop
// combination either recovers the bundle WITH its labels or not at all.
TEST(SegmentOpWalPowerLoss, BundleWithLabelsIsAtomicAcrossPowerLoss) {
  auto vecs = waltest::make_data(kInsert, kDim, 3535);
  std::vector<uint64_t> labels;
  for (size_t i = 0; i < kInsert; ++i) labels.push_back(70000 + i);
  assert_power_loss_matrix(
      "bundle_labels",
      [vecs, labels](QGUpdater &upd) {
        (void)upd.commit_physical_bundle(9, 4, vecs.data(), labels.data(), kInsert);
        upd.writeback(1);  // pwrite pages (unforced) after the kind=8 fsync
      },
      kBaseN + kInsert, /*label_when_committed=*/70000U);
}

// A consolidate maintenance transaction is atomic across power loss (W1 backfill /
// design section 6.3): the END fsync is the single commit point, so once it is
// durable recovery must roll forward to S_new over an index that installed NONE of
// the pages (the forced snapshot -- install pwrites are unforced) or ALL of them
// (the current file). The SIGKILL C9 cut covers the partial-install middle.
TEST(SegmentOpWalPowerLoss, ConsolidateRollsForwardAcrossUnforcedIndexLoss) {
  constexpr size_t kTomb = 4;
  const auto root = battery_root("consolidate_rollforward");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 7373);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const auto run_dir = root / "run";
  copy_tree(template_dir, run_dir);
  const std::string run_prefix = (run_dir / "wal_base").string();
  const std::string index_path = run_prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";

  std::vector<char> forced_index;  // last index fsync == the activation checkpoint (pre-install)
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(run_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    SegmentIoObserver observer;
    observer.on_index_fsync = [&] { forced_index = read_file(index_path); };
    params.io_observer = &observer;
    QGUpdater upd(qg, params);
    for (PID id = 0; id < static_cast<PID>(kTomb); ++id) upd.tombstone(id);
    upd.consolidate(1, /*r_target=*/0, /*reclaim_slots=*/true, /*bloom_consolidate=*/false);
  }
  const auto current_index = read_file(index_path);
  const auto current_wal = read_file(wal_path);  // [flip, begin, pages..., end] (END durable)
  ASSERT_FALSE(forced_index.empty()) << "the activation checkpoint must fsync the index once";
  EXPECT_NE(forced_index, current_index) << "install pwrites are unforced (no index fsync)";

  struct IndexState {
    const char *label;
    const std::vector<char> &index;
  };
  const std::vector<IndexState> states = {
      {"drop-all-install", forced_index},
      {"keep-all-install", current_index},
  };
  std::string s_new_fp;  // wal-2c BLOCKER-2: both install states must land on the same S_new graph
  for (const auto &st : states) {
    SCOPED_TRACE(std::string("consolidate power-loss: ") + st.label);
    const auto first =
        recover_power_loss_state(template_dir, root / "state", st.index, current_wal, max_points);
    ASSERT_FALSE(first.poisoned) << "END durable => must roll forward, not poison";
    EXPECT_EQ(first.num_points, kBaseN);
    EXPECT_EQ(first.live_count, kBaseN - kTomb);
    EXPECT_EQ(first.free_count, kTomb) << "the reclaimed rows land on the free-list (S_new)";
    // wal-2c BLOCKER-2: whether the index dropped ALL installs (forced snapshot) or kept them
    // (current file), the WAL redo must reconstruct the IDENTICAL S_new graph.
    if (s_new_fp.empty()) {
      s_new_fp = first.graph_fp;
    } else {
      EXPECT_EQ(first.graph_fp, s_new_fp) << st.label << " diverged from the S_new graph";
    }
    // wal-2c BLOCKER-2: reopen the FIRST recovery's OUTPUT for the stability check.
    const auto second = recover((root / "state" / "wal_base").string(), max_points);
    EXPECT_EQ(first.index_bytes, second.index_bytes) << "index not byte-stable on double replay";
    EXPECT_EQ(first.wal_bytes, second.wal_bytes) << "wal not byte-stable on double replay";
    EXPECT_EQ(first.graph_fp, second.graph_fp) << "graph fingerprint not stable on double replay";
  }
  std::filesystem::remove_all(root);
}

// wal-2c MAJOR-6: the C7 SIGKILL cut relies on a buffered END staying in THIS process's
// userspace stream buffer -- not portable (std::filebuf may push it to the kernel, so a SIGKILL
// could leave a complete, CRC-valid, un-fsynced END and legitimately recover S_new). This test
// EXPLICITLY materializes the torn-END disk states: a truncated END and a bad-CRC END must both
// truncate the epoch back to S_old, while a structurally complete END rolls forward to S_new.
// It proves the commit point is 'a complete END frame on disk', independent of stream buffering.
TEST(SegmentOpWalPowerLoss, ConsolidateTornEndMaterialization) {
  constexpr size_t kTomb = 4;
  const auto root = battery_root("torn_end");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 8484);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const auto run_dir = root / "run";
  copy_tree(template_dir, run_dir);
  const std::string run_prefix = (run_dir / "wal_base").string();
  const std::string index_path = run_prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  std::vector<char> forced_index;  // pre-install snapshot (activation ckpt; install pwrites unforced)
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(run_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    SegmentIoObserver observer;
    observer.on_index_fsync = [&] { forced_index = read_file(index_path); };
    params.io_observer = &observer;
    QGUpdater upd(qg, params);
    for (PID id = 0; id < static_cast<PID>(kTomb); ++id) upd.tombstone(id);
    upd.consolidate(1, /*r_target=*/0, /*reclaim_slots=*/true, /*bloom_consolidate=*/false);
  }
  const auto complete_wal = read_file(wal_path);   // [flip, begin, pages..., END] (END complete)
  const auto complete_index = read_file(index_path);
  ASSERT_GT(complete_wal.size(), 80U);
  ASSERT_FALSE(forced_index.empty()) << "the activation checkpoint must fsync the index once";
  EXPECT_NE(forced_index, complete_index) << "install pwrites are unforced (no index fsync)";

  // Torn-END variants must be materialized over the PRE-INSTALL (forced) index: the FREE
  // trailers only exist on disk after the (unforced) install, so a torn WAL over the post-
  // install index would spuriously read S_new straight off the trailers.
  auto materialize_and_recover = [&](const std::vector<char> &index,
                                     const std::vector<char> &wal) {
    const auto dst = root / "state";
    copy_tree(template_dir, dst);
    const std::string p = (dst / "wal_base").string();
    write_file(p + waltest::index_suffix(), index);
    write_file(p + waltest::index_suffix() + ".opwal", wal);
    const auto a = recover(p, max_points);
    const auto b = recover(p, max_points);  // reopen a's OUTPUT (idempotence)
    EXPECT_EQ(a.graph_fp, b.graph_fp) << "torn-end recovery not idempotent";
    return a;
  };

  // 1) Truncated END: drop the trailer -> the scan stops before the END frame boundary.
  {
    auto wal = complete_wal;
    wal.resize(wal.size() - 6);
    const auto r = materialize_and_recover(forced_index, wal);
    ASSERT_FALSE(r.poisoned) << "truncated END must recover to S_old, not poison";
    EXPECT_EQ(r.free_count, 0U) << "torn END => epoch truncated => S_old";
    EXPECT_EQ(r.live_count, kBaseN - kTomb);
    EXPECT_EQ(r.epoch, 0U) << "the uncommitted epoch never advances the cursor";
  }
  // 2) Bad-CRC END: flip a byte inside the END payload -> CRC fails, scan stops.
  {
    auto wal = complete_wal;
    wal[wal.size() - 10] = static_cast<char>(wal[wal.size() - 10] ^ 0xFF);
    const auto r = materialize_and_recover(forced_index, wal);
    ASSERT_FALSE(r.poisoned) << "bad-CRC END must recover to S_old, not poison";
    EXPECT_EQ(r.free_count, 0U);
    EXPECT_EQ(r.live_count, kBaseN - kTomb);
    EXPECT_EQ(r.epoch, 0U);
  }
  // 3) Complete END over the PRE-INSTALL index: the redo re-applies the pages => S_new.
  {
    const auto r = materialize_and_recover(forced_index, complete_wal);
    ASSERT_FALSE(r.poisoned);
    EXPECT_EQ(r.free_count, kTomb) << "a complete END frame on disk is the commit point => S_new";
    EXPECT_EQ(r.live_count, kBaseN - kTomb);
    EXPECT_EQ(r.epoch, 1U) << "the committed epoch advances the cursor";
  }
  // 4) Complete END over the POST-INSTALL index: redo is idempotent over installed pages.
  {
    const auto r = materialize_and_recover(complete_index, complete_wal);
    ASSERT_FALSE(r.poisoned);
    EXPECT_EQ(r.free_count, kTomb);
    EXPECT_EQ(r.live_count, kBaseN - kTomb);
    EXPECT_EQ(r.epoch, 1U);
  }
  std::filesystem::remove_all(root);
}

// The label-snapshot checkpoint flip is atomic across power loss (B-07): over an
// already-committed bundle, capture the forced content of index/WAL/slot0/slot1 at
// every fsync boundary, then materialize each TIME-ORDERED boundary state. Every
// state recovers to committed==base+n with the bundle labels -- whether they come
// from WAL replay (old base) or the durable slot (new base). No boundary exposes a
// superblock tuple that points at a not-yet-durable slot (B-05).
TEST(SegmentOpWalPowerLoss, LabelCheckpointBoundaryStatesAreAtomic) {
  const auto root = battery_root("label_ckpt_boundary");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 6262);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const auto run_dir = root / "run";
  copy_tree(template_dir, run_dir);
  const std::string run_prefix = (run_dir / "wal_base").string();
  const std::string index_path = run_prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  const std::string slot0_path = index_path + ".labels.slot0";
  const std::string slot1_path = index_path + ".labels.slot1";

  const auto vecs = waltest::make_data(kInsert, kDim, 828);
  std::vector<uint64_t> labels;
  for (size_t i = 0; i < kInsert; ++i) labels.push_back(70000 + i);

  struct Forced {
    std::vector<char> index, wal, slot0, slot1;
  };
  Forced forced;
  std::vector<Forced> timeline;
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(run_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    SegmentIoObserver observer;
    observer.on_index_fsync = [&] {
      forced.index = read_file(index_path);
      timeline.push_back(forced);
    };
    observer.on_wal_fsync = [&] {
      forced.wal = read_file(wal_path);
      timeline.push_back(forced);
    };
    observer.on_label_slot_fsync = [&] {
      forced.slot0 = read_file(slot0_path);
      forced.slot1 = read_file(slot1_path);
      timeline.push_back(forced);
    };
    params.io_observer = &observer;
    QGUpdater upd(qg, params);  // recovery open: fires no observer callbacks
    // Base durable state before any bundle/checkpoint fsync.
    forced.index = read_file(index_path);
    forced.wal = read_file(wal_path);
    forced.slot0 = read_file(slot0_path);
    forced.slot1 = read_file(slot1_path);
    (void)upd.commit_physical_bundle(11, 5, vecs.data(), labels.data(), kInsert);  // kind=8 fsync
    upd.checkpoint();  // slot fsync -> wal(flip) -> index(superblock) -> wal(reset)
  }
  ASSERT_GE(timeline.size(), 4U) << "expected the label-checkpoint boundary fsync cuts";

  for (size_t i = 0; i < timeline.size(); ++i) {
    SCOPED_TRACE("boundary state " + std::to_string(i));
    auto materialize = [&](const std::filesystem::path &dst) {
      copy_tree(template_dir, dst);
      const std::string p = (dst / "wal_base").string() + waltest::index_suffix();
      write_file(p, timeline[i].index);
      write_file(p + ".opwal", timeline[i].wal);
      write_file(p + ".labels.slot0", timeline[i].slot0);
      write_file(p + ".labels.slot1", timeline[i].slot1);
      return (dst / "wal_base").string();
    };
    const auto first = recover(materialize(root / ("state" + std::to_string(i))), max_points);
    ASSERT_FALSE(first.poisoned) << "boundary state " << i << " must recover, not poison";
    EXPECT_EQ(first.num_points, kBaseN + kInsert) << "the bundle was committed before the checkpoint";
    EXPECT_EQ(first.live_count, kBaseN + kInsert);
    ASSERT_TRUE(first.appended_label.has_value()) << "boundary " << i << " lost the labels";
    EXPECT_EQ(*first.appended_label, 70000U);
    EXPECT_EQ(first.last_txid, 11U);
    // Double replay of the materialized boundary state is byte-stable.
    const auto p2 = materialize(root / ("state2_" + std::to_string(i)));
    const auto a = recover(p2, max_points);
    const auto b = recover(p2, max_points);
    EXPECT_EQ(a.index_bytes, b.index_bytes) << "boundary " << i << " index not byte-stable";
    EXPECT_EQ(a.wal_bytes, b.wal_bytes) << "boundary " << i << " wal not byte-stable";
  }
  std::filesystem::remove_all(root);
}

// ---------------------------------------------------------------------------
// Post-mortem WAL damage: a torn tail or a corrupt earlier frame must never
// resynchronize past the damage (the scanner returns exactly the verified
// prefix). Recovery converges to a prefix and stays byte-stable.
// ---------------------------------------------------------------------------

Recovered run_with_wal_damage(const std::string &name,
                              const std::function<void(std::vector<char> &)> &damage) {
  const auto root = battery_root(name);
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 7654);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const std::string prefix = base.prefix;
  const std::string wal_path = prefix + waltest::index_suffix() + ".opwal";
  auto inserted = waltest::make_data(kInsert, kDim, 246);
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    QGUpdater upd(qg, params);
    for (size_t i = 0; i < kInsert; ++i) upd.allocate_and_insert(inserted.data() + i * kDim);
    upd.publish(upd.allocated_points());
    upd.writeback(1);  // force the WAL fully durable on disk so damage is reproducible
  }
  auto wal_bytes = read_file(wal_path);
  damage(wal_bytes);
  write_file(wal_path, wal_bytes);
  const auto first = recover(prefix, max_points);
  const auto second = recover(prefix, max_points);
  EXPECT_EQ(first.num_points, second.num_points) << name << " not stable";
  EXPECT_EQ(first.index_bytes, second.index_bytes) << name << " index not byte-stable";
  std::filesystem::remove_all(root);
  return first;
}

TEST(SegmentOpWalDamage, TruncatedTailStopsAtVerifiedPrefix) {
  // Chop the final bytes: the last (publish) frame is torn, so recovery stops
  // before it -> the batch is not committed.
  const auto r = run_with_wal_damage("torn_tail", [](std::vector<char> &bytes) {
    if (bytes.size() > 6) bytes.resize(bytes.size() - 6);
  });
  EXPECT_FALSE(r.poisoned);
  EXPECT_TRUE(r.num_points == kBaseN || r.num_points == kBaseN + kInsert);
}

TEST(SegmentOpWalDamage, CorruptEarlyFrameNeverResynchronizesToLaterFrames) {
  // Flip a byte inside the first row_patch payload (after the flip marker).
  // The CRC breaks; the scan stops there and never reaches the later publish,
  // so no committed batch is applied.
  const auto r = run_with_wal_damage("corrupt_mid", [](std::vector<char> &bytes) {
    // The marker frame is ~600 bytes; land the flip well into the next frame.
    if (bytes.size() > 800) bytes[750] = static_cast<char>(bytes[750] ^ 0xFF);
  });
  EXPECT_FALSE(r.poisoned);
  EXPECT_EQ(r.num_points, kBaseN) << "a corrupt early frame must drop the whole later suffix";
}

// Acceptance 2: recovery via WAL replay and recovery via the surviving A/B
// superblock agree. A completed checkpoint leaves both a durable new superblock
// and a WAL marker; recovering it matches recovering the pre-checkpoint WAL over
// the old superblock.
TEST(SegmentOpWalEquivalence, SuperblockAndWalReplayAgree) {
  const auto root = battery_root("equiv");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 1357);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  auto inserted = waltest::make_data(kInsert, kDim, 802);
  std::vector<uint64_t> labels;
  for (size_t i = 0; i < kInsert; ++i) labels.push_back(70000 + i);
  const std::string index_path = base.prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";

  // Capture the pre-checkpoint WAL (row_patches + publish) and the old index.
  std::vector<char> pre_index;
  std::vector<char> pre_wal;
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    QGUpdater upd(qg, params);
    (void)upd.commit_physical_bundle(13, 7, inserted.data(), labels.data(), kInsert);
    upd.writeback(1);
    pre_index = read_file(index_path);
    pre_wal = read_file(wal_path);
    upd.checkpoint();  // durable new superblock (label slot + tx watermark) + WAL reset to a marker
  }
  // Recovery A: via the surviving (new) superblock + marker.
  const auto via_superblock = recover(base.prefix, max_points);
  // Recovery B: roll the index/WAL back to the pre-checkpoint state (old
  // superblock + full WAL) and recover via replay.
  write_file(index_path, pre_index);
  write_file(wal_path, pre_wal);
  const auto via_wal = recover(base.prefix, max_points);
  ASSERT_FALSE(via_superblock.poisoned);
  ASSERT_FALSE(via_wal.poisoned);
  EXPECT_EQ(via_superblock.num_points, via_wal.num_points);
  EXPECT_EQ(via_superblock.live_count, via_wal.live_count);
  EXPECT_EQ(via_superblock.num_points, kBaseN + kInsert);
  // Both recovery paths agree on the label state and the tx watermark: via the
  // slot (new superblock) and via WAL replay (pre-checkpoint) must be identical.
  EXPECT_EQ(via_superblock.last_txid, via_wal.last_txid);
  EXPECT_EQ(via_superblock.last_txid, 13U);
  EXPECT_EQ(via_superblock.appended_label, via_wal.appended_label);
  ASSERT_TRUE(via_superblock.appended_label.has_value());
  EXPECT_EQ(*via_superblock.appended_label, 70000U);
  std::filesystem::remove_all(root);
}

// codex suggestion 6: a residual .reset.tmp is not a correctness hole -- the next
// reset overwrites it (trunc) before renaming, so recovery is unaffected.
TEST(SegmentOpWalDamage, ResidualResetTmpIsHarmless) {
  const auto root = battery_root("reset_tmp");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 2468);
  const size_t max_points = kBaseN + kInsert + 16;
  prepare_wal_base(base.prefix, max_points);
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  write_file(wal_path + ".reset.tmp", std::vector<char>(1234, static_cast<char>(0xAB)));
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.max_points = max_points;
    QGUpdater upd(qg, params);
    auto v = waltest::make_data(4, kDim, 1);
    std::vector<uint64_t> labels = {90000, 90001, 90002, 90003};
    (void)upd.commit_physical_bundle(1, 1, v.data(), labels.data(), 4);
    upd.checkpoint();  // reset_to_single_frame trunc-writes the tmp, then renames
  }
  const auto r = recover(base.prefix, max_points);
  EXPECT_FALSE(r.poisoned);
  EXPECT_EQ(r.num_points, kBaseN + 4);
  std::filesystem::remove_all(root);
}

// codex BLOCKER (fresh-enable crash window): a fresh WAL enable that SIGKILLs after
// the flip fsync but before the superblock write leaves an unstamped superblock and
// an orphan flip in the .opwal. Reopen must DISCARD the orphan and re-stamp -- a
// legal, recoverable window -- not poison.
TEST(SegmentOpWalPowerLoss, FreshEnableCrashAfterFlipIsRecoverable) {
  const auto root = battery_root("fresh_crash");
  std::filesystem::remove_all(root);
  const auto template_dir = root / "template";
  auto base = WalTinyIndex::build(template_dir, kBaseN, 9753);  // uid still 0 (no prepare_wal_base)
  const size_t max_points = kBaseN + kInsert + 16;
  const auto case_dir = root / "case";
  copy_tree(template_dir, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      UpdateParams params;
      params.enable_wal = true;
      params.max_points = max_points;
      params.failpoint_hook = [](SegmentOpFailPoint fp) {
        if (fp == SegmentOpFailPoint::after_flip_append_before_superblock_write) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      QGUpdater upd(qg, params);  // the fresh-enable checkpoint fires the failpoint
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);

  const auto first = recover(case_prefix, max_points);
  EXPECT_FALSE(first.poisoned) << "fresh-enable crash window must be recoverable, not poison";
  EXPECT_EQ(first.num_points, kBaseN);
  const auto second = recover(case_prefix, max_points);  // now stamped: ordinary recovery
  EXPECT_FALSE(second.poisoned);
  EXPECT_EQ(second.num_points, kBaseN);
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace alaya::laser
