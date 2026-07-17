// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// W2 step 8 / BLOCKER-7: the R0-R11 crash matrix for the canonical PID-reuse writer.
//
// A forked child self-SIGKILLs at each cut of one all-reuse bundle over a pre-activated
// v3 pid base with K free PIDs. The parent asserts the child died of SIGKILL (never a
// false-green where the failpoint was skipped and the child committed normally), reopens
// (recovery), and asserts the recovered FULL fingerprint -- counts, applied-op/epoch,
// entry point, medoid set, per-PID trailer + hidden bit + full row bytes (live AND dead),
// the walked free chain, and the label bindings -- equals exactly S_old (every cut before
// the kind=8 fsync) or S_new (every cut at/after it). It reopens the FIRST recovery's
// OUTPUT a SECOND time for double-replay byte stability. A power-loss family materializes
// the three R5 states directly on the WAL bytes: a complete END rolls forward to S_new; a
// torn END (last frame cut mid-write -- a real power-loss) and an absent END both recover
// S_old. Complements the QGUpdater / segment functional families.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "qg_wal_test_support.hpp"

namespace alaya::laser {
namespace {

using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 96;
constexpr size_t kFree = 4;        // base rows tombstoned + reclaimed, then reused
constexpr uint64_t kNewLabelBase = 800000;

std::filesystem::path battery_root(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("qg_reuse_crash_" + name + "_" + std::to_string(::getpid()));
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

UpdateParams reuse_params(size_t max_points,
                          std::function<void(SegmentOpFailPoint)> hook = {},
                          size_t cache_cap = 0) {
  UpdateParams params;
  params.enable_wal = true;
  params.enable_pid_reuse = true;
  params.max_points = max_points;
  params.ef_insert = 64;
  if (cache_cap != 0) params.cache_cap_pages = cache_cap;  // tiny cap forces overlay spill
  params.failpoint_hook = std::move(hook);
  return params;
}

// The bundle vectors + labels every case (and the S_new reference) reuses.
const std::vector<float> &bundle_vecs() {
  static const std::vector<float> v = waltest::make_data(kFree, kDim, 0x5EED);
  return v;
}
std::vector<uint64_t> bundle_labels() {
  std::vector<uint64_t> l(kFree);
  for (size_t i = 0; i < kFree; ++i) l[i] = kNewLabelBase + i;
  return l;
}

// Build a durable, pre-activated v3 pid template: activate via a 1-row append bundle,
// tombstone K base rows, reclaim them into the free-list, checkpoint. The result has K
// free PIDs ready for one all-reuse bundle. No failpoint/observer.
void prepare_reuse_template(const std::string &prefix, size_t max_points) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points));
  const auto seed_vec = waltest::make_data(1, kDim, 0xA11CE);
  const uint64_t seed_label = 111111;
  (void)upd.commit_physical_bundle(1, 1, seed_vec.data(), &seed_label, 1);  // activates v3 pid
  for (PID id = 0; id < static_cast<PID>(kFree); ++id) upd.tombstone(id);
  upd.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);
  upd.checkpoint();  // durable v3 pid base with kFree free PIDs
}

// A generation-aware graph fingerprint: counts, all running watermarks, the routing
// entry + medoid set, every PID's trailer + hidden bit + FULL row bytes (live AND dead --
// captures the free-chain next pointer, reused incarnations, and stale dead-row content),
// the walked free chain, and the sorted label bindings (pid -> gen -> label). Two recovered
// states are the SAME iff their fingerprints match. This is deliberately over-complete so
// a routing/medoid/dead-row divergence the earlier fingerprint missed cannot pass.
std::string capture_fp(QGUpdater &upd) {
  std::string fp;
  auto put = [&](uint64_t v) { fp.append(reinterpret_cast<const char *>(&v), sizeof(v)); };
  const auto n = static_cast<PID>(upd.num_points());
  put(upd.num_points());
  put(upd.allocated_points());
  put(upd.live_count());
  put(upd.free_count());
  put(upd.last_committed_txid());
  put(upd.applied_collection_op_id());
  put(upd.last_completed_consolidate_epoch());
  put(static_cast<uint64_t>(upd.entry_point()));
  put(upd.medoids().size());
  for (PID m : upd.medoids()) put(static_cast<uint64_t>(m));
  for (PID id = 0; id < n; ++id) {
    const auto tr = upd.trailer(id);
    put(tr.flags);
    put(tr.valid_degree);
    put(upd.row_hidden(id) ? 1U : 0U);
    const auto row = upd.debug_read_row(id);  // full node bytes: live rows AND dead rows
    fp.append(row.data(), row.size());
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

struct Recovered {
  bool poisoned = false;
  std::string fp;
  std::vector<char> index_bytes;
  std::vector<char> wal_bytes;
};

Recovered recover(const std::string &prefix, size_t max_points) {
  Recovered r;
  const std::string index_path = prefix + waltest::index_suffix();
  try {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    r.fp = capture_fp(upd);
  } catch (const std::exception &) {
    r.poisoned = true;
  }
  r.index_bytes = read_file(index_path);
  r.wal_bytes = read_file(index_path + ".opwal");
  return r;
}

// S_old = the template (bundle never applied); S_new = template + the all-reuse bundle.
// Built once from a pristine copy of the SAME template each case uses.
struct Refs {
  std::string s_old;
  std::string s_new;
  size_t max_points;
};
const Refs &refs() {
  static const Refs r = [] {
    Refs out;
    out.max_points = kBaseN + 16;
    const auto root = battery_root("refs");
    std::filesystem::remove_all(root);
    const auto tmpl = root / "template";
    auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
    prepare_reuse_template(base.prefix, out.max_points);
    const auto old_dir = root / "old";
    copy_tree(tmpl, old_dir);
    out.s_old = recover((old_dir / "wal_base").string(), out.max_points).fp;
    const auto new_dir = root / "new";
    copy_tree(tmpl, new_dir);
    {
      const std::string prefix = (new_dir / "wal_base").string();
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      QGUpdater upd(qg, reuse_params(out.max_points));
      const auto labels = bundle_labels();
      (void)upd.commit_physical_bundle(2, 2, bundle_vecs().data(), labels.data(), kFree);
    }
    out.s_new = recover((new_dir / "wal_base").string(), out.max_points).fp;
    return out;
  }();
  return r;
}

struct ReuseKill {
  const char *name;
  SegmentOpFailPoint point;
  bool expect_new;       // cut at/after the kind=8 fsync -> roll forward (S_new)
  size_t cache_cap = 0;  // 0 = default (no spill); tiny cap forces overlay spill mid-build
};

class ReuseCrash : public ::testing::TestWithParam<ReuseKill> {};

TEST_P(ReuseCrash, ReopenLandsOnSoldOrSnewAndDoubleReplayStable) {
  const auto param = GetParam();
  const auto &ref = refs();
  ASSERT_NE(ref.s_old, ref.s_new) << "S_old and S_new must differ for the matrix to discriminate";
  const auto root = battery_root(param.name);
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);  // same seed as refs()
  prepare_reuse_template(base.prefix, ref.max_points);
  const auto case_dir = root / "case";
  copy_tree(tmpl, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      const SegmentOpFailPoint target = param.point;
      QGUpdater upd(qg, reuse_params(ref.max_points,
                                     [target](SegmentOpFailPoint fp) {
                                       if (fp == target) {
                                         ::kill(::getpid(), SIGKILL);
                                         ::_exit(99);
                                       }
                                     },
                                     param.cache_cap));
      const auto labels = bundle_labels();
      (void)upd.commit_physical_bundle(2, 2, bundle_vecs().data(), labels.data(), kFree);
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  // BLOCKER-7: the child MUST have died of the injected SIGKILL. A normal exit (0) means the
  // failpoint was never reached, so an S_new expectation would false-green on a clean commit;
  // exit 70 means it threw before the cut. Either invalidates the case.
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << param.name << " child did not die of SIGKILL at the cut (status=" << status << ")";

  const auto first = recover(case_prefix, ref.max_points);
  ASSERT_FALSE(first.poisoned) << param.name << " recovery must not poison";
  EXPECT_EQ(first.fp, param.expect_new ? ref.s_new : ref.s_old)
      << param.name << " landed on the wrong graph (expected " << (param.expect_new ? "S_new" : "S_old")
      << ")";
  // Reopen the FIRST recovery's OUTPUT a second time (not the crash image again).
  const auto second = recover(case_prefix, ref.max_points);
  EXPECT_EQ(first.fp, second.fp) << param.name << " second reopen diverged";
  EXPECT_EQ(first.index_bytes, second.index_bytes) << param.name << " index not byte-stable";
  EXPECT_EQ(first.wal_bytes, second.wal_bytes) << param.name << " wal not byte-stable";
  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    RMatrix, ReuseCrash,
    ::testing::Values(
        // R0: tokens reserved, before any kind=7 -> whole bundle dropped.
        ReuseKill{"R0_reserve", SegmentOpFailPoint::after_reuse_reserve_before_binds, false},
        // R1a: exactly ONE kind=7 bind buffered (a PARTIAL bind set) -> no kind=8 -> dropped.
        ReuseKill{"R1a_partial_bind", SegmentOpFailPoint::after_reuse_first_bind_append, false},
        // R1b: all kind=7 buffered, no kind=1/kind=8 -> canonical lane EOF discards.
        ReuseKill{"R1b_all_binds", SegmentOpFailPoint::after_label_bind_append, false},
        // R2-R4: preimage + final kind=1 all buffered, before the kind=8 append.
        ReuseKill{"R4_before_publish", SegmentOpFailPoint::before_tx_publish_append, false},
        // R4 with a tiny cache cap: the build spills overlay pages (Sync::flush, survives
        // SIGKILL), but the FIRST spill forces the kind=7 lane durable, so a surviving spill
        // is staged inside the (durable) lane and discarded at EOF -> S_old (never applied as
        // an orphan legacy row_patch). Exercises the spill-orphan guard.
        ReuseKill{"R4_spill", SegmentOpFailPoint::before_tx_publish_append, false, /*cap=*/1},
        // R5: kind=8 appended into the WAL's USERSPACE buffer but before its force. A SIGKILL
        // drops the userspace buffer -> no durable canonical frame -> S_old.
        ReuseKill{"R5_end_unforced", SegmentOpFailPoint::after_reuse_tx_publish_append_before_fsync,
                  false},
        // R6: kind=8 forced durable, before install/snapshot -> roll forward.
        ReuseKill{"R6_after_fsync", SegmentOpFailPoint::after_tx_publish_fsync, true},
        // R6b: kind=8 durable + pages installed into the cache (lost on crash), before the
        // snapshot swap -> reopen re-derives S_new from the durable WAL.
        ReuseKill{"R6b_after_install", SegmentOpFailPoint::after_reuse_install_before_snapshot, true},
        // R7: snapshot + routing published, before the reused hidden bits are cleared -> S_new.
        ReuseKill{"R7_after_routing", SegmentOpFailPoint::after_reuse_routing_before_hidden_clear, true},
        // R8: the first reused hidden bit cleared, before the committed watermark store -> S_new
        // (the durable kind=8 makes the whole bundle roll forward regardless of the in-memory
        // publish progress, which is not itself durable).
        ReuseKill{"R8_hidden_partial", SegmentOpFailPoint::after_reuse_hidden_clear_partial_before_commit,
                  true}),
    [](const ::testing::TestParamInfo<ReuseKill> &info) { return info.param.name; });

// R5 three states materialized directly on the committed WAL bytes (no invalid "delete an
// fsync'd tx" model). Run the bundle to its committed END, then:
//   (1) leave the complete END      -> recovery rolls forward   -> S_new
//   (2) cut the last frame mid-write -> a real torn power-loss   -> S_old
//   (3) drop the whole END frame     -> the lane sees no kind=8   -> S_old
class ReuseEndTail : public ::testing::TestWithParam<int> {};

TEST_P(ReuseEndTail, PowerLossEndStates) {
  const int cut = GetParam();  // 0 = keep complete, >0 = bytes to cut off the tail
  const auto &ref = refs();
  const auto root = battery_root("endtail" + std::to_string(cut));
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  prepare_reuse_template(base.prefix, ref.max_points);
  const auto case_dir = root / "case";
  copy_tree(tmpl, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();
  const std::string case_wal = case_prefix + waltest::index_suffix() + ".opwal";
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(ref.max_points));
    const auto labels = bundle_labels();
    (void)upd.commit_physical_bundle(2, 2, bundle_vecs().data(), labels.data(), kFree);
  }
  const auto full_len = static_cast<std::uintmax_t>(read_file(case_wal).size());
  if (cut > 0) {
    ASSERT_GT(full_len, static_cast<std::uintmax_t>(cut));
    std::filesystem::resize_file(case_wal, full_len - static_cast<std::uintmax_t>(cut));
  }
  const auto rec = recover(case_prefix, ref.max_points);
  ASSERT_FALSE(rec.poisoned) << "power-loss recovery must not poison (cut=" << cut << ")";
  EXPECT_EQ(rec.fp, cut == 0 ? ref.s_new : ref.s_old)
      << "cut=" << cut << " landed on the wrong graph";
  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(EndStates, ReuseEndTail,
                         // 0 = complete END (S_new); 4 = last frame trailer cut (torn -> S_old);
                         // 40 = whole END frame dropped (absent -> S_old).
                         ::testing::Values(0, 4, 40),
                         [](const ::testing::TestParamInfo<int> &info) {
                           return "cut" + std::to_string(info.param);
                         });

}  // namespace
}  // namespace alaya::laser
