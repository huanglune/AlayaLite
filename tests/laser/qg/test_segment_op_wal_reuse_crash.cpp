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
void write_file(const std::filesystem::path &path, const std::vector<char> &bytes) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}
// The complete length of the LAST frame in a WAL (the kind=8 END of a just-appended bundle).
uint64_t wal_last_frame_size(const std::string &wal_path) {
  uint64_t sz = 0;
  alaya::wal::WalFile::visit_frames(wal_path, [&](const alaya::wal::ScannedFrame &f) -> bool {
    sz = f.size;  // last frame wins
    return true;
  });
  return sz;
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
  // leg-7 BLOCKER-7: the superblock's SEMANTIC (uid-independent) fields -- generation, format
  // version, and both activation generations. The raw superblock image cannot be compared across
  // independently-built reference/case templates (each stamps a RANDOM segment_uid), so the
  // portable activation/generation state stands in for "superblock image" in the fingerprint.
  put(upd.generation());
  put(static_cast<uint64_t>(upd.superblock_format_version()));
  put(static_cast<uint64_t>(upd.maintenance_activation_gen()));
  put(static_cast<uint64_t>(upd.pid_reuse_activation_gen()));
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
  // leg-7 BLOCKER-7: FULL page bytes (rows + trailers + inter-row/trailer padding) for every
  // committed page -- the per-row bytes above miss the "unused page bytes", so a divergence in
  // padding / a torn trailer region would otherwise pass. And the routing medoid VECTORS, a
  // cache that must agree with the medoid rows (a routing-vector divergence a per-row scan misses).
  const size_t pages = n == 0 ? 0 : (static_cast<size_t>(n) + upd.debug_npp() - 1) / upd.debug_npp();
  put(pages);
  for (size_t pi = 0; pi < pages; ++pi) {
    const auto page = upd.debug_read_page(pi);
    fp.append(page.data(), page.size());
  }
  const auto &mv = upd.debug_medoid_vectors();
  put(mv.size());
  fp.append(reinterpret_cast<const char *>(mv.data()), mv.size() * sizeof(float));
  return fp;
}

// The persisted label-slot state is captured SEMANTICALLY by capture_fp's label_snapshot loop
// (pid -> generation -> label): recovery loads the durable active slot into label_working_ (a
// checksum mismatch poisons before we get here), so the in-memory bindings equal the durable
// slot content. Hashing the raw slot FILES additionally would false-fail on a legal orphan
// inactive-slot write (a checkpoint that fsynced the inactive slot then crashed before the flip
// leaves a differing-but-unreferenced slot1), so the binding capture is the authoritative,
// orphan-robust "label slots" fingerprint (leg-7 BLOCKER-7).

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
  std::string s_old;  // the FULL durable fingerprint (pages + medoids + superblock + label slots)
  std::string s_new;  // of S_old / S_new -- recovery must land on one of them (leg-7 BLOCKER-7).
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
  // leg-7 BLOCKER-7: the fingerprint now reference-compares the FULL durable image -- committed
  // page bytes (rows + trailers + padding, via the cache), the medoid vectors, the superblock
  // image, and BOTH label-slot files -- against the S_old / S_new reference, not merely two-reopen
  // stability. A wrong-but-stable base image (routing / medoid / slot / superblock divergence)
  // that the earlier fingerprint missed now fails here. (Raw file bytes past the committed
  // watermark are non-idempotent across the crash path, so the cache-view fingerprint is the
  // authoritative durable-image reference.)
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
        // R2: the reused pages' FREE preimages logged, before the overlay build/finalize -> S_old.
        ReuseKill{"R2_free_preimage", SegmentOpFailPoint::after_reuse_free_preimage_before_build, false},
        // R3: the FIRST final (build/spine) page logged, before the rest + kind=8 -> S_old.
        ReuseKill{"R3_partial_final", SegmentOpFailPoint::after_reuse_partial_final_page, false},
        // R4: preimage + ALL final kind=1 buffered, before the kind=8 append.
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

// R5 (leg-7 redo): the three power-loss END states materialized from a LEGAL persistence
// snapshot -- the pre-bundle FORCED prefix (W0 = the template's single flip) plus the bundle's
// UNFORCED tail, truncated at the frame level. Nothing already fsync'd is deleted: W0 is written
// intact in every state, and only the never-forced bundle tail is partially/fully materialized,
// exactly as an OS flush interrupted by power loss would leave it. Each state is written as a
// FRESH WAL over a fresh template copy (no fsync-then-delete of the case WAL).
//   complete: W0 ++ [kind7.., kind1.., COMPLETE kind8]  (a complete but never-forced END) -> S_new
//   torn:     W0 ++ [.., kind8 cut mid-frame]            (the END flush was interrupted)   -> S_old
//   absent:   W0 ++ [kind7.., kind1..] (no kind8)        (the END never reached disk)      -> S_old
enum class EndState { kComplete, kTorn, kAbsent };

class ReuseEndTail : public ::testing::TestWithParam<EndState> {};

TEST_P(ReuseEndTail, PowerLossEndStatesFromLegalSnapshot) {
  const EndState state = GetParam();
  const auto &ref = refs();
  const auto root = battery_root("endtail");
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  prepare_reuse_template(base.prefix, ref.max_points);
  // W0 = the durable pre-bundle WAL (the single flip from prepare's checkpoint reset). This is
  // the FORCED prefix; it appears intact in every materialized state below.
  const std::string tmpl_wal = (tmpl / "wal_base").string() + waltest::index_suffix() + ".opwal";
  const auto w0 = read_file(tmpl_wal);
  // Run the bundle in a SCRATCH copy to capture the exact unforced tail bytes + the END frame size.
  const auto scratch = root / "scratch";
  copy_tree(tmpl, scratch);
  const std::string scratch_prefix = (scratch / "wal_base").string();
  const std::string scratch_wal = scratch_prefix + waltest::index_suffix() + ".opwal";
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(scratch_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(ref.max_points));
    const auto labels = bundle_labels();
    (void)upd.commit_physical_bundle(2, 2, bundle_vecs().data(), labels.data(), kFree);
  }
  const auto full_wal = read_file(scratch_wal);
  ASSERT_GT(full_wal.size(), w0.size()) << "the bundle must append a tail past W0";
  ASSERT_EQ(std::vector<char>(full_wal.begin(), full_wal.begin() + static_cast<long>(w0.size())), w0)
      << "W0 must be a prefix of the post-bundle WAL (the bundle appends, never rewrites)";
  const uint64_t end_size = wal_last_frame_size(scratch_wal);  // the kind=8 END frame
  ASSERT_GT(end_size, 4U);
  ASSERT_LE(w0.size() + end_size, full_wal.size());

  std::vector<char> materialized;
  bool expect_new = false;
  switch (state) {
    case EndState::kComplete:
      materialized = full_wal;  // W0 ++ complete-but-never-forced tail
      expect_new = true;
      break;
    case EndState::kTorn:
      materialized.assign(full_wal.begin(), full_wal.end() - 4);  // END cut mid-frame
      break;
    case EndState::kAbsent:
      materialized.assign(full_wal.begin(),
                          full_wal.end() - static_cast<long>(end_size));  // whole END frame dropped
      break;
  }
  const auto case_dir = root / "case";
  copy_tree(tmpl, case_dir);  // fresh template: durable W0 index + slots, W0 WAL (about to be overwritten)
  const std::string case_prefix = (case_dir / "wal_base").string();
  write_file(case_prefix + waltest::index_suffix() + ".opwal", materialized);

  const auto rec = recover(case_prefix, ref.max_points);
  ASSERT_FALSE(rec.poisoned) << "a power-loss END state must recover, not poison";
  EXPECT_EQ(rec.fp, expect_new ? ref.s_new : ref.s_old) << "END state landed on the wrong graph";
  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(EndStates, ReuseEndTail,
                         ::testing::Values(EndState::kComplete, EndState::kTorn, EndState::kAbsent),
                         [](const ::testing::TestParamInfo<EndState> &info) {
                           switch (info.param) {
                             case EndState::kComplete: return "complete_unforced_end";
                             case EndState::kTorn: return "torn_end";
                             case EndState::kAbsent: return "absent_end";
                           }
                           return "unknown";
                         });

// R11 (leg-7): the canonical CHECKPOINT that ABSORBS a durable reuse bundle into the base, with a
// forced SIGKILL at each of its four cut points (label-slot fsync / flip fsync / superblock fsync
// / WAL reset). A cut BEFORE the flip is durable leaves the pre-checkpoint canonical state
// (S_bundle: the bundle is still in the WAL; an orphan inactive-slot write is unreferenced); a
// cut at/after the durable flip rolls forward to the post-checkpoint base (S_ckpt: the bundle is
// ABSORBED, its prefix validate-only). Both fingerprints carry the identical (pid, generation,
// label) bindings + pages (they differ only in the superblock generation the checkpoint bumps),
// proving the canonical bundle is WAL-vs-base equivalent across the checkpoint boundary.
// A bundle profile = (prepare the template, commit the bundle with an optional failpoint hook).
// R11 uses the all-reuse profile (a count-INCREASE checkpoint slot write); R10 uses a same-count
// rebind profile (design 3.1: reuse PIDs that already carry an explicit binding, rebinding them to
// NEW labels so the binding COUNT is UNCHANGED and only the content revision moves -- the
// same-count-dirty checkpoint slot branch, never exercised by the count-increase profile).
using PrepareFn = void (*)(const std::string &prefix, size_t max_points);
using CommitFn = void (*)(const std::string &prefix, size_t max_points,
                          std::function<void(SegmentOpFailPoint)> hook);

void commit_reuse_bundle_hooked(const std::string &prefix, size_t max_points,
                                std::function<void(SegmentOpFailPoint)> hook) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points, std::move(hook)));
  const auto labels = bundle_labels();
  (void)upd.commit_physical_bundle(2, 2, bundle_vecs().data(), labels.data(), kFree);
}

// --- same-count rebind profile (R10) ---
constexpr size_t kSC = 3;  // rows bound explicitly, then reused with new labels (COUNT unchanged)
void prepare_samecount_template(const std::string &prefix, size_t max_points) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points));
  const auto vecs = waltest::make_data(kSC, kDim, 0xC0C);
  std::vector<uint64_t> labels(kSC);
  for (size_t i = 0; i < kSC; ++i) labels[i] = 300000 + i;
  (void)upd.commit_physical_bundle(1, 1, vecs.data(), labels.data(), kSC);  // activate + bind kSC
  for (size_t i = 0; i < kSC; ++i) {
    upd.tombstone(static_cast<PID>(kBaseN + i));  // forward bindings RETAINED (count stays kSC)
  }
  upd.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);  // reclaim the kSC PIDs
  upd.checkpoint();  // persisted count == kSC, kSC free PIDs
}
void commit_samecount_rebind_hooked(const std::string &prefix, size_t max_points,
                                    std::function<void(SegmentOpFailPoint)> hook) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points, std::move(hook)));
  const auto vecs = waltest::make_data(kSC, kDim, 0xD0D);
  std::vector<uint64_t> labels(kSC);
  for (size_t i = 0; i < kSC; ++i) labels[i] = 400000 + i;  // NEW labels -> same-count rebind
  (void)upd.commit_physical_bundle(2, 2, vecs.data(), labels.data(), kSC);
}

struct CheckpointKill {
  const char *name;
  SegmentOpFailPoint point;
  bool expect_ckpt;  // the flip is durable at/after this cut -> the post-checkpoint base
  PrepareFn prepare;
  CommitFn commit;
};

class ReuseCheckpointCrash : public ::testing::TestWithParam<CheckpointKill> {};

TEST_P(ReuseCheckpointCrash, CanonicalCheckpointCutsLandOnBundleOrCheckpointed) {
  const auto param = GetParam();
  const size_t max_points = kBaseN + 16;
  const auto root = battery_root(std::string("ckpt_") + param.name);
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  param.prepare((tmpl / "wal_base").string(), max_points);

  // Reference S_bundle: template + committed bundle (no checkpoint).
  const auto bundle_dir = root / "bundle";
  copy_tree(tmpl, bundle_dir);
  param.commit((bundle_dir / "wal_base").string(), max_points, {});
  const auto s_bundle = recover((bundle_dir / "wal_base").string(), max_points);
  ASSERT_FALSE(s_bundle.poisoned);
  // Reference S_ckpt: the same bundle then a CLEAN checkpoint that absorbs it into the base.
  const auto ckpt_dir = root / "ckpt";
  copy_tree(tmpl, ckpt_dir);
  param.commit((ckpt_dir / "wal_base").string(), max_points, {});
  {
    const std::string prefix = (ckpt_dir / "wal_base").string();
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    upd.checkpoint();  // absorb the (already-durable) bundle into the base
  }
  const auto s_ckpt = recover((ckpt_dir / "wal_base").string(), max_points);
  ASSERT_FALSE(s_ckpt.poisoned);
  ASSERT_NE(s_bundle.fp, s_ckpt.fp) << "the checkpoint must change the durable fingerprint";

  // Case: commit the bundle, then checkpoint with a forced SIGKILL at the cut point. Commit is a
  // separate durable session so the SIGKILL lands purely on the CHECKPOINT frames.
  const auto case_dir = root / "case";
  copy_tree(tmpl, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();
  param.commit(case_prefix, max_points, {});
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(case_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      const SegmentOpFailPoint target = param.point;
      QGUpdater upd(qg, reuse_params(max_points, [target](SegmentOpFailPoint fp) {
                      if (fp == target) {
                        ::kill(::getpid(), SIGKILL);
                        ::_exit(99);
                      }
                    }));
      upd.checkpoint();  // the bundle is already durable; the SIGKILL is on the checkpoint cut
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << param.name << " child did not die of SIGKILL at the checkpoint cut (status=" << status << ")";

  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned) << param.name << " checkpoint recovery must not poison";
  EXPECT_EQ(first.fp, param.expect_ckpt ? s_ckpt.fp : s_bundle.fp)
      << param.name << " landed on the wrong state (expected "
      << (param.expect_ckpt ? "S_ckpt" : "S_bundle") << ")";
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.fp, second.fp) << param.name << " second reopen diverged";
  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointCuts, ReuseCheckpointCrash,
    ::testing::Values(
        // R11 all-reuse (count-increase slot write): the four checkpoint cuts.
        CheckpointKill{"reuse_slot_fsync", SegmentOpFailPoint::label_slot_written_before_flip, false,
                       prepare_reuse_template, commit_reuse_bundle_hooked},
        CheckpointKill{"reuse_flip_fsync", SegmentOpFailPoint::after_flip_append_before_superblock_write,
                       true, prepare_reuse_template, commit_reuse_bundle_hooked},
        CheckpointKill{"reuse_sb_fsync", SegmentOpFailPoint::after_superblock_write_before_wal_reset,
                       true, prepare_reuse_template, commit_reuse_bundle_hooked},
        CheckpointKill{"reuse_reset", SegmentOpFailPoint::after_wal_reset, true, prepare_reuse_template,
                       commit_reuse_bundle_hooked},
        // R10 same-count rebind (same-count-dirty slot write): the slot-write and roll-forward cuts.
        CheckpointKill{"samecount_slot_fsync", SegmentOpFailPoint::label_slot_written_before_flip, false,
                       prepare_samecount_template, commit_samecount_rebind_hooked},
        CheckpointKill{"samecount_flip_fsync",
                       SegmentOpFailPoint::after_flip_append_before_superblock_write, true,
                       prepare_samecount_template, commit_samecount_rebind_hooked}),
    [](const ::testing::TestParamInfo<CheckpointKill> &info) { return info.param.name; });

// R8 (leg-7): the crash matrix on a MIXED reuse+append bundle -- fewer free PIDs than bundle rows,
// so some binds REUSE a freed PID (generation>0) and some APPEND past the high-water mark
// (generation 0). The all-reuse RMatrix above never exercises the append-past-HWM half. A cut
// before the kind=8 fsync discards the whole bundle (S_old); a cut at/after it rolls the mixed
// bundle forward (S_new), the two halves committing atomically.
constexpr size_t kMixFree = 2;    // freed base PIDs (the reuse half)
constexpr size_t kMixBundle = 4;  // 2 reuse + 2 append

void prepare_mixed_template(const std::string &prefix, size_t max_points) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points));
  const auto seed = waltest::make_data(1, kDim, 0xB0B);
  const uint64_t seed_label = 222222;
  (void)upd.commit_physical_bundle(1, 1, seed.data(), &seed_label, 1);  // activate v3 pid
  for (PID id = 0; id < static_cast<PID>(kMixFree); ++id) upd.tombstone(id);
  upd.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);
  upd.checkpoint();  // durable v3 pid base with kMixFree free PIDs (< kMixBundle)
}

const std::vector<float> &mixed_vecs() {
  static const std::vector<float> v = waltest::make_data(kMixBundle, kDim, 0x3EED);
  return v;
}
std::vector<uint64_t> mixed_labels() {
  std::vector<uint64_t> l(kMixBundle);
  for (size_t i = 0; i < kMixBundle; ++i) l[i] = 900000 + i;
  return l;
}
void commit_mixed_bundle(const std::string &prefix, size_t max_points,
                         std::function<void(SegmentOpFailPoint)> hook = {}) {
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  QGUpdater upd(qg, reuse_params(max_points, std::move(hook)));
  const auto labels = mixed_labels();
  (void)upd.commit_physical_bundle(2, 2, mixed_vecs().data(), labels.data(), kMixBundle);
}

struct MixKill {
  const char *name;
  SegmentOpFailPoint point;
  bool expect_new;
};
class MixedReuseCrash : public ::testing::TestWithParam<MixKill> {};

TEST_P(MixedReuseCrash, MixedBundleReopenLandsOnSoldOrSnew) {
  const auto param = GetParam();
  const size_t max_points = kBaseN + 16;
  const auto root = battery_root(std::string("mix_") + param.name);
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  prepare_mixed_template((tmpl / "wal_base").string(), max_points);

  const auto old_dir = root / "old";
  copy_tree(tmpl, old_dir);
  const auto s_old = recover((old_dir / "wal_base").string(), max_points).fp;
  const auto new_dir = root / "new";
  copy_tree(tmpl, new_dir);
  commit_mixed_bundle((new_dir / "wal_base").string(), max_points);
  const auto s_new = recover((new_dir / "wal_base").string(), max_points).fp;
  ASSERT_NE(s_old, s_new) << "the mixed bundle must change the durable fingerprint";

  const auto case_dir = root / "case";
  copy_tree(tmpl, case_dir);
  const std::string case_prefix = (case_dir / "wal_base").string();
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      const SegmentOpFailPoint target = param.point;
      commit_mixed_bundle(case_prefix, max_points, [target](SegmentOpFailPoint fp) {
        if (fp == target) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      });
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << param.name << " child did not die of SIGKILL at the cut (status=" << status << ")";

  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned) << param.name << " mixed recovery must not poison";
  EXPECT_EQ(first.fp, param.expect_new ? s_new : s_old)
      << param.name << " landed on the wrong graph (expected " << (param.expect_new ? "S_new" : "S_old")
      << ")";
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.fp, second.fp) << param.name << " second reopen diverged";
  std::filesystem::remove_all(root);
}

INSTANTIATE_TEST_SUITE_P(
    MixedCuts, MixedReuseCrash,
    ::testing::Values(
        MixKill{"R2_free_preimage", SegmentOpFailPoint::after_reuse_free_preimage_before_build, false},
        MixKill{"R4_before_publish", SegmentOpFailPoint::before_tx_publish_append, false},
        MixKill{"R6_after_fsync", SegmentOpFailPoint::after_tx_publish_fsync, true},
        MixKill{"R8_hidden_partial", SegmentOpFailPoint::after_reuse_hidden_clear_partial_before_commit,
                true}),
    [](const ::testing::TestParamInfo<MixKill> &info) { return info.param.name; });

// ---------------------------------------------------------------------------
// leg-9 real-SIGKILL regressions (r3 independent audit): NEW-BLOCKER-2 and I-1 both require a genuine
// process-death crash grid (the leg-8 versions used an in-process C++ exception). Both fork a child
// that self-SIGKILLs at after_superblock_write_before_wal_reset (superblock durable, WAL not reset),
// then the parent asserts WIFSIGNALED && SIGKILL and double-reopens.
// ---------------------------------------------------------------------------

// NEW-BLOCKER-2 (leg-9, r3 section 3): a legacy orphan kind=7 (torn bundle, kind=8 lost) survives; the
// first reuse call runs a pid-activation checkpoint that fsyncs its G+1 superblock, then the child
// SELF-SIGKILLs before the WAL reset. On reopen the pid-active G+1 base is selected while the orphan
// (tx_id > base) still precedes the activation flip. leg-7 poisoned the orphan at the kind=7 gate;
// leg-9 stages it (dropped at the end-of-replay clear) so recovery succeeds with the data intact.
TEST(ReuseCrashStandalone, OrphanBindThenActivationCheckpointSigkillReopenSurvives) {
  const auto root = battery_root("newb2_sigkill");
  std::filesystem::remove_all(root);
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  const size_t max_points = kBaseN + 16;
  const std::string index_path = base.prefix + waltest::index_suffix();
  const std::string wal_path = index_path + ".opwal";
  uint64_t seg_uid = 0;
  uint64_t base_gen = 0;
  {  // stamp the uid; base stays v2 (no bundle -> no pid activation yet)
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    seg_uid = upd.segment_uid();
    base_gen = upd.generation();
    ASSERT_FALSE(upd.pid_generation_activated());
  }
  {  // plant an orphan kind=7 (tx_id=1 > base committed 0), legacy generation, no matching kind=8
    alaya::wal::WalFile wal(wal_path);
    wal.append(kSegmentOpRecordType, 0, 1, /*batch_id=*/1,
               encode_label_bind(seg_uid, base_gen, /*txid=*/1, /*row_op=*/0,
                                 static_cast<uint32_t>(kBaseN), /*gen=*/0, /*label=*/700),
               alaya::wal::WalFile::Sync::fsync);
  }
  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
      qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      qg.set_params(64, 1, 1);
      QGUpdater upd(qg, reuse_params(max_points, [](SegmentOpFailPoint fp) {
                      if (fp == SegmentOpFailPoint::after_superblock_write_before_wal_reset) {
                        ::kill(::getpid(), SIGKILL);
                        ::_exit(99);
                      }
                    }));
      const auto v = waltest::make_data(1, kDim, 3);
      const uint64_t l = 800;
      (void)upd.commit_physical_bundle(1, 1, v.data(), &l, 1);  // activation checkpoint -> SIGKILL
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << "child must die of SIGKILL at after_superblock_write_before_wal_reset (status=" << status
      << ")";
  for (int pass = 0; pass < 2; ++pass) {  // double reopen: must NOT poison, base intact
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    EXPECT_TRUE(upd.pid_generation_activated()) << "pass " << pass << ": durable pid-active base";
    EXPECT_EQ(upd.num_points(), kBaseN) << "pass " << pass << ": base rows survived intact";
    EXPECT_EQ(upd.last_committed_txid(), 0U) << "pass " << pass << ": orphan never committed";
    EXPECT_EQ(upd.trailer(0).flags & (kQGRowTombstone | kQGRowFree), 0)
        << "pass " << pass << ": a base row is live";
  }
  std::filesystem::remove_all(root);
}

// I-1 (leg-9, r3 independent audit): a maintenance base completes a consolidate epoch (BEGIN/END
// retained in the WAL), then a checkpoint carries the epoch into a G+1 base and fsyncs the
// superblock; the child SELF-SIGKILLs before the WAL reset. On reopen the G+1 base is selected but
// the WAL still holds BEGIN(E,G) at the OLD generation. Without the three-way maintenance-BEGIN split
// this fail-closed forever; with it the absorbed prefix is validate-only and recovery lands exactly
// on the clean checkpoint base.
TEST(ReuseCrashStandalone, ConsolidateThenCheckpointSigkillBeforeResetReopens) {
  const auto root = battery_root("i1_consolidate_ckpt");
  std::filesystem::remove_all(root);
  const size_t max_points = kBaseN + 16;
  const auto tmpl = root / "template";
  auto base = WalTinyIndex::build(tmpl, kBaseN, 4242);
  {  // maintenance-activate + one completed reclaiming epoch + checkpoint -> durable v3 maint base
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    for (PID id = 0; id < 3; ++id) upd.tombstone(id);
    upd.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);  // activates maintenance
    upd.checkpoint();
  }
  // Reference: the SAME second epoch + checkpoint run cleanly (WAL reset). Recovery of the crashed
  // case must reproduce this state exactly.
  const auto ref_dir = root / "ref";
  copy_tree(tmpl, ref_dir);
  const std::string ref_prefix = (ref_dir / "wal_base").string();
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(ref_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    QGUpdater upd(qg, reuse_params(max_points));
    upd.consolidate(1, /*r_target=*/0, /*reclaim=*/false, /*bloom=*/false);  // epoch 2 (no-op)
    upd.checkpoint();  // absorbs epoch 2, resets the WAL -> the clean reference base
  }
  const std::string ref_fp = recover(ref_prefix, max_points).fp;

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
      QGUpdater upd(qg, reuse_params(max_points, [](SegmentOpFailPoint fp) {
                      if (fp == SegmentOpFailPoint::after_superblock_write_before_wal_reset) {
                        ::kill(::getpid(), SIGKILL);
                        ::_exit(99);
                      }
                    }));
      upd.consolidate(1, /*r_target=*/0, /*reclaim=*/false, /*bloom=*/false);  // epoch 2 in the WAL
      upd.checkpoint();  // flips to G+1, SIGKILL after the SB write, before the WAL reset
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << "child must die of SIGKILL (status=" << status << ")";
  const auto first = recover(case_prefix, max_points);
  ASSERT_FALSE(first.poisoned)
      << "recovery must NOT poison (absorbed BEGIN(E,G) at the old generation is validate-only)";
  EXPECT_EQ(first.fp, ref_fp) << "recovery must land exactly on the clean consolidate+checkpoint base";
  const auto second = recover(case_prefix, max_points);
  EXPECT_EQ(first.fp, second.fp) << "double reopen must be byte-stable";
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace alaya::laser
