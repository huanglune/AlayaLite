// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// W2 step 7: MutableLaserSegment token-ization + tombstone ABA + reverse-map
// full-token erase + base-region reuse (design 3.3 / codex B.7 / JC-17). A base
// row is tombstoned + reclaimed, then a canonical bundle reuses its PID with a
// fresh generation; the label -> (pid, generation) token drives an ABA-safe
// tombstone that never kills a newer incarnation, and a reused base binding
// survives reopen (base-region shadowing guard permits generation>0 only).

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "index/disk/mutable_laser_segment.hpp"
#include "index/disk/segment_manifest.hpp"
#include "qg_wal_test_support.hpp"

namespace alaya::disk {
namespace {

namespace waltest = alaya::laser::waltest;
using laser::ResidencyMode;
using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 64;
constexpr uint64_t kLabelBase = 5000;

WalTinyIndex build_segment(const std::filesystem::path &dir, uint32_t seed) {
  auto base = WalTinyIndex::build(dir, kBaseN, seed);
  SegmentManifest manifest;
  manifest.segment_id = "seg_00000001";
  manifest.index_type = DiskIndexType::Laser;
  manifest.metric = core::Metric::l2;
  manifest.dim = kDim;
  manifest.count = kBaseN;
  manifest.ids_file = "ids.u64.bin";
  manifest.vectors_file = "";
  manifest.x_extras["x_laser_filename_prefix"] = "wal_base";
  manifest.x_extras["x_R"] = std::to_string(kDeg);
  manifest.x_extras["x_main_dim"] = std::to_string(kDim);
  manifest.save(dir / "manifest.txt");
  std::ofstream ids(dir / "ids.u64.bin", std::ios::binary);
  for (uint64_t i = 0; i < kBaseN; ++i) {
    const uint64_t label = kLabelBase + i;
    ids.write(reinterpret_cast<const char *>(&label), sizeof(label));
  }
  return base;
}

std::filesystem::path scratch(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("mutable_seg_reuse_" + name + "_" + std::to_string(::getpid()));
}

laser::UpdateParams reuse_params() {
  laser::UpdateParams params;
  params.enable_pid_reuse = true;  // ctor forces enable_wal=true
  params.ef_insert = 64;
  params.max_points = 4 * kBaseN;
  return params;
}

// The nearest label to `vec` in a top-k search (kPidMax-ish sentinel if empty).
uint64_t nearest_label(MutableLaserSegment &seg, const float *vec) {
  DiskSearchOptions opts;
  opts.top_k = 5;
  opts.ef = 64;
  const auto hits = seg.search(vec, opts);
  return hits.empty() ? ~0ULL : hits[0].label;
}

// A base PID whose label resolves to a live token before reuse; after tombstone +
// reclaim + a reuse bundle, the SAME PID carries a generation-1 token under a new
// label, and the old label no longer resolves. Survives reopen (base shadowing OK).
TEST(MutableLaserSegmentReuse, ReusesBasePidWithGenerationTokenAcrossReopen) {
  const auto dir = scratch("basereuse");
  std::filesystem::remove_all(dir);
  auto base = build_segment(dir, 4242);
  const auto newvec = waltest::make_data(1, kDim, 0x51EED);
  const uint64_t victim_label = kLabelBase + 7;  // base PID 7
  const uint64_t new_label = 99999;

  {
    MutableLaserSegment seg(dir, reuse_params(), ResidencyMode::kPagedPool);
    // The victim base label resolves to a gen-0 token at PID 7.
    auto tok0 = seg.token_for_label(victim_label);
    ASSERT_TRUE(tok0.has_value());
    EXPECT_EQ(tok0->pid, static_cast<laser::PID>(7));
    EXPECT_EQ(tok0->pid_generation, 0U);

    // Tombstone by token, reclaim, then commit a canonical reuse bundle (label 99999).
    seg.tombstone(*tok0);
    EXPECT_FALSE(seg.token_for_label(victim_label).has_value());
    seg.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);
    ASSERT_GE(seg.free_count(), 1U);

    const uint64_t labels[1] = {new_label};
    (void)seg.commit_physical_bundle(1, 1, newvec.data(), labels, 1);
    EXPECT_TRUE(seg.pid_generation_activated());

    // The reused PID now carries a generation-1 token under the NEW label; the old
    // label is gone; the new label is searchable and resolves to the reused PID 7.
    auto tok1 = seg.token_for_label(new_label);
    ASSERT_TRUE(tok1.has_value());
    EXPECT_EQ(tok1->pid, static_cast<laser::PID>(7)) << "reused the freed base PID";
    EXPECT_EQ(tok1->pid_generation, 1U) << "reuse bumps the generation to 1";
    EXPECT_FALSE(seg.token_for_label(victim_label).has_value());
    EXPECT_EQ(nearest_label(seg, newvec.data()), new_label);
    seg.checkpoint();
  }
  // Reopen: the generation-1 base-region binding survives the shadowing guard.
  {
    MutableLaserSegment seg(dir, reuse_params(), ResidencyMode::kPagedPool);
    auto tok1 = seg.token_for_label(new_label);
    ASSERT_TRUE(tok1.has_value());
    EXPECT_EQ(tok1->pid, static_cast<laser::PID>(7));
    EXPECT_EQ(tok1->pid_generation, 1U);
    EXPECT_FALSE(seg.token_for_label(victim_label).has_value());
    EXPECT_EQ(nearest_label(seg, newvec.data()), new_label);
  }
  std::filesystem::remove_all(dir);
}

// ABA: a token captured before the PID was reused is STALE; tombstoning it is an
// idempotent no-op that never kills the new incarnation. A token from the future
// (generation ahead of the durable one) is corruption.
TEST(MutableLaserSegmentReuse, StaleTokenTombstoneIsNoOpFutureTokenThrows) {
  const auto dir = scratch("aba");
  std::filesystem::remove_all(dir);
  auto base = build_segment(dir, 313);
  const auto v = waltest::make_data(2, kDim, 0xA1);

  MutableLaserSegment seg(dir, reuse_params(), ResidencyMode::kPagedPool);
  const uint64_t victim = kLabelBase + 3;  // base PID 3
  auto stale = seg.token_for_label(victim);
  ASSERT_TRUE(stale.has_value());
  EXPECT_EQ(stale->pid_generation, 0U);

  // Free PID 3 and reuse it (generation 1, label 70001).
  seg.tombstone(*stale);
  seg.consolidate(1, 0, true, false);
  const uint64_t l1[1] = {70001};
  (void)seg.commit_physical_bundle(1, 1, v.data(), l1, 1);
  auto fresh = seg.token_for_label(70001);
  ASSERT_TRUE(fresh.has_value());
  ASSERT_EQ(fresh->pid, stale->pid);
  ASSERT_EQ(fresh->pid_generation, 1U);

  // Tombstoning the STALE gen-0 token must NOT kill the gen-1 incarnation.
  seg.tombstone(*stale);
  auto after = seg.token_for_label(70001);
  ASSERT_TRUE(after.has_value()) << "stale-token tombstone wrongly erased the new incarnation";
  EXPECT_EQ(after->pid_generation, 1U);
  EXPECT_EQ(nearest_label(seg, v.data()), 70001U) << "reused row still searchable after stale tombstone";

  // A token from the future (generation ahead of durable) is corruption.
  laser::PidToken future{fresh->pid, fresh->pid_generation + 5};
  EXPECT_THROW(seg.tombstone(future), std::exception);
  std::filesystem::remove_all(dir);
}

// R9 (leg-7): the label -> (pid, generation) reverse map is IN-MEMORY, rebuilt from the durable
// label snapshot on every open. A SIGKILL after the canonical bundle's kind=8 is durable but
// BEFORE the segment updates its reverse map (label_to_pid_.insert_or_assign, which runs only
// after the QGUpdater commit returns) must recover a reverse map consistent with the durable
// bindings: the new label resolves to the reused PID's fresh incarnation, the old label is gone,
// and the reused row is searchable. This proves the reverse map is DERIVED, never a separate
// durable structure a crash could desync from the committed bindings.
TEST(MutableLaserSegmentReuse, ReverseMapRecoversAfterSigkillBeforeAdapterUpdate) {
  const auto dir = scratch("r9_reverse_map");
  std::filesystem::remove_all(dir);
  auto base = build_segment(dir, 909);
  const auto newvec = waltest::make_data(1, kDim, 0x9A9);
  const uint64_t victim_label = kLabelBase + 5;  // base PID 5
  const uint64_t new_label = 88888;

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      laser::UpdateParams params = reuse_params();
      // Arm the SIGKILL at the canonical kind=8 fsync (R6): the bundle is durable, but the
      // segment's reverse-map update (after the QGUpdater commit returns) never runs.
      params.failpoint_hook = [](laser::SegmentOpFailPoint fp) {
        if (fp == laser::SegmentOpFailPoint::after_tx_publish_fsync) {
          ::kill(::getpid(), SIGKILL);
          ::_exit(99);
        }
      };
      MutableLaserSegment seg(dir, params, ResidencyMode::kPagedPool);
      auto tok = seg.token_for_label(victim_label);
      if (!tok.has_value()) {
        ::_exit(71);
      }
      seg.tombstone(*tok);
      seg.consolidate(1, /*r_target=*/0, /*reclaim=*/true, /*bloom=*/false);
      const uint64_t labels[1] = {new_label};
      (void)seg.commit_physical_bundle(1, 1, newvec.data(), labels, 1);  // SIGKILL at kind=8 fsync
    } catch (...) {
      ::_exit(70);
    }
    ::_exit(0);
  }
  int status = 0;
  ASSERT_EQ(::waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL)
      << "child must die of SIGKILL at the durable kind=8 (status=" << status << ")";

  // Reopen: rebuild_reverse_index derives the reverse map from the durable (rolled-forward) snapshot.
  MutableLaserSegment seg(dir, reuse_params(), ResidencyMode::kPagedPool);
  auto tok = seg.token_for_label(new_label);
  ASSERT_TRUE(tok.has_value()) << "the reverse map must recover the new label -> reused-PID token";
  EXPECT_EQ(tok->pid, static_cast<laser::PID>(5)) << "reused the freed base PID";
  EXPECT_EQ(tok->pid_generation, 1U) << "reuse bumped the generation to 1";
  EXPECT_FALSE(seg.token_for_label(victim_label).has_value()) << "the old label is gone";
  EXPECT_EQ(nearest_label(seg, newvec.data()), new_label) << "the reused row is searchable";
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::disk
