// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// MutableLaserSegment: the durable, single-writer, mutable segment handle over a
// sealed LASER v2 index. Covers open -> add/search -> tombstone -> flush/
// checkpoint -> close -> reopen (recovery) convergence in both residency modes,
// the label mapping (sidecar below the base count, identity above), and the
// single-writer flock.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
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

constexpr size_t kBaseN = 200;
constexpr uint64_t kLabelBase = 5000;

// Build a valid LASER v2 seg_dir (index + sidecars + manifest.txt + ids sidecar)
// rooted at `dir`. Row i carries label kLabelBase + i.
WalTinyIndex build_segment(const std::filesystem::path &dir, uint32_t seed) {
  auto base = WalTinyIndex::build(dir, kBaseN, seed);  // writes dir/wal_base*_R64_MD64.index
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
  manifest.x_extras["x_laser_index_file"] =
      "wal_base_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
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
         ("mutable_seg_" + name + "_" + std::to_string(::getpid()));
}

class MutableLaserSegmentResidency : public ::testing::TestWithParam<ResidencyMode> {};

TEST_P(MutableLaserSegmentResidency, AddTombstoneFlushReopenConverges) {
  const auto dir = scratch(GetParam() == ResidencyMode::kResidentArena ? "arena" : "paged");
  std::filesystem::remove_all(dir);
  auto base = build_segment(dir, 4242);
  const auto newvec = waltest::make_data(1, kDim, 0xBEEF);
  laser::PID added_pid = 0;

  {
    laser::UpdateParams params;
    params.ef_insert = 64;
    params.max_points = kBaseN + 32;
    MutableLaserSegment seg(dir, params, GetParam());
    EXPECT_EQ(seg.size(), kBaseN);
    EXPECT_EQ(seg.dim(), kDim);

    // A base row's label comes from the sidecar (kLabelBase + row).
    const auto base_query = waltest::make_data(1, kDim, 4242);  // row 0 of the base data
    // (row 0 vector is data[0..dim); reuse the fixture's own generator seed.)
    auto base_hits = seg.search(base.data.data(), DiskSearchOptions{/*top_k=*/5});
    ASSERT_FALSE(base_hits.empty());
    EXPECT_EQ(base_hits[0].label, kLabelBase + 0) << "nearest base row must map through the sidecar";

    // Append a new row: its label is its PID (identity), above the base count.
    added_pid = seg.add(newvec.data());
    EXPECT_EQ(added_pid, static_cast<laser::PID>(kBaseN));
    EXPECT_EQ(seg.size(), kBaseN + 1);
    auto new_hits = seg.search(newvec.data(), DiskSearchOptions{/*top_k=*/5});
    ASSERT_FALSE(new_hits.empty());
    EXPECT_EQ(new_hits[0].label, static_cast<uint64_t>(added_pid)) << "appended row uses identity";

    // Tombstone the appended row: it disappears from results.
    seg.tombstone(added_pid);
    auto after = seg.search(newvec.data(), DiskSearchOptions{/*top_k=*/5});
    for (const auto &hit : after) {
      EXPECT_NE(hit.label, static_cast<uint64_t>(added_pid));
    }
    seg.checkpoint();
  }

  // Reopen (recovery): the appended-then-tombstoned row stays gone; size holds.
  {
    laser::UpdateParams params;
    params.max_points = kBaseN + 32;
    MutableLaserSegment seg(dir, params, GetParam());
    EXPECT_EQ(seg.size(), kBaseN + 1);
    auto hits = seg.search(newvec.data(), DiskSearchOptions{/*top_k=*/5});
    for (const auto &hit : hits) {
      EXPECT_NE(hit.label, static_cast<uint64_t>(added_pid)) << "tombstone must survive reopen";
    }
    // A fresh base query still maps through the sidecar.
    auto base_hits = seg.search(base.data.data(), DiskSearchOptions{/*top_k=*/5});
    ASSERT_FALSE(base_hits.empty());
    EXPECT_EQ(base_hits[0].label, kLabelBase + 0);
  }
  std::filesystem::remove_all(dir);
}

INSTANTIATE_TEST_SUITE_P(Residency, MutableLaserSegmentResidency,
                         ::testing::Values(ResidencyMode::kPagedPool,
                                           ResidencyMode::kResidentArena),
                         [](const ::testing::TestParamInfo<ResidencyMode> &info) {
                           return info.param == ResidencyMode::kResidentArena ? "arena" : "paged";
                         });

TEST(MutableLaserSegment, SecondWriterIsRejectedByFlock) {
  const auto dir = scratch("flock");
  std::filesystem::remove_all(dir);
  build_segment(dir, 7);
  laser::UpdateParams params;
  params.max_points = kBaseN + 32;
  auto first = std::make_unique<MutableLaserSegment>(dir, params, ResidencyMode::kPagedPool);
  EXPECT_THROW((void)std::make_unique<MutableLaserSegment>(dir, params, ResidencyMode::kPagedPool),
               std::runtime_error)
      << "a second writer must fail the single-writer lease";
  first.reset();  // releasing the lease lets a new writer in
  EXPECT_NO_THROW(
      (void)std::make_unique<MutableLaserSegment>(dir, params, ResidencyMode::kPagedPool));
  std::filesystem::remove_all(dir);
}

TEST(MutableLaserSegment, BatchAddPublishesUnderOneWatermark) {
  const auto dir = scratch("batch");
  std::filesystem::remove_all(dir);
  build_segment(dir, 99);
  const size_t n = 10;
  const auto batch = waltest::make_data(n, kDim, 0xF00D);
  laser::UpdateParams params;
  params.ef_insert = 64;
  params.max_points = kBaseN + 32;
  MutableLaserSegment seg(dir, params, ResidencyMode::kPagedPool);
  const auto base_pid = seg.add_batch(batch.data(), n);
  EXPECT_EQ(base_pid, static_cast<laser::PID>(kBaseN));
  EXPECT_EQ(seg.size(), kBaseN + n);
  seg.checkpoint();
  std::filesystem::remove_all(dir);
}

// W0: the single-writer handle mutex serializes add/checkpoint against each other
// while search stays lock-free (poison read gate only). Run concurrent searches
// against a stream of add+checkpoint rounds and assert a race-free convergence.
TEST(MutableLaserSegment, ConcurrentWritesAndSearchesAreRaceFree) {
  const auto dir = scratch("concurrent");
  std::filesystem::remove_all(dir);
  build_segment(dir, 2024);
  constexpr size_t kRounds = 8;
  constexpr size_t kPerRound = 8;
  const auto adds = waltest::make_data(kRounds * kPerRound, kDim, 0xC0DE);
  const auto query = waltest::make_data(1, kDim, 0x5EED);

  std::atomic<size_t> searches{0};
  {
    laser::UpdateParams params;
    params.ef_insert = 64;
    params.max_points = kBaseN + 256;
    MutableLaserSegment seg(dir, params, ResidencyMode::kPagedPool);

    std::atomic<bool> stop{false};
    std::thread reader([&] {
      while (!stop.load(std::memory_order_acquire)) {
        auto hits = seg.search(query.data(), DiskSearchOptions{/*top_k=*/5});
        EXPECT_LE(hits.size(), 5U);
        searches.fetch_add(1, std::memory_order_relaxed);
      }
    });

    for (size_t r = 0; r < kRounds; ++r) {
      for (size_t i = 0; i < kPerRound; ++i) {
        (void)seg.add(adds.data() + (r * kPerRound + i) * kDim);
      }
      seg.checkpoint();  // serialized against add by the handle mutex
    }
    stop.store(true, std::memory_order_release);
    reader.join();

    EXPECT_EQ(seg.size(), kBaseN + kRounds * kPerRound);
  }  // seg destroyed here -> writer lease released before the reopen below
  EXPECT_GT(searches.load(), 0U);

  // Reopen (recovery) converges to the same committed count.
  {
    laser::UpdateParams reopen;
    reopen.max_points = kBaseN + 256;
    MutableLaserSegment seg2(dir, reopen, ResidencyMode::kPagedPool);
    EXPECT_EQ(seg2.size(), kBaseN + kRounds * kPerRound);
  }
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::disk
