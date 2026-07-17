// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Functional (non-crash) durability tests for the QGUpdater op-WAL (enable_wal):
// fresh lineage, clean reopen, durability of published inserts/tombstones across
// a close/reopen with no checkpoint, double-reopen stability, and lineage/scope
// guards. The kill-point crash matrix lives in test_segment_op_wal_crash.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "qg_wal_test_support.hpp"

namespace alaya::laser {
namespace {

using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 256;
constexpr size_t kInsert = 40;
constexpr size_t kTomb = 8;

struct Session {
  QuantizedGraph qg;
  std::unique_ptr<QGUpdater> upd;
  explicit Session(const std::string &prefix, size_t max_points)
      : qg(kBaseN, kDeg, kDim, kDim) {
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.ef_insert = 64;
    params.max_points = max_points;
    params.backlink_mode = UpdateParams::Backlink::kAlphaEvict;
    upd = std::make_unique<QGUpdater>(qg, params);
  }
};

std::filesystem::path scratch_dir(const std::string &name) {
  return std::filesystem::temp_directory_path() /
         ("qg_wal_" + name + "_" + std::to_string(::getpid()));
}

bool row_is_live(QGUpdater &upd, PID id) {
  const auto flags = upd.trailer(id).flags;
  return (flags & (kQGRowTombstone | kQGRowFree)) == 0;
}

TEST(QgUpdaterWal, FreshEnableStampsLineageAndCreatesWal) {
  const auto dir = scratch_dir("fresh");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 11);
  const uint64_t base_gen_before = [&] {
    Session s(base.prefix, kBaseN + 64);
    return s.upd->generation();
  }();
  EXPECT_GT(base_gen_before, 0U);
  EXPECT_TRUE(std::filesystem::exists(base.prefix + waltest::index_suffix() + ".opwal"));
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterWal, PublishedInsertsAndTombstonesSurviveReopenAndCheckpoint) {
  const auto dir = scratch_dir("survive");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 22);
  const auto inserted = waltest::make_data(kInsert, kDim, 777);
  std::vector<PID> ids;

  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    auto &upd = *s.upd;
    for (size_t i = 0; i < kInsert; ++i) {
      ids.push_back(upd.allocate_and_insert(inserted.data() + i * kDim));
    }
    upd.publish(upd.allocated_points());
    for (size_t i = 0; i < kTomb; ++i) upd.tombstone(ids[i]);
    upd.checkpoint();  // forces the WAL and makes the whole state durable
    EXPECT_EQ(upd.num_points(), kBaseN + kInsert);
    EXPECT_EQ(upd.live_count(), kBaseN + kInsert - kTomb);
  }

  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    auto &upd = *s.upd;
    EXPECT_EQ(upd.num_points(), kBaseN + kInsert);
    EXPECT_EQ(upd.live_count(), kBaseN + kInsert - kTomb);
    for (size_t i = 0; i < kTomb; ++i) {
      EXPECT_FALSE(row_is_live(upd, ids[i])) << "tombstoned row " << ids[i] << " must stay dark";
    }
    for (size_t i = kTomb; i < kInsert; ++i) {
      EXPECT_TRUE(row_is_live(upd, ids[i])) << "published insert " << ids[i] << " must be live";
    }
    // Best-effort reachability: a query equal to an inserted vector finds it.
    size_t found = 0;
    for (size_t i = kTomb; i < kInsert; ++i) {
      auto res = upd.search(inserted.data() + i * kDim, 10, 128);
      if (std::find(res.begin(), res.end(), ids[i]) != res.end()) ++found;
    }
    EXPECT_GE(found, (kInsert - kTomb) * 8 / 10) << "recovered inserts should be searchable";
    // Tombstoned rows never appear in results.
    for (size_t i = 0; i < kTomb; ++i) {
      auto res = upd.search(inserted.data() + i * kDim, 10, 128);
      EXPECT_EQ(std::count(res.begin(), res.end(), ids[i]), 0);
    }
  }
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterWal, PublishedInsertsSurviveReopenWithoutCheckpoint) {
  const auto dir = scratch_dir("nockpt");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 33);
  const auto inserted = waltest::make_data(kInsert, kDim, 999);
  std::vector<PID> ids;

  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    auto &upd = *s.upd;
    for (size_t i = 0; i < kInsert; ++i) {
      ids.push_back(upd.allocate_and_insert(inserted.data() + i * kDim));
    }
    upd.publish(upd.allocated_points());  // durable via the WAL, NO checkpoint
  }

  // Reopen: the index pages were never checkpointed, so recovery must redo the
  // op-WAL to reconstruct the published rows.
  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    auto &upd = *s.upd;
    EXPECT_EQ(upd.num_points(), kBaseN + kInsert);
    for (size_t i = 0; i < kInsert; ++i) {
      EXPECT_TRUE(row_is_live(upd, ids[i])) << "insert " << ids[i] << " lost after WAL recovery";
    }
  }
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterWal, DoubleReopenIsByteAndStateStable) {
  const auto dir = scratch_dir("double");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 44);
  const auto inserted = waltest::make_data(kInsert, kDim, 1234);
  const std::string index_path = base.prefix + waltest::index_suffix();

  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    for (size_t i = 0; i < kInsert; ++i) s.upd->allocate_and_insert(inserted.data() + i * kDim);
    s.upd->publish(s.upd->allocated_points());
  }

  auto snapshot = [&]() {
    Session s(base.prefix, kBaseN + kInsert + 16);
    // recovery has just rewritten the file; hash it.
    std::ifstream in(index_path, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return std::make_pair(s.upd->num_points(),
                          std::hash<std::string_view>{}(std::string_view(bytes.data(),
                                                                         bytes.size())));
  };
  const auto first = snapshot();
  const auto second = snapshot();
  EXPECT_EQ(first.first, second.first);
  EXPECT_EQ(first.second, second.second) << "double replay must be byte-stable";
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterWal, ForeignLineageIsRejected) {
  const auto dir = scratch_dir("lineage");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 55);
  {
    Session s(base.prefix, kBaseN + kInsert + 16);
    s.upd->allocate_and_insert(waltest::make_data(1, kDim, 1).data());
    s.upd->publish(s.upd->allocated_points());
  }
  // Overwrite the .opwal with a frame stamped for a different segment uid.
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  {
    alaya::wal::WalFile wal(wal_path);  // truncates/reopens; append a foreign publish
    auto foreign = encode_publish(/*segment_id=*/0xDEADBEEFCAFEF00DULL, /*gen=*/1, /*watermark=*/1);
    wal.append(kSegmentOpRecordType, 0, 1, 0, foreign, alaya::wal::WalFile::Sync::fsync);
  }
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.max_points = kBaseN + kInsert + 16;
  EXPECT_THROW((void)std::make_unique<QGUpdater>(qg, params), std::exception);
  std::filesystem::remove_all(dir);
}

TEST(QgUpdaterWal, ReclaimAndConsolidateAreRejectedUnderWal) {
  const auto dir = scratch_dir("scope");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 66);
  Session s(base.prefix, kBaseN + kInsert + 16);
  EXPECT_THROW(s.upd->consolidate(1), std::logic_error);
  EXPECT_THROW(s.upd->garden(1, GardenParams{}), std::logic_error);
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::laser
