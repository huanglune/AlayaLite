// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Functional (non-crash) durability tests for the QGUpdater op-WAL (enable_wal):
// fresh lineage, clean reopen, durability of published inserts/tombstones across
// a close/reopen with no checkpoint, double-reopen stability, and lineage/scope
// guards. The kill-point crash matrix lives in test_segment_op_wal_crash.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
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

// ---------------------------------------------------------------------------
// W3: label transaction functional durability (commit_physical_bundle + replay
// staging/promotion). The kill-point crash matrix lives in the crash test.
// ---------------------------------------------------------------------------

std::vector<float> labeled_vecs(size_t n, uint32_t seed) { return waltest::make_data(n, kDim, seed); }

// Two bundles committed WITHOUT a checkpoint recover purely by op-WAL replay:
// committed advances and both bundles' explicit labels are promoted.
TEST(QgUpdaterWalLabels, MultiBundlePureReplayRecovers) {
  const auto dir = scratch_dir("multi_bundle");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 101);
  const auto v1 = labeled_vecs(3, 0x1001);
  const auto v2 = labeled_vecs(4, 0x2002);
  const std::vector<uint64_t> l1 = {50000, 50001, 50002};
  const std::vector<uint64_t> l2 = {60000, 60001, 60002, 60003};
  {
    Session s(base.prefix, kBaseN + 64);
    auto r1 = s.upd->commit_physical_bundle(1, 1, v1.data(), l1.data(), l1.size());
    auto r2 = s.upd->commit_physical_bundle(2, 2, v2.data(), l2.data(), l2.size());
    EXPECT_EQ(r1.first, static_cast<PID>(kBaseN));
    EXPECT_EQ(r2.first, static_cast<PID>(kBaseN + 3));
    EXPECT_EQ(s.upd->num_points(), kBaseN + 7);  // NO checkpoint
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN + 7) << "both bundles recovered via replay";
    const auto snap = s.upd->label_snapshot();
    ASSERT_TRUE(snap != nullptr);
    ASSERT_NE(snap->find(static_cast<PID>(kBaseN + 0)), nullptr);
    EXPECT_EQ(*snap->find(static_cast<PID>(kBaseN + 0)), 50000U);
    EXPECT_EQ(*snap->find(static_cast<PID>(kBaseN + 3)), 60000U);
    EXPECT_EQ(*snap->find(static_cast<PID>(kBaseN + 6)), 60003U);
    EXPECT_EQ(s.upd->last_committed_txid(), 2U);
  }
  std::filesystem::remove_all(dir);
}

// A torn bundle (crash before the kind=8 fsync => the whole buffered bundle is
// lost) does not consume its txid: a retry with the SAME txid is accepted.
TEST(QgUpdaterWalLabels, TornBundleRetryWithSameTxidIsAccepted) {
  const auto dir = scratch_dir("torn_retry");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 202);
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  const auto v = labeled_vecs(2, 0x3003);
  const std::vector<uint64_t> labels = {40000, 40001};

  std::uintmax_t marker_len = 0;
  {
    Session s(base.prefix, kBaseN + 64);  // fresh-enable: WAL is a single flip marker
    marker_len = std::filesystem::file_size(wal_path);
  }
  {
    Session s(base.prefix, kBaseN + 64);
    (void)s.upd->commit_physical_bundle(7, 3, v.data(), labels.data(), labels.size());
  }
  // Simulate the torn bundle: the buffered row_patches + kind=7 binds + kind=8 all
  // vanish (kind=8 fsync never completed). Truncate back to the durable marker.
  std::filesystem::resize_file(wal_path, marker_len);
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN) << "torn bundle is not committed";
    EXPECT_EQ(s.upd->last_committed_txid(), 0U) << "torn bundle does not consume its txid";
    // Retry the SAME txid: legal and must succeed.
    auto r = s.upd->commit_physical_bundle(7, 3, v.data(), labels.data(), labels.size());
    EXPECT_EQ(r.first, static_cast<PID>(kBaseN));
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN + 2) << "retry is durable";
    EXPECT_EQ(s.upd->last_committed_txid(), 7U);
  }
  std::filesystem::remove_all(dir);
}

// B-03: the committed txid + applied op-id persist through a checkpoint, so after
// reopen a stale/duplicate txid or a regressed applied op-id is rejected (throws,
// not poison), and a strictly-newer txid with a non-regressed op-id is accepted.
TEST(QgUpdaterWalLabels, CheckpointedTxidAndAppliedOpGatePreconditions) {
  const auto dir = scratch_dir("ckpt_txid");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 303);
  const auto v = labeled_vecs(2, 0x4004);
  const std::vector<uint64_t> labels = {30000, 30001};
  {
    Session s(base.prefix, kBaseN + 64);
    (void)s.upd->commit_physical_bundle(10, 5, v.data(), labels.data(), labels.size());
    s.upd->checkpoint();  // persist last_committed_txid=10, applied=5 into the superblock
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->last_committed_txid(), 10U) << "txid survives checkpoint (B-03)";
    EXPECT_EQ(s.upd->applied_collection_op_id(), 5U);
    const auto v2 = labeled_vecs(1, 0x5005);
    const std::vector<uint64_t> l2 = {31000};
    EXPECT_THROW((void)s.upd->commit_physical_bundle(10, 6, v2.data(), l2.data(), 1),
                 std::invalid_argument)
        << "duplicate txid rejected";
    EXPECT_THROW((void)s.upd->commit_physical_bundle(9, 6, v2.data(), l2.data(), 1),
                 std::invalid_argument)
        << "stale txid rejected";
    EXPECT_THROW((void)s.upd->commit_physical_bundle(11, 4, v2.data(), l2.data(), 1),
                 std::invalid_argument)
        << "regressed applied op-id rejected";
    auto r = s.upd->commit_physical_bundle(11, 5, v2.data(), l2.data(), 1);
    EXPECT_EQ(r.first, static_cast<PID>(kBaseN + 2)) << "strictly-newer txid accepted";
  }
  std::filesystem::remove_all(dir);
}

// A same-txid retry after a torn bundle whose earlier binds survived in the OS page
// cache (process crash on a large bundle) recovers idempotently: the orphan kind=7
// binds are de-duped against the retry's identical binds instead of over-staging
// into a false count mismatch. (The conflicting-label branch is covered by the B-04
// duplicate_row_op_id case.)
TEST(QgUpdaterWalLabels, OrphanBindsFromTornBundleAreDedupedOnRetry) {
  const auto dir = scratch_dir("orphan_dedup");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 505);
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  uint64_t uid = 0;
  {
    Session s(base.prefix, kBaseN + 64);
    uid = s.upd->segment_uid();
  }
  const auto v = labeled_vecs(3, 0x6006);
  const std::vector<uint64_t> labels = {45000, 45001, 45002};
  // Plant an orphan kind=7 for tx=1 (row 0, pid=base, label identical to the retry).
  {
    alaya::wal::WalFile wal(wal_path);
    wal.append(kSegmentOpRecordType, 0, 500, 1,
               encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(kBaseN), 0, labels[0]),
               alaya::wal::WalFile::Sync::fsync);
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN) << "the orphan bind alone is not committed";
    auto r = s.upd->commit_physical_bundle(1, 1, v.data(), labels.data(), labels.size());
    EXPECT_EQ(r.first, static_cast<PID>(kBaseN));
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN + 3) << "retry recovers despite the orphan bind";
    const auto snap = s.upd->label_snapshot();
    ASSERT_NE(snap->find(static_cast<PID>(kBaseN)), nullptr);
    EXPECT_EQ(*snap->find(static_cast<PID>(kBaseN)), 45000U);
    EXPECT_EQ(s.upd->last_committed_txid(), 1U);
  }
  std::filesystem::remove_all(dir);
}

// --- B-04 divergence family: malformed kind=7/8 WALs must poison on replay. ---
using FrameList = std::vector<std::vector<std::byte>>;
struct BadBundle {
  const char *name;
  std::function<void(uint64_t uid, size_t base_n, FrameList &frames, std::vector<uint64_t> &bids)>
      build;
};
class QgUpdaterWalB04 : public ::testing::TestWithParam<BadBundle> {};

TEST_P(QgUpdaterWalB04, MalformedBundlePoisonsReplay) {
  const auto dir = scratch_dir(std::string("b04_") + GetParam().name);
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 404);
  const std::string wal_path = base.prefix + waltest::index_suffix() + ".opwal";
  uint64_t uid = 0;
  {
    Session s(base.prefix, kBaseN + 64);  // fresh-enable; WAL == single marker
    uid = s.upd->segment_uid();
  }
  FrameList frames;
  std::vector<uint64_t> bids;
  GetParam().build(uid, kBaseN, frames, bids);
  ASSERT_EQ(frames.size(), bids.size());
  {
    alaya::wal::WalFile wal(wal_path);  // append after the marker
    for (size_t i = 0; i < frames.size(); ++i) {
      wal.append(kSegmentOpRecordType, 0, 200 + i, bids[i], frames[i],
                 alaya::wal::WalFile::Sync::fsync);
    }
  }
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.max_points = kBaseN + 64;
  EXPECT_THROW((void)std::make_unique<QGUpdater>(qg, params), std::exception)
      << GetParam().name << " must poison on replay";
  std::filesystem::remove_all(dir);
}

INSTANTIATE_TEST_SUITE_P(
    Divergence, QgUpdaterWalB04,
    ::testing::Values(
        BadBundle{"watermark_count_mismatch",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 2, 1, 1));  // new != old+count
                    bids = {1, 1};
                  }},
        BadBundle{"staged_count_mismatch",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 2, 2, 1));  // count 2, 1 staged
                    bids = {1, 1};
                  }},
        BadBundle{"duplicate_row_op_id",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(
                        encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b + 1), 0, 9001));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 2, 2, 1));
                    bids = {1, 1, 1};
                  }},
        BadBundle{"pid_out_of_range",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(
                        encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b + 5), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 1, 1, 1));  // pid b+5 >= new b+1
                    bids = {1, 1};
                  }},
        BadBundle{"batch_id_mismatch",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 1, 1, 1));
                    bids = {1, 2};  // kind=8 frame batch_id != payload tx_id
                  }},
        BadBundle{"nonzero_pid_generation",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 1, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 1, 1, 1));
                    bids = {1, 1};
                  }},
        BadBundle{"second_publish_same_txid",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 1, 1, 1));  // promotes
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 2, 1, 1));  // re-publish tx 1
                    bids = {1, 1, 1};
                  }},
        BadBundle{"applied_op_regression",
                  [](uint64_t uid, size_t b, FrameList &f, std::vector<uint64_t> &bids) {
                    f.push_back(encode_label_bind(uid, 1, 1, 0, static_cast<uint32_t>(b), 0, 9000));
                    f.push_back(encode_tx_publish(uid, 1, 1, b + 1, 1, 5));  // applied 5
                    f.push_back(
                        encode_label_bind(uid, 1, 2, 0, static_cast<uint32_t>(b + 1), 0, 9001));
                    f.push_back(encode_tx_publish(uid, 1, 2, b + 2, 1, 3));  // applied 3 < 5
                    bids = {1, 1, 2, 2};
                  }}),
    [](const ::testing::TestParamInfo<BadBundle> &info) { return info.param.name; });

// codex BLOCKER (staged backlinks): a bundle with deferred reverse edges poisons
// BEFORE commit, so kind=8 never commits a row whose only routing edges live in RAM
// and would vanish on a crash (leaving the row permanently unreachable).
TEST(QgUpdaterWalLabels, StagedBacklinksPoisonBundle) {
  const auto dir = scratch_dir("staged_poison");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 606);
  QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
  qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
  qg.set_params(64, 1, 1);
  UpdateParams params;
  params.enable_wal = true;
  params.ef_insert = 64;
  params.max_points = kBaseN + 64;
  params.stage_backlinks = true;  // deferred reverse edges: illegal under enable_wal
  QGUpdater upd(qg, params);
  const auto v = labeled_vecs(2, 0x7007);
  const std::vector<uint64_t> labels = {33000, 33001};
  EXPECT_THROW((void)upd.commit_physical_bundle(1, 1, v.data(), labels.data(), 2),
               std::runtime_error)
      << "staged backlinks must poison the bundle before commit";
  std::filesystem::remove_all(dir);
}

// codex BLOCKER (internal failure mid-bundle): an exception that does not itself
// poison (simulated via a throwing failpoint) still poisons the handle, so a later
// operation cannot kind=5-publish over the allocation gap and commit the failed
// rows as identity-labeled. On reopen the failed bundle is not committed.
TEST(QgUpdaterWalLabels, InternalFailureMidBundlePoisonsAndDoesNotLeak) {
  const auto dir = scratch_dir("mid_fail");
  std::filesystem::remove_all(dir);
  auto base = WalTinyIndex::build(dir, kBaseN, 707);
  const auto v = labeled_vecs(3, 0x8008);
  const std::vector<uint64_t> labels = {34000, 34001, 34002};
  {
    QuantizedGraph qg(kBaseN, kDeg, kDim, kDim);
    qg.load_disk_index(base.prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = true;
    params.ef_insert = 64;
    params.max_points = kBaseN + 64;
    params.failpoint_hook = [](SegmentOpFailPoint fp) {
      if (fp == SegmentOpFailPoint::after_label_bind_append) {
        throw std::runtime_error("injected mid-bundle failure");  // e.g. std::bad_alloc
      }
    };
    QGUpdater upd(qg, params);
    EXPECT_THROW((void)upd.commit_physical_bundle(5, 3, v.data(), labels.data(), 3),
                 std::runtime_error);
    EXPECT_THROW((void)upd.publish(upd.allocated_points()), std::runtime_error)
        << "the handle must be poisoned after a mid-bundle failure";
  }
  {
    Session s(base.prefix, kBaseN + 64);
    EXPECT_EQ(s.upd->num_points(), kBaseN) << "the failed bundle must not be committed";
    EXPECT_EQ(s.upd->last_committed_txid(), 0U) << "no identity-labeled rows leaked";
  }
  std::filesystem::remove_all(dir);
}

}  // namespace
}  // namespace alaya::laser
