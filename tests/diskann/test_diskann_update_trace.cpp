// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "sift_update_trace.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unordered_set>
#include <vector>

namespace {

struct TraceRound {
  std::vector<uint32_t> deletes;
  std::vector<uint32_t> inserts;
};

TraceRound read_round(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  uint32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  TraceRound round;
  round.deletes.assign(n, 0);
  round.inserts.assign(n, 0);
  in.read(reinterpret_cast<char *>(round.deletes.data()),
          static_cast<std::streamsize>(round.deletes.size() * sizeof(uint32_t)));
  in.read(reinterpret_cast<char *>(round.inserts.data()),
          static_cast<std::streamsize>(round.inserts.size() * sizeof(uint32_t)));
  if (!in) {
    throw std::runtime_error("short trace round");
  }
  return round;
}

TEST(DiskANNUpdateTraceTest, GeneratedTraceMaintainsStableLiveSet) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "diskann_update_trace_test";
  std::filesystem::remove_all(dir);

  const alaya::diskann::bench::UpdateTraceConfig cfg{
      .output_dir = dir,
      .file_prefix = "round_",
      .initial_count = 12,
      .total_count = 20,
      .rounds = 2,
      .update_size = 3,
      .seed = 7,
  };
  alaya::diskann::bench::generate_update_trace(cfg);

  std::unordered_set<uint32_t> live;
  for (uint32_t i = 0; i < cfg.initial_count; ++i) {
    live.insert(i);
  }
  std::unordered_set<uint32_t> inserted;
  for (uint32_t round = 0; round < cfg.rounds; ++round) {
    const TraceRound tr = read_round(dir / ("round_" + std::to_string(round)));
    ASSERT_EQ(tr.deletes.size(), cfg.update_size);
    ASSERT_EQ(tr.inserts.size(), cfg.update_size);
    for (const uint32_t id : tr.deletes) {
      EXPECT_TRUE(live.erase(id)) << "delete id must be live";
    }
    for (const uint32_t id : tr.inserts) {
      EXPECT_GE(id, cfg.initial_count);
      EXPECT_LT(id, cfg.total_count);
      EXPECT_TRUE(inserted.insert(id).second) << "insert ids must not repeat";
      EXPECT_TRUE(live.insert(id).second) << "insert id must not already be live";
    }
    EXPECT_EQ(live.size(), cfg.initial_count);
  }
  EXPECT_TRUE(std::filesystem::exists(dir / "manifest.txt"));
  std::filesystem::remove_all(dir);
}

TEST(DiskANNUpdateTraceTest, YiSequentialTraceMatchesUpdateRunnerOrder) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "diskann_update_trace_yi_test";
  std::filesystem::remove_all(dir);

  alaya::diskann::bench::UpdateTraceConfig cfg{
      .output_dir = dir,
      .file_prefix = "round_",
      .initial_count = 10,
      .total_count = 16,
      .rounds = 2,
      .update_size = 3,
      .seed = 7,
  };
  cfg.mode = alaya::diskann::bench::UpdateTraceMode::YiSequential;
  alaya::diskann::bench::generate_update_trace(cfg);

  const TraceRound round0 = read_round(dir / "round_0");
  EXPECT_EQ(round0.deletes, (std::vector<uint32_t>{1, 2, 3}));
  EXPECT_EQ(round0.inserts, (std::vector<uint32_t>{10, 11, 12}));

  const TraceRound round1 = read_round(dir / "round_1");
  EXPECT_EQ(round1.deletes, (std::vector<uint32_t>{4, 5, 6}));
  EXPECT_EQ(round1.inserts, (std::vector<uint32_t>{13, 14, 15}));

  std::filesystem::remove_all(dir);
}

TEST(DiskANNUpdateTraceTest, RejectsInsufficientReserveIds) {
  const alaya::diskann::bench::UpdateTraceConfig cfg{
      .output_dir = std::filesystem::temp_directory_path() / "diskann_update_trace_bad",
      .file_prefix = "round_",
      .initial_count = 18,
      .total_count = 20,
      .rounds = 2,
      .update_size = 2,
      .seed = 7,
  };
  EXPECT_THROW(alaya::diskann::bench::generate_update_trace(cfg), std::invalid_argument);
}

}  // namespace
