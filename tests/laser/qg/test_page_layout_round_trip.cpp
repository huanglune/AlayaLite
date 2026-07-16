// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"

namespace {

TEST(LaserPageGeometryTest, ReservesV2TrailersWithoutChangingExistingSlackLayouts) {
  struct GeometryCase {
    size_t node_len;
    size_t node_per_page;
    size_t page_size;
  };
  constexpr std::array<GeometryCase, 12> cases{{
      {4096, 1, 8192},    // exact one-sector row grows by one sector
      {8192, 1, 12288},   // gte768 main448
      {12288, 1, 16384},  // gte768 main768
      {2048, 1, 4096},    // exact two-row page drops from npp2 to npp1
      {1024, 3, 4096},    // exact four-row page drops only as far as needed
      {2560, 1, 4096},    // SIFT
      {1920, 2, 4096},    // D0 64+32
      {8960, 1, 12288},
      {13056, 1, 16384},
      {6912, 1, 8192},
      {5888, 1, 8192},
      {15360, 1, 16384},  // dbp1024
  }};

  for (const auto &c : cases) {
    SCOPED_TRACE(c.node_len);
    const auto geometry = alaya::laser::qg_page_geometry(c.node_len);
    EXPECT_EQ(geometry.node_per_page, c.node_per_page);
    EXPECT_EQ(geometry.page_size, c.page_size);
    EXPECT_GE(geometry.page_size - geometry.node_per_page * c.node_len,
              geometry.node_per_page * alaya::laser::kQGRowTrailerSize);
  }
}

constexpr size_t kNumPoints = 1024;
constexpr size_t kMainDim = 64;
constexpr size_t kDim = 128;
constexpr size_t kDegree = 32;
constexpr uint64_t kSeed = 42;

class LaserPageLayoutRoundTripTest : public ::testing::Test {
 protected:
  std::filesystem::path root_;
  std::filesystem::path prefix_;
  std::vector<float> vectors_;

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alaya_laser_page_layout_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    prefix_ = root_ / "tiny";
    vectors_ = write_vectors(prefix_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static std::vector<float> write_vectors(const std::filesystem::path &prefix) {
    std::vector<float> vectors(kNumPoints * kDim);
    std::mt19937 rng(kSeed);
    std::normal_distribution<float> gaussian(0.0F, 1.0F);
    for (float &value : vectors) {
      value = gaussian(rng);
    }

    const std::string data_path = prefix.string() + "_pca_base.fbin";
    std::ofstream out(data_path, std::ios::binary);
    const int n = static_cast<int>(kNumPoints);
    const int d = static_cast<int>(kDim);
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(&d), sizeof(d));
    out.write(reinterpret_cast<const char *>(vectors.data()),
              static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    return vectors;
  }

  static auto make_graph_snapshot() -> alaya::FrozenGraphSnapshot {
    alaya::FrozenGraphSnapshot::Adjacency adjacency(kNumPoints);
    for (uint32_t i = 0; i < kNumPoints; ++i) {
      adjacency[i].reserve(kDegree);
      for (uint32_t j = 0; j < kDegree; ++j) {
        adjacency[i].push_back((i + j + 1) % static_cast<uint32_t>(kNumPoints));
      }
    }
    return alaya::FrozenGraphSnapshot(std::move(adjacency),
                                      /*entry_point=*/0,
                                      static_cast<std::uint32_t>(kDegree));
  }

  static auto open_fd_count() -> size_t {
#ifdef __linux__
    std::error_code ec;
    size_t count = 0;
    for (const auto &entry : std::filesystem::directory_iterator("/proc/self/fd", ec)) {
      (void)entry;
      ++count;
    }
    return ec ? 0 : count;
#else
    return 0;
#endif
  }
};

TEST_F(LaserPageLayoutRoundTripTest, BuilderPacksLowDimNodesPerReadPathLayout) {
  alaya::laser::QuantizedGraph graph(kNumPoints, kDegree, kMainDim, kDim, kSeed);
  alaya::laser::QGBuilder builder(graph, /*ef_build=*/64, /*num_threads=*/2);
  std::vector<std::vector<char>> expected_payloads(kNumPoints);
  builder.set_node_payload_observer(
      [&](alaya::laser::PID id, const char *payload, size_t payload_len) {
        expected_payloads.at(id).assign(payload, payload + payload_len);
      });

  (void)open_fd_count();
  const size_t fds_before = open_fd_count();
  const auto snapshot = make_graph_snapshot();
  builder.build_from_graph(snapshot, prefix_.string().c_str());
  const size_t fds_after = open_fd_count();
  EXPECT_LE(fds_after, fds_before) << "QGBuilder leaked file descriptors";

  const std::filesystem::path index_path = prefix_.string() + "_R" + std::to_string(kDegree) +
                                           "_MD" + std::to_string(kMainDim) + ".index";
  ASSERT_TRUE(std::filesystem::is_regular_file(index_path));

  std::ifstream index(index_path, std::ios::binary);
  ASSERT_TRUE(index.is_open());

  std::vector<uint64_t> metas(alaya::laser::kSectorLen / sizeof(uint64_t), 0);
  index.read(reinterpret_cast<char *>(metas.data()),
             static_cast<std::streamsize>(alaya::laser::kSectorLen));

  const uint64_t node_len = metas[3];
  const uint64_t node_per_page = metas[4];
  const uint64_t page_size =
      ((node_per_page * node_len + alaya::laser::kSectorLen - 1) / alaya::laser::kSectorLen) *
      alaya::laser::kSectorLen;
  const uint64_t expected_file_size =
      alaya::laser::kSectorLen + ((kNumPoints + node_per_page - 1) / node_per_page) * page_size;

  ASSERT_EQ(metas[0], kNumPoints);
  ASSERT_EQ(metas[1], kMainDim);
  ASSERT_GT(node_per_page, 1U);
  ASSERT_EQ(metas[8], expected_file_size);
  ASSERT_EQ(std::filesystem::file_size(index_path), expected_file_size);

  std::vector<char> stored;
  for (uint64_t id = 0; id < kNumPoints; ++id) {
    const uint64_t offset = alaya::laser::kSectorLen + page_size * (id / node_per_page) +
                            (id % node_per_page) * node_len;
    ASSERT_EQ(expected_payloads[id].size(), node_len) << "node " << id;
    stored.assign(node_len, 0);
    index.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    index.read(stored.data(), static_cast<std::streamsize>(stored.size()));
    ASSERT_TRUE(index.good()) << "node " << id << " at offset " << offset;
    EXPECT_EQ(stored, expected_payloads[id]) << "node " << id;
  }

  const uint64_t live_slots_in_last_page = kNumPoints % node_per_page;
  if (live_slots_in_last_page != 0) {
    const uint64_t page_index = kNumPoints / node_per_page;
    const uint64_t trailing_offset =
        alaya::laser::kSectorLen + page_index * page_size + live_slots_in_last_page * node_len;
    const uint64_t trailing_len =
        alaya::laser::kSectorLen + (page_index + 1) * page_size - trailing_offset;
    std::vector<char> trailing(trailing_len);
    index.seekg(static_cast<std::streamoff>(trailing_offset), std::ios::beg);
    index.read(trailing.data(), static_cast<std::streamsize>(trailing.size()));
    ASSERT_TRUE(index.good());
    EXPECT_TRUE(std::all_of(trailing.begin(), trailing.end(), [](char byte) {
      return byte == 0;
    }));
  }
}

}  // namespace
