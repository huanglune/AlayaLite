// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
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

constexpr size_t kNumPoints = 1024;
constexpr size_t kMainDim = 64;
constexpr size_t kDim = 128;
constexpr size_t kDegree = 32;
constexpr uint64_t kSeed = 42;

class LaserPageLayoutRoundTripTest : public ::testing::Test {
 protected:
  std::filesystem::path root_;
  std::filesystem::path prefix_;
  std::filesystem::path vamana_path_;
  std::vector<float> vectors_;

  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("alaya_laser_page_layout_" + std::to_string(::getpid()));
    std::filesystem::remove_all(root_);
    std::filesystem::create_directories(root_);
    prefix_ = root_ / "tiny";
    vamana_path_ = root_ / "tiny.vamana";
    vectors_ = write_vectors(prefix_);
    write_vamana_graph(vamana_path_);
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

  static void write_vamana_graph(const std::filesystem::path &path) {
    const size_t header_size =
        sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
    const size_t record_size = sizeof(uint32_t) + kDegree * sizeof(uint32_t);
    const size_t expected_file_size = header_size + kNumPoints * record_size;
    const uint32_t max_observed_degree = static_cast<uint32_t>(kDegree);
    const uint32_t start = 0;
    const size_t frozen_points = 0;

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(expected_file_size));
    out.write(reinterpret_cast<const char *>(&max_observed_degree), sizeof(max_observed_degree));
    out.write(reinterpret_cast<const char *>(&start), sizeof(start));
    out.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));

    for (uint32_t i = 0; i < kNumPoints; ++i) {
      const uint32_t degree = static_cast<uint32_t>(kDegree);
      out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
      for (uint32_t j = 0; j < kDegree; ++j) {
        const uint32_t neighbor = (i + j + 1) % static_cast<uint32_t>(kNumPoints);
        out.write(reinterpret_cast<const char *>(&neighbor), sizeof(neighbor));
      }
    }
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

  builder.build(vamana_path_.string().c_str(), prefix_.string().c_str());

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
