// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

#include <gtest/gtest.h>

#include <unistd.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kDegree = 64;
constexpr size_t kRows = 2000;
constexpr uint32_t kTopK = 10;
constexpr size_t kEfSearch = 64;
constexpr size_t kBeamWidth = 8;
constexpr size_t kQueryCount = 8;
constexpr size_t kSerialRepeats = 16;
constexpr size_t kConcurrentLanes = 4;
constexpr size_t kConcurrentRepeats = 8;

struct TinyIndex {
  std::filesystem::path directory;
  std::string prefix;
  std::vector<float> rows;

  TinyIndex() = default;
  TinyIndex(const TinyIndex &) = delete;
  auto operator=(const TinyIndex &) -> TinyIndex & = delete;
  TinyIndex(TinyIndex &&other) noexcept
      : directory(std::move(other.directory)),
        prefix(std::move(other.prefix)),
        rows(std::move(other.rows)) {
    other.directory.clear();
  }
  auto operator=(TinyIndex &&other) noexcept -> TinyIndex & {
    if (this != &other) {
      remove();
      directory = std::move(other.directory);
      prefix = std::move(other.prefix);
      rows = std::move(other.rows);
      other.directory.clear();
    }
    return *this;
  }
  ~TinyIndex() { remove(); }

  static auto build() -> TinyIndex {
    TinyIndex index;
    index.directory = std::filesystem::temp_directory_path() /
                      ("laser_paged_determinism_" + std::to_string(::getpid()));
    std::filesystem::remove_all(index.directory);
    std::filesystem::create_directories(index.directory);
    index.prefix = (index.directory / "tiny").string();

    std::mt19937 generator(1001);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    index.rows.resize(kRows * kDim);
    for (auto &value : index.rows) value = distribution(generator);

    const std::string vector_path = index.prefix + "_pca_base.fbin";
    std::ofstream vectors(vector_path, std::ios::binary | std::ios::trunc);
    if (!vectors.is_open()) throw std::runtime_error("failed to create tiny vector fixture");
    const int32_t row_count = static_cast<int32_t>(kRows);
    const int32_t dimension = static_cast<int32_t>(kDim);
    vectors.write(reinterpret_cast<const char *>(&row_count), sizeof(row_count));
    vectors.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    vectors.write(reinterpret_cast<const char *>(index.rows.data()),
                  static_cast<std::streamsize>(index.rows.size() * sizeof(float)));
    vectors.close();

    vamana::VamanaBuildParams params;
    params.R = kDegree;
    params.L = 64;
    params.alpha = 1.2F;
    params.num_threads = 1;
    vamana::VamanaBuilder vamana_builder(index.rows.data(), kRows, kDim, params);
    vamana_builder.build();
    const std::string vamana_path = index.prefix + "_vamana.index";
    vamana::save_graph(
        vamana_builder.graph(), vamana_path, kDegree, vamana_builder.medoid());

    QuantizedGraph graph(kRows, kDegree, kDim, kDim, /*rotator_seed=*/7);
    QGBuilder qg_builder(graph, /*ef_build=*/64, /*num_threads=*/1);
    qg_builder.build(vamana_path.c_str(), index.prefix.c_str());
    return index;
  }

 private:
  void remove() noexcept {
    if (directory.empty()) return;
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    directory.clear();
  }
};

class PagedDeterminismTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { index_ = std::make_unique<TinyIndex>(TinyIndex::build()); }
  static void TearDownTestSuite() { index_.reset(); }

  static auto open_zero_cache_graph() -> std::unique_ptr<QuantizedGraph> {
    auto graph = std::make_unique<QuantizedGraph>(kRows, kDegree, kDim, kDim);
    // A zero-byte cache budget guarantees every expanded node takes the
    // asynchronous page-read path under test.
    graph->load_disk_index(index_->prefix.c_str(), /*search_DRAM_budget=*/0.0F);
    graph->set_params(kEfSearch, kConcurrentLanes, static_cast<int>(kBeamWidth));
    return graph;
  }

  static auto query(size_t query_index) -> const float * {
    const size_t row = (query_index * 197 + 17) % kRows;
    return index_->rows.data() + row * kDim;
  }

  static auto search(QuantizedGraph &graph, size_t query_index) -> std::array<PID, kTopK> {
    std::array<PID, kTopK> results{};
    graph.search(query(query_index),
                 kTopK,
                 results.data(),
                 kEfSearch,
                 kBeamWidth);
    return results;
  }

  static std::unique_ptr<TinyIndex> index_;
};

std::unique_ptr<TinyIndex> PagedDeterminismTest::index_;

TEST_F(PagedDeterminismTest, SerialRepeatedTopKIsBitwiseStable) {
  auto graph = open_zero_cache_graph();
  std::array<std::array<PID, kTopK>, kQueryCount> expected{};
  for (size_t query_index = 0; query_index < kQueryCount; ++query_index) {
    expected[query_index] = search(*graph, query_index);
  }

  for (size_t repeat = 0; repeat < kSerialRepeats; ++repeat) {
    for (size_t query_index = 0; query_index < kQueryCount; ++query_index) {
      EXPECT_EQ(search(*graph, query_index), expected[query_index])
          << "repeat=" << repeat << ", query=" << query_index;
    }
  }
}

TEST_F(PagedDeterminismTest, ConcurrentLanesMatchSerialTopK) {
  auto graph = open_zero_cache_graph();
  std::array<std::array<PID, kTopK>, kQueryCount> expected{};
  for (size_t query_index = 0; query_index < kQueryCount; ++query_index) {
    expected[query_index] = search(*graph, query_index);
  }

  std::barrier start(static_cast<std::ptrdiff_t>(kConcurrentLanes));
  std::atomic_bool mismatch{false};
  std::mutex detail_mutex;
  std::string mismatch_detail;
  std::vector<std::thread> lanes;
  lanes.reserve(kConcurrentLanes);
  for (size_t lane = 0; lane < kConcurrentLanes; ++lane) {
    lanes.emplace_back([&, lane] {
      start.arrive_and_wait();
      for (size_t repeat = 0; repeat < kConcurrentRepeats && !mismatch.load(); ++repeat) {
        for (size_t query_index = 0; query_index < kQueryCount; ++query_index) {
          const auto actual = search(*graph, query_index);
          if (actual == expected[query_index]) continue;
          mismatch.store(true);
          std::lock_guard lock(detail_mutex);
          if (mismatch_detail.empty()) {
            std::ostringstream detail;
            detail << "lane=" << lane << ", repeat=" << repeat << ", query=" << query_index;
            mismatch_detail = detail.str();
          }
          return;
        }
      }
    });
  }
  for (auto &lane : lanes) lane.join();

  EXPECT_FALSE(mismatch.load()) << mismatch_detail;
}

}  // namespace
}  // namespace alaya::laser
