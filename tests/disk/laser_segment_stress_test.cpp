// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/disk/laser_segment.hpp"
#include "utils/platform.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint64_t kCount = 2048;

struct SearchStorage {
  SearchStorage(const float *query)
      : hits(10), offsets(2), counts(1), statuses(1), completeness(1) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    extension_options.effort = 64;
    extension = LaserSegment::make_search_extension(extension_options);
    request.queries = core::TypedTensorView::contiguous(query, 1, kDim);
    request.options.top_k = 10;
    request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = &context;
    request.response = &response;
  }

  core::SearchContext context{};
  LaserSegmentSearchExtension extension_options{};
  core::AlgorithmSearchExtension extension{};
  std::vector<core::SearchHit> hits{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  core::SearchResponse response{};
  core::SearchRequest request{};
};

TEST(LaserSegmentStress, ConcurrentSearchOnlyIsReentrant) {
  const auto fixture = std::filesystem::path(ALAYA_LASER_FIXTURE_DIR);
  if (!engine_supported_v1(DiskIndexType::Laser) || fixture.empty()) {
    GTEST_SKIP() << "LASER fixture is unavailable";
  }
  const auto root = std::filesystem::temp_directory_path() /
                    ("alaya-laser-segment-stress-" + std::to_string(platform::get_pid()));
  std::filesystem::remove_all(root);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup{root};
  std::filesystem::create_directories(root / "segments");
  std::vector<std::uint64_t> labels(kCount);
  std::iota(labels.begin(), labels.end(), std::uint64_t{0});
  LaserSegmentImporter importer(kDim, MetricType::L2, {});
  (void)importer.import_from(fixture, labels.data(), labels.size(), root / "segments/seg_00000001");

  const auto vector_path = fixture / (std::string(ALAYA_LASER_FIXTURE_PREFIX) + "_input.fbin");
  std::ifstream input(vector_path, std::ios::binary);
  std::int32_t count{};
  std::int32_t dim{};
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  ASSERT_EQ(count, static_cast<std::int32_t>(kCount));
  ASSERT_EQ(dim, static_cast<std::int32_t>(kDim));
  std::vector<float> vectors(kCount * kDim);
  input.read(reinterpret_cast<char *>(vectors.data()),
             static_cast<std::streamsize>(vectors.size() * sizeof(float)));
  ASSERT_TRUE(input);

  core::OpenContext open_context;
  auto opened = LaserSegment::open_directory(root / "segments/seg_00000001",
                                             core::OpenOptions{},
                                             open_context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto segment = std::shared_ptr<LaserSegment>(std::move(opened).value());
  SearchStorage baseline(vectors.data() + 43 * kDim);
  ASSERT_TRUE(segment->search(baseline.request).ok());
  ASSERT_EQ(baseline.counts[0], 10U);

  std::atomic_bool failed{};
  std::vector<std::thread> workers;
  for (int worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker] {
      for (int iteration = 0; iteration < 100 && !failed.load(std::memory_order_relaxed);
           ++iteration) {
        SearchStorage call(vectors.data() + 43 * kDim);
        if (!segment->search(call.request).ok() || call.counts[0] != 10U ||
            call.response.score_kind != core::ScoreKind::rank_only) {
          failed.store(true, std::memory_order_relaxed);
          return;
        }
        for (std::size_t index = 0; index < call.counts[0]; ++index) {
          const auto score_bits = std::bit_cast<std::uint32_t>(call.hits[index].score);
          const auto nan =
              (score_bits & 0x7F800000U) == 0x7F800000U && (score_bits & 0x007FFFFFU) != 0;
          if (static_cast<std::uint64_t>(call.hits[index].row_id) >= kCount || !nan ||
              call.hits[index].score_kind != core::ScoreKind::rank_only) {
            failed.store(true, std::memory_order_relaxed);
            return;
          }
        }
        if ((worker + iteration) % 13 == 0) {
          std::this_thread::yield();
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_relaxed));
}

}  // namespace
}  // namespace alaya::disk
