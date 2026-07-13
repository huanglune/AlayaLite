// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <atomic>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/disk/disk_vamana_segment.hpp"

namespace alaya::disk {
namespace {

struct SearchStorage {
  SearchStorage(const float *query, std::uint32_t dim, std::uint64_t top_k)
      : hits(top_k), offsets(2), counts(1), statuses(1), completeness(1) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    extension_options.effort = 48;
    extension = DiskVamanaSegment::make_search_extension(extension_options);
    request.queries = core::TypedTensorView::contiguous(query, 1, dim);
    request.options.top_k = top_k;
    request.options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    request.context = std::addressof(context);
    request.response = std::addressof(response);
  }

  core::SearchContext context{};
  DiskVamanaSearchExtension extension_options{};
  core::AlgorithmSearchExtension extension{};
  std::vector<core::SearchHit> hits{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  core::SearchResponse response{};
  core::SearchRequest request{};
};

TEST(DiskVamanaSegmentStress, ConcurrentSearchOnlyIsReentrant) {
  constexpr std::uint64_t kRows = 257;
  constexpr std::uint32_t kDim = 8;
  constexpr std::uint64_t kTopK = 12;
  std::vector<float> vectors(kRows * kDim);
  std::vector<std::uint64_t> labels(kRows);
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 50'000 + row;
    for (std::uint32_t column = 0; column < kDim; ++column) {
      vectors[row * kDim + column] = static_cast<float>((row + 5) * (column + 11) % 101) / 37.0F;
    }
  }
  const auto root = std::filesystem::temp_directory_path() /
                    ("alaya-disk-vamana-tsan-" + std::to_string(platform::get_pid()));
  std::filesystem::remove_all(root);
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup{root};

  DiskVamanaBuildInput input(core::TypedTensorView::contiguous(vectors.data(), kRows, kDim),
                             labels);
  VamanaSegmentBuildParams build_params;
  build_params.R = 12;
  build_params.L = 48;
  build_params.num_threads = 1;
  build_params.seed = 424242;
  DiskVamanaPublicationOptions options;
  options.collection_root = root;
  options.segment_id = "seg_00000001";
  core::BuildContext build_context;
  auto built = DiskVamanaSegmentFactory::build(input,
                                               core::Metric::l2,
                                               build_params,
                                               options,
                                               build_context);
  ASSERT_TRUE(built.ok()) << built.status().diagnostic();
  auto segment = std::shared_ptr<DiskVamanaSegment>(std::move(built).value());

  SearchStorage baseline(vectors.data() + 17 * kDim, kDim, kTopK);
  ASSERT_TRUE(segment->search(baseline.request).ok());
  std::vector<std::pair<std::uint64_t, std::uint32_t>> expected;
  for (std::size_t index = 0; index < baseline.counts[0]; ++index) {
    expected.emplace_back(static_cast<std::uint64_t>(baseline.hits[index].row_id),
                          std::bit_cast<std::uint32_t>(baseline.hits[index].score));
  }

  std::atomic_bool failed{};
  std::vector<std::thread> workers;
  for (std::uint32_t worker = 0; worker < 8; ++worker) {
    workers.emplace_back([&, worker] {
      for (std::uint32_t iteration = 0; iteration < 200 && !failed.load(); ++iteration) {
        SearchStorage call(vectors.data() + 17 * kDim, kDim, kTopK);
        if (!segment->search(call.request).ok() || call.counts[0] != expected.size()) {
          failed.store(true);
          return;
        }
        for (std::size_t index = 0; index < expected.size(); ++index) {
          if (static_cast<std::uint64_t>(call.hits[index].row_id) != expected[index].first ||
              std::bit_cast<std::uint32_t>(call.hits[index].score) != expected[index].second) {
            failed.store(true);
            return;
          }
        }
        if ((iteration + worker) % 17 == 0) {
          std::this_thread::yield();
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_FALSE(failed.load());
}

}  // namespace
}  // namespace alaya::disk
