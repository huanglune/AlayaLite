// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/diskann_segment.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::uint64_t kRows = 32768;
constexpr std::uint32_t kDim = 32;
constexpr std::uint32_t kTopK = 10;
constexpr std::uint32_t kSearchListSize = 150;

auto make_vectors(std::uint64_t rows = kRows,
                  std::uint32_t dim = kDim,
                  std::uint32_t seed = 20260712) -> std::vector<float> {
  std::mt19937 random(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> vectors(rows * dim);
  for (auto &value : vectors) {
    value = distribution(random);
  }
  return vectors;
}

void build_tsan_index(const std::filesystem::path &directory) {
  constexpr std::uint64_t rows = 4096;
  constexpr std::uint32_t dim = 8;
  auto vectors = make_vectors(rows, dim, 424242);
  std::vector<std::uint64_t> labels(rows);
  for (std::uint64_t row = 0; row < rows; ++row) {
    labels[row] = 50000 + row * 5;
  }
  alaya::diskann::DiskANNBuildParams params;
  params.R = 32;
  params.L = 100;
  params.alpha = 1.2F;
  params.cache_ratio = 0.0;
  params.num_threads = 2;
  params.seed = 20260712;
  alaya::diskann::DiskANNIndex::build(directory.string(),
                                      vectors.data(),
                                      labels.data(),
                                      rows,
                                      dim,
                                      params);
}

void build_index(const std::filesystem::path &directory) {
  auto vectors = make_vectors();
  std::vector<std::uint64_t> labels(kRows);
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 1000000 + row * 7;
  }
  alaya::diskann::DiskANNBuildParams params;
  params.R = 64;
  params.L = 150;
  params.alpha = 1.2F;
  params.cache_ratio = 0.0;
  params.num_threads = 4;
  params.seed = 20260712;
  alaya::diskann::DiskANNIndex::build(directory.string(),
                                      vectors.data(),
                                      labels.data(),
                                      kRows,
                                      kDim,
                                      params);
}

struct ResponseStorage {
  ResponseStorage() : hits(kTopK), offsets(2), counts(1), statuses(1), completeness(1) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
  }

  std::vector<alaya::core::SearchHit> hits;
  std::vector<alaya::core::RowCount> offsets;
  std::vector<alaya::core::RowCount> counts;
  std::vector<alaya::core::Status> statuses;
  std::vector<alaya::core::SearchCompleteness> completeness;
  alaya::core::SearchResponse response;
};

class DirectRunner {
 public:
  DirectRunner(const std::filesystem::path &directory, bool updatable) {
    alaya::diskann::DiskANNLoadParams load;
    load.num_threads = 4;
    load.beam_width = 4;
    load.scratch_search_list_size = kSearchListSize;
    load.updatable = updatable;
    load.search_page_cache = false;
    if (updatable) {
      load.update_io = alaya::diskann::DiskANNUpdateIO::kUring;
      load.update_insert_threads = 4;
      load.update_reconnect_threads = 4;
    }
    index_.load(directory.string(), load);
  }

  void run(const float *query, bool with_probe) {
    std::uint64_t labels[kTopK]{};
    float distances[kTopK]{};
    alaya::diskann::DiskANNSearchParams params;
    params.search_list_size = kSearchListSize;
    params.use_pq = false;
    params.rerank = false;
    params.deterministic = true;
    std::uint32_t count{};
    if (with_probe) {
      const alaya::diskann::BeamSearchCancelProbe probe{this, [](const void *) noexcept {
                                                          return false;
                                                        }};
      index_.search_pipelined(query,
                              1,
                              kTopK,
                              labels,
                              distances,
                              1,
                              1,
                              params,
                              nullptr,
                              nullptr,
                              &count,
                              &probe);
    } else {
      index_.search_pipelined(query,
                              1,
                              kTopK,
                              labels,
                              distances,
                              1,
                              1,
                              params,
                              nullptr,
                              nullptr,
                              &count);
    }
    if (count == 0 || labels[0] == std::numeric_limits<std::uint64_t>::max()) {
      throw std::runtime_error("direct sentinel returned no hits");
    }
  }

 private:
  alaya::diskann::DiskANNIndex index_;
};

class SegmentRunner {
 public:
  explicit SegmentRunner(const std::filesystem::path &directory) {
    alaya::core::OpenContext context;
    auto opened =
        alaya::disk::DiskAnnSegment::open_directory(directory, alaya::core::OpenOptions{}, context);
    if (!opened.ok()) {
      throw std::runtime_error(opened.status().diagnostic());
    }
    auto erased = alaya::disk::DiskAnnSegment::into_any(std::move(opened).value());
    if (!erased.ok()) {
      throw std::runtime_error(erased.status().diagnostic());
    }
    segment_ = std::move(erased).value();
    extension_options_.search_list_size = kSearchListSize;
    extension_options_.use_pq = false;
    extension_options_.rerank = false;
    extension_options_.deterministic = true;
    extension_ = alaya::disk::DiskAnnSegment::make_search_extension(extension_options_);
  }

  void run(const float *query) {
    ResponseStorage storage;
    alaya::core::SearchContext context;
    alaya::core::SearchRequest request;
    request.queries = alaya::core::TypedTensorView::contiguous(query, std::uint64_t{1}, kDim);
    request.options.top_k = kTopK;
    request.options.extensions =
        std::span<const alaya::core::AlgorithmSearchExtension>(&extension_, 1);
    request.context = &context;
    request.response = &storage.response;
    const auto status = segment_.search(std::move(request));
    if (!status.ok() || storage.counts[0] == 0) {
      throw std::runtime_error(status.diagnostic());
    }
  }

 private:
  alaya::core::AnySegment segment_;
  alaya::disk::DiskAnnSegmentSearchExtension extension_options_;
  alaya::core::AlgorithmSearchExtension extension_;
};

template <class Runner>
void serve(Runner &runner, const std::vector<float> &vectors) {
  std::uint64_t sequence{};
  while (std::cin >> sequence) {
    const auto row = (sequence * 104729 + 17) % kRows;
    const float *query = vectors.data() + row * kDim;
    const auto begin = std::chrono::steady_clock::now();
    runner.run(query);
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - begin)
                             .count();
    std::cout << elapsed << '\n' << std::flush;
  }
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    if (argc != 3) {
      std::cerr << "usage: diskann_async_sentinel MODE INDEX_DIR\n";
      return 2;
    }
    const std::string mode = argv[1];
    const std::filesystem::path directory = argv[2];
    if (mode == "build") {
      build_index(directory);
      return 0;
    }
    if (mode == "build-tsan") {
      build_tsan_index(directory);
      return 0;
    }
    const auto vectors = make_vectors();
    if (mode == "direct-updatable" || mode == "direct-readonly" || mode == "direct-probe") {
      DirectRunner direct(directory, mode == "direct-updatable");
      struct Adapter {
        DirectRunner *runner;
        bool probe;
        void run(const float *query) { runner->run(query, probe); }
      } adapter{&direct, mode == "direct-probe"};
      serve(adapter, vectors);
      return 0;
    }
    if (mode == "segment-sync") {
      SegmentRunner segment(directory);
      serve(segment, vectors);
      return 0;
    }
    throw std::invalid_argument("unknown or unavailable sentinel mode: " + mode);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
