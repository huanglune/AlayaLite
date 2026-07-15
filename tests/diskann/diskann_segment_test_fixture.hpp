// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/disk/diskann_segment.hpp"
#include "index/collection/segmented_collection.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
  #include <csignal>
#endif

namespace diskann_test {

using alaya::core::OperationCapability;
using alaya::disk::DiskAnnMutableSegmentFactory;
using alaya::disk::DiskAnnMutableSegmentOptions;
using alaya::disk::DiskAnnSegment;
using alaya::disk::DiskAnnSegmentFactory;
using alaya::disk::DiskAnnSegmentSearchExtension;
using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNSearchParams;

constexpr std::uint64_t kRows = 192;
constexpr std::uint64_t kAsyncRows = 4096;
constexpr std::uint32_t kDim = 8;
constexpr std::uint32_t kTopK = 7;
using namespace std::chrono_literals;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::atomic<std::uint64_t> sequence{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya_diskann_segment_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
             std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct ArtifactBundle {
  explicit ArtifactBundle(const std::filesystem::path &directory) {
    const std::array<std::pair<std::string_view, std::string_view>, 5> specs{{
        {DiskAnnSegment::kMetaArtifactName, "meta.bin"},
        {DiskAnnSegment::kIndexArtifactName, "diskann.index"},
        {DiskAnnSegment::kIdsArtifactName, "ids.bin"},
        {DiskAnnSegment::kCacheIdsArtifactName, "cache_ids.bin"},
        {DiskAnnSegment::kCacheNodesArtifactName, "cache_nodes.bin"},
    }};
    paths.reserve(specs.size());
    locations.reserve(specs.size());
    for (const auto &[name, filename] : specs) {
      (void)name;
      paths.push_back((directory / filename).string());
    }
    for (std::size_t index = 0; index < specs.size(); ++index) {
      locations.emplace_back(specs[index].first, paths[index]);
    }
  }

  [[nodiscard]] auto view() const -> alaya::core::ArtifactView {
    return alaya::core::ArtifactView(locations);
  }

  std::vector<std::string> paths{};
  std::vector<alaya::core::ArtifactLocation> locations{};
};

struct ResponseStorage {
  ResponseStorage(std::uint64_t rows, std::uint64_t top_k)
      : hits(static_cast<std::size_t>(rows * top_k)),
        offsets(static_cast<std::size_t>(rows + 1)),
        counts(static_cast<std::size_t>(rows)),
        statuses(static_cast<std::size_t>(rows)),
        completeness(static_cast<std::size_t>(rows)) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
  }

  alaya::core::SearchResponse response{};
  std::vector<alaya::core::SearchHit> hits{};
  std::vector<alaya::core::RowCount> offsets{};
  std::vector<alaya::core::RowCount> counts{};
  std::vector<alaya::core::Status> statuses{};
  std::vector<alaya::core::SearchCompleteness> completeness{};
};

struct CompletionWaiter {
  void complete(alaya::core::Status value) {
    {
      std::lock_guard lock(mutex);
      status = std::move(value);
      ++count;
    }
    ready.notify_all();
  }

  [[nodiscard]] auto wait(std::uint32_t expected = 1) -> alaya::core::Status {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 10s, [&] {
          return count >= expected;
        })) {
      return alaya::core::Status::error(alaya::core::StatusCode::deadline_exceeded,
                                        alaya::core::OperationStage::completion,
                                        alaya::core::StatusDetail::deadline_reached,
                                        "test completion wait timed out");
    }
    return status;
  }

  std::mutex mutex;
  std::condition_variable ready;
  std::uint32_t count{};
  alaya::core::Status status{};
};

struct ManualLane {
  static void post(void *raw, alaya::core::RuntimeLane::Task callback) noexcept {
    auto &lane = *static_cast<ManualLane *>(raw);
    {
      std::lock_guard lock(lane.mutex);
      lane.task = std::move(callback);
      ++lane.posts;
    }
    lane.ready.notify_one();
  }

  [[nodiscard]] auto take() -> alaya::core::RuntimeLane::Task {
    std::unique_lock lock(mutex);
    if (!ready.wait_for(lock, 10s, [&] {
          return static_cast<bool>(task);
        })) {
      return {};
    }
    return std::move(task);
  }

  std::mutex mutex;
  std::condition_variable ready;
  alaya::core::RuntimeLane::Task task{};
  std::uint32_t posts{};
};

struct SafePointCancellation {
  [[nodiscard]] static auto requested(const void *raw) noexcept -> bool {
    auto &control =
        *const_cast<SafePointCancellation *>(static_cast<const SafePointCancellation *>(raw));
    const auto call = control.calls.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (call >= control.pause_on_call && !control.release.load(std::memory_order_acquire)) {
      control.paused.store(true, std::memory_order_release);
      while (!control.release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    }
    return control.cancel.load(std::memory_order_acquire);
  }

  [[nodiscard]] auto token() const -> alaya::core::CancellationToken {
    alaya::core::CancellationToken token;
    token.state = this;
    token.is_cancelled = &requested;
    return token;
  }

  std::atomic_uint32_t calls{};
  std::atomic_bool paused{};
  std::atomic_bool cancel{};
  std::atomic_bool release{};
  std::uint32_t pause_on_call{3};
};

struct SafePointDeadline {
  [[nodiscard]] static auto requested(const void *raw) noexcept -> bool {
    auto &control = *const_cast<SafePointDeadline *>(static_cast<const SafePointDeadline *>(raw));
    const auto call = control.calls.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (call >= control.expire_on_call) {
      while (std::chrono::steady_clock::now() < control.deadline) {
        std::this_thread::yield();
      }
    }
    return false;
  }

  [[nodiscard]] auto token() const -> alaya::core::CancellationToken {
    alaya::core::CancellationToken token;
    token.state = this;
    token.is_cancelled = &requested;
    return token;
  }

  std::atomic_uint32_t calls{};
  std::uint32_t expire_on_call{3};
  std::chrono::steady_clock::time_point deadline{};
};

template <class Control>
struct PinnedSearchBuffers {
  explicit PinnedSearchBuffers(std::span<const float> source)
      : query(source.begin(), source.end()),
        storage(1, kTopK),
        control(std::make_shared<Control>()) {}

  std::vector<float> query;
  ResponseStorage storage;
  alaya::core::SearchStats stats;
  std::shared_ptr<Control> control;
};

class DiskAnnSegmentTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (const char *external = std::getenv("ALAYA_DISKANN_TSAN_FIXTURE_DIR");
        external != nullptr && *external != '\0') {
      external_async_fixture_ = true;
      async_directory_ = external;
      std::mt19937 random(424242);
      std::normal_distribution<float> distribution(0.0F, 1.0F);
      async_vectors_.resize(kAsyncRows * kDim);
      for (auto &value : async_vectors_) {
        value = distribution(random);
      }
      return;
    }
    temporary_ = std::make_unique<TemporaryDirectory>();
    index_directory_ = temporary_->path() / "index";
    std::mt19937 random(42);
    std::normal_distribution<float> distribution(0.0F, 1.0F);
    vectors_.resize(kRows * kDim);
    for (auto &value : vectors_) {
      value = distribution(random);
    }
    labels_.resize(kRows);
    for (std::uint64_t row = 0; row < kRows; ++row) {
      labels_[row] = 10000 + row * 3;
    }
    DiskANNBuildParams build;
    build.R = 16;
    build.L = 40;
    build.alpha = 1.2F;
    build.cache_ratio = 0.1;
    build.num_threads = 2;
    build.seed = 17;
    alaya::diskann::DiskANNIndex::build(index_directory_.string(),
                                        vectors_.data(),
                                        labels_.data(),
                                        kRows,
                                        kDim,
                                        build);
    pq_directory_ = temporary_->path() / "pq-index";
    build.pq_n_chunks = 2;
    build.pq_train_iters = 3;
    alaya::diskann::DiskANNIndex::build(pq_directory_.string(),
                                        vectors_.data(),
                                        labels_.data(),
                                        kRows,
                                        kDim,
                                        build);

    async_directory_ = temporary_->path() / "async-index";
    async_vectors_.resize(kAsyncRows * kDim);
    for (auto &value : async_vectors_) {
      value = distribution(random);
    }
    std::vector<std::uint64_t> async_labels(kAsyncRows);
    for (std::uint64_t row = 0; row < kAsyncRows; ++row) {
      async_labels[row] = 50000 + row * 5;
    }
    build.R = 32;
    build.L = 100;
    build.cache_ratio = 0.0;
    build.pq_n_chunks = 0;
    build.num_threads = 2;
    alaya::diskann::DiskANNIndex::build(async_directory_.string(),
                                        async_vectors_.data(),
                                        async_labels.data(),
                                        kAsyncRows,
                                        kDim,
                                        build);
  }

  static void TearDownTestSuite() {
    vectors_.clear();
    async_vectors_.clear();
    labels_.clear();
    if (!external_async_fixture_) {
      temporary_.reset();
    }
  }

  [[nodiscard]] static auto open_segment() -> std::unique_ptr<DiskAnnSegment> {
    alaya::core::OpenContext context;
    auto opened =
        DiskAnnSegment::open_directory(index_directory_, alaya::core::OpenOptions{}, context);
    EXPECT_TRUE(opened.ok()) << opened.status().diagnostic();
    return opened.ok() ? std::move(opened).value() : nullptr;
  }

  [[nodiscard]] static auto open_any(const std::filesystem::path &directory = index_directory_)
      -> alaya::core::AnySegment {
    alaya::core::OpenContext context;
    auto opened = DiskAnnSegment::open_directory(directory, alaya::core::OpenOptions{}, context);
    EXPECT_TRUE(opened.ok()) << opened.status().diagnostic();
    if (!opened.ok()) {
      return {};
    }
    auto erased = DiskAnnSegment::into_any(std::move(opened).value());
    EXPECT_TRUE(erased.ok()) << erased.status().diagnostic();
    return erased.ok() ? std::move(erased).value() : alaya::core::AnySegment{};
  }

  [[nodiscard]] static auto request(
      std::span<const float> queries,
      std::uint64_t rows,
      std::uint64_t top_k,
      alaya::core::SearchContext &context,
      ResponseStorage &storage,
      std::span<const alaya::core::AlgorithmSearchExtension> extensions = {})
      -> alaya::core::SearchRequest {
    alaya::core::SearchRequest request;
    request.queries = alaya::core::TypedTensorView::contiguous(queries.data(), rows, kDim);
    request.options.top_k = top_k;
    request.options.extensions = extensions;
    request.context = std::addressof(context);
    request.response = std::addressof(storage.response);
    return request;
  }

  inline static std::unique_ptr<TemporaryDirectory> temporary_{};
  inline static std::filesystem::path index_directory_{};
  inline static std::filesystem::path pq_directory_{};
  inline static std::filesystem::path async_directory_{};
  inline static std::vector<float> vectors_{};
  inline static std::vector<float> async_vectors_{};
  inline static std::vector<std::uint64_t> labels_{};
  inline static bool external_async_fixture_{};
};

}  // namespace diskann_test
