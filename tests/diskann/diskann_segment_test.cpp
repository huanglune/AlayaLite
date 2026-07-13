// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/diskann_segment.hpp"

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

namespace {

using alaya::core::OperationCapability;
using alaya::disk::DiskAnnSegment;
using alaya::disk::DiskAnnSegmentFactory;
using alaya::disk::DiskAnnSegmentLegacyFactory;
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

TEST_F(DiskAnnSegmentTest, SyncSingleAndBatchAreBitwiseEqualToDirectDiskAnn) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  auto direct_result = DiskAnnSegmentLegacyFactory::open(index_directory_);
  ASSERT_TRUE(direct_result.ok()) << direct_result.status().diagnostic();
  auto direct = std::move(direct_result).value();

  DiskAnnSegmentSearchExtension extension_options;
  extension_options.search_list_size = 64;
  extension_options.use_pq = false;
  extension_options.rerank = false;
  extension_options.deterministic = true;
  const auto extension = DiskAnnSegment::make_search_extension(extension_options);
  const std::array extensions{extension};
  const DiskANNSearchParams native_options{/*search_list_size=*/64,
                                           /*use_pq=*/false,
                                           /*rerank=*/false,
                                           /*rerank_count=*/0,
                                           /*deterministic=*/true};

  const auto query = std::span(vectors_).subspan(11 * kDim, kDim);
  std::array<std::uint64_t, kTopK> direct_labels{};
  std::array<float, kTopK> direct_scores{};
  const auto direct_count = direct->search(query.data(),
                                           kTopK,
                                           direct_labels.data(),
                                           direct_scores.data(),
                                           native_options);
  alaya::core::SearchContext context;
  ResponseStorage storage(1, kTopK);
  auto single_request = request(query, 1, kTopK, context, storage, extensions);
  const auto single_status = segment->search(single_request);
  ASSERT_TRUE(single_status.ok()) << single_status.diagnostic();
  ASSERT_EQ(storage.counts[0], direct_count);
  EXPECT_EQ(storage.offsets, (std::vector<alaya::core::RowCount>{0, direct_count}));
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }
  constexpr std::uint64_t kQueries = 3;
  std::vector<float> batch_queries;
  for (const auto row : {std::uint64_t{3}, std::uint64_t{29}, std::uint64_t{101}}) {
    const auto begin = vectors_.begin() + static_cast<std::ptrdiff_t>(row * kDim);
    batch_queries.insert(batch_queries.end(), begin, begin + kDim);
  }
  ResponseStorage batch_storage(kQueries, kTopK);
  auto batch_request = request(batch_queries, kQueries, kTopK, context, batch_storage, extensions);
  const auto batch_status = segment->batch_search(batch_request);
  ASSERT_TRUE(batch_status.ok()) << batch_status.diagnostic();
  std::uint64_t cursor{};
  for (std::uint64_t row = 0; row < kQueries; ++row) {
    std::array<std::uint64_t, kTopK> labels{};
    std::array<float, kTopK> scores{};
    const auto count = direct->search(batch_queries.data() + row * kDim,
                                      kTopK,
                                      labels.data(),
                                      scores.data(),
                                      native_options);
    ASSERT_EQ(batch_storage.counts[row], count);
    ASSERT_EQ(batch_storage.offsets[row], cursor);
    for (std::uint32_t hit = 0; hit < count; ++hit) {
      EXPECT_EQ(batch_storage.hits[cursor + hit].row_id.value, labels[hit]);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(batch_storage.hits[cursor + hit].score),
                std::bit_cast<std::uint32_t>(scores[hit]));
    }
    cursor += count;
  }
  EXPECT_EQ(batch_storage.offsets[kQueries], cursor);

  auto erased_result = DiskAnnSegment::into_any(std::move(segment));
  ASSERT_TRUE(erased_result.ok()) << erased_result.status().diagnostic();
  auto erased = std::move(erased_result).value();
  ResponseStorage native_async_storage(1, kTopK);
  auto native_async_request = request(query, 1, kTopK, context, native_async_storage, extensions);
  const auto native_async_status = erased.search(std::move(native_async_request));
  ASSERT_TRUE(native_async_status.ok()) << native_async_status.diagnostic();
  ASSERT_EQ(native_async_storage.counts[0], direct_count);
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(native_async_storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(native_async_storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }
  ResponseStorage started_storage(1, kTopK);
  auto started_request = request(query, 1, kTopK, context, started_storage, extensions);
  CompletionWaiter started_completion;
  auto started = erased.start_search(std::move(started_request),
                                     alaya::core::SearchCompletion([&](alaya::core::Status status) {
                                       started_completion.complete(std::move(status));
                                     }));
  ASSERT_TRUE(started.ok()) << started.status().diagnostic();
  ASSERT_TRUE(started_completion.wait().ok());
  ASSERT_EQ(started_storage.counts[0], direct_count);
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(started_storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(started_storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }
}

TEST_F(DiskAnnSegmentTest, PqRerankIsBitwiseEqualToDirectDiskAnn) {
  alaya::core::OpenContext open_context;
  auto opened =
      DiskAnnSegment::open_directory(pq_directory_, alaya::core::OpenOptions{}, open_context);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto segment = std::move(opened).value();
  EXPECT_EQ(segment->descriptor().preprocessing,
            alaya::core::MetricPreprocessing::engine_quantized);
  auto direct_result = DiskAnnSegmentLegacyFactory::open(pq_directory_);
  ASSERT_TRUE(direct_result.ok()) << direct_result.status().diagnostic();
  auto direct = std::move(direct_result).value();

  DiskAnnSegmentSearchExtension extension_options;
  extension_options.search_list_size = 64;
  extension_options.use_pq = true;
  extension_options.rerank = true;
  extension_options.rerank_count = 32;
  extension_options.deterministic = true;
  const auto extension = DiskAnnSegment::make_search_extension(extension_options);
  const std::array extensions{extension};
  const DiskANNSearchParams native_options{/*search_list_size=*/64,
                                           /*use_pq=*/true,
                                           /*rerank=*/true,
                                           /*rerank_count=*/32,
                                           /*deterministic=*/true};
  const auto query = std::span(vectors_).subspan(73 * kDim, kDim);
  std::array<std::uint64_t, kTopK> direct_labels{};
  std::array<float, kTopK> direct_scores{};
  const auto direct_count = direct->search(query.data(),
                                           kTopK,
                                           direct_labels.data(),
                                           direct_scores.data(),
                                           native_options);
  alaya::core::SearchContext context;
  ResponseStorage storage(1, kTopK);
  auto search_request = request(query, 1, kTopK, context, storage, extensions);
  const auto status = segment->search(search_request);
  ASSERT_TRUE(status.ok()) << status.diagnostic();
  ASSERT_EQ(storage.counts[0], direct_count);
  EXPECT_EQ(storage.response.result_flags,
            alaya::core::ResultFlag::approximate | alaya::core::ResultFlag::exact_reranked);
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }

  auto erased_result = DiskAnnSegment::into_any(std::move(segment));
  ASSERT_TRUE(erased_result.ok()) << erased_result.status().diagnostic();
  auto erased = std::move(erased_result).value();
  ResponseStorage native_async_storage(1, kTopK);
  auto native_async_request = request(query, 1, kTopK, context, native_async_storage, extensions);
  const auto native_async_status = erased.search(std::move(native_async_request));
  ASSERT_TRUE(native_async_status.ok()) << native_async_status.diagnostic();
  ASSERT_EQ(native_async_storage.counts[0], direct_count);
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(native_async_storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(native_async_storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }
  ResponseStorage started_storage(1, kTopK);
  auto started_request = request(query, 1, kTopK, context, started_storage, extensions);
  CompletionWaiter started_completion;
  auto started = erased.start_search(std::move(started_request),
                                     alaya::core::SearchCompletion([&](alaya::core::Status status) {
                                       started_completion.complete(std::move(status));
                                     }));
  ASSERT_TRUE(started.ok()) << started.status().diagnostic();
  ASSERT_TRUE(started_completion.wait().ok());
  ASSERT_EQ(started_storage.counts[0], direct_count);
  for (std::uint32_t hit = 0; hit < direct_count; ++hit) {
    EXPECT_EQ(started_storage.hits[hit].row_id.value, direct_labels[hit]);
    EXPECT_EQ(std::bit_cast<std::uint32_t>(started_storage.hits[hit].score),
              std::bit_cast<std::uint32_t>(direct_scores[hit]));
  }
}

TEST_F(DiskAnnSegmentTest, DescriptorCapabilitiesAndSyncWaitAdapterAreExplicit) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  const auto descriptor = segment->descriptor();
  EXPECT_EQ(descriptor.algorithm_id, alaya::core::algorithm::diskann);
  EXPECT_EQ(descriptor.format_version, alaya::diskann::DiskANNIndex::kMetaVersion);
  EXPECT_EQ(descriptor.factory_version, 1);
  EXPECT_EQ(descriptor.dim, kDim);
  EXPECT_EQ(descriptor.metric, alaya::core::Metric::l2);
  EXPECT_EQ(descriptor.stored_scalar_type, alaya::core::ScalarType::float32);
  EXPECT_EQ(descriptor.medium, alaya::core::Medium::disk);

  alaya::core::SegmentStats stats;
  ASSERT_TRUE(segment->stats(stats).ok());
  EXPECT_EQ(stats.live_rows, kRows);
  EXPECT_EQ(stats.allocated_rows, kRows);
  EXPECT_GT(stats.resident_bytes, 0);
  EXPECT_EQ(stats.health, alaya::core::SegmentHealth::healthy);

  auto erased_result = DiskAnnSegment::into_any(std::move(segment));
  ASSERT_TRUE(erased_result.ok()) << erased_result.status().diagnostic();
  auto erased = std::move(erased_result).value();
  const auto capabilities = erased.capabilities();
  EXPECT_TRUE(capabilities.supports(OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(OperationCapability::mutation));
  EXPECT_FALSE(capabilities.supports(OperationCapability::save));
  EXPECT_FALSE(capabilities.supports(OperationCapability::export_rows));
  EXPECT_FALSE(capabilities.supports(OperationCapability::checkpoint));
  EXPECT_TRUE(capabilities.concurrency.native_async);
  EXPECT_TRUE(capabilities.concurrency.cooperative_cancel);

  const auto query = std::span(vectors_).subspan(5 * kDim, kDim);
  alaya::core::SearchContext context;
  ResponseStorage storage(1, kTopK);
  auto search_request = request(query, 1, kTopK, context, storage);
  const auto status = erased.search(search_request);
  ASSERT_TRUE(status.ok()) << status.diagnostic();
  EXPECT_EQ(storage.counts[0], kTopK);

  alaya::core::CheckpointContext checkpoint_context;
  alaya::core::CheckpointToken checkpoint_token;
  const auto checkpoint = erased.checkpoint(checkpoint_context, checkpoint_token);
  EXPECT_EQ(checkpoint.code(), alaya::core::StatusCode::not_supported);
  EXPECT_EQ(checkpoint.detail(), alaya::core::StatusDetail::operation_slot_absent);
}

TEST_F(DiskAnnSegmentTest, FactoryGateAndResourceDenialsDoNotAffectDirectIndex) {
  ArtifactBundle artifacts(index_directory_);
  alaya::internal::disk::DiskEngineFeatureFlags disabled;
  disabled.diskann_segment = false;
  alaya::core::OpenContext disabled_context;
  auto rejected = DiskAnnSegmentFactory::open(artifacts.view(),
                                              alaya::core::OpenOptions{},
                                              disabled_context,
                                              disabled);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), alaya::core::StatusCode::not_supported);
  EXPECT_EQ(DiskAnnSegmentFactory::registration.current.implementation_key, "diskann_segment");
  EXPECT_EQ(DiskAnnSegmentFactory::registration.current.engine_factory_key, "diskann");
  EXPECT_EQ(DiskAnnSegmentFactory::registration.legacy.implementation_key, "diskann_index");
  EXPECT_TRUE(DiskAnnSegmentFactory::registration.has_legacy_factory);

  auto direct = DiskAnnSegmentLegacyFactory::open(index_directory_);
  ASSERT_TRUE(direct.ok()) << direct.status().diagnostic();
  EXPECT_EQ(direct.value()->size(), kRows);

  alaya::core::OpenContext resident_denied;
  resident_denied.resident_lease = alaya::core::MemoryLease(0);
  auto denied_resident =
      DiskAnnSegmentFactory::open(artifacts.view(), alaya::core::OpenOptions{}, resident_denied);
  ASSERT_FALSE(denied_resident.ok());
  EXPECT_EQ(denied_resident.status().code(), alaya::core::StatusCode::resource_exhausted);

  alaya::core::OpenContext io_denied;
  io_denied.io_credits.available_requests = 0;
  auto denied_io =
      DiskAnnSegmentFactory::open(artifacts.view(), alaya::core::OpenOptions{}, io_denied);
  ASSERT_FALSE(denied_io.ok());
  EXPECT_EQ(denied_io.status().code(), alaya::core::StatusCode::resource_exhausted);

  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  const auto query = std::span(vectors_).subspan(0, kDim);
  alaya::core::SearchContext search_context;
  search_context.query_scratch_lease = alaya::core::MemoryLease(0);
  ResponseStorage storage(1, kTopK);
  auto search_request = request(query, 1, kTopK, search_context, storage);
  const auto denied_search = segment->search(search_request);
  EXPECT_EQ(denied_search.code(), alaya::core::StatusCode::resource_exhausted);
  EXPECT_EQ(storage.counts[0], 0);
  EXPECT_EQ(storage.completeness[0], alaya::core::SearchCompleteness::failed);
}

TEST_F(DiskAnnSegmentTest, TypedInputEmptyAndCancelledPartialSemanticsAreExplicit) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  const auto query = std::span(vectors_).subspan(7 * kDim, kDim);

  alaya::core::SearchContext context;
  ResponseStorage empty_storage(1, 0);
  auto empty_request = request(query, 1, 0, context, empty_storage);
  ASSERT_TRUE(segment->search(empty_request).ok());
  EXPECT_EQ(empty_storage.offsets, (std::vector<alaya::core::RowCount>{0, 0}));
  EXPECT_EQ(empty_storage.counts[0], 0);
  EXPECT_EQ(empty_storage.completeness[0], alaya::core::SearchCompleteness::complete_k);

  std::array<std::int8_t, kDim> int8_query{};
  ResponseStorage typed_storage(1, kTopK);
  auto typed_request = request(query, 1, kTopK, context, typed_storage);
  typed_request.queries = alaya::core::TypedTensorView::contiguous(int8_query.data(), 1, kDim);
  const auto typed_status = segment->search(typed_request);
  EXPECT_EQ(typed_status.code(), alaya::core::StatusCode::not_supported);

  std::atomic_bool cancelled{true};
  alaya::core::SearchContext cancel_context;
  cancel_context.cancellation = alaya::core::CancellationToken::from_atomic(cancelled);
  ResponseStorage discard_storage(1, kTopK);
  auto discard_request = request(query, 1, kTopK, cancel_context, discard_storage);
  const auto discard = segment->search(discard_request);
  EXPECT_EQ(discard.code(), alaya::core::StatusCode::cancelled);
  EXPECT_FALSE(discard.partial());
  EXPECT_EQ(discard_storage.counts[0], 0);
  EXPECT_EQ(discard_storage.completeness[0], alaya::core::SearchCompleteness::failed);

  ResponseStorage retain_storage(1, kTopK);
  auto retain_request = request(query, 1, kTopK, cancel_context, retain_storage);
  retain_request.options.partial_result_policy = alaya::core::PartialResultPolicy::retain;
  const auto retain = segment->search(retain_request);
  EXPECT_EQ(retain.code(), alaya::core::StatusCode::cancelled);
  EXPECT_TRUE(retain.partial());
  EXPECT_EQ(retain_storage.counts[0], 0);
  EXPECT_EQ(retain_storage.completeness[0], alaya::core::SearchCompleteness::cancelled_partial);
}

TEST_F(DiskAnnSegmentTest, NativeCompletionUsesRequestedLaneExactlyOnceAndAllowsReentry) {
  auto segment = open_any(async_directory_);
  const auto query = std::span(async_vectors_).first(kDim);
  alaya::core::SearchContext context;
  ManualLane lane;
  context.lane.state = &lane;
  context.lane.post = &ManualLane::post;
  context.lane.lane_id = 991;
  ResponseStorage storage(1, kTopK);
  auto search_request = request(query, 1, kTopK, context, storage);
  std::atomic_uint32_t callback_count{};
  std::atomic_bool reentry_ok{};

  auto started =
      segment.start_search(std::move(search_request),
                           alaya::core::SearchCompletion([&](alaya::core::Status status) {
                             callback_count.fetch_add(1, std::memory_order_acq_rel);
                             ResponseStorage nested_storage(1, kTopK);
                             alaya::core::SearchContext nested_context;
                             auto nested_request =
                                 request(query, 1, kTopK, nested_context, nested_storage);
                             const auto nested = segment.search(std::move(nested_request));
                             reentry_ok.store(status.ok() && nested.ok() &&
                                                  nested_storage.counts[0] == kTopK,
                                              std::memory_order_release);
                           }));
  ASSERT_TRUE(started.ok()) << started.status().diagnostic();
  auto handle = std::move(started).value();
  EXPECT_EQ(callback_count.load(std::memory_order_acquire), 0U);
  auto delivery = lane.take();
  ASSERT_TRUE(delivery);
  EXPECT_EQ(lane.posts, 1U);
  EXPECT_EQ(callback_count.load(std::memory_order_acquire), 0U);
  delivery();
  EXPECT_EQ(callback_count.load(std::memory_order_acquire), 1U);
  EXPECT_TRUE(reentry_ok.load(std::memory_order_acquire));
  handle.cancel();
  handle.cancel();
  EXPECT_EQ(callback_count.load(std::memory_order_acquire), 1U);
}

TEST_F(DiskAnnSegmentTest, CompletionIsExactlyOnceUnderCancelStress) {
  constexpr std::size_t kOperations = 16;
  auto segment = open_any(async_directory_);
  std::array<alaya::core::SearchContext, kOperations> contexts{};
  std::array<std::atomic_uint32_t, kOperations> per_operation{};
  std::vector<std::unique_ptr<ResponseStorage>> responses;
  std::vector<alaya::core::OperationHandle> handles;
  responses.reserve(kOperations);
  handles.reserve(kOperations);
  CompletionWaiter all;

  for (std::size_t index = 0; index < kOperations; ++index) {
    responses.push_back(std::make_unique<ResponseStorage>(1, kTopK));
    const auto row = (index * 127) % kAsyncRows;
    const auto query = std::span(async_vectors_).subspan(row * kDim, kDim);
    auto search_request = request(query, 1, kTopK, contexts[index], *responses.back());
    auto started =
        segment.start_search(std::move(search_request),
                             alaya::core::SearchCompletion([&, index](alaya::core::Status status) {
                               per_operation[index].fetch_add(1, std::memory_order_acq_rel);
                               all.complete(std::move(status));
                             }));
    ASSERT_TRUE(started.ok()) << started.status().diagnostic();
    handles.push_back(std::move(started).value());
    if ((index & 1U) != 0) {
      handles.back().cancel();
      handles.back().cancel();
    }
  }
  (void)all.wait(kOperations);
  EXPECT_EQ(all.count, kOperations);
  for (std::size_t index = 0; index < kOperations; ++index) {
    handles[index].cancel();
    handles[index].cancel();
    EXPECT_EQ(per_operation[index].load(std::memory_order_acquire), 1U);
  }
}

TEST_F(DiskAnnSegmentTest, CancelAtDrainedWavePinsBuffersAndAppliesBothPartialPolicies) {
  auto segment = open_any(async_directory_);
  const auto query = std::span(async_vectors_).subspan(31 * kDim, kDim);

  // Prewarm the native pipeline scratch so cancellation is forced by the test
  // probe at a beam safe point, not by first-use allocation timing.
  alaya::core::SearchContext warm_context;
  ResponseStorage warm_storage(1, kTopK);
  auto warm_request = request(query, 1, kTopK, warm_context, warm_storage);
  ASSERT_TRUE(segment.search(std::move(warm_request)).ok());

  for (const auto policy :
       {alaya::core::PartialResultPolicy::discard, alaya::core::PartialResultPolicy::retain}) {
    using Buffers = PinnedSearchBuffers<SafePointCancellation>;
    auto buffers = std::make_shared<Buffers>(query);
    std::weak_ptr<Buffers> weak_buffers = buffers;
    std::weak_ptr<SafePointCancellation> weak_control = buffers->control;
    alaya::core::SearchContext context;
    context.cancellation = buffers->control->token();
    context.stats = &buffers->stats;
    auto search_request = request(buffers->query, 1, kTopK, context, buffers->storage);
    search_request.options.partial_result_policy = policy;
    search_request.lifetime_pin = buffers;
    CompletionWaiter completion;
    std::atomic_bool pin_alive_in_callback{};
    std::atomic_uint64_t completed_count{};
    std::atomic<alaya::core::SearchCompleteness> completed_kind{
        alaya::core::SearchCompleteness::failed};
    std::atomic_uint64_t completed_io{};
    auto started =
        segment.start_search(std::move(search_request),
                             alaya::core::SearchCompletion([&](alaya::core::Status status) {
                               if (auto pinned = weak_buffers.lock()) {
                                 pin_alive_in_callback.store(true, std::memory_order_release);
                                 completed_count.store(pinned->storage.counts[0],
                                                       std::memory_order_release);
                                 completed_kind.store(pinned->storage.completeness[0],
                                                      std::memory_order_release);
                                 completed_io.store(pinned->stats.io_requests,
                                                    std::memory_order_release);
                               }
                               completion.complete(std::move(status));
                             }));
    ASSERT_TRUE(started.ok()) << started.status().diagnostic();
    auto handle = std::move(started).value();
    buffers.reset();

    for (std::uint32_t spin = 0; spin < 100000; ++spin) {
      auto control = weak_control.lock();
      ASSERT_NE(control, nullptr);
      if (control->paused.load(std::memory_order_acquire)) {
        break;
      }
      std::this_thread::yield();
    }
    auto control = weak_control.lock();
    ASSERT_NE(control, nullptr);
    ASSERT_TRUE(control->paused.load(std::memory_order_acquire));
    control->cancel.store(true, std::memory_order_release);
    handle.cancel();
    handle.cancel();
    control->release.store(true, std::memory_order_release);
    control.reset();

    const auto status = completion.wait();
    EXPECT_EQ(status.code(), alaya::core::StatusCode::cancelled);
    EXPECT_EQ(status.partial(), policy == alaya::core::PartialResultPolicy::retain);
    EXPECT_EQ(completion.count, 1U);
    EXPECT_TRUE(pin_alive_in_callback.load(std::memory_order_acquire));
    EXPECT_GT(completed_io.load(std::memory_order_acquire), 0U);
    if (policy == alaya::core::PartialResultPolicy::discard) {
      EXPECT_EQ(completed_count.load(std::memory_order_acquire), 0U);
      EXPECT_EQ(completed_kind.load(std::memory_order_acquire),
                alaya::core::SearchCompleteness::failed);
    } else {
      EXPECT_GT(completed_count.load(std::memory_order_acquire), 0U);
      EXPECT_EQ(completed_kind.load(std::memory_order_acquire),
                alaya::core::SearchCompleteness::cancelled_partial);
    }
    handle.cancel();
    handle.cancel();
    for (std::uint32_t spin = 0; spin < 100000 && !weak_buffers.expired(); ++spin) {
      std::this_thread::yield();
    }
    EXPECT_TRUE(weak_buffers.expired());
  }
}

TEST_F(DiskAnnSegmentTest, TimeoutAtDrainedWavePinsBuffersAndAppliesBothPartialPolicies) {
  auto segment = open_any(async_directory_);
  const auto query = std::span(async_vectors_).subspan(59 * kDim, kDim);
  alaya::core::SearchContext warm_context;
  ResponseStorage warm_storage(1, kTopK);
  auto warm_request = request(query, 1, kTopK, warm_context, warm_storage);
  ASSERT_TRUE(segment.search(std::move(warm_request)).ok());

  for (const auto policy :
       {alaya::core::PartialResultPolicy::discard, alaya::core::PartialResultPolicy::retain}) {
    using Buffers = PinnedSearchBuffers<SafePointDeadline>;
    auto buffers = std::make_shared<Buffers>(query);
    std::weak_ptr<Buffers> weak_buffers = buffers;
    buffers->control->deadline = std::chrono::steady_clock::now() + 200ms;
    alaya::core::SearchContext context;
    context.cancellation = buffers->control->token();
    context.deadline = alaya::core::Deadline::at(buffers->control->deadline);
    context.stats = &buffers->stats;
    auto search_request = request(buffers->query, 1, kTopK, context, buffers->storage);
    search_request.options.partial_result_policy = policy;
    search_request.lifetime_pin = buffers;
    CompletionWaiter completion;
    std::atomic_bool pin_alive_in_callback{};
    std::atomic_uint64_t completed_count{};
    std::atomic<alaya::core::SearchCompleteness> completed_kind{
        alaya::core::SearchCompleteness::failed};
    std::atomic_uint64_t completed_io{};
    std::atomic_uint32_t control_calls{};
    auto started =
        segment.start_search(std::move(search_request),
                             alaya::core::SearchCompletion([&](alaya::core::Status status) {
                               if (auto pinned = weak_buffers.lock()) {
                                 pin_alive_in_callback.store(true, std::memory_order_release);
                                 completed_count.store(pinned->storage.counts[0],
                                                       std::memory_order_release);
                                 completed_kind.store(pinned->storage.completeness[0],
                                                      std::memory_order_release);
                                 completed_io.store(pinned->stats.io_requests,
                                                    std::memory_order_release);
                                 control_calls.store(pinned->control->calls.load(
                                                         std::memory_order_acquire),
                                                     std::memory_order_release);
                               }
                               completion.complete(std::move(status));
                             }));
    ASSERT_TRUE(started.ok()) << started.status().diagnostic();
    auto handle = std::move(started).value();
    buffers.reset();
    const auto status = completion.wait();
    EXPECT_EQ(status.code(), alaya::core::StatusCode::deadline_exceeded);
    EXPECT_EQ(status.partial(), policy == alaya::core::PartialResultPolicy::retain);
    EXPECT_EQ(completion.count, 1U);
    EXPECT_TRUE(pin_alive_in_callback.load(std::memory_order_acquire));
    EXPECT_GE(control_calls.load(std::memory_order_acquire), 3U);
    EXPECT_GT(completed_io.load(std::memory_order_acquire), 0U);
    if (policy == alaya::core::PartialResultPolicy::discard) {
      EXPECT_EQ(completed_count.load(std::memory_order_acquire), 0U);
      EXPECT_EQ(completed_kind.load(std::memory_order_acquire),
                alaya::core::SearchCompleteness::failed);
    } else {
      EXPECT_GT(completed_count.load(std::memory_order_acquire), 0U);
      EXPECT_EQ(completed_kind.load(std::memory_order_acquire),
                alaya::core::SearchCompleteness::cancelled_partial);
    }
    handle.cancel();
    handle.cancel();
    for (std::uint32_t spin = 0; spin < 100000 && !weak_buffers.expired(); ++spin) {
      std::this_thread::yield();
    }
    EXPECT_TRUE(weak_buffers.expired());
  }
}

TEST_F(DiskAnnSegmentTest, FanoutCancellationPropagatesAndWaitsForEveryChild) {
  auto first = open_any(async_directory_);
  auto second = open_any(async_directory_);
  const auto query = std::span(async_vectors_).subspan(83 * kDim, kDim);
  std::array<ResponseStorage, 2> responses{ResponseStorage(1, kTopK), ResponseStorage(1, kTopK)};
  std::array<alaya::core::SearchContext, 2> contexts{};
  std::array<std::shared_ptr<SafePointCancellation>, 2>
      controls{std::make_shared<SafePointCancellation>(),
               std::make_shared<SafePointCancellation>()};
  auto routing_pin = std::make_shared<int>(7);
  std::weak_ptr<int> weak_routing = routing_pin;
  std::array<CompletionWaiter, 2> completions{};
  std::array<alaya::core::OperationHandle, 2> handles{};
  std::array<alaya::core::AnySegment *, 2> children{&first, &second};

  for (std::size_t child = 0; child < children.size(); ++child) {
    contexts[child].cancellation = controls[child]->token();
    auto child_request = request(query, 1, kTopK, contexts[child], responses[child]);
    child_request.options.partial_result_policy = alaya::core::PartialResultPolicy::retain;
    child_request.lifetime_pin = routing_pin;
    auto started =
        children[child]->start_search(std::move(child_request),
                                      alaya::core::SearchCompletion(
                                          [&, child](alaya::core::Status status) {
                                            completions[child].complete(std::move(status));
                                          }));
    ASSERT_TRUE(started.ok()) << started.status().diagnostic();
    handles[child] = std::move(started).value();
  }
  routing_pin.reset();
  EXPECT_FALSE(weak_routing.expired());
  for (const auto &control : controls) {
    for (std::uint32_t spin = 0; spin < 100000 && !control->paused.load(std::memory_order_acquire);
         ++spin) {
      std::this_thread::yield();
    }
    ASSERT_TRUE(control->paused.load(std::memory_order_acquire));
  }
  for (std::size_t child = 0; child < children.size(); ++child) {
    controls[child]->cancel.store(true, std::memory_order_release);
    handles[child].cancel();
    handles[child].cancel();
    controls[child]->release.store(true, std::memory_order_release);
  }
  for (std::size_t child = 0; child < children.size(); ++child) {
    const auto status = completions[child].wait();
    EXPECT_EQ(status.code(), alaya::core::StatusCode::cancelled);
    EXPECT_EQ(completions[child].count, 1U);
    EXPECT_EQ(responses[child].completeness[0], alaya::core::SearchCompleteness::cancelled_partial);
  }
  for (std::uint32_t spin = 0; spin < 100000 && !weak_routing.expired(); ++spin) {
    std::this_thread::yield();
  }
  EXPECT_TRUE(weak_routing.expired());
}

TEST_F(DiskAnnSegmentTest, ConcurrentNativeAsyncAndSyncMixedStress) {
  auto segment = open_any(async_directory_);
  constexpr std::uint32_t kThreads = 4;
  constexpr std::uint32_t kIterations = 4;
  std::atomic_uint32_t failures{};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (std::uint32_t worker = 0; worker < kThreads; ++worker) {
    workers.emplace_back([&, worker] {
      for (std::uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        const auto row = (worker * 257 + iteration * 31) % kAsyncRows;
        const auto query = std::span(async_vectors_).subspan(row * kDim, kDim);
        alaya::core::SearchContext context;
        ResponseStorage storage(1, kTopK);
        auto search_request = request(query, 1, kTopK, context, storage);
        if (((worker + iteration) & 1U) == 0) {
          struct AsyncResult {
            std::atomic_bool done{};
            std::atomic_bool ok{};
          };
          auto completion = std::make_shared<AsyncResult>();
          auto started =
              segment.start_search(std::move(search_request),
                                   alaya::core::SearchCompletion([completion](
                                                                     alaya::core::Status status) {
                                     completion->ok.store(status.ok(), std::memory_order_release);
                                     completion->done.store(true, std::memory_order_release);
                                   }));
          const auto wait_until = std::chrono::steady_clock::now() + 10s;
          while (started.ok() && !completion->done.load(std::memory_order_acquire) &&
                 std::chrono::steady_clock::now() < wait_until) {
            std::this_thread::yield();
          }
          if (!started.ok() || !completion->done.load(std::memory_order_acquire) ||
              !completion->ok.load(std::memory_order_acquire)) {
            failures.fetch_add(1, std::memory_order_acq_rel);
          }
        } else if (!segment.search(std::move(search_request)).ok()) {
          failures.fetch_add(1, std::memory_order_acq_rel);
        }
        if (storage.counts[0] != kTopK) {
          failures.fetch_add(1, std::memory_order_acq_rel);
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(failures.load(std::memory_order_acquire), 0U);
}

TEST_F(DiskAnnSegmentTest, SyncAdapterPerformanceSentinelStaysWithinFivePercent) {
  auto segment = open_segment();
  ASSERT_NE(segment, nullptr);
  DiskAnnSegmentSearchExtension extension_options;
  extension_options.search_list_size = 120;
  extension_options.use_pq = false;
  extension_options.rerank = false;
  extension_options.deterministic = true;
  const auto extension = DiskAnnSegment::make_search_extension(extension_options);
  const std::array extensions{extension};
  const DiskANNSearchParams native_options{/*search_list_size=*/120,
                                           /*use_pq=*/false,
                                           /*rerank=*/false,
                                           /*rerank_count=*/0,
                                           /*deterministic=*/true};
  alaya::core::SearchContext context;
  ResponseStorage storage(1, kTopK);
  auto segment_request =
      request(std::span(vectors_).first(kDim), 1, kTopK, context, storage, extensions);
  std::array<std::uint64_t, kTopK> labels{};
  std::array<float, kTopK> scores{};

  auto run_direct = [&](const float *query) {
    auto result = DiskAnnSegmentLegacyFactory::search_differential(*segment,
                                                                   query,
                                                                   kTopK,
                                                                   labels.data(),
                                                                   scores.data(),
                                                                   native_options);
    ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  };
  auto run_segment = [&](const float *query) {
    segment_request.queries =
        alaya::core::TypedTensorView::contiguous(query, std::uint64_t{1}, kDim);
    const auto status = segment->search(segment_request);
    ASSERT_TRUE(status.ok()) << status.diagnostic();
  };

  constexpr std::uint32_t kWarmup = 12;
  constexpr std::uint32_t kMeasured = 80;
  constexpr std::uint32_t kRounds = 7;
  for (std::uint32_t iteration = 0; iteration < kWarmup; ++iteration) {
    const auto *query = vectors_.data() + (iteration * 13U % kRows) * kDim;
    run_direct(query);
    run_segment(query);
  }

  std::vector<double> ratios;
  ratios.reserve(kRounds);
  std::vector<double> direct_us;
  std::vector<double> segment_us;
  direct_us.reserve(kRounds);
  segment_us.reserve(kRounds);
  for (std::uint32_t round = 0; round < kRounds; ++round) {
    std::chrono::nanoseconds direct_elapsed{};
    std::chrono::nanoseconds segment_elapsed{};
    for (std::uint32_t iteration = 0; iteration < kMeasured; ++iteration) {
      const auto *query = vectors_.data() + ((round * kMeasured + iteration) * 17U % kRows) * kDim;
      const bool segment_first = ((round + iteration) & 1U) != 0;
      if (segment_first) {
        const auto segment_begin = std::chrono::steady_clock::now();
        run_segment(query);
        segment_elapsed += std::chrono::steady_clock::now() - segment_begin;
        const auto direct_begin = std::chrono::steady_clock::now();
        run_direct(query);
        direct_elapsed += std::chrono::steady_clock::now() - direct_begin;
      } else {
        const auto direct_begin = std::chrono::steady_clock::now();
        run_direct(query);
        direct_elapsed += std::chrono::steady_clock::now() - direct_begin;
        const auto segment_begin = std::chrono::steady_clock::now();
        run_segment(query);
        segment_elapsed += std::chrono::steady_clock::now() - segment_begin;
      }
    }
    const auto direct =
        std::chrono::duration<double, std::micro>(direct_elapsed).count() / kMeasured;
    const auto adapter =
        std::chrono::duration<double, std::micro>(segment_elapsed).count() / kMeasured;
    direct_us.push_back(direct);
    segment_us.push_back(adapter);
    ratios.push_back((adapter - direct) / direct);
  }
  const auto median = [](std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
  };
  const auto median_direct = median(direct_us);
  const auto median_segment = median(segment_us);
  const auto median_ratio = median(ratios);
  std::cout << "DiskAnnSegment sync sentinel direct_p50_round_us=" << median_direct
            << " segment_p50_round_us=" << median_segment << " overhead_ratio=" << median_ratio
            << '\n';
  EXPECT_LE(median_ratio, 0.05);
}

}  // namespace
