// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "diskann_segment_test_fixture.hpp"

namespace {
using namespace diskann_test;

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

}  // namespace

