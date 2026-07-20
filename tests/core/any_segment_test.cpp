// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "core/any_segment.hpp"

namespace {

using namespace alaya::core;
using namespace std::chrono_literals;

struct ResponseStorage {
  explicit ResponseStorage(RowCount rows, std::uint64_t top_k)
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

  std::vector<SearchHit> hits;
  std::vector<RowCount> offsets;
  std::vector<RowCount> counts;
  std::vector<Status> statuses;
  std::vector<SearchCompleteness> completeness;
  SearchResponse response;
};

auto make_request(const float *data,
                  RowCount rows,
                  std::uint32_t dim,
                  std::uint64_t top_k,
                  SearchContext &context,
                  ResponseStorage &storage) -> SearchRequest {
  SearchRequest request;
  request.queries = TypedTensorView::contiguous(data, rows, dim);
  request.options.top_k = top_k;
  request.context = &context;
  request.response = &storage.response;
  return request;
}

struct CountingSegment {
  std::atomic_uint32_t calls{};
  mutable std::mutex execution_mutex{};
  std::thread::id last_execution_thread{};

  auto descriptor() const noexcept -> Descriptor {
    Descriptor descriptor;
    descriptor.algorithm_id = 42;
    descriptor.dim = 2;
    descriptor.metric = Metric::l2;
    descriptor.stored_scalar_type = ScalarType::float32;
    return descriptor;
  }

  auto search(const SearchRequest &request) const -> Status { return execute(request); }
  auto batch_search(const SearchRequest &request) const -> Status { return execute(request); }

  auto stats(SegmentStats &stats) const noexcept -> Status {
    stats.live_rows = 7;
    return Status::success();
  }

  [[nodiscard]] auto execution_thread() const -> std::thread::id {
    std::lock_guard lock(execution_mutex);
    return last_execution_thread;
  }

 private:
  auto execute(const SearchRequest &request) const -> Status {
    auto *mutable_self = const_cast<CountingSegment *>(this);
    mutable_self->calls.fetch_add(1, std::memory_order_relaxed);
    {
      std::lock_guard lock(execution_mutex);
      mutable_self->last_execution_thread = std::this_thread::get_id();
    }
    auto &response = *request.response;
    RowCount cursor = 0;
    response.offsets[0] = 0;
    for (RowCount row = 0; row < request.queries.rows; ++row) {
      response.hits[cursor] = SearchHit(SegmentRowId(row),
                                        static_cast<float>(row),
                                        ScoreKind::distance,
                                        Metric::l2,
                                        ResultFlag::approximate);
      ++cursor;
      response.offsets[row + 1] = cursor;
      response.valid_counts[row] = 1;
      response.statuses[row] = Status::success();
      response.completeness[row] = request.options.top_k == 1
                                       ? SearchCompleteness::complete_k
                                       : SearchCompleteness::strategy_incomplete;
    }
    response.query_count = request.queries.rows;
    return Status::success();
  }
};

struct ThrowingSegment {
  auto descriptor() const noexcept -> Descriptor {
    Descriptor descriptor;
    descriptor.dim = 2;
    return descriptor;
  }
  auto search(const SearchRequest &) const -> Status { throw std::runtime_error("fake engine"); }
};

struct BudgetSegment {
  auto descriptor() const noexcept -> Descriptor {
    Descriptor descriptor;
    descriptor.dim = 2;
    return descriptor;
  }
  auto search(const SearchRequest &request) const -> Status {
    return require_lease(request.context->query_scratch_lease,
                         64,
                         OperationStage::search,
                         "fake segment scratch budget denied");
  }
};

struct SlowGate {
  std::mutex mutex;
  std::condition_variable condition;
  Deadline *operation_deadline{};
  bool entered{};
  bool release{};
};

struct SlowSegment {
  std::shared_ptr<SlowGate> gate;

  auto descriptor() const noexcept -> Descriptor {
    Descriptor descriptor;
    descriptor.dim = 2;
    return descriptor;
  }
  auto search(const SearchRequest &request) const -> Status {
    std::unique_lock lock(gate->mutex);
    gate->operation_deadline = std::addressof(request.context->deadline);
    gate->entered = true;
    gate->condition.notify_all();
    gate->condition.wait(lock, [&] {
      return gate->release;
    });
    return Status::success();
  }
};

struct CompletionWaiter {
  std::mutex mutex;
  std::condition_variable ready;
  bool done{};
  std::uint32_t count{};
  Status status{};

  void complete(Status value) {
    {
      std::lock_guard lock(mutex);
      status = std::move(value);
      ++count;
      done = true;
    }
    ready.notify_all();
  }

  auto wait() -> Status {
    std::unique_lock lock(mutex);
    EXPECT_TRUE(ready.wait_for(lock, 5s, [&] {
      return done;
    }));
    return status;
  }
};

TEST(AnySegmentV3, DerivesCapabilitiesFromSlotsAndInstanceConfig) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  const auto capabilities = segment.capabilities();
  EXPECT_TRUE(capabilities.supports(OperationCapability::search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::batch_search));
  EXPECT_TRUE(capabilities.supports(OperationCapability::stats));
  EXPECT_FALSE(capabilities.supports(OperationCapability::mutation));
  EXPECT_FALSE(capabilities.supports(OperationCapability::checkpoint));

  SegmentStats stats;
  EXPECT_TRUE(segment.stats(stats).ok());
  EXPECT_EQ(stats.live_rows, 7U);
}

TEST(AnySegmentV3, TopKZeroAndEmptyBatchCompleteWithoutCallingEngine) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;

  ResponseStorage zero_k(1, 0);
  auto zero_request = make_request(query.data(), 1, 2, 0, context, zero_k);
  EXPECT_TRUE(segment.search(std::move(zero_request)).ok());
  EXPECT_EQ(producer->calls.load(), 0U);
  EXPECT_EQ(zero_k.offsets[0], 0U);
  EXPECT_EQ(zero_k.offsets[1], 0U);
  EXPECT_EQ(zero_k.counts[0], 0U);
  EXPECT_TRUE(zero_k.statuses[0].ok());
  EXPECT_EQ(zero_k.completeness[0], SearchCompleteness::complete_k);

  ResponseStorage empty(0, 3);
  auto empty_request = make_request(nullptr, 0, 2, 3, context, empty);
  EXPECT_TRUE(segment.search(std::move(empty_request)).ok());
  EXPECT_EQ(producer->calls.load(), 0U);
  EXPECT_EQ(empty.offsets[0], 0U);
}

TEST(AnySegmentV3, WritesFlattenedPerQueryResponseWithoutSentinels) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 4> queries{};
  SearchContext context;
  ResponseStorage storage(2, 3);
  auto request = make_request(queries.data(), 2, 2, 3, context, storage);

  ASSERT_TRUE(segment.search(std::move(request)).ok());
  EXPECT_EQ(storage.offsets, (std::vector<RowCount>{0, 1, 2}));
  EXPECT_EQ(storage.counts, (std::vector<RowCount>{1, 1}));
  EXPECT_TRUE(storage.statuses[0].ok());
  EXPECT_TRUE(storage.statuses[1].ok());
  EXPECT_EQ(storage.completeness[0], SearchCompleteness::strategy_incomplete);
  EXPECT_EQ(static_cast<std::uint64_t>(storage.hits[0].row_id), 0U);
  EXPECT_EQ(static_cast<std::uint64_t>(storage.hits[1].row_id), 1U);
}

TEST(AnySegmentV3, SyncWaitRunsSyncAdapterOnCallingThread) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);
  const auto caller = std::this_thread::get_id();

  const auto status = segment.search(std::move(request));

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(producer->execution_thread(), caller);
  EXPECT_EQ(storage.counts[0], 1U);
  EXPECT_TRUE(storage.statuses[0].ok());
}

TEST(AnySegmentV3, AsyncStartRunsSyncAdapterOffCallingThread) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);
  auto waiter = std::make_shared<CompletionWaiter>();
  const auto caller = std::this_thread::get_id();

  auto started = segment.start_search(std::move(request), SearchCompletion([waiter](Status status) {
                                        waiter->complete(std::move(status));
                                      }));
  ASSERT_TRUE(started.ok());
  auto handle = std::move(started).value();
  EXPECT_TRUE(handle.valid());
  EXPECT_TRUE(waiter->wait().ok());
  EXPECT_NE(producer->execution_thread(), caller);
  EXPECT_EQ(storage.counts[0], 1U);
}

TEST(AnySegmentV3, ConvertsEngineExceptionToStatusAndInvalidatesWholeSink) {
  auto erased = AnySegment::from_sync(std::make_shared<ThrowingSegment>());
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ResponseStorage storage(1, 2);
  storage.offsets[0] = 0;
  storage.offsets[1] = 2;
  storage.counts[0] = 2;
  auto request = make_request(query.data(), 1, 2, 2, context, storage);

  const auto status = segment.search(std::move(request));
  EXPECT_EQ(status.code(), StatusCode::internal);
  EXPECT_EQ(status.detail(), StatusDetail::engine_exception);
  EXPECT_EQ(storage.offsets, (std::vector<RowCount>{0, 0}));
  EXPECT_EQ(storage.counts, (std::vector<RowCount>{0}));
  EXPECT_EQ(storage.statuses[0].code(), StatusCode::internal);
  EXPECT_EQ(storage.completeness[0], SearchCompleteness::failed);
}

TEST(AnySegmentV3, ReturnsResourceExhaustedForExplicitZeroLease) {
  auto erased = AnySegment::from_sync(std::make_shared<BudgetSegment>());
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  context.query_scratch_lease = MemoryLease(0);
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);

  const auto status = segment.search(std::move(request));
  EXPECT_EQ(status.code(), StatusCode::resource_exhausted);
  EXPECT_EQ(status.detail(), StatusDetail::budget_denied);
  EXPECT_EQ(storage.statuses[0].code(), StatusCode::resource_exhausted);
}

TEST(AnySegmentV3, CancelKeepsExternalPinUntilExactlyOnceCompletion) {
  auto gate = std::make_shared<SlowGate>();
  auto erased = AnySegment::from_sync(std::make_shared<SlowSegment>(SlowSegment{gate}));
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);
  auto pin = std::make_shared<int>(7);
  std::weak_ptr<int> weak_pin = pin;
  request.lifetime_pin = pin;
  auto waiter = std::make_shared<CompletionWaiter>();
  auto started = segment.start_search(std::move(request), SearchCompletion([waiter](Status status) {
                                        waiter->complete(status);
                                      }));
  ASSERT_TRUE(started.ok());
  auto handle = std::move(started).value();

  {
    std::unique_lock lock(gate->mutex);
    ASSERT_TRUE(gate->condition.wait_for(lock, 5s, [&] {
      return gate->entered;
    }));
  }
  pin.reset();
  EXPECT_FALSE(weak_pin.expired());
  handle.cancel();
  handle.cancel();
  EXPECT_FALSE(weak_pin.expired());
  {
    std::lock_guard lock(waiter->mutex);
    EXPECT_FALSE(waiter->done);
  }
  {
    std::lock_guard lock(gate->mutex);
    gate->release = true;
  }
  gate->condition.notify_all();
  const auto status = waiter->wait();
  EXPECT_EQ(status.code(), StatusCode::cancelled);
  EXPECT_EQ(waiter->count, 1U);
  EXPECT_EQ(storage.counts[0], 0U);
  handle = OperationHandle{};
  // The executor releases its request copy after the completion callback returns, so give it a
  // real deadline; 100 yields elapse in microseconds on a loaded many-core host.
  for (int retry = 0; retry < 2000 && !weak_pin.expired(); ++retry) {
    std::this_thread::sleep_for(1ms);
  }
  EXPECT_TRUE(weak_pin.expired());
}

TEST(AnySegmentV3, DeadlineBeforeExecutionCompletesSafelyWithoutCallingEngine) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  context.deadline = Deadline::at(std::chrono::steady_clock::now() - 1ms);
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);

  const auto status = segment.search(std::move(request));
  EXPECT_EQ(status.code(), StatusCode::deadline_exceeded);
  EXPECT_EQ(producer->calls.load(), 0U);
  EXPECT_EQ(storage.statuses[0].code(), StatusCode::deadline_exceeded);
}

TEST(AnySegmentV3, DeadlineDuringExecutionWaitsForEngineThenInvalidatesResponse) {
  auto gate = std::make_shared<SlowGate>();
  auto erased = AnySegment::from_sync(std::make_shared<SlowSegment>(SlowSegment{gate}));
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);
  Status status;

  std::thread caller([&] {
    status = segment.search(std::move(request));
  });
  bool entered{};
  {
    std::unique_lock lock(gate->mutex);
    entered = gate->condition.wait_for(lock, 5s, [&] {
      return gate->entered;
    });
  }
  if (!entered) {
    {
      std::lock_guard lock(gate->mutex);
      gate->release = true;
    }
    gate->condition.notify_all();
    caller.join();
    FAIL() << "sync adapter did not enter the engine";
  }
  {
    std::lock_guard lock(gate->mutex);
    EXPECT_NE(gate->operation_deadline, nullptr);
    if (gate->operation_deadline != nullptr) {
      *gate->operation_deadline = Deadline::at(std::chrono::steady_clock::now() - 1ms);
    }
    gate->release = true;
  }
  gate->condition.notify_all();
  caller.join();

  EXPECT_EQ(status.code(), StatusCode::deadline_exceeded);
  EXPECT_EQ(storage.counts[0], 0U);
  EXPECT_EQ(storage.statuses[0].code(), StatusCode::deadline_exceeded);
  EXPECT_EQ(storage.completeness[0], SearchCompleteness::failed);
}

struct ManualLane {
  std::mutex mutex;
  std::condition_variable ready;
  RuntimeLane::Task task;

  static void post(void *raw, RuntimeLane::Task callback) noexcept {
    auto &lane = *static_cast<ManualLane *>(raw);
    {
      std::lock_guard lock(lane.mutex);
      lane.task = std::move(callback);
    }
    lane.ready.notify_one();
  }
};

TEST(AnySegmentV3, CompletionUsesRequestedLaneAndNeverRunsInline) {
  auto erased = AnySegment::from_sync(std::make_shared<CountingSegment>());
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ManualLane lane;
  context.lane.state = &lane;
  context.lane.post = &ManualLane::post;
  ResponseStorage storage(1, 1);
  auto request = make_request(query.data(), 1, 2, 1, context, storage);
  std::atomic_bool completed{};

  auto started = segment.start_search(std::move(request), SearchCompletion([&](Status status) {
                                        EXPECT_TRUE(status.ok());
                                        completed.store(true, std::memory_order_release);
                                      }));
  ASSERT_TRUE(started.ok());
  EXPECT_FALSE(completed.load(std::memory_order_acquire));
  RuntimeLane::Task task;
  {
    std::unique_lock lock(lane.mutex);
    ASSERT_TRUE(lane.ready.wait_for(lock, 5s, [&] {
      return static_cast<bool>(lane.task);
    }));
    task = std::move(lane.task);
  }
  EXPECT_FALSE(completed.load(std::memory_order_acquire));
  task();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));
}

TEST(AnySegmentV3, ImmediateCompletionUsesRequestedLaneAndNeverRunsInline) {
  auto producer = std::make_shared<CountingSegment>();
  auto erased = AnySegment::from_sync(producer);
  ASSERT_TRUE(erased.ok());
  auto segment = std::move(erased).value();
  std::array<float, 2> query{};
  SearchContext context;
  ManualLane lane;
  context.lane.state = &lane;
  context.lane.post = &ManualLane::post;
  ResponseStorage storage(1, 0);
  auto request = make_request(query.data(), 1, 2, 0, context, storage);
  std::atomic_bool completed{};

  auto started = segment.start_search(std::move(request), SearchCompletion([&](Status status) {
                                        EXPECT_TRUE(status.ok());
                                        completed.store(true, std::memory_order_release);
                                      }));
  ASSERT_TRUE(started.ok());
  EXPECT_EQ(producer->calls.load(), 0U);
  EXPECT_FALSE(completed.load(std::memory_order_acquire));
  RuntimeLane::Task task;
  {
    std::unique_lock lock(lane.mutex);
    ASSERT_TRUE(lane.ready.wait_for(lock, 5s, [&] {
      return static_cast<bool>(lane.task);
    }));
    task = std::move(lane.task);
  }
  EXPECT_FALSE(completed.load(std::memory_order_acquire));
  task();
  EXPECT_TRUE(completed.load(std::memory_order_acquire));
}

}  // namespace
