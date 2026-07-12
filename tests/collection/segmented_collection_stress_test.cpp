// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fake_mutable_segment.hpp"

namespace alaya::internal::collection {
namespace {

[[nodiscard]] auto open_stress_collection(std::shared_ptr<test::FakeMutableSegment> &producer)
    -> std::shared_ptr<SegmentedCollection> {
  producer = std::make_shared<test::FakeMutableSegment>();
  auto erased = test::make_fake_mutable_any(producer);
  EXPECT_TRUE(erased.ok());
  SegmentRegistration registration;
  registration.segment_id = test::FakeMutableSegment::kSegmentId;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  EXPECT_TRUE(opened.ok());
  return std::move(opened).value();
}

[[nodiscard]] auto upsert(const std::shared_ptr<SegmentedCollection> &collection,
                          std::string id,
                          const std::array<float, 2> &vector) -> core::Result<MutationReceipt> {
  WriteRequest request;
  request.logical_id = core::LogicalId::from_utf8(id);
  request.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  request.mode = WriteMode::upsert;
  core::MutationContext context;
  return collection->write(request, context);
}

TEST(SegmentedCollectionStress, ConcurrentSearchAndMutationPreserveAdmittedWatermarks) {
  std::shared_ptr<test::FakeMutableSegment> producer;
  const auto collection = open_stress_collection(producer);
  ASSERT_TRUE(upsert(collection, "hot-a", {0.0F, 0.0F}).ok());
  ASSERT_TRUE(upsert(collection, "hot-b", {1.0F, 1.0F}).ok());

  constexpr std::uint32_t kSearchThreads = 4;
  constexpr std::uint32_t kSearchIterations = 150;
  constexpr std::uint32_t kMutationIterations = 150;
  std::atomic_bool start{};
  std::atomic_bool failed{};
  std::vector<std::thread> workers;

  for (std::uint32_t thread_index = 0; thread_index < kSearchThreads; ++thread_index) {
    workers.emplace_back([&, thread_index] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      std::uint64_t previous_watermark{};
      const std::array<float, 2> query{static_cast<float>(thread_index), 0.0F};
      for (std::uint32_t iteration = 0; iteration < kSearchIterations; ++iteration) {
        core::SearchContext context;
        CollectionSearchRequest request;
        request.queries = core::TypedTensorView::contiguous(query.data(), 1, 2);
        request.options.top_k = 8;
        request.context = &context;
        auto result = collection->search(request);
        if (!result.ok() || result.value().visibility_watermark < previous_watermark) {
          failed.store(true, std::memory_order_release);
          return;
        }
        previous_watermark = result.value().visibility_watermark;
        const auto &hits = result.value().queries[0].hits;
        for (std::size_t index = 0; index < hits.size(); ++index) {
          if (hits[index].upsert_sequence > result.value().visibility_watermark) {
            failed.store(true, std::memory_order_release);
            return;
          }
          for (std::size_t other = index + 1; other < hits.size(); ++other) {
            if (hits[index].logical_id == hits[other].logical_id) {
              failed.store(true, std::memory_order_release);
              return;
            }
          }
        }
      }
    });
  }

  workers.emplace_back([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (std::uint32_t iteration = 0; iteration < kMutationIterations; ++iteration) {
      const std::array<float, 2> vector{static_cast<float>(iteration), 0.0F};
      if (!upsert(collection, "hot-a", vector).ok()) {
        failed.store(true, std::memory_order_release);
        return;
      }
    }
  });

  workers.emplace_back([&] {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    const auto id = core::LogicalId::from_utf8("hot-b");
    for (std::uint32_t iteration = 0; iteration < kMutationIterations; ++iteration) {
      const std::array<float, 2> vector{0.0F, static_cast<float>(iteration)};
      if (!upsert(collection, "hot-b", vector).ok()) {
        failed.store(true, std::memory_order_release);
        return;
      }
      if ((iteration % 3U) == 0) {
        core::MutationContext context;
        auto erased = collection->erase(id, context);
        if (!erased.ok()) {
          failed.store(true, std::memory_order_release);
          return;
        }
      }
    }
  });

  start.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_FALSE(failed.load(std::memory_order_acquire));
  EXPECT_EQ(producer->maximum_active_mutations(), 1U);
  EXPECT_TRUE(collection->close().ok());
  EXPECT_TRUE(collection->drain().ok());
}

}  // namespace
}  // namespace alaya::internal::collection
