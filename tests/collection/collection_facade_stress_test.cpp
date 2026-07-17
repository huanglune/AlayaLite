// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

class StressTemporaryDirectory {
 public:
  StressTemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-canonical-stress-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }
  ~StressTemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

[[nodiscard]] auto stress_item(std::string id, const std::array<float, 4> &vector)
    -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(id);
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
  return result;
}

TEST(CollectionFacadeStress, ConcurrentSearchAddAndCheckpointRemainConsistent) {
  StressTemporaryDirectory temporary;
  CollectionOptions options;
  options.root = temporary.path();
  options.dim = 4;
  options.metric = core::Metric::l2;
  options.scalar_type = core::ScalarType::float32;
  // Never sealed in this test, so the target algorithm is inert
  // configuration; flat needs no quantization pairing.
  options.target_algorithm = core::algorithm::flat;
  auto created = Collection::create(options);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 4> seed{0.0F, 0.0F, 0.0F, 0.0F};
  ASSERT_TRUE(collection->add(stress_item("seed", seed)).ok());

  std::atomic<bool> writer_done{};
  std::atomic<std::uint64_t> failures{};
  std::mutex diagnostics_mutex;
  std::vector<std::string> diagnostics;
  const auto fail = [&](std::string diagnostic) {
    ++failures;
    std::lock_guard lock(diagnostics_mutex);
    diagnostics.push_back(std::move(diagnostic));
  };
  std::thread writer([&] {
    for (std::uint64_t index = 0; index < 64; ++index) {
      const std::array<float, 4> vector{static_cast<float>(index + 1),
                                        static_cast<float>(index % 7),
                                        1.0F,
                                        -1.0F};
      for (;;) {
        auto receipt = collection->add(stress_item("row-" + std::to_string(index), vector));
        if (receipt.ok()) {
          if (!receipt.value().searchable) {
            fail("write receipt was not searchable");
          }
          break;
        }
        if (receipt.status().code() != core::StatusCode::conflict) {
          fail("write: " + receipt.status().diagnostic());
          break;
        }
        std::this_thread::yield();
      }
    }
    writer_done.store(true, std::memory_order_release);
  });

  std::vector<std::thread> searchers;
  for (unsigned thread = 0; thread < 3; ++thread) {
    searchers.emplace_back([&] {
      do {
        auto result =
            collection->search(core::TypedTensorView::contiguous(seed.data(), 1, seed.size()), 128);
        if (!result.ok() && result.status().code() == core::StatusCode::conflict) {
          std::this_thread::yield();
          continue;
        }
        if (!result.ok() || result.value().offsets.size() != 2 ||
            result.value().valid_counts.size() != 1 ||
            result.value().ids.size() != result.value().distances.size() ||
            result.value().offsets[1] != result.value().valid_counts[0] ||
            result.value().valid_counts[0] > 128) {
          fail(result.ok() ? "search response invariant"
                           : "search: " + result.status().diagnostic());
        }
      } while (!writer_done.load(std::memory_order_acquire));
    });
  }

  std::thread checkpointer([&] {
    for (unsigned index = 0; index < 12; ++index) {
      if (!collection->checkpoint().ok()) {
        fail("checkpoint failed");
      }
    }
  });

  writer.join();
  for (auto &searcher : searchers) {
    searcher.join();
  }
  checkpointer.join();
  for (const auto &diagnostic : diagnostics) {
    ADD_FAILURE() << diagnostic;
  }
  EXPECT_EQ(failures.load(), 0U);
  EXPECT_EQ(collection->size(), 65U);
  EXPECT_EQ(collection->stats().accepted_count, 65U);
  EXPECT_EQ(collection->stats().pending_count, 0U);
  ASSERT_TRUE(collection->checkpoint().ok());
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->size(), 65U);
  EXPECT_EQ(reopened.value()->stats().pending_count, 0U);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionFacadeStress, ConcurrentSealSearchAndWritesRouteToSuccessor) {
  StressTemporaryDirectory temporary;
  CollectionOptions options;
  options.root = temporary.path();
  options.dim = 4;
  options.metric = core::Metric::l2;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::flat;
  auto created = Collection::create(options);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  for (std::uint64_t index = 0; index < 128; ++index) {
    const std::array<float, 4> vector{static_cast<float>(index), 1.0F, 2.0F, 3.0F};
    ASSERT_TRUE(collection->add(stress_item("source-" + std::to_string(index), vector)).ok());
  }

  std::atomic_bool successor_online{};
  std::atomic_bool release_builder{};
  std::atomic_uint64_t failures{};
  core::Result<CollectionSealReceipt> seal_result =
      core::Status::error(core::StatusCode::internal,
                          core::OperationStage::freeze,
                          core::StatusDetail::none);
  std::thread sealer([&] {
    CollectionSealOptions seal_options;
    seal_options.fail_point = CollectionSealFailPoint::after_successor_switch;
    seal_options.failpoint_hook = [&](CollectionSealFailPoint) {
      successor_online.store(true, std::memory_order_release);
      while (!release_builder.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
    };
    seal_result = collection->seal(std::move(seal_options));
  });
  while (!successor_online.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const std::array<float, 4> query{};
  std::thread searcher([&] {
    for (unsigned round = 0; round < 64; ++round) {
      auto result =
          collection->search(core::TypedTensorView::contiguous(query.data(), 1, query.size()), 256);
      if (!result.ok() || result.value().ids.size() != result.value().distances.size() ||
          result.value().valid_counts.size() != 1 || result.value().valid_counts[0] > 256) {
        ++failures;
      }
    }
  });
  std::thread writer([&] {
    for (std::uint64_t index = 0; index < 32; ++index) {
      const std::array<float, 4> vector{static_cast<float>(1000 + index), 4.0F, 5.0F, 6.0F};
      auto receipt = collection->add(stress_item("successor-" + std::to_string(index), vector));
      if (!receipt.ok() || !receipt.value().searchable) {
        ++failures;
      }
    }
  });
  writer.join();
  searcher.join();
  release_builder.store(true, std::memory_order_release);
  sealer.join();

  ASSERT_TRUE(seal_result.ok()) << seal_result.status().diagnostic();
  EXPECT_EQ(seal_result.value().source_segment_id, 2U);
  EXPECT_EQ(seal_result.value().successor_segment_id, 3U);
  EXPECT_EQ(seal_result.value().sealed_rows, 128U);
  EXPECT_EQ(failures.load(), 0U);
  EXPECT_EQ(collection->size(), 160U);
  auto final =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, query.size()), 256);
  ASSERT_TRUE(final.ok()) << final.status().diagnostic();
  EXPECT_EQ(final.value().ids.size(), 160U);
  ASSERT_TRUE(collection->close().ok());
}

}  // namespace
}  // namespace alaya
