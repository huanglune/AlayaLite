// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"
#include "utils/test_paths.hpp"

// Focused coverage for the explicit two-phase rotate-to-successor API
// (Collection::prepare_successor() / Collection::rotate_to_successor()).
// Collection::seal() itself, its four crash fail points, and concurrent
// search/write safety across the write-side (active segment) rotation are
// already covered extensively by collection_facade_test.cpp and
// collection_facade_stress_test.cpp; this file targets what only exists
// once prepare and rotate are separate calls: the routing table staying on
// the predecessor until rotate_to_successor() runs, concurrent search
// safety spanning that window, and crash recovery when a process dies
// after prepare_successor() durably returns but before rotate_to_successor()
// is ever invoked.

namespace alaya {
namespace internal::collection {

class CollectionTestAccess {
 public:
  [[nodiscard]] static auto pin_epoch(const Collection &collection) -> RoutingSnapshotPtr {
    return collection.implementation_->pin_routing_snapshot();
  }
};

}  // namespace internal::collection

namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-rotate-collection-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

[[nodiscard]] auto flat_options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions result;
  result.root = root;
  result.dim = 2;
  result.metric = core::Metric::l2;
  result.scalar_type = core::ScalarType::float32;
  result.target_algorithm = core::algorithm::flat;
  return result;
}

[[nodiscard]] auto item(std::string id, const std::array<float, 2> &vector) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(std::move(id));
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
  return result;
}

[[nodiscard]] auto stress_item(std::string id, const std::array<float, 4> &vector) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(std::move(id));
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
  return result;
}

TEST(CollectionRotate, PrepareThenRotateSwitchesRoutingAtomically) {
  TemporaryDirectory temporary;
  auto created = Collection::create(flat_options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::vector<std::array<float, 2>> vectors{
      {0.0F, 0.0F}, {1.0F, 0.0F}, {2.0F, 0.0F}, {3.0F, 0.0F}};
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    ASSERT_TRUE(collection->add(item("rotate-" + std::to_string(index), vectors[index])).ok());
  }

  auto handle = collection->prepare_successor();
  ASSERT_TRUE(handle.ok()) << handle.status().diagnostic();
  EXPECT_TRUE(handle.value().ready());
  EXPECT_EQ(handle.value().predecessor_segment_ids, (std::vector<std::uint64_t>{2U}));
  EXPECT_EQ(handle.value().successor_segment_id, 4U);

  // The manifest already durably reflects the successor as sealed and the
  // predecessor as gc_pending (stats() reads the on-disk manifest)...
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
  // ...but query routing has not moved: the predecessor segment is still
  // routed and the successor is not, until rotate_to_successor() runs.
  {
    auto epoch = internal::collection::CollectionTestAccess::pin_epoch(*collection);
    EXPECT_NE(epoch->find_segment(2U, 1U), nullptr);
    EXPECT_EQ(epoch->find_segment(4U, 1U), nullptr);
  }

  const std::array<float, 2> query{};
  auto before = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(before.ok()) << before.status().diagnostic();
  EXPECT_EQ(before.value().ids.size(), 4U);

  auto sealed = collection->rotate_to_successor(handle.value());
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().source_segment_id, 2U);
  EXPECT_EQ(sealed.value().sealed_segment_id, 4U);
  EXPECT_EQ(sealed.value().sealed_rows, 4U);
  EXPECT_GT(sealed.value().sealed_bytes, 0U);

  {
    auto epoch = internal::collection::CollectionTestAccess::pin_epoch(*collection);
    EXPECT_EQ(epoch->find_segment(2U, 1U), nullptr);
    EXPECT_NE(epoch->find_segment(4U, 1U), nullptr);
  }

  auto after = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().ids, before.value().ids);
  EXPECT_EQ(after.value().distances, before.value().distances);
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);

  // The new active segment (3) is still empty, so there is nothing left to
  // seal until another write lands.
  auto empty_handle = collection->prepare_successor();
  EXPECT_FALSE(empty_handle.ok());

  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionRotate, PrepareSuccessorTwiceWithoutRotatingReturnsConflict) {
  TemporaryDirectory temporary;
  auto created = Collection::create(flat_options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  ASSERT_TRUE(collection->add(item("only-row", std::array<float, 2>{0.0F, 0.0F})).ok());

  auto handle = collection->prepare_successor();
  ASSERT_TRUE(handle.ok()) << handle.status().diagnostic();

  auto second = collection->prepare_successor();
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().code(), core::StatusCode::conflict);

  // The pending rotation is still valid and can complete normally.
  auto rotated = collection->rotate_to_successor(handle.value());
  ASSERT_TRUE(rotated.ok()) << rotated.status().diagnostic();
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionRotate, RotateToSuccessorRejectsStaleOrForeignHandle) {
  TemporaryDirectory temporary;
  auto created = Collection::create(flat_options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  ASSERT_TRUE(collection->add(item("row-a", std::array<float, 2>{0.0F, 0.0F})).ok());

  auto handle = collection->prepare_successor();
  ASSERT_TRUE(handle.ok()) << handle.status().diagnostic();
  auto rotated = collection->rotate_to_successor(handle.value());
  ASSERT_TRUE(rotated.ok()) << rotated.status().diagnostic();

  // Replaying the same (now-consumed) handle must not silently re-run the
  // switch or touch a segment that is no longer routed.
  auto replay = collection->rotate_to_successor(handle.value());
  EXPECT_FALSE(replay.ok());

  // A handle naming plausible-looking but never-prepared identities is
  // rejected the same way, without side effects.
  CollectionRotationHandle bogus;
  bogus.predecessor_segment_ids = {999U};
  bogus.successor_segment_id = 998U;
  bogus.successor_generation = 1U;
  auto bogus_result = collection->rotate_to_successor(bogus);
  EXPECT_FALSE(bogus_result.ok());

  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionRotate, ConcurrentSearchAndWriteAcrossPrepareAndRotateStayConsistent) {
  TemporaryDirectory temporary;
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

  auto handle = collection->prepare_successor();
  ASSERT_TRUE(handle.ok()) << handle.status().diagnostic();

  std::atomic_bool stop{};
  std::atomic_uint64_t failures{};
  const std::array<float, 4> query{};
  std::thread searcher([&] {
    while (!stop.load(std::memory_order_acquire)) {
      auto result =
          collection->search(core::TypedTensorView::contiguous(query.data(), 1, query.size()), 256);
      // rotate_to_successor() checkpoints internally (same as seal() always
      // has), and SegmentedCollection::checkpoint() briefly gates new
      // admissions (ControlPlaneGate) while it drains in-flight operations.
      // That gate reports a retryable conflict rather than a real failure;
      // see SegmentedCollection::closed_status()'s comment.
      if (!result.ok() && result.status().code() == core::StatusCode::conflict) {
        std::this_thread::yield();
        continue;
      }
      if (!result.ok() || result.value().ids.size() != result.value().distances.size() ||
          result.value().valid_counts.size() != 1 || result.value().valid_counts[0] > 256) {
        ++failures;
      }
    }
  });
  std::thread writer([&] {
    for (std::uint64_t index = 0; index < 32; ++index) {
      const std::array<float, 4> vector{static_cast<float>(1000 + index), 4.0F, 5.0F, 6.0F};
      for (;;) {
        auto receipt = collection->add(stress_item("successor-" + std::to_string(index), vector));
        if (receipt.ok()) {
          if (!receipt.value().searchable) {
            ++failures;
          }
          break;
        }
        if (receipt.status().code() != core::StatusCode::conflict) {
          ++failures;
          break;
        }
        std::this_thread::yield();
      }
    }
  });

  // Widen the window between prepare_successor() having already published
  // the successor durably and rotate_to_successor() actually switching
  // routing, so searchers/writers really do straddle both phases.
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  auto sealed = collection->rotate_to_successor(handle.value());
  writer.join();
  stop.store(true, std::memory_order_release);
  searcher.join();

  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(sealed.value().sealed_rows, 128U);
  EXPECT_EQ(failures.load(), 0U);
  EXPECT_EQ(collection->size(), 160U);
  auto final_result =
      collection->search(core::TypedTensorView::contiguous(query.data(), 1, query.size()), 256);
  ASSERT_TRUE(final_result.ok()) << final_result.status().diagnostic();
  EXPECT_EQ(final_result.value().ids.size(), 160U);
  ASSERT_TRUE(collection->close().ok());
}

#ifndef _WIN32
TEST(CollectionRotate, CrashAfterPrepareSuccessorAutoCompletesOnReopen) {
  const auto root =
      test::tmp_root() / ("alaya-rotate-crash-" + std::to_string(platform::get_pid()));
  std::filesystem::remove_all(root);

  const auto child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    auto created = Collection::create(flat_options(root));
    if (!created.ok()) {
      ::_exit(80);
    }
    if (!created.value()->add(item("rotate-crash-a", std::array<float, 2>{0.0F, 0.0F})).ok() ||
        !created.value()->add(item("rotate-crash-b", std::array<float, 2>{1.0F, 0.0F})).ok()) {
      ::_exit(81);
    }
    auto handle = created.value()->prepare_successor();
    if (!handle.ok()) {
      ::_exit(82);
    }
    // Crash right after prepare_successor() durably returns: the manifest
    // already names a sealed successor, but rotate_to_successor() is never
    // called in this process.
    ::kill(::getpid(), SIGKILL);
    ::_exit(99);
  }
  int child_status{};
  ASSERT_EQ(::waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFSIGNALED(child_status));
  EXPECT_EQ(WTERMSIG(child_status), SIGKILL);

  auto recovered = Collection::open(root);
  ASSERT_TRUE(recovered.ok()) << recovered.status().diagnostic();
  auto collection = std::move(recovered).value();
  // The rotation self-completes on reopen via automatic recovery: no
  // explicit rotate_to_successor() call was ever made in the crashed
  // process, yet the successor is live and the predecessor retired.
  EXPECT_EQ(collection->size(), 2U);
  EXPECT_EQ(collection->stats().sealed_segments_count, 1U);
  ASSERT_TRUE(collection->add(item("rotate-crash-c", std::array<float, 2>{2.0F, 0.0F})).ok());
  const std::array<float, 2> query{};
  auto result = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  EXPECT_EQ(result.value().ids.size(), 3U);
  ASSERT_TRUE(collection->checkpoint().ok());
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto repeated = Collection::open(root);
  ASSERT_TRUE(repeated.ok()) << repeated.status().diagnostic();
  EXPECT_EQ(repeated.value()->size(), 3U);
  EXPECT_EQ(repeated.value()->stats().sealed_segments_count, 1U);
  ASSERT_TRUE(repeated.value()->close().ok());
  std::filesystem::remove_all(root);
}
#endif

}  // namespace
}  // namespace alaya
