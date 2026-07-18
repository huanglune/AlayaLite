// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "qg_wal_test_support.hpp"

namespace alaya::laser {
namespace {

using waltest::kDeg;
using waltest::kDim;
using waltest::WalTinyIndex;

constexpr size_t kBaseN = 256;
constexpr size_t kMaxPoints = kBaseN + 64;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string name) {
    static std::atomic_uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("qg-maintenance-concurrency-" + std::move(name) + "-" + std::to_string(::getpid()) +
             "-" + std::to_string(++serial));
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

class FailpointGate {
 public:
  explicit FailpointGate(SegmentOpFailPoint target, bool throw_on_release = false)
      : target_(target), throw_on_release_(throw_on_release) {}

  void operator()(SegmentOpFailPoint observed) {
    if (observed != target_ || claimed_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    entered_.store(true, std::memory_order_release);
    while (!released_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    if (throw_on_release_) {
      throw std::runtime_error("injected maintenance install failure");
    }
  }

  [[nodiscard]] auto wait_until_entered() -> bool {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered_.load(std::memory_order_acquire)) {
      if (std::chrono::steady_clock::now() >= deadline) {
        return false;
      }
      std::this_thread::yield();
    }
    return true;
  }

  void release() { released_.store(true, std::memory_order_release); }

 private:
  SegmentOpFailPoint target_{};
  bool throw_on_release_{};
  std::atomic_bool claimed_{};
  std::atomic_bool entered_{};
  std::atomic_bool released_{};
};

struct Session {
  QuantizedGraph qg;
  std::unique_ptr<QGUpdater> updater;

  explicit Session(const std::string &prefix,
                   std::function<void(SegmentOpFailPoint)> hook = {},
                   size_t cache_cap_pages = 0,
                   bool resident_arena = false,
                   std::function<void(uint64_t, size_t)> before_index_write_hook = {},
                   bool write_cache = true,
                   bool enable_wal = true)
      : qg(kBaseN, kDeg, kDim, kDim) {
    qg.load_disk_index(prefix.c_str(), 0.0F, /*recovery_mode=*/true);
    if (resident_arena) {
      qg.ensure_resident_arena();
    }
    qg.set_params(64, 1, 1);
    UpdateParams params;
    params.enable_wal = enable_wal;
    params.max_points = kMaxPoints;
    params.write_cache = write_cache;
    params.cache_cap_pages = cache_cap_pages == 0 ? params.cache_cap_pages : cache_cap_pages;
    params.failpoint_hook = std::move(hook);
    params.before_index_write_hook = std::move(before_index_write_hook);
    updater = std::make_unique<QGUpdater>(qg, std::move(params));
  }
};

[[nodiscard]] auto checked_search(QGUpdater &updater, const float *query) -> std::vector<PID> {
  updater.ensure_readable();
  auto result = updater.search(query, 16, kBaseN);
  updater.ensure_readable();
  return result;
}

[[nodiscard]] auto kind_counts(const std::string &prefix) -> std::array<size_t, 9> {
  std::array<size_t, 9> counts{};
  const auto wal_path = prefix + waltest::index_suffix() + ".opwal";
  alaya::wal::WalFile::visit_frames(wal_path, [&](const alaya::wal::ScannedFrame &frame) {
    const auto op = decode_segment_op(frame.payload);
    ++counts[static_cast<size_t>(op.kind)];
    return true;
  });
  return counts;
}

struct StateFingerprint {
  uint64_t page_hash{};
  uint64_t live_count{};
  uint64_t free_count{};
  uint64_t free_head{};
  uint64_t epoch{};
  PID entry{};

  auto operator==(const StateFingerprint &) const -> bool = default;
};

void hash_bytes(uint64_t &hash, const char *data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<unsigned char>(data[i]);
    hash *= 1099511628211ULL;
  }
}

[[nodiscard]] auto fingerprint(QGUpdater &updater) -> StateFingerprint {
  StateFingerprint result;
  result.page_hash = 1469598103934665603ULL;
  for (size_t page = 0; page < updater.file_pages(); ++page) {
    const auto bytes = updater.debug_read_page(page);
    hash_bytes(result.page_hash, bytes.data(), bytes.size());
  }
  result.live_count = updater.live_count();
  result.free_count = updater.free_count();
  result.free_head = updater.free_list_head();
  result.epoch = updater.last_completed_consolidate_epoch();
  result.entry = updater.entry_point();
  return result;
}

[[nodiscard]] auto reopened_fingerprint(const std::string &prefix) -> StateFingerprint {
  Session session(prefix);
  EXPECT_FALSE(session.updater->is_poisoned());
  return fingerprint(*session.updater);
}

[[nodiscard]] auto wait_or_release(FailpointGate &gate) -> bool {
  if (!gate.wait_until_entered()) {
    gate.release();
    return false;
  }
  return true;
}

TEST(QgMaintenanceConcurrency, BuildWindowKeepsOldCommittedSearchVisible) {
  TemporaryDirectory root("build-search");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7101);
  FailpointGate gate(SegmentOpFailPoint::after_consolidate_begin_fsync);
  Session session(base.prefix, [&](auto point) {
    gate(point);
  });
  auto &updater = *session.updater;
  ASSERT_GT(updater.file_pages(), 1U);
  updater.tombstone(1);
  const auto query = waltest::make_data(1, kDim, 7102);
  const auto old_hits = checked_search(updater, query.data());

  auto maintenance = std::async(std::launch::async, [&] {
    updater.consolidate(1, 0, true, false);
  });
  ASSERT_TRUE(wait_or_release(gate));
  for (size_t attempt = 0; attempt < 32; ++attempt) {
    EXPECT_EQ(checked_search(updater, query.data()), old_hits);
  }
  EXPECT_GT(updater.stats().query_seqlock_read_calls, 0U);

  gate.release();
  EXPECT_NO_THROW(maintenance.get());
  EXPECT_FALSE(updater.is_poisoned());
  EXPECT_EQ(updater.free_count(), 1U);
}

TEST(QgMaintenanceConcurrency, OddBeforeWriteBlocksReaderAndClosesEven) {
  TemporaryDirectory root("odd-before-write");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7201);
  FailpointGate gate(SegmentOpFailPoint::after_consolidate_install_version_odd);
  Session session(base.prefix, [&](auto point) {
    gate(point);
  });
  auto &updater = *session.updater;
  updater.tombstone(1);
  const auto query = waltest::make_data(1, kDim, 7202);
  const auto old_page = updater.debug_read_page(0);

  auto maintenance = std::async(std::launch::async, [&] {
    updater.consolidate(1, 0, true, false);
  });
  ASSERT_TRUE(wait_or_release(gate));
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 1U);
  auto page_reader = std::async(std::launch::async, [&] {
    return updater.debug_read_page(0);
  });
  auto search_reader = std::async(std::launch::async, [&] {
    return checked_search(updater, query.data());
  });
  EXPECT_EQ(page_reader.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  EXPECT_EQ(search_reader.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);

  gate.release();
  EXPECT_NO_THROW(maintenance.get());
  const auto observed = page_reader.get();
  const auto final_page = updater.debug_read_page(0);
  EXPECT_NE(final_page, old_page);
  EXPECT_EQ(observed, final_page);
  EXPECT_FALSE(search_reader.get().empty());
  EXPECT_GT(updater.stats().query_seqlock_read_retries, 0U);
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 0U);
}

TEST(QgMaintenanceConcurrency, WriteBeforeEvenKeepsDiskArenaAndReaderAtomic) {
  TemporaryDirectory root("write-before-even");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7301);
  FailpointGate gate(SegmentOpFailPoint::after_consolidate_install_write_before_even);
  Session session(
      base.prefix,
      [&](auto point) {
        gate(point);
      },
      /*cache_cap_pages=*/0,
      /*resident_arena=*/true);
  auto &updater = *session.updater;
  updater.tombstone(1);
  const auto query = waltest::make_data(1, kDim, 7302);

  auto maintenance = std::async(std::launch::async, [&] {
    updater.consolidate(1, 0, true, false);
  });
  ASSERT_TRUE(wait_or_release(gate));
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 1U);
  const auto disk_page = updater.debug_read_disk_page(0);
  const auto arena_rows = updater.debug_read_arena_rows(0);
  ASSERT_LE(arena_rows.size(), disk_page.size());
  EXPECT_TRUE(std::equal(arena_rows.begin(), arena_rows.end(), disk_page.begin()));

  auto page_reader = std::async(std::launch::async, [&] {
    return updater.debug_read_page(0);
  });
  auto search_reader = std::async(std::launch::async, [&] {
    return checked_search(updater, query.data());
  });
  EXPECT_EQ(page_reader.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);
  EXPECT_EQ(search_reader.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);

  gate.release();
  EXPECT_NO_THROW(maintenance.get());
  EXPECT_EQ(page_reader.get(), updater.debug_read_page(0));
  EXPECT_FALSE(search_reader.get().empty());
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 0U);
}

TEST(QgMaintenanceConcurrency, InstallFailureUnblocksReaderAndReopenRollsForward) {
  TemporaryDirectory root("install-failure");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7401);
  FailpointGate gate(SegmentOpFailPoint::after_consolidate_install_write_before_even,
                     /*throw_on_release=*/true);
  {
    Session session(base.prefix, [&](auto point) {
      gate(point);
    });
    auto &updater = *session.updater;
    updater.tombstone(1);
    const auto query = waltest::make_data(1, kDim, 7402);
    auto maintenance = std::async(std::launch::async, [&] {
      try {
        updater.consolidate(1, 0, true, false);
        return false;
      } catch (...) {
        return true;
      }
    });
    ASSERT_TRUE(wait_or_release(gate));
    EXPECT_EQ(updater.debug_page_version(0) & 1U, 1U);
    auto reader = std::async(std::launch::async, [&] {
      try {
        (void)checked_search(updater, query.data());
        return false;
      } catch (...) {
        return true;
      }
    });
    EXPECT_EQ(reader.wait_for(std::chrono::milliseconds(30)), std::future_status::timeout);

    gate.release();
    EXPECT_TRUE(maintenance.get());
    EXPECT_TRUE(reader.get());
    EXPECT_TRUE(updater.is_poisoned());
    EXPECT_EQ(updater.debug_page_version(0) & 1U, 0U);
  }

  const auto first = reopened_fingerprint(base.prefix);
  EXPECT_EQ(first.live_count, kBaseN - 1);
  EXPECT_EQ(first.free_count, 1U);
  EXPECT_EQ(first.epoch, 1U);
  const auto second = reopened_fingerprint(base.prefix);
  EXPECT_EQ(second, first);
}

TEST(QgMaintenanceConcurrency, StatvfsFailureBeforeBeginDoesNotLatch) {
  TemporaryDirectory root("statvfs");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7501);
  std::atomic_bool armed{true};
  Session session(base.prefix, [&](SegmentOpFailPoint point) {
    if (point == SegmentOpFailPoint::before_consolidate_statvfs &&
        armed.exchange(false, std::memory_order_acq_rel)) {
      throw std::system_error(std::make_error_code(std::errc::io_error),
                              "injected statvfs failure");
    }
  });
  auto &updater = *session.updater;
  updater.tombstone(1);
  EXPECT_THROW(updater.consolidate(1, 0, true, false), std::system_error);
  EXPECT_FALSE(updater.is_poisoned());
  EXPECT_EQ(kind_counts(base.prefix)[static_cast<size_t>(SegmentOpKind::consolidate_begin)], 0U);

  const auto replacement = waltest::make_data(1, kDim, 7502);
  const auto appended = updater.allocate_and_insert(replacement.data());
  updater.publish(updater.allocated_points());
  EXPECT_GE(appended, kBaseN);
  EXPECT_FALSE(checked_search(updater, replacement.data()).empty());
  EXPECT_NO_THROW(updater.checkpoint());
  EXPECT_NO_THROW(updater.consolidate(1, 0, true, false));
  EXPECT_FALSE(updater.is_poisoned());
  EXPECT_EQ(updater.free_count(), 1U);
}

TEST(QgMaintenanceConcurrency, BaselineFlushFailurePoisonsClosesEvenAndPrecedesBegin) {
  TemporaryDirectory root("baseline-flush");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7551);
  std::atomic_bool armed{};
  std::atomic_bool preflight_seen{};
  Session session(
      base.prefix,
      [&](SegmentOpFailPoint point) {
        if (point == SegmentOpFailPoint::before_consolidate_statvfs &&
            armed.load(std::memory_order_acquire)) {
          preflight_seen.store(true, std::memory_order_release);
        }
      },
      /*cache_cap_pages=*/0,
      /*resident_arena=*/false,
      [&](uint64_t, size_t) {
        if (armed.exchange(false, std::memory_order_acq_rel)) {
          throw std::system_error(std::make_error_code(std::errc::no_space_on_device),
                                  "injected baseline pwrite ENOSPC");
        }
      });
  auto &updater = *session.updater;

  // Activate maintenance first, then leave a committed tombstone page dirty in
  // the shared write cache so the next consolidate has real baseline work.
  ASSERT_NO_THROW(updater.consolidate(1, 0, true, false));
  updater.tombstone(1);
  const auto counts_before = kind_counts(base.prefix);
  preflight_seen.store(false, std::memory_order_release);
  armed.store(true, std::memory_order_release);

  EXPECT_THROW(updater.consolidate(1, 0, true, false), std::system_error);
  EXPECT_TRUE(preflight_seen.load(std::memory_order_acquire));
  EXPECT_TRUE(updater.is_poisoned());
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 0U);
  const auto counts_after = kind_counts(base.prefix);
  EXPECT_EQ(counts_after[static_cast<size_t>(SegmentOpKind::consolidate_begin)],
            counts_before[static_cast<size_t>(SegmentOpKind::consolidate_begin)]);

  // The raw page reader bypasses the poison gate and therefore proves the
  // seqlock itself was closed instead of merely observing ensure_readable().
  auto reader = std::async(std::launch::async, [&] {
    return updater.debug_read_page(0).size();
  });
  EXPECT_EQ(reader.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(reader.get(), updater.debug_read_page(0).size());
  EXPECT_THROW(updater.ensure_readable(), std::runtime_error);
}

TEST(QgMaintenanceConcurrency, DirectPageWriteFailurePoisonsAndClosesEven) {
  TemporaryDirectory root("direct-write");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7552);
  std::atomic_bool armed{};
  Session session(
      base.prefix,
      {},
      /*cache_cap_pages=*/0,
      /*resident_arena=*/false,
      [&](uint64_t, size_t) {
        if (armed.exchange(false, std::memory_order_acq_rel)) {
          throw std::system_error(std::make_error_code(std::errc::io_error),
                                  "injected direct page pwrite failure");
        }
      },
      /*write_cache=*/false,
      /*enable_wal=*/false);
  auto &updater = *session.updater;
  armed.store(true, std::memory_order_release);

  EXPECT_THROW(updater.tombstone(1), std::system_error);
  EXPECT_TRUE(updater.is_poisoned());
  EXPECT_EQ(updater.debug_page_version(0) & 1U, 0U);
  auto reader = std::async(std::launch::async, [&] {
    return updater.debug_read_page(0).size();
  });
  EXPECT_EQ(reader.wait_for(std::chrono::seconds(1)), std::future_status::ready);
  EXPECT_EQ(reader.get(), updater.debug_read_page(0).size());
}

TEST(QgMaintenanceConcurrency, ReclaimOverlayHonorsPageCap) {
  TemporaryDirectory root("reclaim-cap");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7553);
  Session session(base.prefix, {}, /*cache_cap_pages=*/1);
  auto &updater = *session.updater;
  for (size_t raw_id = 0; raw_id < kBaseN; ++raw_id) {
    updater.tombstone(static_cast<PID>(raw_id));
  }

  ASSERT_NO_THROW(updater.consolidate(1, 0, true, false));
  EXPECT_EQ(updater.free_count(), kBaseN);
  EXPECT_GE(updater.stats().maintenance_peak_overlay_pages, 1U);
  EXPECT_LE(updater.stats().maintenance_peak_overlay_pages, updater.cache_cap_pages() + 1);

  // A second epoch exercises the existing free-chain dependency walk as well
  // as the canonical next-pointer rewrite, with no newly eligible rows.
  ASSERT_NO_THROW(updater.consolidate(1, 0, true, false));
  EXPECT_EQ(updater.free_count(), kBaseN);
  EXPECT_LE(updater.stats().maintenance_peak_overlay_pages, updater.cache_cap_pages() + 1);
}

TEST(QgMaintenanceConcurrency, EnospcAfterBeginPoisonsAndReopenRollsBack) {
  TemporaryDirectory root("enospc");
  auto base = WalTinyIndex::build(root.path(), kBaseN, 7601);
  {
    std::atomic_bool armed{true};
    Session session(
        base.prefix,
        [&](SegmentOpFailPoint point) {
          if (point == SegmentOpFailPoint::after_consolidate_overlay_modify_before_spill &&
              armed.exchange(false, std::memory_order_acq_rel)) {
            throw std::system_error(std::make_error_code(std::errc::no_space_on_device),
                                    "injected maintenance WAL ENOSPC");
          }
        },
        /*cache_cap_pages=*/1);
    auto &updater = *session.updater;
    updater.tombstone(1);
    EXPECT_THROW(updater.consolidate(1, 0, true, false), std::system_error);
    EXPECT_TRUE(updater.is_poisoned());
    const auto before_checkpoint = kind_counts(base.prefix);
    EXPECT_GT(before_checkpoint[static_cast<size_t>(SegmentOpKind::consolidate_begin)], 0U);
    EXPECT_EQ(before_checkpoint[static_cast<size_t>(SegmentOpKind::consolidate_end)], 0U);
    EXPECT_THROW(updater.checkpoint(), std::exception);
    const auto after_checkpoint = kind_counts(base.prefix);
    EXPECT_EQ(after_checkpoint[static_cast<size_t>(SegmentOpKind::superblock_flip)],
              before_checkpoint[static_cast<size_t>(SegmentOpKind::superblock_flip)]);
  }

  const auto first = reopened_fingerprint(base.prefix);
  EXPECT_EQ(first.live_count, kBaseN - 1);
  EXPECT_EQ(first.free_count, 0U);
  EXPECT_EQ(first.epoch, 0U);
  const auto second = reopened_fingerprint(base.prefix);
  EXPECT_EQ(second, first);
}

}  // namespace
}  // namespace alaya::laser
