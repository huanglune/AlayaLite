// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file uring_reactor_test.cpp
 * @brief UringReactor: awaitable reads resumed on a libcoro thread pool.
 */

#include "storage/io/uring_reactor.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#include "utils/coro_gate.hpp"

#if defined(__linux__)
  #include <fcntl.h>
  #include <unistd.h>
#endif

namespace {

using alaya::IORequest;
using alaya::UringReactor;

class UringReactorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!UringReactor::is_available()) {
      GTEST_SKIP() << "io_uring not available on this kernel";
    }
    path_ = std::filesystem::temp_directory_path() /
            ("uring_reactor_test_" + std::to_string(::getpid()));
    payload_.resize(kFileSize);
    for (size_t i = 0; i < payload_.size(); ++i) {
      payload_[i] = static_cast<char>((i * 131 + 7) & 0xFF);
    }
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    out.write(payload_.data(), static_cast<std::streamsize>(payload_.size()));
    out.close();
    fd_ = ::open(path_.c_str(), O_RDONLY);
    ASSERT_GE(fd_, 0);
  }

  void TearDown() override {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  static constexpr size_t kFileSize = 1 << 20;  // 1 MiB
  std::filesystem::path path_;
  std::vector<char> payload_;
  int fd_ = -1;
};

TEST_F(UringReactorTest, SingleReadReturnsData) {
  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 2, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  std::vector<char> buf(4096, 0);
  auto task = [&]() -> coro::task<int32_t> {
    co_await pool.schedule();
    co_return co_await reactor.read(pool, fd_, buf.data(), 4096, 8192);
  };
  const int32_t got = coro::sync_wait(task());
  ASSERT_EQ(got, 4096);
  EXPECT_EQ(0, std::memcmp(buf.data(), payload_.data() + 8192, 4096));
}

TEST_F(UringReactorTest, BatchWaveCompletesAllReads) {
  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 2, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  constexpr uint32_t kReads = 64;
  constexpr size_t kLen = 4096;
  std::vector<char> buf(kReads * kLen, 0);
  std::vector<IORequest> reqs(kReads);
  for (uint32_t i = 0; i < kReads; ++i) {
    // Scattered, overlapping offsets across the file.
    const uint64_t off = (static_cast<uint64_t>(i) * 12800) % (kFileSize - kLen);
    reqs[i] = IORequest{buf.data() + i * kLen, kLen, off, nullptr};
  }
  auto task = [&]() -> coro::task<uint32_t> {
    co_await pool.schedule();
    co_return co_await reactor.read_batch(pool, fd_, reqs.data(), kReads);
  };
  const uint32_t failures = coro::sync_wait(task());
  EXPECT_EQ(failures, 0U);
  for (uint32_t i = 0; i < kReads; ++i) {
    ASSERT_EQ(reqs[i].result_, static_cast<int32_t>(kLen)) << "read " << i;
    EXPECT_EQ(0, std::memcmp(buf.data() + i * kLen, payload_.data() + reqs[i].offset_, kLen))
        << "read " << i;
  }
}

TEST_F(UringReactorTest, EmptyBatchDoesNotSuspend) {
  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 1, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  auto task = [&]() -> coro::task<uint32_t> {
    co_await pool.schedule();
    co_return co_await reactor.read_batch(pool, fd_, nullptr, 0);
  };
  EXPECT_EQ(coro::sync_wait(task()), 0U);
}

TEST_F(UringReactorTest, ShortReadPastEofCountsAsFailure) {
  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 1, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  std::vector<char> buf(4096, 0);
  std::vector<IORequest> reqs(1);
  reqs[0] = IORequest{buf.data(), 4096, kFileSize - 1024, nullptr};  // only 1024 bytes left
  auto task = [&]() -> coro::task<uint32_t> {
    co_await pool.schedule();
    co_return co_await reactor.read_batch(pool, fd_, reqs.data(), 1);
  };
  EXPECT_EQ(coro::sync_wait(task()), 1U);
  EXPECT_EQ(reqs[0].result_, 1024);
}

TEST_F(UringReactorTest, ConcurrentWavesFromManyTasks) {
  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 8, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  constexpr uint32_t kTasks = 32;
  constexpr uint32_t kReadsPerTask = 16;
  constexpr size_t kLen = 4096;
  std::atomic<uint32_t> total_failures{0};

  auto one_wave = [&](uint32_t task_id) -> coro::task<> {
    co_await pool.schedule();
    std::vector<char> buf(kReadsPerTask * kLen);
    std::vector<IORequest> reqs(kReadsPerTask);
    for (uint32_t i = 0; i < kReadsPerTask; ++i) {
      const uint64_t off = (static_cast<uint64_t>(task_id * 7919 + i * 4096)) % (kFileSize - kLen);
      reqs[i] = IORequest{buf.data() + i * kLen, kLen, off, nullptr};
    }
    total_failures += co_await reactor.read_batch(pool, fd_, reqs.data(), kReadsPerTask);
    for (uint32_t i = 0; i < kReadsPerTask; ++i) {
      if (std::memcmp(buf.data() + i * kLen, payload_.data() + reqs[i].offset_, kLen) != 0) {
        total_failures += 1000;  // corruption sentinel
      }
    }
  };
  auto run = [&]() -> coro::task<> {
    std::vector<coro::task<>> tasks;
    tasks.reserve(kTasks);
    for (uint32_t t = 0; t < kTasks; ++t) {
      tasks.emplace_back(one_wave(t));
    }
    co_await coro::when_all(std::move(tasks));
  };
  coro::sync_wait(run());
  EXPECT_EQ(total_failures.load(), 0U);
}

// AsyncGate: far more coroutines than pooled objects AND than pool threads,
// each holding its object across a real reactor suspension. A thread-blocking
// pool would deadlock here (every thread parked in acquire while all objects
// are held by suspended coroutines); the suspending gate must complete all
// tasks with exclusive ownership throughout.
TEST_F(UringReactorTest, AsyncGateSuspendsInsteadOfBlockingUnderOversubscription) {
  struct Pooled {
    std::atomic<int> owners{0};
  };
  constexpr uint32_t kItems = 2;
  constexpr uint32_t kTasks = 64;
  constexpr uint32_t kLen = 4096;

  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 2, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  alaya::AsyncGate<Pooled> gate;
  std::vector<Pooled> items(kItems);
  for (auto &item : items) {
    gate.add(&item);
  }

  std::atomic<uint32_t> ownership_violations{0};
  std::atomic<uint32_t> completed{0};
  std::atomic<uint32_t> io_failures{0};

  auto one = [&](uint32_t task_id) -> coro::task<> {
    co_await pool.schedule();
    Pooled *item = co_await gate.acquire(pool);
    if (item->owners.fetch_add(1, std::memory_order_acq_rel) != 0) {
      ownership_violations.fetch_add(1, std::memory_order_relaxed);
    }
    // Suspend while holding the pooled object — the deadlock-shaped moment.
    std::vector<char> buf(kLen);
    IORequest req{buf.data(), kLen, (task_id % 64) * kLen, nullptr};
    io_failures += co_await reactor.read_batch(pool, fd_, &req, 1);
    if (item->owners.fetch_sub(1, std::memory_order_acq_rel) != 1) {
      ownership_violations.fetch_add(1, std::memory_order_relaxed);
    }
    gate.release(item);
    completed.fetch_add(1, std::memory_order_relaxed);
    co_return;
  };

  auto run = [&]() -> coro::task<> {
    std::vector<coro::task<>> tasks;
    tasks.reserve(kTasks);
    for (uint32_t t = 0; t < kTasks; ++t) {
      tasks.emplace_back(one(t));
    }
    co_await coro::when_all(std::move(tasks));
  };
  coro::sync_wait(run());

  EXPECT_EQ(completed.load(), kTasks);
  EXPECT_EQ(ownership_violations.load(), 0U);
  EXPECT_EQ(io_failures.load(), 0U);
  EXPECT_EQ(gate.free_count(), kItems);
}

// The update-search shape: every coroutine issues many SEQUENTIAL waves,
// reusing one requests vector (clear + emplace) across them, while other
// coroutines' waves poll the same ring. A completion dispatched after its
// wave's read_batch returned would write result_ into the next wave's
// requests — this is exactly the corruption TSan flagged in the e2e run.
TEST_F(UringReactorTest, SequentialWaveReuseUnderConcurrency) {
  constexpr uint32_t kCoros = 8;
  constexpr uint32_t kWavesPerCoro = 400;
  constexpr uint32_t kLen = 4096;

  UringReactor reactor;
  coro::thread_pool pool{
      {.thread_count = 4, .on_thread_start_functor = nullptr, .on_thread_stop_functor = nullptr}};
  std::atomic<uint32_t> corruption{0};
  std::atomic<uint32_t> io_failures{0};

  auto one = [&](uint32_t coro_id) -> coro::task<> {
    co_await pool.schedule();
    std::vector<char> buf(8 * kLen);
    std::vector<IORequest> reqs;  // reused across waves like the async searches
    for (uint32_t w = 0; w < kWavesPerCoro; ++w) {
      const uint32_t width = 1 + ((coro_id + w) % 8);
      reqs.clear();
      for (uint32_t i = 0; i < width; ++i) {
        const uint64_t off = ((coro_id * 131 + w * 17 + i) * kLen) % (kFileSize - kLen);
        reqs.emplace_back(buf.data() + i * kLen, kLen, off, nullptr);
      }
      io_failures += co_await reactor.read_batch(pool, fd_, reqs.data(), width);
      for (uint32_t i = 0; i < width; ++i) {
        if (reqs[i].result_ != static_cast<int32_t>(kLen)) {
          corruption.fetch_add(1, std::memory_order_relaxed);
        }
        // Poison the slot: a late completion for THIS wave would overwrite it
        // (and be caught by the next iteration's checks or by TSan).
        reqs[i].result_ = -12345;
      }
    }
  };

  auto run = [&]() -> coro::task<> {
    std::vector<coro::task<>> tasks;
    tasks.reserve(kCoros);
    for (uint32_t c = 0; c < kCoros; ++c) {
      tasks.emplace_back(one(c));
    }
    co_await coro::when_all(std::move(tasks));
  };
  coro::sync_wait(run());

  EXPECT_EQ(corruption.load(), 0U);
  EXPECT_EQ(io_failures.load(), 0U);
}

}  // namespace
