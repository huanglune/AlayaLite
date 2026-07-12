// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <array>
#include <coroutine>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <vector>

#include "storage/io/alignment.hpp"
#include "storage/io/page_awaitable.hpp"
#include "storage/io/page_reader_factory.hpp"

namespace alaya::storage::io {
namespace {

class TemporaryFile {
 public:
  TemporaryFile() {
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-page-reader-" + std::to_string(++sequence_));
    std::ofstream out(path_, std::ios::binary);
    for (std::size_t i = 0; i < 8192; ++i) out.put(static_cast<char>(i % 251));
  }
  ~TemporaryFile() { std::filesystem::remove(path_); }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  inline static std::atomic_uint64_t sequence_{0};
  std::filesystem::path path_;
};

class QueueExecutor final : public ResumeExecutor {
 public:
  auto execute(std::coroutine_handle<> handle) noexcept -> bool override {
    queued.push_back(handle);
    return true;
  }
  void drain() {
    while (!queued.empty()) {
      auto handle = queued.front();
      queued.pop_front();
      handle.resume();
    }
  }
  std::deque<std::coroutine_handle<>> queued;
};

struct TestTask {
  struct promise_type {
    auto get_return_object() -> TestTask {
      return TestTask{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    auto initial_suspend() noexcept { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    void return_void() noexcept {}
    void unhandled_exception() { std::terminate(); }
  };
  explicit TestTask(std::coroutine_handle<promise_type> value) : handle(value) {}
  TestTask(const TestTask &) = delete;
  ~TestTask() { handle.destroy(); }
  void start() { handle.resume(); }
  [[nodiscard]] auto done() const { return handle.done(); }
  std::coroutine_handle<promise_type> handle;
};

TEST(PageReaderAlignment, ValidatesAndRoundsWithoutFixedSectorAssumption) {
  std::array<std::byte, 16> bytes{};
  ReadRequest request{.offset = 8, .buffer = bytes};
  EXPECT_TRUE(validate_read_request(request, {1, 8, 8, 4, false}));
  request.offset = 3;
  EXPECT_FALSE(validate_read_request(request, {1, 8, 8, 4, false}));
  EXPECT_EQ(align_down(17, 8), 16);
  EXPECT_EQ(align_up(17, 8), 24);
  EXPECT_FALSE(align_up(17, 0).has_value());
  EXPECT_EQ(conservative_constraints(OpenMode::buffered).buffer_alignment, 1);
  EXPECT_EQ(conservative_constraints(OpenMode::direct).buffer_alignment, 4096);
  EXPECT_EQ(constraints_from_sector_sizes(512, 4096).buffer_alignment, 4096);
}

TEST(SyncPageReaderContract, ReadsSinglePageInline) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  std::array<std::byte, 32> bytes{};
  ReadResult result;
  bool called = false;
  const auto callback = [](void *context, ReadResult value) noexcept {
    auto *pair = static_cast<std::pair<bool *, ReadResult *> *>(context);
    *pair->first = true;
    *pair->second = value;
  };
  std::pair context{&called, &result};
  ReadRequest request{.id = 7, .offset = 251, .buffer = bytes};
  auto handle = reader.submit({&request, 1}, {callback, &context});
  EXPECT_TRUE(called);  // D8: completion occurs before submit returns.
  EXPECT_TRUE(handle);
  EXPECT_EQ(result.id, 7);
  EXPECT_EQ(result.status, ReadStatus::ok);
  EXPECT_EQ(result.bytes, bytes.size());
  EXPECT_EQ(bytes[0], std::byte{0});
}

TEST(SyncPageReaderContract, BatchCompletesEachRequestExactlyOnce) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  std::array<std::array<std::byte, 16>, 3> buffers{};
  std::array<ReadRequest, 3> requests{};
  for (std::size_t i = 0; i < requests.size(); ++i) {
    requests[i] = {.id = i + 1, .offset = i * 16, .buffer = buffers[i]};
  }
  std::map<RequestId, int> calls;
  const auto callback = [](void *context, ReadResult result) noexcept {
    ++(*static_cast<std::map<RequestId, int> *>(context))[result.id];
  };
  auto handle = reader.submit(requests, {callback, &calls});
  EXPECT_EQ(calls, (std::map<RequestId, int>{{1, 1}, {2, 1}, {3, 1}}));
  EXPECT_EQ(handle.cancel(), CancelResult::already_complete);
}

TEST(SyncPageReaderContract, ReportsShortReadAtAndAcrossEof) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  std::array<std::byte, 32> first{};
  std::array<std::byte, 8> second{};
  std::array requests{ReadRequest{.id = 1, .offset = 8180, .buffer = first},
                      ReadRequest{.id = 2, .offset = 9000, .buffer = second}};
  const auto results = read_pages_blocking(reader, requests);
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].status, ReadStatus::short_read);
  EXPECT_EQ(results[0].bytes, 12);
  EXPECT_EQ(results[1].status, ReadStatus::short_read);
  EXPECT_EQ(results[1].bytes, 0);
}

TEST(SyncPageReaderContract, ExpiredDeadlineCompletesAfterIoIsSafe) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  std::array<std::byte, 16> bytes{};
  ReadRequest request{.id = 3,
                      .offset = 0,
                      .buffer = bytes,
                      .deadline = Clock::now() - std::chrono::seconds(1)};
  const auto results = read_pages_blocking(reader, {&request, 1});
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::timed_out);
  EXPECT_EQ(results[0].bytes, bytes.size());
}

TEST(SyncPageReaderContract, RejectsWholeInvalidBatchAndWorkAfterShutdown) {
  TemporaryFile file;
  SyncPageReader reader(file.path(), {.mode = OpenMode::buffered, .queue_depth = 1});
  std::array<std::byte, 1> a{}, b{};
  std::array requests{ReadRequest{.buffer = a}, ReadRequest{.buffer = b}};
  EXPECT_THROW((void)reader.submit(requests, {}), std::invalid_argument);
  reader.shutdown();
  reader.shutdown();
  EXPECT_THROW((void)reader.submit({}, {}), std::runtime_error);
}

TEST(SyncPageReaderContract, BlockingLayerHandlesEmptyAndNormalBatch) {
  TemporaryFile file;
  auto reader = open_page_reader(file.path());
  EXPECT_TRUE(read_pages_blocking(*reader, {}).empty());
  std::array<std::byte, 64> bytes{};
  ReadRequest request{.id = 9, .offset = 502, .buffer = bytes};
  auto results = read_pages_blocking(*reader, {&request, 1});
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::ok);
  EXPECT_EQ(bytes.front(), std::byte{0});
}

TEST(SyncPageReaderContract, AwaitableDefersResumeThroughExecutor) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  QueueExecutor executor;
  std::vector<ReadResult> results;
  std::array<std::byte, 128> bytes{};
  auto make_operation = [&]() -> TestTask {
    ReadRequest request{.id = 42, .offset = 0, .buffer = bytes};
    results = co_await read_pages(reader, executor, {&request, 1});
  };
  auto operation = make_operation();

  operation.start();
  EXPECT_FALSE(operation.done());
  ASSERT_EQ(executor.queued.size(), 1);  // no inline coroutine re-entry
  EXPECT_TRUE(results.empty());
  executor.drain();
  EXPECT_TRUE(operation.done());
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::ok);
  EXPECT_EQ(bytes[127], std::byte{127});  // buffer stayed live until completion
}

TEST(SyncPageReaderContract, EmptyAwaitableDoesNotUseExecutor) {
  TemporaryFile file;
  SyncPageReader reader(file.path());
  QueueExecutor executor;
  bool resumed = false;
  auto make_operation = [&]() -> TestTask {
    auto results = co_await read_pages(reader, executor, {});
    resumed = results.empty();
  };
  auto operation = make_operation();
  operation.start();
  EXPECT_TRUE(operation.done());
  EXPECT_TRUE(resumed);
  EXPECT_TRUE(executor.queued.empty());
}

}  // namespace
}  // namespace alaya::storage::io
