// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <coroutine>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
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
    for (std::size_t i = 0; i < 65536; ++i) out.put(static_cast<char>(i % 251));
  }
  ~TemporaryFile() { std::filesystem::remove(path_); }
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  inline static std::atomic_uint64_t sequence_{0};
  std::filesystem::path path_;
};

class AlignedBuffer {
 public:
  AlignedBuffer(std::size_t size, std::size_t alignment) : size_(size) {
    alignment = std::max(alignment, sizeof(void *));
    if (::posix_memalign(&memory_, alignment, size) != 0) throw std::bad_alloc();
  }
  ~AlignedBuffer() { std::free(memory_); }
  AlignedBuffer(const AlignedBuffer &) = delete;
  [[nodiscard]] auto span() -> std::span<std::byte> {
    return {static_cast<std::byte *>(memory_), size_};
  }

 private:
  void *memory_ = nullptr;
  std::size_t size_ = 0;
};

class QueueExecutor final : public ResumeExecutor {
 public:
  auto execute(std::coroutine_handle<> handle) noexcept -> bool override {
    {
      std::lock_guard lock(mutex_);
      queued_.push_back(handle);
    }
    cv_.notify_one();
    return true;
  }
  void wait_and_drain() {
    std::coroutine_handle<> handle;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [&] {
        return !queued_.empty();
      });
      handle = queued_.front();
      queued_.pop_front();
    }
    handle.resume();
  }
  [[nodiscard]] auto empty() {
    std::lock_guard lock(mutex_);
    return queued_.empty();
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::coroutine_handle<>> queued_;
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
  ~TestTask() { handle.destroy(); }
  void start() { handle.resume(); }
  [[nodiscard]] auto done() const { return handle.done(); }
  std::coroutine_handle<promise_type> handle;
};

class PageReaderContract : public testing::TestWithParam<PageReaderBackend> {
 protected:
  auto open(const TemporaryFile &file, std::uint32_t depth = 16) {
    return open_page_reader(file.path(),
                            {.mode = OpenMode::automatic, .queue_depth = depth},
                            GetParam());
  }
};

#if defined(__linux__) && defined(ALAYA_LASER_USE_LIBAIO) && ALAYA_LASER_USE_LIBAIO
INSTANTIATE_TEST_SUITE_P(AllBackends,
                         PageReaderContract,
                         testing::Values(PageReaderBackend::sync,
                                         PageReaderBackend::libaio,
                                         PageReaderBackend::threadpool));
#else
INSTANTIATE_TEST_SUITE_P(AllBackends,
                         PageReaderContract,
                         testing::Values(PageReaderBackend::sync, PageReaderBackend::threadpool));
#endif

TEST(PageReaderAlignment, ValidatesAndRoundsWithoutFixedSectorAssumption) {
  alignas(16) std::array<std::byte, 16> bytes{};
  ReadRequest request{.offset = 8, .buffer = bytes};
  EXPECT_TRUE(validate_read_request(request, {1, 8, 8, 4, false}));
  request.offset = 3;
  EXPECT_FALSE(validate_read_request(request, {1, 8, 8, 4, false}));
  EXPECT_EQ(align_down(17, 8), 16);
  EXPECT_EQ(align_up(17, 8), 24);
  EXPECT_FALSE(align_up(17, 0).has_value());
  EXPECT_EQ(conservative_constraints(OpenMode::buffered).buffer_alignment, 1);
  EXPECT_EQ(conservative_constraints(OpenMode::direct).buffer_alignment, 4096);
}

TEST_P(PageReaderContract, ReadsSinglePage) {
  TemporaryFile file;
  auto reader = open(file);
  const auto constraints = reader->constraints();
  AlignedBuffer buffer(constraints.size_alignment, constraints.buffer_alignment);
  ReadRequest request{.id = 7, .offset = constraints.offset_alignment, .buffer = buffer.span()};
  const auto results = read_pages_blocking(*reader, {&request, 1});
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].id, 7);
  EXPECT_EQ(results[0].status, ReadStatus::ok);
  EXPECT_EQ(results[0].bytes, buffer.span().size());
}

TEST_P(PageReaderContract, BatchCompletesEachRequestExactlyOnce) {
  TemporaryFile file;
  auto reader = open(file);
  const auto c = reader->constraints();
  AlignedBuffer storage(3 * c.size_alignment, c.buffer_alignment);
  std::array<ReadRequest, 3> requests;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    requests[i] = {.id = i + 1,
                   .offset = i * c.offset_alignment,
                   .buffer = storage.span().subspan(i * c.size_alignment, c.size_alignment)};
  }
  std::mutex mutex;
  std::condition_variable cv;
  std::map<RequestId, int> calls;
  struct Context {
    std::mutex *mutex;
    std::condition_variable *cv;
    std::map<RequestId, int> *calls;
  } context{&mutex, &cv, &calls};
  const auto callback = [](void *raw, ReadResult result) noexcept {
    auto &context = *static_cast<Context *>(raw);
    std::lock_guard lock(*context.mutex);
    ++(*context.calls)[result.id];
    context.cv->notify_one();
  };
  auto handle = reader->submit(requests, {callback, &context});
  std::unique_lock lock(mutex);
  cv.wait(lock, [&] {
    return calls.size() == requests.size();
  });
  EXPECT_EQ(calls, (std::map<RequestId, int>{{1, 1}, {2, 1}, {3, 1}}));
  lock.unlock();
  EXPECT_EQ(handle.cancel(), CancelResult::already_complete);
}

TEST_P(PageReaderContract, ReportsShortReadAtAndAcrossEof) {
  TemporaryFile file;
  auto reader = open(file);
  const auto c = reader->constraints();
  AlignedBuffer first(c.size_alignment, c.buffer_alignment);
  AlignedBuffer second(c.size_alignment, c.buffer_alignment);
  std::array requests{ReadRequest{.id = 1, .offset = 65536, .buffer = first.span()},
                      ReadRequest{.id = 2,
                                  .offset = 65536 + c.offset_alignment,
                                  .buffer = second.span()}};
  const auto results = read_pages_blocking(*reader, requests);
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0].status, ReadStatus::short_read);
  EXPECT_EQ(results[0].bytes, 0);
  EXPECT_EQ(results[1].status, ReadStatus::short_read);
  EXPECT_EQ(results[1].bytes, 0);
}

TEST_P(PageReaderContract, ExpiredDeadlineCompletesAfterIoIsSafe) {
  TemporaryFile file;
  auto reader = open(file);
  const auto c = reader->constraints();
  AlignedBuffer buffer(c.size_alignment, c.buffer_alignment);
  ReadRequest request{.id = 3,
                      .offset = 0,
                      .buffer = buffer.span(),
                      .deadline = Clock::now() - std::chrono::seconds(1)};
  const auto results = read_pages_blocking(*reader, {&request, 1});
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::timed_out);
  EXPECT_EQ(results[0].bytes, buffer.span().size());
}

TEST_P(PageReaderContract, RejectsWholeInvalidBatchAndWorkAfterShutdown) {
  TemporaryFile file;
  auto reader = open(file, 1);
  AlignedBuffer buffer(2, 8);
  std::array requests{ReadRequest{.buffer = buffer.span().first(1)},
                      ReadRequest{.buffer = buffer.span().last(1)}};
  EXPECT_THROW((void)reader->submit(requests, {}), std::invalid_argument);
  reader->shutdown();
  reader->shutdown();
  EXPECT_THROW((void)reader->submit({}, {}), std::runtime_error);
}

TEST_P(PageReaderContract, BlockingLayerHandlesEmptyAndNormalBatch) {
  TemporaryFile file;
  auto reader = open(file);
  EXPECT_TRUE(read_pages_blocking(*reader, {}).empty());
  const auto c = reader->constraints();
  AlignedBuffer buffer(c.size_alignment, c.buffer_alignment);
  ReadRequest request{.id = 9, .offset = 0, .buffer = buffer.span()};
  auto results = read_pages_blocking(*reader, {&request, 1});
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::ok);
}

TEST_P(PageReaderContract, AwaitableDefersResumeThroughExecutor) {
  TemporaryFile file;
  auto reader = open(file);
  QueueExecutor executor;
  std::vector<ReadResult> results;
  const auto c = reader->constraints();
  AlignedBuffer buffer(c.size_alignment, c.buffer_alignment);
  auto make_operation = [&]() -> TestTask {
    ReadRequest request{.id = 42, .offset = 0, .buffer = buffer.span()};
    results = co_await read_pages(*reader, executor, {&request, 1});
  };
  auto operation = make_operation();
  operation.start();
  EXPECT_FALSE(operation.done());
  EXPECT_TRUE(results.empty());
  executor.wait_and_drain();
  EXPECT_TRUE(operation.done());
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].status, ReadStatus::ok);
}

TEST_P(PageReaderContract, EmptyAwaitableDoesNotUseExecutor) {
  TemporaryFile file;
  auto reader = open(file);
  QueueExecutor executor;
  bool resumed = false;
  auto make_operation = [&]() -> TestTask {
    auto results = co_await read_pages(*reader, executor, {});
    resumed = results.empty();
  };
  auto operation = make_operation();
  operation.start();
  EXPECT_TRUE(operation.done());
  EXPECT_TRUE(resumed);
  EXPECT_TRUE(executor.empty());
}

#if defined(__linux__) && defined(ALAYA_LASER_USE_LIBAIO) && ALAYA_LASER_USE_LIBAIO
TEST(LibaioPageReaderTest, RejectsMisalignedDirectIo) {
  TemporaryFile file;
  LibaioPageReader reader(file.path());
  std::vector<std::byte> bytes(reader.constraints().size_alignment + 1);
  ReadRequest request{.offset = 1,
                      .buffer = {bytes.data() + 1, reader.constraints().size_alignment}};
  EXPECT_THROW((void)reader.submit({&request, 1}, {}), std::invalid_argument);
}

TEST(LibaioPageReaderTest, SupportsConcurrentBatchesInFlight) {
  TemporaryFile file;
  LibaioPageReader reader(file.path(), {.queue_depth = 16});
  const auto c = reader.constraints();
  std::array<std::unique_ptr<AlignedBuffer>, 8> buffers;
  std::array<ReadRequest, 8> requests;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    buffers[i] = std::make_unique<AlignedBuffer>(c.size_alignment, c.buffer_alignment);
    requests[i] = {.id = i, .offset = i * c.offset_alignment, .buffer = buffers[i]->span()};
  }
  std::atomic_size_t completed{0};
  const auto callback = [](void *raw, ReadResult result) noexcept {
    EXPECT_EQ(result.status, ReadStatus::ok);
    static_cast<std::atomic_size_t *>(raw)->fetch_add(1, std::memory_order_release);
  };
  std::vector<std::thread> submitters;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    submitters.emplace_back([&, i] {
      (void)reader.submit({&requests[i], 1}, {callback, &completed});
    });
  }
  for (auto &thread : submitters) thread.join();
  reader.shutdown();
  EXPECT_EQ(completed.load(std::memory_order_acquire), requests.size());
}
#endif

TEST(ThreadpoolPageReaderTest, BoundedQueueRejectsOversizedBatch) {
  TemporaryFile file;
  ThreadpoolPageReader reader(file.path(), {.queue_depth = 2});
  std::array<std::byte, 3> bytes{};
  std::array requests{ReadRequest{.buffer = {&bytes[0], 1}},
                      ReadRequest{.buffer = {&bytes[1], 1}},
                      ReadRequest{.buffer = {&bytes[2], 1}}};
  EXPECT_THROW((void)reader.submit(requests, {}), std::invalid_argument);
}

}  // namespace
}  // namespace alaya::storage::io
