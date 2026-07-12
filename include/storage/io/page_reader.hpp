// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <chrono>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <system_error>

namespace alaya::storage::io {

using RequestId = std::uint64_t;
using Clock = std::chrono::steady_clock;
using FileHandle = std::intptr_t;

enum class OpenMode : std::uint8_t { buffered, direct, automatic };
enum class CancelResult : std::uint8_t { requested, already_complete, unsupported };

struct ReaderOptions {
  OpenMode mode = OpenMode::automatic;
  std::uint32_t queue_depth = 128;
};

struct ReadConstraints {
  std::size_t buffer_alignment = 1;
  std::size_t offset_alignment = 1;
  std::size_t size_alignment = 1;
  std::uint32_t max_batch = 1;
  bool direct = false;
};

struct ReadRequest {
  RequestId id = 0;
  std::uint64_t offset = 0;
  std::span<std::byte> buffer{};
  Clock::time_point deadline = Clock::time_point::max();
};

enum class ReadStatus : std::uint8_t {
  ok,
  invalid_argument,
  short_read,
  cancelled,
  timed_out,
  io_error,
  shutting_down
};

struct ReadResult {
  RequestId id = 0;
  std::size_t bytes = 0;
  ReadStatus status = ReadStatus::ok;
  std::error_code error{};
};

using CompletionFn = void (*)(void *, ReadResult) noexcept;

struct Completion {
  CompletionFn fn = nullptr;
  void *context = nullptr;
};

class BatchHandle {
 public:
  BatchHandle() noexcept = default;
  BatchHandle(BatchHandle &&) noexcept = default;
  auto operator=(BatchHandle &&) noexcept -> BatchHandle & = default;
  BatchHandle(const BatchHandle &) = delete;
  auto operator=(const BatchHandle &) -> BatchHandle & = delete;
  ~BatchHandle() = default;

  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }
  [[nodiscard]] auto cancel() noexcept -> CancelResult {
    return state_ == nullptr ? CancelResult::already_complete : state_->cancel();
  }

 private:
  struct State {
    virtual ~State() = default;
    [[nodiscard]] virtual auto cancel() noexcept -> CancelResult = 0;
  };

  explicit BatchHandle(std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}
  std::shared_ptr<State> state_;
  friend class PageReader;
};

class PageReader {
 public:
  PageReader() = default;
  PageReader(const PageReader &) = delete;
  auto operator=(const PageReader &) -> PageReader & = delete;
  virtual ~PageReader() = default;

  [[nodiscard]] virtual auto constraints() const noexcept -> ReadConstraints = 0;
  [[nodiscard]] virtual auto submit(std::span<const ReadRequest> requests, Completion completion)
      -> BatchHandle = 0;
  virtual void shutdown() noexcept = 0;

 protected:
  template <class CancelFn>
  [[nodiscard]] static auto make_batch_handle(CancelFn cancel) -> BatchHandle {
    struct State final : BatchHandle::State {
      explicit State(CancelFn fn) : cancel_(std::move(fn)) {}
      auto cancel() noexcept -> CancelResult override { return cancel_(); }
      CancelFn cancel_;
    };
    return BatchHandle(std::make_shared<State>(std::move(cancel)));
  }
};

class ResumeExecutor {
 public:
  virtual ~ResumeExecutor() = default;
  // A successful call enqueues the handle; it must not resume it inline.
  virtual auto execute(std::coroutine_handle<>) noexcept -> bool = 0;
};

}  // namespace alaya::storage::io
