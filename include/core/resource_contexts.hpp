// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory_resource>
#include <utility>

#include "core/status.hpp"

namespace alaya::core {

inline constexpr std::uint64_t kUnlimitedResource = std::numeric_limits<std::uint64_t>::max();

struct MemoryLease {
  VersionedStructHeader header{};
  std::uint64_t available_bytes{kUnlimitedResource};
  const void *lease_token{};
  std::uint64_t reserved[3]{};

  MemoryLease() : header(current_struct_header<MemoryLease>()) {}
  explicit MemoryLease(std::uint64_t bytes)
      : header(current_struct_header<MemoryLease>()), available_bytes(bytes) {}

  [[nodiscard]] auto permits(std::uint64_t bytes) const noexcept -> bool {
    return available_bytes == kUnlimitedResource || bytes <= available_bytes;
  }
};

struct MemoryReservation {
  using Grow = Status (*)(void *, std::uint64_t, MemoryLease &) noexcept;

  VersionedStructHeader header{};
  MemoryLease lease{};
  void *state{};
  Grow grow{};
  std::uint64_t reserved[3]{};

  MemoryReservation() : header(current_struct_header<MemoryReservation>()) {}
  explicit MemoryReservation(std::uint64_t bytes)
      : header(current_struct_header<MemoryReservation>()), lease(bytes) {}

  [[nodiscard]] auto permits(std::uint64_t bytes) const noexcept -> bool {
    return lease.permits(bytes);
  }

  [[nodiscard]] auto ensure(std::uint64_t bytes, OperationStage stage, const char *diagnostic)
      -> Status {
    if (permits(bytes)) {
      return Status::success();
    }
    if (grow != nullptr) {
      auto status = grow(state, bytes, lease);
      if (!status.ok()) {
        return status;
      }
      if (permits(bytes)) {
        return Status::success();
      }
    }
    return Status::error(StatusCode::resource_exhausted,
                         stage,
                         StatusDetail::budget_denied,
                         diagnostic,
                         Retryability::retryable_with_backoff);
  }
};

struct IoCredits {
  VersionedStructHeader header{};
  std::uint64_t available_requests{kUnlimitedResource};
  std::uint64_t available_bytes{kUnlimitedResource};
  const void *credit_token{};
  std::uint64_t reserved[3]{};

  IoCredits() : header(current_struct_header<IoCredits>()) {}
};

struct Deadline {
  VersionedStructHeader header{};
  std::uint64_t steady_clock_nanoseconds{};
  bool enabled{};
  std::uint8_t reserved_bytes[7]{};
  std::uint64_t reserved[2]{};

  Deadline() : header(current_struct_header<Deadline>()) {}

  [[nodiscard]] static auto at(std::chrono::steady_clock::time_point value) -> Deadline {
    Deadline deadline;
    deadline.enabled = true;
    deadline.steady_clock_nanoseconds = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count());
    return deadline;
  }

  [[nodiscard]] auto expired() const noexcept -> bool {
    if (!enabled) {
      return false;
    }
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
    return now >= 0 && static_cast<std::uint64_t>(now) >= steady_clock_nanoseconds;
  }
};

struct CancellationToken {
  using IsCancelled = bool (*)(const void *) noexcept;

  VersionedStructHeader header{};
  const void *state{};
  IsCancelled is_cancelled{};
  std::uint64_t reserved[3]{};

  CancellationToken() : header(current_struct_header<CancellationToken>()) {}

  [[nodiscard]] auto requested() const noexcept -> bool {
    return is_cancelled != nullptr && is_cancelled(state);
  }

  [[nodiscard]] static auto from_atomic(const std::atomic_bool &flag) -> CancellationToken {
    CancellationToken token;
    token.state = std::addressof(flag);
    token.is_cancelled = [](const void *raw) noexcept {
      return static_cast<const std::atomic_bool *>(raw)->load(std::memory_order_acquire);
    };
    return token;
  }
};

struct RuntimeLane {
  using Task = std::function<void()>;
  using Post = void (*)(void *, Task) noexcept;

  VersionedStructHeader header{};
  void *state{};
  Post post{};
  std::uint64_t lane_id{};
  std::uint64_t reserved[3]{};

  RuntimeLane() : header(current_struct_header<RuntimeLane>()) {}

  [[nodiscard]] auto configured() const noexcept -> bool { return post != nullptr; }

  void dispatch(Task task) const noexcept {
    if (post != nullptr) {
      post(state, std::move(task));
    } else {
      task();
    }
  }
};

struct SearchStats {
  VersionedStructHeader header{};
  std::uint64_t visited{};
  std::uint64_t io_bytes{};
  std::uint64_t io_requests{};
  std::uint64_t cache_hits{};
  std::uint64_t filter_candidates{};
  std::uint64_t rerank_count{};
  std::uint64_t budget_wait_nanoseconds{};
  std::uint64_t reserved[4]{};

  SearchStats() : header(current_struct_header<SearchStats>()) {}
};

struct OpenContext {
  VersionedStructHeader header{};
  MemoryLease resident_lease{};
  MemoryLease cache_lease{};
  MemoryLease scratch_pool_lease{};
  IoCredits io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  std::uint64_t reserved[4]{};

  OpenContext() : header(current_struct_header<OpenContext>()) {}
};

struct BuildContext {
  VersionedStructHeader header{};
  MemoryReservation growing_reservation{};
  std::pmr::memory_resource *temporary_memory{std::pmr::get_default_resource()};
  IoCredits io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  void *telemetry{};
  std::uint64_t reserved[4]{};

  BuildContext() : header(current_struct_header<BuildContext>()) {}
};

struct MutationContext {
  VersionedStructHeader header{};
  MemoryReservation pending_reservation{};
  MemoryReservation stage_reservation{};
  IoCredits wal_io_credits{};
  IoCredits io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  const void *transaction_token{};
  std::uint64_t reserved[4]{};

  MutationContext() : header(current_struct_header<MutationContext>()) {}
};

struct SealContext {
  VersionedStructHeader header{};
  MemoryReservation snapshot_reservation{};
  MemoryReservation build_reservation{};
  IoCredits io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  const void *target_config{};
  std::uint64_t target_config_size{};
  std::uint64_t reserved[4]{};

  SealContext() : header(current_struct_header<SealContext>()) {}
};

struct SearchContext {
  VersionedStructHeader header{};
  MemoryLease query_scratch_lease{};
  IoCredits io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  SearchStats *stats{};
  std::uint64_t reserved[4]{};

  SearchContext() : header(current_struct_header<SearchContext>()) {}
};

enum class DurabilityTarget : std::uint8_t { searchable = 0, wal_fsync = 1, full_checkpoint = 2 };

struct CheckpointContext {
  VersionedStructHeader header{};
  IoCredits dirty_page_io_credits{};
  IoCredits wal_io_credits{};
  Deadline deadline{};
  CancellationToken cancellation{};
  RuntimeLane lane{};
  DurabilityTarget durability_target{DurabilityTarget::wal_fsync};
  std::uint8_t reserved_bytes[7]{};
  std::uint64_t reserved[4]{};

  CheckpointContext() : header(current_struct_header<CheckpointContext>()) {}
};

[[nodiscard]] inline auto validate_runtime_control(const Deadline &deadline,
                                                   const CancellationToken &cancellation,
                                                   OperationStage stage) -> Status {
  if (cancellation.requested()) {
    return Status::error(StatusCode::cancelled,
                         stage,
                         StatusDetail::cancellation_requested,
                         "operation was cancelled");
  }
  if (deadline.expired()) {
    return Status::error(StatusCode::deadline_exceeded,
                         stage,
                         StatusDetail::deadline_reached,
                         "operation deadline was reached");
  }
  return Status::success();
}

[[nodiscard]] inline auto require_lease(const MemoryLease &lease,
                                        std::uint64_t bytes,
                                        OperationStage stage,
                                        const char *diagnostic) -> Status {
  if (!lease.permits(bytes)) {
    return Status::error(StatusCode::resource_exhausted,
                         stage,
                         StatusDetail::budget_denied,
                         diagnostic,
                         Retryability::retryable_with_backoff);
  }
  return Status::success();
}

}  // namespace alaya::core
