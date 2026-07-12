// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <exception>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "core/versioned.hpp"

namespace alaya::core {

enum class StatusCode : std::uint8_t {
  ok = 0,
  invalid_argument = 1,
  not_supported = 2,
  conflict = 3,
  not_found = 4,
  resource_exhausted = 5,
  deadline_exceeded = 6,
  cancelled = 7,
  io_error = 8,
  corruption = 9,
  closed = 10,
  internal = 11,
};

enum class OperationStage : std::uint8_t {
  none = 0,
  admission = 1,
  validation = 2,
  open = 3,
  build = 4,
  search = 5,
  mutation_prepare = 6,
  mutation_stage = 7,
  mutation_publish = 8,
  mutation_abort = 9,
  mutation_replay = 10,
  save = 11,
  export_rows = 12,
  checkpoint = 13,
  freeze = 14,
  stats = 15,
  close = 16,
  drain = 17,
  completion = 18,
};

enum class Retryability : std::uint8_t {
  not_retryable = 0,
  retryable = 1,
  retryable_with_backoff = 2,
};

// Stable programmatic detail. diagnostic() is deliberately not a branch key.
enum class StatusDetail : std::uint16_t {
  none = 0,
  malformed_struct = 1,
  unsupported_abi_version = 2,
  null_data = 3,
  dimension_mismatch = 4,
  unsupported_scalar_type = 5,
  arithmetic_overflow = 6,
  sink_too_small = 7,
  unknown_extension = 8,
  budget_denied = 9,
  cancellation_requested = 10,
  deadline_reached = 11,
  engine_exception = 12,
  allocation_failure = 13,
  already_exists = 14,
  readonly_instance = 15,
  operation_slot_absent = 16,
  invalid_score = 17,
  invalid_stride = 18,
};

class Status {
 public:
  Status() : header(current_struct_header<Status>()) {}

  Status(StatusCode code,
         OperationStage stage,
         StatusDetail detail = StatusDetail::none,
         Retryability retryability = Retryability::not_retryable,
         bool partial = false,
         std::string diagnostic = {})
      : header(current_struct_header<Status>()),
        code_(code),
        stage_(stage),
        retryability_(retryability),
        partial_(partial),
        detail_(detail),
        diagnostic_(std::move(diagnostic)) {}

  [[nodiscard]] static auto success() -> Status { return {}; }

  [[nodiscard]] static auto error(StatusCode code,
                                  OperationStage stage,
                                  StatusDetail detail = StatusDetail::none,
                                  std::string diagnostic = {},
                                  Retryability retryability = Retryability::not_retryable,
                                  bool partial = false) -> Status {
    return {code, stage, detail, retryability, partial, std::move(diagnostic)};
  }

  [[nodiscard]] auto ok() const noexcept -> bool { return code_ == StatusCode::ok; }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] auto code() const noexcept -> StatusCode { return code_; }
  [[nodiscard]] auto stage() const noexcept -> OperationStage { return stage_; }
  [[nodiscard]] auto retryability() const noexcept -> Retryability { return retryability_; }
  [[nodiscard]] auto partial() const noexcept -> bool { return partial_; }
  [[nodiscard]] auto detail() const noexcept -> StatusDetail { return detail_; }
  [[nodiscard]] auto diagnostic() const noexcept -> const std::string & { return diagnostic_; }

  VersionedStructHeader header{};

 private:
  StatusCode code_{StatusCode::ok};
  OperationStage stage_{OperationStage::none};
  Retryability retryability_{Retryability::not_retryable};
  bool partial_{};
  StatusDetail detail_{StatusDetail::none};
  std::uint16_t reserved_{};
  std::string diagnostic_{};
  std::uint64_t reserved_tail_[2]{};
};

template <class T>
class Result {
 public:
  Result(Status status)
      : header(current_struct_header<Result>()), status_(std::move(status)), value_(std::nullopt) {
    if (status_.ok()) {
      status_ = Status::error(StatusCode::internal,
                              OperationStage::validation,
                              StatusDetail::malformed_struct,
                              "successful Result must contain a value");
    }
  }

  Result(T value)
      : header(current_struct_header<Result>()),
        status_(Status::success()),
        value_(std::move(value)) {}

  [[nodiscard]] auto ok() const noexcept -> bool { return status_.ok() && value_.has_value(); }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] auto status() const noexcept -> const Status & { return status_; }
  [[nodiscard]] auto value() & -> T & { return value_.value(); }
  [[nodiscard]] auto value() const & -> const T & { return value_.value(); }
  [[nodiscard]] auto value() && -> T { return std::move(value_).value(); }

  VersionedStructHeader header{};

 private:
  Status status_{};
  std::optional<T> value_{};
  std::uint64_t reserved_[2]{};
};

[[nodiscard]] inline auto status_from_exception(OperationStage stage) noexcept -> Status {
  try {
    throw;
  } catch (const std::bad_alloc &error) {
    return Status::error(StatusCode::resource_exhausted,
                         stage,
                         StatusDetail::allocation_failure,
                         error.what(),
                         Retryability::retryable_with_backoff);
  } catch (const std::invalid_argument &error) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::engine_exception,
                         error.what());
  } catch (const std::out_of_range &error) {
    return Status::error(StatusCode::invalid_argument,
                         stage,
                         StatusDetail::engine_exception,
                         error.what());
  } catch (const std::exception &error) {
    return Status::error(StatusCode::internal, stage, StatusDetail::engine_exception, error.what());
  } catch (...) {
    return Status::error(StatusCode::internal,
                         stage,
                         StatusDetail::engine_exception,
                         "non-standard exception crossed an engine boundary");
  }
}

}  // namespace alaya::core
