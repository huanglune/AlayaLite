// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

#include "core/capabilities.hpp"

namespace alaya::core {

// Contract v3 is frozen. Engines register operations in this table; they must
// not grow request/response/context structs or establish a parallel boundary.
struct OperationHandle {
  using Cancel = void (*)(void *) noexcept;  // NOLINT(readability/casting)

  VersionedStructHeader header{};
  std::shared_ptr<void> state{};
  Cancel cancel_operation{};
  std::uint64_t reserved[4]{};

  OperationHandle() : header(current_struct_header<OperationHandle>()) {}
  OperationHandle(std::shared_ptr<void> operation_state, Cancel cancel)
      : header(current_struct_header<OperationHandle>()),
        state(std::move(operation_state)),
        cancel_operation(cancel) {}

  void cancel() const noexcept {
    if (state != nullptr && cancel_operation != nullptr) {
      cancel_operation(state.get());
    }
  }

  [[nodiscard]] auto valid() const noexcept -> bool {
    return state != nullptr && cancel_operation != nullptr;
  }
};

struct SearchCompletion {
  VersionedStructHeader header{};
  std::function<void(Status)> callback{};
  std::uint64_t reserved[4]{};

  SearchCompletion() : header(current_struct_header<SearchCompletion>()) {}
  explicit SearchCompletion(std::function<void(Status)> completion)
      : header(current_struct_header<SearchCompletion>()), callback(std::move(completion)) {}
};

struct AnySegmentOperationTable {
  using StartSearch = Result<OperationHandle> (*)(const std::shared_ptr<void> &,
                                                  SearchRequest,
                                                  SearchCompletion) noexcept;
  using MutationOperation = Status (*)(const std::shared_ptr<void> &,
                                       const OpaqueOperationRequest &,
                                       MutationContext &,
                                       MutationToken *) noexcept;
  using MutationTokenOperation = Status (*)(const std::shared_ptr<void> &,
                                            MutationToken &,
                                            MutationContext &) noexcept;
  using Save = Status (*)(const std::shared_ptr<void> &,
                          ArtifactWriter &,
                          const SaveOptions &,
                          ArtifactManifest &) noexcept;
  using ExportRows = Status (*)(const std::shared_ptr<void> &,
                                const OpaqueOperationRequest &,
                                ExportCursor &) noexcept;
  using Checkpoint = Status (*)(const std::shared_ptr<void> &,
                                CheckpointContext &,
                                CheckpointToken &) noexcept;
  using FreezeSnapshot = Status (*)(const std::shared_ptr<void> &,
                                    SealContext &,
                                    FreezeToken &) noexcept;
  using Stats = Status (*)(const std::shared_ptr<void> &, SegmentStats &) noexcept;
  using Close = Status (*)(const std::shared_ptr<void> &) noexcept;
  using Drain = Status (*)(const std::shared_ptr<void> &, const Deadline &) noexcept;

  std::uint32_t table_size{};
  std::uint32_t table_version{kOperationTableVersion};
  StartSearch start_search{};  // required
  StartSearch start_batch_search{};
  MutationOperation prepare_mutation{};
  MutationTokenOperation stage_mutation{};
  MutationTokenOperation publish_mutation{};
  MutationTokenOperation abort_mutation{};
  MutationOperation replay_mutation{};
  Save save{};
  ExportRows export_rows{};
  Checkpoint checkpoint{};
  FreezeSnapshot freeze_snapshot{};
  Stats stats{};
  Close close{};
  Drain drain{};
  std::uint64_t reserved[8]{};
};

namespace detail {

struct AsyncOperationState {
  std::atomic_bool cancelled{};
  std::atomic_bool completion_started{};
  SearchRequest request{};
  SearchContext context{};
  CancellationToken external_cancellation{};
  SearchCompletion completion{};

  static auto combined_cancelled(const void *raw) noexcept -> bool {
    const auto &state = *static_cast<const AsyncOperationState *>(raw);
    return state.cancelled.load(std::memory_order_acquire) ||
           state.external_cancellation.requested();
  }

  void bind_context() {
    context = *request.context;
    if (request.options.deadline_steady_nanoseconds != 0 &&
        (!context.deadline.enabled ||
         request.options.deadline_steady_nanoseconds < context.deadline.steady_clock_nanoseconds)) {
      context.deadline.enabled = true;
      context.deadline.steady_clock_nanoseconds = request.options.deadline_steady_nanoseconds;
    }
    external_cancellation = context.cancellation;
    context.cancellation.state = this;
    context.cancellation.is_cancelled = &combined_cancelled;
    request.context = std::addressof(context);
  }

  void cancel() noexcept { cancelled.store(true, std::memory_order_release); }

  void finish(Status status, const std::shared_ptr<AsyncOperationState> &self) noexcept {
    if (completion_started.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    auto delivery = [self, status = std::move(status)]() mutable {
      try {
        self->completion.callback(std::move(status));
      } catch (...) {
        // Completion code is outside the engine boundary. It cannot cause a
        // second completion or unwind through a runtime lane.
      }
      self->request.lifetime_pin.reset();
      self->completion.callback = {};
    };
    context.lane.dispatch(std::move(delivery));
  }
};

inline void invalidate_failed_response(SearchRequest &request, const Status &status) noexcept {
  if (request.response != nullptr &&
      request.options.partial_result_policy == PartialResultPolicy::discard) {
    request.response->invalidate(status);
  }
}

inline auto apply_terminal_control(SearchRequest &request, Status status) -> Status {
  if (!status.ok()) {
    invalidate_failed_response(request, status);
    return status;
  }
  const auto control = validate_runtime_control(request.context->deadline,
                                                request.context->cancellation,
                                                OperationStage::search);
  if (control.ok()) {
    return status;
  }
  if (request.options.partial_result_policy == PartialResultPolicy::retain &&
      request.response != nullptr) {
    auto partial = Status::error(control.code(),
                                 OperationStage::search,
                                 control.detail(),
                                 control.diagnostic(),
                                 control.retryability(),
                                 true);
    for (RowCount query = 0; query < request.response->query_count; ++query) {
      if (request.response->statuses[query].ok()) {
        request.response->statuses[query] = partial;
        request.response->completeness[query] = SearchCompleteness::cancelled_partial;
      }
    }
    return partial;
  }
  request.response->invalidate(control);
  return control;
}

template <class Segment, bool Batch>
struct SyncSearchState final : AsyncOperationState {
  std::shared_ptr<Segment> segment{};

  void run(const std::shared_ptr<SyncSearchState> &self) noexcept {
    Status status;
    try {
      status = validate_runtime_control(this->context.deadline,
                                        this->context.cancellation,
                                        OperationStage::search);
      if (status.ok()) {
        if constexpr (Batch) {
          status = segment->batch_search(this->request);
        } else {
          status = segment->search(this->request);
        }
      }
    } catch (...) {
      status = status_from_exception(OperationStage::search);
    }
    status = apply_terminal_control(this->request, std::move(status));
    this->finish(std::move(status), self);
  }
};

template <class Segment, bool Batch>
auto start_sync_search(const std::shared_ptr<void> &instance,
                       SearchRequest request,
                       SearchCompletion completion) noexcept -> Result<OperationHandle> {
  try {
    auto state = std::make_shared<SyncSearchState<Segment, Batch>>();
    state->segment = std::static_pointer_cast<Segment>(instance);
    state->request = std::move(request);
    state->completion = std::move(completion);
    state->bind_context();
    std::thread([state] {
      state->run(state);
    }).detach();
    return OperationHandle(state, [](void *raw) noexcept {
      static_cast<SyncSearchState<Segment, Batch> *>(raw)->cancel();
    });
  } catch (...) {
    return status_from_exception(OperationStage::admission);
  }
}

inline auto start_immediate(SearchRequest request,
                            SearchCompletion completion,
                            Status status) noexcept -> Result<OperationHandle> {
  try {
    auto state = std::make_shared<AsyncOperationState>();
    state->request = std::move(request);
    state->completion = std::move(completion);
    state->bind_context();
    std::thread([state, status = std::move(status)]() mutable {
      auto delivered = status;
      if (state->cancelled.load(std::memory_order_acquire) && delivered.ok()) {
        delivered = Status::error(StatusCode::cancelled,
                                  OperationStage::search,
                                  StatusDetail::cancellation_requested,
                                  "operation was cancelled before completion");
        invalidate_failed_response(state->request, delivered);
      }
      state->finish(std::move(delivered), state);
    }).detach();
    return OperationHandle(state, [](void *raw) noexcept {
      static_cast<AsyncOperationState *>(raw)->cancel();
    });
  } catch (...) {
    return status_from_exception(OperationStage::admission);
  }
}

template <class Segment>
struct SyncOperationAdapter {
  static auto table() noexcept -> const AnySegmentOperationTable & {
    static const AnySegmentOperationTable operations = [] {
      AnySegmentOperationTable value{};
      value.table_size = sizeof(AnySegmentOperationTable);
      value.table_version = kOperationTableVersion;
      value.start_search = &start_sync_search<Segment, false>;
      if constexpr (BatchSearchable<Segment>) {
        value.start_batch_search = &start_sync_search<Segment, true>;
      }
      if constexpr (Mutable<Segment>) {
        value.prepare_mutation = [](const std::shared_ptr<void> &instance,
                                    const OpaqueOperationRequest &request,
                                    MutationContext &context,
                                    MutationToken *token) noexcept {
          try {
            if (token == nullptr) {
              return Status::error(StatusCode::invalid_argument,
                                   OperationStage::mutation_prepare,
                                   StatusDetail::null_data,
                                   "mutation token output is null");
            }
            return std::static_pointer_cast<Segment>(instance)->prepare_mutation(request,
                                                                                 context,
                                                                                 *token);
          } catch (...) {
            return status_from_exception(OperationStage::mutation_prepare);
          }
        };
        value.stage_mutation = [](const std::shared_ptr<void> &instance,
                                  MutationToken &token,
                                  MutationContext &context) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->stage_mutation(token, context);
          } catch (...) {
            return status_from_exception(OperationStage::mutation_stage);
          }
        };
        value.publish_mutation = [](const std::shared_ptr<void> &instance,
                                    MutationToken &token,
                                    MutationContext &context) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->publish_mutation(token, context);
          } catch (...) {
            return status_from_exception(OperationStage::mutation_publish);
          }
        };
        value.abort_mutation = [](const std::shared_ptr<void> &instance,
                                  MutationToken &token,
                                  MutationContext &context) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->abort_mutation(token, context);
          } catch (...) {
            return status_from_exception(OperationStage::mutation_abort);
          }
        };
        value.replay_mutation = [](const std::shared_ptr<void> &instance,
                                   const OpaqueOperationRequest &request,
                                   MutationContext &context,
                                   MutationToken *) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->replay_mutation(request, context);
          } catch (...) {
            return status_from_exception(OperationStage::mutation_replay);
          }
        };
      }
      if constexpr (Saveable<Segment>) {
        value.save = [](const std::shared_ptr<void> &instance,
                        ArtifactWriter &writer,
                        const SaveOptions &options,
                        ArtifactManifest &manifest) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->save(writer, options, manifest);
          } catch (...) {
            return status_from_exception(OperationStage::save);
          }
        };
      }
      if constexpr (Exportable<Segment>) {
        value.export_rows = [](const std::shared_ptr<void> &instance,
                               const OpaqueOperationRequest &request,
                               ExportCursor &cursor) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->export_rows(request, cursor);
          } catch (...) {
            return status_from_exception(OperationStage::export_rows);
          }
        };
      }
      if constexpr (Checkpointable<Segment>) {
        value.checkpoint = [](const std::shared_ptr<void> &instance,
                              CheckpointContext &context,
                              CheckpointToken &token) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->checkpoint(context, token);
          } catch (...) {
            return status_from_exception(OperationStage::checkpoint);
          }
        };
      }
      if constexpr (Freezable<Segment>) {
        value.freeze_snapshot = [](const std::shared_ptr<void> &instance,
                                   SealContext &context,
                                   FreezeToken &token) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->freeze_snapshot(context, token);
          } catch (...) {
            return status_from_exception(OperationStage::freeze);
          }
        };
      }
      if constexpr (StatsProvider<Segment>) {
        value.stats = [](const std::shared_ptr<void> &instance, SegmentStats &stats) noexcept {
          return std::static_pointer_cast<Segment>(instance)->stats(stats);
        };
      }
      if constexpr (Closable<Segment>) {
        value.close = [](const std::shared_ptr<void> &instance) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->close();
          } catch (...) {
            return status_from_exception(OperationStage::close);
          }
        };
        value.drain = [](const std::shared_ptr<void> &instance, const Deadline &deadline) noexcept {
          try {
            return std::static_pointer_cast<Segment>(instance)->drain(deadline);
          } catch (...) {
            return status_from_exception(OperationStage::drain);
          }
        };
      }
      return value;
    }();
    return operations;
  }
};

}  // namespace detail

class AnySegment {
 public:
  AnySegment() = default;

  template <Searchable Segment>
  [[nodiscard]] static auto from_sync(std::shared_ptr<Segment> segment,
                                      SegmentInstanceConfig config = {}) -> Result<AnySegment> {
    if (segment == nullptr) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::admission,
                           StatusDetail::null_data,
                           "AnySegment instance is null");
    }
    auto descriptor = segment->descriptor();
    return from_raw(std::move(segment),
                    std::addressof(detail::SyncOperationAdapter<Segment>::table()),
                    std::move(descriptor),
                    std::move(config));
  }

  [[nodiscard]] static auto from_raw(std::shared_ptr<void> instance,
                                     const AnySegmentOperationTable *operations,
                                     Descriptor descriptor,
                                     SegmentInstanceConfig config = {}) -> Result<AnySegment> {
    if (instance == nullptr || operations == nullptr) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::admission,
                           StatusDetail::null_data,
                           "AnySegment instance or operation table is null");
    }
    constexpr auto required_table_size = offsetof(AnySegmentOperationTable, start_batch_search) +
                                         sizeof(AnySegmentOperationTable::StartSearch);
    if (operations->table_version != kOperationTableVersion ||
        operations->table_size < required_table_size || operations->start_search == nullptr) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::admission,
                           StatusDetail::malformed_struct,
                           "AnySegment operation table is incompatible or lacks search");
    }
    AnySegment segment;
    segment.instance_ = std::move(instance);
    segment.operations_ = operations;
    segment.descriptor_ = std::move(descriptor);
    segment.config_ = std::move(config);
    return segment;
  }

  [[nodiscard]] auto descriptor() const noexcept -> Descriptor { return descriptor_; }

  [[nodiscard]] auto capabilities() const noexcept -> RuntimeCapabilities {
    RuntimeCapabilities capabilities;
    if (operations_ == nullptr) {
      return capabilities;
    }
    auto enable = [&](OperationCapability capability, bool available) {
      const auto bit = capability_bit(capability);
      if (available && (config_.enabled_operations & bit) != 0) {
        capabilities.operations |= bit;
      }
    };
    enable(OperationCapability::search, operations_->start_search != nullptr);
    enable(OperationCapability::batch_search, operations_->start_batch_search != nullptr);
    const auto mutation_bundle =
        operations_->prepare_mutation != nullptr && operations_->stage_mutation != nullptr &&
        operations_->publish_mutation != nullptr && operations_->abort_mutation != nullptr &&
        operations_->replay_mutation != nullptr && !config_.readonly;
    enable(OperationCapability::mutation, mutation_bundle);
    enable(OperationCapability::save, operations_->save != nullptr);
    enable(OperationCapability::export_rows, operations_->export_rows != nullptr);
    enable(OperationCapability::checkpoint, operations_->checkpoint != nullptr);
    enable(OperationCapability::freeze, operations_->freeze_snapshot != nullptr);
    enable(OperationCapability::stats, operations_->stats != nullptr);
    enable(OperationCapability::close, operations_->close != nullptr);
    enable(OperationCapability::drain, operations_->drain != nullptr);
    capabilities.concurrency = config_.concurrency;
    return capabilities;
  }

  [[nodiscard]] auto start_search(SearchRequest request, SearchCompletion completion) const
      -> Result<OperationHandle> {
    return start(false, std::move(request), std::move(completion));
  }

  [[nodiscard]] auto start_batch_search(SearchRequest request, SearchCompletion completion) const
      -> Result<OperationHandle> {
    return start(true, std::move(request), std::move(completion));
  }

  [[nodiscard]] auto search(SearchRequest request) const -> Status {
    if (request.context == nullptr) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::validation,
                           StatusDetail::null_data,
                           "SearchContext is null");
    }
    // A sync wrapper must not wait for a callback queued onto its own lane.
    SearchContext wait_context = *request.context;
    wait_context.lane = RuntimeLane{};
    request.context = std::addressof(wait_context);

    struct WaitState {
      std::mutex mutex;
      std::condition_variable ready;
      bool done{};
      Status status{};
    };
    auto wait = std::make_shared<WaitState>();
    SearchCompletion completion([wait](Status status) {
      {
        std::lock_guard lock(wait->mutex);
        wait->status = std::move(status);
        wait->done = true;
      }
      wait->ready.notify_one();
    });
    auto started = request.queries.rows == 1
                       ? start_search(std::move(request), std::move(completion))
                       : start_batch_search(std::move(request), std::move(completion));
    if (!started.ok()) {
      return started.status();
    }
    auto handle = std::move(started).value();
    std::unique_lock lock(wait->mutex);
    wait->ready.wait(lock, [&] {
      return wait->done;
    });
    return wait->status;
  }

  [[nodiscard]] auto stats(SegmentStats &stats) const -> Status {
    if (operations_ == nullptr || operations_->stats == nullptr ||
        !capabilities().supports(OperationCapability::stats)) {
      return unsupported(OperationStage::stats);
    }
    return operations_->stats(instance_, stats);
  }

  [[nodiscard]] auto prepare_mutation(const OpaqueOperationRequest &request,
                                      MutationContext &context,
                                      MutationToken &token) -> Status {
    if (operations_ == nullptr || operations_->prepare_mutation == nullptr ||
        !capabilities().supports(OperationCapability::mutation)) {
      return unsupported(OperationStage::mutation_prepare);
    }
    return operations_->prepare_mutation(instance_, request, context, &token);
  }

  [[nodiscard]] auto stage_mutation(MutationToken &token, MutationContext &context) -> Status {
    if (operations_ == nullptr || operations_->stage_mutation == nullptr ||
        !capabilities().supports(OperationCapability::mutation)) {
      return unsupported(OperationStage::mutation_stage);
    }
    return operations_->stage_mutation(instance_, token, context);
  }

  [[nodiscard]] auto publish_mutation(MutationToken &token, MutationContext &context) -> Status {
    if (operations_ == nullptr || operations_->publish_mutation == nullptr ||
        !capabilities().supports(OperationCapability::mutation)) {
      return unsupported(OperationStage::mutation_publish);
    }
    return operations_->publish_mutation(instance_, token, context);
  }

  [[nodiscard]] auto abort_mutation(MutationToken &token, MutationContext &context) -> Status {
    if (operations_ == nullptr || operations_->abort_mutation == nullptr ||
        !capabilities().supports(OperationCapability::mutation)) {
      return unsupported(OperationStage::mutation_abort);
    }
    return operations_->abort_mutation(instance_, token, context);
  }

  [[nodiscard]] auto replay_mutation(const OpaqueOperationRequest &request,
                                     MutationContext &context) -> Status {
    if (operations_ == nullptr || operations_->replay_mutation == nullptr ||
        !capabilities().supports(OperationCapability::mutation)) {
      return unsupported(OperationStage::mutation_replay);
    }
    return operations_->replay_mutation(instance_, request, context, nullptr);
  }

  [[nodiscard]] auto save(ArtifactWriter &writer,
                          const SaveOptions &options,
                          ArtifactManifest &manifest) const -> Status {
    if (operations_ == nullptr || operations_->save == nullptr ||
        !capabilities().supports(OperationCapability::save)) {
      return unsupported(OperationStage::save);
    }
    return operations_->save(instance_, writer, options, manifest);
  }

  [[nodiscard]] auto export_rows(const OpaqueOperationRequest &request, ExportCursor &cursor) const
      -> Status {
    if (operations_ == nullptr || operations_->export_rows == nullptr ||
        !capabilities().supports(OperationCapability::export_rows)) {
      return unsupported(OperationStage::export_rows);
    }
    return operations_->export_rows(instance_, request, cursor);
  }

  [[nodiscard]] auto checkpoint(CheckpointContext &context, CheckpointToken &token) -> Status {
    if (operations_ == nullptr || operations_->checkpoint == nullptr ||
        !capabilities().supports(OperationCapability::checkpoint)) {
      return unsupported(OperationStage::checkpoint);
    }
    return operations_->checkpoint(instance_, context, token);
  }

  [[nodiscard]] auto freeze_snapshot(SealContext &context, FreezeToken &token) -> Status {
    if (operations_ == nullptr || operations_->freeze_snapshot == nullptr ||
        !capabilities().supports(OperationCapability::freeze)) {
      return unsupported(OperationStage::freeze);
    }
    return operations_->freeze_snapshot(instance_, context, token);
  }

  [[nodiscard]] auto close() -> Status {
    if (operations_ == nullptr || operations_->close == nullptr ||
        !capabilities().supports(OperationCapability::close)) {
      return unsupported(OperationStage::close);
    }
    return operations_->close(instance_);
  }

  [[nodiscard]] auto drain(const Deadline &deadline) -> Status {
    if (operations_ == nullptr || operations_->drain == nullptr ||
        !capabilities().supports(OperationCapability::drain)) {
      return unsupported(OperationStage::drain);
    }
    return operations_->drain(instance_, deadline);
  }

 private:
  [[nodiscard]] auto start(bool batch, SearchRequest request, SearchCompletion completion) const
      -> Result<OperationHandle> {
    if (operations_ == nullptr || instance_ == nullptr) {
      return Status::error(StatusCode::closed,
                           OperationStage::admission,
                           StatusDetail::operation_slot_absent,
                           "AnySegment is empty");
    }
    if (!is_current_struct(request) || !is_current_struct(request.options) ||
        !is_current_struct(completion) || !completion.callback) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::validation,
                           StatusDetail::malformed_struct,
                           "search request, options, or completion is malformed");
    }
    if (request.context == nullptr || request.response == nullptr ||
        !is_current_struct(*request.context)) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::validation,
                           StatusDetail::null_data,
                           "search context or response is null/incompatible");
    }
    const auto tensor_status =
        validate_tensor(request.queries, descriptor_.dim, OperationStage::validation);
    if (!tensor_status.ok()) {
      return tensor_status;
    }
    const auto response_status = validate_response(*request.response,
                                                   request.queries.rows,
                                                   request.options.top_k,
                                                   OperationStage::validation);
    if (!response_status.ok()) {
      return response_status;
    }
    if ((!batch && request.queries.rows != 1) || (batch && request.queries.rows == 1)) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::validation,
                           StatusDetail::malformed_struct,
                           batch ? "batch search requires zero or multiple query rows"
                                 : "single search requires exactly one query row");
    }
    if (request.options.top_k == 0 || request.queries.rows == 0) {
      initialize_empty_response(*request.response,
                                request.queries.rows,
                                request.options.top_k == 0
                                    ? SearchCompleteness::complete_k
                                    : SearchCompleteness::eligible_exhausted);
      return detail::start_immediate(std::move(request), std::move(completion), Status::success());
    }
    const auto slot = batch ? operations_->start_batch_search : operations_->start_search;
    const auto capability = batch ? OperationCapability::batch_search : OperationCapability::search;
    if (slot == nullptr || !capabilities().supports(capability)) {
      return unsupported(OperationStage::admission);
    }
    return slot(instance_, std::move(request), std::move(completion));
  }

  [[nodiscard]] static auto unsupported(OperationStage stage) -> Status {
    return Status::error(StatusCode::not_supported,
                         stage,
                         StatusDetail::operation_slot_absent,
                         "AnySegment operation slot is unavailable");
  }

  std::shared_ptr<void> instance_{};
  const AnySegmentOperationTable *operations_{};
  Descriptor descriptor_{};
  SegmentInstanceConfig config_{};
};

static_assert(offsetof(AnySegmentOperationTable, table_size) == 0);
static_assert(offsetof(AnySegmentOperationTable, table_version) == sizeof(std::uint32_t));

}  // namespace alaya::core
