// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_01: batch write, search/read, and checkpoint/seal/rotate/compact/gc wrappers.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::add_batch(std::span<const CollectionItem> items,
                                         CollectionBatchMutationMode mode,
                                         CollectionWriteOptions options)
    -> core::Result<CollectionBatchMutationReceipt> {
  std::vector<CollectionBatchRow> rows;
  rows.reserve(items.size());
  for (const auto &item : items) {
    rows.push_back(CollectionBatchRow{CollectionMutationAction::add,
                                      item.logical_id,
                                      item.vector,
                                      item.metadata,
                                      item.document,
                                      item.retry_token});
  }
  return mutate_batch(rows, mode, std::move(options));
}

[[nodiscard]] auto Collection::upsert_batch(std::span<const CollectionItem> items,
                                            CollectionBatchMutationMode mode,
                                            CollectionWriteOptions options)
    -> core::Result<CollectionBatchMutationReceipt> {
  std::vector<CollectionBatchRow> rows;
  rows.reserve(items.size());
  for (const auto &item : items) {
    rows.push_back(CollectionBatchRow{CollectionMutationAction::upsert,
                                      item.logical_id,
                                      item.vector,
                                      item.metadata,
                                      item.document,
                                      item.retry_token});
  }
  return mutate_batch(rows, mode, std::move(options));
}

[[nodiscard]] auto Collection::search(const core::TypedTensorView &query,
                                      const core::SearchOptions &options,
                                      core::SearchContext &context)
    -> core::Result<CollectionSearchResponse> {
  return search(query, options, context, CollectionFilter{});
}

[[nodiscard]] auto Collection::search(const core::TypedTensorView &query,
                                      const core::SearchOptions &options,
                                      core::SearchContext &context,
                                      const CollectionFilter &filter)
    -> core::Result<CollectionSearchResponse> {
  if (query.rows != 1) {
    return error(core::StatusCode::invalid_argument,
                 core::OperationStage::validation,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection single search requires exactly one query row");
  }
  return execute_search(query, options, context, filter);
}

[[nodiscard]] auto Collection::search(const core::TypedTensorView &query, std::uint64_t top_k)
    -> core::Result<CollectionSearchResponse> {
  core::SearchOptions options(top_k);
  core::SearchContext context;
  return search(query, options, context);
}

[[nodiscard]] auto Collection::search(const core::TypedTensorView &query,
                                      std::uint64_t top_k,
                                      const CollectionFilter &filter,
                                      core::FilterPolicy policy)
    -> core::Result<CollectionSearchResponse> {
  core::SearchOptions options(top_k);
  options.filter_policy = policy;
  core::SearchContext context;
  return search(query, options, context, filter);
}

[[nodiscard]] auto Collection::batch_search(const core::TypedTensorView &queries,
                                            const core::SearchOptions &options,
                                            core::SearchContext &context)
    -> core::Result<CollectionSearchResponse> {
  return batch_search(queries, options, context, CollectionFilter{});
}

[[nodiscard]] auto Collection::batch_search(const core::TypedTensorView &queries,
                                            const core::SearchOptions &options,
                                            core::SearchContext &context,
                                            const CollectionFilter &filter)
    -> core::Result<CollectionSearchResponse> {
  return execute_search(queries, options, context, filter);
}

[[nodiscard]] auto Collection::batch_search(const core::TypedTensorView &queries,
                                            std::uint64_t top_k)
    -> core::Result<CollectionSearchResponse> {
  core::SearchOptions options(top_k);
  core::SearchContext context;
  return batch_search(queries, options, context);
}

[[nodiscard]] auto Collection::batch_search(const core::TypedTensorView &queries,
                                            std::uint64_t top_k,
                                            const CollectionFilter &filter,
                                            core::FilterPolicy policy)
    -> core::Result<CollectionSearchResponse> {
  core::SearchOptions options(top_k);
  options.filter_policy = policy;
  core::SearchContext context;
  return batch_search(queries, options, context, filter);
}

[[nodiscard]] auto Collection::get_by_id(const core::LogicalId &logical_id,
                                         CollectionProjection projection)
    -> core::Result<CollectionRecord> {
  return implementation_->get_by_id(logical_id, projection);
}

[[nodiscard]] auto Collection::records(CollectionProjection projection, std::size_t limit)
    -> core::Result<std::vector<CollectionRecord>> {
  return scan(CollectionFilter{}, limit, projection);
}

[[nodiscard]] auto Collection::scan(const CollectionFilter &filter,
                                    std::size_t limit,
                                    CollectionProjection projection)
    -> core::Result<std::vector<CollectionRecord>> {
  return implementation_->scalar_query(filter, limit, projection);
}

[[nodiscard]] auto Collection::checkpoint(core::CheckpointContext &context)
    -> core::Result<CollectionCheckpointReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::checkpoint); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return checkpoint_locked(context);
}

[[nodiscard]] auto Collection::checkpoint() -> core::Result<CollectionCheckpointReceipt> {
  core::CheckpointContext context;
  context.durability_target = core::DurabilityTarget::full_checkpoint;
  return checkpoint(context);
}

[[nodiscard]] auto Collection::consolidate(CollectionConsolidateOptions options)
    -> core::Result<CollectionConsolidateReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::checkpoint); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  auto receipt = implementation_->consolidate(options.num_threads,
                                              options.r_target,
                                              options.reclaim_slots,
                                              options.bloom_consolidate);
  if (!receipt.ok()) {
    return receipt.status();
  }
  return CollectionConsolidateReceipt{receipt.value().active_segment_id,
                                      receipt.value().active_generation};
}

[[nodiscard]] auto Collection::seal(CollectionSealOptions options)
    -> core::Result<CollectionSealReceipt> {
  core::SealContext context;
  return seal(context, std::move(options));
}

[[nodiscard]] auto Collection::seal(core::SealContext &context, CollectionSealOptions options)
    -> core::Result<CollectionSealReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::freeze); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return seal_locked(context, options);
}

[[nodiscard]] auto Collection::prepare_successor(CollectionSealOptions options)
    -> core::Result<CollectionRotationHandle> {
  core::SealContext context;
  return prepare_successor(context, std::move(options));
}

[[nodiscard]] auto Collection::prepare_successor(core::SealContext &context,
                                                 CollectionSealOptions options)
    -> core::Result<CollectionRotationHandle> {
  if (const auto writable = ensure_writable(core::OperationStage::freeze); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return prepare_successor_locked(context, options);
}

[[nodiscard]] auto Collection::rotate_to_successor(const CollectionRotationHandle &handle)
    -> core::Result<CollectionSealReceipt> {
  core::SealContext context;
  return rotate_to_successor(handle, context);
}

[[nodiscard]] auto Collection::rotate_to_successor(const CollectionRotationHandle &handle,
                                                   core::SealContext &context)
    -> core::Result<CollectionSealReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::freeze); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return rotate_to_successor_locked(handle, context);
}

[[nodiscard]] auto Collection::compact() -> core::Result<CollectionCompactReceipt> {
  core::SealContext context;
  return compact(context);
}

[[nodiscard]] auto Collection::compact(core::SealContext &context)
    -> core::Result<CollectionCompactReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::freeze); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return compact_locked(context);
}

[[nodiscard]] auto Collection::gc() -> core::Result<CollectionGcReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::save); !writable.ok()) {
    return writable;
  }
  std::lock_guard lock(control_mutex_);
  return gc_locked();
}
}  // namespace alaya
