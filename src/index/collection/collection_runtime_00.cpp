// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// collection_runtime_00: create/open, single-row and batch write entries.
// One compile-cost-balanced Collection runtime unit; see CMakeLists.txt.

#include "index/collection/collection.hpp"

namespace alaya {

[[nodiscard]] auto Collection::create(CollectionOptions options)
    -> core::Result<std::shared_ptr<Collection>> {
  auto status = validate_options(options, core::OperationStage::build);
  if (!status.ok()) {
    return status;
  }
  try {
    const auto schema_path = facade_schema_path(options.root);
    if (std::filesystem::exists(schema_path) ||
        std::filesystem::exists(options.root / "schema.json") ||
        std::filesystem::exists(options.root / ".alaya_internal" /
                                internal::collection::kCollectionWalNamespace)) {
      return error(core::StatusCode::conflict,
                   core::OperationStage::build,
                   core::StatusDetail::already_exists,
                   "canonical Collection target already contains a collection layout");
    }
    std::filesystem::create_directories(options.root);
    auto process_lock =
        internal::collection::CollectionProcessLock::acquire(options.root, false, true);
    if (!process_lock.ok()) {
      return process_lock.status();
    }
    status = write_facade_schema(options);
    if (!status.ok()) {
      return status;
    }
    internal::collection::CollectionControlState state;
    state.auto_seal_rows = options.auto_seal_rows;
    status = internal::collection::CollectionControlStore::save(options.root, state);
    if (!status.ok()) {
      return status;
    }
    // 2B: materialize the empty active LASER segment directory before open so
    // make_active_registration can open it (create never re-creates on reopen).
    if (options.active_engine == core::algorithm::laser) {
      status = create_active_laser_segment(options, kActiveSegmentId, kActiveSegmentGeneration);
      if (!status.ok()) {
        return status;
      }
    }
    auto opened = open_segmented(options, state);
    if (!opened.ok()) {
      return opened.status();
    }
    auto result = std::shared_ptr<Collection>(new Collection(std::move(options),
                                                             std::move(opened).value(),
                                                             std::move(state),
                                                             false,
                                                             std::move(process_lock).value()));
    core::CheckpointContext context;
    context.durability_target = core::DurabilityTarget::full_checkpoint;
    auto checkpoint = result->checkpoint(context);
    if (!checkpoint.ok()) {
      return checkpoint.status();
    }
    return result;
  } catch (...) {
    return core::status_from_exception(core::OperationStage::build);
  }
}

[[nodiscard]] auto Collection::open(const std::filesystem::path &root)
    -> core::Result<std::shared_ptr<Collection>> {
  return open(root, CollectionOpenOptions{});
}

[[nodiscard]] auto Collection::open(const std::filesystem::path &root,
                                    CollectionOpenOptions open_options)
    -> core::Result<std::shared_ptr<Collection>> {
  if (root.empty()) {
    return error(core::StatusCode::invalid_argument,
                 core::OperationStage::open,
                 core::StatusDetail::malformed_struct,
                 "canonical Collection root is empty");
  }
  try {
    if (std::filesystem::is_regular_file(facade_schema_path(root))) {
      auto process_lock =
          internal::collection::CollectionProcessLock::acquire(root,
                                                               open_options.read_only,
                                                               !open_options.read_only);
      if (!process_lock.ok()) {
        return process_lock.status();
      }
      auto options = read_facade_schema(root);
      if (!options.ok()) {
        return options.status();
      }
      if (internal::collection::CollectionControlStore::exists(root)) {
        auto loaded_state = internal::collection::CollectionControlStore::load(root);
        if (!loaded_state.ok()) {
          return loaded_state.status();
        }
        auto state = std::move(loaded_state).value();
        options.value().auto_seal_rows = state.auto_seal_rows;
        auto status = validate_options(options.value(), core::OperationStage::open);
        if (!status.ok()) {
          return status;
        }
        if (open_options.read_only) {
          if (state.phase != internal::collection::CollectionControlPhase::idle) {
            return readonly_open_requires_recovery(
                "Collection control state is not idle; open in read-write mode to recover it "
                "first");
          }
          if (options.value().active_engine == core::algorithm::laser) {
            return readonly_open_requires_recovery(
                "read-only open of an active LASER mutation engine is not supported");
          }
        } else {
          status = normalize_control_state_before_open(root, state);
          if (!status.ok()) {
            return status;
          }
        }
        auto opened = open_segmented(options.value(), state, open_options.read_only);
        if (!opened.ok()) {
          return opened.status();
        }
        auto result = std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                                 std::move(opened).value(),
                                                                 std::move(state),
                                                                 open_options.read_only,
                                                                 std::move(process_lock).value()));
        if (!open_options.read_only) {
          status = result->recover_control_state();
          if (!status.ok()) {
            return status;
          }
        }
        return result;
      }
      if (open_options.read_only) {
        return readonly_open_requires_recovery(
            "read-only Collection open cannot initialize missing control state; open in "
            "read-write mode first");
      }
      internal::collection::CollectionControlState state;
      state.auto_seal_rows = options.value().auto_seal_rows;
      auto opened = open_segmented(options.value(), state, false);
      if (!opened.ok()) {
        return opened.status();
      }
      auto status = internal::collection::CollectionControlStore::save(root, state);
      if (!status.ok()) {
        return status;
      }
      return std::shared_ptr<Collection>(new Collection(std::move(options).value(),
                                                        std::move(opened).value(),
                                                        std::move(state),
                                                        false,
                                                        std::move(process_lock).value()));
    }

    return error(core::StatusCode::not_found,
                 core::OperationStage::open,
                 core::StatusDetail::none,
                 "no canonical Collection layout found at this path");
  } catch (...) {
    return core::status_from_exception(core::OperationStage::open);
  }
}

[[nodiscard]] auto Collection::add(const CollectionItem &item, CollectionWriteOptions options)
    -> core::Result<CollectionMutationReceipt> {
  options.retry_token =
      item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
  return write(item, internal::collection::WriteMode::insert_only, std::move(options));
}

[[nodiscard]] auto Collection::upsert(const CollectionItem &item, CollectionWriteOptions options)
    -> core::Result<CollectionMutationReceipt> {
  options.retry_token =
      item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
  return write(item, internal::collection::WriteMode::upsert, std::move(options));
}

[[nodiscard]] auto Collection::replace(const CollectionItem &item, CollectionWriteOptions options)
    -> core::Result<CollectionMutationReceipt> {
  options.retry_token =
      item.retry_token.empty() ? std::move(options.retry_token) : item.retry_token;
  return write(item, internal::collection::WriteMode::replace, std::move(options));
}

[[nodiscard]] auto Collection::remove(const core::LogicalId &logical_id,
                                      CollectionWriteOptions options)
    -> core::Result<CollectionMutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
  core::MutationContext context;
  return implementation_->erase(logical_id, context, std::move(options));
}

[[nodiscard]] auto Collection::mutate_batch(std::span<const CollectionBatchRow> rows,
                                            CollectionBatchMutationMode mode,
                                            CollectionWriteOptions options)
    -> core::Result<CollectionBatchMutationReceipt> {
  if (const auto writable = ensure_writable(core::OperationStage::admission); !writable.ok()) {
    return writable;
  }
  std::vector<internal::collection::BatchRowMutation> native_rows;
  native_rows.reserve(rows.size());
  for (const auto &row : rows) {
    internal::collection::BatchRowMutation native;
    native.logical_id = row.logical_id;
    native.vector = row.vector;
    native.metadata = row.metadata;
    native.document = row.document;
    native.retry_token = row.retry_token;
    switch (row.action) {
      case CollectionMutationAction::add:
        native.action = internal::collection::RowMutationAction::write;
        native.write_mode = internal::collection::WriteMode::insert_only;
        break;
      case CollectionMutationAction::upsert:
        native.action = internal::collection::RowMutationAction::write;
        native.write_mode = internal::collection::WriteMode::upsert;
        break;
      case CollectionMutationAction::replace:
        native.action = internal::collection::RowMutationAction::write;
        native.write_mode = internal::collection::WriteMode::replace;
        break;
      case CollectionMutationAction::remove:
        native.action = internal::collection::RowMutationAction::erase;
        native.write_mode = internal::collection::WriteMode::upsert;
        break;
    }
    native_rows.push_back(std::move(native));
  }
  internal::collection::BatchMutationRequest request;
  request.rows = native_rows;
  request.mode = mode;
  request.options = std::move(options);
  core::MutationContext context;
  auto receipt = implementation_->mutate_batch(request, context);
  if (receipt.ok()) {
    maybe_auto_seal();
  }
  return receipt;
}
}  // namespace alaya
