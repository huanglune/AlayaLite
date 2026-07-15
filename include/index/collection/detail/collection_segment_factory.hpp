// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <filesystem>
#include <string>

#include "index/collection/detail/collection_target_builder.hpp"

namespace alaya::internal::collection::detail {

class CollectionSegmentFactory {
 public:
  [[nodiscard]] static auto open_entry(const std::filesystem::path &collection_root,
                                       const SegmentEntryV2 &entry,
                                       const CollectionSchema &schema,
                                       core::OpenContext &context)
      -> core::Result<core::AnySegment> {
    const auto *registration = find_collection_target_registration(entry.factory_key);
    if (registration == nullptr) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::open,
                                 core::StatusDetail::operation_slot_absent,
                                 "Collection segment factory key '" + entry.factory_key +
                                     "' is not supported");
    }
    if (entry.algorithm_id != registration->algorithm_id) {
      return core::Status::error(core::StatusCode::corruption,
                                 core::OperationStage::open,
                                 core::StatusDetail::malformed_struct,
                                 "Collection segment factory key disagrees with algorithm_id");
    }
    return registration->open(collection_root, entry, schema, context);
  }
};

}  // namespace alaya::internal::collection::detail
