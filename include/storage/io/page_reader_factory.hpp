// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <memory>

#include "storage/io/sync_page_reader.hpp"

namespace alaya::storage::io {

// M1 exposes the index-neutral factory contract with the synchronous reference
// backend. M2 will select platform asynchronous backends here.
[[nodiscard]] inline auto open_page_reader(const std::filesystem::path &path,
                                           const ReaderOptions &options = {})
    -> std::unique_ptr<PageReader> {
  return std::make_unique<SyncPageReader>(path, options);
}

}  // namespace alaya::storage::io
