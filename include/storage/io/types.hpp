// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>

namespace alaya {

using AsyncIOCallback = void (*)(void *arg, int32_t result);

struct IORequest {
  void *buffer_{nullptr};
  size_t size_{0};
  uint64_t offset_{0};
  int32_t result_{0};
  void *user_data_{nullptr};

  IORequest() = default;

  IORequest(void *buf, size_t sz, uint64_t off, void *ud = nullptr)
      : buffer_(buf), size_(sz), offset_(off), user_data_(ud) {}

  [[nodiscard]] auto is_success() const -> bool {
    return result_ > 0 && static_cast<size_t>(result_) == size_;
  }
};

}  // namespace alaya
