// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file bitwise.hpp
 * @brief Binary vector operations for RaBitQ quantization.
 *
 * Provides utilities for:
 * - popcount: Count set bits in binary vectors
 * - pack_binary: Convert 0/1 integer arrays to packed uint64 format
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include "platform/detect.hpp"

namespace alaya::laser::space {

/** @brief Packs 0/1 integer array into uint64 bit-packed format. */
inline void pack_binary(const int *ALAYA_RESTRICT bin_x,
                        uint64_t *ALAYA_RESTRICT binary,
                        size_t length) {
  for (size_t i = 0; i < length; i += 64) {
    uint64_t cur = 0;
    for (size_t j = 0; j < 64; ++j) {
      cur |= (static_cast<uint64_t>(bin_x[i + j]) << (63 - j));
    }
    *binary = cur;
    ++binary;
  }
}

}  // namespace alaya::laser::space
