// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>

namespace alaya::core {

using AlgorithmId = std::uint64_t;

// Stable registry identities are pure core data. Domain enum conversion lives
// in index/compat.hpp and must never be pulled back into this header.
namespace algorithm {
inline constexpr AlgorithmId flat = 1;
inline constexpr AlgorithmId hnsw =
    2;  // retired engine — id reserved, never reuse (capability gate rejects)
inline constexpr AlgorithmId nsg =
    3;  // retired engine — id reserved, never reuse (capability gate rejects)
inline constexpr AlgorithmId fusion =
    4;  // retired engine — id reserved, never reuse (capability gate rejects)
inline constexpr AlgorithmId qg = 5;
inline constexpr AlgorithmId vamana =
    6;  // retired engine — id reserved, never reuse (capability gate rejects)
inline constexpr AlgorithmId laser = 7;
inline constexpr AlgorithmId diskann =
    8;  // retired engine — id reserved, never reuse (capability gate rejects)
}  // namespace algorithm

}  // namespace alaya::core
