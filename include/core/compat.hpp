// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "core/algorithm_registry.hpp"

namespace alaya::core::compat {

// Source bridge only. Domain type conversion moved to index/compat.hpp so the
// core header closure remains independent of index, disk, and metric headers.
inline constexpr auto kAlgorithmFlat [[deprecated("use core::algorithm::flat")]] = algorithm::flat;
inline constexpr auto kAlgorithmHnsw [[deprecated("use core::algorithm::hnsw")]] = algorithm::hnsw;
inline constexpr auto kAlgorithmNsg [[deprecated("use core::algorithm::nsg")]] = algorithm::nsg;
inline constexpr auto kAlgorithmFusion [[deprecated("use core::algorithm::fusion")]] =
    algorithm::fusion;
inline constexpr auto kAlgorithmQg [[deprecated("use core::algorithm::qg")]] = algorithm::qg;
inline constexpr auto kAlgorithmVamana [[deprecated("use core::algorithm::vamana")]] =
    algorithm::vamana;
inline constexpr auto kAlgorithmLaser [[deprecated("use core::algorithm::laser")]] =
    algorithm::laser;

}  // namespace alaya::core::compat
