// SPDX-License-Identifier: AGPL-3.0-only
#include "index/graph/diskann/diskann_index.hpp"

#include <type_traits>

static_assert(std::is_default_constructible_v<alaya::diskann::DiskANNBuildParams>);
static_assert(std::is_default_constructible_v<alaya::diskann::DiskANNLoadParams>);
static_assert(!std::is_copy_constructible_v<alaya::diskann::DiskANNIndex>);

auto main() -> int { return 0; }
