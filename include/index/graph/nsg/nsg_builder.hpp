// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/nsg/detail/nsg_builder_kernel.hpp"

namespace alaya {

// Source bridge retained only until NsgSegment becomes the registered runtime.
template <typename DistanceSpaceType>
using NSGBuilder = detail::NsgBuilderKernel<DistanceSpaceType>;

}  // namespace alaya
