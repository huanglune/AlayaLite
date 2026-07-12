// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/fusion/detail/fusion_builder_kernel.hpp"

namespace alaya {

// Source bridge retained only until FusionSegment becomes the registered runtime.
template <typename DistanceSpaceType,
          typename PrimaryGraph,
          typename SecondaryGraph,
          typename DataType = typename DistanceSpaceType::DataTypeAlias,
          typename DistanceType = typename DistanceSpaceType::DistanceTypeAlias,
          typename IDType = typename DistanceSpaceType::IDTypeAlias>
using FusionGraphBuilder = detail::FusionBuilderKernel<DistanceSpaceType,
                                                       PrimaryGraph,
                                                       SecondaryGraph,
                                                       DataType,
                                                       DistanceType,
                                                       IDType>;

}  // namespace alaya
