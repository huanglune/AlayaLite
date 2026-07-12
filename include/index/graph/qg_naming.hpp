// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/qg/qg_builder.hpp"
#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER
  #include "index/graph/laser/qg/qg_builder.hpp"
#endif

// These facade names disambiguate AlayaLite's two unrelated QG surfaces without
// changing their historical names or physical include paths.
namespace alaya::memory_qg {

template <typename DistanceSpaceType>
using Builder = ::alaya::QGBuilder<DistanceSpaceType>;

template <typename DataType>
using Quantizer = ::alaya::RaBitQQuantizer<DataType>;

}  // namespace alaya::memory_qg

namespace alaya::disk_laser_qg {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER
using Builder = ::alaya::laser::QGBuilder;
using Factor = ::alaya::laser::Factor;
using Graph = ::alaya::laser::QuantizedGraph;
using Query = ::alaya::laser::QGQuery;
using Scanner = ::alaya::laser::QGScanner;
#endif

}  // namespace alaya::disk_laser_qg
