// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include "index/graph/qg/qg_builder.hpp"
#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER
  #include "index/graph/laser/qg/qg_builder.hpp"
#endif

// These facade names disambiguate the topology-only in-memory builder from
// LASER's persisted and searchable QG implementation. memory_qg intentionally
// exposes no Segment or artifact reader.

namespace alaya::disk_laser_qg {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER
using Builder = ::alaya::laser::QGBuilder;
using Factor = ::alaya::laser::Factor;
using Graph = ::alaya::laser::QuantizedGraph;
using Query = ::alaya::laser::QGQuery;
using Scanner = ::alaya::laser::QGScanner;
#endif

}  // namespace alaya::disk_laser_qg
