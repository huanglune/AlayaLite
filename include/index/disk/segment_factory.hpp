// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include "index/disk/disk_flat_builder.hpp"
#include "index/disk/disk_flat_searcher.hpp"
#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"

#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 1
#else
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 0
#endif

namespace alaya::disk {

[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool {
  switch (type) {
    case DiskIndexType::Flat:
      return true;
    case DiskIndexType::Laser:
#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
      return true;
#else
      return false;
#endif
  }
  return false;
}

// load_segment_from_manifest() (a DiskIndexType-generic factory entry
// point) and its detail::throw_unsupported_engine()/
// detail::laser_residency_request() helpers used to live here, but had zero
// production callers: every real segment-open path goes through a
// type-specific route instead (DiskFlatSegment::open(), LaserSegment::
// open()/open_directory()/open_collection(),
// CollectionSegmentFactory::open_entry()). Removed (U2-c manifest,
// decision 8, confirmed zero callers before deletion); the residency
// selection logic in particular moved to laser_segment.hpp, which is now
// the actual (and only) production call site, consulted from
// LaserSegment::open().

}  // namespace alaya::disk

#undef ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
