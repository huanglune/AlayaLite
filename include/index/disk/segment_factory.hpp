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
#include "index/disk/vamana_segment_builder.hpp"
#include "index/disk/vamana_segment_searcher.hpp"

#if defined(ALAYA_ENABLE_LASER) && (ALAYA_ENABLE_LASER + 0) != 0
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 1
#else
  #define ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED 0
#endif

#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
  #include "index/disk/laser_segment_searcher.hpp"
#endif

namespace alaya::disk {

[[nodiscard]] constexpr auto engine_supported_v1(DiskIndexType type) noexcept -> bool {
  switch (type) {
    case DiskIndexType::Flat:
      return true;
    case DiskIndexType::Vamana:
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

namespace detail {

[[noreturn]] inline auto throw_unsupported_engine(DiskIndexType type) -> void {
  throw std::runtime_error(std::string("DiskSegmentFactory: engine '") +
                           std::string(index_type_to_string(type)) + "' not implemented in v1");
}

}  // namespace detail

[[nodiscard]] inline auto load_segment_from_manifest(const std::filesystem::path &seg_dir)
    -> std::shared_ptr<SegmentSearcher> {
  const auto sm = SegmentManifest::load(seg_dir / "manifest.txt");
  if (!engine_supported_v1(sm.index_type)) {
    detail::throw_unsupported_engine(sm.index_type);
  }
  std::shared_ptr<SegmentSearcher> searcher;
  switch (sm.index_type) {
    case DiskIndexType::Flat:
      searcher = std::make_shared<DiskFlatSegmentSearcher>(seg_dir);
      break;
    case DiskIndexType::Vamana:
      searcher = std::make_shared<VamanaSegmentSearcher>(seg_dir);
      break;
    case DiskIndexType::Laser:
#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
      searcher = std::make_shared<LaserSegmentSearcher>(seg_dir);
      break;
#else
      detail::throw_unsupported_engine(sm.index_type);
#endif
  }
  if (!engine_supported_v1(searcher->type())) {
    throw std::runtime_error(
        "DiskSegmentFactory: invariant violated — searcher engine is not registered in v1");
  }
  return searcher;
}

}  // namespace alaya::disk

#undef ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED
