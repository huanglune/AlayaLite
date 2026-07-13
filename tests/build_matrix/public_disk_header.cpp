// SPDX-License-Identifier: AGPL-3.0-only
#include "index/disk/segment_factory.hpp"

#include <type_traits>

static_assert(std::is_abstract_v<alaya::disk::SegmentSearcher>);

auto golden_disk_compile() -> alaya::disk::DiskIndexType {
  return alaya::disk::DiskIndexType::Flat;
}
