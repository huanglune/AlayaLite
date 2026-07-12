// SPDX-License-Identifier: AGPL-3.0-only
#include "index/disk/disk_collection.hpp"

#include <type_traits>

static_assert(std::is_class_v<alaya::disk::DiskCollection>);
static_assert(std::is_abstract_v<alaya::disk::SegmentSearcher>);

auto golden_disk_compile() -> alaya::disk::DiskIndexType {
  return alaya::disk::DiskIndexType::Flat;
}
