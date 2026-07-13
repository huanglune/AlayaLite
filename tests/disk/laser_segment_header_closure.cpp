// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/laser_segment.hpp"

int main() {
  static_assert(!alaya::core::Saveable<alaya::disk::LaserSegment>);
  alaya::core::OpenContext context;
  const auto opened = alaya::disk::LaserSegmentFactory::open(alaya::core::ArtifactView{},
                                                             alaya::core::OpenOptions{},
                                                             context);
  return !opened.ok() && opened.status().code() == alaya::core::StatusCode::not_supported ? 0 : 1;
}
