// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/diskann_segment.hpp"

int main() {
  static_assert(alaya::disk::DiskAnnSegment::kAlgorithmId == alaya::core::algorithm::diskann);
  return 0;
}
