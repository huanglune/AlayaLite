// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/disk_flat_segment.hpp"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::cerr << "usage: artifact_disk_flat_segment_generator OUTPUT_DIR\n";
    return 2;
  }
  constexpr std::uint64_t kRows = 64;
  constexpr std::uint32_t kDim = 8;
  constexpr std::uint64_t kSeed = 20260712;
  std::vector<float> vectors(kRows * kDim);
  std::vector<std::uint64_t> labels(kRows);
  std::uint64_t state = kSeed;
  auto next = [&] {
    state ^= state >> 12U;
    state ^= state << 25U;
    state ^= state >> 27U;
    return state * 0x2545F4914F6CDD1DULL;
  };
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 2000 + row;
    for (std::uint32_t col = 0; col < kDim; ++col) {
      const auto value = static_cast<std::int32_t>(next() >> 40U);
      vectors[row * kDim + col] = static_cast<float>(value) / 8'388'608.0F;
    }
  }

  alaya::disk::DiskFlatBuildInput input(alaya::core::TypedTensorView::contiguous(vectors.data(),
                                                                                 kRows,
                                                                                 kDim),
                                        labels);
  alaya::disk::DiskFlatPublicationOptions options;
  options.collection_root = argv[1];
  options.segment_id = "seg_00000001";
  options.segment_generation = 1;
  options.manifest_generation = 1;
  options.collection_features.manifest_v2_writer = true;
  alaya::core::BuildContext context;
  auto built =
      alaya::disk::DiskFlatSegmentFactory::build(input, alaya::core::Metric::l2, options, context);
  if (!built.ok()) {
    std::cerr << built.status().diagnostic() << '\n';
    return 1;
  }
  return 0;
}
