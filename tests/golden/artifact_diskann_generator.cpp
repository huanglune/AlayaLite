// SPDX-License-Identifier: AGPL-3.0-only
#include "index/graph/diskann/diskann_index.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::cerr << "usage: artifact_diskann_generator OUTPUT_DIR\n";
    return 2;
  }
  constexpr uint64_t kRows = 64;
  constexpr uint64_t kDim = 8;
  std::vector<float> vectors(kRows * kDim);
  std::vector<uint64_t> labels(kRows);
  for (uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 1000 + row;
    for (uint64_t col = 0; col < kDim; ++col) {
      vectors[row * kDim + col] =
          std::sin(static_cast<float>((row + 1) * (col + 3)) * 0.125F);
    }
  }
  alaya::diskann::DiskANNBuildParams params;
  params.R = 8;
  params.L = 24;
  params.alpha = 1.2F;
  params.cache_ratio = 0.125;
  params.num_threads = 1;
  params.seed = 424242;
  alaya::diskann::DiskANNIndex::build(argv[1], vectors.data(), labels.data(), kRows, kDim,
                                      params);
  return 0;
}
