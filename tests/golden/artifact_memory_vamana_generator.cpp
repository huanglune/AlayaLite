// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/graph/vamana/vamana_mem_segment.hpp"

namespace {

constexpr std::uint32_t kRows = 128;
constexpr std::uint32_t kDim = 8;

auto vectors() -> std::vector<float> {
  std::vector<float> result(kRows * kDim);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t col = 0; col < kDim; ++col) {
      const auto value = (row * 29U + col * 17U + (row % 11U) * (col + 3U)) % 257U;
      result[row * kDim + col] = static_cast<float>(value) / 257.0F;
    }
  }
  return result;
}

}  // namespace

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::cerr << "usage: artifact_memory_vamana_generator OUTPUT_DIR\n";
    return 2;
  }
  const std::filesystem::path output(argv[1]);
  std::filesystem::create_directories(output);
  const auto graph_path = (output / "graph.index").string();
  const auto data_path = (output / "vectors.fbin").string();
  const auto data = vectors();

  alaya::VamanaMemBuildOptions options;
  options.max_neighbors = 8;
  options.construction_effort = 32;
  options.alpha = 1.2F;
  options.thread_count = 1;
  options.max_candidates = 64;
  options.seed = 424242;
  alaya::core::BuildContext build_context;
  auto segment =
      alaya::VamanaMemSegment::build(alaya::VamanaMemSegment::BuildInput(
                                         alaya::core::TypedTensorView::contiguous(data.data(),
                                                                                  kRows,
                                                                                  kDim)),
                                     options,
                                     build_context);

  const std::array locations{
      alaya::core::ArtifactLocation(alaya::VamanaMemSegment::kGraphArtifactName, graph_path),
      alaya::core::ArtifactLocation(alaya::VamanaMemSegment::kDataArtifactName, data_path),
  };
  alaya::core::ArtifactWriter writer{std::span<const alaya::core::ArtifactLocation>(locations)};
  alaya::core::ArtifactManifest manifest;
  const auto status = segment->save(writer, {}, manifest);
  if (!status.ok()) {
    throw std::runtime_error(status.diagnostic());
  }
  if (manifest.schema_version != 1 ||
      manifest.format_version != alaya::VamanaMemSegment::kFormatVersion ||
      manifest.algorithm_id != alaya::core::algorithm::vamana || manifest.artifacts.size() != 2) {
    throw std::runtime_error("unexpected Vamana-memory artifact manifest");
  }
  return 0;
}
