// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "index/graph/hnsw/hnsw_segment.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"

namespace {

using RawSpace = alaya::RawSpace<>;
using Sq8Space = alaya::SQ8Space<>;

struct Fbin {
  std::uint32_t rows{};
  std::uint32_t dim{};
  std::vector<float> values;
};

auto read_fbin(const std::filesystem::path &path) -> Fbin {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open memory-graph golden input: " + path.string());
  }
  Fbin result;
  input.read(reinterpret_cast<char *>(&result.rows), sizeof(result.rows));
  input.read(reinterpret_cast<char *>(&result.dim), sizeof(result.dim));
  result.values.resize(static_cast<std::size_t>(result.rows) * result.dim);
  input.read(reinterpret_cast<char *>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("invalid memory-graph golden input: " + path.string());
  }
  return result;
}

template <typename Segment, typename SearchSpace, typename BuildOptions>
void build_and_save(const std::filesystem::path &root,
                    const std::string &engine,
                    const float *vectors,
                    std::uint32_t rows,
                    std::uint32_t dim,
                    std::uint32_t capacity,
                    const BuildOptions &options) {
  std::filesystem::create_directories(root);
  auto build_space = std::make_shared<RawSpace>(capacity, dim, alaya::core::Metric::l2);
  build_space->fit(vectors, rows);

  std::shared_ptr<SearchSpace> search_space;
  if constexpr (std::is_same_v<SearchSpace, RawSpace>) {
    search_space = build_space;
  } else {
    search_space = std::make_shared<SearchSpace>(capacity, dim, alaya::core::Metric::l2);
    search_space->fit(vectors, rows);
  }

  alaya::core::BuildContext context;
  auto segment =
      Segment::build(typename Segment::BuildInput(alaya::core::TypedTensorView::contiguous(vectors,
                                                                                           rows,
                                                                                           dim),
                                                  search_space,
                                                  build_space),
                     options,
                     context);

  const auto graph_path = (root / (engine + "_l2_8.index")).string();
  const auto data_path = (root / "raw.data").string();
  const auto quant_path =
      std::is_same_v<SearchSpace, RawSpace> ? std::string{} : (root / "sq8.data").string();
  const std::array locations{
      alaya::core::ArtifactLocation(Segment::kGraphArtifactName, graph_path),
      alaya::core::ArtifactLocation(Segment::kDataArtifactName, data_path),
      alaya::core::ArtifactLocation(Segment::kQuantArtifactName, quant_path),
  };
  alaya::core::ArtifactWriter writer{std::span<const alaya::core::ArtifactLocation>(locations)};
  alaya::core::ArtifactManifest manifest;
  const auto status = segment->save(writer, {}, manifest);
  if (!status.ok()) {
    throw std::runtime_error(status.diagnostic());
  }
  if (manifest.schema_version != 1 || manifest.format_version != Segment::kFormatVersion) {
    throw std::runtime_error("unexpected memory-graph artifact manifest");
  }
}

template <typename RawSegment, typename QuantizedSegment, typename BuildOptions>
void build_pair(const std::filesystem::path &output,
                const std::string &engine,
                const float *vectors,
                std::uint32_t rows,
                std::uint32_t dim,
                std::uint32_t capacity,
                const BuildOptions &options) {
  build_and_save<RawSegment, RawSpace>(output / ("memory_" + engine + "_none"),
                                       engine,
                                       vectors,
                                       rows,
                                       dim,
                                       capacity,
                                       options);
  build_and_save<QuantizedSegment, Sq8Space>(output / ("memory_" + engine + "_sq8"),
                                             engine,
                                             vectors,
                                             rows,
                                             dim,
                                             capacity,
                                             options);
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    if (argc != 3) {
      std::cerr << "usage: artifact_memory_graph_generator OUTPUT_DIR VECTORS_80_FBIN\n";
      return 2;
    }
    const std::filesystem::path output(argv[1]);
    const auto input = read_fbin(argv[2]);
    if (input.rows != 80 || input.dim != 8) {
      throw std::runtime_error("memory-graph golden input must be 80x8");
    }

    alaya::HnswBuildOptions hnsw;
    hnsw.max_neighbors = 8;
    hnsw.ef_construction = 32;
    hnsw.thread_count = 1;
    build_pair<alaya::HnswSegment<RawSpace>, alaya::HnswSegment<Sq8Space, RawSpace>>(output,
                                                                                     "hnsw",
                                                                                     input.values
                                                                                         .data(),
                                                                                     64,
                                                                                     input.dim,
                                                                                     80,
                                                                                     hnsw);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "artifact_memory_graph_generator: " << error.what() << '\n';
    return 1;
  }
}
