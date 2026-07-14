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

#include "index/disk/detail/disk_collection_v1.hpp"
#include "index/graph/fusion/fusion_segment.hpp"
#include "index/graph/hnsw/hnsw_segment.hpp"
#include "index/graph/nsg/nsg_segment.hpp"
#include "space/raw_space.hpp"
#include "space/sq8_space.hpp"
#include "storage/container/sequential_storage.hpp"

namespace {

using RawSpace = alaya::RawSpace<float,
                                 float,
                                 std::uint32_t,
                                 alaya::SequentialStorage<float, std::uint32_t>>;
using Sq8Space = alaya::SQ8Space<float,
                                 float,
                                 std::uint32_t,
                                 alaya::SequentialStorage<std::uint8_t, std::uint32_t>,
                                 alaya::EmptyScalarData>;

struct Fbin {
  std::uint32_t rows{};
  std::uint32_t dim{};
  std::vector<float> values;
};

auto read_fbin(const std::filesystem::path &path) -> Fbin {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open retained-v1 input: " + path.string());
  }
  Fbin result;
  input.read(reinterpret_cast<char *>(&result.rows), sizeof(result.rows));
  input.read(reinterpret_cast<char *>(&result.dim), sizeof(result.dim));
  result.values.resize(static_cast<std::size_t>(result.rows) * result.dim);
  input.read(reinterpret_cast<char *>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!input || input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("invalid retained-v1 input: " + path.string());
  }
  return result;
}

template <typename Segment, typename SearchSpace, typename BuildOptions>
void build_memory_artifact(const std::filesystem::path &root,
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
}

template <typename Segment, typename Options>
void build_quantization_pair(const std::filesystem::path &output,
                             const std::string &engine,
                             const float *vectors,
                             std::uint32_t rows,
                             std::uint32_t dim,
                             std::uint32_t capacity,
                             const Options &options) {
  build_memory_artifact<Segment, RawSpace>(output / ("memory_" + engine + "_none"),
                                           engine,
                                           vectors,
                                           rows,
                                           dim,
                                           capacity,
                                           options);
  using QuantizedSegment = std::conditional_t<
      std::is_same_v<Segment, alaya::HnswSegment<RawSpace, RawSpace>>,
      alaya::HnswSegment<Sq8Space, RawSpace>,
      std::conditional_t<std::is_same_v<Segment, alaya::NsgSegment<RawSpace, RawSpace>>,
                         alaya::NsgSegment<Sq8Space, RawSpace>,
                         alaya::FusionSegment<Sq8Space, RawSpace>>>;
  build_memory_artifact<QuantizedSegment, Sq8Space>(output / ("memory_" + engine + "_sq8"),
                                                    engine,
                                                    vectors,
                                                    rows,
                                                    dim,
                                                    capacity,
                                                    options);
}

void build_disk_artifacts(const std::filesystem::path &output, const Fbin &input) {
  constexpr std::uint64_t kRows = 64;
  std::array<std::uint64_t, kRows> labels{};
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = 1000 + row;
  }
  {
    alaya::disk::DiskCollection collection(output / "disk_flat",
                                           input.dim,
                                           alaya::core::Metric::l2,
                                           alaya::disk::DiskIndexType::Flat);
    collection.add_batch(input.values.data(), labels.data(), kRows);
    collection.flush();
  }
  {
    alaya::disk::VamanaSegmentBuildParams options;
    options.R = 8;
    options.L = 24;
    options.alpha = 1.2F;
    options.seed = 424242;
    options.num_threads = 1;
    alaya::disk::DiskCollection collection(output / "disk_vamana",
                                           input.dim,
                                           alaya::core::Metric::l2,
                                           alaya::disk::DiskIndexType::Vamana,
                                           alaya::disk::DiskCollection::kDefaultMaxPendingBytes,
                                           options);
    collection.add_batch(input.values.data(), labels.data(), kRows);
    collection.flush();
  }
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    if (argc != 3) {
      std::cerr << "usage: artifact_legacy_v1_generator OUTPUT_DIR VECTORS_80_FBIN\n";
      return 2;
    }
    const std::filesystem::path output(argv[1]);
    const auto input = read_fbin(argv[2]);
    if (input.rows != 80 || input.dim != 8) {
      throw std::runtime_error("retained-v1 golden input must be 80x8");
    }

    alaya::HnswBuildOptions hnsw;
    hnsw.max_neighbors = 8;
    hnsw.ef_construction = 32;
    hnsw.thread_count = 1;
    build_quantization_pair<alaya::HnswSegment<RawSpace, RawSpace>>(output,
                                                                    "hnsw",
                                                                    input.values.data(),
                                                                    64,
                                                                    input.dim,
                                                                    80,
                                                                    hnsw);

    alaya::NsgBuildOptions nsg;
    nsg.max_neighbors = 8;
    nsg.ef_construction = 32;
    nsg.thread_count = 1;
    build_quantization_pair<alaya::NsgSegment<RawSpace, RawSpace>>(output,
                                                                   "nsg",
                                                                   input.values.data(),
                                                                   input.rows,
                                                                   input.dim,
                                                                   96,
                                                                   nsg);

    alaya::FusionBuildOptions fusion;
    fusion.max_neighbors = 8;
    fusion.ef_construction = 32;
    fusion.thread_count = 1;
    build_quantization_pair<alaya::FusionSegment<RawSpace, RawSpace>>(output,
                                                                      "fusion",
                                                                      input.values.data(),
                                                                      input.rows,
                                                                      input.dim,
                                                                      96,
                                                                      fusion);
    build_disk_artifacts(output, input);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "artifact_legacy_v1_generator: " << error.what() << '\n';
    return 1;
  }
}
