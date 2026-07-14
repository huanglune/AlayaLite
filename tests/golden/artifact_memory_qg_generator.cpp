// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "index/neighbor.hpp"
#include "space/rabitq_space.hpp"
#include "core/value_types.hpp"
#include "utils/rabitq_utils/rotator.hpp"

namespace {

using Space = alaya::RaBitQSpace<>;

constexpr std::uint32_t kRows = 48;
constexpr std::uint32_t kDim = 64;
constexpr std::uint32_t kSeed = 20260712;

auto vectors() -> std::vector<float> {
  std::vector<float> result(kRows * kDim);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t col = 0; col < kDim; ++col) {
      const auto value = (row * 37U + col * 19U + (row % 7U) * col) % 251U;
      result[row * kDim + col] = static_cast<float>(value) / 251.0F;
    }
  }
  return result;
}

void install_fixed_graph(const std::shared_ptr<Space> &space) {
  for (std::uint32_t row = 0; row < kRows; ++row) {
    std::vector<alaya::Neighbor<std::uint32_t, float>> neighbors;
    neighbors.reserve(Space::kDegreeBound);
    for (std::uint32_t offset = 1; offset <= Space::kDegreeBound; ++offset) {
      const auto neighbor = (row + offset) % kRows;
      neighbors.emplace_back(neighbor, space->get_distance(row, neighbor));
    }
    space->update_nei(row, neighbors);
  }
  space->set_ep(0);
}

void overwrite_rotator_with_seed(std::string_view artifact_path) {
  constexpr std::streamoff kRotatorOffset = sizeof(alaya::core::Metric) + sizeof(std::uint32_t) +
                                            sizeof(std::uint32_t) + sizeof(std::uint32_t) +
                                            sizeof(alaya::RotatorType) + sizeof(std::uint32_t);
  constexpr std::size_t kFlipBytes = 4 * kDim / 8;
  std::array<std::uint8_t, kFlipBytes> flips{};
  std::mt19937 generator(kSeed);
  for (auto &value : flips) {
    value = static_cast<std::uint8_t>(generator() & 0xffU);
  }

  std::fstream artifact(std::string(artifact_path),
                        std::ios::binary | std::ios::in | std::ios::out);
  if (!artifact) {
    throw std::runtime_error("cannot reopen QG seed artifact");
  }
  artifact.seekp(kRotatorOffset);
  artifact.write(reinterpret_cast<const char *>(flips.data()),
                 static_cast<std::streamsize>(flips.size()));
  if (!artifact) {
    throw std::runtime_error("cannot write deterministic QG rotator");
  }
}

}  // namespace

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::cerr << "usage: artifact_memory_qg_generator OUTPUT_DIR\n";
    return 2;
  }
  const std::filesystem::path output(argv[1]);
  std::filesystem::create_directories(output);
  const auto seed_artifact = output / "seed.tmp";
  const auto final_artifact = output / "rabitq.data";
  const auto data = vectors();

  // RaBitQ's retained default constructor intentionally uses random_device.
  // Write one valid space, replace only its serialized FhtKac flip bytes with
  // a fixed seed, reload that v1 artifact, then refill every persisted byte.
  auto seed_space = std::make_shared<Space>(kRows, kDim, alaya::core::Metric::l2);
  seed_space->fit(data.data(), kRows);
  install_fixed_graph(seed_space);
  seed_space->save(seed_artifact.string());
  overwrite_rotator_with_seed(seed_artifact.string());

  auto deterministic = std::make_shared<Space>();
  deterministic->load(seed_artifact.string());
  deterministic->fit(data.data(), kRows);
  install_fixed_graph(deterministic);
  deterministic->save(final_artifact.string());
  std::filesystem::remove(seed_artifact);
  return 0;
}
