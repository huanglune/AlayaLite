// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "index/disk/laser_segment.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "space/rabitq_space.hpp"
#include "core/value_types.hpp"
#include "platform/detect.hpp"

#ifndef ALAYA_LASER_FIXTURE_DIR
  #define ALAYA_LASER_FIXTURE_DIR ""
#endif

#ifndef ALAYA_LASER_FIXTURE_PREFIX
  #define ALAYA_LASER_FIXTURE_PREFIX "dsqg_seg_00000001"
#endif

namespace alaya::disk {
namespace {

using MemorySpace = RaBitQSpace<>;
using MemorySegment = QgSegment<MemorySpace>;

constexpr std::uint32_t kRows = 128;
constexpr std::uint32_t kCapacity = 144;
constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kDegree = 64;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-rabitq-format-separation-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_;
};

auto make_vectors() -> std::vector<float> {
  std::vector<float> vectors(static_cast<std::size_t>(kRows) * kDim);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto value = (row * 37U + column * 19U + (row % 7U) * column) % 251U;
      vectors[static_cast<std::size_t>(row) * kDim + column] = static_cast<float>(value) / 251.0F;
    }
  }
  return vectors;
}

void build_memory_qg(const std::filesystem::path &artifact) {
  const auto vectors = make_vectors();
  auto space = std::make_shared<MemorySpace>(kCapacity, kDim, core::Metric::l2);
  space->fit(vectors.data(), kRows);
  core::BuildContext build_context;
  QgBuildOptions build_options;
  build_options.ef_build = 64;
  build_options.thread_count = 1;
  auto segment =
      MemorySegment::build({core::TypedTensorView::contiguous(vectors.data(), kRows, kDim), space},
                           build_options,
                           build_context);
  const auto artifact_string = artifact.string();
  const std::array locations{core::ArtifactLocation(MemorySegment::kArtifactName, artifact_string)};
  core::ArtifactWriter writer{std::span<const core::ArtifactLocation>(locations)};
  core::ArtifactManifest manifest;
  const auto status = segment->save(writer, {}, manifest);
  if (!status.ok()) {
    throw std::runtime_error(status.diagnostic());
  }
}

void write_nonempty(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary);
  constexpr char byte = '\0';
  output.write(&byte, 1);
  if (!output) {
    throw std::runtime_error("cannot write cross-format test sidecar");
  }
}

void wrap_memory_qg_as_laser_segment(const std::filesystem::path &memory_artifact,
                                     const std::filesystem::path &segment_directory) {
  std::filesystem::create_directories(segment_directory);
  const std::string prefix{"dsqg_seg_00000001"};
  const std::string index_name =
      prefix + "_R" + std::to_string(kDegree) + "_MD" + std::to_string(kDim) + ".index";
  std::filesystem::copy_file(memory_artifact, segment_directory / index_name);
  write_nonempty(segment_directory / (index_name + "_rotator"));
  write_nonempty(segment_directory / (index_name + "_cache_ids"));
  write_nonempty(segment_directory / (index_name + "_cache_nodes"));

  std::vector<std::uint64_t> labels(kRows);
  for (std::uint64_t row = 0; row < kRows; ++row) {
    labels[row] = row;
  }
  std::ofstream ids(segment_directory / "ids.u64.bin", std::ios::binary);
  ids.write(reinterpret_cast<const char *>(labels.data()),
            static_cast<std::streamsize>(labels.size() * sizeof(labels.front())));
  if (!ids) {
    throw std::runtime_error("cannot write cross-format test ids");
  }

  SegmentManifest manifest;
  manifest.segment_id = "seg_00000001";
  manifest.index_type = DiskIndexType::Laser;
  manifest.metric = core::Metric::l2;
  manifest.dim = kDim;
  manifest.count = kRows;
  manifest.ids_file = "ids.u64.bin";
  manifest.vectors_file.clear();
  manifest.x_extras["x_laser_filename_prefix"] = prefix;
  manifest.x_extras["x_R"] = std::to_string(kDegree);
  manifest.x_extras["x_main_dim"] = std::to_string(kDim);
  manifest.x_extras["x_laser_index_file"] = index_name;
  manifest.x_extras["x_laser_rotator_file"] = index_name + "_rotator";
  manifest.x_extras["x_laser_cache_ids_file"] = index_name + "_cache_ids";
  manifest.x_extras["x_laser_cache_nodes_file"] = index_name + "_cache_nodes";
  manifest.save(segment_directory / "manifest.txt");
}

auto fixture_index() -> std::filesystem::path {
  const std::string prefix(ALAYA_LASER_FIXTURE_PREFIX);
  return std::filesystem::path(ALAYA_LASER_FIXTURE_DIR) / (prefix + "_R64_MD128.index");
}

TEST(RaBitQFormatSeparation, MemoryWireFormatIsRejectedByLaserOpen) {
  TemporaryDirectory temporary;
  const auto memory_artifact = temporary.path() / "memory.qg";
  const auto laser_directory = temporary.path() / "seg_00000001";
  build_memory_qg(memory_artifact);
  wrap_memory_qg_as_laser_segment(memory_artifact, laser_directory);

  core::OpenContext context;
  const auto opened = LaserSegment::open_directory(laser_directory, {}, context);
  ASSERT_FALSE(opened.ok());
  EXPECT_EQ(opened.status().code(), core::StatusCode::io_error);
  EXPECT_NE(opened.status().diagnostic().find("disagrees with LASER index metadata count"),
            std::string::npos)
      << opened.status().diagnostic();
}

TEST(RaBitQFormatSeparation, LaserWireFormatIsRejectedByMemoryOpen) {
  const auto laser_artifact = fixture_index();
  if (!std::filesystem::is_regular_file(laser_artifact)) {
    GTEST_SKIP() << "LASER fixture is unavailable: " << laser_artifact;
  }
  const auto artifact_string = laser_artifact.string();
  const std::array locations{core::ArtifactLocation(MemorySegment::kArtifactName, artifact_string)};
  core::OpenContext context;
  try {
    (void)MemorySegment::open(core::ArtifactView(locations), {}, context);
    FAIL() << "LASER wire format was accepted as memory RaBitQ";
  } catch (const std::invalid_argument &error) {
    EXPECT_NE(std::string(error.what()).find("artifact is not memory RaBitQ format"),
              std::string::npos)
        << error.what();
  }
}

}  // namespace
}  // namespace alaya::disk
