/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

#include "index/laser/utils/vamana_graph_reader.hpp"
#include "utils/vector_file_reader.hpp"

namespace alaya {
namespace {

auto make_temp_path(const char *label) -> std::filesystem::path {
  auto suffix = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  auto dir = std::filesystem::temp_directory_path() / ("laser_file_readers_test_" + suffix);
  std::filesystem::create_directories(dir);
  return dir / label;
}

void write_fvecs(const std::filesystem::path &path,
                 const std::vector<float> &data,
                 uint32_t num_vectors,
                 uint32_t dim) {
  std::ofstream out(path, std::ios::binary);
  for (uint32_t i = 0; i < num_vectors; ++i) {
    auto d = static_cast<int32_t>(dim);
    out.write(reinterpret_cast<const char *>(&d), sizeof(d));
    out.write(reinterpret_cast<const char *>(data.data() + static_cast<size_t>(i) * dim),
              static_cast<std::streamsize>(dim * sizeof(float)));
  }
}

void write_fbin(const std::filesystem::path &path,
                const std::vector<float> &data,
                uint32_t num_vectors,
                uint32_t dim) {
  std::ofstream out(path, std::ios::binary);
  int32_t hdr[2] = {static_cast<int32_t>(num_vectors), static_cast<int32_t>(dim)};
  out.write(reinterpret_cast<const char *>(hdr), sizeof(hdr));
  out.write(reinterpret_cast<const char *>(data.data()),
            static_cast<std::streamsize>(static_cast<size_t>(num_vectors) * dim * sizeof(float)));
}

void write_vamana(const std::filesystem::path &path,
                  uint32_t max_degree,
                  uint32_t entry_point,
                  const std::vector<std::vector<uint32_t>> &adj) {
  size_t body_size = 0;
  for (const auto &nbrs : adj) {
    body_size += sizeof(uint32_t) + nbrs.size() * sizeof(uint32_t);
  }
  constexpr size_t kHdrSz = sizeof(size_t) + 2 * sizeof(uint32_t) + sizeof(size_t);
  size_t file_size = kHdrSz + body_size;
  size_t frozen = 0;

  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(&file_size), sizeof(file_size));
  out.write(reinterpret_cast<const char *>(&max_degree), sizeof(max_degree));
  out.write(reinterpret_cast<const char *>(&entry_point), sizeof(entry_point));
  out.write(reinterpret_cast<const char *>(&frozen), sizeof(frozen));
  for (const auto &nbrs : adj) {
    auto deg = static_cast<uint32_t>(nbrs.size());
    out.write(reinterpret_cast<const char *>(&deg), sizeof(deg));
    out.write(reinterpret_cast<const char *>(nbrs.data()),
              static_cast<std::streamsize>(nbrs.size() * sizeof(uint32_t)));
  }
}

TEST(FvecsFileReaderTest, OpenAndReadSequential) {
  constexpr uint32_t kNumVectors = 100;
  constexpr uint32_t kDim = 16;
  auto path = make_temp_path("vectors.fvecs");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      data[static_cast<size_t>(i) * kDim + j] = static_cast<float>(i * kDim + j) * 0.1F;
    }
  }
  write_fvecs(path, data, kNumVectors, kDim);

  FvecsFileReader reader;
  reader.open(path.string());
  EXPECT_EQ(reader.dim(), kDim);
  EXPECT_EQ(reader.num_vectors(), kNumVectors);

  std::vector<float> out(static_cast<size_t>(kNumVectors) * kDim);
  reader.read_sequential(0, kNumVectors, out.data());

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], data[i]);
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FvecsFileReaderTest, ReadByIds) {
  constexpr uint32_t kNumVectors = 50;
  constexpr uint32_t kDim = 8;
  auto path = make_temp_path("vectors_by_id.fvecs");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      data[static_cast<size_t>(i) * kDim + j] = static_cast<float>(i * kDim + j) * 0.1F;
    }
  }
  write_fvecs(path, data, kNumVectors, kDim);

  FvecsFileReader reader;
  reader.open(path.string());

  std::vector<uint32_t> ids = {5, 10, 0, 49};
  std::vector<float> out(ids.size() * kDim);
  reader.read_by_ids(ids.data(), static_cast<uint32_t>(ids.size()), out.data());

  for (size_t i = 0; i < ids.size(); ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      auto expected = data[static_cast<size_t>(ids[i]) * kDim + j];
      auto actual = out[i * kDim + j];
      EXPECT_FLOAT_EQ(actual, expected);
    }
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FvecsFileReaderTest, OpenInvalidFile) {
  auto path = make_temp_path("does_not_exist.fvecs");

  FvecsFileReader reader;
  EXPECT_THROW(reader.open(path.string()), std::runtime_error);

  std::filesystem::remove_all(path.parent_path());
}

TEST(FbinFileReaderTest, OpenAndReadSequential) {
  constexpr uint32_t kNumVectors = 100;
  constexpr uint32_t kDim = 16;
  auto path = make_temp_path("vectors.fbin");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      data[static_cast<size_t>(i) * kDim + j] = static_cast<float>(i * kDim + j) * 0.1F;
    }
  }
  write_fbin(path, data, kNumVectors, kDim);

  FbinFileReader reader;
  reader.open(path.string());
  EXPECT_EQ(reader.dim(), kDim);
  EXPECT_EQ(reader.num_vectors(), kNumVectors);

  std::vector<float> out(static_cast<size_t>(kNumVectors) * kDim);
  reader.read_sequential(0, kNumVectors, out.data());

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], data[i]);
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FbinFileReaderTest, ReadByIds) {
  constexpr uint32_t kNumVectors = 50;
  constexpr uint32_t kDim = 8;
  auto path = make_temp_path("vectors_by_id.fbin");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t i = 0; i < kNumVectors; ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      data[static_cast<size_t>(i) * kDim + j] = static_cast<float>(i * kDim + j) * 0.1F;
    }
  }
  write_fbin(path, data, kNumVectors, kDim);

  FbinFileReader reader;
  reader.open(path.string());

  std::vector<uint32_t> ids = {3, 7, 0, 49};
  std::vector<float> out(ids.size() * kDim);
  reader.read_by_ids(ids.data(), static_cast<uint32_t>(ids.size()), out.data());

  for (size_t i = 0; i < ids.size(); ++i) {
    for (uint32_t j = 0; j < kDim; ++j) {
      auto expected = data[static_cast<size_t>(ids[i]) * kDim + j];
      auto actual = out[i * kDim + j];
      EXPECT_FLOAT_EQ(actual, expected);
    }
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FbinFileReaderTest, OpenInvalidFile) {
  auto path = make_temp_path("does_not_exist.fbin");

  FbinFileReader reader;
  EXPECT_THROW(reader.open(path.string()), std::runtime_error);

  std::filesystem::remove_all(path.parent_path());
}

TEST(FloatVectorFileReaderTest, AutoDetectsFvecs) {
  constexpr uint32_t kNumVectors = 32;
  constexpr uint32_t kDim = 12;
  auto path = make_temp_path("auto_reader.fvecs");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t vec_id = 0; vec_id < kNumVectors; ++vec_id) {
    for (uint32_t dim = 0; dim < kDim; ++dim) {
      data[static_cast<size_t>(vec_id) * kDim + dim] = static_cast<float>(vec_id * 10 + dim);
    }
  }
  write_fvecs(path, data, kNumVectors, kDim);

  FloatVectorFileReader reader;
  reader.open(path.string());
  EXPECT_EQ(reader.num_vectors(), kNumVectors);
  EXPECT_EQ(reader.dim(), kDim);

  std::vector<uint32_t> ids{0, 4, 31};
  std::vector<float> out(ids.size() * kDim);
  reader.read_by_ids(ids.data(), static_cast<uint32_t>(ids.size()), out.data());
  for (size_t i = 0; i < ids.size(); ++i) {
    for (uint32_t dim = 0; dim < kDim; ++dim) {
      auto expected = data[static_cast<size_t>(ids[i]) * kDim + dim];
      EXPECT_FLOAT_EQ(out[i * kDim + dim], expected);
    }
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FloatVectorFileReaderTest, AutoDetectsFbin) {
  constexpr uint32_t kNumVectors = 20;
  constexpr uint32_t kDim = 6;
  auto path = make_temp_path("auto_reader.fbin");

  std::vector<float> data(static_cast<size_t>(kNumVectors) * kDim);
  for (uint32_t vec_id = 0; vec_id < kNumVectors; ++vec_id) {
    for (uint32_t dim = 0; dim < kDim; ++dim) {
      data[static_cast<size_t>(vec_id) * kDim + dim] = static_cast<float>(vec_id * 100 + dim);
    }
  }
  write_fbin(path, data, kNumVectors, kDim);

  FloatVectorFileReader reader;
  reader.open(path.string());
  EXPECT_EQ(reader.num_vectors(), kNumVectors);
  EXPECT_EQ(reader.dim(), kDim);

  std::vector<float> out(static_cast<size_t>(kNumVectors) * kDim);
  reader.read_sequential(0, kNumVectors, out.data());
  for (size_t idx = 0; idx < out.size(); ++idx) {
    EXPECT_FLOAT_EQ(out[idx], data[idx]);
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(FloatVectorFileReaderTest, RejectsUnsupportedExtension) {
  auto path = make_temp_path("auto_reader.invalid");
  {
    std::ofstream out(path, std::ios::binary);
    out << "invalid";
  }

  FloatVectorFileReader reader;
  EXPECT_THROW(reader.open(path.string()), std::invalid_argument);

  std::filesystem::remove_all(path.parent_path());
}

TEST(VamanaGraphReaderTest, OpenAndReadMetadata) {
  auto path = make_temp_path("graph.vamana");

  constexpr uint32_t kMaxDegree = 3;
  constexpr uint32_t kEntryPoint = 1;
  std::vector<std::vector<uint32_t>> adj = {
      {1, 2},
      {0, 2, 3},
      {1},
      {1, 2},
  };
  write_vamana(path, kMaxDegree, kEntryPoint, adj);

  VamanaGraphReader reader;
  reader.open(path.string());

  EXPECT_EQ(reader.num_nodes(), 4U);
  EXPECT_EQ(reader.max_degree(), kMaxDegree);
  EXPECT_EQ(reader.entry_point(), kEntryPoint);

  std::filesystem::remove_all(path.parent_path());
}

TEST(VamanaGraphReaderTest, ReadChunk) {
  auto path = make_temp_path("graph_chunk.vamana");

  constexpr uint32_t kMaxDegree = 3;
  constexpr uint32_t kEntryPoint = 1;
  std::vector<std::vector<uint32_t>> adj = {
      {1, 2},
      {0, 2, 3},
      {1},
      {1, 2},
  };
  write_vamana(path, kMaxDegree, kEntryPoint, adj);

  VamanaGraphReader reader;
  reader.open(path.string());

  std::vector<std::vector<uint32_t>> out;
  reader.read_chunk(0, 4, out);

  ASSERT_EQ(out.size(), adj.size());
  for (size_t i = 0; i < adj.size(); ++i) {
    EXPECT_EQ(out[i], adj[i]);
  }

  std::filesystem::remove_all(path.parent_path());
}

TEST(VamanaGraphReaderTest, ComputeInDegrees) {
  auto path = make_temp_path("graph_indeg.vamana");

  constexpr uint32_t kMaxDegree = 3;
  constexpr uint32_t kEntryPoint = 1;
  std::vector<std::vector<uint32_t>> adj = {
      {1, 2},
      {0, 2, 3},
      {1},
      {1, 2},
  };
  write_vamana(path, kMaxDegree, kEntryPoint, adj);

  VamanaGraphReader reader;
  reader.open(path.string());

  auto in_degrees = reader.compute_in_degrees();
  std::vector<uint32_t> expected = {1, 3, 3, 1};

  EXPECT_EQ(in_degrees, expected);

  std::filesystem::remove_all(path.parent_path());
}

TEST(VamanaGraphReaderTest, OpenInvalidFile) {
  auto path = make_temp_path("does_not_exist.vamana");

  VamanaGraphReader reader;
  EXPECT_THROW(reader.open(path.string()), std::runtime_error);

  std::filesystem::remove_all(path.parent_path());
}

}  // namespace
}  // namespace alaya
