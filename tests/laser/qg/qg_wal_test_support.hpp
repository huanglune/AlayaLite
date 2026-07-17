// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Shared fixture for the op-WAL functional + crash tests: builds a tiny LASER
// v2 quantized-graph index (ring Vamana -> QGBuilder -> non-WAL migrate +
// checkpoint) so a QGUpdater with enable_wal=true can open it as a clean base.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"

namespace alaya::laser::waltest {

inline constexpr size_t kDim = 64;
inline constexpr size_t kDeg = 64;  // R64/d64: node_per_page == 2, with trailer slack.

inline std::vector<float> make_data(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (auto &v : data) v = dist(gen);
  return data;
}

inline void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * n * dim));
}

inline void write_ring_vamana(const std::string &path, size_t n, uint32_t degree) {
  const size_t header_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
  const size_t record_size = sizeof(uint32_t) * (static_cast<size_t>(degree) + 1);
  const size_t expected_file_size = header_size + n * record_size;
  const uint32_t start = 0;
  const size_t frozen_points = 0;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(expected_file_size));
  out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
  out.write(reinterpret_cast<const char *>(&start), sizeof(start));
  out.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
  for (uint32_t i = 0; i < n; ++i) {
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    for (uint32_t j = 0; j < degree; ++j) {
      const uint32_t neighbor = (i + j + 1) % static_cast<uint32_t>(n);
      out.write(reinterpret_cast<const char *>(&neighbor), sizeof(neighbor));
    }
  }
}

inline std::string index_suffix() {
  return "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
}

// Builds a v2 index rooted at `<dir>/<name>` with `base_n` rows and returns its
// prefix. After this call the .index carries a clean v2 A/B superblock, so a
// QGUpdater with enable_wal=true opens it as a fresh lineage.
struct WalTinyIndex {
  std::filesystem::path dir;
  std::string prefix;
  std::vector<float> data;
  size_t base_n;

  static WalTinyIndex build(const std::filesystem::path &dir, size_t base_n, uint32_t seed) {
    WalTinyIndex t;
    t.dir = dir;
    t.base_n = base_n;
    std::filesystem::create_directories(dir);
    t.prefix = (dir / "wal_base").string();
    t.data = make_data(base_n, kDim, seed);

    const std::string vamana_path = t.prefix + "_vamana.index";
    write_fbin(t.prefix + "_pca_base.fbin", t.data.data(), static_cast<int32_t>(base_n),
               static_cast<int32_t>(kDim));
    write_ring_vamana(vamana_path, base_n, kDeg);
    {
      QuantizedGraph qg(base_n, kDeg, kDim, kDim, /*rotator_seed=*/7);
      QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/1);
      builder.build(vamana_path.c_str(), t.prefix.c_str());
    }
    // Migrate v1 -> v2 and checkpoint a clean base WITHOUT the WAL.
    {
      QuantizedGraph qg(base_n, kDeg, kDim, kDim);
      qg.load_disk_index(t.prefix.c_str(), 0.0F);
      qg.set_params(64, 1, 1);
      QGUpdater upd(qg, UpdateParams{});
      upd.checkpoint();
    }
    return t;
  }
};

}  // namespace alaya::laser::waltest
