/**
 * @file vamana_graph_reader.hpp
 * @brief Offset-indexed reader for Vamana graph files with chunked neighbor access.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya {

class VamanaGraphReader {
 public:
  VamanaGraphReader() = default;

  VamanaGraphReader(const VamanaGraphReader &) = delete;
  VamanaGraphReader &operator=(const VamanaGraphReader &) = delete;
  VamanaGraphReader(VamanaGraphReader &&) = default;
  VamanaGraphReader &operator=(VamanaGraphReader &&) = default;

  void open(const std::string &path) {
    path_ = path;
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open vamana: " + path);

    in.read(reinterpret_cast<char *>(&file_size_), sizeof(size_t));
    in.read(reinterpret_cast<char *>(&max_degree_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&entry_point_), sizeof(uint32_t));
    size_t frozen = 0;
    in.read(reinterpret_cast<char *>(&frozen), sizeof(size_t));

    constexpr size_t kHdrSz = sizeof(size_t) + 2 * sizeof(uint32_t) + sizeof(size_t);
    size_t pos = kHdrSz;
    while (pos < file_size_) {
      offsets_.push_back(pos);
      uint32_t deg = 0;
      in.read(reinterpret_cast<char *>(&deg), sizeof(uint32_t));
      degrees_.push_back(deg);
      in.seekg(static_cast<std::streamoff>(deg * sizeof(uint32_t)), std::ios::cur);
      pos += sizeof(uint32_t) + static_cast<size_t>(deg) * sizeof(uint32_t);
    }
  }

  void read_chunk(uint32_t start, uint32_t count, std::vector<std::vector<uint32_t>> &out) const {
    out.resize(count);
    std::ifstream in(path_, std::ios::binary);
    for (uint32_t i = 0; i < count; ++i) {
      in.seekg(static_cast<std::streamoff>(offsets_[start + i] + sizeof(uint32_t)));
      out[i].resize(degrees_[start + i]);
      in.read(reinterpret_cast<char *>(out[i].data()),
              static_cast<std::streamsize>(degrees_[start + i] * sizeof(uint32_t)));
    }
  }

  [[nodiscard]] auto compute_in_degrees() const -> std::vector<uint32_t> {
    auto n = static_cast<uint32_t>(offsets_.size());
    std::vector<uint32_t> in_deg(n, 0);
    std::ifstream in(path_, std::ios::binary);
    for (uint32_t nid = 0; nid < n; ++nid) {
      in.seekg(static_cast<std::streamoff>(offsets_[nid] + sizeof(uint32_t)));
      std::vector<uint32_t> nbrs(degrees_[nid]);
      in.read(reinterpret_cast<char *>(nbrs.data()),
              static_cast<std::streamsize>(degrees_[nid] * sizeof(uint32_t)));
      for (auto id : nbrs) in_deg[id]++;
    }
    return in_deg;
  }

  [[nodiscard]] auto num_nodes() const -> uint32_t {
    return static_cast<uint32_t>(offsets_.size());
  }
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }
  [[nodiscard]] auto entry_point() const -> uint32_t { return entry_point_; }

 private:
  std::string path_;
  size_t file_size_{0};
  uint32_t max_degree_{0};
  uint32_t entry_point_{0};
  std::vector<size_t> offsets_;
  std::vector<uint32_t> degrees_;
};

}  // namespace alaya
