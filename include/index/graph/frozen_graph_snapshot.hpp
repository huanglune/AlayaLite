// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace alaya {

// Owning, move-only snapshot of a finalized graph topology. The snapshot has
// no mutable adjacency accessor so downstream seal paths all observe the same
// graph that the source builder produced.
class FrozenGraphSnapshot {
 public:
  using Adjacency = std::vector<std::vector<std::uint32_t>>;

  FrozenGraphSnapshot(Adjacency adjacency,
                      std::uint32_t entry_point,
                      std::uint32_t max_degree,
                      std::uint64_t frozen_pts = 0)
      : adjacency_(std::move(adjacency)),
        entry_point_(entry_point),
        max_degree_(max_degree),
        frozen_pts_(frozen_pts) {}

  // Moving the outer adjacency vector transfers all per-node edge buffers in
  // constant time; no edge list is copied.
  explicit FrozenGraphSnapshot(vamana::VamanaBuilder &&builder, std::uint64_t frozen_pts = 0)
      : entry_point_(builder.medoid()), max_degree_(builder.max_degree()), frozen_pts_(frozen_pts) {
    adjacency_ = std::move(builder).release_graph();
  }

  FrozenGraphSnapshot(const FrozenGraphSnapshot &) = delete;
  auto operator=(const FrozenGraphSnapshot &) -> FrozenGraphSnapshot & = delete;
  FrozenGraphSnapshot(FrozenGraphSnapshot &&) noexcept = default;
  auto operator=(FrozenGraphSnapshot &&) noexcept -> FrozenGraphSnapshot & = default;

  [[nodiscard]] static auto from_vamana(vamana::VamanaBuilder &&builder,
                                        std::uint64_t frozen_pts = 0) -> FrozenGraphSnapshot {
    return FrozenGraphSnapshot(std::move(builder), frozen_pts);
  }

  [[nodiscard]] auto adjacency() const noexcept -> const Adjacency & { return adjacency_; }
  [[nodiscard]] auto graph() const noexcept -> const Adjacency & { return adjacency_; }
  [[nodiscard]] auto entry_point() const noexcept -> std::uint32_t { return entry_point_; }
  [[nodiscard]] auto medoid() const noexcept -> std::uint32_t { return entry_point_; }
  [[nodiscard]] auto num_points() const noexcept -> std::size_t { return adjacency_.size(); }
  [[nodiscard]] auto max_degree() const noexcept -> std::uint32_t { return max_degree_; }
  [[nodiscard]] auto frozen_pts() const noexcept -> std::uint64_t { return frozen_pts_; }

  // Validate the topology invariants shared by all seal consumers.
  void validate() const {
    if (adjacency_.empty()) {
      throw std::invalid_argument("FrozenGraphSnapshot: graph must contain at least one node");
    }
    if (static_cast<std::size_t>(entry_point_) >= adjacency_.size()) {
      throw std::invalid_argument("FrozenGraphSnapshot: entry point " +
                                  std::to_string(entry_point_) +
                                  " is out of range for N=" + std::to_string(adjacency_.size()));
    }

    for (std::size_t node = 0; node < adjacency_.size(); ++node) {
      const auto &neighbors = adjacency_[node];
      if (neighbors.size() > max_degree_) {
        throw std::invalid_argument("FrozenGraphSnapshot: node " + std::to_string(node) +
                                    " has degree " + std::to_string(neighbors.size()) +
                                    " exceeding max_degree=" + std::to_string(max_degree_));
      }
      for (const std::uint32_t neighbor : neighbors) {
        if (static_cast<std::size_t>(neighbor) >= adjacency_.size()) {
          throw std::invalid_argument("FrozenGraphSnapshot: node " + std::to_string(node) +
                                      " has out-of-range neighbor " + std::to_string(neighbor) +
                                      " for N=" + std::to_string(adjacency_.size()));
        }
        if (static_cast<std::size_t>(neighbor) == node) {
          throw std::invalid_argument("FrozenGraphSnapshot: node " + std::to_string(node) +
                                      " has a self-loop");
        }
      }
    }
  }

  void save(const std::filesystem::path &path) const {
    validate();
    vamana::save_graph(adjacency_, path, max_degree_, entry_point_, frozen_pts_);
  }

  [[nodiscard]] static auto load(const std::filesystem::path &path) -> FrozenGraphSnapshot {
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("FrozenGraphSnapshot::load: not a regular file: " + path.string());
    }

    constexpr std::uint64_t kHeaderSize = 24;
    const auto actual_size = static_cast<std::uint64_t>(std::filesystem::file_size(path));
    if (actual_size < kHeaderSize) {
      throw std::runtime_error("FrozenGraphSnapshot::load: truncated header in " + path.string());
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw std::runtime_error("FrozenGraphSnapshot::load: cannot open " + path.string());
    }

    std::uint64_t expected_size = 0;
    std::uint32_t max_degree = 0;
    std::uint32_t entry_point = 0;
    std::uint64_t frozen_pts = 0;
    read_exact(input, expected_size, "expected_file_size", path);
    read_exact(input, max_degree, "max_degree", path);
    read_exact(input, entry_point, "entry_point", path);
    read_exact(input, frozen_pts, "frozen_pts", path);
    if (expected_size != actual_size) {
      throw std::runtime_error("FrozenGraphSnapshot::load: declared size " +
                               std::to_string(expected_size) + " differs from actual size " +
                               std::to_string(actual_size) + " in " + path.string());
    }

    const std::uint64_t records_bytes = expected_size - kHeaderSize;
    std::uint64_t consumed = 0;
    Adjacency adjacency;
    while (consumed < records_bytes) {
      if (records_bytes - consumed < sizeof(std::uint32_t)) {
        throw std::runtime_error("FrozenGraphSnapshot::load: truncated degree record in " +
                                 path.string());
      }
      std::uint32_t degree = 0;
      read_exact(input, degree, "node degree", path);
      consumed += sizeof(std::uint32_t);
      if (degree > max_degree) {
        throw std::runtime_error(
            "FrozenGraphSnapshot::load: node " + std::to_string(adjacency.size()) + " degree " +
            std::to_string(degree) + " exceeds max_degree=" + std::to_string(max_degree));
      }

      const auto neighbor_bytes = static_cast<std::uint64_t>(degree) * sizeof(std::uint32_t);
      if (neighbor_bytes > records_bytes - consumed) {
        throw std::runtime_error("FrozenGraphSnapshot::load: truncated neighbors for node " +
                                 std::to_string(adjacency.size()) + " in " + path.string());
      }
      std::vector<std::uint32_t> neighbors(degree);
      if (degree != 0) {
        input.read(reinterpret_cast<char *>(neighbors.data()),
                   static_cast<std::streamsize>(neighbor_bytes));
        if (!input) {
          throw std::runtime_error("FrozenGraphSnapshot::load: failed reading neighbors in " +
                                   path.string());
        }
      }
      consumed += neighbor_bytes;
      adjacency.push_back(std::move(neighbors));
    }

    FrozenGraphSnapshot snapshot(std::move(adjacency), entry_point, max_degree, frozen_pts);
    try {
      snapshot.validate();
    } catch (const std::invalid_argument &error) {
      throw std::runtime_error("FrozenGraphSnapshot::load: " + std::string(error.what()) +
                               " (path: " + path.string() + ")");
    }
    return snapshot;
  }

 private:
  template <typename T>
  static void read_exact(std::ifstream &input,
                         T &value,
                         const char *field,
                         const std::filesystem::path &path) {
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!input) {
      throw std::runtime_error("FrozenGraphSnapshot::load: failed reading " + std::string(field) +
                               " from " + path.string());
    }
  }

  Adjacency adjacency_;
  std::uint32_t entry_point_ = 0;
  std::uint32_t max_degree_ = 0;
  std::uint64_t frozen_pts_ = 0;
};

}  // namespace alaya
