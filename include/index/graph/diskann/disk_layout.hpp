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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

#include "utils/math.hpp"
#include "utils/memory.hpp"

namespace alaya {

// ============================================================================
// Constants for DiskANN disk layout
// ============================================================================

constexpr size_t kDiskSectorSize = 4096;              ///< 4KB sector size for Direct IO alignment
constexpr uint32_t kDiskANNMagicNumber = 0x444B414E;  ///< "DKAN" in hex (DiskANN magic)
constexpr uint32_t kDiskANNVersion = 2;               ///< Current version (2 = PQ support)

// ============================================================================
// DiskIndexHeader - Metadata for the DiskANN index file
// ============================================================================

/**
 * @brief Header structure for DiskANN index file.
 *
 * This structure is stored at the beginning of the index file and contains
 * all necessary metadata to interpret the rest of the file. The header is
 * padded to be exactly 4KB (one sector) for Direct IO compatibility.
 *
 * File Layout (without PQ):
 * +------------------+
 * | DiskIndexHeader  | <- 4KB (sector 0)
 * +------------------+
 * | DiskNode[0]      | <- 4KB * nodes_per_sector (sector 1+)
 * | DiskNode[1]      |
 * | ...              |
 * | DiskNode[n-1]    |
 * +------------------+
 *
 * File Layout (with PQ enabled):
 * +------------------+
 * | DiskIndexHeader  | <- 4KB (sector 0)
 * +------------------+
 * | PQ Codebook      | <- M × 256 × (D/M) × sizeof(float), aligned to 4KB
 * +------------------+
 * | PQ Codes         | <- N × M bytes, aligned to 4KB
 * +------------------+
 * | DiskNode[0]      | <- Full vectors + graph for reranking
 * | DiskNode[1]      |
 * | ...              |
 * +------------------+
 */
struct alignas(kDiskSectorSize) DiskIndexHeader {
  /// Metadata fields for the index header (add new fields here freely)
  struct Metadata {
    // Magic number and version for file validation
    uint32_t magic_;    ///< Magic number to identify file format (kDiskANNMagicNumber)
    uint32_t version_;  ///< Version of the disk format

    // Core parameters from DiskANN algorithm
    uint32_t max_degree_;  ///< R: Maximum out-degree of each node in the graph
    float alpha_;          ///< α: Distance threshold multiplier for pruning (typically 1.2)

    // Data dimensions
    uint32_t dimension_;   ///< Dimension of the vector data
    uint64_t num_points_;  ///< Total number of points (vectors) in the index

    // Disk layout information
    uint64_t
        node_sector_size_;  ///< Size of each node in bytes (must be multiple of kDiskSectorSize)
    uint64_t data_offset_;  ///< Byte offset where node data begins (after header)

    // Entry point for search
    uint64_t medoid_id_;  ///< ID of the medoid (entry point for search)

    // Additional metadata
    uint64_t index_build_time_;  ///< Unix timestamp when index was built
    uint32_t data_type_;         ///< Data type identifier (0=float, 1=int8, 2=uint8)

    // PQ (Product Quantization) parameters for in-memory navigation
    uint32_t pq_enabled_;          ///< Whether PQ is enabled (0=no, 1=yes)
    uint32_t pq_num_subspaces_;    ///< M: Number of PQ subspaces
    uint64_t pq_codebook_offset_;  ///< Byte offset to PQ codebook section
    uint64_t pq_codebook_size_;    ///< Size of PQ codebook in bytes
    uint64_t pq_codes_offset_;     ///< Byte offset to PQ codes section
    uint64_t pq_codes_size_;       ///< Size of PQ codes section in bytes
    uint64_t nodes_offset_;        ///< Byte offset to DiskNode array (for reranking)

    // Add new fields here - no need to manually adjust padding
  };

  /// Union to ensure the header is exactly 4KB
  union {
    Metadata meta_;                     ///< Actual metadata fields
    uint8_t padding_[kDiskSectorSize];  ///< Force 4KB size
  };

  DiskIndexHeader() : padding_{} {}

  /**
   * @brief Initialize header with given parameters.
   *
   * @param max_degree Maximum out-degree R of the graph
   * @param alpha Distance threshold multiplier α
   * @param dimension Vector dimension
   * @param num_points Number of points in the index
   * @param node_sector_size Size of each node on disk (in bytes)
   */
  void init(uint32_t max_degree,
            float alpha,
            uint32_t dimension,
            uint64_t num_points,
            uint64_t node_sector_size) {
    meta_.magic_ = kDiskANNMagicNumber;
    meta_.version_ = kDiskANNVersion;
    meta_.max_degree_ = max_degree;
    meta_.alpha_ = alpha;
    meta_.dimension_ = dimension;
    meta_.num_points_ = num_points;
    meta_.node_sector_size_ = node_sector_size;
    meta_.data_offset_ = kDiskSectorSize;  // Data starts right after header
    meta_.medoid_id_ = 0;
    meta_.index_build_time_ = 0;
    meta_.data_type_ = 0;  // Default to float

    // Initialize PQ fields to disabled
    meta_.pq_enabled_ = 0;
    meta_.pq_num_subspaces_ = 0;
    meta_.pq_codebook_offset_ = 0;
    meta_.pq_codebook_size_ = 0;
    meta_.pq_codes_offset_ = 0;
    meta_.pq_codes_size_ = 0;
    meta_.nodes_offset_ = kDiskSectorSize;  // Without PQ, nodes start right after header
  }

  /**
   * @brief Initialize PQ parameters and compute section offsets.
   *
   * Call this after init() to enable PQ-based in-memory navigation.
   * This sets up the file layout with codebook and PQ codes sections.
   *
   * @param num_subspaces Number of PQ subspaces (M)
   */
  void init_pq(uint32_t num_subspaces) {
    meta_.pq_enabled_ = 1;
    meta_.pq_num_subspaces_ = num_subspaces;

    // Calculate codebook size: M subspaces × 256 centroids × (D/M) dims × sizeof(float)
    uint32_t subspace_dim = meta_.dimension_ / num_subspaces;
    size_t codebook_raw_size =
        static_cast<size_t>(num_subspaces) * 256 * subspace_dim * sizeof(float);
    meta_.pq_codebook_size_ = math::round_up_pow2(codebook_raw_size, kDiskSectorSize);

    // Calculate PQ codes size: N points × M bytes
    size_t codes_raw_size = meta_.num_points_ * num_subspaces;
    meta_.pq_codes_size_ = math::round_up_pow2(codes_raw_size, kDiskSectorSize);

    // Set offsets: Header -> Codebook -> PQ Codes -> Nodes
    meta_.pq_codebook_offset_ = kDiskSectorSize;
    meta_.pq_codes_offset_ = meta_.pq_codebook_offset_ + meta_.pq_codebook_size_;
    meta_.nodes_offset_ = meta_.pq_codes_offset_ + meta_.pq_codes_size_;

    // Update data_offset for backward compatibility
    meta_.data_offset_ = meta_.nodes_offset_;
  }

  /**
   * @brief Check if PQ is enabled.
   */
  [[nodiscard]] auto is_pq_enabled() const -> bool { return meta_.pq_enabled_ != 0; }

  /**
   * @brief Get the byte offset for a specific node.
   *
   * @param node_id Node ID
   * @return Byte offset to the node data
   */
  [[nodiscard]] auto get_node_offset(uint64_t node_id) const -> uint64_t {
    return meta_.nodes_offset_ + node_id * meta_.node_sector_size_;
  }

  /**
   * @brief Validate the header integrity.
   *
   * @return true if header is valid, false otherwise
   */
  [[nodiscard]] auto is_valid() const -> bool {
    if (meta_.magic_ != kDiskANNMagicNumber) {
      return false;
    }
    // Support version 1 (no PQ) and version 2 (with PQ support)
    if (meta_.version_ != kDiskANNVersion && meta_.version_ != 1) {
      return false;
    }
    if (meta_.dimension_ == 0 || meta_.num_points_ == 0) {
      return false;
    }
    if (meta_.node_sector_size_ == 0 || meta_.node_sector_size_ % kDiskSectorSize != 0) {
      return false;
    }
    // Validate PQ fields if enabled
    if (meta_.pq_enabled_ != 0) {
      if (meta_.pq_num_subspaces_ == 0 || meta_.dimension_ % meta_.pq_num_subspaces_ != 0) {
        return false;
      }
      if (meta_.pq_codebook_offset_ < kDiskSectorSize) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief Save header to file stream.
   *
   * @param writer Output file stream
   */
  void save(std::ofstream &writer) const {
    static_assert(sizeof(DiskIndexHeader) == kDiskSectorSize,
                  "DiskIndexHeader must be exactly 4KB");
    writer.write(reinterpret_cast<const char *>(this), sizeof(DiskIndexHeader));
  }

  /**
   * @brief Load header from file stream.
   *
   * @param reader Input file stream
   */
  void load(std::ifstream &reader) {
    static_assert(sizeof(DiskIndexHeader) == kDiskSectorSize,
                  "DiskIndexHeader must be exactly 4KB");
    reader.read(reinterpret_cast<char *>(this), sizeof(DiskIndexHeader));
    if (!is_valid()) {
      throw std::runtime_error("Invalid DiskANN index header");
    }
  }
};

static_assert(sizeof(DiskIndexHeader) == kDiskSectorSize,
              "DiskIndexHeader must be exactly 4KB for Direct IO");

// ============================================================================
// DiskNode - On-disk representation of a graph node
// ============================================================================

/**
 * @brief Structure representing a node stored on disk.
 *
 * Each DiskNode contains:
 * - The original vector data
 * - Number of neighbors
 * - Neighbor IDs
 *
 * The node is padded to be a multiple of 4KB for Direct IO compatibility.
 *
 * Memory Layout within a DiskNode:
 * +------------------------+
 * | num_neighbors (4B)     |
 * +------------------------+
 * | neighbor_ids[0..R-1]   | <- R * 4 bytes
 * +------------------------+
 * | vector_data[0..dim-1]  | <- dim * sizeof(DataType) bytes
 * +------------------------+
 * | padding                | <- Fill to 4KB boundary
 * +------------------------+
 *
 * @tparam DataType The data type of vector elements (default: float)
 * @tparam IDType The data type for node IDs (default: uint32_t)
 */
template <typename DataType = float, typename IDType = uint32_t>
struct DiskNode {
  static_assert(std::is_trivial_v<DataType> && std::is_standard_layout_v<DataType>,
                "DataType must be a POD type");
  static_assert(std::is_trivial_v<IDType> && std::is_standard_layout_v<IDType>,
                "IDType must be a POD type");

  /**
   * @brief Calculate the required sector size for a node.
   *
   * This function computes the minimum number of sectors needed to store
   * a node with the given dimension and maximum degree.
   *
   * @param dimension Vector dimension
   * @param max_degree Maximum number of neighbors (R)
   * @return Size in bytes, aligned to kDiskSectorSize
   */
  [[nodiscard]] static constexpr auto calc_node_sector_size(uint32_t dimension, uint32_t max_degree)
      -> size_t {
    // Layout: num_neighbors(4B) + neighbor_ids(R*4B) + vector_data(dim*sizeof(DataType))
    size_t raw_size = sizeof(uint32_t) +             // num_neighbors
                      max_degree * sizeof(IDType) +  // neighbor_ids
                      dimension * sizeof(DataType);  // vector_data
    return math::round_up_pow2(raw_size, kDiskSectorSize);
  }

  /**
   * @brief Get offset to neighbor count within node data.
   * @return Byte offset to num_neighbors field
   */
  [[nodiscard]] static constexpr auto offset_num_neighbors() -> size_t { return 0; }

  /**
   * @brief Get offset to neighbor IDs array within node data.
   * @return Byte offset to neighbor_ids array
   */
  [[nodiscard]] static constexpr auto offset_neighbor_ids() -> size_t { return sizeof(uint32_t); }

  /**
   * @brief Get offset to vector data within node data.
   *
   * @param max_degree Maximum number of neighbors (R)
   * @return Byte offset to vector_data array
   */
  [[nodiscard]] static constexpr auto offset_vector_data(uint32_t max_degree) -> size_t {
    return sizeof(uint32_t) + max_degree * sizeof(IDType);
  }

  /**
   * @brief Serializer/Deserializer helper for DiskNode.
   *
   * This class provides methods to read/write node data from/to raw buffers.
   * It handles the conversion between the structured node representation
   * and the flat disk layout.
   */
  struct Accessor {
    uint8_t *data_;        ///< Pointer to the raw node data buffer
    uint32_t dimension_;   ///< Vector dimension
    uint32_t max_degree_;  ///< Maximum number of neighbors
    size_t node_size_;     ///< Total size of node in bytes

    Accessor() = default;

    /**
     * @brief Initialize accessor with node parameters.
     *
     * @param data Pointer to raw node data buffer
     * @param dimension Vector dimension
     * @param max_degree Maximum number of neighbors
     */
    Accessor(uint8_t *data, uint32_t dimension, uint32_t max_degree)
        : data_(data),
          dimension_(dimension),
          max_degree_(max_degree),
          node_size_(calc_node_sector_size(dimension, max_degree)) {}

    /**
     * @brief Get pointer to the number of neighbors.
     * @return Pointer to num_neighbors field
     */
    [[nodiscard]] auto num_neighbors_ptr() -> uint32_t * {
      return reinterpret_cast<uint32_t *>(data_ + offset_num_neighbors());
    }

    /**
     * @brief Get the number of neighbors.
     * @return Number of neighbors
     */
    [[nodiscard]] auto num_neighbors() const -> uint32_t {
      return *reinterpret_cast<const uint32_t *>(data_ + offset_num_neighbors());
    }

    /**
     * @brief Set the number of neighbors.
     * @param count Number of neighbors
     */
    void set_num_neighbors(uint32_t count) { *num_neighbors_ptr() = count; }

    /**
     * @brief Get pointer to the neighbor IDs array.
     * @return Pointer to neighbor_ids array
     */
    [[nodiscard]] auto neighbor_ids() -> IDType * {
      return reinterpret_cast<IDType *>(data_ + offset_neighbor_ids());
    }

    /**
     * @brief Get const pointer to the neighbor IDs array.
     * @return Const pointer to neighbor_ids array
     */
    [[nodiscard]] auto neighbor_ids() const -> const IDType * {
      return reinterpret_cast<const IDType *>(data_ + offset_neighbor_ids());
    }

    /**
     * @brief Get pointer to the vector data.
     * @return Pointer to vector_data array
     */
    [[nodiscard]] auto vector_data() -> DataType * {
      return reinterpret_cast<DataType *>(data_ + offset_vector_data(max_degree_));
    }

    /**
     * @brief Get const pointer to the vector data.
     * @return Const pointer to vector_data array
     */
    [[nodiscard]] auto vector_data() const -> const DataType * {
      return reinterpret_cast<const DataType *>(data_ + offset_vector_data(max_degree_));
    }

    /**
     * @brief Get a specific neighbor ID.
     *
     * @param index Index of the neighbor (0-based)
     * @return Neighbor ID at the given index
     */
    [[nodiscard]] auto get_neighbor(uint32_t index) const -> IDType {
      return neighbor_ids()[index];
    }

    /**
     * @brief Set a specific neighbor ID.
     *
     * @param index Index of the neighbor (0-based)
     * @param id Neighbor ID to set
     */
    void set_neighbor(uint32_t index, IDType id) { neighbor_ids()[index] = id; }

    /**
     * @brief Copy vector data into the node.
     *
     * @param vec Pointer to vector data to copy
     */
    void set_vector(const DataType *vec) {
      std::memcpy(vector_data(), vec, dimension_ * sizeof(DataType));
    }

    /**
     * @brief Copy neighbor IDs into the node.
     *
     * @param neighbors Pointer to neighbor IDs to copy
     * @param count Number of neighbors
     */
    void set_neighbors(const IDType *neighbors, uint32_t count) {
      set_num_neighbors(count);
      std::memcpy(neighbor_ids(), neighbors, count * sizeof(IDType));
      // Fill remaining slots with invalid ID (-1)
      for (uint32_t i = count; i < max_degree_; ++i) {
        neighbor_ids()[i] = static_cast<IDType>(-1);
      }
    }

    /**
     * @brief Clear the node data.
     */
    void clear() { std::memset(data_, 0, node_size_); }

    /**
     * @brief Initialize a node with vector and neighbors.
     *
     * @param vec Pointer to vector data
     * @param neighbors Pointer to neighbor IDs
     * @param num_nbrs Number of neighbors
     */
    void init(const DataType *vec, const IDType *neighbors, uint32_t num_nbrs) {
      clear();
      set_vector(vec);
      set_neighbors(neighbors, num_nbrs);
    }
  };
};

// ============================================================================
// DiskNodeBuffer - Aligned buffer for reading/writing DiskNodes
// ============================================================================

/**
 * @brief Aligned buffer for disk I/O operations.
 *
 * This class manages a 4KB-aligned memory buffer suitable for Direct IO.
 * It can hold one or more disk nodes and provides methods for aligned I/O.
 *
 * @tparam DataType The data type of vector elements
 * @tparam IDType The data type for node IDs
 */
template <typename DataType = float, typename IDType = uint32_t>
class DiskNodeBuffer {
 public:
  using NodeType = DiskNode<DataType, IDType>;
  using AccessorType = typename NodeType::Accessor;

 private:
  AlignedBuffer data_;      ///< 4KB-aligned buffer for Direct IO
  uint32_t dimension_{0};   ///< Vector dimension
  uint32_t max_degree_{0};  ///< Maximum number of neighbors
  size_t node_size_{0};     ///< Size of each node in bytes
  size_t num_nodes_{0};     ///< Number of nodes that fit in buffer

 public:
  DiskNodeBuffer() = default;
  ~DiskNodeBuffer() = default;

  DiskNodeBuffer(const DiskNodeBuffer &) = delete;
  auto operator=(const DiskNodeBuffer &) -> DiskNodeBuffer & = delete;

  DiskNodeBuffer(DiskNodeBuffer &&) noexcept = default;
  auto operator=(DiskNodeBuffer &&) noexcept -> DiskNodeBuffer & = default;

  /**
   * @brief Allocate aligned buffer for specified number of nodes.
   *
   * @param dimension Vector dimension
   * @param max_degree Maximum number of neighbors
   * @param num_nodes Number of nodes to allocate space for (default: 1)
   */
  void allocate(uint32_t dimension, uint32_t max_degree, size_t num_nodes = 1) {
    dimension_ = dimension;
    max_degree_ = max_degree;
    node_size_ = NodeType::calc_node_sector_size(dimension, max_degree);
    num_nodes_ = num_nodes;

    data_.resize(node_size_ * num_nodes);
    std::memset(data_.data(), 0, data_.size());
  }

  /**
   * @brief Get accessor for node at given index.
   *
   * @param index Node index in buffer (0-based)
   * @return Accessor for the node
   */
  [[nodiscard]] auto get_node(size_t index) -> AccessorType {
    if (index >= num_nodes_) {
      throw std::out_of_range("Node index out of range");
    }
    return AccessorType(data_.data() + index * node_size_, dimension_, max_degree_);
  }

  /// @brief Get raw data pointer.
  [[nodiscard]] auto data() -> uint8_t * { return data_.data(); }

  /// @brief Get const raw data pointer.
  [[nodiscard]] auto data() const -> const uint8_t * { return data_.data(); }

  /// @brief Get total buffer size.
  [[nodiscard]] auto size() const -> size_t { return data_.size(); }

  /// @brief Get single node size.
  [[nodiscard]] auto node_size() const -> size_t { return node_size_; }

  /// @brief Get number of nodes in buffer.
  [[nodiscard]] auto num_nodes() const -> size_t { return num_nodes_; }

  /// @brief Check if buffer is allocated.
  [[nodiscard]] auto is_valid() const -> bool { return !data_.empty(); }
};

}  // namespace alaya
