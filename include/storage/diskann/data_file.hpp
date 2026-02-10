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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "storage/buffer/buffer_pool.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/macros.hpp"
#include "utils/memory.hpp"

namespace alaya {

// ============================================================================
// Constants for DataFile
// ============================================================================
constexpr size_t kDataBlockSize = 4096;
inline thread_local AlignedBuffer tl_io_buffer(kDataBlockSize);

/**
 * @brief Neighbor table view mapped onto raw memory
 *
 * memory layout: [count (4B)] [id0] [id1] ...
 */
template <typename IDType>
struct NeighborList {
  uint32_t num_neighbors_;

  auto neighbor_ids() -> IDType * { return reinterpret_cast<IDType *>(this + 1); }
  auto neighbor_ids() const -> const IDType * { return reinterpret_cast<const IDType *>(this + 1); }

  // Vector-like access
  auto begin() -> IDType * { return neighbor_ids(); }
  auto end() -> IDType * { return neighbor_ids() + num_neighbors_; }
  auto begin() const -> const IDType * { return neighbor_ids(); }
  auto end() const -> const IDType * { return neighbor_ids() + num_neighbors_; }

  auto size() const -> size_t { return num_neighbors_; }
  auto empty() const -> bool { return num_neighbors_ == 0; }

  NeighborList() = default;
  ~NeighborList() = default;
  ALAYA_NON_COPYABLE(NeighborList);
};

// ============================================================================
// DataFile - Header-less block-based data file with Direct I/O support
// ============================================================================

/**
 * @brief Header-less data file for graph/vector data with Direct I/O support.
 *
 * Data is stored in 4KB sector-aligned blocks. Within each block, node rows
 * are packed without internal alignment. All metadata (capacity, dim, etc.)
 * is stored in MetaFile, not in this file.
 *
 * File Layout (no header, data starts at offset 0):
 * +-------------------------------------------+ Offset 0 (Block 0)
 * | Node 0: [neighbors][vector]               |
 * | Node 1: [neighbors][vector]               |
 * | ... (nodes_per_block nodes)               |
 * | [padding to 4KB boundary]                 |
 * +-------------------------------------------+ Offset 4096 (Block 1)
 * | Node N: [neighbors][vector]               |
 * | ...                                       |
 * | [padding to 4KB boundary]                 |
 * +-------------------------------------------+ ...
 *
 * Per-node row layout (no padding):
 * [num_neighbors (4B)] [neighbor_ids (R * sizeof(IDType))] [vector (dim * sizeof(DataType))]
 *
 * @tparam DataType Vector element type (float, int8_t, etc.)
 * @tparam IDType Node ID type (uint32_t, uint64_t)
 */
template <typename DataType = float,
          typename IDType = uint32_t,
          ReplacerStrategy ReplacerType = LRUReplacer>
class DataFile {
  static_assert(std::is_trivial_v<DataType>, "DataType must be trivial");
  static_assert(std::is_trivial_v<IDType>, "IDType must be trivial");

 public:
  using BufferPoolType = BufferPool<IDType, ReplacerType>;
  using PageHandle = typename BufferPoolType::PageHandle;

  /**
   * @brief RAII node reference combining read/write access with buffer pool pinning.
   *
   * Unifies the former Viewer (read-only) and Editor (read-write) into a single type.
   * Holds a PageHandle to keep the underlying page pinned while the reference
   * is alive. Supports zero-copy reads and in-place writes.
   *
   * - In inspect_node: passed as const NodeRef& (only read methods accessible)
   * - In modify_node / batch_modify: passed as NodeRef& (read + write)
   */
  class NodeRef {
   public:
    NodeRef() = default;

    NodeRef(PageHandle &&handle,
            uint8_t *node_ptr,
            uint32_t dim,
            uint32_t max_deg,
            size_t vec_offset)
        : handle_(std::move(handle)),
          node_ptr_(node_ptr),
          dim_(dim),
          max_deg_(max_deg),
          vec_offset_(vec_offset) {}

    ALAYA_NON_COPYABLE_BUT_MOVABLE(NodeRef);
    ~NodeRef() = default;

    [[nodiscard]] auto empty() const -> bool { return node_ptr_ == nullptr; }

    // --- Read interface (zero-copy) ---

    [[nodiscard]] auto vector() const -> std::span<const DataType> {
      return {reinterpret_cast<const DataType *>(node_ptr_ + vec_offset_), dim_};
    }

    [[nodiscard]] auto neighbors() const -> std::span<const IDType> {
      uint32_t count = *reinterpret_cast<const uint32_t *>(node_ptr_);
      const auto *ids = reinterpret_cast<const IDType *>(node_ptr_ + sizeof(uint32_t));
      return {ids, count};
    }

    // --- Write interface ---

    void set_vector(std::span<const DataType> data) {
      if (data.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
      }
      mark_dirty();
      auto *dst = reinterpret_cast<DataType *>(node_ptr_ + vec_offset_);
      std::copy(data.begin(), data.end(), dst);
    }

    void set_neighbors(std::span<const IDType> nbrs) {
      if (nbrs.size() > max_deg_) {
        throw std::length_error("Too many neighbors");
      }
      mark_dirty();
      auto &list = *reinterpret_cast<NeighborList<IDType> *>(node_ptr_);
      list.num_neighbors_ = static_cast<uint32_t>(nbrs.size());
      std::copy(nbrs.begin(), nbrs.end(), list.begin());

      // Optional: padding it
      std::fill(list.end(), list.neighbor_ids() + max_deg_, static_cast<IDType>(-1));
    }

    auto mutable_vector() -> std::span<DataType> {
      return {reinterpret_cast<DataType *>(node_ptr_ + vec_offset_), dim_};
    }

    auto mutable_neighbors() -> NeighborList<IDType> & {
      return *reinterpret_cast<NeighborList<IDType> *>(node_ptr_);
    }

    void mark_dirty() { handle_.mark_dirty(); }
    [[nodiscard]] auto handle() const -> const PageHandle & { return handle_; }

   private:
    PageHandle handle_;
    uint8_t *node_ptr_{nullptr};
    uint32_t dim_{0};
    uint32_t max_deg_{0};
    size_t vec_offset_{0};
  };

  ALAYA_NON_COPYABLE_BUT_MOVABLE(DataFile);

  explicit DataFile(BufferPoolType *bp = nullptr) : buffer_pool_(bp) {
    if (buffer_pool_ == nullptr) {
      throw std::invalid_argument("DataFile requires a valid BufferPool");
    }
  }
  ~DataFile() { close(); }

  // ==========================================================================
  // Node Access Methods
  // ==========================================================================
  [[nodiscard]] auto get_node(uint32_t node_id) -> NodeRef {
    // 1. Calculate offsets (Compiler optimizes this heavily)
    uint32_t block_id = node_id / nodes_per_block_;
    uint64_t block_offset = static_cast<uint64_t>(block_id) * kDataBlockSize;
    uint32_t node_offset = (node_id % nodes_per_block_) * row_size_;

    // 2. Fetch page (Thread-safe, RAII)
    PageHandle handle = buffer_pool_->get_or_read(block_id,
                                                  *file_,
                                                  block_offset,
                                                  reinterpret_cast<uint8_t *>(tl_io_buffer.data()));

    if (handle.empty()) {
      throw std::runtime_error("Failed to read node: " + std::to_string(node_id));
    }

    uint8_t *node_ptr = handle.mutable_data() + node_offset;
    return NodeRef(std::move(handle), node_ptr, dim_, max_degree_, vector_offset_);
  }

  void prefetch_blocks(std::span<const uint32_t> block_ids) {
    if (block_ids.empty()) {
      return;
    }

    for (uint32_t block_id : block_ids) {
      uint64_t offset = static_cast<uint64_t>(block_id) * kDataBlockSize;
      buffer_pool_->prefetch(block_id, *file_, offset);
    }
  }

  // -------------------------------------------------------------------------
  // File operations
  // -------------------------------------------------------------------------

  /**
   * @brief Create a new data file.
   *
   * @param path File path for the data file
   * @param capacity Maximum number of nodes
   * @param dim Vector dimension
   * @param max_degree Maximum out-degree (R)
   */
  void create(std::string_view path, uint32_t capacity, uint32_t dim, uint32_t max_degree) {
    if (is_open_) {
      throw std::runtime_error("DataFile already open");
    }

    path_ = std::string(path);
    init_layout(capacity, dim, max_degree);

    file_ = std::make_unique<DirectFileIO>();
    file_->open(path_, DirectFileIO::Mode::kReadWrite);

    // Pre-allocate file by writing zeros to the last block
    uint64_t total_size = total_file_size();

    if (total_size > 0) {
      // Write last block to extend file to full size
      AlignedBuffer zeros(kDataBlockSize);
      std::memset(zeros.data(), 0, kDataBlockSize);
      file_->write(reinterpret_cast<char *>(zeros.data()),
                   kDataBlockSize,
                   total_size - kDataBlockSize);
    }

    is_open_ = true;
    is_writable_ = true;
  }

  /**
   * @brief Open an existing data file.
   *
   * Layout parameters come from MetaFile (not from any file header).
   *
   * @param path File path to the data file
   * @param capacity Maximum number of nodes (from MetaFile)
   * @param dim Vector dimension (from MetaFile)
   * @param max_degree Maximum out-degree (from MetaFile)
   * @param writable Whether to open for writing
   */
  void open(std::string_view path,
            uint32_t capacity,
            uint32_t dim,
            uint32_t max_degree,
            bool writable = false) {
    if (is_open_) {
      throw std::runtime_error("DataFile already open");
    }

    path_ = std::string(path);
    init_layout(capacity, dim, max_degree);

    file_ = std::make_unique<DirectFileIO>();
    file_->open(path_, writable ? DirectFileIO::Mode::kReadWrite : DirectFileIO::Mode::kRead);

    is_open_ = true;
    is_writable_ = writable;
  }

  /**
   * @brief Close the data file.
   */
  /**
   * @brief Flush all dirty buffer pool pages back to the data file.
   */
  void flush() {
    if (!is_open_ || !file_) {
      return;
    }
    buffer_pool_->flush_all([this](uint32_t block_id, const uint8_t *data) -> void {
      file_->write(reinterpret_cast<const char *>(data),
                   kDataBlockSize,
                   static_cast<uint64_t>(block_id) * kDataBlockSize);
    });
  }

  void close() {
    if (!is_open_) {
      return;
    }

    if (file_) {
      file_->close();
      file_.reset();
    }

    path_.clear();
    is_open_ = false;
    is_writable_ = false;
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  [[nodiscard]] auto capacity() const -> uint32_t { return capacity_; }
  [[nodiscard]] auto dimension() const -> uint32_t { return dim_; }
  [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }
  [[nodiscard]] auto nodes_per_block() const -> uint32_t { return nodes_per_block_; }
  [[nodiscard]] auto row_size() const -> size_t { return row_size_; }

  /// Map a node ID to its containing block index.
  [[nodiscard]] auto block_id_of(uint32_t node_id) const -> uint32_t {
    return node_id / nodes_per_block_;
  }

  [[nodiscard]] auto num_blocks() const -> uint32_t {
    return (capacity_ + nodes_per_block_ - 1) / nodes_per_block_;
  }
  [[nodiscard]] auto is_open() const -> bool { return is_open_; }
  [[nodiscard]] auto is_writable() const -> bool { return is_writable_; }
  [[nodiscard]] auto path() const -> const std::string & { return path_; }

  /**
   * @brief Write a raw block to the data file (used by BufferPool flush callback).
   */
  void write_block(uint32_t block_id, const uint8_t *data) {
    if (!is_open_ || !is_writable_ || !file_) {
      return;
    }
    file_->write(reinterpret_cast<const char *>(data),
                 kDataBlockSize,
                 static_cast<uint64_t>(block_id) * kDataBlockSize);
  }

  /**
   * @brief Grow the data file to accommodate a new capacity.
   *
   * Extends the file by writing zero blocks for the new region.
   * Must be called when MetaFile auto-grows beyond the data file's capacity.
   *
   * @param new_capacity New capacity (must be > current capacity)
   */
  void grow(uint32_t new_capacity) {
    if (!is_writable_) {
      throw std::runtime_error("Cannot grow read-only DataFile");
    }
    if (new_capacity <= capacity_) {
      return;
    }

    uint32_t old_num_blocks = num_blocks();
    capacity_ = new_capacity;
    uint32_t new_num_blocks = num_blocks();

    // Write zero blocks for the new region
    if (new_num_blocks > old_num_blocks) {
      AlignedBuffer zeros(kDataBlockSize);
      std::memset(zeros.data(), 0, kDataBlockSize);
      // Write last block to extend file to full new size
      uint64_t last_block_offset = static_cast<uint64_t>(new_num_blocks - 1) * kDataBlockSize;
      file_->write(reinterpret_cast<char *>(zeros.data()), kDataBlockSize, last_block_offset);
    }
  }

  /**
   * @brief Get total file size in bytes.
   */
  [[nodiscard]] auto total_file_size() const -> uint64_t {
    uint64_t num_blocks =
        (static_cast<uint64_t>(capacity_) + nodes_per_block_ - 1) / nodes_per_block_;
    return num_blocks * kDataBlockSize;
  }

  /**
   * @brief Preload a block into the buffer pool cache.
   */
  auto preload_block(uint32_t block_id) const -> void {
    if (!is_open_) {
      throw std::runtime_error("Not open");
    }
    if (block_id >= num_blocks()) {
      throw std::out_of_range("Block ID out of range");
    }
    // Touch the block to load it into the buffer pool
    buffer_pool_->get_or_read(block_id,
                              *file_,
                              static_cast<uint64_t>(block_id) * kDataBlockSize,
                              reinterpret_cast<uint8_t *>(tl_io_buffer.data()));
  }

 private:
  /**
   * @brief Initialize layout parameters from dim and max_degree.
   */
  void init_layout(uint32_t capacity, uint32_t dim, uint32_t max_degree) {
    dim_ = dim;
    max_degree_ = max_degree;
    capacity_ = capacity;

    // Per-node row: [num_neighbors (4B)] [neighbor_ids (R * sizeof(IDType))] [vector]
    neighbor_size_ = sizeof(uint32_t) + static_cast<size_t>(max_degree) * sizeof(IDType);
    vector_offset_ = neighbor_size_;
    row_size_ = neighbor_size_ + static_cast<size_t>(dim) * sizeof(DataType);
    nodes_per_block_ = static_cast<uint32_t>(kDataBlockSize / row_size_);

    if (nodes_per_block_ == 0) {
      throw std::invalid_argument(
          "Node row size exceeds block size — increase block size or reduce parameters");
    }
  }

  auto validate_id(uint32_t node_id) const -> void {
    if (!is_open_) {
      throw std::runtime_error("Not open");
    }
    if (node_id >= capacity_) {
      throw std::out_of_range("Node ID out of range");
    }
  }

  // Layout parameters
  uint32_t dim_{0};              // Vector dimension
  uint32_t max_degree_{0};       // Maximum out-degree (R)
  uint32_t capacity_{0};         // Maximum number of nodes (from MetaFile)
  size_t row_size_{0};           // Total bytes per node row
  size_t neighbor_size_{0};      // sizeof(uint32_t) + max_degree * sizeof(IDType)
  size_t vector_offset_{0};      // = neighbor_size_ (vector starts after neighbors)
  uint32_t nodes_per_block_{0};  // floor(kDataBlockSize / row_size)

  // I/O
  std::string path_;
  std::unique_ptr<DirectFileIO> file_;
  BufferPoolType *buffer_pool_{nullptr};  // Reference to external buffer pool
  bool is_open_{false};
  bool is_writable_{false};
};

}  // namespace alaya
