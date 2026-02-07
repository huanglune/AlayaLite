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
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "storage/buffer/buffer_pool.hpp"
#include "storage/io/direct_file_io.hpp"
#include "utils/macros.hpp"
#include "utils/memory.hpp"
#include "utils/types.hpp"

namespace alaya {

// ============================================================================
// Constants for DataFile
// ============================================================================
constexpr size_t kDataBlockSize = 4096;
inline thread_local AlignedBuffer tl_io_buffer(kDataBlockSize);

// ============================================================================
// NodeViewer and NodeEditor forward declarations
// ============================================================================

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

  class Viewer {
   public:
    /**
     * @brief Construct a Viewer for a node row in the data file.
     *
     * @param row_ptr pointer to the start of the node row
     * @param dim vector dimension
     * @param max_deg maximum degree (R)
     * @param vec_offset offset to the vector data within the row
     */
    Viewer(const char *row_ptr, uint32_t dim, uint32_t max_degree, size_t vec_offset)
        : row_ptr_(row_ptr), dim_(dim), max_degree_(max_degree), vec_offset_(vec_offset) {}

    [[nodiscard]] auto vector_view() const -> std::span<const DataType> {
      const auto *vec_ptr = reinterpret_cast<const DataType *>(row_ptr_ + vec_offset_);
      return std::span<const DataType>(vec_ptr, dim_);
    }

    [[nodiscard]] auto neighbors_view() const -> const NeighborList<IDType> & {
      return *reinterpret_cast<const NeighborList<IDType> *>(row_ptr_);
    }

    void copy_vector_to(std::span<DataType> buffer) const {
      if (buffer.size() != dim_) {
        throw std::invalid_argument("Vector dim mismatch");
      }
      auto view = vector_view();
      std::copy(view.begin(), view.end(), buffer.begin());
    }

    auto copy_neighbors_to(std::span<IDType> buffer) const -> uint32_t {
      const auto &nbrs = neighbors_view();
      size_t count = std::min(nbrs.size(), buffer.size());
      std::copy_n(nbrs.begin(), count, buffer.begin());
      return static_cast<uint32_t>(count);
    }

   private:
    const char *row_ptr_;
    uint32_t dim_;
    uint32_t max_degree_;
    size_t vec_offset_;
  };

  class Editor {
   public:
    Editor(byte *row_ptr, uint32_t dim, uint32_t max_degree, size_t vec_offset)
        : row_ptr_(row_ptr), dim_(dim), max_degree_(max_degree), vec_offset_(vec_offset) {}

    auto set_vector(std::span<const DataType> new_vec) -> void {
      if (new_vec.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
      }

      auto *dest = reinterpret_cast<DataType *>(row_ptr_ + vec_offset_);
      std::copy(new_vec.begin(), new_vec.end(), dest);
    }

    auto set_neighbors(std::span<const IDType> new_nbrs) -> void {
      if (new_nbrs.size() > max_degree_) {
        throw std::length_error("Too many neighbors");
      }
      // get variable reference
      auto &list = *reinterpret_cast<NeighborList<IDType> *>(row_ptr_);
      list.num_neighbors_ = static_cast<uint32_t>(new_nbrs.size());
      std::copy(new_nbrs.begin(), new_nbrs.end(), list.begin());

      // fill remaining slots with -1 (invalid ID)
      std::fill(list.end(), list.neighbor_ids() + max_degree_, static_cast<IDType>(-1));
    }

    // ---------------------------------------------------------
    // Advanced Interface: Get mutable view (for in-place sorting, in-place cropping)
    // ---------------------------------------------------------

    /**
     * @brief Get a mutable span to the vector data for in-place modification.
     * @return Mutable span of vector data
     */
    auto mutable_vector() -> std::span<DataType> {
      auto *ptr = reinterpret_cast<DataType *>(row_ptr_ + vec_offset_);
      return std::span<DataType>(ptr, dim_);
    }

    /**
     * @brief Get a mutable reference to the neighbor list for in-place modification.
     * @return Mutable reference to NeighborList
     */
    auto mutable_neighbors() -> NeighborList<IDType> & {
      return *reinterpret_cast<NeighborList<IDType> *>(row_ptr_);
    }

    [[nodiscard]] auto as_viewer() const -> Viewer {
      return Viewer(reinterpret_cast<const char *>(row_ptr_), dim_, max_degree_, vec_offset_);
    }

   private:
    byte *row_ptr_;
    uint32_t dim_;
    uint32_t max_degree_;
    size_t vec_offset_;
  };

  ALAYA_NON_COPYABLE_BUT_MOVABLE(DataFile);

  explicit DataFile(BufferPoolType *bp = nullptr) : buffer_pool_(bp) {
    if (buffer_pool_ == nullptr) {
      throw std::invalid_argument("DataFile requires a valid BufferPool");
    }
  }
  ~DataFile() { close(); }

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
  // Vector operations
  // -------------------------------------------------------------------------
  template <typename Func>
  void inspect_node(uint32_t node_id, Func &&visitor) const {
    validate_id(node_id);
    auto [block_offset, row_offset] = get_block_and_row_offset(node_id);

    // Cache key = block_id so that all nodes in the same 4KB block share one cache entry.
    // This increases effective cache utilization by nodes_per_block_ (typically 3-5x).
    uint32_t block_id = node_id / nodes_per_block_;
    PageHandle handle;
    handle = buffer_pool_->get_or_read(block_id,
                                       *file_,
                                       block_offset,
                                       reinterpret_cast<uint8_t *>(tl_io_buffer.data()));

    if (handle.empty()) {
      throw std::runtime_error("Failed to read node");
    }

    Viewer viewer(reinterpret_cast<const char *>(handle.data() + row_offset),
                  dim_,
                  max_degree_,
                  vector_offset_);
    visitor(viewer);

    // 4. Function ends -> handle destructs -> reference count decreases -> page becomes evictable
  }

  template <typename Func>
  void modify_node(uint32_t node_id, Func &&modifier) {
    if (!is_writable_) {
      throw std::runtime_error("Read only");
    }
    validate_id(node_id);
    auto [block_offset, row_offset] = get_block_and_row_offset(node_id);

    // get Handle (RAII)
    // Modification operations also need to be "read" into the Cache first, because we need to
    // preserve other data in the block that does not require modification.
    // Cache key = block_id so that all nodes in the same 4KB block share one cache entry.
    uint32_t block_id = node_id / nodes_per_block_;
    PageHandle handle;
    if (tl_io_buffer.size() < kDataBlockSize) {
      tl_io_buffer.resize(kDataBlockSize);
    }
    handle = buffer_pool_->get_or_read(block_id,
                                       *file_,
                                       block_offset,
                                       reinterpret_cast<uint8_t *>(tl_io_buffer.data()));

    if (handle.empty()) {
      throw std::runtime_error("Failed to read node for modification");
    }

    Editor editor(reinterpret_cast<byte *>(handle.mutable_data() + row_offset),
                  dim_,
                  max_degree_,
                  vector_offset_);
    modifier(editor);

    // 3. Write-Through
    // Write the memory in the Cache back to the disk directly.
    // Advantage: No need to malloc another buffer for dirty data.
    // Note: The handle must remain alive (pinned) during the write, which is guaranteed by RAII.
    file_->write(reinterpret_cast<char *>(handle.mutable_data()), kDataBlockSize, block_offset);

    // 4. End
    // The Cache is the latest, and the disk is also the latest.
  }

  auto read_vector(uint32_t node_id, std::span<DataType> buffer) const -> void {
    inspect_node(node_id, [&](const Viewer &v) -> auto {
      v.copy_vector_to(buffer);
    });
  }

  auto read_neighbors(uint32_t node_id, std::span<IDType> buffer) const -> uint32_t {
    uint32_t count = 0;
    inspect_node(node_id, [&](const Viewer &v) -> auto {
      count = v.copy_neighbors_to(buffer);
    });
    return count;
  }

  auto write_vector(uint32_t node_id, std::span<const DataType> buffer) -> void {
    modify_node(node_id, [&](Editor &e) -> auto {
      e.set_vector(buffer);
    });
  }

  auto write_neighbors(uint32_t node_id, std::span<const IDType> buffer) -> void {
    modify_node(node_id, [&](Editor &e) -> auto {
      e.set_neighbors(buffer);
    });
  }

  /**
   * @brief Modify a range of nodes with one disk write per block.
   *
   * For sequential node writes (e.g., during index building), this is much
   * more efficient than calling modify_node() per node, because all nodes
   * sharing the same 4KB block are written to disk only once.
   *
   * Bypasses the BufferPool and uses direct I/O for maximum throughput.
   *
   * @param start_id First node ID to modify (inclusive)
   * @param end_id Last node ID to modify (exclusive)
   * @param modifier Callback: (uint32_t node_id, Editor &editor) -> void
   */
  template <typename Func>
  void batch_modify(uint32_t start_id, uint32_t end_id, Func &&modifier) {
    if (!is_writable_) {
      throw std::runtime_error("Read only");
    }
    if (end_id > capacity_) {
      throw std::out_of_range("Node ID range out of bounds");
    }

    AlignedBuffer block_buf(kDataBlockSize);
    uint32_t cur = start_id;

    while (cur < end_id) {
      uint32_t block_idx = cur / nodes_per_block_;
      uint32_t block_end_node = std::min((block_idx + 1) * nodes_per_block_, end_id);
      block_end_node = std::min(block_end_node, capacity_);

      uint64_t block_offset = static_cast<uint64_t>(block_idx) * kDataBlockSize;

      // Read existing block data; zero-fill if sparse/unwritten
      auto read_bytes = file_->read(block_buf.data(), kDataBlockSize, block_offset);
      if (read_bytes < static_cast<ssize_t>(kDataBlockSize)) {
        std::memset(block_buf.data(), 0, kDataBlockSize);
      }

      // Modify all nodes in this block
      for (uint32_t id = cur; id < block_end_node; ++id) {
        size_t row_offset = static_cast<size_t>(id % nodes_per_block_) * row_size_;
        Editor editor(reinterpret_cast<byte *>(block_buf.data() + row_offset),
                      dim_,
                      max_degree_,
                      vector_offset_);
        modifier(id, editor);
      }

      // Single write for entire block
      file_->write(block_buf.data(), kDataBlockSize, block_offset);

      cur = block_end_node;
    }
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
   * @brief Preload a specific block into the buffer pool.
   *
   * Used by the searcher to warm up the cache at open() time,
   * reading blocks sequentially for maximum I/O throughput.
   *
   * @param block_id Block index (0-based)
   */
  void preload_block(uint32_t block_id) const {
    if (!is_open_ || !file_) {
      return;
    }
    uint64_t block_offset = static_cast<uint64_t>(block_id) * kDataBlockSize;
    buffer_pool_->get_or_read(block_id,
                              *file_,
                              block_offset,
                              reinterpret_cast<uint8_t *>(tl_io_buffer.data()));
    // Handle destructs immediately -> unpin -> page becomes evictable but stays cached
  }

  /**
   * @brief Batch-prefetch multiple blocks into the buffer pool via io_uring.
   *
   * Identifies cache misses among the given block_ids, submits all reads to
   * io_uring in a single batch (parallel SSD reads), then inserts results into
   * the buffer pool. This converts N sequential preads into 1 batch submission,
   * reducing total I/O latency from N * latency to ~1 * latency.
   *
   * @param block_ids Sorted span of block IDs to prefetch (may contain duplicates)
   */
  void prefetch_blocks(std::span<const uint32_t> block_ids) const {
    if (!is_open_ || !file_ || block_ids.empty()) {
      return;
    }

    // 1. Deduplicate and filter to cache misses
    std::vector<uint32_t> missing;
    missing.reserve(block_ids.size());
    auto prev = static_cast<uint32_t>(-1);
    for (auto blk : block_ids) {
      if (blk == prev) {
        continue;  // Skip duplicates (input is sorted)
      }
      prev = blk;
      auto handle = buffer_pool_->get(blk);
      if (handle.empty()) {
        missing.push_back(blk);
      }
    }
    if (missing.empty()) {
      return;
    }

    // 2. Allocate aligned buffers and prepare IORequests
    std::vector<AlignedBuffer> buffers;
    buffers.reserve(missing.size());
    std::vector<IORequest> requests;
    requests.reserve(missing.size());
    for (auto blk : missing) {
      buffers.emplace_back(kDataBlockSize);
      requests.emplace_back(buffers.back().data(),
                            kDataBlockSize,
                            static_cast<uint64_t>(blk) * kDataBlockSize);
    }

    // 3. Submit all reads to io_uring in one batch (parallel SSD reads)
    // Thread-safe: each thread has its own io_uring ring (thread-local)
    file_->submit_reads(requests);

    // 4. Insert completed reads into buffer pool (no additional disk I/O)
    for (size_t i = 0; i < missing.size(); ++i) {
      if (requests[i].is_success()) {
        buffer_pool_->put(missing[i], reinterpret_cast<uint8_t *>(buffers[i].data()));
        // Handle destructs → unpin → page stays cached
      }
    }
  }

  /**
   * @brief Get a vector view directly from a pinned block handle.
   *
   * Used by block-local pinning: the caller keeps one PageHandle alive per
   * unique block and extracts vectors without additional pin/unpin overhead.
   *
   * @param handle Pinned page handle for the block containing node_id
   * @param node_id The node whose vector to extract
   * @return Span pointing into the handle's data (valid while handle is alive)
   */
  [[nodiscard]] auto get_vector_in_block(const PageHandle &handle, uint32_t node_id) const
      -> std::span<const DataType> {
    size_t row_offset = static_cast<size_t>(node_id % nodes_per_block_) * row_size_;
    const auto *vec_ptr =
        reinterpret_cast<const DataType *>(handle.data() + row_offset + vector_offset_);
    return std::span<const DataType>(vec_ptr, dim_);
  }

  /**
   * @brief Get total file size in bytes.
   */
  [[nodiscard]] auto total_file_size() const -> uint64_t {
    uint64_t num_blocks =
        (static_cast<uint64_t>(capacity_) + nodes_per_block_ - 1) / nodes_per_block_;
    return num_blocks * kDataBlockSize;
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

  auto get_block_and_row_offset(uint32_t node_id) const -> std::pair<uint64_t, size_t> {
    return {static_cast<size_t>(node_id / nodes_per_block_) * kDataBlockSize,
            static_cast<size_t>(node_id % nodes_per_block_) * row_size_};
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
