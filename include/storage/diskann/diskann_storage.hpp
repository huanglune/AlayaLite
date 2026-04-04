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

#include <cstdint>
#include <exception>
#include <filesystem>  // NOLINT(build/c++17)
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "data_file.hpp"
#include "meta_file.hpp"
#include "utils/macros.hpp"

namespace alaya {

// ============================================================================
// File extension constants
// ============================================================================

constexpr std::string_view kMetaFileExtension = ".meta";
constexpr std::string_view kDataFileExtension = ".data";

// ============================================================================
// DiskANNStorage - Unified storage manager for three-file architecture
// ============================================================================

/**
 * @brief Unified storage manager for two-file DiskANN architecture.
 *
 * Coordinates access to:
 * - MetaFile: In-memory metadata and validity tracking
 * - DataFile: Direct I/O for full vector/graph access (with BufferPool caching)
 *
 * File naming convention:
 * - base_path.meta - Metadata file
 * - base_path.data - Graph/vector data file
 *
 * @tparam DataType Vector element type (float, int8_t, etc.)
 * @tparam IDType Node ID type (uint32_t, uint64_t)
 */
template <typename DataType = float,
          typename IDType = uint32_t,
          ReplacerStrategy ReplacerType = LRUReplacer>
class DiskANNStorage {
 public:
  using BufferPoolType = BufferPool<IDType, ReplacerType>;
  using DataFileType = DataFile<DataType, IDType, ReplacerType>;
  using NodeRef = typename DataFileType::NodeRef;

  ALAYA_NON_COPYABLE_BUT_MOVABLE(DiskANNStorage);

  explicit DiskANNStorage(BufferPoolType *bp) : data_(bp) {
    if (bp == nullptr) {
      throw std::invalid_argument("DiskANNStorage requires a valid BufferPool");
    }
  }
  ~DiskANNStorage() noexcept {
    try {
      close();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }

  // -------------------------------------------------------------------------
  // Path generation helpers
  // -------------------------------------------------------------------------

  [[nodiscard]] static auto meta_path(std::string_view base) -> std::string {
    return std::string(base) + std::string(kMetaFileExtension);
  }
  [[nodiscard]] static auto data_path(std::string_view base) -> std::string {
    return std::string(base) + std::string(kDataFileExtension);
  }

  // -------------------------------------------------------------------------
  // File operations
  // -------------------------------------------------------------------------

  /**
   * @brief Create a new DiskANN index with two-file architecture.
   *
   * @param base_path Base path for index files (without extension)
   * @param capacity Maximum number of vectors
   * @param dim Vector dimension
   * @param max_degree Maximum out-degree (R)
   * @param num_pq_subspaces Unused (kept for API compatibility, must be 0)
   * @param metric_type Metric type enum (0=L2, 1=IP, 2=COS)
   * @param data_type Data type enum
   */
  void create(std::string_view base_path,
              uint32_t capacity,
              uint32_t dim,
              uint32_t max_degree,
              uint32_t num_pq_subspaces = 0,
              uint32_t metric_type = 0,
              uint32_t data_type = 0) {
    (void)num_pq_subspaces;
    if (is_open_) {
      throw std::runtime_error("DiskANNStorage already open");
    }

    base_path_ = std::string(base_path);
    try {
      // Create metadata file (may round up capacity for bitmap alignment)
      meta_.create(meta_path(base_path), capacity, dim, max_degree, metric_type, data_type);

      // Use MetaFile's actual capacity (rounded up) for data file
      uint32_t actual_capacity = meta_.capacity();

      // Create data file
      data_.create(data_path(base_path), actual_capacity, dim, max_degree);

      is_open_ = true;
    } catch (...) {
      cleanup_failed_create();
      base_path_.clear();
      is_open_ = false;
      throw;
    }

    // Remove stale PQ sidecar only after successful creation, so a failed
    // rebuild-in-place does not additionally destroy the old .pq file.
    std::string legacy_pq = std::string(base_path) + ".pq";
    if (std::filesystem::exists(legacy_pq)) {
      std::error_code ec;
      std::filesystem::remove(legacy_pq, ec);
      if (ec) {
        // Index was created successfully but stale .pq couldn't be removed.
        // Close the new index and report the error — open() would reject it.
        close();
        throw std::runtime_error("Index created but failed to remove stale PQ sidecar at " +
                                 legacy_pq + ": " + ec.message() +
                                 ". Remove it manually, then re-open.");
      }
    }
  }

  /**
   * @brief Open an existing DiskANN index.
   *
   * @param base_path Base path for index files (without extension)
   * @param writable Whether to open for writing
   */
  void open(std::string_view base_path, bool writable = false) {
    if (is_open_) {
      throw std::runtime_error("DiskANNStorage already open");
    }

    base_path_ = std::string(base_path);
    try {
      // Fail fast if a legacy PQ sidecar exists
      std::string legacy_pq = std::string(base_path) + ".pq";
      if (std::filesystem::exists(legacy_pq)) {
        throw std::runtime_error(
            "Legacy PQ sidecar found at " + legacy_pq +
            ". PQ indexes are no longer supported — rebuild the index without PQ.");
      }

      // Open metadata file
      meta_.open(meta_path(base_path));

      // Open data file
      data_.open(data_path(base_path),
                 meta_.capacity(),
                 meta_.dimension(),
                 meta_.max_degree(),
                 writable);

      is_open_ = true;
    } catch (...) {
      data_.close();
      meta_.close();
      base_path_.clear();
      throw;
    }
  }

  /**
   * @brief Close all files.
   */
  void close() {
    if (!is_open_) {
      return;
    }

    std::exception_ptr close_error;
    try {
      data_.close();
    } catch (...) {
      if (close_error == nullptr) {
        close_error = std::current_exception();
      }
    }
    try {
      meta_.close();
    } catch (...) {
      if (close_error == nullptr) {
        close_error = std::current_exception();
      }
    }

    base_path_.clear();
    is_open_ = false;

    if (close_error != nullptr) {
      std::rethrow_exception(close_error);
    }
  }

  /**
   * @brief Save metadata changes to disk.
   */
  void save_meta() {
    if (is_open_) {
      meta_.save();
    }
  }

  // -------------------------------------------------------------------------
  // File access
  // -------------------------------------------------------------------------

  [[nodiscard]] auto meta() -> MetaFile & { return meta_; }
  [[nodiscard]] auto meta() const -> const MetaFile & { return meta_; }

  [[nodiscard]] auto data() -> DataFileType & { return data_; }
  [[nodiscard]] auto data() const -> const DataFileType & { return data_; }

  // -------------------------------------------------------------------------
  // Status
  // -------------------------------------------------------------------------

  [[nodiscard]] auto is_open() const -> bool { return is_open_; }
  [[nodiscard]] auto base_path() const -> const std::string & { return base_path_; }

  // -------------------------------------------------------------------------
  // Convenience accessors (delegate to meta file)
  // -------------------------------------------------------------------------

  [[nodiscard]] auto dimension() const -> uint32_t { return meta_.dimension(); }
  [[nodiscard]] auto capacity() const -> uint32_t { return meta_.capacity(); }
  [[nodiscard]] auto max_degree() const -> uint32_t { return meta_.max_degree(); }
  [[nodiscard]] auto entry_point() const -> uint32_t { return meta_.entry_point(); }
  [[nodiscard]] auto num_active() const -> uint64_t { return meta_.num_active_points(); }
  [[nodiscard]] auto num_points() const -> uint32_t {
    return static_cast<uint32_t>(meta_.num_active_points());
  }
  [[nodiscard]] auto metric_type() const -> uint32_t { return meta_.header().metric_type_; }

  // -------------------------------------------------------------------------
  // Entry point management
  // -------------------------------------------------------------------------

  void set_entry_point(uint32_t entry_id) { meta_.set_entry_point(entry_id); }
  void set_build_timestamp(uint64_t timestamp) { meta_.set_build_timestamp(timestamp); }
  void set_alpha(float alpha) { meta_.set_alpha(alpha); }

  // -------------------------------------------------------------------------
  // ID mapping (external_id <-> internal_slot)
  // -------------------------------------------------------------------------

  /**
   * @brief Resolve external ID to internal disk slot. O(1).
   */
  [[nodiscard]] auto resolve(IDType external_id) const -> uint32_t {
    return meta_.resolve(static_cast<uint32_t>(external_id));
  }

  /**
   * @brief Resolve internal slot to external ID. O(1).
   */
  [[nodiscard]] auto resolve_reverse(uint32_t internal_slot) const -> IDType {
    return static_cast<IDType>(meta_.resolve_reverse(internal_slot));
  }

  /**
   * @brief Check if an external ID has a valid mapping.
   */
  [[nodiscard]] auto has_mapping(IDType external_id) const -> bool {
    return meta_.has_mapping(static_cast<uint32_t>(external_id));
  }

  // -------------------------------------------------------------------------
  // Node operations (visitor pattern) — operate on internal slot IDs
  // -------------------------------------------------------------------------
  /**
   * @brief Get NodeRef
   */
  [[nodiscard]] auto get_node(uint32_t internal_slot) -> NodeRef {
    return data_.get_node(internal_slot);
  }

  // -------------------------------------------------------------------------
  // Node operations (external ID API)
  // -------------------------------------------------------------------------

  /**
   * @brief Check if an internal slot is valid (active).
   */
  [[nodiscard]] auto is_valid(uint32_t internal_slot) const -> bool {
    return meta_.is_valid(internal_slot);
  }

  /**
   * @brief Allocate a new internal slot.
   *
   * @return Internal slot ID, or -1 if full
   */
  [[nodiscard]] auto allocate_node_id() -> int32_t {
    if (meta_.is_full()) {
      data_.grow(meta_.next_growth_capacity());
    }
    return meta_.allocate_slot();
  }

 private:
  MetaFile meta_;
  std::string base_path_;
  DataFileType data_;

  bool is_open_{false};

  void cleanup_failed_create() {
    data_.close();
    meta_.close_without_save();

    std::error_code ec;
    std::filesystem::remove(meta_path(base_path_), ec);
    std::filesystem::remove(meta_path(base_path_) + ".tmp", ec);
    std::filesystem::remove(data_path(base_path_), ec);
  }
};

}  // namespace alaya
