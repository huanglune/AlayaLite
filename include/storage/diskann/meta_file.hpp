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

#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include "utils/bitset.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/math.hpp"

namespace alaya {

// ============================================================================
// Constants for MetaFile
// ============================================================================

constexpr uint32_t kMetaFileMagic = 0x4D455441;          // "META" in hex
constexpr uint32_t kMetaFileVersion = 1;                 // Current version
constexpr size_t kMetaFileHeaderSize = 128;              // Header size in bytes
constexpr uint64_t kGrowthChunkSize = 64 * 1024 * 1024;  // 64MB growth chunk
constexpr uint32_t kInvalidMapping = UINT32_MAX;         // Sentinel for unmapped IDs

// ============================================================================
// FreeList - Stack-based free slot tracker for O(1) allocation/deallocation
// ============================================================================

/**
 * @brief Stack-based free slot tracker.
 *
 * Provides O(1) allocation and deallocation by maintaining a stack of free slot IDs.
 * When a slot is freed, its ID is pushed onto the stack.
 * When allocating, we pop from the stack if available.
 *
 * File format:
 * - uint32_t count: number of free slots
 * - uint32_t slots[count]: array of free slot IDs
 */
class FreeList {
 public:
  FreeList() = default;

  void push(uint32_t slot_id) { free_slots_.push_back(slot_id); }
  [[nodiscard]] auto pop() -> int32_t {
    if (free_slots_.empty()) {
      return -1;
    }
    auto slot = static_cast<int32_t>(free_slots_.back());
    free_slots_.pop_back();
    return slot;
  }
  [[nodiscard]] auto empty() const noexcept -> bool { return free_slots_.empty(); }
  [[nodiscard]] auto size() const noexcept -> size_t { return free_slots_.size(); }
  void clear() { free_slots_.clear(); }
  [[nodiscard]] auto data() noexcept -> uint32_t * { return free_slots_.data(); }
  [[nodiscard]] auto data() const noexcept -> const uint32_t * { return free_slots_.data(); }
  void resize(size_t count) { free_slots_.resize(count); }
  [[nodiscard]] auto size_in_bytes() const noexcept -> size_t {
    return sizeof(uint32_t) + free_slots_.size() * sizeof(uint32_t);
  }

 private:
  std::vector<uint32_t> free_slots_;
};

// ============================================================================
// MetaFileHeader - Fixed-size header for metadata file
// ============================================================================

/**
 * @brief Fixed header for metadata file (128 bytes, cache-line aligned).
 *
 * Layout:
 * +------------------------+ Offset 0
 * | magic_ (4B)            | 0x4D455441 ("META")
 * | version_ (4B)          | Version number
 * +------------------------+ Offset 8
 * | num_capacity_ (4B)     | Maximum number of vectors
 * | num_frozen_points_(4B) | Number of frozen (committed) points
 * +------------------------+ Offset 16
 * | entry_point_id_ (4B)   | Entry point (medoid) for search
 * | max_degree_ (4B)       | Maximum out-degree (R)
 * +------------------------+ Offset 24
 * | dim_ (4B)              | Vector dimension
 * | dim_aligned_ (4B)      | Aligned dimension (for SIMD, 64B multiple)
 * +------------------------+ Offset 32
 * | data_type_ (4B)        | Data type enum (float32, int8, etc.)
 * | metric_type_ (4B)      | Metric type enum (L2, IP, COS)
 * +------------------------+ Offset 40
 * | alpha_ (4B float)      | Pruning alpha parameter
 * | reserved1_ (4B)        | Reserved for future use
 * +------------------------+ Offset 48
 * | build_timestamp_ (8B)  | Unix timestamp of index build
 * +------------------------+ Offset 56
 * | bitmap_offset_ (8B)    | Offset to validity bitmap (= 128)
 * +------------------------+ Offset 64
 * | freelist_offset_ (8B)  | Offset to optional freelist (0 if none)
 * +------------------------+ Offset 72
 * | freelist_size_ (8B)    | Size of freelist section
 * +------------------------+ Offset 80
 * | num_active_points_(8B) | Current number of active (non-deleted) points
 * +------------------------+ Offset 88
 * | delete_watermark_ (8B) | Highest deleted node_id + 1 (rebuild scan boundary)
 * +------------------------+ Offset 96
 * | idmap_offset_ (8B)     | Offset to external→internal ID mapping table
 * +------------------------+ Offset 104
 * | idmap_count_ (8B)      | Number of entries in the ID mapping
 * +------------------------+ Offset 112
 * | checksum_ (8B)         | CRC64 checksum of header (excluding this field)
 * +------------------------+ Offset 120
 * | padding_[8]            | Reserved for future expansion
 * +------------------------+ Offset 128
 */
struct alignas(64) MetaFileHeader {
  uint32_t magic_{kMetaFileMagic};
  uint32_t version_{kMetaFileVersion};

  uint32_t num_capacity_{0};
  uint32_t num_frozen_points_{0};

  uint32_t entry_point_id_{0};
  uint32_t max_degree_{0};

  uint32_t dim_{0};
  uint32_t dim_aligned_{0};

  uint32_t data_type_{0};    // DiskDataType enum
  uint32_t metric_type_{0};  // MetricType enum

  float alpha_{1.2F};
  uint32_t reserved1_{0};

  uint64_t build_timestamp_{0};

  uint64_t bitmap_offset_{kMetaFileHeaderSize};  // After header

  uint64_t freelist_offset_{0};
  uint64_t freelist_size_{0};

  uint64_t num_active_points_{0};
  uint64_t delete_watermark_{0};

  uint64_t idmap_offset_{0};  // Offset to external→internal ID mapping table
  uint64_t idmap_count_{0};   // Number of entries in the ID mapping

  uint64_t checksum_{0};

  uint8_t padding_[8]{};  // Pad to 128 bytes

  /**
   * @brief Initialize header with given parameters.
   *
   * @param capacity Maximum number of vectors
   * @param dim Vector dimension
   * @param max_degree Maximum out-degree (R)
   * @param metric_type Metric type (L2, IP, COS)
   * @param data_type Data type (float32, int8, etc.)
   */
  void init(uint32_t capacity,
            uint32_t dim,
            uint32_t max_degree,
            uint32_t metric_type = 0,
            uint32_t data_type = 0) {
    magic_ = kMetaFileMagic;
    version_ = kMetaFileVersion;
    num_capacity_ = capacity;
    num_frozen_points_ = 0;
    entry_point_id_ = 0;
    max_degree_ = max_degree;
    dim_ = dim;
    // Round up to 64-byte boundary for SIMD alignment, then convert back to element count
    dim_aligned_ = math::round_up_pow2(dim * sizeof(float), 64) / sizeof(float);
    data_type_ = data_type;
    metric_type_ = metric_type;
    alpha_ = 1.2F;
    build_timestamp_ = 0;
    bitmap_offset_ = kMetaFileHeaderSize;
    freelist_offset_ = 0;  // Freelist is not persisted (rebuilt from bitmap on load)
    freelist_size_ = 0;
    num_active_points_ = 0;
    delete_watermark_ = 0;
    idmap_offset_ = 0;
    idmap_count_ = 0;
    checksum_ = 0;
  }

  [[nodiscard]] auto bitset_size_bytes() const -> size_t {
    return static_cast<size_t>(num_capacity_ / 8);
  }

  /**
   * @brief Compute CRC64 checksum of header (excluding checksum field).
   * @return Computed checksum value
   */
  [[nodiscard]] auto compute_checksum() const -> uint64_t {
    // Simple XOR-based checksum (can be replaced with CRC64)
    uint64_t sum = 0;
    const auto *data = reinterpret_cast<const uint64_t *>(this);
    // Exclude last 24 bytes (checksum_ + padding_)
    for (size_t i = 0;
         i < (kMetaFileHeaderSize - sizeof(padding_) - sizeof(checksum_)) / sizeof(uint64_t);
         ++i) {
      sum ^= data[i];
      sum = (sum << 7) | (sum >> 57);  // Rotate
    }
    return sum;
  }

  void update_checksum() { checksum_ = compute_checksum(); }
  [[nodiscard]] auto verify_checksum() const -> bool { return checksum_ == compute_checksum(); }

  /**
   * @brief Validate the header integrity.
   * @return true if header is valid, false otherwise
   */
  [[nodiscard]] auto is_valid() const -> bool {
    if (magic_ != kMetaFileMagic) {
      return false;
    }
    if (version_ > kMetaFileVersion) {
      return false;
    }
    if (dim_ == 0 || num_capacity_ == 0) {
      return false;
    }
    return true;
  }
};

static_assert(sizeof(MetaFileHeader) == kMetaFileHeaderSize,
              "MetaFileHeader must be exactly 128 bytes");

// ============================================================================
// MetaFile - In-memory representation of the metadata file
// ============================================================================

/**
 * @brief In-memory representation of the metadata file.
 *
 * Manages:
 * - Header metadata
 * - Validity bitmap (which node IDs are active)
 * - Optional freelist for O(1) slot allocation
 *
 * The metadata file is designed to be fully loaded into memory for fast access.
 * It supports dynamic insert/delete operations through the validity bitmap.
 */
class MetaFile {
 public:
  MetaFile() = default;
  ~MetaFile() { close(); }

  // Non-copyable
  ALAYA_NON_COPYABLE(MetaFile);

  // Movable
  MetaFile(MetaFile &&other) noexcept
      : path_(std::move(other.path_)),
        header_(other.header_),
        validity_bitmap_(std::move(other.validity_bitmap_)),
        freelist_(std::move(other.freelist_)),
        id_map_(std::move(other.id_map_)),
        reverse_map_(std::move(other.reverse_map_)),
        dirty_(other.dirty_),
        is_open_(other.is_open_) {
    other.is_open_ = false;
    other.dirty_ = false;
  }

  auto operator=(MetaFile &&other) noexcept -> MetaFile & {
    if (this != &other) {
      close();
      path_ = std::move(other.path_);
      header_ = other.header_;
      validity_bitmap_ = std::move(other.validity_bitmap_);
      freelist_ = std::move(other.freelist_);
      id_map_ = std::move(other.id_map_);
      reverse_map_ = std::move(other.reverse_map_);
      dirty_ = other.dirty_;
      is_open_ = other.is_open_;
      other.is_open_ = false;
      other.dirty_ = false;
    }
    return *this;
  }

  /**
   * @brief Create a new metadata file.
   *
   * @param path File path for the metadata file
   * @param capacity Maximum number of vectors
   * @param dim Vector dimension
   * @param max_degree Maximum out-degree (R)
   * @param metric_type Metric type enum
   * @param data_type Data type enum
   */
  void create(std::string_view path,
              uint32_t capacity,
              uint32_t dim,
              uint32_t max_degree,
              uint32_t metric_type = 0,
              uint32_t data_type = 0) {
    if (is_open_) {
      throw std::runtime_error("MetaFile already open");
    }

    path_ = std::string(path);
    capacity = math::round_up_pow2(capacity, 64);  // Round up for Bitset 64-bit word alignment

    // Initialize header
    header_.init(capacity, dim, max_degree, metric_type, data_type);

    // Initialize validity bitmap (all zeros = no valid nodes)
    validity_bitmap_.resize(capacity);
    validity_bitmap_.reset();
    freelist_.clear();

    // Initialize ID mapping vectors (all unmapped)
    id_map_.assign(capacity, kInvalidMapping);
    reverse_map_.assign(capacity, kInvalidMapping);

    dirty_ = true;
    is_open_ = true;

    // Save to disk
    save();
  }

  /**
   * @brief Open an existing metadata file.
   *
   * @param path File path to the metadata file
   */
  void open(std::string_view path) {
    if (is_open_) {
      throw std::runtime_error("MetaFile already open");
    }

    path_ = std::string(path);

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open metadata file: " + path_);
    }

    // Read header
    file.read(reinterpret_cast<char *>(&header_), sizeof(MetaFileHeader));
    if (!file) {
      throw std::runtime_error("Failed to read metadata header");
    }

    // Validate header
    if (!header_.is_valid()) {
      throw std::runtime_error("Invalid metadata header");
    }

    if (!header_.verify_checksum()) {
      throw std::runtime_error("Metadata header checksum mismatch");
    }

    // Read bitmap data directly into bitset
    validity_bitmap_.resize(header_.num_capacity_);
    file.seekg(static_cast<std::streamoff>(header_.bitmap_offset_));
    file.read(reinterpret_cast<char *>(validity_bitmap_.data()),
              static_cast<std::streamsize>(header_.bitset_size_bytes()));

    // Rebuild freelist from bitmap (not persisted to disk)
    rebuild_freelist();

    // Read ID mapping if present
    if (header_.idmap_offset_ > 0 && header_.idmap_count_ > 0) {
      auto map_size = header_.idmap_count_;
      id_map_.resize(map_size);
      reverse_map_.resize(map_size);

      file.seekg(static_cast<std::streamoff>(header_.idmap_offset_));
      file.read(reinterpret_cast<char *>(id_map_.data()),
                static_cast<std::streamsize>(map_size * sizeof(uint32_t)));
      file.read(reinterpret_cast<char *>(reverse_map_.data()),
                static_cast<std::streamsize>(map_size * sizeof(uint32_t)));
    } else {
      // No mapping on disk — initialize empty maps
      id_map_.assign(header_.num_capacity_, kInvalidMapping);
      reverse_map_.assign(header_.num_capacity_, kInvalidMapping);
    }

    dirty_ = false;
    is_open_ = true;
  }

  /**
   * @brief Save metadata to disk.
   *
   * Performs atomic save using temporary file + rename.
   */
  void save() {
    if (!is_open_) {
      throw std::runtime_error("MetaFile not open");
    }

    // Freelist is not persisted — rebuilt from bitmap on load
    header_.freelist_offset_ = 0;
    header_.freelist_size_ = 0;

    // Compute ID map offset (after bitmap)
    header_.idmap_offset_ = header_.bitmap_offset_ + header_.bitset_size_bytes();
    header_.idmap_count_ = static_cast<uint64_t>(id_map_.size());

    std::string temp_path = path_ + ".tmp";

    {
      std::ofstream file(temp_path, std::ios::binary | std::ios::trunc);
      if (!file) {
        throw std::runtime_error("Failed to create temporary metadata file");
      }

      // Write header with updated checksum
      MetaFileHeader header_copy = header_;
      header_copy.update_checksum();
      file.write(reinterpret_cast<const char *>(&header_copy), sizeof(MetaFileHeader));

      // Write validity bitmap directly from bitset
      file.write(reinterpret_cast<const char *>(validity_bitmap_.data()),
                 static_cast<std::streamsize>(header_.bitset_size_bytes()));

      // Write ID mapping vectors
      file.write(reinterpret_cast<const char *>(id_map_.data()),
                 static_cast<std::streamsize>(id_map_.size() * sizeof(uint32_t)));
      file.write(reinterpret_cast<const char *>(reverse_map_.data()),
                 static_cast<std::streamsize>(reverse_map_.size() * sizeof(uint32_t)));

      file.flush();
      // Note: For production, should call fsync() here
    }

    // Atomic rename
    if (std::rename(temp_path.c_str(), path_.c_str()) != 0) {
      throw std::runtime_error("Failed to rename temporary metadata file");
    }

    dirty_ = false;
  }

  /**
   * @brief Close the metadata file.
   *
   * Saves if dirty before closing.
   */
  void close() {
    if (!is_open_) {
      return;
    }

    if (dirty_) {
      save();
    }

    path_.clear();
    validity_bitmap_ = Bitset();
    freelist_.clear();
    id_map_.clear();
    reverse_map_.clear();
    header_ = MetaFileHeader();
    dirty_ = false;
    is_open_ = false;
  }

  // -------------------------------------------------------------------------
  // Node validity management
  // -------------------------------------------------------------------------

  /**
   * @brief Check if a node ID is valid (active).
   *
   * @param node_id Node ID to check
   * @return true if the node is valid, false otherwise
   */
  [[nodiscard]] auto is_valid(uint32_t node_id) const -> bool {
    if (node_id >= header_.num_capacity_) {
      return false;
    }
    return validity_bitmap_.get(node_id);
  }

  /**
   * @brief Mark a node ID as valid (active).
   *
   * @param node_id Node ID to mark as valid
   */
  void set_valid(uint32_t node_id) {
    if (node_id >= header_.num_capacity_) {
      throw std::out_of_range("Node ID out of range");
    }
    if (!validity_bitmap_.get(node_id)) {
      validity_bitmap_.set(node_id);
      ++header_.num_active_points_;
      dirty_ = true;
    }
  }

  /**
   * @brief Free a slot (alias for set_invalid).
   *
   * @param node_id Node ID to free
   */
  void free_slot(uint32_t node_id) { set_invalid(node_id); }

  /**
   * @brief Mark a node ID as invalid (deleted).
   *
   * @param node_id Node ID to mark as invalid
   */
  void set_invalid(uint32_t node_id) {
    if (node_id >= header_.num_capacity_) {
      throw std::out_of_range("Node ID out of range");
    }
    if (validity_bitmap_.get(node_id)) {
      validity_bitmap_.reset(node_id);
      --header_.num_active_points_;
      // Add to freelist for O(1) reallocation
      freelist_.push(node_id);
      // Track farthest deletion for rebuild_freelist scan boundary
      header_.delete_watermark_ =
          std::max(static_cast<uint64_t>(node_id) + 1, header_.delete_watermark_);
      dirty_ = true;
    }
  }

  // -------------------------------------------------------------------------
  // Slot allocation
  // -------------------------------------------------------------------------

  /**
   * @brief Allocate a free slot and return its ID.
   *
   * Uses freelist for O(1) allocation when slots have been freed.
   * Falls back to bitmap scan, then auto-grows if full.
   *
   * @return Allocated node ID
   */
  [[nodiscard]] auto allocate_slot() -> int32_t {
    // Try freelist first (O(1) for previously freed slots)
    int32_t slot = freelist_.pop();
    if (slot >= 0) {
      validity_bitmap_.set(static_cast<size_t>(slot));
      ++header_.num_active_points_;
      dirty_ = true;
      return slot;
    }

    // Fall back to bitmap scan
    int found = validity_bitmap_.find_first_zero();
    if (found >= 0 && static_cast<uint32_t>(found) < header_.num_capacity_) {
      validity_bitmap_.set(static_cast<size_t>(found));
      ++header_.num_active_points_;
      dirty_ = true;
      return found;
    }

    // No free slots — grow capacity dynamically
    uint32_t old_capacity = header_.num_capacity_;
    uint32_t new_capacity = old_capacity + std::max(old_capacity / 2, uint32_t{1024});
    LOG_INFO("MetaFile: Auto-growing capacity from {} to {}", old_capacity, new_capacity);
    grow(new_capacity);

    validity_bitmap_.set(old_capacity);
    ++header_.num_active_points_;
    dirty_ = true;
    return static_cast<int32_t>(old_capacity);
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  [[nodiscard]] auto header() -> MetaFileHeader & { return header_; }
  [[nodiscard]] auto header() const -> const MetaFileHeader & { return header_; }

  [[nodiscard]] auto validity_bitmap() -> Bitset & { return validity_bitmap_; }
  [[nodiscard]] auto validity_bitmap() const -> const Bitset & { return validity_bitmap_; }

  [[nodiscard]] auto num_active_points() const -> uint64_t { return header_.num_active_points_; }
  [[nodiscard]] auto capacity() const -> uint32_t { return header_.num_capacity_; }
  [[nodiscard]] auto dimension() const -> uint32_t { return header_.dim_; }
  [[nodiscard]] auto max_degree() const -> uint32_t { return header_.max_degree_; }
  [[nodiscard]] auto entry_point() const -> uint32_t { return header_.entry_point_id_; }

  [[nodiscard]] auto is_open() const -> bool { return is_open_; }
  [[nodiscard]] auto is_dirty() const -> bool { return dirty_; }
  [[nodiscard]] auto path() const -> const std::string & { return path_; }

  void set_entry_point(uint32_t entry_id) {
    header_.entry_point_id_ = entry_id;
    dirty_ = true;
  }

  void set_build_timestamp(uint64_t timestamp) {
    header_.build_timestamp_ = timestamp;
    dirty_ = true;
  }

  void set_alpha(float alpha) {
    header_.alpha_ = alpha;
    dirty_ = true;
  }

  // -------------------------------------------------------------------------
  // ID mapping (external_id <-> internal_slot)
  // -------------------------------------------------------------------------

  /**
   * @brief Insert a mapping from external ID to internal slot.
   *
   * @param external_id User-specified external ID
   * @param internal_slot Allocated disk slot
   */
  void insert_mapping(uint32_t external_id, uint32_t internal_slot) {
    if (external_id >= id_map_.size()) {
      throw std::out_of_range("External ID out of range");
    }
    if (internal_slot >= reverse_map_.size()) {
      throw std::out_of_range("Internal slot out of range");
    }
    id_map_[external_id] = internal_slot;
    reverse_map_[internal_slot] = external_id;
    dirty_ = true;
  }

  /**
   * @brief Remove a mapping by external ID.
   *
   * @param external_id External ID to unmap
   */
  void remove_mapping(uint32_t external_id) {
    if (external_id >= id_map_.size()) {
      return;
    }
    uint32_t internal_slot = id_map_[external_id];
    if (internal_slot != kInvalidMapping) {
      reverse_map_[internal_slot] = kInvalidMapping;
    }
    id_map_[external_id] = kInvalidMapping;
    dirty_ = true;
  }

  /**
   * @brief Resolve external ID to internal disk slot. O(1).
   *
   * @param external_id External ID to resolve
   * @return Internal slot, or kInvalidMapping if unmapped
   */
  [[nodiscard]] auto resolve(uint32_t external_id) const -> uint32_t {
    if (external_id >= id_map_.size()) {
      return kInvalidMapping;
    }
    return id_map_[external_id];
  }

  /**
   * @brief Resolve internal slot to external ID. O(1).
   *
   * @param internal_slot Internal slot to resolve
   * @return External ID, or kInvalidMapping if unmapped
   */
  [[nodiscard]] auto resolve_reverse(uint32_t internal_slot) const -> uint32_t {
    if (internal_slot >= reverse_map_.size()) {
      return kInvalidMapping;
    }
    return reverse_map_[internal_slot];
  }

  /**
   * @brief Check if an external ID has a mapping.
   */
  [[nodiscard]] auto has_mapping(uint32_t external_id) const -> bool {
    return external_id < id_map_.size() && id_map_[external_id] != kInvalidMapping;
  }

  /**
   * @brief Grow the capacity of the metadata file.
   *
   * Expands the validity bitmap and updates the header.
   * New slots are initialized as invalid (free).
   * Note: Callers are responsible for growing corresponding data/graph files.
   *
   * @param new_capacity The new capacity (must be > current capacity)
   */
  void grow(uint32_t new_capacity) {
    new_capacity = math::round_up_pow2(new_capacity, 64);  // maintain 64-alignment
    if (new_capacity <= header_.num_capacity_) {
      return;
    }
    header_.num_capacity_ = new_capacity;
    validity_bitmap_.resize(new_capacity);
    id_map_.resize(new_capacity, kInvalidMapping);
    reverse_map_.resize(new_capacity, kInvalidMapping);
    dirty_ = true;
  }

 private:
  /**
   * @brief Rebuild freelist from the validity bitmap.
   *
   * Scans [0, delete_watermark_) in the bitmap to find deleted slots
   * (unset bits below the deletion watermark) and adds them to the freelist
   * for O(1) reuse. Called on open() instead of reading freelist from disk.
   * TODO(hl): Optimize when needed
   */
  void rebuild_freelist() {
    freelist_.clear();
    for (uint32_t i = 0; i < static_cast<uint32_t>(header_.delete_watermark_); ++i) {
      if (!validity_bitmap_.get(i)) {
        freelist_.push(i);
      }
    }
  }

  std::string path_;
  MetaFileHeader header_;
  Bitset validity_bitmap_;
  FreeList freelist_;
  std::vector<uint32_t>
      id_map_;  // id_map_[external_id] = internal_slot (kInvalidMapping if unmapped)
  std::vector<uint32_t>
      reverse_map_;  // reverse_map_[internal_slot] = external_id (kInvalidMapping if unmapped)
  mutable bool dirty_{false};
  bool is_open_{false};
};

}  // namespace alaya
