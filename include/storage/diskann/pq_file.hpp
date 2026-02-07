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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>

#include "utils/math.hpp"
#include "utils/memory.hpp"

namespace alaya {

// ============================================================================
// Constants for PQFile
// ============================================================================

constexpr uint32_t kPQFileMagic = 0x50514441;   // "PQDA" in hex
constexpr uint32_t kPQFileVersion = 1;          // Current version
constexpr size_t kPQFileHeaderSize = 64;        // Header size in bytes
constexpr size_t kPQGrowthChunk = 67108864;     // 64MB growth chunk
constexpr uint32_t kDefaultNumCentroids = 256;  // Default K value

// ============================================================================
// PQFileHeader - Fixed-size header for PQ data file
// ============================================================================

/**
 * @brief PQ file header (64 bytes, cache-line aligned).
 *
 * Layout:
 * +------------------------+ Offset 0
 * | magic_ (4B)            | 0x50514441 ("PQDA")
 * | version_ (4B)          | Version number
 * +------------------------+ Offset 8
 * | num_subspaces_ (4B)    | M: Number of PQ subspaces
 * | num_centroids_ (4B)    | K: Centroids per subspace (typically 256)
 * +------------------------+ Offset 16
 * | subspace_dim_ (4B)     | D/M: Dimension per subspace
 * | dim_original_ (4B)     | Original vector dimension
 * +------------------------+ Offset 24
 * | num_vectors_ (8B)      | N: Current number of encoded vectors
 * +------------------------+ Offset 32
 * | file_capacity_ (8B)    | Pre-allocated file size for expansion
 * +------------------------+ Offset 40
 * | codebook_offset_ (8B)  | Offset to codebook data (= 64)
 * +------------------------+ Offset 48
 * | codes_offset_ (8B)     | Offset to PQ codes
 * +------------------------+ Offset 56
 * | codes_capacity_ (8B)   | Pre-allocated space for codes
 * +------------------------+ Offset 64 (End of header)
 */
struct alignas(64) PQFileHeader {
  uint32_t magic_{kPQFileMagic};
  uint32_t version_{kPQFileVersion};

  uint32_t num_subspaces_{0};    // M
  uint32_t num_centroids_{256};  // K

  uint32_t subspace_dim_{0};  // D/M
  uint32_t dim_original_{0};  // D

  uint64_t num_vectors_{0};  // N (current count)

  uint64_t file_capacity_{0};  // Total file capacity

  uint64_t codebook_offset_{kPQFileHeaderSize};
  uint64_t codes_offset_{0};
  uint64_t codes_capacity_{0};  // Pre-allocated codes space

  /**
   * @brief Initialize header with given parameters.
   *
   * @param dim Original vector dimension
   * @param num_subspaces Number of PQ subspaces (M)
   * @param initial_capacity Initial number of vectors to pre-allocate
   * @param num_centroids Number of centroids per subspace (K)
   */
  void init(uint32_t dim,
            uint32_t num_subspaces,
            uint64_t initial_capacity,
            uint32_t num_centroids = kDefaultNumCentroids) {
    magic_ = kPQFileMagic;
    version_ = kPQFileVersion;
    num_subspaces_ = num_subspaces;
    num_centroids_ = num_centroids;
    subspace_dim_ = dim / num_subspaces;
    dim_original_ = dim;
    num_vectors_ = 0;

    // Calculate codebook size: M * K * (D/M) * sizeof(float)
    size_t codebook_size =
        static_cast<size_t>(num_subspaces) * num_centroids * subspace_dim_ * sizeof(float);

    codebook_offset_ = kPQFileHeaderSize;
    codes_offset_ = codebook_offset_ + codebook_size;

    // Pre-allocate space for codes: capacity * M bytes
    codes_capacity_ = initial_capacity * num_subspaces;
    file_capacity_ = codes_offset_ + codes_capacity_;
  }

  /**
   * @brief Get codebook size in bytes.
   */
  [[nodiscard]] auto codebook_size() const -> size_t {
    return static_cast<size_t>(num_subspaces_) * num_centroids_ * subspace_dim_ * sizeof(float);
  }

  /**
   * @brief Get current codes size in bytes.
   */
  [[nodiscard]] auto codes_size() const -> size_t { return num_vectors_ * num_subspaces_; }

  /**
   * @brief Validate the header integrity.
   */
  [[nodiscard]] auto is_valid() const -> bool {
    if (magic_ != kPQFileMagic) {
      return false;
    }
    if (version_ > kPQFileVersion) {
      return false;
    }
    if (num_subspaces_ == 0 || num_centroids_ == 0 || subspace_dim_ == 0) {
      return false;
    }
    return true;
  }
};

static_assert(sizeof(PQFileHeader) == kPQFileHeaderSize, "PQFileHeader must be exactly 64 bytes");

// ============================================================================
// PQFile - Memory-mapped PQ file for fast access
// ============================================================================

/**
 * @brief Memory-mapped PQ file for fast access.
 *
 * Uses mmap for zero-copy access to codebook and PQ codes.
 * Supports dynamic expansion with chunk-based growth strategy.
 *
 * File Layout:
 * +------------------------+ Offset 0
 * | PQFileHeader (64B)     |
 * +------------------------+ Offset 64 (codebook_offset_)
 * | Codebook               | M * K * (D/M) * sizeof(float)
 * +------------------------+ Offset codes_offset_
 * | PQ Codes               | N * M bytes (with pre-allocated capacity)
 * +------------------------+
 */
class PQFile {
 public:
  PQFile() = default;
  ~PQFile() { close(); }

  // Non-copyable
  PQFile(const PQFile &) = delete;
  auto operator=(const PQFile &) -> PQFile & = delete;

  // Movable
  PQFile(PQFile &&other) noexcept
      : path_(std::move(other.path_)),
        fd_(other.fd_),
        header_(other.header_),
        mmap_addr_(other.mmap_addr_),
        mmap_size_(other.mmap_size_),
        codebook_(other.codebook_),
        codes_(other.codes_),
        is_open_(other.is_open_),
        is_writable_(other.is_writable_) {
    other.fd_ = -1;
    other.mmap_addr_ = nullptr;
    other.mmap_size_ = 0;
    other.codebook_ = nullptr;
    other.codes_ = nullptr;
    other.is_open_ = false;
  }

  auto operator=(PQFile &&other) noexcept -> PQFile & {
    if (this != &other) {
      close();
      path_ = std::move(other.path_);
      fd_ = other.fd_;
      header_ = other.header_;
      mmap_addr_ = other.mmap_addr_;
      mmap_size_ = other.mmap_size_;
      codebook_ = other.codebook_;
      codes_ = other.codes_;
      is_open_ = other.is_open_;
      is_writable_ = other.is_writable_;
      other.fd_ = -1;
      other.mmap_addr_ = nullptr;
      other.mmap_size_ = 0;
      other.codebook_ = nullptr;
      other.codes_ = nullptr;
      other.is_open_ = false;
    }
    return *this;
  }

  /**
   * @brief Create a new PQ file.
   *
   * @param path File path for the PQ file
   * @param dim Original vector dimension
   * @param num_subspaces Number of PQ subspaces (M)
   * @param initial_capacity Initial number of vectors to pre-allocate
   * @param num_centroids Number of centroids per subspace (K)
   */
  void create(std::string_view path,
              uint32_t dim,
              uint32_t num_subspaces,
              uint64_t initial_capacity,
              uint32_t num_centroids = kDefaultNumCentroids) {
    if (is_open_) {
      throw std::runtime_error("PQFile already open");
    }

    path_ = std::string(path);

    // Initialize header
    header_.init(dim, num_subspaces, initial_capacity, num_centroids);

    // Create and truncate file
    fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to create PQ file: " + path_);
    }

    // Extend file to capacity
    if (ftruncate(fd_, static_cast<off_t>(header_.file_capacity_)) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to extend PQ file");
    }

    // Write header
    if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to write PQ file header");
    }

    // Memory map
    mmap_size_ = header_.file_capacity_;
    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      mmap_addr_ = nullptr;
      throw std::runtime_error("Failed to mmap PQ file");
    }

    setup_mmap_pointers();
    is_open_ = true;
    is_writable_ = true;
  }

  /**
   * @brief Open an existing PQ file.
   *
   * @param path File path to the PQ file
   * @param writable Whether to open for writing
   */
  void open(std::string_view path, bool writable = false) {
    if (is_open_) {
      throw std::runtime_error("PQFile already open");
    }

    path_ = std::string(path);
    is_writable_ = writable;

    // Open file
    int flags = writable ? O_RDWR : O_RDONLY;
    fd_ = ::open(path_.c_str(), flags);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open PQ file: " + path_);
    }

    // Get file size
    struct stat st{};
    if (fstat(fd_, &st) != 0) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to stat PQ file");
    }

    // Read header
    if (::pread(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Failed to read PQ file header");
    }

    // Validate header
    if (!header_.is_valid()) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Invalid PQ file header");
    }

    // Memory map
    mmap_size_ = static_cast<size_t>(st.st_size);
    int prot = PROT_READ | (writable ? PROT_WRITE : 0);
    mmap_addr_ = mmap(nullptr, mmap_size_, prot, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
      ::close(fd_);
      fd_ = -1;
      mmap_addr_ = nullptr;
      throw std::runtime_error("Failed to mmap PQ file");
    }

    setup_mmap_pointers();
    is_open_ = true;
  }

  /**
   * @brief Close the PQ file.
   */
  void close() {
    if (!is_open_) {
      return;
    }

    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
      if (is_writable_) {
        // Write header to mmap region, then sync everything to disk
        std::memcpy(mmap_addr_, &header_, sizeof(header_));
        msync(mmap_addr_, mmap_size_, MS_SYNC);
      }
      munmap(mmap_addr_, mmap_size_);
      mmap_addr_ = nullptr;
    }

    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }

    path_.clear();
    codebook_ = nullptr;
    codes_ = nullptr;
    mmap_size_ = 0;
    is_open_ = false;
    is_writable_ = false;
  }

  // -------------------------------------------------------------------------
  // Write operations (during build)
  // -------------------------------------------------------------------------

  /**
   * @brief Write codebook data.
   *
   * @param codebook_data Pointer to codebook data (M * K * D/M floats)
   */
  void write_codebook(const float *codebook_data) {
    if (!is_open_ || !is_writable_) {
      throw std::runtime_error("PQ file not open for writing");
    }

    size_t codebook_size = header_.codebook_size();
    std::memcpy(codebook_, codebook_data, codebook_size);
  }

  /**
   * @brief Write PQ codes for vectors.
   *
   * @param codes Pointer to PQ codes (N * M bytes)
   * @param num_vectors Number of vectors
   */
  void write_codes(const uint8_t *codes, uint64_t num_vectors) {
    if (!is_open_ || !is_writable_) {
      throw std::runtime_error("PQ file not open for writing");
    }

    ensure_capacity(num_vectors);

    size_t codes_size = num_vectors * header_.num_subspaces_;
    std::memcpy(codes_, codes, codes_size);
    header_.num_vectors_ = num_vectors;

    // Update header in mmap
    std::memcpy(mmap_addr_, &header_, sizeof(header_));
  }

  /**
   * @brief Append a single PQ code.
   *
   * @param code Pointer to PQ code (M bytes)
   * @return Index of the appended code
   */
  auto append_code(const uint8_t *code) -> uint64_t {
    if (!is_open_ || !is_writable_) {
      throw std::runtime_error("PQ file not open for writing");
    }

    ensure_capacity(header_.num_vectors_ + 1);

    uint64_t index = header_.num_vectors_;
    std::memcpy(codes_ + index * header_.num_subspaces_, code, header_.num_subspaces_);
    ++header_.num_vectors_;

    // Update header in mmap
    std::memcpy(mmap_addr_, &header_, sizeof(header_));

    return index;
  }

  /**
   * @brief Update a PQ code at given index.
   *
   * @param index Vector index
   * @param code Pointer to new PQ code (M bytes)
   */
  void update_code(uint64_t index, const uint8_t *code) {
    if (!is_open_ || !is_writable_) {
      throw std::runtime_error("PQ file not open for writing");
    }

    if (index >= header_.num_vectors_) {
      throw std::out_of_range("Code index out of range");
    }

    std::memcpy(codes_ + index * header_.num_subspaces_, code, header_.num_subspaces_);
  }

  // -------------------------------------------------------------------------
  // Read operations (mmap-based)
  // -------------------------------------------------------------------------

  /**
   * @brief Get pointer to the entire codebook.
   *
   * Layout: [subspace_0: K centroids][subspace_1: K centroids]...
   * Each centroid has D/M floats.
   */
  [[nodiscard]] auto get_codebook() const -> const float * { return codebook_; }

  /**
   * @brief Get pointer to a specific centroid.
   *
   * @param subspace Subspace index (0 to M-1)
   * @param centroid_id Centroid index (0 to K-1)
   * @return Pointer to the centroid (D/M floats)
   */
  [[nodiscard]] auto get_centroid(uint32_t subspace, uint32_t centroid_id) const -> const float * {
    size_t offset = static_cast<size_t>(subspace) * header_.num_centroids_ * header_.subspace_dim_ +
                    static_cast<size_t>(centroid_id) * header_.subspace_dim_;
    return codebook_ + offset;
  }

  /**
   * @brief Get pointer to a PQ code for a vector.
   *
   * @param vector_id Vector index
   * @return Pointer to the PQ code (M bytes)
   */
  [[nodiscard]] auto get_code(uint64_t vector_id) const -> const uint8_t * {
    if (vector_id >= header_.num_vectors_) {
      throw std::out_of_range("Vector ID out of range");
    }
    return codes_ + vector_id * header_.num_subspaces_;
  }

  // -------------------------------------------------------------------------
  // ADC (Asymmetric Distance Computation) helpers
  // -------------------------------------------------------------------------

  /**
   * @brief Compute ADC lookup table for a query vector.
   *
   * The ADC table stores distances from query subvectors to all centroids.
   * Table layout: [subspace_0: K distances][subspace_1: K distances]...
   *
   * @param query Query vector (D floats)
   * @param adc_table Output ADC table (M * K floats)
   */
  void compute_adc_table(const float *query, float *adc_table) const {
    for (uint32_t m = 0; m < header_.num_subspaces_; ++m) {
      const float *query_subvec = query + m * header_.subspace_dim_;
      float *table_row = adc_table + m * header_.num_centroids_;

      for (uint32_t k = 0; k < header_.num_centroids_; ++k) {
        const float *centroid = get_centroid(m, k);
        float dist = 0.0F;
        for (uint32_t d = 0; d < header_.subspace_dim_; ++d) {
          float diff = query_subvec[d] - centroid[d];
          dist += diff * diff;
        }
        table_row[k] = dist;
      }
    }
  }

  /**
   * @brief Compute approximate distance using ADC table and PQ code.
   *
   * @param adc_table Pre-computed ADC table (M * K floats)
   * @param code PQ code (M bytes)
   * @return Approximate squared L2 distance
   */
  [[nodiscard]] auto compute_distance(const float *adc_table, const uint8_t *code) const -> float {
    float dist = 0.0F;
    for (uint32_t m = 0; m < header_.num_subspaces_; ++m) {
      dist += adc_table[m * header_.num_centroids_ + code[m]];
    }
    return dist;
  }

  /**
   * @brief Compute approximate distance for a vector ID.
   *
   * @param adc_table Pre-computed ADC table
   * @param vector_id Vector index
   * @return Approximate squared L2 distance
   */
  [[nodiscard]] auto compute_distance(const float *adc_table, uint64_t vector_id) const -> float {
    return compute_distance(adc_table, get_code(vector_id));
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  [[nodiscard]] auto header() const -> const PQFileHeader & { return header_; }
  [[nodiscard]] auto num_subspaces() const -> uint32_t { return header_.num_subspaces_; }
  [[nodiscard]] auto num_centroids() const -> uint32_t { return header_.num_centroids_; }
  [[nodiscard]] auto subspace_dim() const -> uint32_t { return header_.subspace_dim_; }
  [[nodiscard]] auto num_vectors() const -> uint64_t { return header_.num_vectors_; }
  [[nodiscard]] auto code_size() const -> uint32_t { return header_.num_subspaces_; }

  [[nodiscard]] auto is_open() const -> bool { return is_open_; }
  [[nodiscard]] auto is_writable() const -> bool { return is_writable_; }
  [[nodiscard]] auto path() const -> const std::string & { return path_; }

 private:
  /**
   * @brief Ensure file has capacity for given number of codes.
   *
   * Implements chunk-based growth strategy to avoid frequent remapping.
   */
  void ensure_capacity(uint64_t new_codes_count) {
    size_t needed = header_.codes_offset_ + new_codes_count * header_.num_subspaces_;
    if (needed <= header_.file_capacity_) {
      return;
    }

    // Chunk growth strategy: max(old * 1.5, old + 64MB)
    size_t new_capacity =
        std::max(header_.file_capacity_ * 3 / 2, header_.file_capacity_ + kPQGrowthChunk);
    new_capacity = std::max(new_capacity, needed);

    // Extend file
    if (ftruncate(fd_, static_cast<off_t>(new_capacity)) != 0) {
      throw std::runtime_error("Failed to extend PQ file");
    }

    // Unmap old mapping
    if (mmap_addr_ != nullptr && mmap_addr_ != MAP_FAILED) {
      munmap(mmap_addr_, mmap_size_);
    }

    // Create new mapping
    mmap_size_ = new_capacity;
    mmap_addr_ = mmap(nullptr, mmap_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (mmap_addr_ == MAP_FAILED) {
      mmap_addr_ = nullptr;
      throw std::runtime_error("Failed to remap PQ file");
    }

    // Update header
    header_.file_capacity_ = new_capacity;
    header_.codes_capacity_ = new_capacity - header_.codes_offset_;

    // Re-establish pointers
    setup_mmap_pointers();

    // Update header in mmap
    std::memcpy(mmap_addr_, &header_, sizeof(header_));
  }

  /**
   * @brief Setup pointers into mmap region.
   */
  void setup_mmap_pointers() {
    auto *base = static_cast<uint8_t *>(mmap_addr_);
    codebook_ = reinterpret_cast<float *>(base + header_.codebook_offset_);
    codes_ = base + header_.codes_offset_;
  }

  std::string path_;
  int fd_{-1};
  PQFileHeader header_;

  // Memory mapping
  void *mmap_addr_{nullptr};
  size_t mmap_size_{0};

  // Cached pointers into mmap region
  float *codebook_{nullptr};
  uint8_t *codes_{nullptr};

  bool is_open_{false};
  bool is_writable_{false};
};

}  // namespace alaya
