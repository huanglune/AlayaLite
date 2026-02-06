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
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "space/quant/pq.hpp"
#include "space/space_concepts.hpp"
#include "storage/data/sequential_storage.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"
#include "utils/math.hpp"
#include "utils/platform.hpp"
#include "utils/prefetch.hpp"
#include "utils/types.hpp"

namespace alaya {

/**
 * @brief PQ-based vector space for memory-efficient approximate distance computation.
 *
 * PQSpace stores vectors as PQ codes (M bytes per vector) and provides fast
 * approximate distance computation using ADC (Asymmetric Distance Computation).
 * This is ideal for DiskANN's in-memory navigation phase where we need to
 * compute many approximate distances quickly.
 *
 * Usage pattern for DiskANN:
 * 1. Build phase: Use RawSpace for exact distances
 * 2. Search phase: Use PQSpace for fast approximate navigation
 *
 * @tparam DataType The data type of original vectors (e.g., float)
 * @tparam DistanceType The distance type (default: float)
 * @tparam IDType The ID type (default: uint32_t)
 */
template <typename DataType = float, typename DistanceType = float, typename IDType = uint32_t>
class PQSpace {
 public:
  using DataTypeAlias = DataType;
  using IDTypeAlias = IDType;
  using DistanceTypeAlias = DistanceType;

  // For Space concept compatibility - PQ codes are uint8_t
  using DistDataType = uint8_t;

 private:
  IDType capacity_{0};                 ///< Maximum number of vectors
  uint32_t dim_{0};                    ///< Original vector dimension
  uint32_t num_subspaces_{0};          ///< Number of PQ subspaces (M)
  MetricType metric_{MetricType::L2};  ///< Metric type (currently only L2 supported for PQ)

  PQQuantizer<DataType> quantizer_;  ///< PQ codebook and encoding
  std::vector<uint8_t> pq_codes_;    ///< PQ codes storage: N × M bytes
  IDType item_cnt_{0};               ///< Number of stored vectors
  IDType delete_cnt_{0};             ///< Number of deleted vectors (for compatibility)

 public:
  PQSpace() = default;
  ~PQSpace() = default;
  ALAYA_NON_COPYABLE_BUT_MOVABLE(PQSpace);

  /**
   * @brief Construct PQSpace with parameters.
   *
   * @param capacity Maximum number of vectors
   * @param dim Original vector dimension
   * @param num_subspaces Number of PQ subspaces M (dim must be divisible by M)
   * @param metric Metric type (currently only L2 is fully supported)
   */
  PQSpace(IDType capacity, uint32_t dim, uint32_t num_subspaces, MetricType metric = MetricType::L2)
      : capacity_(capacity),
        dim_(dim),
        num_subspaces_(num_subspaces),
        metric_(metric),
        quantizer_(dim, num_subspaces) {
    if (metric != MetricType::L2) {
      LOG_WARN("PQSpace: Only L2 metric is fully supported. IP/COS may give inaccurate results.");
    }
    pq_codes_.reserve(static_cast<size_t>(capacity) * num_subspaces);
  }

  /**
   * @brief Construct PQSpace from existing quantizer.
   *
   * @param capacity Maximum number of vectors
   * @param quantizer Pre-trained PQ quantizer
   * @param metric Metric type
   */
  PQSpace(IDType capacity, PQQuantizer<DataType> quantizer, MetricType metric = MetricType::L2)
      : capacity_(capacity),
        dim_(quantizer.dim()),
        num_subspaces_(quantizer.num_subspaces()),
        metric_(metric),
        quantizer_(std::move(quantizer)) {
    pq_codes_.reserve(static_cast<size_t>(capacity) * num_subspaces_);
  }

  // ==========================================================================
  // Space concept interface
  // ==========================================================================

  /**
   * @brief Get vector dimension.
   */
  [[nodiscard]] auto get_dim() const -> uint32_t { return dim_; }

  /**
   * @brief Get data size per vector (PQ code size in bytes).
   */
  [[nodiscard]] auto get_data_size() const -> size_t { return num_subspaces_; }

  /**
   * @brief Get capacity.
   */
  [[nodiscard]] auto get_capacity() const -> IDType { return capacity_; }

  /**
   * @brief Get number of stored vectors.
   */
  [[nodiscard]] auto get_data_num() const -> IDType { return item_cnt_; }

  /**
   * @brief Set metric function (no-op for PQ, metric is fixed at construction).
   */
  void set_metric_function() {
    // PQ distance is computed via ADC table lookup, no function pointer needed
  }

  /**
   * @brief Get distance function pointer (for Space concept compatibility).
   *
   * Note: This returns a placeholder. Use QueryContext for actual distance computation.
   */
  [[nodiscard]] auto get_dist_func() const -> DistFunc<DataType, DistanceType> {
    // Return nullptr - actual distance computation uses ADC table lookup
    return nullptr;
  }

  /**
   * @brief Fit data into the space (train codebook and encode vectors).
   *
   * @param data Raw vector data (num_vectors × dim)
   * @param num_vectors Number of vectors
   */
  void fit(const DataType *data, IDType num_vectors) {
    if (num_vectors > capacity_) {
      throw std::runtime_error("Number of vectors exceeds capacity");
    }

    // Train quantizer if not already trained
    if (quantizer_.codebook_bytes() == 0) {
      quantizer_ = PQQuantizer<DataType>(dim_, num_subspaces_);
    }
    quantizer_.fit(data, num_vectors);

    // Encode all vectors
    pq_codes_.resize(static_cast<size_t>(num_vectors) * num_subspaces_);
    quantizer_.batch_encode(data, num_vectors, pq_codes_.data());
    item_cnt_ = num_vectors;

    LOG_INFO("PQSpace: Fitted {} vectors, code size = {} bytes/vector",
             num_vectors,
             num_subspaces_);
  }

  /**
   * @brief Get approximate distance between two stored vectors.
   *
   * Note: This computes distance by decoding both vectors, which is slow.
   * For search, use QueryContext with precomputed ADC table instead.
   *
   * @param i First vector ID
   * @param j Second vector ID
   * @return Approximate squared L2 distance
   */
  auto get_distance(IDType i, IDType j) -> DistanceType {
    // Decode both vectors and compute exact distance on reconstructed vectors
    std::vector<DataType> vec_i(dim_);
    std::vector<DataType> vec_j(dim_);
    quantizer_.decode(get_code(i), vec_i.data());
    quantizer_.decode(get_code(j), vec_j.data());

    float dist = 0.0F;
    for (uint32_t d = 0; d < dim_; ++d) {
      float diff = static_cast<float>(vec_i[d]) - static_cast<float>(vec_j[d]);
      dist += diff * diff;
    }
    return static_cast<DistanceType>(dist);
  }

  // ==========================================================================
  // PQ-specific interface
  // ==========================================================================

  /**
   * @brief Get PQ codes for a vector by ID.
   *
   * @param id Vector ID
   * @return Pointer to PQ codes (num_subspaces bytes)
   */
  [[nodiscard]] auto get_code(IDType id) const -> const uint8_t * {
    return pq_codes_.data() + static_cast<size_t>(id) * num_subspaces_;
  }

  /**
   * @brief Get mutable PQ codes pointer.
   */
  [[nodiscard]] auto get_code(IDType id) -> uint8_t * {
    return pq_codes_.data() + static_cast<size_t>(id) * num_subspaces_;
  }

  /**
   * @brief Get all PQ codes.
   */
  [[nodiscard]] auto get_all_codes() const -> const uint8_t * { return pq_codes_.data(); }

  /**
   * @brief Get the quantizer.
   */
  [[nodiscard]] auto get_quantizer() const -> const PQQuantizer<DataType> & { return quantizer_; }

  /**
   * @brief Get mutable quantizer reference.
   */
  [[nodiscard]] auto get_quantizer() -> PQQuantizer<DataType> & { return quantizer_; }

  /**
   * @brief Get number of subspaces.
   */
  [[nodiscard]] auto num_subspaces() const -> uint32_t { return num_subspaces_; }

  /**
   * @brief Insert a vector (encode and store PQ codes).
   *
   * @param data Raw vector data
   * @return Assigned ID
   */
  auto insert(const DataType *data) -> IDType {
    if (item_cnt_ >= capacity_) {
      return static_cast<IDType>(-1);
    }

    IDType id = item_cnt_++;
    pq_codes_.resize(static_cast<size_t>(item_cnt_) * num_subspaces_);
    quantizer_.encode(data, get_code(id));
    return id;
  }

  /**
   * @brief Remove a vector by ID (mark as deleted).
   */
  auto remove(IDType id) -> IDType {
    // PQ codes are stored contiguously, actual removal would require reindex
    // For now, just track deletion count for compatibility
    delete_cnt_++;
    return id;
  }

  /**
   * @brief Set PQ codes directly (for loading from disk).
   *
   * @param codes PQ codes array (num_vectors × num_subspaces)
   * @param num_vectors Number of vectors
   */
  void set_codes(const uint8_t *codes, IDType num_vectors) {
    if (num_vectors > capacity_) {
      throw std::runtime_error("Number of vectors exceeds capacity");
    }
    pq_codes_.resize(static_cast<size_t>(num_vectors) * num_subspaces_);
    std::memcpy(pq_codes_.data(), codes, pq_codes_.size());
    item_cnt_ = num_vectors;
  }

  // ==========================================================================
  // QueryContext for efficient ADC-based distance computation
  // ==========================================================================

  /**
   * @brief Query context with precomputed ADC table for fast distance lookups.
   *
   * Usage:
   * ```cpp
   * auto ctx = pq_space.get_query_context(query_vector);
   * float dist = ctx(vector_id);  // Fast ADC lookup
   * ```
   */
  class QueryContext {
   private:
    const PQSpace &space_;
    std::vector<float> adc_table_;  ///< M × 256 lookup table

   public:
    /**
     * @brief Construct QueryContext from raw query vector.
     *
     * @param space Reference to PQSpace
     * @param query Raw query vector
     */
    QueryContext(const PQSpace &space, const DataType *query) : space_(space) {
      adc_table_.resize(space_.num_subspaces_ * PQQuantizer<DataType>::kNumCentroids);
      space_.quantizer_.compute_adc_table(query, adc_table_.data());
    }

    /**
     * @brief Construct QueryContext from stored vector ID.
     *
     * @param space Reference to PQSpace
     * @param id Query vector ID (will decode and compute ADC table)
     */
    QueryContext(const PQSpace &space, IDType id) : space_(space) {
      // Decode the vector first
      std::vector<DataType> query(space_.dim_);
      space_.quantizer_.decode(space_.get_code(id), query.data());

      adc_table_.resize(space_.num_subspaces_ * PQQuantizer<DataType>::kNumCentroids);
      space_.quantizer_.compute_adc_table(query.data(), adc_table_.data());
    }

    ~QueryContext() = default;

    QueryContext(const QueryContext &) = delete;
    auto operator=(const QueryContext &) -> QueryContext & = delete;
    QueryContext(QueryContext &&) = default;
    auto operator=(QueryContext &&) -> QueryContext & = default;

    /**
     * @brief Compute approximate distance to a vector using ADC table lookup.
     *
     * @param id Target vector ID
     * @return Approximate squared L2 distance
     */
    [[nodiscard]] auto operator()(IDType id) const -> DistanceType {
      return static_cast<DistanceType>(
          space_.quantizer_.compute_distance_with_table(adc_table_.data(), space_.get_code(id)));
    }

    /**
     * @brief Compute distance given PQ codes directly.
     *
     * @param codes PQ codes pointer
     * @return Approximate squared L2 distance
     */
    [[nodiscard]] auto compute(const uint8_t *codes) const -> DistanceType {
      return static_cast<DistanceType>(
          space_.quantizer_.compute_distance_with_table(adc_table_.data(), codes));
    }

    /**
     * @brief Get the ADC table for external use (e.g., SIMD batch computation).
     */
    [[nodiscard]] auto adc_table() const -> const float * { return adc_table_.data(); }
  };

  /**
   * @brief Create a QueryContext for efficient distance computation.
   *
   * @param query Raw query vector
   * @return QueryContext with precomputed ADC table
   */
  [[nodiscard]] auto get_query_context(const DataType *query) const -> QueryContext {
    return QueryContext(*this, query);
  }

  /**
   * @brief Create a QueryContext from a stored vector ID.
   *
   * @param id Query vector ID
   * @return QueryContext with precomputed ADC table
   */
  [[nodiscard]] auto get_query_context(IDType id) const -> QueryContext {
    return QueryContext(*this, id);
  }

  // Alias for compatibility with other Space implementations
  using QueryComputer = QueryContext;
  [[nodiscard]] auto get_query_computer(const DataType *query) const -> QueryContext {
    return get_query_context(query);
  }
  [[nodiscard]] auto get_query_computer(IDType id) const -> QueryContext {
    return get_query_context(id);
  }

  // ==========================================================================
  // Persistence
  // ==========================================================================

  /**
   * @brief Save PQSpace to file.
   */
  void save(std::string_view filename) const {
    std::ofstream writer(std::string(filename), std::ios::binary);
    if (!writer.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    writer.write(reinterpret_cast<const char *>(&capacity_), sizeof(capacity_));
    writer.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<const char *>(&num_subspaces_), sizeof(num_subspaces_));
    writer.write(reinterpret_cast<const char *>(&metric_), sizeof(metric_));
    writer.write(reinterpret_cast<const char *>(&item_cnt_), sizeof(item_cnt_));
    writer.write(reinterpret_cast<const char *>(&delete_cnt_), sizeof(delete_cnt_));

    // Save quantizer (codebook)
    quantizer_.save(writer);

    // Save PQ codes
    size_t code_bytes = static_cast<size_t>(item_cnt_) * num_subspaces_;
    writer.write(reinterpret_cast<const char *>(pq_codes_.data()), code_bytes);

    LOG_INFO("PQSpace: Saved to {} ({} vectors, {} bytes codes)", filename, item_cnt_, code_bytes);
  }

  /**
   * @brief Load PQSpace from file.
   */
  void load(std::string_view filename) {
    std::ifstream reader(std::string(filename), std::ios::binary);
    if (!reader.is_open()) {
      throw std::runtime_error("Cannot open file " + std::string(filename));
    }

    reader.read(reinterpret_cast<char *>(&capacity_), sizeof(capacity_));
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&num_subspaces_), sizeof(num_subspaces_));
    reader.read(reinterpret_cast<char *>(&metric_), sizeof(metric_));
    reader.read(reinterpret_cast<char *>(&item_cnt_), sizeof(item_cnt_));
    reader.read(reinterpret_cast<char *>(&delete_cnt_), sizeof(delete_cnt_));

    // Load quantizer (codebook)
    quantizer_.load(reader);

    // Load PQ codes
    size_t code_bytes = static_cast<size_t>(item_cnt_) * num_subspaces_;
    pq_codes_.resize(code_bytes);
    reader.read(reinterpret_cast<char *>(pq_codes_.data()), code_bytes);

    LOG_INFO("PQSpace: Loaded from {} ({} vectors, {} bytes codes)",
             filename,
             item_cnt_,
             code_bytes);
  }

  /**
   * @brief Prefetch PQ codes for a vector (for cache optimization).
   */
  void prefetch_by_id(IDType id) const {
    mem_prefetch_l1(get_code(id), (num_subspaces_ + 63) / 64);
  }
};

}  // namespace alaya
