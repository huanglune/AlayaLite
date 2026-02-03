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
#include <stdexcept>
#include <vector>

#include "utils/kmeans.hpp"
#include "utils/log.hpp"

namespace alaya {

/**
 * @brief Product Quantization (PQ) for vector compression.
 *
 * PQ divides a D-dimensional vector into M subspaces, each with D/M dimensions.
 * Each subspace is quantized independently using K centroids (typically K=256).
 * This results in a compressed representation of M bytes per vector.
 *
 * Reference: "Product Quantization for Nearest Neighbor Search" (Jégou et al., 2011)
 *
 * @tparam DataType The data type of input values (e.g., float).
 */
template <typename DataType = float>
class PQQuantizer {
 public:
  static constexpr uint32_t kNumCentroids = 256;  ///< Number of centroids per subspace (K)

 private:
  uint32_t dim_{0};            ///< Original vector dimension (D)
  uint32_t num_subspaces_{0};  ///< Number of subspaces (M)
  uint32_t subspace_dim_{0};   ///< Dimension per subspace (D/M)

  /// Codebook: M subspaces × K centroids × (D/M) dimensions
  /// Layout: [subspace_0][centroid_0..255][dims] [subspace_1][centroid_0..255][dims] ...
  std::vector<DataType> codebook_;

  // K-means configuration
  KMeans<DataType> kmeans_{kNumCentroids, 20, 3};

 public:
  PQQuantizer() = default;
  ~PQQuantizer() = default;

  PQQuantizer(const PQQuantizer &other) = default;
  auto operator=(const PQQuantizer &other) -> PQQuantizer & = default;
  PQQuantizer(PQQuantizer &&other) noexcept = default;
  auto operator=(PQQuantizer &&other) noexcept -> PQQuantizer & = default;

  /**
   * @brief Construct PQQuantizer with specified parameters.
   *
   * @param dim Original vector dimension (must be divisible by num_subspaces)
   * @param num_subspaces Number of subspaces M (typically 8, 16, 32, or 64)
   */
  PQQuantizer(uint32_t dim, uint32_t num_subspaces) : dim_(dim), num_subspaces_(num_subspaces) {
    if (dim % num_subspaces != 0) {
      throw std::invalid_argument("Dimension must be divisible by num_subspaces");
    }
    subspace_dim_ = dim / num_subspaces;
    codebook_.resize(static_cast<size_t>(num_subspaces) * kNumCentroids * subspace_dim_);
  }

  /**
   * @brief Get the original vector dimension.
   */
  [[nodiscard]] auto dim() const -> uint32_t { return dim_; }

  /**
   * @brief Get the number of subspaces.
   */
  [[nodiscard]] auto num_subspaces() const -> uint32_t { return num_subspaces_; }

  /**
   * @brief Get the dimension per subspace.
   */
  [[nodiscard]] auto subspace_dim() const -> uint32_t { return subspace_dim_; }

  /**
   * @brief Get the code size in bytes (equals num_subspaces).
   */
  [[nodiscard]] auto code_size() const -> uint32_t { return num_subspaces_; }

  /**
   * @brief Get pointer to codebook data.
   */
  [[nodiscard]] auto codebook_data() const -> const DataType * { return codebook_.data(); }

  /**
   * @brief Get codebook size in bytes.
   */
  [[nodiscard]] auto codebook_bytes() const -> size_t {
    return codebook_.size() * sizeof(DataType);
  }

  /**
   * @brief Get pointer to centroids for a specific subspace.
   *
   * @param subspace_idx Subspace index [0, M)
   * @return Pointer to K×(D/M) centroids for this subspace
   */
  [[nodiscard]] auto get_subspace_centroids(uint32_t subspace_idx) const -> const DataType * {
    return codebook_.data() + static_cast<size_t>(subspace_idx) * kNumCentroids * subspace_dim_;
  }

  /**
   * @brief Get pointer to a specific centroid.
   *
   * @param subspace_idx Subspace index [0, M)
   * @param centroid_idx Centroid index [0, 256)
   * @return Pointer to the centroid vector
   */
  [[nodiscard]] auto get_centroid(uint32_t subspace_idx, uint32_t centroid_idx) const
      -> const DataType * {
    return get_subspace_centroids(subspace_idx) + static_cast<size_t>(centroid_idx) * subspace_dim_;
  }

  /**
   * @brief Train the codebook using K-means clustering.
   *
   * @param data Pointer to training data (num_vectors × dim)
   * @param num_vectors Number of training vectors
   */
  void fit(const DataType *data, size_t num_vectors) {
    if (num_vectors < kNumCentroids) {
      throw std::invalid_argument("Need at least 256 vectors for PQ training");
    }

    LOG_INFO("PQ: Training codebook with {} vectors, M={}, D/M={}",
             num_vectors,
             num_subspaces_,
             subspace_dim_);

    // Train each subspace independently
    for (uint32_t m = 0; m < num_subspaces_; ++m) {
      train_subspace(data, num_vectors, m);
    }

    LOG_INFO("PQ: Codebook training completed");
  }

  /**
   * @brief Encode a single vector to PQ codes.
   *
   * @param vec Input vector (dim dimensions)
   * @param codes Output PQ codes (num_subspaces bytes)
   */
  void encode(const DataType *vec, uint8_t *codes) const {
    for (uint32_t m = 0; m < num_subspaces_; ++m) {
      const DataType *subvec = vec + m * subspace_dim_;
      codes[m] = find_nearest_centroid(m, subvec);
    }
  }

  /**
   * @brief Batch encode multiple vectors.
   *
   * @param data Input vectors (num_vectors × dim)
   * @param num_vectors Number of vectors
   * @param codes Output PQ codes (num_vectors × num_subspaces)
   */
  void batch_encode(const DataType *data, size_t num_vectors, uint8_t *codes) const {
    for (size_t i = 0; i < num_vectors; ++i) {
      encode(data + i * dim_, codes + i * num_subspaces_);
    }
  }

  /**
   * @brief Decode PQ codes back to approximate vector.
   *
   * @param codes Input PQ codes (num_subspaces bytes)
   * @param vec Output reconstructed vector (dim dimensions)
   */
  void decode(const uint8_t *codes, DataType *vec) const {
    for (uint32_t m = 0; m < num_subspaces_; ++m) {
      const DataType *centroid = get_centroid(m, codes[m]);
      std::memcpy(vec + m * subspace_dim_, centroid, subspace_dim_ * sizeof(DataType));
    }
  }

  /**
   * @brief Compute ADC (Asymmetric Distance Computation) lookup table.
   *
   * For a query vector, precompute squared L2 distances from each query subvector
   * to all centroids in that subspace. This enables O(M) distance computation
   * using table lookups.
   *
   * @param query Query vector (dim dimensions)
   * @param adc_table Output table (num_subspaces × 256 floats)
   */
  void compute_adc_table(const DataType *query, float *adc_table) const {
    for (uint32_t m = 0; m < num_subspaces_; ++m) {
      const DataType *query_sub = query + m * subspace_dim_;
      float *table_row = adc_table + m * kNumCentroids;

      for (uint32_t k = 0; k < kNumCentroids; ++k) {
        const DataType *centroid = get_centroid(m, k);
        table_row[k] = KMeans<DataType>::compute_l2_sqr(query_sub, centroid, subspace_dim_);
      }
    }
  }

  /**
   * @brief Compute approximate distance using ADC table lookup.
   *
   * @param adc_table Precomputed ADC table for query
   * @param codes PQ codes of the target vector
   * @return Approximate squared L2 distance
   */
  [[nodiscard]] auto compute_distance_with_table(const float *adc_table, const uint8_t *codes) const
      -> float {
    float dist = 0.0F;
    for (uint32_t m = 0; m < num_subspaces_; ++m) {
      dist += adc_table[m * kNumCentroids + codes[m]];
    }
    return dist;
  }

  /**
   * @brief Save quantizer to file stream.
   */
  void save(std::ofstream &writer) const {
    writer.write(reinterpret_cast<const char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<const char *>(&num_subspaces_), sizeof(num_subspaces_));
    writer.write(reinterpret_cast<const char *>(&subspace_dim_), sizeof(subspace_dim_));
    writer.write(reinterpret_cast<const char *>(codebook_.data()), codebook_bytes());
    LOG_INFO("PQ: Quantizer saved (codebook size: {} bytes)", codebook_bytes());
  }

  /**
   * @brief Load quantizer from file stream.
   */
  void load(std::ifstream &reader) {
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&num_subspaces_), sizeof(num_subspaces_));
    reader.read(reinterpret_cast<char *>(&subspace_dim_), sizeof(subspace_dim_));

    codebook_.resize(static_cast<size_t>(num_subspaces_) * kNumCentroids * subspace_dim_);
    reader.read(reinterpret_cast<char *>(codebook_.data()), codebook_bytes());
    LOG_INFO("PQ: Quantizer loaded (M={}, D/M={}, codebook size: {} bytes)",
             num_subspaces_,
             subspace_dim_,
             codebook_bytes());
  }

 private:
  /**
   * @brief Find the nearest centroid for a subvector.
   *
   * @param subspace_idx Subspace index
   * @param subvec Subvector to quantize
   * @return Index of nearest centroid [0, 255]
   */
  [[nodiscard]] auto find_nearest_centroid(uint32_t subspace_idx, const DataType *subvec) const
      -> uint8_t {
    const DataType *centroids = get_subspace_centroids(subspace_idx);
    return static_cast<uint8_t>(kmeans_.find_nearest(subvec, centroids, subspace_dim_));
  }

  /**
   * @brief Train centroids for a single subspace using K-means.
   *
   * @param data Full training data
   * @param num_vectors Number of training vectors
   * @param subspace_idx Subspace index to train
   */
  void train_subspace(const DataType *data, size_t num_vectors, uint32_t subspace_idx) {
    // Extract subvectors for this subspace
    std::vector<DataType> subvectors(num_vectors * subspace_dim_);
    for (size_t i = 0; i < num_vectors; ++i) {
      const DataType *src = data + i * dim_ + subspace_idx * subspace_dim_;
      DataType *dst = subvectors.data() + i * subspace_dim_;
      std::memcpy(dst, src, subspace_dim_ * sizeof(DataType));
    }

    // Get pointer to this subspace's centroids
    DataType *centroids =
        codebook_.data() + static_cast<size_t>(subspace_idx) * kNumCentroids * subspace_dim_;

    // Run K-means clustering
    kmeans_.fit(subvectors.data(), num_vectors, subspace_dim_, centroids);
  }
};

}  // namespace alaya
