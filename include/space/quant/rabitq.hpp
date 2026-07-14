/*
 * Copyright 2025 VectorDB.NTU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>

#include "core/log.hpp"
#include "core/value_types.hpp"
#include "space/quant/rabitq_core.hpp"
#include "space/quant/rabitq/defines.hpp"
#include "space/quant/rabitq/fastscan.hpp"

namespace alaya {
template <typename DataType>
struct RaBitQQuantizer {
 private:
  uint32_t dim_{0};         ///< dimension
  uint32_t padded_dim_{0};  ///< padded dimension

  /**
   * @brief pack 0/1 uncompacted integer data (binary_code) to compacted bytes data (compact_code)
   *
   * @param binary_code uncompact quantization code, e.g., [1,0,1,0,0,1,1,0]
   * @param compact_code compact quantization code, e.g., [10100110]
   */
  void pack_binary(const int *ALAYA_RESTRICT binary_code, uint8_t *ALAYA_RESTRICT compact_code) {
    constexpr size_t kTypeBits = sizeof(uint8_t) * 8;
    // i points to the start point of each batch
    for (size_t i = 0; i < padded_dim_; i += kTypeBits) {
      uint8_t cur = 0;
      // j represents offset within a byte,j∈[0,kTypeBits-1]
      for (size_t j = 0; j < kTypeBits; ++j) {
        cur |= (static_cast<uint8_t>(binary_code[i + j]) << (kTypeBits - 1 - j));
      }
      *compact_code = cur;
      // 1 byte(8 bits) each batch
      ++compact_code;
    }
  }

  /**
   * @brief Calculate factors(f_add and f_scale) and quantization code for one neighbor
   *
   * @param data Rotatated neighbor data, len: padded_dim
   * @param centroid Rotated centroid data pointer, len: padded_dim
   * @param binary_code Store uncompacted quantization code, every int is either 1 or 0
   * @param f_add One of factors
   * @param f_rescale One of factors
   */
  void cal_fac_and_qc(const DataType *data,
                      const DataType *centroid,
                      int *binary_code,
                      DataType &f_add,
                      DataType &f_rescale,
                      const core::Metric metric) {
    const auto factors =
        RaBitQCore::memory_factors(data, centroid, padded_dim_, binary_code, metric);
    f_add = factors.base;
    // The legacy memory scan consumes <s/2,q>. L2 historically stores twice the canonical
    // <s,q> scale; IP/COS already store the canonical scale and must remain byte-compatible.
    f_rescale =
        metric == core::Metric::l2 ? 2 * factors.signed_query_scale : factors.signed_query_scale;
  }

 public:
  RaBitQQuantizer() = default;
  ~RaBitQQuantizer() = default;

  RaBitQQuantizer(const RaBitQQuantizer &) = delete;
  auto operator=(const RaBitQQuantizer &) -> RaBitQQuantizer & = delete;

  RaBitQQuantizer(const RaBitQQuantizer &&) = delete;
  auto operator=(const RaBitQQuantizer &&) -> RaBitQQuantizer & = delete;

  explicit RaBitQQuantizer(const uint32_t &dim, const uint32_t &padded_dim)
      : dim_(dim), padded_dim_(padded_dim) {}

  // use one vertex as the centroid and quantize its neighbors
  auto batch_quantize(const DataType *rotated_neighbors /* len: num * dim */,
                      const DataType *rotated_centroid /* single centroid, len: dim */,
                      size_t num /* total number of the neighbors in this batch */,
                      /* The following pointers point to where the result data is stored */
                      uint8_t *bin_code,
                      DataType *f_add,
                      DataType *f_rescale,
                      const core::Metric metric) -> void {
    // for compacted quantization code storage
    std::vector<uint8_t> compact_codes(num * padded_dim_ / 8);  // 1 bit/dim

    /// todo: parallelable?
    for (size_t i = 0; i < num; ++i) {                           // ith neighbor
      auto rotated_nei = rotated_neighbors + (i * padded_dim_);  // start pointer

      // for uncompacted quantization code storage
      std::vector<int> binary_code(padded_dim_);
      cal_fac_and_qc(rotated_nei,
                     rotated_centroid,
                     binary_code.data(),
                     f_add[i],
                     f_rescale[i],
                     metric);

      // the number of bits in every uint8_t
      constexpr size_t kTypeBits = 8;
      // padded_dim_ / kTypeBits denotes the total number of uint8_t needed for a single neighbor's
      // quantization code
      auto compact_code =
          reinterpret_cast<uint8_t *>(compact_codes.data()) + (padded_dim_ / kTypeBits * i);
      // pack 0/1 uncompacted integer data (binary_code) to compacted bytes data (compact_code)
      pack_binary(binary_code.data(), compact_code);
    }

    // restructure quantization codes for later fastscan computation in querying phase
    fastscan::pack_codes(padded_dim_, compact_codes.data(), num, bin_code);
  }

  auto save(std::ofstream &writer) -> void {
    writer.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    writer.write(reinterpret_cast<char *>(&padded_dim_), sizeof(padded_dim_));

    LOG_INFO("rabitq quantizer is saved.");
  }

  auto load(std::ifstream &reader) -> void {
    reader.read(reinterpret_cast<char *>(&dim_), sizeof(dim_));
    reader.read(reinterpret_cast<char *>(&padded_dim_), sizeof(padded_dim_));

    LOG_INFO("rabitq quantizer is loaded.");
  }
};
};  // namespace alaya
