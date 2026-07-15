// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once
#include "scalar_quantized_space.hpp"
#include "simd/distance_ip.hpp"
#include "simd/distance_l2.hpp"
#include "space/quant/sq8.hpp"

namespace alaya {

struct SQ8Traits {
  static constexpr const char *name = "SQ8Space";

  template <typename DataType>
  using Quantizer = SQ8Quantizer<DataType>;

  static constexpr uint32_t data_size(size_t dim) {
    return static_cast<uint32_t>(dim * sizeof(uint8_t));
  }

  template <typename DataType, typename DistanceType>
  static constexpr auto l2_func = simd::l2_sqr_sq8<DataType, DistanceType>;

  template <typename DataType, typename DistanceType>
  static constexpr auto ip_func = simd::ip_sqr_sq8<DataType, DistanceType>;
};

template <typename DataType = float,
          typename DistanceType = float,
          typename IDType = uint32_t,
          typename DataStorage = SequentialStorage<uint8_t, IDType>>
using SQ8Space = ScalarQuantizedSpace<SQ8Traits, DataType, DistanceType, IDType, DataStorage>;

static_assert(Space<SQ8Space<>>);

}  // namespace alaya
