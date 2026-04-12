/**
 * @file fht_rotator.hpp
 * @brief Fast Hadamard Transform (FHT) based vector rotation.
 *
 * Applies randomized FHT rotation to vectors before quantization.
 * This rotation normalizes the distribution of vector components,
 * improving the accuracy of binary quantization in RaBitQ.
 *
 * The rotation consists of:
 * 1. Element-wise multiplication with random signs
 * 2. Fast Hadamard Transform (O(d log d) complexity)
 */
// NOLINTBEGIN

#pragma once

#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>

#include "simd/fht.hpp"
#include "utils/aligned_array.hpp"
#include "utils/memory.hpp"

namespace symqg {

/**
 * @brief Applies randomized Fast Hadamard Transform rotation to vectors.
 *
 * Used during both indexing (to rotate neighbor vectors) and search
 * (to rotate query vectors) with the same random signs for consistency.
 */
class FHTRotator {
  using data_type = data::Array<float, std::vector<size_t>, alaya::AlignedAlloc<float, 64>>;

 private:
  std::function<void(float *)> fht_float_ = alaya::simd::helper_float_6;
  size_t iter_ = 0;
  size_t remain_ = 0;
  size_t dimension_ = 0;
  size_t paded_dim_ = 0;
  data_type mat_;

 public:
  FHTRotator() = default;

  explicit FHTRotator(size_t dim)
      : dimension_(dim),
        paded_dim_(1 << alaya::math::ceil_log2(dim)),
        mat_(std::vector<size_t>{1, paded_dim_}) {
    size_t log_b = alaya::math::ceil_log2(dim);

    std::uniform_int_distribution<int> bernoulli(0, 1);
    std::random_device rdd;
    std::mt19937_64 gen(rdd());
    for (size_t i = 0; i < paded_dim_; ++i) {
      mat_[i] =
          static_cast<float>((2 * bernoulli(gen)) - 1) / std::sqrt(static_cast<float>(paded_dim_));
    }
#if defined(__AVX512F__)
    remain_ = dimension_ & 0b1111;
    iter_ = dimension_ - remain_;
#elif defined(__AVX2__)
    remain_ = dimension_ & 0b111;
    iter_ = dimension_ - remain_;
#else
    remain_ = dimension_ & 0b11;
    iter_ = dimension_ - remain_;
#endif
    switch (log_b) {
      case 6:
        this->fht_float_ = alaya::simd::helper_float_6;
        break;
      case 7:
        this->fht_float_ = alaya::simd::helper_float_7;
        break;
      case 8:
        this->fht_float_ = alaya::simd::helper_float_8;
        break;
      case 9:
        this->fht_float_ = alaya::simd::helper_float_9;
        break;
      case 10:
        this->fht_float_ = alaya::simd::helper_float_10;
        break;
      case 11:
        this->fht_float_ = alaya::simd::helper_float_11;
        break;
      default:
        std::cerr << "dimension of vector is too big\n";
        abort();
        break;
    }
  }

  ~FHTRotator() = default;

  /**
   * @brief       rotate the scr vector by FHTRotator
   *
   * @param src   raw query vector, length dimension_
   * @param dst   rotated query vector, length B, must be aligned to 64 bytes
   */
  void rotate(const float *__restrict__ src, float *__restrict__ dst) const {
    size_t idx = 0;
// NOLINTBEGIN(portability-simd-intrinsics)
#if defined(__AVX512F__)
    for (; idx < iter_; idx += 16) {
      __m512 ss = _mm512_loadu_ps(&src[idx]);
      __m512 mm = _mm512_load_ps(&mat_.at(idx));
      ss = _mm512_mul_ps(ss, mm);
      _mm512_store_ps(&dst[idx], ss);
    }
#elif defined(__AVX2__)
    for (; idx < iter_; idx += 8) {
      __m256 ss = _mm256_loadu_ps(&src[idx]);
      __m256 mm = _mm256_load_ps(&mat_.at(idx));
      ss = _mm256_mul_ps(ss, mm);
      _mm256_store_ps(&dst[idx], ss);
    }
// NOLINTEND(portability-simd-intrinsics)
#else
    for (idx = 0; idx < iter_; ++idx) {
      dst[idx] = src[idx] * mat_.at(idx);
    }
#endif
    for (; idx < dimension_; ++idx) {
      dst[idx] = src[idx] * mat_.at(idx);
    }
    std::fill(dst + dimension_, dst + paded_dim_, 0.0F);
    fht_float_(dst);
  }

  void load(std::ifstream &input) { mat_.load(input); }

  void save(std::ofstream &output) const { mat_.save(output); }
};
}  // namespace symqg
// NOLINTEND
