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
#include <random>
#include <thread>
#include <vector>

namespace alaya {

// thread-safe random integer generator
template <typename IDType>
inline auto rand_integer(IDType min, IDType max) -> IDType {
  static thread_local std::mt19937 generator(
      std::random_device{}() +  // NOLINT(whitespace/braces)
      std::hash<std::thread::id>()(std::this_thread::get_id()));
  std::uniform_int_distribution<IDType> distribution(min, max);
  return distribution(generator);
}

template <typename IDType>
inline void gen_random(std::mt19937 &rng, IDType *addr, const int size, const int n) {
  for (int i = 0; i < size; ++i) {
    addr[i] = rng() % n;
  }
  std::sort(addr, addr + size);
  for (int i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  int off = rng() % n;
  for (int i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % n;
  }
}

struct RandomGenerator {
  std::mt19937 mt_;

  explicit RandomGenerator(int64_t seed = 1234) : mt_(static_cast<unsigned int>(seed)) {}

  /// random positive integer
  auto rand_int() -> int { return mt_() & 0x7fffffff; }

  /// random int64_t
  auto rand_int64() -> int64_t {
    return static_cast<int64_t>(rand_int()) | static_cast<int64_t>(rand_int()) << 31;
  }

  /// generate random integer between 0 and max-1
  auto rand_int(int max) -> int { return mt_() % max; }

  /// between 0 and 1
  auto rand_float() -> float { return mt_() / static_cast<float>(std::mt19937::max()); }

  auto rand_double() -> double { return mt_() / static_cast<double>(std::mt19937::max()); }
};

/**
 * @brief Generate random float vectors for testing purposes.
 *
 * @tparam T The data type of the vectors (default: float)
 * @param num_vectors Number of vectors to generate
 * @param dim Dimension of each vector
 * @param seed Random seed for reproducibility (default: 42)
 * @param min_val Minimum value for random distribution (default: -1.0)
 * @param max_val Maximum value for random distribution (default: 1.0)
 * @return std::vector<T> Flattened vector containing all generated vectors
 */
template <typename T = float>
inline auto generate_random_vectors(size_t num_vectors,
                                    uint32_t dim,
                                    uint32_t seed = 42,
                                    T min_val = static_cast<T>(-1.0),
                                    T max_val = static_cast<T>(1.0)) -> std::vector<T> {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(min_val, max_val);
  std::vector<T> data(num_vectors * dim);
  for (auto &v : data) {
    v = dist(rng);
  }
  return data;
}

}  // namespace alaya
