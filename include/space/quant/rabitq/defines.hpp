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

#pragma once

#include "kernels/linalg/types.hpp"
#include "platform/detect.hpp"


namespace alaya {

template <typename T>
auto dot_product(const T *ALAYA_RESTRICT vec0, const T *ALAYA_RESTRICT vec1, size_t dim) -> T {
  kernels::linalg::ConstVectorMap<T> v0(vec0, dim);
  kernels::linalg::ConstVectorMap<T> v1(vec1, dim);
  return v0.dot(v1);
}

template <typename T>
inline auto l2_sqr(const T *ALAYA_RESTRICT vec0, size_t dim) -> T {
  kernels::linalg::ConstVectorMap<T> v0(vec0, dim);
  return v0.dot(v0);
}

template <typename T>
inline auto l2_sqr(const T *ALAYA_RESTRICT vec0, const T *ALAYA_RESTRICT vec1, size_t dim) -> T {
  kernels::linalg::ConstVectorMap<T> v0(vec0, dim);
  kernels::linalg::ConstVectorMap<T> v1(vec1, dim);
  return (v0 - v1).dot(v0 - v1);
}
}  // namespace alaya
