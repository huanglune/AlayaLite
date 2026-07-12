// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <Eigen/Dense>

namespace alaya::kernels::linalg {

using Index = Eigen::Index;

template <typename T>
using RowMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using ColMajorMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
template <typename T>
using RowMajorMatrixMap = Eigen::Map<RowMajorMatrix<T>>;
template <typename T>
using ConstRowMajorMatrixMap = Eigen::Map<const RowMajorMatrix<T>>;
template <typename T>
using RowMajorArray = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <typename T>
using RowMajorArrayMap = Eigen::Map<RowMajorArray<T>>;
template <typename T>
using ConstRowMajorArrayMap = Eigen::Map<const RowMajorArray<T>>;
template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T>
using VectorMap = Eigen::Map<Vector<T>>;
template <typename T>
using ConstVectorMap = Eigen::Map<const Vector<T>>;

}  // namespace alaya::kernels::linalg
