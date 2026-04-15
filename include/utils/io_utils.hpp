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
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <stdexcept>
#include <vector>

#include "log.hpp"

namespace alaya {

// ============================================================================
// Legacy output-parameter loaders (used by dataset_utils, DiskANN bench, tests)
// ============================================================================

/**
 * @brief Load vectors from a .fvecs file (per-row dim prefix format).
 */
template <typename T>
inline void load_fvecs(const std::filesystem::path &filepath,
                       std::vector<T> &data,
                       uint32_t &num,
                       uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);

  if (!reader.is_open()) {
    LOG_CRITICAL("Open fvecs file error {}.", filepath.string());
    exit(-1);
  }

  num = 0;
  data.clear();

  while (!reader.eof()) {
    reader.read(reinterpret_cast<char *>(&dim), 4);
    if (reader.eof()) {
      break;
    }
    if (dim == 0) {
      LOG_CRITICAL("file {} is empty.", filepath.string());
      exit(-1);
    }
    std::vector<T> vec(dim);
    reader.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(T));
    if (reader.gcount() != dim * static_cast<int>(sizeof(T))) {
      LOG_CRITICAL("file {} is not valid.", filepath.string());
      exit(-1);
    }
    data.insert(data.end(), vec.begin(), vec.end());
    num++;
  }

  reader.close();
}

/**
 * @brief Load vectors from a .ivecs file (per-row dim prefix format).
 */
template <typename T>
inline void load_ivecs(const std::filesystem::path &filepath,
                       std::vector<T> &data,
                       uint32_t &num,
                       uint32_t &dim) {
  std::ifstream file(filepath, std::ios::binary);

  if (!file.is_open()) {
    LOG_CRITICAL("Open ivecs file error {}.", filepath.string());
    exit(-1);
  }

  file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (file.fail()) {
    throw std::runtime_error("Failed to read dimension from file: " + filepath.string());
  }

  file.seekg(0, std::ios::end);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  num = file_size / (sizeof(uint32_t) + dim * sizeof(T));
  data.resize(num * dim);

  for (uint32_t i = 0; i < num; ++i) {
    file.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
    file.read(reinterpret_cast<char *>(data.data() + (i * dim)), dim * sizeof(T));
    if (file.fail()) {
      throw std::runtime_error("Failed to read data from file: " + filepath.string());
    }
  }
}

// ============================================================================
// Modern return-value loaders
// ============================================================================

/**
 * @brief Result type for bulk-loading a vector file into memory.
 */
template <typename T>
struct VecFileData {
  uint32_t num_{0};
  uint32_t dim_{0};
  std::vector<T> data_;
};

/**
 * @brief Load all vectors from a .fvecs/.ivecs file into contiguous memory.
 *
 * Format: repeated [int32 dim][dim * sizeof(T) data] per row.
 */
template <typename T>
inline auto load_fvecs_all(const std::filesystem::path &filepath) -> VecFileData<T> {
  std::ifstream reader(filepath, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Cannot open fvecs file: " + filepath.string());
  }

  int32_t dim = 0;
  reader.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
  if (!reader || dim <= 0) {
    throw std::runtime_error("Invalid fvecs header: " + filepath.string());
  }

  reader.seekg(0, std::ios::end);
  size_t file_size = reader.tellg();
  size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(T);
  auto num = static_cast<uint32_t>(file_size / row_bytes);

  VecFileData<T> result;
  result.num_ = num;
  result.dim_ = static_cast<uint32_t>(dim);
  result.data_.resize(static_cast<size_t>(num) * dim);

  reader.seekg(0);
  for (uint32_t i = 0; i < num; ++i) {
    int32_t d = 0;
    reader.read(reinterpret_cast<char *>(&d), sizeof(int32_t));
    reader.read(reinterpret_cast<char *>(result.data_.data() + static_cast<size_t>(i) * dim),
                static_cast<std::streamsize>(dim * sizeof(T)));
  }
  return result;
}

/**
 * @brief Load all vectors from a .fbin/.ibin file into contiguous memory.
 *
 * Format: [int32 num_rows][int32 dim][num_rows * dim * sizeof(T) data].
 */
template <typename T>
inline auto load_bin_all(const std::filesystem::path &filepath) -> VecFileData<T> {
  std::ifstream reader(filepath, std::ios::binary);
  if (!reader.is_open()) {
    throw std::runtime_error("Cannot open bin file: " + filepath.string());
  }

  int32_t hdr[2]{};
  reader.read(reinterpret_cast<char *>(hdr), sizeof(hdr));
  if (!reader || hdr[0] <= 0 || hdr[1] <= 0) {
    throw std::runtime_error("Invalid bin header: " + filepath.string());
  }

  VecFileData<T> result;
  result.num_ = static_cast<uint32_t>(hdr[0]);
  result.dim_ = static_cast<uint32_t>(hdr[1]);
  result.data_.resize(static_cast<size_t>(result.num_) * result.dim_);
  reader.read(reinterpret_cast<char *>(result.data_.data()),
              static_cast<std::streamsize>(result.data_.size() * sizeof(T)));
  if (!reader) {
    throw std::runtime_error("Failed to read bin payload: " + filepath.string());
  }
  return result;
}

/**
 * @brief Auto-detect format and load float vectors (.fbin or .fvecs).
 */
inline auto load_float_vectors(const std::filesystem::path &filepath) -> VecFileData<float> {
  if (filepath.extension() == ".fbin") {
    return load_bin_all<float>(filepath);
  }
  return load_fvecs_all<float>(filepath);
}

/**
 * @brief Auto-detect format and load int32 vectors (.ibin or .ivecs).
 */
inline auto load_int_vectors(const std::filesystem::path &filepath) -> VecFileData<int32_t> {
  if (filepath.extension() == ".ibin") {
    return load_bin_all<int32_t>(filepath);
  }
  return load_fvecs_all<int32_t>(filepath);
}

}  // namespace alaya
