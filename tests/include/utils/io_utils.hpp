// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya {

namespace detail {

inline void check_open(const std::ifstream &stream, const std::filesystem::path &path) {
  if (!stream.is_open()) {
    throw std::runtime_error("Cannot open file: " + path.string());
  }
}

}  // namespace detail

template <typename T>
void load_fvecs(const std::filesystem::path &filepath,
                std::vector<T> &data,
                uint32_t &num,
                uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);
  detail::check_open(reader, filepath);

  num = 0;
  data.clear();

  while (reader.read(reinterpret_cast<char *>(&dim), 4)) {
    if (dim == 0) {
      throw std::runtime_error("Zero dimension in file: " + filepath.string());
    }
    auto offset = data.size();
    data.resize(offset + dim);
    reader.read(reinterpret_cast<char *>(data.data() + offset), dim * sizeof(T));
    if (!reader) {
      throw std::runtime_error("Truncated record in file: " + filepath.string());
    }
    ++num;
  }
}

template <typename T>
void load_ivecs(const std::filesystem::path &filepath,
                std::vector<T> &data,
                uint32_t &num,
                uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);
  detail::check_open(reader, filepath);

  reader.read(reinterpret_cast<char *>(&dim), sizeof(uint32_t));
  if (!reader) {
    throw std::runtime_error("Cannot read dimension from: " + filepath.string());
  }

  reader.seekg(0, std::ios::end);
  auto file_size = static_cast<std::size_t>(reader.tellg());
  reader.seekg(0, std::ios::beg);

  num = static_cast<uint32_t>(file_size / (sizeof(uint32_t) + dim * sizeof(T)));
  data.resize(num * dim);

  for (uint32_t i = 0; i < num; ++i) {
    uint32_t row_dim{};
    reader.read(reinterpret_cast<char *>(&row_dim), sizeof(uint32_t));
    reader.read(reinterpret_cast<char *>(data.data() + (i * dim)), dim * sizeof(T));
    if (!reader) {
      throw std::runtime_error("Truncated record at row " + std::to_string(i) +
                               " in: " + filepath.string());
    }
  }
}

template <typename T>
void load_bvecs(const std::filesystem::path &filepath,
                std::vector<T> &data,
                uint32_t &num,
                uint32_t &dim) {
  std::ifstream reader(filepath, std::ios::binary);
  detail::check_open(reader, filepath);

  reader.read(reinterpret_cast<char *>(&dim), 4);
  reader.seekg(0, std::ios::end);
  auto file_size = static_cast<std::size_t>(reader.tellg());
  reader.seekg(0, std::ios::beg);

  num = static_cast<uint32_t>(file_size / (4 + dim));
  data.resize(num * dim);

  std::vector<uint8_t> row(dim);
  for (uint32_t i = 0; i < num; ++i) {
    uint32_t row_dim{};
    reader.read(reinterpret_cast<char *>(&row_dim), 4);
    reader.read(reinterpret_cast<char *>(row.data()), dim);
    for (uint32_t j = 0; j < dim; ++j) {
      data[i * dim + j] = static_cast<T>(row[j]);
    }
  }
}

template <typename T>
void save_ivecs(const std::filesystem::path &filepath, const T *data, uint32_t num, uint32_t dim) {
  std::ofstream writer(filepath, std::ios::binary);
  if (!writer.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filepath.string());
  }

  writer.write(reinterpret_cast<const char *>(&num), 4);
  for (uint32_t i = 0; i < num; ++i) {
    writer.write(reinterpret_cast<const char *>(&dim), 4);
    writer.write(reinterpret_cast<const char *>(data + (i * dim)), dim * sizeof(T));
  }
}

}  // namespace alaya
