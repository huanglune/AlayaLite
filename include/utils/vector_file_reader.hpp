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

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "storage/io/io_uring_engine.hpp"

namespace alaya {

namespace detail {

struct FbinVectorFormat {
  static void init(int fd,
                   const std::string &path,
                   uint32_t &dim,
                   uint32_t &num_vectors,
                   size_t &vector_stride,
                   size_t &vector_bytes,
                   uint64_t &vector_offset_base) {
    int32_t hdr[2]{};
    if (::pread(fd, hdr, sizeof(hdr), 0) != static_cast<ssize_t>(sizeof(hdr)) || hdr[0] <= 0 ||
        hdr[1] <= 0) {
      throw std::runtime_error("Invalid fbin: " + path);
    }
    num_vectors = static_cast<uint32_t>(hdr[0]);
    dim = static_cast<uint32_t>(hdr[1]);
    vector_bytes = static_cast<size_t>(dim) * sizeof(float);
    vector_stride = vector_bytes;
    vector_offset_base = sizeof(hdr);
  }
};

struct FvecsVectorFormat {
  static void init(int fd,
                   const std::string &path,
                   uint32_t &dim,
                   uint32_t &num_vectors,
                   size_t &vector_stride,
                   size_t &vector_bytes,
                   uint64_t &vector_offset_base) {
    int32_t dim_raw = 0;
    if (::pread(fd, &dim_raw, sizeof(dim_raw), 0) != static_cast<ssize_t>(sizeof(dim_raw)) ||
        dim_raw <= 0) {
      throw std::runtime_error("Invalid fvecs: " + path);
    }

    dim = static_cast<uint32_t>(dim_raw);
    vector_bytes = static_cast<size_t>(dim) * sizeof(float);
    vector_stride = sizeof(int32_t) + vector_bytes;
    vector_offset_base = sizeof(int32_t);

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      throw std::runtime_error("Failed to stat fvecs file: " + path);
    }
    auto file_size = static_cast<size_t>(st.st_size);
    if (file_size % vector_stride != 0) {
      throw std::runtime_error("Corrupted fvecs file: " + path);
    }
    num_vectors = static_cast<uint32_t>(file_size / vector_stride);
  }
};

}  // namespace detail

template <typename Format>
class BasicVectorFileReader {
 public:
  BasicVectorFileReader() = default;
  ~BasicVectorFileReader() { close_fd(); }

  BasicVectorFileReader(const BasicVectorFileReader &) = delete;
  auto operator=(const BasicVectorFileReader &) -> BasicVectorFileReader & = delete;
  BasicVectorFileReader(BasicVectorFileReader &&) = delete;
  auto operator=(BasicVectorFileReader &&) -> BasicVectorFileReader & = delete;

  void open(const std::string &path) {
    close_fd();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Cannot open: " + path);
    }

    try {
      Format::init(fd_,
                   path,
                   dim_,
                   num_vectors_,
                   vector_stride_,
                   vector_bytes_,
                   vector_offset_base_);
    } catch (...) {
      close_fd();
      throw;
    }

    try {
      engine_ = std::make_unique<IOUringEngine>(kQueueDepth);
    } catch (...) {
      engine_.reset();
    }
  }

  void read_by_ids(const uint32_t *ids, uint32_t count, float *out) {
    batch_read(
        count,
        [&](uint32_t i) {
          return ids[i];
        },
        out);
  }

  void read_sequential(uint32_t start, uint32_t count, float *out) {
    batch_read(
        count,
        [&](uint32_t i) {
          return start + i;
        },
        out);
  }

  [[nodiscard]] auto dim() const -> uint32_t { return dim_; }
  [[nodiscard]] auto num_vectors() const -> uint32_t { return num_vectors_; }

 private:
  int fd_{-1};
  uint32_t dim_{0};
  uint32_t num_vectors_{0};
  size_t vector_stride_{0};
  size_t vector_bytes_{0};
  uint64_t vector_offset_base_{0};
  std::unique_ptr<IOUringEngine> engine_;
  static constexpr uint32_t kQueueDepth = 256;

  void close_fd() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  template <typename IndexFn>
  void batch_read(uint32_t count, IndexFn index_fn, float *out) {
    if (!engine_) {
      for (uint32_t i = 0; i < count; ++i) {
        auto vector_id = static_cast<uint64_t>(index_fn(i));
        auto offset = vector_offset_base_ + vector_id * vector_stride_;
        auto bytes = ::pread(fd_,
                             out + static_cast<size_t>(i) * dim_,
                             vector_bytes_,
                             static_cast<off_t>(offset));
        if (bytes != static_cast<ssize_t>(vector_bytes_)) {
          throw std::runtime_error("Failed to read vector payload");
        }
      }
      return;
    }

    constexpr uint32_t kBatch = 256;
    std::vector<IORequest> reqs(kBatch);
    for (uint32_t s = 0; s < count; s += kBatch) {
      auto n = std::min(kBatch, count - s);
      for (uint32_t i = 0; i < n; ++i) {
        auto vector_id = static_cast<uint64_t>(index_fn(s + i));
        auto offset = vector_offset_base_ + vector_id * vector_stride_;
        reqs[i] = IORequest(out + static_cast<size_t>(s + i) * dim_, vector_bytes_, offset);
      }
      auto span = std::span<IORequest>(reqs.data(), n);
      engine_->wait(engine_->submit_reads(fd_, span), -1);
    }
  }
};

using FbinFileReader = BasicVectorFileReader<detail::FbinVectorFormat>;
using FvecsFileReader = BasicVectorFileReader<detail::FvecsVectorFormat>;

class FloatVectorFileReader {
 public:
  void open(const std::filesystem::path &path) { open(path.string()); }

  void open(const std::string &path) {
    auto ext = std::filesystem::path(path).extension().string();
    if (ext == ".fvecs") {
      fvecs_.open(path);
      active_reader_ = ActiveReader::kFvecs;
      return;
    }
    if (ext == ".fbin") {
      fbin_.open(path);
      active_reader_ = ActiveReader::kFbin;
      return;
    }
    throw std::invalid_argument("Unsupported vector file extension: " + ext +
                                " (expected .fbin or .fvecs)");
  }

  void read_by_ids(const uint32_t *ids, uint32_t count, float *out) {
    switch (active_reader_) {
      case ActiveReader::kFbin:
        fbin_.read_by_ids(ids, count, out);
        return;
      case ActiveReader::kFvecs:
        fvecs_.read_by_ids(ids, count, out);
        return;
      case ActiveReader::kNone:
        throw std::runtime_error("FloatVectorFileReader is not opened");
    }
  }

  void read_sequential(uint32_t start, uint32_t count, float *out) {
    switch (active_reader_) {
      case ActiveReader::kFbin:
        fbin_.read_sequential(start, count, out);
        return;
      case ActiveReader::kFvecs:
        fvecs_.read_sequential(start, count, out);
        return;
      case ActiveReader::kNone:
        throw std::runtime_error("FloatVectorFileReader is not opened");
    }
  }

  [[nodiscard]] auto dim() const -> uint32_t {
    switch (active_reader_) {
      case ActiveReader::kFbin:
        return fbin_.dim();
      case ActiveReader::kFvecs:
        return fvecs_.dim();
      case ActiveReader::kNone:
        throw std::runtime_error("FloatVectorFileReader is not opened");
    }
    throw std::runtime_error("FloatVectorFileReader has invalid reader state");
  }

  [[nodiscard]] auto num_vectors() const -> uint32_t {
    switch (active_reader_) {
      case ActiveReader::kFbin:
        return fbin_.num_vectors();
      case ActiveReader::kFvecs:
        return fvecs_.num_vectors();
      case ActiveReader::kNone:
        throw std::runtime_error("FloatVectorFileReader is not opened");
    }
    throw std::runtime_error("FloatVectorFileReader has invalid reader state");
  }

 private:
  enum class ActiveReader : uint8_t {
    kNone = 0,
    kFbin,
    kFvecs,
  };

  FbinFileReader fbin_;
  FvecsFileReader fvecs_;
  ActiveReader active_reader_{ActiveReader::kNone};
};

}  // namespace alaya
