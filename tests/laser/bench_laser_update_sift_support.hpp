// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace alaya::laser::bench {

// Bench-only, file-backed view of an fbin matrix. The complete file is mapped
// into virtual address space so row() remains a trivial stable-pointer lookup;
// callers release completed sequential batch windows with discard_rows().
// Thus resident base pages remain reclaimable cgroup file pages rather than an
// anonymous std::vector allocation.
class MappedFloatMatrix {
 public:
  explicit MappedFloatMatrix(const std::string &path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      const int error = errno;
      ::close(fd);
      throw std::runtime_error("cannot stat " + path + ": " + std::strerror(error));
    }
    if (st.st_size < static_cast<off_t>(2 * sizeof(int32_t)) ||
        static_cast<uintmax_t>(st.st_size) > std::numeric_limits<size_t>::max()) {
      ::close(fd);
      throw std::runtime_error("bad fbin size: " + path);
    }

    mapped_bytes_ = static_cast<size_t>(st.st_size);
    mapping_ = ::mmap(nullptr, mapped_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    const int mmap_error = errno;
    ::close(fd);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      throw std::runtime_error("cannot mmap " + path + ": " + std::strerror(mmap_error));
    }

    int32_t header_n = 0;
    int32_t header_dim = 0;
    std::memcpy(&header_n, mapping_, sizeof(header_n));
    std::memcpy(&header_dim,
                static_cast<const char *>(mapping_) + sizeof(header_n),
                sizeof(header_dim));
    if (header_n <= 0 || header_dim <= 0) {
      reset();
      throw std::runtime_error("bad fbin header: " + path);
    }

    n = static_cast<uint32_t>(header_n);
    dim = static_cast<uint32_t>(header_dim);
    const size_t rows = static_cast<size_t>(n);
    const size_t width = static_cast<size_t>(dim);
    if (rows > std::numeric_limits<size_t>::max() / width ||
        rows * width > (std::numeric_limits<size_t>::max() - 2 * sizeof(int32_t)) / sizeof(float)) {
      reset();
      throw std::runtime_error("fbin shape overflows address space: " + path);
    }
    const size_t required = 2 * sizeof(int32_t) + rows * width * sizeof(float);
    if (mapped_bytes_ < required) {
      reset();
      throw std::runtime_error("short read: " + path);
    }
    data_ =
        reinterpret_cast<const float *>(static_cast<const char *>(mapping_) + 2 * sizeof(int32_t));

    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
      reset();
      throw std::runtime_error("cannot determine mmap page size");
    }
    page_size_ = static_cast<size_t>(page_size);
    if (::madvise(mapping_, mapped_bytes_, MADV_SEQUENTIAL) != 0) {
      const int error = errno;
      reset();
      throw std::runtime_error("MADV_SEQUENTIAL failed for " + path + ": " + std::strerror(error));
    }
  }

  MappedFloatMatrix(const MappedFloatMatrix &) = delete;
  auto operator=(const MappedFloatMatrix &) -> MappedFloatMatrix & = delete;

  ~MappedFloatMatrix() { reset(); }

  [[nodiscard]] const float *row(size_t i) const {
    if (i >= n) {
      throw std::out_of_range("MappedFloatMatrix::row outside matrix");
    }
    return data_ + i * static_cast<size_t>(dim);
  }

  // Drop only complete pages behind a finished sequential batch. Keeping the
  // page containing end_row avoids faulting unrelated rows in the next batch.
  void discard_rows(size_t begin_row, size_t end_row) const {
    if (begin_row > end_row || end_row > n) {
      throw std::out_of_range("MappedFloatMatrix::discard_rows outside matrix");
    }
    if (begin_row == end_row) return;

    const uintptr_t begin = reinterpret_cast<uintptr_t>(row(begin_row));
    const uintptr_t end = end_row == n ? reinterpret_cast<uintptr_t>(data_) +
                                             static_cast<size_t>(n) * dim * sizeof(float)
                                       : reinterpret_cast<uintptr_t>(row(end_row));
    const uintptr_t advised_begin = begin - begin % page_size_;
    const uintptr_t advised_end = end - end % page_size_;
    if (advised_end <= advised_begin) return;
    if (::madvise(reinterpret_cast<void *>(advised_begin),
                  advised_end - advised_begin,
                  MADV_DONTNEED) != 0) {
      throw std::runtime_error(std::string("MADV_DONTNEED failed: ") + std::strerror(errno));
    }
  }

  uint32_t n = 0;
  uint32_t dim = 0;

 private:
  void reset() noexcept {
    if (mapping_ != nullptr) {
      ::munmap(mapping_, mapped_bytes_);
      mapping_ = nullptr;
    }
    data_ = nullptr;
    mapped_bytes_ = 0;
  }

  void *mapping_ = nullptr;
  const float *data_ = nullptr;
  size_t mapped_bytes_ = 0;
  size_t page_size_ = 0;
};

inline size_t parse_cache_cap_pages(const std::string &value) {
  if (value.empty() || value.front() == '-') {
    throw std::invalid_argument("bad --cache_cap_pages value: " + value);
  }
  size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("bad --cache_cap_pages value: " + value);
  }
  return static_cast<size_t>(parsed);
}

template <typename UpdateParamsT>
void apply_cache_cap_pages(size_t requested, UpdateParamsT &params) {
  if (requested != 0) params.cache_cap_pages = requested;
}

}  // namespace alaya::laser::bench
