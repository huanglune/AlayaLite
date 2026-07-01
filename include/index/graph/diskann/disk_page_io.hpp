// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file disk_page_io.hpp
 * @brief Sector-aligned read-modify-write of node records in diskann.index.
 *
 * `DiskPageIO` (design D6) is the single place that writes the disk index during
 * in-place updates. It owns its own `O_DIRECT | O_RDWR` file descriptor (separate
 * from the search reader's read-only descriptor); in serial-update mode (a global
 * mutex serialises search and update) the two descriptors never race, and because
 * both use O_DIRECT a completed pwrite is visible to a subsequent pread.
 *
 * Each operation read-modify-writes one sector-aligned page so co-resident nodes
 * survive updates. Appends extend the file with ftruncate. Linux O_DIRECT is
 * required for the private syscall path; non-Linux update calls throw loudly.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
  #include <fcntl.h>
  #include <sys/stat.h>
  #include <unistd.h>
#endif

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/disk_page_cache.hpp"
#include "index/graph/laser/utils/memory.hpp"

namespace alaya::diskann {

class DiskPageIO {
 public:
  /// A decoded node record: raw coordinates plus the live neighbor id list.
  struct NodeData {
    std::vector<float> coords;   ///< dim float32 entries
    std::vector<uint32_t> nbrs;  ///< n_nbrs neighbor ids
  };

  DiskPageIO(const std::string &index_path,
             const DiskLayoutGeometry &geom,
             size_t page_cache_capacity = 0)
      : geom_(geom), page_cache_(page_cache_capacity) {
    try {
      page_buf_ =
          static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
      flush_buf_ =
          static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
      open_rw(index_path);  // sets fd_ + file_size_
    } catch (...) {
      close_fd();
      free_buffers();
      throw;
    }
  }

  ~DiskPageIO() {
    close_fd();
    free_buffers();
  }

  DiskPageIO(const DiskPageIO &) = delete;
  DiskPageIO &operator=(const DiskPageIO &) = delete;
  DiskPageIO(DiskPageIO &&) = delete;
  DiskPageIO &operator=(DiskPageIO &&) = delete;

  /// Read the full node record (coords + neighbors) of @p id.
  NodeData read_node(uint32_t id) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    read_page(id);
    const NodeRecordView view{page_buf_ + geom_.offset_to_node(id), geom_.dim};
    NodeData d;
    d.coords.assign(view.coords(), view.coords() + geom_.dim);
    const uint32_t n = view.n_nbrs();
    d.nbrs.assign(view.nbrs(), view.nbrs() + n);
    return d;
  }

  /// Read only neighbor lists for a delete batch. IDs sharing a disk page reuse
  /// the loaded page buffer, avoiding repeated O_DIRECT reads and coords copies.
  std::vector<std::vector<uint32_t>> read_neighbors_batch(const uint32_t *ids, uint32_t count) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    std::vector<std::vector<uint32_t>> out(count);
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return geom_.get_page_offset(ids[a]) < geom_.get_page_offset(ids[b]);
    });
    bool page_loaded = false;
    uint64_t loaded_off = 0;
    for (const uint32_t pos : order) {
      const uint32_t id = ids[pos];
      const uint64_t off = geom_.get_page_offset(id);
      if (!page_loaded || off != loaded_off) {
        read_page(id);
        loaded_off = off;
        page_loaded = true;
      }
      out[pos] = neighbors_from_loaded_page(id);
    }
    return out;
  }

  /// Parallel variant for Yi-style delete batches. It bypasses the shared page
  /// buffer and page cache; callers must serialize it against writes.
  std::vector<std::vector<uint32_t>> read_neighbors_batch_parallel(const uint32_t *ids,
                                                                   uint32_t count,
                                                                   uint32_t threads) {
#if defined(__linux__)
    if (count == 0) {
      return {};
    }
    if (threads <= 1) {
      return read_neighbors_batch(ids, count);
    }
    std::vector<uint32_t> order = page_sorted_order(ids, count);
    std::vector<NeighborPageGroup> groups = neighbor_page_groups(ids, order);
    const uint32_t workers =
        std::min<uint32_t>({threads, static_cast<uint32_t>(groups.size()), count});
    if (workers <= 1) {
      return read_neighbors_batch(ids, count);
    }

    std::vector<std::vector<uint32_t>> out(count);
    std::exception_ptr error;
    std::mutex error_mutex;
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (uint32_t worker = 0; worker < workers; ++worker) {
      pool.emplace_back([&, worker]() {
        try {
          read_neighbor_group_stride(ids, order, groups, worker, workers, out);
        } catch (...) {
          std::lock_guard<std::mutex> lock(error_mutex);
          if (error == nullptr) {
            error = std::current_exception();
          }
        }
      });
    }
    for (auto &thread : pool) {
      thread.join();
    }
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    return out;
#else
    (void)threads;
    return read_neighbors_batch(ids, count);
#endif
  }

  /// Write a complete node record `[coords | n_nbrs | nbr_ids]`. Read-modify-write
  /// that preserves co-resident nodes; extends the file for a new-append slot.
  void write_node(uint32_t id, const float *coords, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node: n_nbrs exceeds max_degree");
    }
    load_page_for_write(id);
    char *rec = page_buf_ + geom_.offset_to_node(id);
    std::memset(rec, 0, geom_.node_len);  // clear stale bytes (esp. a reused slot's tail)
    pack_node_record(rec, coords, nbr_ids, n_nbrs, geom_.dim);
    write_page(id);
    vec_cache_[id].assign(coords, coords + geom_.dim);  // keep the coords cache coherent
  }

  /// Overwrite only the `n_nbrs` + `nbr_ids` fields, leaving coords untouched.
  void write_node_neighbors(uint32_t id, uint32_t n_nbrs, const uint32_t *nbr_ids) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    if (n_nbrs > geom_.max_degree) {
      throw std::invalid_argument("DiskPageIO::write_node_neighbors: n_nbrs exceeds max_degree");
    }
    read_page(id);  // must read first to preserve coords + co-resident nodes
    char *rec = page_buf_ + geom_.offset_to_node(id);
    const uint64_t coords_bytes = geom_.dim * sizeof(float);
    // Zero the whole neighbor region first so no stale trailing ids survive.
    std::memset(rec + coords_bytes, 0, geom_.node_len - coords_bytes);
    std::memcpy(rec + coords_bytes, &n_nbrs, sizeof(n_nbrs));
    if (n_nbrs > 0) {
      std::memcpy(rec + coords_bytes + sizeof(uint32_t),
                  nbr_ids,
                  static_cast<size_t>(n_nbrs) * sizeof(uint32_t));
    }
    write_page(id);
  }

  /// Coords of @p id, served from the transient cache when present (design D8).
  std::vector<float> read_coords_cached(uint32_t id) {
    std::lock_guard<std::mutex> lock(io_mutex_);
    auto it = vec_cache_.find(id);
    if (it != vec_cache_.end()) {
      return it->second;
    }
    read_page(id);
    const NodeRecordView view{page_buf_ + geom_.offset_to_node(id), geom_.dim};
    auto res = vec_cache_.emplace(id, std::vector<float>(view.coords(), view.coords() + geom_.dim));
    return res.first->second;
  }

  /// Drop the transient coords cache (call between independent update operations).
  void clear_cache() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    vec_cache_.clear();
  }

  /// Write all dirty cached pages to disk; clean pages stay cached.
  void flush_dirty_pages() {
    std::lock_guard<std::mutex> lock(io_mutex_);
    flush_dirty_pages_unlocked();
  }

  [[nodiscard]] uint64_t file_size() const { return file_size_; }
  [[nodiscard]] const DiskLayoutGeometry &geometry() const { return geom_; }

 private:
  void flush_dirty_pages_unlocked() {
    page_cache_.flush_dirty([this](uint64_t page_off, const char *page) {
      write_page_to_disk(page_off, page);
    });
  }

  std::vector<uint32_t> neighbors_from_loaded_page(uint32_t id) const {
    const NodeRecordView view{page_buf_ + geom_.offset_to_node(id), geom_.dim};
    const uint32_t n = view.n_nbrs();
    return std::vector<uint32_t>(view.nbrs(), view.nbrs() + n);
  }

  struct NeighborPageGroup {
    uint32_t begin = 0;
    uint32_t end = 0;
  };

  std::vector<uint32_t> page_sorted_order(const uint32_t *ids, uint32_t count) const {
    std::vector<uint32_t> order(count);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return geom_.get_page_offset(ids[a]) < geom_.get_page_offset(ids[b]);
    });
    return order;
  }

  std::vector<NeighborPageGroup> neighbor_page_groups(const uint32_t *ids,
                                                      const std::vector<uint32_t> &order) const {
    std::vector<NeighborPageGroup> groups;
    for (uint32_t begin = 0; begin < order.size();) {
      uint32_t end = begin + 1;
      const uint64_t page_off = geom_.get_page_offset(ids[order[begin]]);
      while (end < order.size() && geom_.get_page_offset(ids[order[end]]) == page_off) {
        ++end;
      }
      groups.push_back(NeighborPageGroup{begin, end});
      begin = end;
    }
    return groups;
  }

  std::vector<uint32_t> neighbors_from_page(const char *page, uint32_t id) const {
    const NodeRecordView view{page + geom_.offset_to_node(id), geom_.dim};
    const uint32_t n = view.n_nbrs();
    return std::vector<uint32_t>(view.nbrs(), view.nbrs() + n);
  }

#if defined(__linux__)
  void read_neighbor_group_stride(const uint32_t *ids,
                                  const std::vector<uint32_t> &order,
                                  const std::vector<NeighborPageGroup> &groups,
                                  uint32_t worker,
                                  uint32_t workers,
                                  std::vector<std::vector<uint32_t>> &out) const {
    char *buf =
        static_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(geom_.page_size));
    try {
      for (uint32_t group_id = worker; group_id < groups.size(); group_id += workers) {
        const NeighborPageGroup group = groups[group_id];
        const uint32_t first_id = ids[order[group.begin]];
        const uint64_t page_off = geom_.get_page_offset(first_id);
        const ssize_t r = ::pread(fd_, buf, geom_.page_size, static_cast<off_t>(page_off));
        if (r != static_cast<ssize_t>(geom_.page_size)) {
          throw std::runtime_error("DiskPageIO::read_neighbors_batch_parallel: short/failed pread");
        }
        for (uint32_t pos = group.begin; pos < group.end; ++pos) {
          const uint32_t out_pos = order[pos];
          out[out_pos] = neighbors_from_page(buf, ids[out_pos]);
        }
      }
      alaya::laser::memory::align_free(buf);
    } catch (...) {
      alaya::laser::memory::align_free(buf);
      throw;
    }
  }
#endif

  /// Ensure the page holding @p id exists, with page_buf_ holding its current
  /// content (existing page: RMW) or zeros (freshly extended page).
  void load_page_for_write(uint32_t id) {
    const uint64_t page_off = geom_.get_page_offset(id);
    const uint64_t page_end = page_off + geom_.page_size;
    if (page_end <= file_size_) {
      read_page(id);
    } else {
      extend_to(page_end);                         // ftruncate; OS zero-fills the new region
      std::memset(page_buf_, 0, geom_.page_size);  // fresh in-memory page
    }
  }

  // ---- platform-gated syscalls (Linux O_DIRECT) ----
  void open_rw(const std::string &path);
  void read_page(uint32_t id);
  void write_page(uint32_t id);
  void write_page_to_disk(uint64_t page_off, const char *page);
  void extend_to(uint64_t new_size);
  void close_fd();

  void free_buffers() {
    if (page_buf_ != nullptr) {
      alaya::laser::memory::align_free(page_buf_);
      page_buf_ = nullptr;
    }
    if (flush_buf_ != nullptr) {
      alaya::laser::memory::align_free(flush_buf_);
      flush_buf_ = nullptr;
    }
  }

  DiskLayoutGeometry geom_;
  int fd_ = -1;
  uint64_t file_size_ = 0;
  char *page_buf_ = nullptr;
  char *flush_buf_ = nullptr;
  DiskPageCache page_cache_;
  std::unordered_map<uint32_t, std::vector<float>> vec_cache_;
  mutable std::mutex io_mutex_;
};

#if defined(__linux__)

inline void DiskPageIO::open_rw(const std::string &path) {
  fd_ = ::open(path.c_str(), O_DIRECT | O_RDWR);  // NOLINT(hicpp-vararg)
  if (fd_ < 0) {
    throw std::runtime_error("DiskPageIO::open_rw: cannot open (O_DIRECT|O_RDWR) " + path);
  }
  struct stat st{};
  if (::fstat(fd_, &st) != 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("DiskPageIO::open_rw: fstat failed " + path);
  }
  file_size_ = static_cast<uint64_t>(st.st_size);
}

inline void DiskPageIO::read_page(uint32_t id) {
  const uint64_t off = geom_.get_page_offset(id);
  if (page_cache_.read(off, page_buf_, geom_.page_size)) {
    return;
  }
  const ssize_t r = ::pread(fd_, page_buf_, geom_.page_size, static_cast<off_t>(off));
  if (r != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::read_page: short/failed pread at " + std::to_string(off) +
                             " (got " + std::to_string(r) + ")");
  }
  page_cache_.write(off,
                    page_buf_,
                    geom_.page_size,
                    false,
                    [this](uint64_t page_off, const char *page) {
                      write_page_to_disk(page_off, page);
                    });
}

inline void DiskPageIO::write_page(uint32_t id) {
  const uint64_t off = geom_.get_page_offset(id);
  if (page_cache_.enabled()) {
    page_cache_.write(off,
                      page_buf_,
                      geom_.page_size,
                      true,
                      [this](uint64_t page_off, const char *page) {
                        write_page_to_disk(page_off, page);
                      });
    return;
  }
  write_page_to_disk(off, page_buf_);
}

inline void DiskPageIO::write_page_to_disk(uint64_t page_off, const char *page) {
  const char *write_buf = page;
  if (page != page_buf_) {
    std::memcpy(flush_buf_, page, geom_.page_size);
    write_buf = flush_buf_;
  }
  const ssize_t w = ::pwrite(fd_, write_buf, geom_.page_size, static_cast<off_t>(page_off));
  if (w != static_cast<ssize_t>(geom_.page_size)) {
    throw std::runtime_error("DiskPageIO::write_page: short/failed pwrite at " +
                             std::to_string(page_off) + " (got " + std::to_string(w) + ")");
  }
}

inline void DiskPageIO::extend_to(uint64_t new_size) {
  if (::ftruncate(fd_, static_cast<off_t>(new_size)) != 0) {
    throw std::runtime_error("DiskPageIO::extend_to: ftruncate failed to " +
                             std::to_string(new_size));
  }
  file_size_ = new_size;
}

inline void DiskPageIO::close_fd() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

#else  // !__linux__ : in-place updates require Linux O_DIRECT — fail loudly when used.

inline void DiskPageIO::open_rw(const std::string &) {
  throw std::runtime_error("DiskPageIO: in-place DiskANN updates require Linux (O_DIRECT)");
}
inline void DiskPageIO::read_page(uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::write_page(uint32_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::write_page_to_disk(uint64_t, const char *) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::extend_to(uint64_t) {
  throw std::runtime_error("DiskPageIO: unsupported platform (needs Linux O_DIRECT)");
}
inline void DiskPageIO::close_fd() {}

#endif  // __linux__

}  // namespace alaya::diskann
