/**
 * @file thread_date.hpp
 * @brief Thread-affine memory owner for Laser search contexts.
 *
 * ThreadDate owns all memory for one search thread. It is allocated once
 * at set_params() time and recycled via a ConcurrentQueue pool.
 * LaserSearchContext is constructed as a lightweight view over ThreadDate
 * at query start.
 *
 * Memory layout:
 * - sector_scratch_: 2 * max_beam_width * page_size (aligned to kSectorLen)
 * - pca_query_scratch_: (dimension + residual_dimension) floats
 * - All LaserSearchContext buffers: LUT, rotated_query, byte_query,
 *   scan_result, scan_float, appro_dist, io_events, frontier_reqs
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include <utility>

#include "index/laser/laser_search_context.hpp"
#include "index/laser/utils/memory.hpp"
#include "index/laser/utils/search_buffer.hpp"

namespace symqg {

constexpr size_t kSectorLen = 4096;

struct ThreadDate {
  // I/O context for libaio
  io_context_t ctx_ = nullptr;

  // Search pool (beam search frontier)
  buffer::SearchBuffer search_pool_;

  // Sector-aligned scratch for async I/O reads
  char *sector_scratch_ = nullptr;

  // PCA-transformed query buffer
  float *pca_query_scratch_ = nullptr;

  // Pre-allocated search context buffers
  uint8_t *lut_buf_ = nullptr;
  float *rotated_query_buf_ = nullptr;
  uint8_t *byte_query_buf_ = nullptr;
  uint16_t *scan_result_buf_ = nullptr;
  float *scan_float_buf_ = nullptr;
  float *appro_dist_buf_ = nullptr;
  io_event *io_events_buf_ = nullptr;
  AlignedRead *frontier_reqs_buf_ = nullptr;
  iocb *iocb_buf_ = nullptr;
  iocb **iocb_ptrs_buf_ = nullptr;

  // Reusable search context (view over the above buffers)
  LaserSearchContext search_ctx_;

  ThreadDate() = default;
  ThreadDate(const ThreadDate &) = delete;
  auto operator=(const ThreadDate &) -> ThreadDate & = delete;

  ThreadDate(ThreadDate &&other) noexcept
      : ctx_(std::exchange(other.ctx_, nullptr)),
        search_pool_(std::move(other.search_pool_)),
        sector_scratch_(std::exchange(other.sector_scratch_, nullptr)),
        pca_query_scratch_(std::exchange(other.pca_query_scratch_, nullptr)),
        lut_buf_(std::exchange(other.lut_buf_, nullptr)),
        rotated_query_buf_(std::exchange(other.rotated_query_buf_, nullptr)),
        byte_query_buf_(std::exchange(other.byte_query_buf_, nullptr)),
        scan_result_buf_(std::exchange(other.scan_result_buf_, nullptr)),
        scan_float_buf_(std::exchange(other.scan_float_buf_, nullptr)),
        appro_dist_buf_(std::exchange(other.appro_dist_buf_, nullptr)),
        io_events_buf_(std::exchange(other.io_events_buf_, nullptr)),
        frontier_reqs_buf_(std::exchange(other.frontier_reqs_buf_, nullptr)),
        iocb_buf_(std::exchange(other.iocb_buf_, nullptr)),
        iocb_ptrs_buf_(std::exchange(other.iocb_ptrs_buf_, nullptr)),
        search_ctx_(std::move(other.search_ctx_)) {}

  auto operator=(ThreadDate &&other) noexcept -> ThreadDate & {
    if (this == &other) {
      return *this;
    }
    deallocate();
    ctx_ = std::exchange(other.ctx_, nullptr);
    search_pool_ = std::move(other.search_pool_);
    sector_scratch_ = std::exchange(other.sector_scratch_, nullptr);
    pca_query_scratch_ = std::exchange(other.pca_query_scratch_, nullptr);
    lut_buf_ = std::exchange(other.lut_buf_, nullptr);
    rotated_query_buf_ = std::exchange(other.rotated_query_buf_, nullptr);
    byte_query_buf_ = std::exchange(other.byte_query_buf_, nullptr);
    scan_result_buf_ = std::exchange(other.scan_result_buf_, nullptr);
    scan_float_buf_ = std::exchange(other.scan_float_buf_, nullptr);
    appro_dist_buf_ = std::exchange(other.appro_dist_buf_, nullptr);
    io_events_buf_ = std::exchange(other.io_events_buf_, nullptr);
    frontier_reqs_buf_ = std::exchange(other.frontier_reqs_buf_, nullptr);
    iocb_buf_ = std::exchange(other.iocb_buf_, nullptr);
    iocb_ptrs_buf_ = std::exchange(other.iocb_ptrs_buf_, nullptr);
    search_ctx_ = std::move(other.search_ctx_);
    return *this;
  }

  /**
   * @brief Allocate all buffers for the given search parameters.
   * Called once at set_params() time.
   */
  void allocate(size_t padded_dim,
                size_t degree_bound,
                size_t max_beam_width,
                size_t page_size,
                size_t ef_search,
                size_t full_dim,
                size_t num_points) {
    // Sector scratch: 2 * max_beam_width pages, sector-aligned
    sector_scratch_ = reinterpret_cast<char *>(
        memory::align_allocate<kSectorLen>(2 * max_beam_width * page_size));

    // PCA query scratch
    pca_query_scratch_ = new float[full_dim];

    // LUT buffer: padded_dim * 4 bytes (padded_dim / 4 sub-codebooks * 16 entries)
    lut_buf_ = new (std::align_val_t(64)) uint8_t[padded_dim * 4];

    // Rotated query: padded_dim floats, 64-byte aligned
    rotated_query_buf_ = new (std::align_val_t(64)) float[padded_dim];

    // Byte query: padded_dim bytes, 64-byte aligned
    byte_query_buf_ = new (std::align_val_t(64)) uint8_t[padded_dim];

    // Scan result: degree_bound uint16_t values
    scan_result_buf_ = new uint16_t[degree_bound];

    // Scan float: degree_bound floats
    scan_float_buf_ = new float[degree_bound];

    // Approximate distances: degree_bound floats
    appro_dist_buf_ = new float[degree_bound];

    // I/O events: max_beam_width events
    io_events_buf_ = new io_event[max_beam_width];

    // Frontier read requests: 2 * max_beam_width requests
    frontier_reqs_buf_ = new AlignedRead[2 * max_beam_width];

    // Pre-allocated iocb scratch for submit_reqs (avoids per-call heap alloc)
    iocb_buf_ = new iocb[2 * max_beam_width];
    iocb_ptrs_buf_ = new iocb *[2 * max_beam_width];

    // Search pool
    search_pool_.resize(ef_search);

    // Initialize search context as a view over the allocated buffers
    // Beam search correctness depends on being able to mark every visited node.
    // A smaller hash table silently drops inserts when full, which can cause
    // repeated revisits and effectively unbounded query time on larger graphs.
    size_t visited_capacity = num_points;
    search_ctx_.init(lut_buf_,
                     rotated_query_buf_,
                     byte_query_buf_,
                     scan_result_buf_,
                     scan_float_buf_,
                     appro_dist_buf_,
                     io_events_buf_,
                     frontier_reqs_buf_,
                     iocb_buf_,
                     iocb_ptrs_buf_,
                     padded_dim,
                     degree_bound,
                     max_beam_width,
                     visited_capacity);
  }

  /**
   * @brief Free all owned memory.
   */
  void deallocate() {
    if (sector_scratch_ != nullptr) {
      std::free(sector_scratch_);
      sector_scratch_ = nullptr;
    }
    delete[] pca_query_scratch_;
    pca_query_scratch_ = nullptr;

    ::operator delete[](lut_buf_, std::align_val_t(64));
    lut_buf_ = nullptr;

    ::operator delete[](rotated_query_buf_, std::align_val_t(64));
    rotated_query_buf_ = nullptr;

    ::operator delete[](byte_query_buf_, std::align_val_t(64));
    byte_query_buf_ = nullptr;

    delete[] scan_result_buf_;
    scan_result_buf_ = nullptr;

    delete[] scan_float_buf_;
    scan_float_buf_ = nullptr;

    delete[] appro_dist_buf_;
    appro_dist_buf_ = nullptr;

    delete[] io_events_buf_;
    io_events_buf_ = nullptr;

    delete[] frontier_reqs_buf_;
    frontier_reqs_buf_ = nullptr;

    delete[] iocb_buf_;
    iocb_buf_ = nullptr;

    delete[] iocb_ptrs_buf_;
    iocb_ptrs_buf_ = nullptr;
  }
};

}  // namespace symqg
