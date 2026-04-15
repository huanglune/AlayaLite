/**
 * @file laser_search_context.hpp
 * @brief Zero-allocation search context for Laser index queries.
 *
 * LaserSearchContext provides typed accessors to pre-allocated scratch buffers
 * owned by ThreadData. It is constructed as a lightweight view at query start
 * and passed by reference through the entire search call chain.
 *
 * All per-query and per-node heap allocations are eliminated by reusing these
 * pre-allocated buffers with generation-tagged reset.
 */

#pragma once

#include <libaio.h>

#include <cstdint>

#include "index/laser/io/aligned_file_reader.hpp"
#include "index/laser/laser_types.hpp"

namespace symqg {

/**
 * @brief Aggregates all buffer pointers and config for search context initialization.
 * Replaces the 14-parameter init() call with a single struct.
 */
struct SearchContextBuffers {
  // Buffer pointers (borrowed from ThreadData — no ownership)
  uint8_t *lut_ = nullptr;
  float *rotated_query_ = nullptr;
  uint8_t *byte_query_ = nullptr;
  uint16_t *scan_result_ = nullptr;
  float *scan_float_ = nullptr;
  float *appro_dist_ = nullptr;
  io_event *io_events_ = nullptr;
  AlignedRead *frontier_reqs_ = nullptr;
  iocb *iocb_buf_ = nullptr;
  iocb **iocb_ptrs_ = nullptr;

  // Dimension and capacity config
  size_t padded_dim_ = 0;
  size_t degree_bound_ = 0;
  size_t max_beam_width_ = 0;
  size_t visited_capacity_ = 0;
};

class LaserSearchContext {
 public:
  LaserSearchContext() = default;

  /**
   * @brief Construct a context as a view over pre-allocated buffers.
   * All pointers are borrowed from ThreadData — no ownership.
   */
  void init(const SearchContextBuffers &bufs) {
    lut_ = bufs.lut_;
    rotated_query_ = bufs.rotated_query_;
    byte_query_ = bufs.byte_query_;
    scan_result_ = bufs.scan_result_;
    scan_float_ = bufs.scan_float_;
    appro_dist_ = bufs.appro_dist_;
    io_events_ = bufs.io_events_;
    frontier_reqs_ = bufs.frontier_reqs_;
    iocb_buf_ = bufs.iocb_buf_;
    iocb_ptrs_buf_ = bufs.iocb_ptrs_;

    padded_dim_ = bufs.padded_dim_;
    degree_bound_ = bufs.degree_bound_;
    max_beam_width_ = bufs.max_beam_width_;

    ongoing_table_ = OngoingTable(2 * bufs.max_beam_width_);
    visited_set_ = TaggedVisitedSet(bufs.visited_capacity_);
    prepared_ring_ = FixedRingBuffer<std::pair<PID, char *>>(2 * bufs.max_beam_width_);
    free_slot_stack_ = FixedStack<char *>(2 * bufs.max_beam_width_);
    cache_nhoods_.reserve(bufs.max_beam_width_);
  }

  /**
   * @brief O(1) reset between queries. Increments generation tags.
   */
  void reset() {
    ongoing_table_.reset();
    visited_set_.reset();
    prepared_ring_.reset();
    free_slot_stack_.reset_empty();
    cache_nhoods_.clear();
  }

  // Scratch buffer accessors (no allocation, returns pre-allocated memory)
  [[nodiscard]] auto lut() -> uint8_t * { return lut_; }
  [[nodiscard]] auto rotated_query() -> float * { return rotated_query_; }
  [[nodiscard]] auto byte_query() -> uint8_t * { return byte_query_; }
  [[nodiscard]] auto scan_result() -> uint16_t * { return scan_result_; }
  [[nodiscard]] auto scan_float() -> float * { return scan_float_; }
  [[nodiscard]] auto appro_dist() -> float * { return appro_dist_; }
  [[nodiscard]] auto io_events() -> io_event * { return io_events_; }
  [[nodiscard]] auto frontier_reqs() -> AlignedRead * { return frontier_reqs_; }
  [[nodiscard]] auto iocb_buf() -> iocb * { return iocb_buf_; }
  [[nodiscard]] auto iocb_ptrs_buf() -> iocb ** { return iocb_ptrs_buf_; }

  // Generation-tagged data structures
  [[nodiscard]] auto ongoing_table() -> OngoingTable & { return ongoing_table_; }
  [[nodiscard]] auto visited_set() -> TaggedVisitedSet & { return visited_set_; }
  [[nodiscard]] auto prepared_ring() -> FixedRingBuffer<std::pair<PID, char *>> & {
    return prepared_ring_;
  }
  [[nodiscard]] auto free_slot_stack() -> FixedStack<char *> & { return free_slot_stack_; }
  [[nodiscard]] auto cache_nhoods() -> std::vector<std::pair<PID, char *>> & {
    return cache_nhoods_;
  }
  [[nodiscard]] auto result_buffer() -> ResultBuffer & { return result_buffer_; }

 private:
  // Borrowed buffer pointers (owned by ThreadData)
  uint8_t *lut_ = nullptr;
  float *rotated_query_ = nullptr;
  uint8_t *byte_query_ = nullptr;
  uint16_t *scan_result_ = nullptr;
  float *scan_float_ = nullptr;
  float *appro_dist_ = nullptr;
  io_event *io_events_ = nullptr;
  AlignedRead *frontier_reqs_ = nullptr;
  iocb *iocb_buf_ = nullptr;
  iocb **iocb_ptrs_buf_ = nullptr;

  size_t padded_dim_ = 0;
  size_t degree_bound_ = 0;
  size_t max_beam_width_ = 0;

  // Zero-alloc data structures with generation-tagged reset
  OngoingTable ongoing_table_;
  TaggedVisitedSet visited_set_;
  FixedRingBuffer<std::pair<PID, char *>> prepared_ring_;
  FixedStack<char *> free_slot_stack_;
  std::vector<std::pair<PID, char *>> cache_nhoods_;
  ResultBuffer result_buffer_{0};
};

}  // namespace symqg
