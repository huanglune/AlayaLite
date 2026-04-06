/**
 * @file laser_search_context.hpp
 * @brief Zero-allocation search context for Laser index queries.
 *
 * LaserSearchContext provides typed accessors to pre-allocated scratch buffers
 * owned by ThreadDate. It is constructed as a lightweight view at query start
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
#include "index/laser/utils/search_buffer.hpp"

namespace symqg {

class LaserSearchContext {
   public:
    LaserSearchContext() = default;

    /**
     * @brief Construct a context as a view over pre-allocated buffers.
     * All pointers are borrowed from ThreadDate — no ownership.
     */
    void init(
        uint8_t* lut_buf,
        float* rotated_query_buf,
        uint8_t* byte_query_buf,
        uint16_t* scan_result_buf,
        float* scan_float_buf,
        float* appro_dist_buf,
        io_event* io_events_buf,
        AlignedRead* frontier_reqs_buf,
        iocb* iocb_buf,
        iocb** iocb_ptrs_buf,
        size_t padded_dim,
        size_t degree_bound,
        size_t max_beam_width,
        size_t visited_capacity
    ) {
        lut_ = lut_buf;
        rotated_query_ = rotated_query_buf;
        byte_query_ = byte_query_buf;
        scan_result_ = scan_result_buf;
        scan_float_ = scan_float_buf;
        appro_dist_ = appro_dist_buf;
        io_events_ = io_events_buf;
        frontier_reqs_ = frontier_reqs_buf;
        iocb_buf_ = iocb_buf;
        iocb_ptrs_buf_ = iocb_ptrs_buf;

        padded_dim_ = padded_dim;
        degree_bound_ = degree_bound;
        max_beam_width_ = max_beam_width;

        ongoing_table_ = OngoingTable(2 * max_beam_width);
        visited_set_ = TaggedVisitedSet(visited_capacity);
        prepared_ring_ = FixedRingBuffer<std::pair<PID, char*>>(2 * max_beam_width);
        free_slot_stack_ = FixedStack<char*>(2 * max_beam_width);
        cache_nhoods_.reserve(max_beam_width);
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
    [[nodiscard]] auto lut() -> uint8_t* { return lut_; }
    [[nodiscard]] auto rotated_query() -> float* { return rotated_query_; }
    [[nodiscard]] auto byte_query() -> uint8_t* { return byte_query_; }
    [[nodiscard]] auto scan_result() -> uint16_t* { return scan_result_; }
    [[nodiscard]] auto scan_float() -> float* { return scan_float_; }
    [[nodiscard]] auto appro_dist() -> float* { return appro_dist_; }
    [[nodiscard]] auto io_events() -> io_event* { return io_events_; }
    [[nodiscard]] auto frontier_reqs() -> AlignedRead* { return frontier_reqs_; }
    [[nodiscard]] auto iocb_buf() -> iocb* { return iocb_buf_; }
    [[nodiscard]] auto iocb_ptrs_buf() -> iocb** { return iocb_ptrs_buf_; }

    // Generation-tagged data structures
    [[nodiscard]] auto ongoing_table() -> OngoingTable& { return ongoing_table_; }
    [[nodiscard]] auto visited_set() -> TaggedVisitedSet& { return visited_set_; }
    [[nodiscard]] auto prepared_ring() -> FixedRingBuffer<std::pair<PID, char*>>& {
        return prepared_ring_;
    }
    [[nodiscard]] auto free_slot_stack() -> FixedStack<char*>& { return free_slot_stack_; }
    [[nodiscard]] auto cache_nhoods()
        -> std::vector<std::pair<PID, char*>>& { return cache_nhoods_; }
    [[nodiscard]] auto result_buffer() -> buffer::ResultBuffer& { return result_buffer_; }

   private:
    // Borrowed buffer pointers (owned by ThreadDate)
    uint8_t* lut_ = nullptr;
    float* rotated_query_ = nullptr;
    uint8_t* byte_query_ = nullptr;
    uint16_t* scan_result_ = nullptr;
    float* scan_float_ = nullptr;
    float* appro_dist_ = nullptr;
    io_event* io_events_ = nullptr;
    AlignedRead* frontier_reqs_ = nullptr;
    iocb* iocb_buf_ = nullptr;
    iocb** iocb_ptrs_buf_ = nullptr;

    size_t padded_dim_ = 0;
    size_t degree_bound_ = 0;
    size_t max_beam_width_ = 0;

    // Zero-alloc data structures with generation-tagged reset
    OngoingTable ongoing_table_;
    TaggedVisitedSet visited_set_;
    FixedRingBuffer<std::pair<PID, char*>> prepared_ring_;
    FixedStack<char*> free_slot_stack_;
    std::vector<std::pair<PID, char*>> cache_nhoods_;
    buffer::ResultBuffer result_buffer_{0};
};

}  // namespace symqg
