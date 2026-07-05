// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file search_scratch.hpp
 * @brief Per-thread scratch state for the DiskANN cached beam search.
 *
 * Each search thread borrows one @c ThreadData from a @c ConcurrentQueue for
 * the duration of a query (LASER pattern, design D8). It bundles:
 *   - the visited bitset (dedup of popped nodes),
 *   - the @c retset frontier (NeighborPriorityQueue, reused from Vamana),
 *   - @c exact_dists: exact L2 distances of nodes actually read from disk/cache,
 *   - @c pq_table: the per-query @c n_chunks x 256 PQ distance table (PQ mode),
 *   - @c sector_scratch: a sector-aligned double buffer for async page reads,
 *   - @c ctx_: the thread's AlignedFileReader I/O context.
 *
 * The scratch is owned by the index and reused across queries; reset_query()
 * clears only the per-query state.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/visited_bitset.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/vamana/robust_prune.hpp"

namespace alaya::diskann {

struct ThreadDataScratchConfig {
  uint64_t n_page_slots = 0;
  uint64_t page_size = 0;
  uint32_t pq_table_entries = 0;
  uint64_t max_slot_id = 0;
  uint32_t max_degree = 0;
  uint32_t search_list_size = 0;
  uint64_t query_dim = 0;
};

struct NeighborScratchView {
  const uint32_t *data_ptr = nullptr;
  uint32_t len = 0;

  [[nodiscard]] const uint32_t *data() const { return data_ptr; }
  [[nodiscard]] size_t size() const { return len; }
  [[nodiscard]] const uint32_t *begin() const { return data_ptr; }
  [[nodiscard]] const uint32_t *end() const {
    return data_ptr == nullptr ? nullptr : data_ptr + len;
  }
};

struct InFlightSlot {
  bool occupied = false;
  uint32_t id = 0;
  uint64_t page_slot = 0;
  const char *record = nullptr;
};

struct ThreadData {
  // --- Per-query mutable search state ---
  VisitedBitset visited_bits;                   ///< ids popped/seeded
  alaya::vamana::NeighborPriorityQueue retset;  ///< exploration frontier
  std::vector<float> exact_dists;               ///< node id -> exact L2 sqr or NaN
  std::vector<uint32_t> exact_dirty;            ///< exact_dists entries written this query
  std::vector<float> pq_table;                  ///< n_chunks*256 (empty if no PQ)
  std::vector<float> pq_qres;                   ///< dim floats for PQ query residual
  std::vector<uint32_t> nbrs_buf;               ///< contiguous cached neighbor lists
  std::vector<std::pair<uint32_t, uint32_t>> nbrs_offsets;  ///< id -> (start, len)
  std::vector<uint32_t> nbrs_dirty;                         ///< offsets written this query
  uint32_t nbrs_buf_next = 0;
  uint32_t nbrs_slot_len = 0;
  std::vector<uint32_t> nbrs_free_offsets;
  std::vector<InFlightSlot> inflight;  ///< indexed by scratch page slot
  std::vector<AlignedRead> io_reqs;
  std::vector<AlignedReadEvent> io_events;
  std::vector<uint64_t> free_page_slots;
  std::vector<AlignedRead> rerank_reqs;

  // --- I/O scratch (allocated once, reused) ---
  char *sector_scratch = nullptr;  ///< n_page_slots * page_size bytes, sector-aligned.
                                   ///< One page slot per in-flight read; the No-PQ pipeline
                                   ///< depth equals the slot count (PQ caps in-flight at
                                   ///< beam_width, so it ignores the surplus).
  uint64_t sector_scratch_bytes = 0;
  char *wave_scratch = nullptr;  ///< Reactor wave buffer for the async update search
                                 ///< (one page per neighbor of an expansion). Kept apart
                                 ///< from sector_scratch so the sync pipelines' depth —
                                 ///< which equals the sector slot count — is untouched.
  uint64_t wave_scratch_bytes = 0;
  IOContext ctx_{};  ///< AIO context (owned via reader.register_thread())

  void resize_slot_capacity(uint64_t max_slot_id) {
    if (max_slot_id <= visited_bits.size_bits()) {
      return;
    }
    visited_bits.resize(max_slot_id);
    exact_dists.resize(max_slot_id, std::numeric_limits<float>::quiet_NaN());
    nbrs_offsets.resize(max_slot_id, {0, 0});
  }

  /// Reset only the per-query state; keeps allocated buffers.
  void reset_query(size_t search_list_size) {
    visited_bits.clear();
    reset_exact_dists();
    reset_neighbors();
    clear_inflight();
    retset.reserve(search_list_size);
    retset.clear();
  }

  /// Allocate the sector page buffer, flat hot-path structures, and PQ scratch.
  void alloc_scratch(const ThreadDataScratchConfig &config) {
    if (config.n_page_slots == 0 || config.page_size == 0 || config.max_slot_id == 0) {
      throw std::invalid_argument("ThreadData::alloc_scratch: invalid zero-sized config");
    }
    sector_scratch_bytes = config.n_page_slots * config.page_size;
    sector_scratch = reinterpret_cast<char *>(
        alaya::laser::memory::align_allocate<kSectorLen>(sector_scratch_bytes));
    if (config.pq_table_entries > 0) {
      pq_table.assign(config.pq_table_entries, 0.0f);
    }
    if (config.query_dim > 0) {
      pq_qres.assign(config.query_dim, 0.0f);
    }
    visited_bits.resize(config.max_slot_id);
    exact_dists.assign(config.max_slot_id, std::numeric_limits<float>::quiet_NaN());
    exact_dirty.reserve(config.search_list_size);
    nbrs_slot_len = config.max_degree;
    nbrs_buf.assign(static_cast<size_t>(config.search_list_size) * nbrs_slot_len, 0);
    nbrs_offsets.assign(config.max_slot_id, {0, 0});
    nbrs_dirty.reserve(config.search_list_size);
    nbrs_free_offsets.reserve(config.search_list_size);
    inflight.assign(config.n_page_slots, {});
    io_reqs.reserve(config.n_page_slots);
    io_events.reserve(config.n_page_slots);
    free_page_slots.reserve(config.n_page_slots);
    rerank_reqs.resize(1);
    reset_neighbors();
  }

  /// Grow the wave buffer to at least @p bytes (sector-aligned; contents lost).
  void ensure_wave_scratch(uint64_t bytes) {
    if (wave_scratch_bytes >= bytes) {
      return;
    }
    if (wave_scratch != nullptr) {
      alaya::laser::memory::align_free(wave_scratch);
      wave_scratch = nullptr;
      wave_scratch_bytes = 0;
    }
    wave_scratch =
        reinterpret_cast<char *>(alaya::laser::memory::align_allocate<kSectorLen>(bytes));
    wave_scratch_bytes = bytes;
  }

  void set_exact_dist(uint32_t id, float distance) {
    if (id >= exact_dists.size()) {
      throw std::out_of_range("ThreadData::set_exact_dist: id out of range");
    }
    if (is_missing_exact(exact_dists[id])) {
      exact_dirty.push_back(id);
    }
    exact_dists[id] = distance;
  }

  [[nodiscard]] float exact_dist(uint32_t id) const {
    if (id >= exact_dists.size()) {
      throw std::out_of_range("ThreadData::exact_dist: id out of range");
    }
    return exact_dists[id];
  }

  [[nodiscard]] static bool is_missing_exact(float value) {
    static constexpr uint32_t kExponentMask = 0x7F800000u;
    static constexpr uint32_t kMantissaMask = 0x007FFFFFu;
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & kExponentMask) == kExponentMask && (bits & kMantissaMask) != 0;
  }

  void cache_neighbors(uint32_t id, const uint32_t *nbrs, uint32_t len) {
    if (id >= nbrs_offsets.size()) {
      throw std::out_of_range("ThreadData::cache_neighbors: id out of range");
    }
    if (len > nbrs_slot_len) {
      throw std::runtime_error("ThreadData::cache_neighbors: neighbor list exceeds slot size");
    }
    if (len == 0) {
      nbrs_offsets[id] = {0, 0};
      return;
    }
    if (nbrs_free_offsets.empty()) {
      throw std::runtime_error("ThreadData::cache_neighbors: neighbor scratch exhausted");
    }
    const uint32_t start = nbrs_free_offsets.back();
    nbrs_free_offsets.pop_back();
    std::copy_n(nbrs, len, nbrs_buf.begin() + start);
    nbrs_offsets[id] = {start, len};
    nbrs_dirty.push_back(id);
    nbrs_buf_next += nbrs_slot_len;
  }

  [[nodiscard]] NeighborScratchView cached_neighbors(uint32_t id) const {
    if (id >= nbrs_offsets.size()) {
      throw std::out_of_range("ThreadData::cached_neighbors: id out of range");
    }
    const auto [start, len] = nbrs_offsets[id];
    return {len == 0 ? nullptr : nbrs_buf.data() + start, len};
  }

  void release_cached_neighbors(uint32_t id) {
    if (id >= nbrs_offsets.size()) {
      throw std::out_of_range("ThreadData::release_cached_neighbors: id out of range");
    }
    const auto [start, len] = nbrs_offsets[id];
    if (len == 0) {
      return;
    }
    nbrs_offsets[id] = {0, 0};
    nbrs_free_offsets.push_back(start);
    nbrs_buf_next -= nbrs_slot_len;
  }

  void set_inflight(uint64_t page_slot, uint32_t id, const char *record) {
    if (page_slot >= inflight.size()) {
      throw std::out_of_range("ThreadData::set_inflight: page slot out of range");
    }
    inflight[page_slot] = {true, id, page_slot, record};
  }

  [[nodiscard]] bool has_inflight() const {
    return std::any_of(inflight.begin(), inflight.end(), [](const InFlightSlot &slot) {
      return slot.occupied;
    });
  }

  [[nodiscard]] bool remove_inflight(uint32_t id, InFlightSlot &out) {
    for (InFlightSlot &slot : inflight) {
      if (!slot.occupied || slot.id != id) {
        continue;
      }
      out = slot;
      slot.occupied = false;
      return true;
    }
    return false;
  }

  /// Release the sector buffer (vectors free with the ThreadData instance).
  void free_scratch() {
    if (sector_scratch != nullptr) {
      alaya::laser::memory::align_free(sector_scratch);
      sector_scratch = nullptr;
    }
    sector_scratch_bytes = 0;
    if (wave_scratch != nullptr) {
      alaya::laser::memory::align_free(wave_scratch);
      wave_scratch = nullptr;
    }
    wave_scratch_bytes = 0;
  }

 private:
  void reset_exact_dists() {
    for (const uint32_t id : exact_dirty) {
      exact_dists[id] = std::numeric_limits<float>::quiet_NaN();
    }
    exact_dirty.clear();
  }

  void reset_neighbors() {
    for (const uint32_t id : nbrs_dirty) {
      nbrs_offsets[id] = {0, 0};
    }
    nbrs_dirty.clear();
    nbrs_buf_next = 0;
    nbrs_free_offsets.clear();
    if (nbrs_slot_len == 0) {
      return;
    }
    const uint32_t n_slots = static_cast<uint32_t>(nbrs_buf.size() / nbrs_slot_len);
    for (uint32_t slot = 0; slot < n_slots; ++slot) {
      nbrs_free_offsets.push_back(slot * nbrs_slot_len);
    }
  }

  void clear_inflight() {
    for (InFlightSlot &slot : inflight) {
      slot.occupied = false;
    }
  }
};

}  // namespace alaya::diskann
