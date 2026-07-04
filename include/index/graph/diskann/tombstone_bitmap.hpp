// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file tombstone_bitmap.hpp
 * @brief Deleted-slot bitmap for in-place DiskANN updates.
 *
 * One bit per slot id: `set(id)` marks a slot deleted (tombstoned), `clear(id)`
 * marks it live, `is_deleted(id)` queries the status. Out-of-range ids read as
 * *live*, so a freshly appended slot beyond the current capacity is implicitly
 * live without forcing a grow.
 *
 * Concurrency: mutators (set/clear) are expected to run under the index's
 * exclusive update lock, but `is_deleted()` is called lock-free from hot
 * reconnect paths while other threads allocate ("dark" tombstoned slots) and
 * publish. Storage is therefore pointer-stable and word-atomic: a fixed table
 * of atomically published 8 KiB chunks. Growing allocates and publishes a new
 * chunk; existing chunks never move, so lock-free readers can never observe a
 * reallocation (the previous flat-vector storage relocated under readers).
 * Reads are relaxed — a racing reader may see a bit a moment early or late,
 * which the update protocol already tolerates; it must never fault.
 *
 * Searches don't walk the live bitmap: they take a `TombstoneSnapshot` (flat,
 * immutable copy) under the update lock's shared side.
 *
 * The on-disk format is unchanged: an explicit little-endian
 * [magic | n_words | words] blob, decoupled from the in-memory representation.
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya::diskann {

static_assert(std::endian::native == std::endian::little,
              "TombstoneBitmap on-disk format assumes little-endian host");

/// Immutable flat copy of a TombstoneBitmap, taken under the update lock and
/// handed to searches. Cheap to query, trivially assignable/reusable.
class TombstoneSnapshot {
 public:
  [[nodiscard]] bool is_deleted(uint64_t id) const {
    if (id >= num_bits_) {
      return false;
    }
    return ((words_[id >> 6] >> (id & 63)) & uint64_t{1}) != 0;
  }

  [[nodiscard]] uint64_t capacity() const { return num_bits_; }

  void reset(uint64_t n_words) {
    words_.assign(n_words, 0);
    num_bits_ = n_words * 64;
  }

  [[nodiscard]] uint64_t *data() { return words_.data(); }

 private:
  std::vector<uint64_t> words_;
  uint64_t num_bits_ = 0;
};

class TombstoneBitmap {
 public:
  static constexpr uint64_t kMagic = 0x414C5954424D5031ULL;  // "ALYTBMP1"

  /// 8 KiB chunks (2^16 bits); a full table addresses every uint32 slot id.
  static constexpr uint64_t kChunkBitsLog2 = 16;
  static constexpr uint64_t kChunkBits = uint64_t{1} << kChunkBitsLog2;
  static constexpr uint64_t kChunkWords = kChunkBits / 64;
  static constexpr uint64_t kMaxChunks = uint64_t{1} << 16;

  TombstoneBitmap() : table_(new ChunkTable()) {}

  /// Pre-size the bitmap to hold at least @p initial_bits (all live).
  explicit TombstoneBitmap(uint64_t initial_bits) : TombstoneBitmap() {
    ensure_capacity(initial_bits);
  }

  TombstoneBitmap(const TombstoneBitmap &) = delete;
  TombstoneBitmap &operator=(const TombstoneBitmap &) = delete;

  ~TombstoneBitmap() { release_chunks(); }

  /// Drop all state and pre-size for @p initial_bits (all live). Exclusive use
  /// only (load/reset paths) — not safe against concurrent readers.
  void reset(uint64_t initial_bits) {
    release_chunks();
    num_bits_.store(0, std::memory_order_relaxed);
    count_.store(0, std::memory_order_relaxed);
    ensure_capacity(initial_bits);
  }

  /// Mark slot @p id as deleted (grows the bitmap if @p id is out of range).
  /// Runs under the caller's exclusive update lock; publication of a new chunk
  /// is release-ordered so lock-free readers see a fully zeroed chunk.
  void set(uint64_t id) {
    Chunk *chunk = chunk_grow(id);
    const uint64_t mask = uint64_t{1} << (id & 63);
    const uint64_t old =
        chunk->words[(id >> 6) & (kChunkWords - 1)].fetch_or(mask, std::memory_order_relaxed);
    if ((old & mask) == 0) {
      count_.fetch_add(1, std::memory_order_relaxed);
    }
  }

  /// Mark slot @p id as live. Clearing an out-of-range id is a no-op (already
  /// live), so no allocation is forced.
  void clear(uint64_t id) {
    Chunk *chunk = chunk_at(id);
    if (chunk == nullptr) {
      return;
    }
    const uint64_t mask = uint64_t{1} << (id & 63);
    const uint64_t old =
        chunk->words[(id >> 6) & (kChunkWords - 1)].fetch_and(~mask, std::memory_order_relaxed);
    if ((old & mask) != 0) {
      count_.fetch_sub(1, std::memory_order_relaxed);
    }
  }

  /// True iff slot @p id is tombstoned. Out-of-range ids are live (false).
  /// Lock-free and safe against concurrent set/clear/grow.
  [[nodiscard]] bool is_deleted(uint64_t id) const {
    const Chunk *chunk = chunk_at(id);
    if (chunk == nullptr) {
      return false;
    }
    const uint64_t w = chunk->words[(id >> 6) & (kChunkWords - 1)].load(std::memory_order_relaxed);
    return ((w >> (id & 63)) & uint64_t{1}) != 0;
  }

  /// Number of tombstoned (set) bits.
  [[nodiscard]] uint64_t count() const { return count_.load(std::memory_order_relaxed); }

  /// Bit capacity (chunk-aligned, >= the highest id ever touched + 1).
  [[nodiscard]] uint64_t capacity() const { return num_bits_.load(std::memory_order_relaxed); }

  /// Flat immutable copy for searches. Callers synchronize against mutators
  /// (the index takes the update lock's shared side).
  void snapshot_into(TombstoneSnapshot &out) const {
    const uint64_t bits = capacity();
    const uint64_t n_words = bits / 64;
    out.reset(n_words);
    uint64_t *dst = out.data();
    for (uint64_t w = 0; w < n_words; ++w) {
      const Chunk *chunk = (*table_)[w / kChunkWords].load(std::memory_order_acquire);
      dst[w] = chunk == nullptr
                   ? 0
                   : chunk->words[w & (kChunkWords - 1)].load(std::memory_order_relaxed);
    }
  }

  /// Write [magic | num_words | words] into @p out. Reusable so an enclosing
  /// container (e.g. SlotAllocator) can embed the bitmap in its own file.
  void serialize(std::ostream &out) const {
    const uint64_t magic = kMagic;
    const uint64_t n_words = capacity() / 64;
    out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char *>(&n_words), sizeof(n_words));
    std::vector<uint64_t> buf(kChunkWords);
    for (uint64_t w = 0; w < n_words; w += kChunkWords) {
      const Chunk *chunk = (*table_)[w / kChunkWords].load(std::memory_order_acquire);
      const uint64_t span = std::min(kChunkWords, n_words - w);
      for (uint64_t i = 0; i < span; ++i) {
        buf[i] = chunk == nullptr ? 0 : chunk->words[i].load(std::memory_order_relaxed);
      }
      out.write(reinterpret_cast<const char *>(buf.data()),
                static_cast<std::streamsize>(span * sizeof(uint64_t)));
    }
  }

  /// Read state previously written by serialize(). The deleted count is
  /// recomputed via popcount, so it is never trusted from disk. Exclusive use
  /// only (load paths).
  void deserialize(std::istream &in) {
    uint64_t magic = 0;
    uint64_t n_words = 0;
    in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char *>(&n_words), sizeof(n_words));
    if (!in || magic != kMagic) {
      throw std::runtime_error("TombstoneBitmap::deserialize: bad magic/truncated");
    }
    reset(0);
    ensure_capacity(n_words * 64);
    uint64_t recount = 0;
    std::vector<uint64_t> buf(kChunkWords);
    for (uint64_t w = 0; w < n_words; w += kChunkWords) {
      const uint64_t span = std::min(kChunkWords, n_words - w);
      in.read(reinterpret_cast<char *>(buf.data()),
              static_cast<std::streamsize>(span * sizeof(uint64_t)));
      if (!in) {
        throw std::runtime_error("TombstoneBitmap::deserialize: words truncated");
      }
      Chunk *chunk = (*table_)[w / kChunkWords].load(std::memory_order_relaxed);
      for (uint64_t i = 0; i < span; ++i) {
        chunk->words[i].store(buf[i], std::memory_order_relaxed);
        recount += static_cast<uint64_t>(std::popcount(buf[i]));
      }
    }
    count_.store(recount, std::memory_order_relaxed);
  }

  /// Persist to a standalone @p path (truncating).
  void save(const std::string &path) const {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("TombstoneBitmap::save: cannot open " + path);
    }
    serialize(out);
    if (!out) {
      throw std::runtime_error("TombstoneBitmap::save: write failed " + path);
    }
  }

  /// Restore from a standalone @p path written by save().
  void load(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      throw std::runtime_error("TombstoneBitmap::load: cannot open " + path);
    }
    deserialize(in);
  }

 private:
  struct Chunk {
    std::array<std::atomic<uint64_t>, kChunkWords> words{};
  };
  using ChunkTable = std::array<std::atomic<Chunk *>, kMaxChunks>;

  /// Published chunk holding bit @p id, or nullptr (= all live). Lock-free;
  /// the chunk table is fixed-size, so readers never race a table relocation.
  [[nodiscard]] Chunk *chunk_at(uint64_t id) const {
    const uint64_t c = id >> kChunkBitsLog2;
    if (c >= kMaxChunks) {
      return nullptr;
    }
    return (*table_)[c].load(std::memory_order_acquire);
  }

  /// chunk_at, allocating-and-publishing the chunk if absent (writer paths).
  [[nodiscard]] Chunk *chunk_grow(uint64_t id) {
    const uint64_t c = id >> kChunkBitsLog2;
    if (c >= kMaxChunks) {
      throw std::out_of_range("TombstoneBitmap: id beyond addressable range");
    }
    Chunk *chunk = (*table_)[c].load(std::memory_order_acquire);
    if (chunk != nullptr) {
      return chunk;
    }
    auto fresh = std::make_unique<Chunk>();
    Chunk *expected = nullptr;
    if ((*table_)[c].compare_exchange_strong(expected,
                                             fresh.get(),
                                             std::memory_order_release,
                                             std::memory_order_acquire)) {
      chunk = fresh.release();
      bump_capacity((c + 1) * kChunkBits);
      return chunk;
    }
    return expected;  // another writer published it first
  }

  /// Grow so that bit index (bits-1) is addressable. New chunks are zero (live).
  void ensure_capacity(uint64_t bits) {
    if (bits == 0) {
      return;
    }
    for (uint64_t c = 0; c * kChunkBits < bits; ++c) {
      (void)chunk_grow(c * kChunkBits);
    }
    bump_capacity((bits + kChunkBits - 1) / kChunkBits * kChunkBits);
  }

  void bump_capacity(uint64_t bits) {
    uint64_t cur = num_bits_.load(std::memory_order_relaxed);
    while (cur < bits && !num_bits_.compare_exchange_weak(cur,
                                                          bits,
                                                          std::memory_order_release,
                                                          std::memory_order_relaxed)) {
    }
  }

  void release_chunks() {
    for (auto &slot : *table_) {
      delete slot.exchange(nullptr, std::memory_order_relaxed);
    }
  }

  std::unique_ptr<ChunkTable> table_;
  std::atomic<uint64_t> num_bits_{0};
  std::atomic<uint64_t> count_{0};
};

}  // namespace alaya::diskann
