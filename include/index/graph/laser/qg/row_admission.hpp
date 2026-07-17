// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <bit>
#include <cstdint>
#include <unordered_set>
#include <vector>

// Segment admission contract: one predicate seam for user filters and
// tombstone visibility. See docs/design/segment-admission-contract.md.
//
// This header is a leaf: it depends only on the standard library (no
// collection/core includes), so it can be used from the kernel's hottest
// loop without pulling in the Collection layer.
namespace alaya::laser {

// RowAdmission v1 = include-semantics bitmap over a segment's row/PID
// capacity (bit set -> admissible). A POD *view*: it never owns storage.
// Callers materialize the backing std::vector<uint64_t> (or supply an
// already word-aligned payload) and keep it alive for at least the
// duration of one query's kernel calls.
//
// Non-atomic by design: per-call admission is an immutable snapshot for the
// duration of one query. The contract's phase-separation guarantee (search
// never overlaps a mutation phase on the same segment view) is what makes
// that safe -- see contract section 4.
struct RowAdmission {
  const uint64_t *words = nullptr;
  uint64_t capacity = 0;  // number of representable bit positions (rows/PIDs)
  uint64_t popcount = 0;  // cached count of set bits (planner density hook; unused in v1)

  // Bit test with a defensive bounds check: a pid at or beyond `capacity`
  // (e.g. a row admitted into the segment after this snapshot was built) is
  // reported inadmissible rather than reading out of bounds.
  [[nodiscard]] auto test(uint64_t pid) const noexcept -> bool {
    if (words == nullptr || pid >= capacity) {
      return false;
    }
    const uint64_t word = words[pid >> 6U];
    return ((word >> (pid & 63U)) & 1ULL) != 0ULL;
  }
};

[[nodiscard]] inline auto admission_words_for_capacity(uint64_t capacity) noexcept -> uint64_t {
  return (capacity + 63U) >> 6U;
}

[[nodiscard]] inline auto admission_popcount_words(const uint64_t *words,
                                                    uint64_t num_words) noexcept -> uint64_t {
  uint64_t total = 0;
  for (uint64_t i = 0; i < num_words; ++i) {
    total += static_cast<uint64_t>(std::popcount(words[i]));
  }
  return total;
}

// Zero out bit positions >= capacity in the trailing (partial) word so
// popcount and any bulk scan over `storage` stay exact. Assumes
// storage.size() == admission_words_for_capacity(capacity).
inline void admission_clear_tail_bits(std::vector<uint64_t> &storage, uint64_t capacity) noexcept {
  const uint64_t rem = capacity & 63U;
  if (rem != 0 && !storage.empty()) {
    const uint64_t mask = (uint64_t{1} << rem) - 1U;
    storage.back() &= mask;
  }
}

// Wrap an externally-owned, word-aligned bitmap payload with zero copy.
// `payload_size` is in bytes (matches core::SegmentFilterView.payload_size).
// The caller is responsible for validating alignment and that the payload
// covers at least admission_words_for_capacity(capacity) words -- this
// factory trusts the precondition and only clamps the popcount scan so a
// short buffer cannot be read out of bounds.
[[nodiscard]] inline auto admission_from_bitmap_payload(const void *payload,
                                                         uint64_t payload_size,
                                                         uint64_t capacity) -> RowAdmission {
  const auto *words = static_cast<const uint64_t *>(payload);
  const uint64_t needed_words = admission_words_for_capacity(capacity);
  const uint64_t available_words = payload_size / sizeof(uint64_t);
  const uint64_t scannable_words = available_words < needed_words ? available_words : needed_words;
  RowAdmission admission;
  admission.words = words;
  admission.capacity = capacity;
  admission.popcount = words == nullptr ? 0 : admission_popcount_words(words, scannable_words);
  return admission;
}

// Materialize a bitmap from an explicit row/PID list (deduplicated
// implicitly: setting the same bit twice does not double-count popcount).
// `storage` is caller-owned and must outlive the returned view.
[[nodiscard]] inline auto admission_from_sorted_rows(const uint64_t *rows,
                                                      uint64_t n,
                                                      uint64_t capacity,
                                                      std::vector<uint64_t> &storage)
    -> RowAdmission {
  storage.assign(admission_words_for_capacity(capacity), uint64_t{0});
  uint64_t set_count = 0;
  for (uint64_t i = 0; i < n; ++i) {
    const uint64_t row = rows[i];
    if (row >= capacity) {
      continue;  // defensive: a row outside capacity cannot be represented
    }
    const uint64_t mask = uint64_t{1} << (row & 63U);
    uint64_t &word = storage[row >> 6U];
    if ((word & mask) == 0ULL) {
      word |= mask;
      ++set_count;
    }
  }
  RowAdmission admission;
  admission.words = storage.data();
  admission.capacity = capacity;
  admission.popcount = set_count;
  return admission;
}

// Materialize the complement of an exclude set (e.g. QGUpdater's tombstone
// set): every row in [0, capacity) is admissible except those named in
// `excluded`. This is the migration/testing adapter that lets the legacy
// hash-probe exclude-set seam and the v1 bitmap seam be compared directly
// (contract acceptance #2, tombstone parity) -- it is not meant to run in
// the per-query hot path, where the live bitmap is maintained incrementally
// instead.
template <typename T>
[[nodiscard]] inline auto admission_from_exclude_set(const std::unordered_set<T> &excluded,
                                                      uint64_t capacity,
                                                      std::vector<uint64_t> &storage)
    -> RowAdmission {
  storage.assign(admission_words_for_capacity(capacity), ~uint64_t{0});
  admission_clear_tail_bits(storage, capacity);
  for (const auto &row : excluded) {
    const auto r = static_cast<uint64_t>(row);
    if (r >= capacity) {
      continue;
    }
    storage[r >> 6U] &= ~(uint64_t{1} << (r & 63U));
  }
  RowAdmission admission;
  admission.words = storage.data();
  admission.capacity = capacity;
  admission.popcount = admission_popcount_words(storage.data(), storage.size());
  return admission;
}

// Compose two admission sources (e.g. user filter AND live/tombstone
// bitmap) into one, at O(capacity/64). Per contract section 3, this is a
// setup-time cost paid once per query, never per candidate. Absent inputs
// (words == nullptr) are treated as all-zero rather than all-admissible --
// callers that want "one source only" semantics skip this call entirely
// and hand the kernel that single RowAdmission directly (contract: "if
// only one source is active it is used directly, no copy").
[[nodiscard]] inline auto admission_and(const RowAdmission &a,
                                        const RowAdmission &b,
                                        std::vector<uint64_t> &storage) -> RowAdmission {
  const uint64_t capacity = a.capacity < b.capacity ? a.capacity : b.capacity;
  const uint64_t num_words = admission_words_for_capacity(capacity);
  storage.assign(num_words, uint64_t{0});
  for (uint64_t i = 0; i < num_words; ++i) {
    const uint64_t aw = a.words != nullptr ? a.words[i] : uint64_t{0};
    const uint64_t bw = b.words != nullptr ? b.words[i] : uint64_t{0};
    storage[i] = aw & bw;
  }
  admission_clear_tail_bits(storage, capacity);
  RowAdmission result;
  result.words = storage.data();
  result.capacity = capacity;
  result.popcount = admission_popcount_words(storage.data(), storage.size());
  return result;
}

}  // namespace alaya::laser
