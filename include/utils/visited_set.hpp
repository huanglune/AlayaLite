/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * ...
 */

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

#include "utils/math.hpp"
#include "utils/platform.hpp"

namespace alaya {

class DenseVisitedSet {
 public:
  using VersionType = uint16_t;

  DenseVisitedSet() = default;

  DenseVisitedSet(const DenseVisitedSet &) = delete;
  auto operator=(const DenseVisitedSet &) -> DenseVisitedSet & = delete;
  DenseVisitedSet(DenseVisitedSet &&) = default;
  auto operator=(DenseVisitedSet &&) -> DenseVisitedSet & = default;
  ~DenseVisitedSet() = default;

  auto reset(size_t num_elements) -> void {
    if (visited_.size() < num_elements) {
      visited_.resize(num_elements, 0);
      version_ = 0;
    }

    ++version_;

    // Handle overflow wrap-around (triggered every 65536 queries)
    if (ALAYA_UNLIKELY(version_ == 0)) {
      std::memset(visited_.data(), 0, visited_.size() * sizeof(VersionType));
      version_ = 1;
    }
  }

  [[nodiscard]] __attribute__((always_inline)) auto is_visited(uint32_t id) const -> bool {
    assert(id < visited_.size());
    return visited_[id] == version_;
  }

  __attribute__((always_inline)) auto mark(uint32_t id) -> void {
    assert(id < visited_.size());
    visited_[id] = version_;
  }

  auto prefetch(uint32_t id) const -> void { __builtin_prefetch(&visited_[id], 1, 3); }

 private:
  std::vector<VersionType> visited_;
  VersionType version_{0};
};

class SparseVisitedSet {
 public:
  struct Entry {
    uint32_t id;   // NOLINT(readability-identifier-naming)
    uint32_t gen;  // NOLINT(readability-identifier-naming)
  };
  static_assert(sizeof(Entry) == 8, "Entry must be 8 bytes");

  SparseVisitedSet() = default;

  explicit SparseVisitedSet(size_t capacity)
      : capacity_(size_t{1} << alaya::math::ceil_log2(capacity)),
        mask_(capacity_ - 1),
        table_(capacity_) {
    std::memset(table_.data(), 0, capacity_ * sizeof(Entry));
  }

  auto reset() -> void {
    ++gen_;
    if (ALAYA_UNLIKELY(gen_ == 0)) {
      std::memset(table_.data(), 0, capacity_ * sizeof(Entry));
      gen_ = 1;
    }
  }

  [[nodiscard]] auto get(uint32_t id) const -> bool {
    size_t idx = id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      const auto &entry = table_[probe];
      if (entry.gen != gen_) {
        return false;
      }
      if (entry.id == id) {
        return true;
      }
    }
    return false;
  }

  auto set(uint32_t id) -> void {
    size_t idx = id & mask_;
    for (size_t i = 0; i < capacity_; ++i) {
      size_t probe = (idx + i) & mask_;
      auto &entry = table_[probe];
      if (entry.gen != gen_) {
        entry = {id, gen_};
        return;
      }
      if (entry.id == id) {
        return;
      }
    }
  }

 private:
  size_t capacity_ = 0;
  size_t mask_ = 0;
  uint32_t gen_ = 1;
  std::vector<Entry> table_;
};

struct GlobalDenseVisitedSet {
  static auto get(size_t num_elements) -> DenseVisitedSet & {
    static thread_local DenseVisitedSet visited;
    visited.reset(num_elements);
    return visited;
  }
};

}  // namespace alaya
