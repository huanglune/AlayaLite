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

namespace alaya::diskann {

using VisitedVersionType = uint16_t;

class VisitedList {
 public:
  VisitedList() = default;

  VisitedList(const VisitedList &) = delete;
  auto operator=(const VisitedList &) -> VisitedList & = delete;

  void reset(size_t num_elements) {
    if (visited_.size() < num_elements) {
      visited_.resize(num_elements, 0);
      version_ = 0;
    }

    version_++;

    // Handle overflow wrap-around (triggered every 65536 queries)
    if (version_ == 0) {
      std::memset(visited_.data(), 0, visited_.size() * sizeof(VisitedVersionType));
      version_ = 1;
    }
  }

  [[nodiscard]] __attribute__((always_inline)) auto is_visited(uint32_t id) const -> bool {
    assert(id < visited_.size());
    return visited_[id] == version_;
  }

  __attribute__((always_inline)) void mark(uint32_t id) {
    assert(id < visited_.size());
    visited_[id] = version_;
  }

  void prefetch(uint32_t id) const { __builtin_prefetch(&visited_[id], 1, 3); }

 private:
  std::vector<VisitedVersionType> visited_;
  VisitedVersionType version_{0};
};

struct GlobalVisitedList {
  static auto get(size_t num_elements) -> VisitedList & {
    // static thread_local:
    // 1. static: Only the first run will initialize
    // 2. thread_local: Each thread has its own copy, independent of others
    static thread_local VisitedList vl;

    // Reset on each retrieval (version iteration)
    vl.reset(num_elements);
    return vl;
  }
};

}  // namespace alaya::diskann
