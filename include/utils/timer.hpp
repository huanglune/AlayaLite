// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <chrono>
#include <cstdint>

namespace alaya {
class Timer {
  using clock_ = std::chrono::steady_clock;
  clock_::time_point m_beg_;

 public:
  Timer() : m_beg_(clock_::now()) {}

  void reset() { m_beg_ = clock_::now(); }

  // returns elapsed time in `us`
  [[nodiscard]] auto elapsed() const -> uint64_t {
    const auto elapsed_time = clock_::now() - m_beg_;
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed_time).count());
  }

  [[nodiscard]] auto elapsed_us() const -> double { return static_cast<double>(elapsed()); }
  [[nodiscard]] auto elapsed_ms() const -> double { return elapsed_us() / 1'000.0; }
  [[nodiscard]] auto elapsed_s() const -> double { return elapsed_us() / 1'000'000.0; }
};

}  // namespace alaya
