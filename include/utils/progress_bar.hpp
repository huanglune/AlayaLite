/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "utils/log.hpp"
#include "utils/macros.hpp"

namespace alaya {

/// @brief Thread-safe animated progress bar with spinner, color, and ETA.
///
/// Renders a tqdm/uv-style progress bar to stderr with:
///   - Braille dot spinner animation (~10 fps)
///   - ANSI color: green filled bar, yellow gradient edge, dim empty portion
///   - ETA estimation based on elapsed time
///   - Green checkmark on completion
///
/// Falls back to periodic LOG_INFO when stderr is not a terminal.
///
/// Usage:
/// @code
///   ProgressBar bar("Building index", total_count);
///   // from any thread:
///   bar.tick();
///   // automatic finish in destructor, or call bar.finish() explicitly
/// @endcode
class ProgressBar {
  using Clock = std::chrono::steady_clock;

 public:
  explicit ProgressBar(std::string prefix, uint64_t total)
      : prefix_(std::move(prefix)),
        total_(total),
        current_(0),
        start_time_(Clock::now()),
        last_render_us_(0),
        last_fallback_pct_(0),
        finished_(false),
        rendering_(false),
        is_tty_(isatty(fileno(stderr)) != 0) {
    if (is_tty_ && total_ > 0) {
      render_line(0, /*is_final=*/false);
    }
  }

  ALAYA_NON_COPYABLE_NON_MOVABLE(ProgressBar);

  ~ProgressBar() { finish(); }

  /// Thread-safe: increment progress by 1.
  void tick() {
    uint64_t cur = current_.fetch_add(1, std::memory_order_relaxed) + 1;

    if (is_tty_) {
      auto now_us =
          std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_time_).count();
      int64_t last_us = last_render_us_.load(std::memory_order_relaxed);

      if (now_us - last_us >= kRenderIntervalUs || cur >= total_) {
        bool expected = false;
        if (rendering_.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
          last_render_us_.store(now_us, std::memory_order_relaxed);
          render_line(cur, /*is_final=*/false);
          rendering_.store(false, std::memory_order_release);
        }
      }
    } else {
      // Non-TTY fallback: LOG_INFO every 10%
      if (total_ > 0) {
        uint64_t pct = cur * 100 / total_;
        uint64_t last = last_fallback_pct_.load(std::memory_order_relaxed);
        if (pct / 10 > last / 10 || cur >= total_) {
          last_fallback_pct_.store(pct, std::memory_order_relaxed);
          LOG_INFO("{}: [{}/{}] ({}%)", prefix_, cur, total_, pct);
        }
      }
    }
  }

  /// Finalize the bar: render completion state and print newline.
  /// Safe to call multiple times; only the first call takes effect.
  void finish() {
    if (finished_.exchange(true)) {
      return;
    }
    if (is_tty_) {
      render_line(current_.load(std::memory_order_relaxed), /*is_final=*/true);
      std::fprintf(stderr, "\n");
    }
  }

 private:
  // ── Timing & layout ──────────────────────────────────────────────────
  static constexpr int64_t kRenderIntervalUs = 100'000;  // 100 ms (~10 fps)
  static constexpr uint32_t kBarWidth = 30;
  static constexpr uint32_t kPrefixWidth = 24;
  static constexpr uint32_t kSpinnerFrameCount = 10;

  // ── ANSI escape codes ────────────────────────────────────────────────
  static constexpr const char *kReset = "\033[0m";
  static constexpr const char *kBold = "\033[1m";
  static constexpr const char *kDim = "\033[2m";
  static constexpr const char *kGreen = "\033[32m";
  static constexpr const char *kYellow = "\033[33m";
  static constexpr const char *kBoldCyan = "\033[1;36m";
  static constexpr const char *kBoldGreen = "\033[1;32m";
  static constexpr const char *kClearLine = "\r\033[K";

  // ── Unicode symbols (UTF-8 encoded) ──────────────────────────────────
  static constexpr const char *kFullBlock = "\xe2\x96\x88";   // █ U+2588
  static constexpr const char *kDarkShade = "\xe2\x96\x93";   // ▓ U+2593
  static constexpr const char *kLightShade = "\xe2\x96\x91";  // ░ U+2591
  static constexpr const char *kCheckMark = "\xe2\x9c\x93";   // ✓ U+2713

  // ── Braille dot spinner frames ───────────────────────────────────────
  static constexpr const char *kSpinnerFrames[kSpinnerFrameCount] = {
      "\xe2\xa0\x8b",
      "\xe2\xa0\x99",
      "\xe2\xa0\xb9",
      "\xe2\xa0\xb8",
      "\xe2\xa0\xbc",
      "\xe2\xa0\xb4",
      "\xe2\xa0\xa6",
      "\xe2\xa0\xa7",
      "\xe2\xa0\x87",
      "\xe2\xa0\x8f",
  };

  /// Pad or truncate prefix to fixed width for alignment.
  auto format_prefix() -> std::string {
    if (prefix_.size() <= kPrefixWidth) {
      std::string result = prefix_;
      result.append(kPrefixWidth - prefix_.size(), ' ');
      return result;
    }
    return prefix_.substr(0, kPrefixWidth - 3) + "...";
  }

  /// Build the colored bar string with gradient edge.
  auto build_bar(uint32_t filled) -> std::string {
    std::string bar;
    bar.reserve(kBarWidth * 10);

    // Filled portion (green)
    if (filled > 0) {
      bar += kGreen;
      for (uint32_t i = 0; i < filled && i < kBarWidth; ++i) {
        bar += kFullBlock;
      }
    }

    // Gradient edge (yellow)
    if (filled < kBarWidth) {
      bar += kYellow;
      bar += kDarkShade;
      bar += kReset;
    }

    // Empty portion (dim)
    if (filled + 1 < kBarWidth) {
      bar += kDim;
      for (uint32_t i = filled + 1; i < kBarWidth; ++i) {
        bar += kLightShade;
      }
      bar += kReset;
    }

    return bar;
  }

  /// Render one frame of the progress bar.
  /// @param is_final  true = show green checkmark; false = show spinning indicator
  void render_line(uint64_t current, bool is_final) {
    double frac = (total_ > 0) ? static_cast<double>(current) / static_cast<double>(total_) : 0.0;
    frac = std::min(frac, 1.0);
    auto filled = static_cast<uint32_t>(frac * kBarWidth);
    double elapsed_s = std::chrono::duration<double>(Clock::now() - start_time_).count();

    std::string line;
    line.reserve(256);
    line += kClearLine;

    // Leading icon: spinner or checkmark
    if (is_final) {
      line += kBoldGreen;
      line += kCheckMark;
    } else {
      line += kBoldCyan;
      line += kSpinnerFrames[spinner_frame_++ % kSpinnerFrameCount];
    }
    line += kReset;
    line += ' ';

    // Prefix (bold)
    line += kBold;
    line += format_prefix();
    line += kReset;
    line += "  ";

    // Colored bar
    line += build_bar(filled);
    line += "  ";

    // Percentage (bold)
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f%%", frac * 100.0);
    line += kBold;
    line += buf;
    line += kReset;
    line += ' ';

    // Count and elapsed time
    std::snprintf(buf,
                  sizeof(buf),
                  "(%lu/%lu) %.1fs",
                  static_cast<unsigned long>(current),
                  static_cast<unsigned long>(total_),
                  elapsed_s);
    line += buf;

    // ETA (dim, only during progress)
    if (!is_final && current > 0 && current < total_) {
      double eta_s =
          elapsed_s * static_cast<double>(total_ - current) / static_cast<double>(current);
      std::snprintf(buf, sizeof(buf), " ETA %.1fs", eta_s);
      line += kDim;
      line += buf;
      line += kReset;
    }

    std::fwrite(line.data(), 1, line.size(), stderr);
    std::fflush(stderr);
  }

  // ── Member state ─────────────────────────────────────────────────────
  std::string prefix_;
  uint64_t total_;
  std::atomic<uint64_t> current_;
  Clock::time_point start_time_;
  std::atomic<int64_t> last_render_us_;
  std::atomic<uint64_t> last_fallback_pct_;
  std::atomic<bool> finished_;
  std::atomic<bool> rendering_;  // CAS spinlock for render exclusion
  uint32_t spinner_frame_{0};    // only modified under rendering_ lock
  bool is_tty_;
};

}  // namespace alaya
