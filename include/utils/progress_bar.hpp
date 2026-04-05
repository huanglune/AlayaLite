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
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>

#include "utils/console.hpp"
#include "utils/log.hpp"
#include "utils/macros.hpp"

namespace alaya {

/// @brief Thread-safe animated progress bar with uv-style aesthetics.
///
/// Renders a progress bar to stderr matching uv's visual style:
///   - Braille dot spinner animation (~10 fps)
///   - Green filled blocks (█), dim empty blocks (░)
///   - Percentage + ETA display
///   - On completion: bar replaced with green checkmark summary line
///
/// Falls back to periodic fprintf(stderr) when stderr is not a terminal.
/// Respects ALAYA_QUIET=1: suppresses all stderr output when set.
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
        is_tty_(isatty(fileno(stderr)) != 0),
        quiet_(is_quiet()) {
    if (!quiet_ && is_tty_ && total_ > 0) {
      render_line(0, /*is_final=*/false);
    }
  }

  ALAYA_NON_COPYABLE_NON_MOVABLE(ProgressBar);

  ~ProgressBar() { finish(); }

  /// Thread-safe: increment progress by 1.
  void tick() {
    uint64_t cur = current_.fetch_add(1, std::memory_order_relaxed) + 1;

    if (quiet_) {
      return;
    }

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
      // Non-TTY fallback: print every 10%
      if (total_ > 0) {
        uint64_t pct = cur * 100 / total_;
        uint64_t last = last_fallback_pct_.load(std::memory_order_relaxed);
        if (pct / 10 > last / 10 || cur >= total_) {
          last_fallback_pct_.store(pct, std::memory_order_relaxed);
          std::fprintf(stderr, "  %s: %" PRIu64 "%%\n", prefix_.c_str(), pct);
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
    if (quiet_) {
      return;
    }
    auto current = current_.load(std::memory_order_relaxed);
    if (is_tty_) {
      render_line(current, /*is_final=*/true);
      std::fprintf(stderr, "\n");
      console::notify_progress_complete();
    } else if (total_ > 0 && current >= total_) {
      console::notify_progress_complete();
    }
  }

 private:
  // ── Timing & layout ──────────────────────────────────────────────────
  static constexpr int64_t kRenderIntervalUs = 100'000;  // 100 ms (~10 fps)
  static constexpr uint32_t kBarWidth = 20;
  static constexpr uint32_t kSpinnerFrameCount = 10;

  // ── ANSI escape codes ────────────────────────────────────────────────
  static constexpr const char *kReset = "\033[0m";
  static constexpr const char *kBold = "\033[1m";
  static constexpr const char *kDim = "\033[2m";
  static constexpr const char *kGreen = "\033[32m";
  static constexpr const char *kBoldCyan = "\033[1;36m";
  static constexpr const char *kClearLine = "\r\033[K";

  // ── Unicode symbols (UTF-8 encoded) ──────────────────────────────────
  static constexpr const char *kFullBlock = "\xe2\x96\x88";   // █ U+2588
  static constexpr const char *kLightShade = "\xe2\x96\x91";  // ░ U+2591

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

  /// Build the uv-style bar: green ████ + dim ░░░░.
  auto build_bar(uint32_t filled) -> std::string {
    std::string bar;
    bar.reserve(kBarWidth * 8);

    if (filled > 0) {
      bar += kGreen;
      for (uint32_t i = 0; i < filled && i < kBarWidth; ++i) {
        bar += kFullBlock;
      }
      bar += kReset;
    }

    if (filled < kBarWidth) {
      bar += kDim;
      for (uint32_t i = filled; i < kBarWidth; ++i) {
        bar += kLightShade;
      }
      bar += kReset;
    }

    return bar;
  }

  /// Render the in-progress bar line.
  void render_progress(double frac,
                       uint32_t filled,
                       double elapsed_s,
                       uint64_t current,
                       std::string &line) {
    // 2-space indent + spinner
    line += "  ";
    line += kBoldCyan;
    line += kSpinnerFrames[spinner_frame_++ % kSpinnerFrameCount];
    line += kReset;
    line += ' ';

    // Prefix (bold)
    line += kBold;
    line += prefix_;
    line += kReset;
    line += "  ";

    // Bar
    line += build_bar(filled);
    line += "  ";

    // Percentage
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%3.0f%%", frac * 100.0);
    line += buf;

    // ETA (dim)
    if (current > 0 && current < total_) {
      double eta_s =
          elapsed_s * static_cast<double>(total_ - current) / static_cast<double>(current);
      std::snprintf(buf, sizeof(buf), "  ETA %.1fs", eta_s);
      line += kDim;
      line += buf;
      line += kReset;
    }
  }

  /// Render the completion line (reuses console's shared formatter).
  void render_completion(double elapsed_s, std::string &line) {
    line += console::detail::format_completion_line(prefix_, elapsed_s);
  }

  /// Render one frame of the progress bar.
  void render_line(uint64_t current, bool is_final) {
    double frac = (total_ > 0) ? static_cast<double>(current) / static_cast<double>(total_) : 0.0;
    frac = std::min(frac, 1.0);
    auto filled = static_cast<uint32_t>(frac * kBarWidth);
    double elapsed_s = std::chrono::duration<double>(Clock::now() - start_time_).count();

    std::string line;
    line.reserve(256);
    line += kClearLine;

    if (is_final) {
      render_completion(elapsed_s, line);
    } else {
      render_progress(frac, filled, elapsed_s, current, line);
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
  bool quiet_;
};

}  // namespace alaya
