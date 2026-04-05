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

#include <atomic>
#include <cstdio>
#include <string>
#include <string_view>

#include "utils/log.hpp"  // for is_quiet()

namespace alaya::console {

/// @brief Check if stderr is a TTY (cached once per process).
inline auto is_tty() -> bool {
  static const bool kTty = isatty(fileno(stderr)) != 0;
  return kTty;
}

// ── ANSI escape codes (only used when TTY) ─────────────────────────────
namespace detail {
inline constexpr const char *kReset = "\033[0m";
inline constexpr const char *kBold = "\033[1m";
inline constexpr const char *kDim = "\033[2m";
inline constexpr const char *kBoldCyan = "\033[1;36m";
inline constexpr const char *kBoldGreen = "\033[1;32m";
inline constexpr const char *kBoldRed = "\033[1;31m";

inline constexpr uint32_t kLineWidth = 60;

inline auto progress_completion_flag() -> std::atomic<bool> & {
  static std::atomic<bool> flag{false};
  return flag;
}

/// @brief Visible column width of a UTF-8 string (codepoint count).
/// Skips continuation bytes (10xxxxxx) so multibyte chars count as 1.
inline auto visible_width(std::string_view s) -> int {
  int width = 0;
  for (auto c : s) {
    if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) {
      ++width;
    }
  }
  return width;
}

/// @brief Format a uv-style completion line (no trailing newline).
///
/// TTY:     "  \033[1;32m✓\033[0m {label}{pad}\033[2m{time}\033[0m"
/// non-TTY: "  * {label}{pad}{time}"
inline auto format_completion_line(std::string_view label, double elapsed_seconds) -> std::string {
  char time_buf[16];
  std::snprintf(time_buf, sizeof(time_buf), "%.1fs", elapsed_seconds);
  std::string time_str(time_buf);

  // visible columns: 2(indent) + 2("✓ " or "* ") + label_vis + pad + time
  auto label_vis = visible_width(label);
  auto label_vis_max = static_cast<int>(kLineWidth) - 4 - static_cast<int>(time_str.size());
  std::string display_label(label);
  if (label_vis > label_vis_max) {
    // Truncate by visible width — trim bytes until we fit
    int vis = 0;
    size_t cut = 0;
    for (; cut < label.size(); ++cut) {
      if ((static_cast<unsigned char>(label[cut]) & 0xC0) != 0x80) {
        if (vis >= label_vis_max - 3) {
          break;
        }
        ++vis;
      }
    }
    display_label = std::string(label.substr(0, cut)) + "...";
    label_vis = vis + 3;
  }

  int pad = label_vis_max - label_vis;
  if (pad < 1) {
    pad = 1;
  }

  std::string line;
  line.reserve(128);
  if (is_tty()) {
    line += "  ";
    line += kBoldGreen;
    line += "\xe2\x9c\x93";  // ✓ U+2713
    line += kReset;
    line += ' ';
    line += display_label;
    line.append(static_cast<size_t>(pad), ' ');
    line += kDim;
    line += time_str;
    line += kReset;
  } else {
    line += "  * ";
    line += display_label;
    line.append(static_cast<size_t>(pad), ' ');
    line += time_str;
  }
  return line;
}
}  // namespace detail

/// @brief Record that a progress bar already emitted the completion state.
inline void notify_progress_complete() {
  detail::progress_completion_flag().store(true, std::memory_order_relaxed);
}

/// @brief Show a temporary status line for a new phase.
/// On TTY: prints a braille spinner "⠏ description..." that will be overwritten
/// by phase_done() or the next ProgressBar.
inline void phase_start(std::string_view /*name*/, std::string_view description) {
  detail::progress_completion_flag().store(false, std::memory_order_relaxed);
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "  %s\xe2\xa0\x8f%s %.*s...",
                 detail::kBoldCyan,
                 detail::kReset,
                 static_cast<int>(description.size()),
                 description.data());
    std::fflush(stderr);
  }
}

/// @brief Print a uv-style phase completion line to stderr.
/// Skipped if a ProgressBar already printed its own completion line.
inline void phase_done(std::string_view name, double elapsed_seconds) {
  if (detail::progress_completion_flag().exchange(false, std::memory_order_relaxed)) {
    return;
  }
  if (is_quiet()) {
    return;
  }
  std::string output;
  if (is_tty()) {
    output += "\r\033[K";  // clear phase_start status line
  }
  output += detail::format_completion_line(name, elapsed_seconds);
  output += '\n';
  std::fwrite(output.data(), 1, output.size(), stderr);
}

/// @brief Print a uv-style summary line to stderr.
inline void summary(std::string_view label, std::string_view detail_text, double elapsed_seconds) {
  detail::progress_completion_flag().store(false, std::memory_order_relaxed);
  if (is_quiet()) {
    return;
  }
  std::string combined;
  combined.reserve(label.size() + 3 + detail_text.size());
  combined += label;
  if (is_tty()) {
    combined += " \xe2\x80\x94 ";  // " — " U+2014
  } else {
    combined += " -- ";
  }
  combined += detail_text;

  auto line = detail::format_completion_line(combined, elapsed_seconds);
  line += '\n';
  std::fwrite(line.data(), 1, line.size(), stderr);
}

/// @brief Print an error banner to stderr.
inline void error(std::string_view message) {
  detail::progress_completion_flag().store(false, std::memory_order_relaxed);
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "  %s\xe2\x9c\x97 %.*s%s\n",
                 detail::kBoldRed,
                 static_cast<int>(message.size()),
                 message.data(),
                 detail::kReset);
  } else {
    std::fprintf(stderr, "  ! %.*s\n", static_cast<int>(message.size()), message.data());
  }
}

}  // namespace alaya::console
