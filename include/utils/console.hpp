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

#include <cstdio>
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
inline constexpr const char *kGreen = "\033[32m";
inline constexpr const char *kRed = "\033[31m";
inline constexpr const char *kBoldGreen = "\033[1;32m";
inline constexpr const char *kBoldRed = "\033[1;31m";
}  // namespace detail

/// @brief Print a phase-start banner to stderr.
/// TTY:     "\033[1m▸ Phase 1: KMeans partitioning\033[0m\n"
/// non-TTY: "> Phase 1: KMeans partitioning\n"
inline void phase_start(std::string_view name, std::string_view description) {
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "%s\xe2\x96\xb8 %.*s: %.*s%s\n",
                 detail::kBold,
                 static_cast<int>(name.size()),
                 name.data(),
                 static_cast<int>(description.size()),
                 description.data(),
                 detail::kReset);
  } else {
    std::fprintf(stderr,
                 "> %.*s: %.*s\n",
                 static_cast<int>(name.size()),
                 name.data(),
                 static_cast<int>(description.size()),
                 description.data());
  }
}

/// @brief Print a phase-done banner to stderr.
/// TTY:     "\033[1;32m✓ Phase 1 complete (4.3s)\033[0m\n"
/// non-TTY: "* Phase 1 complete (4.3s)\n"
inline void phase_done(std::string_view name, double elapsed_seconds) {
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "%s\xe2\x9c\x93 %.*s complete (%.1fs)%s\n",
                 detail::kBoldGreen,
                 static_cast<int>(name.size()),
                 name.data(),
                 elapsed_seconds,
                 detail::kReset);
  } else {
    std::fprintf(stderr,
                 "* %.*s complete (%.1fs)\n",
                 static_cast<int>(name.size()),
                 name.data(),
                 elapsed_seconds);
  }
}

/// @brief Print a summary banner to stderr.
/// TTY:     "\033[1;32m✓ Index built — 20000 vectors in 45.6s\033[0m\n"
/// non-TTY: "* Index built -- 20000 vectors in 45.6s\n"
inline void summary(std::string_view label, std::string_view detail_text, double elapsed_seconds) {
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "%s\xe2\x9c\x93 %.*s \xe2\x80\x94 %.*s in %.1fs%s\n",
                 detail::kBoldGreen,
                 static_cast<int>(label.size()),
                 label.data(),
                 static_cast<int>(detail_text.size()),
                 detail_text.data(),
                 elapsed_seconds,
                 detail::kReset);
  } else {
    std::fprintf(stderr,
                 "* %.*s -- %.*s in %.1fs\n",
                 static_cast<int>(label.size()),
                 label.data(),
                 static_cast<int>(detail_text.size()),
                 detail_text.data(),
                 elapsed_seconds);
  }
}

/// @brief Print an error banner to stderr.
/// TTY:     "\033[1;31m✗ Build failed: out of memory\033[0m\n"
/// non-TTY: "! Build failed: out of memory\n"
inline void error(std::string_view message) {
  if (is_quiet()) {
    return;
  }
  if (is_tty()) {
    std::fprintf(stderr,
                 "%s\xe2\x9c\x97 %.*s%s\n",
                 detail::kBoldRed,
                 static_cast<int>(message.size()),
                 message.data(),
                 detail::kReset);
  } else {
    std::fprintf(stderr, "! %.*s\n", static_cast<int>(message.size()), message.data());
  }
}

}  // namespace alaya::console
