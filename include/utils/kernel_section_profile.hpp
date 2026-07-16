// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Probe-only kernel section profiler (U-line kernel-gap audit). Compiled out
// unless ALAYA_KERNEL_SECTION_PROFILE is defined; intended for single-thread
// profiling runs only (the accumulator is a plain global). Sections align the
// two graph kernels 1:1:
//   prep  — per-query setup (rotate/quantize/LUT build, medoid seeding)
//   exact — fp32 exact distance of the popped node
//   scan  — fastscan estimate of the popped node's neighbors
//   pool  — neighbor loop: pool inserts + visited checks (+ prefetch)
// Everything else (pop/visited of the popped node, result pool) is "other" =
// total wall time minus the sections.

#pragma once

#ifdef ALAYA_KERNEL_SECTION_PROFILE

#include <x86intrin.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>

namespace alaya::ksp {

struct Stats {
  std::uint64_t prep = 0;
  std::uint64_t exact = 0;
  std::uint64_t scan = 0;
  std::uint64_t pool = 0;
  std::uint64_t pops = 0;
  std::uint64_t queries = 0;
};

inline Stats g_stats;  // NOLINT

inline void reset() { g_stats = Stats{}; }

inline void report(const char *tag) {
  const Stats &s = g_stats;
  if (s.queries == 0) {
    return;
  }
  const auto q = static_cast<double>(s.queries);
  std::printf("ksp,%s,queries,%" PRIu64 ",pops_per_q,%.1f,cyc_per_q,prep,%.0f,exact,%.0f,scan,%.0f,pool,%.0f\n",
              tag,
              s.queries,
              static_cast<double>(s.pops) / q,
              static_cast<double>(s.prep) / q,
              static_cast<double>(s.exact) / q,
              static_cast<double>(s.scan) / q,
              static_cast<double>(s.pool) / q);
  std::fflush(stdout);
  reset();
}

}  // namespace alaya::ksp

#define ALAYA_KSP_BEGIN(sec) const std::uint64_t alaya_ksp_b_##sec = __rdtsc()
#define ALAYA_KSP_END(sec) ::alaya::ksp::g_stats.sec += __rdtsc() - alaya_ksp_b_##sec
#define ALAYA_KSP_COUNT(field) ++::alaya::ksp::g_stats.field
#define ALAYA_KSP_REPORT(tag) ::alaya::ksp::report(tag)
// The memqg build path shares the instrumented kernel (find_candidates does
// ~N*ef_build pops); benches must reset right before the search sweep.
#define ALAYA_KSP_RESET() ::alaya::ksp::reset()

#else

#define ALAYA_KSP_BEGIN(sec)
#define ALAYA_KSP_END(sec)
#define ALAYA_KSP_COUNT(field)
#define ALAYA_KSP_REPORT(tag)
#define ALAYA_KSP_RESET()

#endif  // ALAYA_KERNEL_SECTION_PROFILE
