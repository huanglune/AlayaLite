// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Standalone A/B timing harness for space/quant/rabitq/dispatch.hpp: "dispatch
// on" (the process-detected best tier, AVX-512 on this host) vs. "forced
// generic" (the portable scalar tier called directly, bypassing dispatch),
// for each of the six converted kernels. Build only, not registered with
// ctest — run manually and paste the numbers into the delivery report.
//
// The repository has no prior benchmark number for this refactor ("17-24%"
// mentioned in planning docs has no traceable source); this harness produces
// fresh, reproducible numbers instead.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "index/graph/detail/timer.hpp"
#include "simd/cpu_features.hpp"
#include "simd/fastscan.hpp"
#include "space/quant/rabitq/dispatch.hpp"

namespace {

using alaya::Timer;

constexpr int kAccumulateIters = 4'000'000;
constexpr int kEstimateIters = 4'000'000;
constexpr int kFusedIters = 2'000'000;
constexpr int kFlipSignIters = 2'000'000;
constexpr int kKacsWalkIters = 4'000'000;
constexpr int kScalarQuantIters = 2'000'000;

void print_row(const std::string &name, double generic_ms, double dispatched_ms, long iters) {
  const double speedup = generic_ms / dispatched_ms;
  const double generic_ns_per_op = (generic_ms * 1.0e6) / static_cast<double>(iters);
  const double dispatched_ns_per_op = (dispatched_ms * 1.0e6) / static_cast<double>(iters);
  std::cout << std::left << std::setw(34) << name << std::right << std::fixed
            << std::setprecision(1) << std::setw(10) << generic_ns_per_op << " ns/op (generic) "
            << std::setw(10) << dispatched_ns_per_op << " ns/op (dispatched) " << std::setw(7)
            << std::setprecision(2) << speedup << "x\n";
}

// ---------------------------------------------------------------------------
// 1. accumulate
// ---------------------------------------------------------------------------
void bench_accumulate(std::mt19937 &rng, size_t dim) {
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::vector<uint8_t> codes(dim << 2);
  std::vector<uint8_t> lut(dim << 2);
  std::generate(codes.begin(), codes.end(), [&] {
    return byte_dist(rng);
  });
  std::generate(lut.begin(), lut.end(), [&] {
    return byte_dist(rng);
  });
  std::array<uint16_t, 32> result{};

  volatile uint16_t sink = 0;

  Timer timer;
  for (int i = 0; i < kAccumulateIters; ++i) {
    ::alaya::simd::fastscan::accumulate_generic(dim, codes.data(), lut.data(), result.data());
    sink = static_cast<uint16_t>(sink ^ result[i & 31]);
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_accumulate_func();
  for (int i = 0; i < kAccumulateIters; ++i) {
    fn(dim, codes.data(), lut.data(), result.data());
    sink = static_cast<uint16_t>(sink ^ result[i & 31]);
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("accumulate (dim=" + std::to_string(dim) + ")",
            generic_ms,
            dispatched_ms,
            kAccumulateIters);
}

// ---------------------------------------------------------------------------
// 2. estimate_distances
// ---------------------------------------------------------------------------
void bench_estimate_distances(std::mt19937 &rng) {
  std::uniform_real_distribution<float> coef_dist(-5.0F, 5.0F);
  std::uniform_int_distribution<int> seg_dist(0, 4000);
  alignas(64) std::array<uint16_t, alaya::rabitq_simd::kBatchSize> nth_segments{};
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> f_add{};
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> f_rescale{};
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> result{};
  for (size_t i = 0; i < alaya::rabitq_simd::kBatchSize; ++i) {
    nth_segments[i] = static_cast<uint16_t>(seg_dist(rng));
    f_add[i] = coef_dist(rng);
    f_rescale[i] = coef_dist(rng);
  }

  volatile float sink = 0;

  Timer timer;
  for (int i = 0; i < kEstimateIters; ++i) {
    alaya::rabitq_simd::detail::estimate_distances_generic(nth_segments.data(),
                                                           f_add.data(),
                                                           f_rescale.data(),
                                                           1.0F,
                                                           0.5F,
                                                           0.25F,
                                                           result.data());
    sink = sink + result[i & 31];
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_estimate_distances_func();
  for (int i = 0; i < kEstimateIters; ++i) {
    fn(nth_segments.data(), f_add.data(), f_rescale.data(), 1.0F, 0.5F, 0.25F, result.data());
    sink = sink + result[i & 31];
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("estimate_distances", generic_ms, dispatched_ms, kEstimateIters);
}

// ---------------------------------------------------------------------------
// 3. accumulate_and_estimate_distances (the fused hot kernel memqg calls per
//    candidate node, rabitq_space.hpp QueryComputer::batch_est_dist)
// ---------------------------------------------------------------------------
void bench_fused(std::mt19937 &rng, size_t dim) {
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_real_distribution<float> coef_dist(-3.0F, 3.0F);
  std::vector<uint8_t> codes(dim << 2);
  std::vector<uint8_t> lut(dim << 2);
  std::generate(codes.begin(), codes.end(), [&] {
    return byte_dist(rng);
  });
  std::generate(lut.begin(), lut.end(), [&] {
    return byte_dist(rng);
  });
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> f_add{};
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> f_rescale{};
  for (size_t i = 0; i < alaya::rabitq_simd::kBatchSize; ++i) {
    f_add[i] = coef_dist(rng);
    f_rescale[i] = coef_dist(rng);
  }
  alignas(64) std::array<float, alaya::rabitq_simd::kBatchSize> result{};
  alignas(64) std::array<uint16_t, alaya::rabitq_simd::kBatchSize> nth_segments{};

  volatile float sink = 0;

  // "forced generic": hand-composed from the two known-generic primitives,
  // bypassing dispatch entirely (mirrors what the pre-refactor #else branch
  // did on a non-AVX-512 build, and what
  // rabitq_dispatch_test.cpp's differential fuzz test uses as its oracle).
  Timer timer;
  for (int i = 0; i < kFusedIters; ++i) {
    ::alaya::simd::fastscan::accumulate_generic(dim, codes.data(), lut.data(), nth_segments.data());
    alaya::rabitq_simd::detail::estimate_distances_generic(nth_segments.data(),
                                                           f_add.data(),
                                                           f_rescale.data(),
                                                           1.0F,
                                                           0.5F,
                                                           0.25F,
                                                           result.data());
    sink = sink + result[i & 31];
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_accumulate_and_estimate_distances_func();
  for (int i = 0; i < kFusedIters; ++i) {
    fn(codes.data(),
       lut.data(),
       f_add.data(),
       f_rescale.data(),
       1.0F,
       0.5F,
       0.25F,
       result.data(),
       dim);
    sink = sink + result[i & 31];
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("accumulate_and_estimate_distances (dim=" + std::to_string(dim) + ")",
            generic_ms,
            dispatched_ms,
            kFusedIters);
}

// ---------------------------------------------------------------------------
// 4. flip_sign
// ---------------------------------------------------------------------------
void bench_flip_sign(std::mt19937 &rng, size_t dim) {
  std::uniform_real_distribution<float> val_dist(-10.0F, 10.0F);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::vector<uint8_t> flip(dim / 8);
  std::generate(flip.begin(), flip.end(), [&] {
    return byte_dist(rng);
  });
  std::vector<float> data(dim);
  std::generate(data.begin(), data.end(), [&] {
    return val_dist(rng);
  });

  volatile float sink = 0;

  Timer timer;
  for (int i = 0; i < kFlipSignIters; ++i) {
    alaya::rabitq_simd::detail::flip_sign_generic(flip.data(), data.data(), dim);
    sink = sink + data[static_cast<size_t>(i) % dim];
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_flip_sign_func();
  for (int i = 0; i < kFlipSignIters; ++i) {
    fn(flip.data(), data.data(), dim);
    sink = sink + data[static_cast<size_t>(i) % dim];
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("flip_sign (dim=" + std::to_string(dim) + ")",
            generic_ms,
            dispatched_ms,
            kFlipSignIters);
}

// ---------------------------------------------------------------------------
// 5. kacs_walk
// ---------------------------------------------------------------------------
void bench_kacs_walk(std::mt19937 &rng, size_t len) {
  std::uniform_real_distribution<float> val_dist(-10.0F, 10.0F);
  std::vector<float> data(len);
  std::generate(data.begin(), data.end(), [&] {
    return val_dist(rng);
  });

  volatile float sink = 0;

  Timer timer;
  for (int i = 0; i < kKacsWalkIters; ++i) {
    alaya::rabitq_simd::detail::kacs_walk_generic(data.data(), len);
    sink = sink + data[static_cast<size_t>(i) % len];
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_kacs_walk_func();
  for (int i = 0; i < kKacsWalkIters; ++i) {
    fn(data.data(), len);
    sink = sink + data[static_cast<size_t>(i) % len];
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("kacs_walk (len=" + std::to_string(len) + ")",
            generic_ms,
            dispatched_ms,
            kKacsWalkIters);
}

// ---------------------------------------------------------------------------
// 6. scalar_quantize_optimized
// ---------------------------------------------------------------------------
void bench_scalar_quantize_optimized(std::mt19937 &rng, size_t table_length) {
  std::uniform_real_distribution<float> val_dist(-7.0F, 7.0F);
  std::vector<float> vec0(table_length);
  std::generate(vec0.begin(), vec0.end(), [&] {
    return val_dist(rng);
  });
  const float lo = *std::min_element(vec0.begin(), vec0.end());
  const float hi = *std::max_element(vec0.begin(), vec0.end());
  const float delta = (hi - lo) / 255.0F;
  std::vector<uint8_t> result(table_length);

  volatile uint8_t sink = 0;

  Timer timer;
  for (int i = 0; i < kScalarQuantIters; ++i) {
    alaya::rabitq_simd::detail::scalar_quantize_optimized_generic(result.data(),
                                                                  vec0.data(),
                                                                  table_length,
                                                                  lo,
                                                                  delta);
    sink = static_cast<uint8_t>(sink ^ result[static_cast<size_t>(i) % table_length]);
  }
  const double generic_ms = timer.elapsed_ms();

  timer.reset();
  const auto fn = alaya::rabitq_simd::get_scalar_quantize_optimized_func();
  for (int i = 0; i < kScalarQuantIters; ++i) {
    fn(result.data(), vec0.data(), table_length, lo, delta);
    sink = static_cast<uint8_t>(sink ^ result[static_cast<size_t>(i) % table_length]);
  }
  const double dispatched_ms = timer.elapsed_ms();

  (void)sink;
  print_row("scalar_quantize_optimized (len=" + std::to_string(table_length) + ")",
            generic_ms,
            dispatched_ms,
            kScalarQuantIters);
}

}  // namespace

auto main() -> int {
  const auto &features = alaya::simd::get_cpu_features();
  std::cout << "host cpu features: avx512f=" << features.avx512f_
            << " avx512bw=" << features.avx512bw_ << " avx2=" << features.avx2_
            << " fma=" << features.fma_ << '\n';
  std::cout << "rabitq_simd dispatch resolved to: " << alaya::rabitq_simd::get_rabitq_simd_name()
            << "\n\n";

  std::mt19937 rng(0xB5A5E11EU);

  bench_accumulate(rng, 128);
  bench_estimate_distances(rng);
  bench_fused(rng, 128);
  bench_flip_sign(rng, 256);
  bench_kacs_walk(rng, 256);
  bench_scalar_quantize_optimized(rng, 512);

  return 0;
}
