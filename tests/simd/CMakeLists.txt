# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  l2_sqr_test
  GTEST
  SRCS l2_sqr_test.cpp
)
alaya_cc_target(
  ip_test
  GTEST
  SRCS ip_test.cpp
)
alaya_cc_target(
  fht_test
  GTEST
  SRCS fht_test.cpp
)
alaya_cc_target(
  cpu_features_test
  GTEST
  SRCS cpu_features_test.cpp
)

# Standalone micro-benchmarks: built, never registered with ctest — run manually.
alaya_cc_target(l2_sqr_full_benchmark SRCS l2_sqr_full_benchmark.cpp)
alaya_cc_target(l2_sqr_sq8_benchmark SRCS l2_sqr_sq8_benchmark.cpp)
alaya_cc_target(l2_sqr_sq4_benchmark SRCS l2_sqr_sq4_benchmark.cpp)
alaya_cc_target(ip_full_benchmark SRCS ip_full_benchmark.cpp)
alaya_cc_target(ip_sq8_benchmark SRCS ip_sq8_benchmark.cpp)
alaya_cc_target(ip_sq4_benchmark SRCS ip_sq4_benchmark.cpp)
alaya_cc_target(fht_benchmark SRCS fht_benchmark.cpp)

alaya_add_test(
  NAME simd_test_l2_sqr
  TARGET l2_sqr_test
  LABELS simd
)
alaya_add_test(
  NAME simd_test_ip
  TARGET ip_test
  LABELS simd
)
alaya_add_test(
  NAME simd_test_fht
  TARGET fht_test
  LABELS simd
)
alaya_add_test(
  NAME simd_test_cpu_features
  TARGET cpu_features_test
  LABELS simd
)
