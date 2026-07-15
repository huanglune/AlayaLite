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
