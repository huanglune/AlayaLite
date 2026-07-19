# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  qg_builder_test
  GTEST
  SRCS qg_builder_test.cpp
)
alaya_cc_target(
  qg_hashset_test
  GTEST
  SRCS qg_hashset_test.cpp
)
alaya_cc_target(
  coro_mutex_test
  GTEST
  SRCS mutex_test.cpp
)
alaya_add_test(
  NAME index_test_qg_builder
  TARGET qg_builder_test
  LABELS unit builder
)
alaya_add_test(
  NAME index_test_coro_mutex
  TARGET coro_mutex_test
  LABELS unit
)
alaya_add_test(
  NAME index_test_qg_hashset
  TARGET qg_hashset_test
  LABELS unit builder
)
