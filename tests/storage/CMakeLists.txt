# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  storage_test
  GTEST
  SRCS sequential_storage_test.cpp
)
alaya_cc_target(
  static_storage_test
  GTEST
  SRCS static_storage_test.cpp
)
alaya_cc_target(
  rocksdb_storage_test
  GTEST
  SRCS rocksdb_storage_test.cpp
)

alaya_add_test(
  NAME storage_test_sequential_storage
  TARGET storage_test
  LABELS storage
)
alaya_add_test(
  NAME storage_test_static_storage
  TARGET static_storage_test
  LABELS storage
)
alaya_add_test(
  NAME storage_test_rocksdb_storage
  TARGET rocksdb_storage_test
  LABELS storage
)
