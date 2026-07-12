# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(segmented_collection_header_closure SRCS segmented_collection_header_closure.cpp)
alaya_cc_target(legacy_memory_adapter_header_closure SRCS legacy_memory_adapter_header_closure.cpp)
alaya_cc_target(legacy_disk_adapter_header_closure SRCS legacy_disk_adapter_header_closure.cpp)

alaya_cc_target(segmented_collection_test SRCS segmented_collection_test.cpp GTEST)
alaya_add_test(
  NAME segmented_collection_test
  TARGET segmented_collection_test
  LABELS unit collection
  TIMEOUT 60
)

alaya_cc_target(legacy_adapter_test SRCS legacy_adapter_test.cpp GTEST)
alaya_add_test(
  NAME legacy_adapter_test
  TARGET legacy_adapter_test
  LABELS unit collection
  TIMEOUT 60
)
