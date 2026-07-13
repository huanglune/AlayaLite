# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(segmented_collection_header_closure SRCS segmented_collection_header_closure.cpp)
alaya_cc_target(legacy_memory_adapter_header_closure SRCS legacy_memory_adapter_header_closure.cpp)
alaya_cc_target(legacy_disk_adapter_header_closure SRCS legacy_disk_adapter_header_closure.cpp)
alaya_cc_target(manifest_v2_header_closure SRCS manifest_v2_header_closure.cpp)

alaya_cc_target(
  manifest_v2_test
  SRCS manifest_v2_test.cpp
  GTEST
)
alaya_add_test(
  NAME manifest_v2_test
  TARGET manifest_v2_test
  LABELS unit collection persistence
  TIMEOUT 60
)

alaya_cc_target(
  segmented_collection_test
  SRCS segmented_collection_test.cpp
  GTEST
)
alaya_add_test(
  NAME segmented_collection_test
  TARGET segmented_collection_test
  LABELS unit collection
  TIMEOUT 60
)

alaya_cc_target(
  legacy_adapter_test
  SRCS legacy_adapter_test.cpp
  GTEST
)
alaya_add_test(
  NAME legacy_adapter_test
  TARGET legacy_adapter_test
  LABELS unit collection
  TIMEOUT 60
)

alaya_cc_target(
  segmented_collection_stress_test
  SRCS segmented_collection_stress_test.cpp
  GTEST
)
alaya_add_test(
  NAME segmented_collection_stress_test
  TARGET segmented_collection_stress_test
  LABELS collection stress tsan
  TIMEOUT 120
)

alaya_cc_target(
  heterogeneous_segment_integration_test
  SRCS heterogeneous_segment_integration_test.cpp
  GTEST
)
alaya_add_test(
  NAME heterogeneous_segment_integration_test
  TARGET heterogeneous_segment_integration_test
  LABELS integration collection disk segment
  TIMEOUT 180
)
