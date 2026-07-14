# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(segmented_collection_header_closure SRCS segmented_collection_header_closure.cpp)
alaya_cc_target(collection_header_closure SRCS collection_header_closure.cpp)

alaya_cc_target(
  collection_facade_test
  SRCS collection_facade_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_facade_test
  TARGET collection_facade_test
  LABELS unit collection canonical
  TIMEOUT 60
)
alaya_cc_target(
  collection_facade_importer_test
  SRCS collection_facade_importer_test.cpp
  GTEST
  DEFS ALAYA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
alaya_add_test(
  NAME collection_facade_importer_test
  TARGET collection_facade_importer_test
  LABELS integration collection canonical migration
  TIMEOUT 180
)
alaya_cc_target(
  collection_facade_stress_test
  SRCS collection_facade_stress_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_facade_stress_test
  TARGET collection_facade_stress_test
  LABELS collection canonical stress tsan
  TIMEOUT 120
)
alaya_cc_target(logical_wal_header_closure SRCS logical_wal_header_closure.cpp)
alaya_cc_target(mutation_wal_codec_header_closure SRCS mutation_wal_codec_header_closure.cpp)
alaya_cc_target(collection_checkpoint_header_closure SRCS collection_checkpoint_header_closure.cpp)
alaya_cc_target(filter_seal_gc_header_closure SRCS filter_seal_gc_header_closure.cpp)
alaya_cc_target(collection_flat_target_header_closure SRCS collection_flat_target_header_closure.cpp)
alaya_cc_target(legacy_importer_header_closure SRCS legacy_importer_header_closure.cpp)

alaya_cc_target(
  legacy_importer_test
  SRCS legacy_importer_test.cpp
  GTEST
  DEFS ALAYA_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
alaya_add_test(
  NAME legacy_importer_test
  TARGET legacy_importer_test
  LABELS integration collection persistence migration
  TIMEOUT 180
)
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
  logical_wal_test
  SRCS logical_wal_test.cpp
  GTEST
)
alaya_add_test(
  NAME logical_wal_test
  TARGET logical_wal_test
  LABELS unit collection persistence wal
  TIMEOUT 60
)

alaya_cc_target(
  wal_coordinator_test
  SRCS wal_coordinator_test.cpp
  GTEST
)
alaya_add_test(
  NAME wal_coordinator_test
  TARGET wal_coordinator_test
  LABELS unit
         collection
         persistence
         wal
         crash
  TIMEOUT 120
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
