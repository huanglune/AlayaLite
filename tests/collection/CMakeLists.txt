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
  collection_rotate_test
  SRCS collection_rotate_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_rotate_test
  TARGET collection_rotate_test
  LABELS unit collection canonical crash
  TIMEOUT 60
)

alaya_cc_target(
  collection_hnsw_seal_test
  SRCS collection_hnsw_seal_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_hnsw_seal_test
  TARGET collection_hnsw_seal_test
  LABELS integration collection memory
  TIMEOUT 180
)

alaya_cc_target(
  collection_qg_seal_test
  SRCS collection_qg_seal_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_qg_seal_test
  TARGET collection_qg_seal_test
  LABELS integration collection memory
  TIMEOUT 300
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

# Collection-level end-to-end tests for the segment admission contract (docs/design/segment-admission-contract.md): a
# real LASER segment executing a bitmap traversal filter, and the auto-policy fallback when a qg memory segment in the
# same collection still rejects the filter kind.
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    segmented_collection_laser_filter_test
    GTEST LASER
    SRCS segmented_collection_laser_filter_test.cpp
  )
  alaya_add_test(
    NAME segmented_collection_laser_filter_test
    TARGET segmented_collection_laser_filter_test
    LABELS integration collection laser segment
    TIMEOUT 120
  )

  # End-to-end coverage for LASER as a Collection target algorithm (U2-c W1):
  # create/upsert/seal/search/filter-search/close/reopen/rotate_to_successor
  # through the public Collection facade, plus the metric=inner_product ->
  # flat fallback.
  alaya_cc_target(
    collection_laser_target_test
    GTEST LASER
    SRCS collection_laser_target_test.cpp
  )
  alaya_add_test(
    NAME collection_laser_target_test
    TARGET collection_laser_target_test
    LABELS integration collection laser
    TIMEOUT 180
  )
endif()
