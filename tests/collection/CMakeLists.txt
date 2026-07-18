# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(segmented_collection_header_closure SRCS segmented_collection_header_closure.cpp)
alaya_cc_target(collection_header_closure SRCS collection_header_closure.cpp)

# Collection tests share one PCH per compile profile. Header-closure targets stay deliberately PCH-free so they continue
# to prove that public headers compile in isolation.
alaya_cc_target(
  alaya_collection_test_pch
  SRCS ../pch_anchor.cpp
  GTEST
)
target_precompile_headers(alaya_collection_test_pch PRIVATE <gtest/gtest.h> <alaya/collection.hpp>)

alaya_cc_target(
  collection_facade_test
  SRCS collection_facade_test.cpp
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
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
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
)
alaya_add_test(
  NAME collection_rotate_test
  TARGET collection_rotate_test
  LABELS unit collection canonical crash
  TIMEOUT 60
)

alaya_cc_target(
  collection_qg_seal_test
  SRCS collection_qg_seal_test.cpp
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
)
alaya_add_test(
  NAME collection_qg_seal_test
  TARGET collection_qg_seal_test
  LABELS integration collection memory
  TIMEOUT 300
)

alaya_cc_target(
  collection_qg_ip_norm_test
  SRCS collection_qg_ip_norm_test.cpp
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
)
alaya_add_test(
  NAME collection_qg_ip_norm_test
  TARGET collection_qg_ip_norm_test
  LABELS integration collection memory
  TIMEOUT 180
)

alaya_cc_target(
  collection_qg_recall_floor_test
  SRCS collection_qg_recall_floor_test.cpp
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
)
alaya_add_test(
  NAME collection_qg_recall_floor_test
  TARGET collection_qg_recall_floor_test
  LABELS integration collection memory
  TIMEOUT 300
)

alaya_cc_target(
  collection_facade_stress_test
  SRCS collection_facade_stress_test.cpp
  GTEST PCH_REUSE_FROM alaya_collection_test_pch
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
  collection_maintenance_interleaving_test
  SRCS collection_maintenance_interleaving_test.cpp
  GTEST
)
alaya_add_test(
  NAME collection_maintenance_interleaving_test
  TARGET collection_maintenance_interleaving_test
  LABELS unit collection wal tsan
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
    alaya_collection_laser_test_pch
    SRCS ../pch_anchor.cpp
    GTEST LASER
  )
  target_precompile_headers(
    alaya_collection_laser_test_pch
    PRIVATE
    <gtest/gtest.h>
    <utils/laser_test_pch.hpp>
    <index/collection/segmented_collection.hpp>
  )

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
  # create/upsert/seal/search/filter-search/close/reopen/rotate_to_successor through the public Collection facade, plus
  # native inner_product admission (the deeper metric semantics live in collection_laser_metric_wiring_test below).
  alaya_cc_target(
    collection_laser_target_test
    GTEST LASER
    SRCS collection_laser_target_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
  )
  alaya_add_test(
    NAME collection_laser_target_test
    TARGET collection_laser_target_test
    LABELS integration collection laser
    TIMEOUT 180
  )

  # LASER recall floor lock (D-lite manifest W2): mirrors collection_qg_recall_floor_test.cpp for the LASER on-disk QG
  # target, which had no recall regression guard before this file. l2 unit/nonunit tiers plus a resident_arena residency
  # tier, each asserting engine identity against the persisted sealed-segment manifest (not
  # Collection::target_implementation_key() -- see the test file header for why).
  alaya_cc_target(
    collection_laser_recall_floor_test
    GTEST LASER
    SRCS collection_laser_recall_floor_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
  )
  alaya_add_test(
    NAME collection_laser_recall_floor_test
    TARGET collection_laser_recall_floor_test
    LABELS integration collection laser
    TIMEOUT 300
  )

  # Sealed/read-only LASER metric production wiring: same-data IP parity with memory QG, cosine normalization semantics,
  # native proofs, and reopen.
  alaya_cc_target(
    collection_laser_metric_wiring_test
    GTEST LASER
    SRCS collection_laser_metric_wiring_test.cpp
  )
  alaya_add_test(
    NAME collection_laser_metric_wiring_test
    TARGET collection_laser_metric_wiring_test
    LABELS integration collection laser
    TIMEOUT 300
  )

  if(ALAYA_ENABLE_MUTABLE_LASER)
    # Compile-time capability-surface proof for the active LASER Collection adapter (codex B-10): links LASER so the AIO
    # page-reader symbols resolve.
    alaya_cc_target(
      mutable_laser_adapter_header_closure
      LASER
      SRCS mutable_laser_adapter_header_closure.cpp
    )

    # Adapter-in-isolation coverage: explicit-txid idempotency (B-01/B-02), previous tombstoning (B-05), and the
    # post-commit failure latch (B-04).
    alaya_cc_target(
      mutable_laser_adapter_test
      GTEST LASER
      SRCS mutable_laser_adapter_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
    )
    alaya_add_test(
      NAME mutable_laser_adapter_test
      TARGET mutable_laser_adapter_test
      LABELS unit collection laser
      TIMEOUT 120
    )

    # End-to-end active LASER engine: create/write/search/remove/close/reopen through the public Collection facade (B-08
    # config persistence + B-01 reopen restore).
    alaya_cc_target(
      collection_active_laser_test
      GTEST LASER
      SRCS collection_active_laser_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
    )
    alaya_add_test(
      NAME collection_active_laser_test
      TARGET collection_active_laser_test
      LABELS integration collection laser
      TIMEOUT 180
    )

    alaya_cc_target(
      collection_active_laser_maintenance_test
      GTEST LASER
      SRCS collection_active_laser_maintenance_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
    )
    alaya_add_test(
      NAME collection_active_laser_maintenance_test
      TARGET collection_active_laser_maintenance_test
      LABELS integration
             collection
             laser
             wal
             tsan
      TIMEOUT 300 RUN_SERIAL
    )

    alaya_cc_target(
      collection_active_laser_recovery_test
      GTEST LASER
      SRCS collection_active_laser_recovery_test.cpp PCH_REUSE_FROM alaya_collection_laser_test_pch
    )
    alaya_add_test(
      NAME collection_active_laser_recovery_test
      TARGET collection_active_laser_recovery_test
      LABELS integration
             collection
             laser
             wal
             crash
      TIMEOUT 300
    )
  endif()
endif()
