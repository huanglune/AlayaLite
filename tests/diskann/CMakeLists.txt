# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# DiskANN disk index tests.
#
# Layout / PQ / node-cache / trace tests are header-only over std + SIMD and need no I/O backend. Beam search / index /
# e2e use the LASER AlignedFileReader (libaio on Linux) and therefore need the LASER consumer surface (alaya_laser, its
# compile options, and ALAYA_ENABLE_LASER).

alaya_cc_target(
  test_diskann_layout
  GTEST
  SRCS test_diskann_layout.cpp
)
alaya_cc_target(
  test_diskann_pq
  GTEST
  SRCS test_diskann_pq.cpp
)
alaya_cc_target(
  test_diskann_node_cache
  GTEST
  SRCS test_diskann_node_cache.cpp
)
alaya_cc_target(
  test_diskann_tombstone_slot
  GTEST
  SRCS test_diskann_tombstone_slot.cpp
)
alaya_cc_target(
  test_diskann_update_trace
  GTEST
  SRCS test_diskann_update_trace.cpp
)

alaya_add_test(
  NAME test_diskann_layout
  TARGET test_diskann_layout
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_pq
  TARGET test_diskann_pq
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_node_cache
  TARGET test_diskann_node_cache
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_tombstone_slot
  TARGET test_diskann_tombstone_slot
  LABELS diskann
)
alaya_add_test(
  NAME test_diskann_update_trace
  TARGET test_diskann_update_trace
  LABELS diskann
)

if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    diskann_segment_header_closure
    LASER
    SRCS diskann_segment_header_closure.cpp
  )
  alaya_cc_target(
    diskann_segment_test
    GTEST LASER
    SRCS diskann_segment_test.cpp
  )
  alaya_cc_target(
    diskann_segment_mutable_test
    GTEST LASER
    SRCS diskann_segment_mutable_test.cpp
  )
  alaya_cc_target(
    diskann_segment_async_test
    GTEST LASER
    SRCS diskann_segment_async_test.cpp
  )
  alaya_cc_target(
    test_diskann_beam_search
    GTEST LASER
    SRCS test_diskann_beam_search.cpp
  )
  alaya_cc_target(
    test_diskann_index
    GTEST LASER
    SRCS test_diskann_index.cpp
  )
  alaya_cc_target(
    test_diskann_e2e
    GTEST LASER
    SRCS test_diskann_e2e.cpp
  )
  alaya_cc_target(
    test_diskann_disk_page_io
    GTEST LASER
    SRCS test_diskann_disk_page_io.cpp
  )
  alaya_cc_target(
    test_diskann_tombstone_search
    GTEST LASER
    SRCS test_diskann_tombstone_search.cpp
  )
  alaya_cc_target(
    test_diskann_update_e2e
    GTEST LASER
    SRCS test_diskann_update_e2e.cpp
  )
  alaya_cc_target(
    test_diskann_build_from_graph
    GTEST LASER
    SRCS test_diskann_build_from_graph.cpp
  )

  alaya_add_test(
    NAME test_diskann_beam_search
    TARGET test_diskann_beam_search
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_segment
    TARGET diskann_segment_test
    LABELS diskann segment
  )
  alaya_add_test(
    NAME test_diskann_segment_mutable
    TARGET diskann_segment_mutable_test
    LABELS diskann segment
    TIMEOUT 180
  )
  alaya_add_test(
    NAME test_diskann_segment_async
    TARGET diskann_segment_async_test
    LABELS diskann segment
    TIMEOUT 180
  )
  alaya_add_test(
    NAME test_diskann_index
    TARGET test_diskann_index
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_disk_page_io
    TARGET test_diskann_disk_page_io
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_tombstone_search
    TARGET test_diskann_tombstone_search
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_update_e2e
    TARGET test_diskann_update_e2e
    LABELS diskann
  )
  alaya_add_test(
    NAME test_diskann_build_from_graph
    TARGET test_diskann_build_from_graph
    LABELS diskann seal
    TIMEOUT 600
  )
  # The end-to-end build+search test is heavier; give it a generous timeout.
  alaya_add_test(
    NAME test_diskann_e2e
    TARGET test_diskann_e2e
    LABELS diskann e2e
    TIMEOUT 600
  )

endif()
