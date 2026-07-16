# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  graph_test
  GTEST
  SRCS graph_test.cpp
)
alaya_cc_target(
  hnsw_test
  GTEST
  SRCS hnsw_test.cpp
)
alaya_cc_target(
  qg_segment_test
  GTEST
  SRCS qg_segment_test.cpp
)
alaya_cc_target(
  rabitq_test
  GTEST
  SRCS rabitq_test.cpp
)
alaya_cc_target(
  rabitq_performance_test
  GTEST
  SRCS rabitq_performance_test.cpp
)
alaya_cc_target(
  graph_search_test
  GTEST
  SRCS search_test.cpp
)
alaya_cc_target(
  coro_mutex_test
  GTEST
  SRCS mutex_test.cpp
)
alaya_cc_target(
  query_utils_test
  GTEST
  SRCS query_utils_test.cpp
)

alaya_add_test(
  NAME index_test_graph
  TARGET graph_test
  LABELS unit
)
alaya_add_test(
  NAME index_test_hnsw
  TARGET hnsw_test
  LABELS unit
)
alaya_add_test(
  NAME index_test_qg_segment
  TARGET qg_segment_test
  LABELS unit segment
)
alaya_add_test(
  NAME index_test_qg_segment_tsan
  TARGET qg_segment_test
  FILTER QgSegmentTest.ConcurrentSearchOnlyIsReentrant
  LABELS tsan
)
alaya_add_test(
  NAME index_test_rabitq
  TARGET rabitq_test
  LABELS unit
)
alaya_add_test(
  NAME index_test_rabitq_performance
  TARGET rabitq_performance_test
  LABELS performance
  TIMEOUT 1800
)
alaya_add_test(
  NAME index_test_graph_search
  TARGET graph_search_test
  FILTER SearchTest.*
  LABELS unit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME index_test_graph_search_unit
  TARGET graph_search_test
  FILTER GraphSearchJobUnitTest.*
  LABELS unit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME index_test_graph_search_rabitq
  TARGET graph_search_test
  FILTER RaBitQGraphSearchJobTest.*
  LABELS unit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME index_test_coro_mutex
  TARGET coro_mutex_test
  LABELS unit
)
alaya_add_test(
  NAME index_test_query_utils
  TARGET query_utils_test
  LABELS utils
)
