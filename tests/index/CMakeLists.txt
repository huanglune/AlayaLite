# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  fusion_graph_test
  GTEST
  SRCS fusion_graph_test.cpp
)
alaya_cc_target(
  fusion_segment_test
  GTEST
  SRCS fusion_segment_test.cpp
)
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
  nndescent_test
  GTEST
  SRCS nndescent_test.cpp
)
alaya_cc_target(
  nsg_test
  GTEST
  SRCS nsg_test.cpp
)
alaya_cc_target(
  nsg_segment_test
  GTEST
  SRCS nsg_segment_test.cpp
)
alaya_cc_target(
  qg_segment_test
  GTEST
  SRCS qg_segment_test.cpp
)
alaya_cc_target(
  vamana_mem_segment_test
  GTEST
  SRCS vamana_mem_segment_test.cpp
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
alaya_cc_target(rabitq_benchmark SRCS rabitq_benchmark.cpp)

alaya_add_test(NAME index_test_fusion_graph TARGET fusion_graph_test)
alaya_add_test(NAME index_test_fusion_segment TARGET fusion_segment_test)
alaya_add_test(
  NAME index_test_fusion_segment_tsan
  TARGET fusion_segment_test
  FILTER FusionSegmentTest.ConcurrentSearchOnlyIsReentrant
  LABELS tsan
)
alaya_add_test(NAME index_test_graph TARGET graph_test)
alaya_add_test(NAME index_test_hnsw TARGET hnsw_test)
alaya_add_test(NAME index_test_nndescent TARGET nndescent_test)
alaya_add_test(NAME index_test_nsg TARGET nsg_test)
alaya_add_test(NAME index_test_nsg_segment TARGET nsg_segment_test)
alaya_add_test(
  NAME index_test_nsg_segment_tsan
  TARGET nsg_segment_test
  FILTER NsgSegmentTest.ConcurrentSearchOnlyIsReentrant
  LABELS tsan
)
alaya_add_test(NAME index_test_qg_segment TARGET qg_segment_test)
alaya_add_test(
  NAME index_test_qg_segment_tsan
  TARGET qg_segment_test
  FILTER QgSegmentTest.ConcurrentSearchOnlyIsReentrant
  LABELS tsan
)
alaya_add_test(NAME index_test_vamana_mem_segment TARGET vamana_mem_segment_test)
alaya_add_test(
  NAME index_test_vamana_mem_segment_tsan
  TARGET vamana_mem_segment_test
  FILTER VamanaMemSegmentTest.ConcurrentSearchOnlyIsReentrant
  LABELS tsan
)
alaya_add_test(NAME index_test_rabitq TARGET rabitq_test)
alaya_add_test(
  NAME index_test_rabitq_performance
  TARGET rabitq_performance_test
  LABELS performance
)
