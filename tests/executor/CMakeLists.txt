# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  search_test
  GTEST
  SRCS search_test.cpp
)
alaya_cc_target(
  mutex_test
  GTEST
  SRCS mutex_test.cpp
)

alaya_add_test(
  NAME executor_test_search_graph
  TARGET search_test
  FILTER SearchTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME executor_test_search_unit
  TARGET search_test
  FILTER GraphSearchJobUnitTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME executor_test_search_rabitq_graph_job
  TARGET search_test
  FILTER RaBitQGraphSearchJobTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(NAME executor_test_mutex TARGET mutex_test)
