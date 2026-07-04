# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  coro_test
  GTEST
  SRCS coro_test.cpp
)
alaya_cc_target(
  search_test
  GTEST
  SRCS search_test.cpp
)
alaya_cc_target(
  update_test
  GTEST
  SRCS update_test.cpp
)
alaya_cc_target(
  mutex_test
  GTEST
  SRCS mutex_test.cpp
)
alaya_cc_target(
  scheduler_test
  GTEST
  SRCS scheduler_test.cpp
)
alaya_cc_target(
  task_queue_test
  GTEST
  SRCS task_queue_test.cpp
)
alaya_cc_target(
  worker_test
  GTEST
  SRCS worker_test.cpp
)

alaya_add_test(NAME executor_test_coro TARGET coro_test)
# search_test bundles several suites; register them separately so ctest can schedule and report them independently. They
# read data files relative to the build dir, hence the explicit working directory.
alaya_add_test(
  NAME executor_test_search_graph
  TARGET search_test
  FILTER SearchTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME executor_test_search_hybrid
  TARGET search_test
  FILTER HybridSearchTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME executor_test_search_rabitq_graph_job
  TARGET search_test
  FILTER RaBitQGraphSearchJobTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(
  NAME executor_test_search_rabitq_hybrid
  TARGET search_test
  FILTER RaBitQHybridSearchTest.*
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
alaya_add_test(NAME executor_test_update TARGET update_test)
alaya_add_test(NAME executor_test_mutex TARGET mutex_test)
alaya_add_test(NAME executor_test_scheduler TARGET scheduler_test)
alaya_add_test(NAME executor_test_task_queue TARGET task_queue_test)
alaya_add_test(NAME executor_test_worker TARGET worker_test)
