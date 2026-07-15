# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  vamana_build_dispatch_test
  GTEST
  SRCS vamana_build_dispatch_test.cpp
)
alaya_cc_target(
  vamana_greedy_search_test
  GTEST
  SRCS test_vamana_greedy_search.cpp
)
alaya_cc_target(
  vamana_reader_test
  GTEST
  SRCS test_vamana_reader.cpp
)
alaya_add_test(NAME vamana_test_build_dispatch TARGET vamana_build_dispatch_test)
alaya_add_test(NAME vamana_test_greedy_search TARGET vamana_greedy_search_test)
alaya_add_test(NAME vamana_test_reader TARGET vamana_reader_test)

# test_vamana_alignment — Gate 1 alignment harness, hand-written main with exit codes 10/11/12 for L1/L2/L3 tiers (not a
# GTest target). Requires CLI args (--alaya_index, --diskann_index, --data_path, --query_path, --gt_path), so it is not
# registered with add_test — run manually.
alaya_cc_target(test_vamana_alignment SRCS test_vamana_alignment.cpp)
