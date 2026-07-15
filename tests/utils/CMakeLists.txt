# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  evaluate_test
  GTEST
  SRCS evaluate_test.cpp
)
alaya_cc_target(
  dataset_utils_test
  GTEST
  SRCS dataset_utils_test.cpp
)
alaya_cc_target(
  math_test
  GTEST
  SRCS math_test.cpp
)

alaya_add_test(
  NAME utils_test_evaluate
  TARGET evaluate_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_dataset_utils
  TARGET dataset_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_math
  TARGET math_test
  LABELS utils
)
