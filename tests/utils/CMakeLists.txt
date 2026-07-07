# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  query_utils_test
  GTEST
  SRCS query_utils_test.cpp
)
alaya_cc_target(
  log_test
  GTEST
  SRCS log_test.cpp
)
alaya_cc_target(
  evaluate_test
  GTEST
  SRCS evaluate_test.cpp
)
alaya_cc_target(
  metric_type_test
  GTEST
  SRCS metric_type_test.cpp
)
alaya_cc_target(
  data_utils_test
  GTEST
  SRCS data_utils_test.cpp
)
alaya_cc_target(
  dataset_utils_test
  GTEST
  SRCS dataset_utils_test.cpp
)
alaya_cc_target(
  index_encoding_test
  GTEST
  SRCS index_encoding_test.cpp
)
alaya_cc_target(
  metadata_filter_test
  GTEST
  SRCS metadata_filter_test.cpp
)
alaya_cc_target(
  rotator_utils_test
  GTEST
  SRCS rabitq_utils/rotator_test.cpp
)
alaya_cc_target(
  lut_utils_test
  GTEST
  SRCS rabitq_utils/lut_test.cpp
)
alaya_cc_target(
  math_test
  GTEST
  SRCS math_test.cpp
)

alaya_add_test(
  NAME utils_test_query_utils
  TARGET query_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_log
  TARGET log_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_evaluate
  TARGET evaluate_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_metric_type
  TARGET metric_type_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_data_utils
  TARGET data_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_dataset_utils
  TARGET dataset_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_index_encoding
  TARGET index_encoding_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_metadata_filter
  TARGET metadata_filter_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_rotator_utils
  TARGET rotator_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_lut_utils
  TARGET lut_utils_test
  LABELS utils
)
alaya_add_test(
  NAME utils_test_math
  TARGET math_test
  LABELS utils
)
