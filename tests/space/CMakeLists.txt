# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  quant_test
  GTEST
  SRCS quant/sq4_test.cpp
)
alaya_cc_target(
  quant_sq8_test
  GTEST
  SRCS quant/sq8_test.cpp
)
alaya_cc_target(
  raw_space_test
  GTEST
  SRCS raw_space_test.cpp
)
alaya_cc_target(
  sq4_space_test
  GTEST
  SRCS sq4_space_test.cpp
)
alaya_cc_target(
  sq8_space_test
  GTEST
  SRCS sq8_space_test.cpp
)
alaya_cc_target(
  rabitq_space_test
  GTEST
  SRCS rabitq_space_test.cpp
)

alaya_add_test(
  NAME space_test_quant
  TARGET quant_test
  LABELS space
)
alaya_add_test(
  NAME space_test_quant_sq8
  TARGET quant_sq8_test
  LABELS space
)
alaya_add_test(
  NAME space_test_raw_space
  TARGET raw_space_test
  LABELS space
)
alaya_add_test(
  NAME space_test_sq4_space
  TARGET sq4_space_test
  LABELS space
)
alaya_add_test(
  NAME space_test_sq8_space
  TARGET sq8_space_test
  LABELS space
)
alaya_add_test(
  NAME space_test_rabitq_space
  TARGET rabitq_space_test
  LABELS space
)

alaya_cc_target(
  rabitq_rotator_test
  GTEST
  SRCS rabitq_utils/rotator_test.cpp
)
alaya_cc_target(
  rabitq_lut_test
  GTEST
  SRCS rabitq_utils/lut_test.cpp
)
alaya_add_test(
  NAME space_test_rabitq_rotator
  TARGET rabitq_rotator_test
  LABELS space
)
alaya_add_test(
  NAME space_test_rabitq_lut
  TARGET rabitq_lut_test
  LABELS space
)
