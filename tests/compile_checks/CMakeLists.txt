# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# Compile-only checks: verify headers compile without LASER and that public API types satisfy their documented contracts
# (concepts, traits). This directory is only added when ALAYA_ENABLE_LASER=OFF.

alaya_cc_target(no_laser_header_closure SRCS no_laser_header_closure.cpp)
alaya_cc_target(public_graph_closure SRCS public_graph_header.cpp)
alaya_cc_target(public_space_closure SRCS public_space_header.cpp)
alaya_cc_target(public_disk_closure SRCS public_disk_header.cpp)

alaya_add_test(
  NAME compile_check_no_laser
  TARGET no_laser_header_closure
  LABELS compile
)
alaya_add_test(
  NAME compile_check_graph
  TARGET public_graph_closure
  LABELS compile
)
alaya_add_test(
  NAME compile_check_space
  TARGET public_space_closure
  LABELS compile
)
alaya_add_test(
  NAME compile_check_disk
  TARGET public_disk_closure
  LABELS compile
)
