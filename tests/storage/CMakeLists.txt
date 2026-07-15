# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

alaya_cc_target(
  storage_test
  GTEST
  SRCS sequential_storage_test.cpp
)
alaya_cc_target(
  static_storage_test
  GTEST
  SRCS static_storage_test.cpp
)

alaya_cc_target(
  page_reader_contract_test
  GTEST
  SRCS page_reader_contract_test.cpp
)
if(TARGET AIO::aio)
  target_link_libraries(page_reader_contract_test PRIVATE AIO::aio)
  target_compile_definitions(page_reader_contract_test PRIVATE ALAYA_LASER_USE_LIBAIO=1)
endif()

alaya_cc_target(
  mmap_file_test
  GTEST
  SRCS mmap_file_test.cpp
)

alaya_add_test(
  NAME storage_test_mmap_file
  TARGET mmap_file_test
  LABELS storage
)
alaya_add_test(
  NAME storage_test_sequential_storage
  TARGET storage_test
  LABELS storage
)
alaya_add_test(
  NAME storage_test_static_storage
  TARGET static_storage_test
  LABELS storage
)

alaya_add_test(
  NAME storage_test_page_reader_contract
  TARGET page_reader_contract_test
  LABELS storage
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  alaya_cc_target(
    uring_reactor_test
    GTEST
    SRCS uring_reactor_test.cpp
  )
  alaya_add_test(
    NAME storage_test_uring_reactor
    TARGET uring_reactor_test
    LABELS storage
  )
endif()
