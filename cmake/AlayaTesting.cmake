# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaTesting.cmake - declarative helpers for test/bench executables.
#
# alaya_cc_target(<name> SRCS <files...> [GTEST] [LASER] [BARE] [PCH_REUSE_FROM <target>] [LIBS <targets...>] [DEFS
# <defines...>] [OPTS <options...>]) Creates the executable, links AlayaLite (headers + all third-party libs) and the
# standard build flags, applies coverage instrumentation, and optionally the GTest libraries and/or the LASER consumer
# surface (alaya_laser + ALAYA_ENABLE_LASER=1 + the SIMD/vectorization options LASER translation units require). BARE
# skips the implicit AlayaLite link for targets that must control their exact link set (the LASER backend unit tests);
# such targets still get alaya_build_flags and list everything else through LIBS. PCH_REUSE_FROM shares a precompiled
# header from a target with an identical compile profile.
#
# alaya_add_test(NAME <name> TARGET <target> [FILTER <gtest-filter>] [LABELS <labels...>] [TIMEOUT <seconds>]
# [WORKING_DIRECTORY <dir>] [RUN_SERIAL] [COVERAGE_RUN_SERIAL]) Registers one ctest entry for an alaya_cc_target
# executable. COVERAGE_RUN_SERIAL applies only when ENABLE_COVERAGE=ON, so memory-bound coverage tests can avoid
# destructive CPU contention without serializing the Release gate. A target may be registered several times with
# different FILTERs (e.g. one ctest entry per GTest suite in a binary).
#
# Target names, test names, and labels are part of the project's external contract — CI scripts build targets by name
# and select tests by label — so helpers take both names explicitly instead of deriving one from the other.

include_guard(GLOBAL)

# --- Test config generation (once per configure) ---
set(ALAYA_TEST_DATA_DIR
    "${CMAKE_SOURCE_DIR}/data"
    CACHE PATH "Root directory for test datasets"
)
set(ALAYA_BUILD_GRAPH_DIR
    ""
    CACHE PATH "Root directory for build-graph artifacts"
)
set(ALAYA_DISKANN_SEARCH_BIN
    "search_memory_index"
    CACHE FILEPATH "Path to DiskANN search_memory_index binary"
)
configure_file(
  ${CMAKE_SOURCE_DIR}/tests/include/utils/test_config.hpp.in ${CMAKE_BINARY_DIR}/generated/test_config.hpp @ONLY
)

# --- Dataset download targets (opt-in, never part of default build) ---
set(ALAYA_DOWNLOAD_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/download_dataset.sh")
set(ALAYA_DATASETS siftsmall deep1m)

add_custom_target(test-data)
foreach(ds IN LISTS ALAYA_DATASETS)
  add_custom_target(
    test-data-${ds}
    COMMAND ${ALAYA_DOWNLOAD_SCRIPT} ${ds} ${ALAYA_TEST_DATA_DIR}
    COMMENT "Downloading dataset: ${ds}"
    VERBATIM
  )
  add_dependencies(test-data test-data-${ds})
endforeach()

function(alaya_cc_target target_name)
  set(flag_keywords GTEST LASER BARE)
  set(one_value_keywords PCH_REUSE_FROM)
  set(multi_value_keywords SRCS LIBS DEFS OPTS)
  cmake_parse_arguments(
    ARG
    "${flag_keywords}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  if(NOT ARG_SRCS)
    message(FATAL_ERROR "alaya_cc_target(${target_name}): SRCS is required")
  endif()
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "alaya_cc_target(${target_name}): unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  set(sources)
  foreach(source_file IN LISTS ARG_SRCS)
    list(APPEND sources ${CMAKE_CURRENT_SOURCE_DIR}/${source_file})
  endforeach()

  add_executable(${target_name} ${sources})
  if(ARG_BARE)
    target_link_libraries(${target_name} PRIVATE alaya_build_flags ${ARG_LIBS})
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
  else()
    target_link_libraries(${target_name} PRIVATE AlayaLite alaya_build_flags ${ARG_LIBS})
  endif()
  target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/tests/include ${CMAKE_BINARY_DIR}/generated)

  if(ARG_GTEST)
    target_link_libraries(${target_name} PRIVATE GTest::gtest GTest::gtest_main)
    set_property(TARGET ${target_name} PROPERTY ALAYA_GTEST TRUE)
  endif()

  if(ARG_LASER)
    if(NOT TARGET alaya_laser)
      message(FATAL_ERROR "alaya_cc_target(${target_name}): LASER requested but ALAYA_ENABLE_LASER=OFF")
    endif()
    target_link_libraries(${target_name} PRIVATE alaya_laser)
    target_compile_definitions(${target_name} PRIVATE ALAYA_ENABLE_LASER=1)
    target_compile_options(${target_name} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  endif()

  if(ARG_DEFS)
    target_compile_definitions(${target_name} PRIVATE ${ARG_DEFS})
  endif()
  if(ARG_OPTS)
    target_compile_options(${target_name} PRIVATE ${ARG_OPTS})
  endif()

  add_coverage_to_target(${target_name})

  if(ARG_PCH_REUSE_FROM)
    if(NOT TARGET ${ARG_PCH_REUSE_FROM})
      message(
        FATAL_ERROR "alaya_cc_target(${target_name}): PCH_REUSE_FROM target '${ARG_PCH_REUSE_FROM}' does not exist"
      )
    endif()
    target_precompile_headers(${target_name} REUSE_FROM ${ARG_PCH_REUSE_FROM})
  endif()
endfunction()

function(alaya_add_test)
  set(flag_keywords RUN_SERIAL COVERAGE_RUN_SERIAL)
  set(one_value_keywords
      NAME
      TARGET
      FILTER
      TIMEOUT
      WORKING_DIRECTORY
  )
  set(multi_value_keywords LABELS)
  cmake_parse_arguments(
    ARG
    "${flag_keywords}"
    "${one_value_keywords}"
    "${multi_value_keywords}"
    ${ARGN}
  )

  if(NOT ARG_NAME OR NOT ARG_TARGET)
    message(FATAL_ERROR "alaya_add_test: NAME and TARGET are required")
  endif()
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "alaya_add_test(${ARG_NAME}): unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  set(test_args)
  get_target_property(is_gtest ${ARG_TARGET} ALAYA_GTEST)
  if(is_gtest)
    list(APPEND test_args --gtest_brief=1)
  endif()
  if(ARG_FILTER)
    list(APPEND test_args --gtest_filter=${ARG_FILTER})
  endif()
  add_test(NAME ${ARG_NAME} COMMAND $<TARGET_FILE:${ARG_TARGET}> ${test_args})

  if(ARG_LABELS)
    set_tests_properties(${ARG_NAME} PROPERTIES LABELS "${ARG_LABELS}")
  endif()
  if(ARG_TIMEOUT)
    set_tests_properties(${ARG_NAME} PROPERTIES TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_WORKING_DIRECTORY)
    set_tests_properties(${ARG_NAME} PROPERTIES WORKING_DIRECTORY ${ARG_WORKING_DIRECTORY})
  endif()
  if(ARG_RUN_SERIAL OR (ARG_COVERAGE_RUN_SERIAL AND ENABLE_COVERAGE))
    set_tests_properties(${ARG_NAME} PROPERTIES RUN_SERIAL TRUE)
  endif()
endfunction()
