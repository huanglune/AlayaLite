# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# Disk segment tests: flat engine unconditionally, LASER engine when ALAYA_ENABLE_LASER.

alaya_cc_target(
  disk_types_test
  GTEST
  SRCS disk_types_test.cpp
)
alaya_cc_target(
  segment_manifest_test
  GTEST
  SRCS segment_manifest_test.cpp
)
alaya_cc_target(
  disk_flat_builder_test
  GTEST
  SRCS disk_flat_builder_test.cpp
)
alaya_cc_target(
  disk_flat_searcher_test
  GTEST
  SRCS disk_flat_searcher_test.cpp
)
alaya_cc_target(
  disk_flat_segment_test
  GTEST
  SRCS disk_flat_segment_test.cpp
)
alaya_cc_target(disk_flat_segment_header_closure SRCS disk_flat_segment_header_closure.cpp)
alaya_cc_target(laser_segment_header_closure SRCS laser_segment_header_closure.cpp)
alaya_cc_target(
  disk_flat_segment_stress_test
  GTEST
  SRCS disk_flat_segment_stress_test.cpp
)
alaya_add_test(
  NAME disk_test_types
  TARGET disk_types_test
  LABELS unit
)
alaya_add_test(
  NAME disk_test_segment_manifest
  TARGET segment_manifest_test
  LABELS unit
)
alaya_add_test(
  NAME disk_test_flat_builder
  TARGET disk_flat_builder_test
  LABELS unit
)
alaya_add_test(
  NAME disk_test_flat_searcher
  TARGET disk_flat_searcher_test
  LABELS unit
)
alaya_add_test(
  NAME disk_test_flat_segment
  TARGET disk_flat_segment_test
  LABELS disk segment
)
alaya_add_test(
  NAME disk_test_flat_segment_stress
  TARGET disk_flat_segment_stress_test
  LABELS disk segment stress tsan
  TIMEOUT 120
)
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    laser_fixture_builder
    LASER
    SRCS fixtures/laser_fixture_builder.cpp
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_PREFIX dsqg_seg_00000001)
  set(ALAYA_LASER_SEGMENT_FIXTURE_DIR ${CMAKE_CURRENT_BINARY_DIR}/fixtures/laser_segment)
  set(ALAYA_LASER_SEGMENT_FIXTURE_INPUT
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_input.fbin
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_INDEX
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_R64_MD128.index
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS
      ${ALAYA_LASER_SEGMENT_FIXTURE_INPUT}
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_rotator
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_cache_ids
      ${ALAYA_LASER_SEGMENT_FIXTURE_INDEX}_cache_nodes
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/.laser_fixture_provenance.json
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca.bin
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids_indices
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_ENV)
  if(ALAYA_ASAN_RUNTIME)
    list(APPEND ALAYA_LASER_SEGMENT_FIXTURE_ENV LD_PRELOAD=${ALAYA_ASAN_RUNTIME} ASAN_OPTIONS=detect_leaks=0)
  endif()

  set(ALAYA_LASER_SEGMENT_FIXTURE_AVAILABLE OFF)
  if(BUILD_PYTHON)
    add_custom_command(
      OUTPUT ${ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS}
      BYPRODUCTS ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca_base.fbin
                 ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_vamana_graph.index
      COMMAND ${CMAKE_COMMAND} -E make_directory ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
      COMMAND
        ${CMAKE_COMMAND} -E env ${ALAYA_LASER_SEGMENT_FIXTURE_ENV} ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py --output-dir ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
        --extension $<TARGET_FILE:_alayalitepy> --native-builder $<TARGET_FILE:laser_fixture_builder> --prefix
        ${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX} --count 2048 --dim 128 --R 64 --seed 42 --optional-sidecars
      DEPENDS ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py _alayalitepy laser_fixture_builder
      COMMENT "Generating deterministic LASER disk fixture"
      VERBATIM
    )
    add_custom_target(laser_segment_fixture DEPENDS ${ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS})
    set(ALAYA_LASER_SEGMENT_FIXTURE_AVAILABLE ON)
  else()
    message(STATUS "Skipping LASER disk fixture generation because BUILD_PYTHON=OFF")
  endif()

  set(ALAYA_LASER_SEGMENT_TEST_CANDIDATES
      test_laser_segment_importer.cpp
      test_laser_segment_searcher.cpp
      test_segment_factory_laser.cpp
      test_laser_searcher_thread_safety.cpp
      laser_segment_test.cpp
      laser_segment_stress_test.cpp
      rabitq_format_separation_test.cpp
      test_unified_laser_admission.cpp
  )
  set(ALAYA_LASER_SEGMENT_TEST_TARGETS)
  foreach(laser_segment_test_source IN LISTS ALAYA_LASER_SEGMENT_TEST_CANDIDATES)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${laser_segment_test_source})
      continue()
    endif()
    get_filename_component(_laser_segment_test_target ${laser_segment_test_source} NAME_WE)
    alaya_cc_target(
      ${_laser_segment_test_target}
      GTEST LASER
      SRCS ${laser_segment_test_source}
    )
    if(ALAYA_LASER_SEGMENT_FIXTURE_AVAILABLE)
      add_dependencies(${_laser_segment_test_target} laser_segment_fixture)
      target_compile_definitions(
        ${_laser_segment_test_target} PRIVATE ALAYA_LASER_FIXTURE_DIR="${ALAYA_LASER_SEGMENT_FIXTURE_DIR}"
                                              ALAYA_LASER_FIXTURE_PREFIX="${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}"
      )
    endif()
    if(_laser_segment_test_target STREQUAL "laser_segment_test")
      alaya_add_test(
        NAME ${_laser_segment_test_target}
        TARGET ${_laser_segment_test_target}
        LABELS laser segment
        TIMEOUT 180
      )
    elseif(_laser_segment_test_target STREQUAL "laser_segment_stress_test")
      alaya_add_test(
        NAME ${_laser_segment_test_target}
        TARGET ${_laser_segment_test_target}
        LABELS laser segment stress tsan
        TIMEOUT 180
      )
    else()
      alaya_add_test(
        NAME ${_laser_segment_test_target}
        TARGET ${_laser_segment_test_target}
        LABELS laser
      )
    endif()
    list(APPEND ALAYA_LASER_SEGMENT_TEST_TARGETS ${_laser_segment_test_target})
  endforeach()
  if(ALAYA_LASER_SEGMENT_TEST_TARGETS)
    add_custom_target(test_laser_segment DEPENDS ${ALAYA_LASER_SEGMENT_TEST_TARGETS})
  endif()
else()
  set(ALAYA_LASER_NEGATIVE_TEST_CANDIDATES test_laser_segment_importer.cpp test_laser_segment_searcher.cpp
                                           test_segment_factory_laser.cpp
  )
  set(ALAYA_LASER_NEGATIVE_TEST_TARGETS)
  foreach(laser_negative_test_source IN LISTS ALAYA_LASER_NEGATIVE_TEST_CANDIDATES)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${laser_negative_test_source})
      continue()
    endif()
    get_filename_component(_laser_negative_test_stem ${laser_negative_test_source} NAME_WE)
    set(_laser_negative_test_target ${_laser_negative_test_stem}_negative)
    alaya_cc_target(
      ${_laser_negative_test_target}
      GTEST
      SRCS ${laser_negative_test_source}
    )
    alaya_add_test(NAME ${_laser_negative_test_target} TARGET ${_laser_negative_test_target})
    list(APPEND ALAYA_LASER_NEGATIVE_TEST_TARGETS ${_laser_negative_test_target})
  endforeach()
  if(ALAYA_LASER_NEGATIVE_TEST_TARGETS)
    add_custom_target(test_laser_segment_negative DEPENDS ${ALAYA_LASER_NEGATIVE_TEST_TARGETS})
  endif()
endif()
