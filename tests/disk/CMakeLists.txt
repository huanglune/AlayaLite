# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# Disk collection tests: flat + Vamana engines unconditionally, LASER engine when ALAYA_ENABLE_LASER.

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
  mmap_file_test
  GTEST
  SRCS mmap_file_test.cpp
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
  disk_collection_test
  GTEST
  SRCS disk_collection_test.cpp
)
alaya_cc_target(
  test_disk_collection_lock
  GTEST
  SRCS test_disk_collection_lock.cpp
)
alaya_cc_target(
  segment_factory_test
  GTEST
  SRCS segment_factory_test.cpp
)
alaya_cc_target(
  disk_collection_factory_dispatch_test
  GTEST
  SRCS disk_collection_factory_dispatch_test.cpp
)
alaya_cc_target(
  test_vamana_reader
  GTEST
  SRCS test_vamana_reader.cpp
)
alaya_cc_target(
  test_vamana_greedy_search
  GTEST
  SRCS test_vamana_greedy_search.cpp
)
alaya_cc_target(
  test_vamana_segment_builder
  GTEST
  SRCS test_vamana_segment_builder.cpp
)
alaya_cc_target(
  test_vamana_segment_searcher
  GTEST
  SRCS test_vamana_segment_searcher.cpp
)
alaya_cc_target(
  test_segment_factory_vamana
  GTEST
  SRCS test_segment_factory_vamana.cpp
)
alaya_cc_target(
  test_disk_collection_vamana
  GTEST
  SRCS test_disk_collection_vamana.cpp
)
alaya_cc_target(
  test_vamana_adapter_overhead
  GTEST
  SRCS test_vamana_adapter_overhead.cpp
)

alaya_add_test(NAME disk_test_types TARGET disk_types_test)
alaya_add_test(NAME disk_test_segment_manifest TARGET segment_manifest_test)
alaya_add_test(NAME disk_test_mmap_file TARGET mmap_file_test)
alaya_add_test(NAME disk_test_flat_builder TARGET disk_flat_builder_test)
alaya_add_test(NAME disk_test_flat_searcher TARGET disk_flat_searcher_test)
alaya_add_test(NAME disk_test_collection TARGET disk_collection_test)
alaya_add_test(NAME test_disk_collection_lock TARGET test_disk_collection_lock)
alaya_add_test(NAME disk_test_segment_factory TARGET segment_factory_test)
alaya_add_test(NAME disk_test_collection_factory_dispatch TARGET disk_collection_factory_dispatch_test)
alaya_add_test(NAME test_vamana_reader TARGET test_vamana_reader)
alaya_add_test(NAME test_vamana_greedy_search TARGET test_vamana_greedy_search)
alaya_add_test(NAME test_vamana_segment_builder TARGET test_vamana_segment_builder)
alaya_add_test(NAME test_vamana_segment_searcher TARGET test_vamana_segment_searcher)
alaya_add_test(NAME test_segment_factory_vamana TARGET test_segment_factory_vamana)
alaya_add_test(NAME test_disk_collection_vamana TARGET test_disk_collection_vamana)
alaya_add_test(NAME test_vamana_adapter_overhead TARGET test_vamana_adapter_overhead)

if(ALAYA_ENABLE_LASER)
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
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca.bin
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids
      ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_medoids_indices
  )
  set(ALAYA_LASER_SEGMENT_FIXTURE_ENV ALAYALITE_LASER_BUILD_DIR=${CMAKE_BINARY_DIR})
  if(ALAYA_ASAN_RUNTIME)
    list(APPEND ALAYA_LASER_SEGMENT_FIXTURE_ENV LD_PRELOAD=${ALAYA_ASAN_RUNTIME} ASAN_OPTIONS=detect_leaks=0)
  endif()

  set(ALAYA_LASER_SEGMENT_FIXTURE_AVAILABLE OFF)
  if(BUILD_PYTHON)
    # The fixture generator imports the freshly built _alayalitepy from the build tree, so it depends on the module
    # TARGET: CMake orders it after the module link and re-runs it when the module changes. Never invoke a nested `cmake
    # --build` from inside a build rule here — two build systems compiling the same objects concurrently corrupt
    # PCH/object files (observed as cc1plus ICEs) and .ninja_deps.
    add_custom_command(
      OUTPUT ${ALAYA_LASER_SEGMENT_FIXTURE_OUTPUTS}
      BYPRODUCTS ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_pca_base.fbin
                 ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}/${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}_vamana_graph.index
                 ${CMAKE_CURRENT_BINARY_DIR}/fixtures/python_stage
      COMMAND ${CMAKE_COMMAND} -E make_directory ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
      COMMAND
        ${CMAKE_COMMAND} -E env ${ALAYA_LASER_SEGMENT_FIXTURE_ENV} ${Python_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py --output-dir ${ALAYA_LASER_SEGMENT_FIXTURE_DIR}
        --prefix ${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX} --count 2048 --dim 128 --R 64 --seed 42 --optional-sidecars
      DEPENDS ${CMAKE_SOURCE_DIR}/tests/disk/fixtures/build_laser_fixture.py _alayalitepy
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
      test_disk_collection_laser.cpp
      test_laser_adapter_overhead.cpp
      test_laser_searcher_thread_safety.cpp
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
    if(_laser_segment_test_target STREQUAL "test_laser_adapter_overhead")
      alaya_add_test(
        NAME ${_laser_segment_test_target}
        TARGET ${_laser_segment_test_target}
        LABELS laser performance
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

  # The hybrid Flat/Vamana/Laser batch_search test only needs the fixture, the alaya_laser link, and
  # ALAYA_ENABLE_LASER=1 for its Laser-engine branch; the Flat/Vamana branches run everywhere.
  alaya_cc_target(
    test_disk_collection_batch_search
    GTEST LASER
    SRCS test_disk_collection_batch_search.cpp
  )
  if(ALAYA_LASER_SEGMENT_FIXTURE_AVAILABLE)
    add_dependencies(test_disk_collection_batch_search laser_segment_fixture)
    target_compile_definitions(
      test_disk_collection_batch_search PRIVATE ALAYA_LASER_FIXTURE_DIR="${ALAYA_LASER_SEGMENT_FIXTURE_DIR}"
                                                ALAYA_LASER_FIXTURE_PREFIX="${ALAYA_LASER_SEGMENT_FIXTURE_PREFIX}"
    )
  endif()
else()
  alaya_cc_target(
    test_disk_collection_batch_search
    GTEST
    SRCS test_disk_collection_batch_search.cpp
  )

  set(ALAYA_LASER_NEGATIVE_TEST_CANDIDATES test_laser_segment_importer.cpp test_laser_segment_searcher.cpp
                                           test_segment_factory_laser.cpp test_disk_collection_laser.cpp
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

alaya_add_test(NAME test_disk_collection_batch_search TARGET test_disk_collection_batch_search)
