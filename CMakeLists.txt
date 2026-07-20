# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaLite - header-only vector database core with a pybind11 extension module.
#
# cmake-format: off
# This file only orchestrates; each concern lives in cmake/:
#   AlayaConan.cmake        Conan dependency provider registration (must precede project())
#   AlayaOptions.cmake      every user-facing switch + option validation
#   AlayaPreflight.cmake    early, actionable environment checks (compiler floor, Python headers)
#   AlayaToolchain.cmake    C++ standard, ccache, fast linkers, sanitizer detection
#   AlayaFlags.cmake        compile flags as the linkable alaya_build_flags INTERFACE target
#   AlayaDependencies.cmake find_package for all third-party libs (THIRD_PARTY_LIBS)
#   AlayaLaser.cmake        LASER disk-index backend wiring (alaya_laser INTERFACE target)
#   AlayaTesting.cmake      alaya_cc_target / alaya_add_test helpers used by tests/*
# cmake-format: on
#
# Quick start (dev): cmake --preset release && cmake --build --preset release && ctest --preset release The same tree
# also configures with a plain `cmake -B build/Release -DCMAKE_BUILD_TYPE=Release` (CI does), and is driven by
# scikit-build-core for wheel builds (see pyproject.toml). The 3.24 floor comes from the Conan dependency provider
# (CMAKE_PROJECT_TOP_LEVEL_INCLUDES).

cmake_minimum_required(VERSION 3.24...3.31)

include(cmake/AlayaConan.cmake)

project(AlayaLite LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()

include(cmake/AlayaOptions.cmake)
include(cmake/AlayaPreflight.cmake)
alaya_preflight()

include(cmake/AlayaToolchain.cmake)
include(cmake/AlayaFlags.cmake)
include(cmake/AlayaDependencies.cmake)
include(cmake/AlayaLaser.cmake)
include(cmake/Codecov.cmake)
include(cmake/PrintSummary.cmake)

# ---------------------------------------------------------------------------------------------------------------------
# The core consumer surface plus the non-template Collection runtime.
# ---------------------------------------------------------------------------------------------------------------------
add_library(AlayaLite INTERFACE)
target_include_directories(
  AlayaLite INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(AlayaLite INTERFACE ${THIRD_PARTY_LIBS})
target_compile_features(AlayaLite INTERFACE cxx_std_20)

add_library(
  alaya_segmented_collection_runtime STATIC
  src/index/collection/segmented_collection_lifecycle.cpp src/index/collection/segmented_collection_mutation.cpp
  src/index/collection/segmented_collection_search.cpp src/index/collection/segmented_collection_utility.cpp
)
target_include_directories(
  alaya_segmented_collection_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(alaya_segmented_collection_runtime PUBLIC ${THIRD_PARTY_LIBS})
target_compile_features(alaya_segmented_collection_runtime PUBLIC cxx_std_20)
alaya_apply_build_flags(alaya_segmented_collection_runtime)
add_coverage_to_target(alaya_segmented_collection_runtime)

# Collection's non-template members, split across translation units by compile cost and built into a portable and a
# LASER owner. Each collection_runtime_NN.cpp header names that unit's concern.
set(_alaya_collection_runtime_sources
    src/index/collection/collection_runtime_00.cpp
    src/index/collection/collection_runtime_01.cpp
    src/index/collection/collection_runtime_02.cpp
    src/index/collection/collection_runtime_03.cpp
    src/index/collection/collection_runtime_04.cpp
    src/index/collection/collection_runtime_05.cpp
    src/index/collection/collection_runtime_06.cpp
    src/index/collection/collection_runtime_07.cpp
    src/index/collection/collection_runtime_08.cpp
    src/index/collection/collection_runtime_09.cpp
    src/index/collection/collection_runtime_10.cpp
)

add_library(alaya_collection_runtime STATIC ${_alaya_collection_runtime_sources})
target_include_directories(
  alaya_collection_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(alaya_collection_runtime PUBLIC alaya_segmented_collection_runtime ${THIRD_PARTY_LIBS})
target_compile_features(alaya_collection_runtime PUBLIC cxx_std_20)
alaya_apply_build_flags(alaya_collection_runtime)
add_coverage_to_target(alaya_collection_runtime)

if(ALAYA_ENABLE_LASER)
  add_library(alaya_collection_laser_runtime STATIC ${_alaya_collection_runtime_sources})
  target_include_directories(
    alaya_collection_laser_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(
    alaya_collection_laser_runtime PUBLIC alaya_segmented_collection_runtime alaya_laser ${THIRD_PARTY_LIBS}
  )
  target_compile_definitions(alaya_collection_laser_runtime PRIVATE ALAYA_ENABLE_LASER=1)
  target_compile_features(alaya_collection_laser_runtime PUBLIC cxx_std_20)
  target_compile_options(alaya_collection_laser_runtime PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  alaya_apply_build_flags(alaya_collection_laser_runtime)
  add_coverage_to_target(alaya_collection_laser_runtime)
  set(_alaya_collection_runtime_link
      "$<IF:$<BOOL:$<TARGET_PROPERTY:ALAYA_LASER_CONSUMER>>,alaya_collection_laser_runtime,alaya_collection_runtime>"
  )
else()
  set(_alaya_collection_runtime_link alaya_collection_runtime)
endif()
target_link_libraries(AlayaLite INTERFACE ${_alaya_collection_runtime_link})
unset(_alaya_collection_runtime_link)
unset(_alaya_collection_runtime_sources)

if(BUILD_TESTING)
  include(CTest)
  find_package(GTest REQUIRED)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  include(cmake/AlayaTesting.cmake)
  add_subdirectory(tests)
  add_subdirectory(benchmarks)
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()

print_build_summary()
