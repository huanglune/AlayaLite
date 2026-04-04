cmake_minimum_required(VERSION 3.15...3.31)

# Project definition
set(PROJECT_NAME AlayaLite)
project(${PROJECT_NAME} LANGUAGES CXX)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE
  )
endif()

# Include utility functions
include(cmake/PrintSummary.cmake)
include(cmake/Codecov.cmake)

option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(ENABLE_CLANG_TIDY "Enable clang-tidy static analysis during build" OFF)
option(BUILD_TESTING "Build tests" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_PYTHON "Build Python bindings" OFF)

# ============================================================================
# 1. C++ STANDARD AND COMPILER CONFIGURATION
# ============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/EHsc /utf-8 /O2)
else()
  add_compile_options(-Wall -Ofast)
  if(BUILD_TESTING OR BUILD_BENCHMARKS)
    add_compile_options(-g -fno-omit-frame-pointer)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-flto=auto)
    add_link_options(-flto=auto)
  endif()
endif()

# ============================================================================
# 1. Dependency Management (Auto Conan Setup)
# ============================================================================
include(cmake/ConanSetup.cmake)
if(APPLE)
  file(GLOB OPENMP_BREW_PATHS "/usr/local/opt/libomp" # Intel Mac
       "/opt/homebrew/opt/libomp" # Apple Silicon
  )
  list(APPEND CMAKE_PREFIX_PATH ${OPENMP_BREW_PATHS})
endif()
find_package(concurrentqueue REQUIRED)
find_package(spdlog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(unordered_dense REQUIRED)
find_package(tomlplusplus REQUIRED)
if(BUILD_BENCHMARKS)
  find_package(benchmark REQUIRED)
endif()
if(BUILD_PYTHON)
  find_package(pybind11 REQUIRED)
endif()

# ============================================================================
# 1. clang-tidy (after find_package so OpenMP include dirs are available)
# ============================================================================
set(CLANG_TIDY_PREFERRED_VERSION
    "18"
    CACHE STRING "Preferred clang-tidy version (keep in sync with CI)"
)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES "clang-tidy-${CLANG_TIDY_PREFERRED_VERSION}" clang-tidy)

  if(CLANG_TIDY_EXE)
    if(NOT CLANG_TIDY_EXE MATCHES "clang-tidy-${CLANG_TIDY_PREFERRED_VERSION}")
      message(WARNING "Using ${CLANG_TIDY_EXE} instead of clang-tidy-${CLANG_TIDY_PREFERRED_VERSION}. "
                      "Results may differ from CI. "
                      "Install: sudo apt-get install clang-tidy-${CLANG_TIDY_PREFERRED_VERSION}"
      )
    endif()

    set(CLANG_TIDY_COMMAND ${CLANG_TIDY_EXE})

    # GCC + clang-tidy: GCC's omp.h may not be on clang-tidy's search path. Create a shim that redirects to GCC's actual
    # omp.h location. Note: -print-file-name returns the input unchanged when not found, so the EXISTS check handles
    # that case.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=include/omp.h
        OUTPUT_VARIABLE GCC_OPENMP_HEADER
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(EXISTS "${GCC_OPENMP_HEADER}")
        set(CLANG_TIDY_COMPAT_DIR "${CMAKE_BINARY_DIR}/clang-tidy-compat")
        file(MAKE_DIRECTORY "${CLANG_TIDY_COMPAT_DIR}")
        file(WRITE "${CLANG_TIDY_COMPAT_DIR}/omp.h" "#pragma once\n#include <${GCC_OPENMP_HEADER}>\n")
        list(APPEND CLANG_TIDY_COMMAND "--extra-arg=-isystem${CLANG_TIDY_COMPAT_DIR}")
        message(STATUS "clang-tidy using OpenMP compatibility header: ${CLANG_TIDY_COMPAT_DIR}/omp.h")
      endif()
    endif()

    foreach(ct_openmp_include_dir IN LISTS OpenMP_CXX_INCLUDE_DIRS)
      if(IS_DIRECTORY "${ct_openmp_include_dir}")
        list(APPEND CLANG_TIDY_COMMAND "--extra-arg=-isystem${ct_openmp_include_dir}")
      endif()
    endforeach()

    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_COMMAND})
    message(STATUS "clang-tidy enabled: ${CLANG_TIDY_EXE}")
  else()
    message(WARNING "clang-tidy not found, static analysis skipped. "
                    "Install: sudo apt-get install clang-tidy-${CLANG_TIDY_PREFERRED_VERSION}"
    )
  endif()
endif()

set(THIRD_PARTY_LIBS
    spdlog::spdlog_header_only
    concurrentqueue::concurrentqueue
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    unordered_dense::unordered_dense
)

if(UNIX AND NOT APPLE)
  find_package(libcoro REQUIRED)
  find_package(liburing REQUIRED)
  list(APPEND THIRD_PARTY_LIBS libcoro::libcoro liburing::liburing)
endif()

# ============================================================================
# PROJECT TARGETS
# ============================================================================
include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(
  ${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME} INTERFACE ${THIRD_PARTY_LIBS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# ============================================================================
# SUBDIRECTORIES
# ============================================================================
if(BUILD_TESTING)
  include(CTest)
  find_package(GTest)
  set(GTEST_LIBS GTest::gtest GTest::gtest_main)
  add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(BUILD_PYTHON)
  add_subdirectory(python)
endif()
print_build_summary()
