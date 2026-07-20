# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaLaser.cmake - the LASER disk-index module: backend wiring and the alaya_laser INTERFACE target.
#
# Backend selection (libaio / thread pool; the Windows IOCP backend was removed) is validated and pinned in
# AlayaOptions.cmake; this module resolves the backend's system dependencies and defines the consumer surface.

include_guard(GLOBAL)

if(NOT ALAYA_ENABLE_LASER)
  return()
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
  if(APPLE)
    message(FATAL_ERROR "Laser module requires OpenMP. On macOS install Homebrew libomp "
                        "(brew install libomp) or configure with -DALAYA_ENABLE_LASER=OFF."
    )
  else()
    find_package(OpenMP REQUIRED)
  endif()
endif()

if(ALAYA_LASER_USE_IOCP)
  message(FATAL_ERROR "The Windows IOCP LASER backend was removed; LASER is unavailable on Windows. "
                      "Configure with -DALAYA_ENABLE_LASER=OFF instead."
  )
elseif(ALAYA_LASER_USE_THREADPOOL)
  set(_alaya_laser_backend_libs)
  set(_alaya_laser_backend_definition ALAYA_LASER_USE_THREADPOOL=1)
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_alaya_laser_backend_message "thread pool (macOS)")
  else()
    set(_alaya_laser_backend_message "thread pool (portable fallback)")
  endif()
else()
  find_library(AIO_LIBRARY aio)
  find_path(AIO_INCLUDE_DIR libaio.h)
  if(NOT AIO_LIBRARY OR NOT AIO_INCLUDE_DIR)
    message(
      FATAL_ERROR
        "Laser module requires libaio for the default Linux backend. Install it via:\n"
        "  sudo apt-get install libaio-dev   (Debian / Ubuntu)\n"
        "  sudo dnf install libaio-devel     (Fedora / RHEL)\n"
        "or configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module, or "
        "-DALAYA_LASER_USE_THREADPOOL=ON to use the portable backend on Linux."
    )
  endif()
  if(NOT TARGET AIO::aio)
    add_library(AIO::aio UNKNOWN IMPORTED)
    set_target_properties(
      AIO::aio PROPERTIES IMPORTED_LOCATION "${AIO_LIBRARY}" INTERFACE_INCLUDE_DIRECTORIES "${AIO_INCLUDE_DIR}"
    )
  endif()
  set(_alaya_laser_backend_libs AIO::aio)
  set(_alaya_laser_backend_definition ALAYA_LASER_USE_LIBAIO=1)
  set(_alaya_laser_backend_message "libaio (Linux x86_64)")
endif()

message(STATUS "LASER I/O backend: ${_alaya_laser_backend_message}")

# INTERFACE target that exposes the ported Laser header tree plus its runtime dependency surface. Consumers (pybind
# module, alignment test, examples) link against this to pick up include paths and transitive libs.
add_library(alaya_laser INTERFACE)
target_include_directories(
  alaya_laser INTERFACE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  alaya_laser
  INTERFACE ${_alaya_laser_backend_libs}
            Eigen3::Eigen
            OpenMP::OpenMP_CXX
            concurrentqueue::concurrentqueue
            spdlog::spdlog_header_only
)
target_compile_definitions(alaya_laser INTERFACE ${_alaya_laser_backend_definition})
target_compile_features(alaya_laser INTERFACE cxx_std_20)

# -mavx2 -mfma is the LASER baseline for Eigen and non-handwritten SIMD code; handwritten LASER kernels use
# function-level target attributes for runtime AVX-512 dispatch.
if(MSVC)
  # MSVC: /arch:AVX2 enables AVX2 + FMA at the codegen level. There is no MSVC analogue of -ftree-vectorize (it
  # auto-vectorizes under /O2 by default). EIGEN_DONT_PARALLELIZE is passed via /D.
  set(_ALAYA_LASER_CONSUMER_OPTIONS
      /arch:AVX2 /DEIGEN_DONT_PARALLELIZE
      CACHE INTERNAL "Compile options a Laser-consuming target must apply PRIVATEly"
  )
else()
  # LASER keeps AVX2+FMA as its target baseline even when the core build is validating its generic x86 baseline.
  set(_alaya_laser_isa_options ${ALAYA_SIMD_COMPILE_OPTIONS})
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64"
     AND NOT ALAYA_NATIVE_ARCH
     AND NOT ALAYA_X86_AVX2_BASELINE
  )
    set(_alaya_laser_isa_options -mavx2 -mfma)
  endif()
  set(_ALAYA_LASER_CONSUMER_OPTIONS
      ${_alaya_laser_isa_options} -ftree-vectorize -DEIGEN_DONT_PARALLELIZE
      CACHE INTERNAL "Compile options a Laser-consuming target must apply PRIVATEly"
  )
endif()

# Compile the mutable-segment/updater bridge once. Public Collection headers depend only on the narrow updater API,
# while updater implementation and WAL/replay bodies stay behind this target.
#
# Linux-only: the updater io path uses O_DIRECT and sync_file_range, and mutable LASER has never shipped elsewhere.
# Sealed-LASER platforms (macOS wheels) build the read path only and had no TU instantiating the updater before the
# split either, so gating the bridge keeps the pre-split platform map unchanged.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(alaya_laser_runtime STATIC ${CMAKE_SOURCE_DIR}/src/index/graph/laser/qg/qg_updater_runtime.cpp)
  target_include_directories(
    alaya_laser_runtime PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  )
  target_link_libraries(
    alaya_laser_runtime
    PUBLIC ${_alaya_laser_backend_libs}
           Eigen3::Eigen
           OpenMP::OpenMP_CXX
           concurrentqueue::concurrentqueue
           spdlog::spdlog_header_only
  )
  target_compile_definitions(alaya_laser_runtime PUBLIC ${_alaya_laser_backend_definition})
  target_compile_features(alaya_laser_runtime PUBLIC cxx_std_20)
  target_compile_options(alaya_laser_runtime PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  if(ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # This target is created before Codecov.cmake is included, so apply the same instrumentation that
    # add_coverage_to_target() gives test executables. Otherwise updater calls moved behind the bridge disappear from
    # coverage_c++.info.
    target_compile_options(alaya_laser_runtime PRIVATE --coverage -O0 -g)
    target_link_libraries(alaya_laser_runtime PRIVATE --coverage)
  endif()
  target_link_libraries(alaya_laser INTERFACE alaya_laser_runtime)
endif()
