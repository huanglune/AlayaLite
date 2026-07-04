# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaOptions.cmake - every user-facing build switch in one place.
#
# Options are declared here (and only here) so `cmake -LH` and this file are the complete catalogue. Validation that
# only involves option/platform combinations also lives here; toolchain-dependent checks live in AlayaPreflight.cmake.

include_guard(GLOBAL)

option(ENABLE_COVERAGE "Enable code coverage instrumentation" OFF)
option(BUILD_PYTHON "Build Python extension module" ON)
option(BUILD_TESTING "Build tests" OFF)
option(ALAYA_NATIVE_ARCH "Compile with -march=native for host-specific builds" OFF)
option(ALAYA_ALLOW_NATIVE_PACKAGE "Allow BUILD_PYTHON with host-specific -march=native" OFF)
option(ALAYA_USE_CCACHE "Use ccache as compiler launcher when available" ON)
option(ALAYA_USE_FAST_LINKER "Use mold or lld when available" ON)
option(ALAYA_AUTO_CONAN "Run conan install automatically when the toolchain is missing" ON)

if(BUILD_PYTHON
   AND ALAYA_NATIVE_ARCH
   AND NOT ALAYA_ALLOW_NATIVE_PACKAGE
)
  message(FATAL_ERROR "ALAYA_NATIVE_ARCH=ON makes Python package builds host-specific and unsafe to distribute. "
                      "Use -DALAYA_ALLOW_NATIVE_PACKAGE=ON only for local, non-distributable builds."
  )
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Laser on-disk QG index module. Linux x86_64 defaults to libaio; macOS defaults to the portable thread-pool backend;
# Windows x64 defaults to the IOCP backend. Other platforms skip it silently by default.
# ---------------------------------------------------------------------------------------------------------------------
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
   OR CMAKE_SYSTEM_NAME STREQUAL "Darwin"
   OR (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
)
  set(ALAYA_ENABLE_LASER_DEFAULT ON)
else()
  set(ALAYA_ENABLE_LASER_DEFAULT OFF)
endif()
option(ALAYA_ENABLE_LASER "Build the Laser disk-index module" ${ALAYA_ENABLE_LASER_DEFAULT})

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT ON)
else()
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT OFF)
endif()
option(ALAYA_LASER_USE_THREADPOOL "Use the portable thread-pool LASER I/O backend"
       ${ALAYA_LASER_USE_THREADPOOL_DEFAULT}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
  set(ALAYA_LASER_USE_IOCP_DEFAULT ON)
else()
  set(ALAYA_LASER_USE_IOCP_DEFAULT OFF)
endif()
option(ALAYA_LASER_USE_IOCP "Use the Windows IOCP LASER I/O backend" ${ALAYA_LASER_USE_IOCP_DEFAULT})

if(ALAYA_LASER_USE_IOCP AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64"
                                )
)
  message(FATAL_ERROR "ALAYA_LASER_USE_IOCP=ON requires Windows AMD64/x86_64. On other platforms use "
                      "-DALAYA_LASER_USE_THREADPOOL=ON or the libaio default."
  )
endif()

if(ALAYA_ENABLE_LASER)
  # Pin the backend per platform. macOS and Windows have exactly one supported backend, so force it; anything else must
  # be Linux x86_64 (libaio or thread pool).
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ALAYA_LASER_USE_THREADPOOL
        ON
        CACHE BOOL "Use the portable thread-pool LASER I/O backend" FORCE
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64")
    set(ALAYA_LASER_USE_IOCP
        ON
        CACHE BOOL "Use the Windows IOCP LASER I/O backend" FORCE
    )
  elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(
      FATAL_ERROR
        "ALAYA_ENABLE_LASER=ON but this platform is not supported yet. "
        "Supported platforms: Linux x86_64 (libaio), macOS (thread pool), Windows x64 (IOCP). "
        "Configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module."
    )
  endif()
endif()
