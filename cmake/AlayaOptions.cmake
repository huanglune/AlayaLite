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
option(ALAYA_X86_AVX2_BASELINE "Use AVX2+FMA as the x86 compile baseline" ON)
option(ALAYA_ALLOW_NATIVE_PACKAGE "Allow BUILD_PYTHON with host-specific -march=native" OFF)
option(ALAYA_USE_CCACHE "Use ccache as compiler launcher when available" ON)
option(ALAYA_USE_FAST_LINKER "Use mold or lld when available" ON)
# Consumed before project() by AlayaConan.cmake; declared here so it shows up in cmake -LH with the other switches.
option(ALAYA_AUTO_CONAN "Resolve C++ dependencies via the bundled Conan dependency provider" ON)

if(BUILD_PYTHON
   AND ALAYA_NATIVE_ARCH
   AND NOT ALAYA_ALLOW_NATIVE_PACKAGE
)
  message(FATAL_ERROR "ALAYA_NATIVE_ARCH=ON makes Python package builds host-specific and unsafe to distribute. "
                      "Use -DALAYA_ALLOW_NATIVE_PACKAGE=ON only for local, non-distributable builds."
  )
endif()

# ---------------------------------------------------------------------------------------------------------------------
# Laser on-disk QG index module. Linux x86_64 defaults to libaio; macOS defaults to the portable thread-pool backend.
# Windows is excluded since the IOCP backend was removed (the module no longer compiles there); other platforms skip it
# silently by default.
# ---------------------------------------------------------------------------------------------------------------------
if((CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64") OR CMAKE_SYSTEM_NAME STREQUAL
                                                                                             "Darwin"
)
  set(ALAYA_ENABLE_LASER_DEFAULT ON)
else()
  set(ALAYA_ENABLE_LASER_DEFAULT OFF)
endif()
option(ALAYA_ENABLE_LASER "Build the Laser disk-index module" ${ALAYA_ENABLE_LASER_DEFAULT})

# Keep the default/platform admission aligned with collection.hpp's ALAYA_COLLECTION_HAS_ACTIVE_LASER condition
# (ALAYA_ENABLE_LASER && __linux__). Linux may explicitly disable this test capability; explicitly enabling it on
# another platform is an error.
set(ALAYA_ENABLE_MUTABLE_LASER_DEFAULT OFF)
if(ALAYA_ENABLE_LASER AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ALAYA_ENABLE_MUTABLE_LASER_DEFAULT ON)
endif()
option(ALAYA_ENABLE_MUTABLE_LASER
       "Build Linux-only mutable LASER test targets (requires LASER; non-Linux ON is an error)"
       ${ALAYA_ENABLE_MUTABLE_LASER_DEFAULT}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT ON)
else()
  set(ALAYA_LASER_USE_THREADPOOL_DEFAULT OFF)
endif()
option(ALAYA_LASER_USE_THREADPOOL "Use the portable thread-pool LASER I/O backend"
       ${ALAYA_LASER_USE_THREADPOOL_DEFAULT}
)

# The Windows IOCP LASER backend was removed; the option is kept only so an explicit -DALAYA_LASER_USE_IOCP=ON fails
# loudly in AlayaLaser.cmake instead of silently selecting a backend that no longer exists.
option(ALAYA_LASER_USE_IOCP "Removed Windows IOCP LASER I/O backend (unsupported)" OFF)

if(ALAYA_ENABLE_LASER)
  # Pin the backend per platform. macOS has exactly one supported backend, so force it; anything else must be Linux
  # x86_64 (libaio or thread pool). Windows lost its only backend when IOCP was removed.
  if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(ALAYA_LASER_USE_THREADPOOL
        ON
        CACHE BOOL "Use the portable thread-pool LASER I/O backend" FORCE
    )
  elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(
      FATAL_ERROR
        "ALAYA_ENABLE_LASER=ON but this platform is not supported. "
        "Supported platforms: Linux x86_64 (libaio or thread pool), macOS (thread pool). "
        "Configure with -DALAYA_ENABLE_LASER=OFF to skip the Laser module."
    )
  endif()
endif()

if(ALAYA_ENABLE_MUTABLE_LASER AND NOT ALAYA_ENABLE_LASER)
  message(FATAL_ERROR "ALAYA_ENABLE_MUTABLE_LASER=ON requires ALAYA_ENABLE_LASER=ON.")
elseif(ALAYA_ENABLE_MUTABLE_LASER AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(
    FATAL_ERROR
      "ALAYA_ENABLE_MUTABLE_LASER=ON is supported only on Linux; the mutable updater uses Linux-only APIs. "
      "Keep ALAYA_ENABLE_LASER=ON for sealed LASER support and configure with -DALAYA_ENABLE_MUTABLE_LASER=OFF."
  )
endif()
