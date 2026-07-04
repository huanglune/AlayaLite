# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# ConanSetup.cmake - locate (or produce) the Conan-generated dependency toolchain and include it.
#
# Dependencies come from Conan 2 (see conanfile.py). `scripts/conan_build/conan_install.py` wraps profile detection and
# `conan install`; it is invoked automatically on first configure unless ALAYA_AUTO_CONAN=OFF. The generator output
# location follows conanfile.py's cmake_layout and is intentionally anchored to the SOURCE tree, not the binary dir, so
# one `conan install` serves every build tree of the same build type (Makefile dirs, presets, scikit-build).

include_guard(GLOBAL)

# cmake_layout output path: Windows -> build/generators/, Others -> build/<type>/generators/
if(WIN32)
  set(CONAN_GENERATORS_DIR "${CMAKE_SOURCE_DIR}/build/generators")
else()
  set(CONAN_GENERATORS_DIR "${CMAKE_SOURCE_DIR}/build/${CMAKE_BUILD_TYPE}/generators")
endif()
set(CONAN_TOOLCHAIN_FILE "${CONAN_GENERATORS_DIR}/conan_toolchain.cmake")

if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
  set(CONAN_INSTALL_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/conan_build/conan_install.py")
  set(_conan_manual_command "${Python_EXECUTABLE} ${CONAN_INSTALL_SCRIPT} --build-type ${CMAKE_BUILD_TYPE}")

  if(NOT ALAYA_AUTO_CONAN)
    message(FATAL_ERROR "Conan toolchain not found (${CONAN_TOOLCHAIN_FILE}) and ALAYA_AUTO_CONAN=OFF.\n"
                        "Install dependencies manually first:\n  ${_conan_manual_command}"
    )
  endif()

  if(NOT DEFINED Python_EXECUTABLE)
    # Reached only when a caller skips alaya_preflight(); keep a fallback so this module stays self-contained.
    find_package(Python REQUIRED COMPONENTS Interpreter)
  endif()

  message(STATUS "Running: ${CONAN_INSTALL_SCRIPT} (first configure only; disable with -DALAYA_AUTO_CONAN=OFF)")
  execute_process(
    COMMAND ${Python_EXECUTABLE} "${CONAN_INSTALL_SCRIPT}" --project-dir "${CMAKE_SOURCE_DIR}" --build-type
            "${CMAKE_BUILD_TYPE}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE _conan_result
  )

  if(NOT _conan_result EQUAL 0)
    message(
      FATAL_ERROR
        "Conan install failed with exit code ${_conan_result}. The dependency error is printed above; the usual "
        "environment causes are:\n"
        "  - no C compiler / make / ninja for source-built packages (conan builds missing binaries locally)\n"
        "  - no network access to conancenter on first run\n"
        "Re-run it by hand to iterate faster:\n  ${_conan_manual_command}"
    )
  endif()
endif()

# CMake caches `<Package>_DIR` entries. If a build directory is reconfigured from Debug to Release (or the reverse),
# those cached entries can keep pointing at the previous Conan generator directory, stripping dependency include paths
# from generated compile commands.
file(GLOB _conan_config_files "${CONAN_GENERATORS_DIR}/*Config.cmake" "${CONAN_GENERATORS_DIR}/*-config.cmake")
foreach(conan_config_file IN LISTS _conan_config_files)
  get_filename_component(_conan_config_name "${conan_config_file}" NAME)
  string(REGEX REPLACE "(-config|Config)\\.cmake$" "" _conan_package_name "${_conan_config_name}")
  set(_conan_package_dir_var "${_conan_package_name}_DIR")
  if(DEFINED CACHE{${_conan_package_dir_var}})
    get_property(
      _conan_cached_dir
      CACHE "${_conan_package_dir_var}"
      PROPERTY VALUE
    )
    if(_conan_cached_dir AND NOT _conan_cached_dir STREQUAL CONAN_GENERATORS_DIR)
      unset(${_conan_package_dir_var} CACHE)
    endif()
  endif()
endforeach()

if(NOT EXISTS "${CONAN_TOOLCHAIN_FILE}")
  message(FATAL_ERROR "Conan toolchain not found: ${CONAN_TOOLCHAIN_FILE}")
endif()
include("${CONAN_TOOLCHAIN_FILE}")
