# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaConan.cmake - register the Conan dependency provider. Included from the top-level CMakeLists BEFORE project().
#
# Dependency resolution uses the official cmake-conan provider (cmake/vendor/conan_provider.cmake, pinned from
# https://github.com/conan-io/cmake-conan @ b1593849dd84): the first find_package() triggers `conan install` with a
# host profile derived from the actual CMake toolchain state (compiler, build type, arch, cppstd), and the generated
# CMakeDeps config packages land under ${CMAKE_BINARY_DIR}/conan. Compared to the previous hand-rolled integration
# this needs no toolchain-file include, no configure-time wrapper script, and no source-tree-anchored generator
# directory — but it does require the `conan` executable (>= 2.0.5) on PATH. PEP 517 builds (uv sync / uv build /
# cibuildwheel) get it automatically from [build-system].requires; interactive users install it once.
#
# Opt out with -DALAYA_AUTO_CONAN=OFF and provide dependencies yourself:
#   conan install . --build=missing -s build_type=Release \
#     && cmake -B build/manual -DALAYA_AUTO_CONAN=OFF -DCMAKE_PREFIX_PATH=<generators dir from the install output>

include_guard(GLOBAL)

# option(ALAYA_AUTO_CONAN) is declared later in AlayaOptions.cmake; before project() only an explicit -D shows up.
if(DEFINED ALAYA_AUTO_CONAN AND NOT ALAYA_AUTO_CONAN)
  message(STATUS "ALAYA_AUTO_CONAN=OFF: dependency provider disabled, expecting a manual conan install")
elseif(DEFINED CMAKE_PROJECT_TOP_LEVEL_INCLUDES)
  message(STATUS "CMAKE_PROJECT_TOP_LEVEL_INCLUDES already set; not registering the bundled Conan provider")
else()
  find_program(ALAYA_CONAN_EXECUTABLE conan)
  if(NOT ALAYA_CONAN_EXECUTABLE)
    message(
      FATAL_ERROR
        "The `conan` executable was not found on PATH; it drives all C++ dependencies. Install it once with:\n"
        "  uv tool install conan     (or: pipx install conan / pip install --user conan)\n"
        "then reconfigure. Alternatively pass -DALAYA_AUTO_CONAN=OFF and manage dependencies yourself."
    )
  endif()
  set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES ${CMAKE_CURRENT_LIST_DIR}/vendor/conan_provider.cmake)
endif()
