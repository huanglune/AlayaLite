# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaPreflight.cmake - fail fast, with instructions.
#
# Historically a missing tool surfaced as an inscrutable error several layers deep (a Conan dependency failing to
# compile, or FindPython dying inside pybind11). This module checks the environment up front and, when something is
# missing, says exactly what to install or which cache variable to pass.

include_guard(GLOBAL)
include(${CMAKE_CURRENT_LIST_DIR}/AlayaPython.cmake)

macro(alaya_preflight)
  # -------------------------------------------------------------------------------------------------------------------
  # Compiler floor. The codebase is C++20 with coroutines (libcoro) and heavy template metaprogramming; these are the
  # oldest toolchains the project actually builds with (CI: gcc-11/gcc-13, AppleClang 15+, MSVC 2022).
  # -------------------------------------------------------------------------------------------------------------------
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
    message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old: AlayaLite needs GCC >= 11 for C++20 "
                        "coroutines. Install a newer GCC or pass -DCMAKE_CXX_COMPILER=g++-13."
    )
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^Clang$" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(FATAL_ERROR "Clang ${CMAKE_CXX_COMPILER_VERSION} is too old: AlayaLite needs Clang >= 14 for C++20. "
                        "Install a newer Clang or pass -DCMAKE_CXX_COMPILER=clang++-16."
    )
  elseif(MSVC AND MSVC_VERSION LESS 1930)
    message(FATAL_ERROR "MSVC toolset ${MSVC_VERSION} is too old: AlayaLite needs Visual Studio 2022 (toolset "
                        "19.30+) for C++20."
    )
  endif()

  # -------------------------------------------------------------------------------------------------------------------
  # Python. The interpreter is always needed (Conan bootstrap + test fixtures); the C API headers only when the
  # extension module is built. Probe QUIETly first so we can turn "Could NOT find Python" into an actionable message.
  # -------------------------------------------------------------------------------------------------------------------
  if(BUILD_PYTHON)
    set(_alaya_python_components Interpreter Development.Module)
  else()
    set(_alaya_python_components Interpreter)
  endif()

  alaya_python_hints()
  find_package(Python QUIET COMPONENTS ${_alaya_python_components} BYPASS_PROVIDER)
  if(NOT Python_FOUND)
    if(BUILD_PYTHON
       AND Python_Interpreter_FOUND
       AND NOT Python_Development.Module_FOUND
    )
      message(
        FATAL_ERROR
          "Python interpreter ${Python_EXECUTABLE} has no C API headers (Development.Module), so the pybind11 "
          "module cannot be built. Pick one:\n"
          "  1. create the project venv, which ships headers:   uv sync        (then reconfigure)\n"
          "  2. point CMake at another interpreter:             -DPython_EXECUTABLE=/path/to/python\n"
          "  3. install the system headers:                     sudo apt-get install python3-dev\n"
          "  4. skip the extension module entirely:             -DBUILD_PYTHON=OFF"
      )
    elseif(NOT Python_Interpreter_FOUND)
      message(FATAL_ERROR "No Python interpreter found. Install Python 3.9+ (or run `uv sync` to create the "
                          "project venv) or pass -DPython_EXECUTABLE=/path/to/python."
      )
    endif()
  endif()
  # Bind the probe result for real (also maps legacy Python3_* hints and the .venv preference).
  alaya_find_python(${_alaya_python_components})
  unset(_alaya_python_components)
endmacro()
