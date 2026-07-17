# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaPython.cmake - single source of truth for Python discovery.
#
# Everything in this project resolves Python through CMake's FindPython module (the `Python_*` variable family): the
# pybind11 module (interpreter + Development.Module) and the LASER test fixture (interpreter). pybind11 is pointed at
# the same result via PYBIND11_FINDPYTHON=ON, so there is exactly one interpreter per build tree and one hint variable
# to override it: -DPython_EXECUTABLE=<path>.
#
# Resolution order: 1. An explicit -DPython_EXECUTABLE / -DPython_ROOT_DIR wins (this is also how scikit-build-core
# drives wheel builds — it passes FindPython hints for the interpreter cibuildwheel selected). 2. -DPython3_EXECUTABLE
# is accepted as a legacy alias and mapped onto Python_EXECUTABLE. 3. An activated virtualenv ($ENV{VIRTUAL_ENV}) is
# honoured via Python_FIND_VIRTUALENV. 4. The project-local uv venv (<repo>/.venv, created by `uv sync`) is used when
# present. 5. Whatever FindPython locates on PATH.
#
# System interpreters frequently lack the python3-dev headers; the uv-managed .venv always ships them, which is why the
# project prefers it.

include_guard(GLOBAL)

# Apply the hint chain above. Idempotent; called before every FindPython probe.
macro(alaya_python_hints)
  if(NOT DEFINED Python_EXECUTABLE AND DEFINED Python3_EXECUTABLE)
    message(STATUS "Mapping legacy -DPython3_EXECUTABLE onto Python_EXECUTABLE (${Python3_EXECUTABLE})")
    set(Python_EXECUTABLE
        "${Python3_EXECUTABLE}"
        CACHE FILEPATH "Path to the Python interpreter" FORCE
    )
  endif()

  if(NOT DEFINED Python_EXECUTABLE
     AND NOT DEFINED Python_ROOT_DIR
     AND NOT DEFINED SKBUILD
     AND NOT DEFINED ENV{VIRTUAL_ENV}
  )
    if(WIN32)
      set(_alaya_venv_python "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    else()
      set(_alaya_venv_python "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${_alaya_venv_python}")
      message(STATUS "Using project virtualenv Python: ${_alaya_venv_python} (override with -DPython_EXECUTABLE)")
      set(Python_EXECUTABLE
          "${_alaya_venv_python}"
          CACHE FILEPATH "Path to the Python interpreter" FORCE
      )
    endif()
    unset(_alaya_venv_python)
  endif()

  # Resolve an activated virtualenv before any system interpreter.
  set(Python_FIND_VIRTUALENV FIRST)
endmacro()

# alaya_find_python(<components...>) - hints + find_package(Python REQUIRED). Safe to call repeatedly with growing
# component lists; FindPython accumulates components in the cache. BYPASS_PROVIDER keeps Python discovery away from the
# Conan dependency provider: Python is never a Conan package, and the first intercepted find_package() must be a real
# C++ dependency so the derived profile sees the fully configured toolchain (CMAKE_CXX_STANDARD -> cppstd).
macro(alaya_find_python)
  alaya_python_hints()
  find_package(Python REQUIRED COMPONENTS ${ARGN} BYPASS_PROVIDER)
endmacro()
