# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaToolchain.cmake - language level and build accelerators.
#
# Owns the compiler-agnostic knobs: C++ standard, PIC, compile_commands.json, ccache, fast linkers, and sanitizer
# detection. Optimization/warning flags live in AlayaFlags.cmake.

include_guard(GLOBAL)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------------------------------------------------
# Sanitizer detection. `make build-san` (and the asan preset) inject -fsanitize=... through CMAKE_CXX_FLAGS; downstream
# code needs to know (optimization level drops to -O1, the LASER fixture preloads libasan).
# ---------------------------------------------------------------------------------------------------------------------
set(ALAYA_SANITIZER_BUILD OFF)
set(ALAYA_ASAN_RUNTIME)
if(CMAKE_CXX_FLAGS MATCHES "(^| )-fsanitize=")
  set(ALAYA_SANITIZER_BUILD ON)
  if(CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*address")
    execute_process(
      COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
      OUTPUT_VARIABLE ALAYA_ASAN_RUNTIME
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # GCC echoes the input name verbatim when the file is not on its search path; clear it so downstream LD_PRELOAD
    # guards stay correct.
    if(NOT EXISTS "${ALAYA_ASAN_RUNTIME}")
      set(ALAYA_ASAN_RUNTIME)
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------------------------------------------------
# ccache. The pybind11 module (_alayalitepy) is a single translation unit that expands dispatch.hpp's nested macros into
# ~288 template instantiations. Single-TU means -j N cannot parallelize it, so repeat builds lean on ccache. pch_defines
# + time/mtime/ctime sloppiness is required for PCH to actually hit cache; `cmake -E env` wraps each compile so
# CCACHE_SLOPPINESS is in scope for ccache itself, not just CMake's configure step.
# ---------------------------------------------------------------------------------------------------------------------
if(ALAYA_USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER
        ${CMAKE_COMMAND}
        -E
        env
        CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime
        ${CCACHE_PROGRAM}
    )
    set(CMAKE_CXX_COMPILER_LAUNCHER
        ${CMAKE_COMMAND}
        -E
        env
        CCACHE_SLOPPINESS=pch_defines,time_macros,include_file_mtime,include_file_ctime
        ${CCACHE_PROGRAM}
    )
    message(STATUS "ccache enabled: ${CCACHE_PROGRAM}")
  endif()
endif()

if(ALAYA_USE_FAST_LINKER AND NOT MSVC)
  find_program(MOLD_PROGRAM mold)
  find_program(LLD_PROGRAM ld.lld)
  if(MOLD_PROGRAM)
    add_link_options(-fuse-ld=mold)
    message(STATUS "Fast linker: mold (${MOLD_PROGRAM})")
  elseif(LLD_PROGRAM)
    add_link_options(-fuse-ld=lld)
    message(STATUS "Fast linker: lld (${LLD_PROGRAM})")
  endif()
endif()
