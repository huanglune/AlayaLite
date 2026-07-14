# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# AlayaFlags.cmake - compile flags as linkable INTERFACE targets instead of directory-global state.
#
# Every in-tree target opts in by linking `alaya_build_flags` (the alaya_cc_* helpers do this automatically). Nothing
# leaks to out-of-tree consumers of the AlayaLite INTERFACE library, and per-target overrides (LASER's -O3, coverage's
# -O0) stay ordinary target_compile_options calls that appear later on the command line.
#
# Flag policy: - Warnings (-Wall, /EHsc /utf-8) apply in every configuration. - SIMD baseline (-mavx2 -mfma on x86_64
# when ALAYA_X86_AVX2_BASELINE=ON, or -march=native with ALAYA_NATIVE_ARCH=ON) applies in every configuration. Runtime
# dispatched SIMD kernels use function-level target attributes and also compile with the generic baseline. -
# -Ofast/-funroll-loops are Release-only. Debug builds now really are debuggable (this deliberately diverges from the
# pre-rewrite behavior, where -Ofast overrode -g/-O0 in every configuration). MSVC needs no explicit /O2: its Release
# configuration already carries it, and forcing it globally used to fight /Od+/RTC1 in Debug (D9025). - Sanitizer builds
# (detected in AlayaToolchain.cmake from -fsanitize in CMAKE_CXX_FLAGS) pin -O1 in all configurations so UBSan/ASan
# reports keep usable stacks without making the suite unbearably slow.

include_guard(GLOBAL)

set(ALAYA_SIMD_COMPILE_OPTIONS)
if(NOT MSVC)
  if(ALAYA_NATIVE_ARCH)
    list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -march=native)
  elseif(ALAYA_X86_AVX2_BASELINE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND ALAYA_SIMD_COMPILE_OPTIONS -mavx2 -mfma)
  endif()
endif()

add_library(alaya_build_flags INTERFACE)

if(MSVC)
  target_compile_options(alaya_build_flags INTERFACE /EHsc /utf-8)
else()
  target_compile_options(alaya_build_flags INTERFACE -Wall ${ALAYA_SIMD_COMPILE_OPTIONS})
  if(ALAYA_SANITIZER_BUILD)
    target_compile_options(alaya_build_flags INTERFACE -O1)
  else()
    target_compile_options(alaya_build_flags INTERFACE $<$<CONFIG:Release>:-Ofast> $<$<CONFIG:Release>:-funroll-loops>)
  endif()
endif()

# Convenience for targets not created through the alaya_cc_* helpers (e.g. the pybind11 module).
function(alaya_apply_build_flags target_name)
  target_link_libraries(${target_name} PRIVATE alaya_build_flags)
endfunction()
