# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# _alayalitepy - the canonical Collection pybind11 extension module.

set(PYTHON_MODULE_NAME _alayalitepy)

# ---------------------------------------------------------------------------------------------------------------------
# The module. pybind11 resolves Python through FindPython (PYBIND11_FINDPYTHON=ON set before find_package(pybind11) in
# AlayaDependencies.cmake), i.e. the same interpreter as everything else.
# ---------------------------------------------------------------------------------------------------------------------
pybind11_add_module(
  ${PYTHON_MODULE_NAME}
  MODULE
  NO_EXTRAS
  src/pybind.cpp
  src/collection_binding_factory_impl.cpp
  src/collection_binding_lifecycle.cpp
  src/collection_binding_management_impl.cpp
  src/collection_binding_mutation.cpp
  src/collection_binding_mutation_impl.cpp
  src/collection_binding_read_impl.cpp
  src/collection_binding_search.cpp
  src/collection_binding_search_impl.cpp
  src/collection_binding_types.cpp
)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE AlayaLite pybind11::module)
alaya_apply_build_flags(${PYTHON_MODULE_NAME})
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE -Wno-nonnull)
endif()

# Precompile the recurring pybind11 and STL surface used by the canonical Collection binding.
target_precompile_headers(
  ${PYTHON_MODULE_NAME}
  PRIVATE
  <pybind11/pybind11.h>
  <pybind11/numpy.h>
  <pybind11/stl.h>
  <pybind11/cast.h>
  <pybind11/pytypes.h>
  <vector>
  <string>
  <memory>
  <unordered_map>
)

# LASER remains linked for the canonical disk reader and the selected-SIMD diagnostic. SIMD-dependent flags are PRIVATE.
if(ALAYA_ENABLE_LASER)
  target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE alaya_laser)
  set_property(TARGET ${PYTHON_MODULE_NAME} PROPERTY ALAYA_LASER_CONSUMER TRUE)
  target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE ALAYA_ENABLE_LASER=1)
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC 11 can ICE while auto-vectorizing LASER-enabled binding code. The handwritten LASER kernels still use target
    # attributes; this only disables generic tree vectorization for the binding translation units.
    target_compile_options(${PYTHON_MODULE_NAME} PRIVATE -fno-tree-vectorize)
  endif()
endif()

install(
  TARGETS ${PYTHON_MODULE_NAME}
  LIBRARY DESTINATION alayalite
  RUNTIME DESTINATION alayalite
)

if(BUILD_TESTING)
  add_test(NAME python_binding_incremental_test
           COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/tests/binding/native_incremental_ctest.py
                   $<TARGET_FILE:${PYTHON_MODULE_NAME}>
  )
  set_tests_properties(python_binding_incremental_test PROPERTIES LABELS "unit;python;binding" TIMEOUT 60)
endif()
