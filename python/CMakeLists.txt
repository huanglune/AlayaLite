set(PYTHON_MODULE_NAME _alayalitepy)

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED
)

# Automatically process various attributes (such as suffixes, visibility, etc.)
pybind11_add_module(${PYTHON_MODULE_NAME} MODULE NO_EXTRAS src/pybind.cpp)
target_include_directories(${PYTHON_MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE AlayaLite pybind11::module)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE -Wno-nonnull)
endif()

# Precompile pybind11 + Eigen + STL headers. _alayalitepy is a single-TU module whose dispatch.hpp expansion drags in
# pybind11/numpy.h, Eigen, and the bulk of <vector>/<string>/<memory>/<unordered_map> hundreds of times through template
# instantiation. A PCH cuts the parse+template-instantiation cost on the recurring header surface and is the single
# biggest lever for this TU's compile time. Requires CMake >= 3.16; gate accordingly.
if(NOT CMAKE_VERSION VERSION_LESS 3.16)
  target_precompile_headers(
    ${PYTHON_MODULE_NAME}
    PRIVATE
    <pybind11/pybind11.h>
    <pybind11/numpy.h>
    <pybind11/stl.h>
    <pybind11/cast.h>
    <pybind11/pytypes.h>
    <Eigen/Core>
    <vector>
    <string>
    <memory>
    <unordered_map>
  )
endif()

# The Laser on-disk QG pybind is compiled as a submodule of _alayalitepy. SIMD-dependent flags (-march=native, etc.) are
# applied PRIVATE so they don't leak to anything else that might link the Laser INTERFACE target downstream.
if(ALAYA_ENABLE_LASER)
  target_link_libraries(${PYTHON_MODULE_NAME} PRIVATE alaya_laser)
  target_compile_definitions(${PYTHON_MODULE_NAME} PRIVATE ALAYA_ENABLE_LASER=1)
  target_compile_options(${PYTHON_MODULE_NAME} PRIVATE ${_ALAYA_LASER_CONSUMER_OPTIONS})
endif()

install(
  TARGETS ${PYTHON_MODULE_NAME}
  LIBRARY DESTINATION alayalite
  RUNTIME DESTINATION alayalite
)
