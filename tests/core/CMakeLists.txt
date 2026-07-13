alaya_cc_target(core_value_types_test SRCS core_value_types_test.cpp GTEST)
alaya_add_test(NAME core_value_types_test TARGET core_value_types_test LABELS unit core)

alaya_cc_target(any_segment_test SRCS any_segment_test.cpp GTEST)
alaya_add_test(NAME any_segment_test TARGET any_segment_test LABELS unit core)

alaya_cc_target(core_value_types_header_closure SRCS value_types_header_closure.cpp)
alaya_cc_target(core_capabilities_header_closure SRCS capabilities_header_closure.cpp)
alaya_cc_target(core_status_header_closure SRCS status_header_closure.cpp)
alaya_cc_target(core_resource_contexts_header_closure SRCS resource_contexts_header_closure.cpp)
alaya_cc_target(core_algorithm_registry_header_closure SRCS algorithm_registry_header_closure.cpp)
alaya_cc_target(core_any_segment_header_closure SRCS any_segment_header_closure.cpp)

add_test(
  NAME core_header_boundaries
  COMMAND ${CMAKE_COMMAND} -DALAYA_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/check_core_header_boundaries.cmake
)
set_tests_properties(core_header_boundaries PROPERTIES LABELS "unit;core")
