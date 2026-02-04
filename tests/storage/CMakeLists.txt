add_executable(storage_test ${CMAKE_CURRENT_SOURCE_DIR}/sequential_storage_test.cpp)
add_executable(static_storage_test ${CMAKE_CURRENT_SOURCE_DIR}/static_storage_test.cpp)
add_executable(direct_file_io_test ${CMAKE_CURRENT_SOURCE_DIR}/direct_file_io_test.cpp)
add_executable(buffer_pool_test ${CMAKE_CURRENT_SOURCE_DIR}/buffer_pool_test.cpp)

target_link_libraries(storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(static_storage_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(direct_file_io_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})
target_link_libraries(buffer_pool_test PRIVATE ${THIRD_PARTY_LIBS} ${GTEST_LIBS})

add_coverage_to_target(storage_test)
add_coverage_to_target(static_storage_test)
add_coverage_to_target(direct_file_io_test)
add_coverage_to_target(buffer_pool_test)

add_test(NAME storage_test_sequential_storage COMMAND $<TARGET_FILE:storage_test>)
add_test(NAME storage_test_static_storage COMMAND $<TARGET_FILE:static_storage_test>)
add_test(NAME storage_test_direct_file_io COMMAND $<TARGET_FILE:direct_file_io_test>)
add_test(NAME storage_test_buffer_pool COMMAND $<TARGET_FILE:buffer_pool_test>)
