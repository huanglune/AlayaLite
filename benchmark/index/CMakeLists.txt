include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(diskann_update_bench ${CMAKE_CURRENT_SOURCE_DIR}/diskann_update_bench.cpp)
target_link_libraries(diskann_update_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)

add_executable(graph_search_bench ${CMAKE_CURRENT_SOURCE_DIR}/graph_search_bench.cpp)
target_link_libraries(graph_search_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)
