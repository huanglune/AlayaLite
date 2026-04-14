include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_executable(diskann_update_bench ${CMAKE_CURRENT_SOURCE_DIR}/diskann_update_bench.cpp)
target_link_libraries(diskann_update_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)

add_executable(graph_search_bench ${CMAKE_CURRENT_SOURCE_DIR}/graph_search_bench.cpp)
target_link_libraries(graph_search_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)

add_executable(diskann_search_bench ${CMAKE_CURRENT_SOURCE_DIR}/diskann_search_bench.cpp)
target_link_libraries(diskann_search_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)

add_executable(build_vamana_graph ${CMAKE_CURRENT_SOURCE_DIR}/build_vamana_graph.cpp)
target_link_libraries(build_vamana_graph PRIVATE ${THIRD_PARTY_LIBS})

if(LASER_AVAILABLE)
  add_executable(laser_build_bench ${CMAKE_CURRENT_SOURCE_DIR}/laser_build_bench.cpp)
  target_link_libraries(laser_build_bench PRIVATE ${THIRD_PARTY_LIBS})
endif()
