add_executable(diskann_update_bench ${CMAKE_CURRENT_SOURCE_DIR}/diskann_update_bench.cpp)
target_link_libraries(diskann_update_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)

add_executable(hnsw_bench ${CMAKE_CURRENT_SOURCE_DIR}/hnsw_bench.cpp)
target_link_libraries(hnsw_bench PRIVATE ${THIRD_PARTY_LIBS} tomlplusplus::tomlplusplus)
