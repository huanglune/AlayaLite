add_executable(diskann_searcher_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/diskann_searcher_benchmark.cpp)

target_link_libraries(diskann_searcher_benchmark PRIVATE ${THIRD_PARTY_LIBS} benchmark::benchmark)
