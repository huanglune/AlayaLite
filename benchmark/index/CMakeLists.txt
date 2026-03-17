add_executable(diskann_searcher_benchmark ${CMAKE_CURRENT_SOURCE_DIR}/diskann_searcher_benchmark.cpp)

target_compile_definitions(diskann_searcher_benchmark PRIVATE ALAYA_PROJECT_ROOT="${CMAKE_SOURCE_DIR}")
target_link_libraries(diskann_searcher_benchmark PRIVATE ${THIRD_PARTY_LIBS} benchmark::benchmark)
