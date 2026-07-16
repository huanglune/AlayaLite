# DiskANN benchmarks — build only, not registered with ctest.
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    bench_diskann_sift
    LASER
    SRCS bench_diskann_sift.cpp
  )
  alaya_cc_target(
    bench_diskann_sift_update
    LASER
    SRCS bench_diskann_sift_update.cpp
  )
  alaya_cc_target(
    diskann_async_sentinel
    LASER
    SRCS diskann_async_sentinel.cpp
  )
  alaya_cc_target(
    topology_preserving_seal_poc
    LASER
    SRCS topology_preserving_seal_poc.cpp
  )
endif()
alaya_cc_target(make_diskann_sift_update_trace SRCS make_diskann_sift_update_trace.cpp)
target_include_directories(make_diskann_sift_update_trace PRIVATE ${CMAKE_SOURCE_DIR}/tests/diskann)
