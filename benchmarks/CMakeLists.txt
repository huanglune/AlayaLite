add_subdirectory(simd)
add_subdirectory(rabitq)
add_subdirectory(laser)

# AnySegment synchronous-adapter microbenchmark — build only, not registered with ctest. It keeps the engine work
# deliberately small so the fixed adapter execution overhead remains visible in P50/P99 and QPS.
alaya_cc_target(any_segment_sync_benchmark SRCS any_segment_sync_benchmark.cpp)

# Result-contract A/B harness — build only, not registered with ctest. It constructs/reuses real resident-arena LASER
# artifacts and drives them through SegmentedCollection with the historical rank-only contract versus the opt-in
# numeric-distance prototype.
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    result_contract_benchmark
    LASER
    SRCS result_contract_benchmark.cpp
  )

  # LASER C-lanes harness — build only, not registered with ctest. Direct mode pairs the selected residency with a
  # resident-arena reference; Collection mode pairs the segment and public Collection faces.
  alaya_cc_target(
    parity_lanes_benchmark
    LASER
    SRCS parity_lanes_benchmark.cpp
  )
endif()
