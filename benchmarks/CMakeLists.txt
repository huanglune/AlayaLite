add_subdirectory(simd)
add_subdirectory(rabitq)

# Result-contract A/B harness — build only, not registered with ctest. It constructs/reuses real resident-arena LASER
# artifacts and drives them through SegmentedCollection with the historical rank-only contract versus the opt-in
# numeric-distance prototype.
if(ALAYA_ENABLE_LASER)
  alaya_cc_target(
    result_contract_benchmark
    LASER
    SRCS result_contract_benchmark.cpp
  )
endif()
