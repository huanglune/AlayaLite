# SIMD micro-benchmarks — build only, not registered with ctest.
alaya_cc_target(l2_sqr_full_benchmark SRCS l2_sqr_full_benchmark.cpp)
alaya_cc_target(l2_sqr_sq8_benchmark SRCS l2_sqr_sq8_benchmark.cpp)
alaya_cc_target(l2_sqr_sq4_benchmark SRCS l2_sqr_sq4_benchmark.cpp)
alaya_cc_target(ip_full_benchmark SRCS ip_full_benchmark.cpp)
alaya_cc_target(ip_sq8_benchmark SRCS ip_sq8_benchmark.cpp)
alaya_cc_target(ip_sq4_benchmark SRCS ip_sq4_benchmark.cpp)
alaya_cc_target(fht_benchmark SRCS fht_benchmark.cpp)
