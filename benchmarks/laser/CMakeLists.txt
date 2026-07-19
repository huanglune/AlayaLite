# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

if(NOT ALAYA_ENABLE_LASER OR NOT ALAYA_ENABLE_MUTABLE_LASER_TESTS)
  return()
endif()

set(_laser_benchmark_opts
    -O3
    -DNDEBUG
    -Wall
    -Wextra
    ${_ALAYA_LASER_CONSUMER_OPTIONS}
)

# SIFT-scale update experiment harness (build / insert / delete / eval modes). Not a ctest — run manually; see the usage
# string in the source.
alaya_cc_target(
  bench_laser_update_sift
  BARE
  SRCS bench_laser_update_sift.cpp bench_laser_oracles.cpp
  LIBS alaya_laser
  OPTS ${_laser_benchmark_opts}
)
# The vendored FFHT header contains non-inline helper definitions. Keep the bench's split source files maintainable
# while compiling them as one translation unit, as the original single-source target did.
set_target_properties(bench_laser_update_sift PROPERTIES UNITY_BUILD ON)
