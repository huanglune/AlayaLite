# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

# LASER module tests. These targets manage their exact link set (BARE): the backend file-reader tests deliberately do
# NOT link alaya_laser so they can pin a backend definition themselves regardless of the platform default, and the
# compile smoke tests link only the LASER surface. All of them build with the standard test-only flags plus the LASER
# consumer options (see _ALAYA_LASER_CONSUMER_OPTIONS).

if(NOT ALAYA_ENABLE_LASER)
  return()
endif()

set(_laser_test_opts
    -O3
    -DNDEBUG
    -Wall
    -Wextra
    ${_ALAYA_LASER_CONSUMER_OPTIONS}
)

set(_laser_reader_libs Eigen3::Eigen concurrentqueue::concurrentqueue spdlog::spdlog_header_only)
if(TARGET OpenMP::OpenMP_CXX)
  list(APPEND _laser_reader_libs OpenMP::OpenMP_CXX)
endif()

# Gate L1 — mechanical port compile smoke test. Verifies the Laser headers under include/index/graph/laser/ compile
# under AlayaLite's toolchain. See openspec/changes/port-laser-disk-index tasks.md item 2.13.
alaya_cc_target(
  test_laser_compile
  BARE
  SRCS test_laser_compile.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_compile
  TARGET test_laser_compile
  LABELS unit
)

# Round-trip test for FHTRotator::dump_signs: seeded determinism, seed sensitivity, and on-disk layout. See the test
# source for details.
alaya_cc_target(
  test_rotator_dump
  BARE
  SRCS test_rotator_dump.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_rotator_dump
  TARGET test_rotator_dump
  LABELS unit
)

alaya_cc_target(
  test_laser_page_layout_round_trip
  BARE GTEST
  SRCS qg/test_page_layout_round_trip.cpp
  LIBS alaya_laser
  DEFS ALAYA_LASER_TESTING
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_page_layout_round_trip
  TARGET test_laser_page_layout_round_trip
  LABELS laser
)

# Backend file-reader tests: pin the backend definition per target instead of linking alaya_laser, so each backend is
# exercised on every platform that can compile it.
alaya_cc_target(
  test_threadpool_file_reader
  BARE GTEST
  SRCS utils/test_threadpool_file_reader.cpp
  LIBS ${_laser_reader_libs}
  DEFS ALAYA_LASER_USE_THREADPOOL=1
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_threadpool_file_reader
  TARGET test_threadpool_file_reader
  LABELS laser threadpool
)

alaya_cc_target(
  laser_simd_dispatch_test
  BARE GTEST
  SRCS simd_dispatch_test.cpp
  LIBS alaya_laser
  OPTS -Wall -Wextra ${_ALAYA_LASER_CONSUMER_OPTIONS}
)
alaya_add_test(
  NAME laser_test_simd_dispatch
  TARGET laser_simd_dispatch_test
  LABELS laser simd
)

alaya_cc_target(
  rabitq_factor_equivalence_test
  BARE GTEST
  SRCS rabitq_factor_equivalence_test.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_rabitq_factor_equivalence
  TARGET rabitq_factor_equivalence_test
  LABELS laser rabitq
)

alaya_cc_target(
  qg_naming_contract_test
  GTEST LASER
  SRCS qg_naming_contract_test.cpp
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_qg_naming_contract
  TARGET qg_naming_contract_test
  LABELS laser rabitq contract
)

# Streaming-update research prototype (qg_updater.hpp): patch primitives, builder byte-equality, insert/tombstone
# end-to-end on a tiny index.
alaya_cc_target(
  test_qg_updater_unit
  BARE GTEST
  SRCS qg/test_qg_updater_unit.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_qg_updater_unit
  TARGET test_qg_updater_unit
  LABELS laser
)

# SEGMENT_OP op-WAL payload codec (segment_op_wal.hpp): round-trip + validation.
alaya_cc_target(
  test_segment_op_wal
  BARE GTEST
  SRCS qg/test_segment_op_wal.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_segment_op_wal
  TARGET test_segment_op_wal
  LABELS laser wal persistence
)

# Residency seams behind UnifiedLaserSegment: prefetch budget default, ResidencyProvider dispatch, ensure_resident_arena
# file materialization, and the QGUpdater write_at -> arena mirror.
alaya_cc_target(
  test_unified_residency
  BARE GTEST
  SRCS qg/test_unified_residency.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_unified_residency
  TARGET test_unified_residency
  LABELS laser
)

# Bench-only fbin mmap view used by update modes under cgroup RAM caps.
alaya_cc_target(
  test_bench_laser_update_sift_support
  BARE GTEST
  SRCS test_bench_laser_update_sift_support.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_bench_laser_update_sift_support
  TARGET test_bench_laser_update_sift_support
  LABELS laser
)

# SIFT-scale update experiment harness (build / insert / delete / eval modes). Not a ctest — run manually; see the usage
# string in the source.
alaya_cc_target(
  bench_laser_update_sift
  BARE
  SRCS bench_laser_update_sift.cpp bench_laser_oracles.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
# The vendored FFHT header contains non-inline helper definitions. Keep the bench's split source files maintainable
# while compiling them as one translation unit, as the original single-source target did.
set_target_properties(bench_laser_update_sift PROPERTIES UNITY_BUILD ON)
