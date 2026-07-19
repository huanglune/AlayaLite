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

# Paged search must consume asynchronous reads in a logical order that is independent of backend completion timing. The
# same source is exercised with the configured libaio and thread-pool builds, including the TSan lane.
alaya_cc_target(
  test_laser_paged_determinism
  BARE GTEST LASER
  SRCS qg/test_paged_determinism.cpp
  OPTS ${_laser_test_opts}
)
if(ALAYA_LASER_USE_THREADPOOL)
  set(_laser_paged_backend_label threadpool)
else()
  set(_laser_paged_backend_label libaio)
endif()
alaya_add_test(
  NAME laser_test_paged_determinism
  TARGET test_laser_paged_determinism
  LABELS laser
         paged
         determinism
         disk
         tsan
         ${_laser_paged_backend_label}
  TIMEOUT 180 RUN_SERIAL
)
if(ALAYA_LASER_USE_THREADPOOL)
  # This controls only the deprecated AlignedFileReader used while building the fixture. The search path under test uses
  # ThreadpoolPageReader (which ignores this variable) and still runs its backend plus four query lanes concurrently
  # under TSan.
  set_tests_properties(laser_test_paged_determinism PROPERTIES ENVIRONMENT "ALAYA_LASER_IO_THREADS=1")
endif()

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
  LABELS laser threadpool RUN_SERIAL
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

if(ALAYA_ENABLE_MUTABLE_LASER_TESTS)
  # All consumers below have the BARE+GTEST+LASER+_laser_test_opts profile. Keeping the owner profile identical is
  # required by CMake's REUSE_FROM contract; ALAYA_LASER_TESTING and non-GTest targets intentionally opt out.
  alaya_cc_target(
    alaya_laser_qg_test_pch
    BARE GTEST
    SRCS ../pch_anchor.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts}
  )
  target_precompile_headers(alaya_laser_qg_test_pch PRIVATE <gtest/gtest.h> <utils/laser_test_pch.hpp>)

  # Streaming-update research prototype (qg_updater.hpp): patch primitives, builder byte-equality, insert/tombstone
  # end-to-end on a tiny index.
  alaya_cc_target(
    test_qg_updater_unit
    BARE GTEST
    SRCS qg/test_qg_updater_unit.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
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

  # QGUpdater op-WAL functional durability (enable_wal): reopen/recovery round-trips.
  alaya_cc_target(
    test_qg_updater_wal
    BARE GTEST
    SRCS qg/test_qg_updater_wal.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  alaya_add_test(
    NAME laser_test_qg_updater_wal
    TARGET test_qg_updater_wal
    LABELS laser wal persistence
  )

  alaya_cc_target(
    test_qg_maintenance_concurrency
    BARE GTEST
    SRCS qg/test_qg_maintenance_concurrency.cpp
    LIBS alaya_laser
    DEFS ALAYA_LASER_TESTING
    OPTS ${_laser_test_opts}
  )
  alaya_add_test(
    NAME laser_test_qg_maintenance_concurrency
    TARGET test_qg_maintenance_concurrency
    LABELS laser wal persistence tsan
    TIMEOUT 120
  )

  # G1 crash matrix: SIGKILL/failpoint cuts + a power-loss persistence model.
  alaya_cc_target(
    test_segment_op_wal_crash
    BARE GTEST
    SRCS qg/test_segment_op_wal_crash.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  # 420s ceiling: coverage-instrumented runs put the crash matrix past the default 120s (CI-only timeouts on
  # otherwise-green runs; instrumentation costs 2-4x).
  alaya_add_test(
    NAME laser_test_segment_op_wal_crash
    TARGET test_segment_op_wal_crash
    LABELS laser wal persistence crash
    TIMEOUT 420 COVERAGE_RUN_SERIAL
  )

  # MutableLaserSegment: durable single-writer mutable segment handle (W3).
  alaya_cc_target(
    test_mutable_laser_segment
    BARE GTEST
    SRCS qg/test_mutable_laser_segment.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  alaya_add_test(
    NAME laser_test_mutable_laser_segment
    TARGET test_mutable_laser_segment
    LABELS laser wal disk persistence
    TIMEOUT 120
  )

  # Residency seams behind UnifiedLaserSegment: prefetch budget default, ResidencyProvider dispatch,
  # ensure_resident_arena file materialization, and the QGUpdater write_at -> arena mirror.
  alaya_cc_target(
    test_unified_residency
    BARE GTEST
    SRCS qg/test_unified_residency.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  alaya_add_test(
    NAME laser_test_unified_residency
    TARGET test_unified_residency
    LABELS laser
  )
endif()

# RowAdmission (segment admission contract v1): the bitmap POD view + its factories (bitmap payload wrap, sorted-rows
# materialization, exclude-set complement, AND composition). Pure kernel-level unit test -- no QuantizedGraph/disk index
# needed, so this target links nothing beyond gtest.
alaya_cc_target(
  test_row_admission
  BARE GTEST
  SRCS qg/test_row_admission.cpp
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_row_admission
  TARGET test_row_admission
  LABELS laser unit
)

# Segment admission contract kernel-level tests: tombstone parity between the legacy result_filter_ exclude-set path and
# the RowAdmission bitmap path (contract acceptance #2), bitmap-filter correctness, and the admission vs. exclude-set
# performance A/B (contract acceptance #4).
alaya_cc_target(
  test_admission_contract
  BARE GTEST
  SRCS qg/test_admission_contract.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
alaya_add_test(
  NAME laser_test_admission_contract
  TARGET test_admission_contract
  LABELS laser
)

# Regression test for a data-dependent heap overflow in QGBuilder's out-of-core build path (init_from_vamana() reading
# unclamped per-node degrees; see qg_builder.hpp and the test source for the full root-cause chain). Builds several tiny
# indices back-to-back in one process, which used to crash intermittently pre-fix.
alaya_cc_target(
  test_qg_builder_oom_regression
  BARE GTEST
  SRCS qg/test_qg_builder_oom_regression.cpp
  LIBS alaya_laser
  OPTS ${_laser_test_opts}
)
# 420s ceiling: 192-249s measured under ENABLE_COVERAGE instrumentation.
alaya_add_test(
  NAME laser_test_qg_builder_oom_regression
  TARGET test_qg_builder_oom_regression
  LABELS laser
  TIMEOUT 420
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

if(ALAYA_ENABLE_MUTABLE_LASER_TESTS)
  # SIFT-scale update experiment harness (build / insert / delete / eval modes). Not a ctest — run manually; see the
  # usage string in the source.
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

  # wal-2c: W2 canonical PID-reuse writer functional tests (activation, all-reuse, mixed, generation chain, no-reuse
  # byte-invariance). The R0-R11 SIGKILL matrix is test_segment_op_wal_reuse_crash.cpp (registered below in the same
  # section).
  alaya_cc_target(
    test_qg_updater_reuse
    BARE GTEST
    SRCS qg/test_qg_updater_reuse.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  alaya_add_test(
    NAME laser_test_qg_updater_reuse
    TARGET test_qg_updater_reuse
    LABELS laser wal persistence
  )

  # wal-2c: W2 step 7 MutableLaserSegment token-ization + tombstone ABA + base reuse.
  alaya_cc_target(
    test_mutable_laser_segment_reuse
    BARE GTEST
    SRCS qg/test_mutable_laser_segment_reuse.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  alaya_add_test(
    NAME laser_test_mutable_laser_segment_reuse
    TARGET test_mutable_laser_segment_reuse
    LABELS laser wal persistence
  )

  # wal-2c: W2 step 8 canonical PID-reuse R0-R11 SIGKILL/power-loss crash matrix.
  alaya_cc_target(
    test_segment_op_wal_reuse_crash
    BARE GTEST
    SRCS qg/test_segment_op_wal_reuse_crash.cpp
    LIBS alaya_laser
    OPTS ${_laser_test_opts} PCH_REUSE_FROM alaya_laser_qg_test_pch
  )
  # 420s ceiling: same coverage-lane margin as the crash matrix above (53s+ measured instrumented; slower runners were
  # within one variance step of the 120s default).
  alaya_add_test(
    NAME laser_test_segment_op_wal_reuse_crash
    TARGET test_segment_op_wal_reuse_crash
    LABELS laser wal persistence crash
    TIMEOUT 420
  )
endif()
# laser-ip: LASER inner-product (IP) distance kernel tests. Appended by branch feat/laser-ip-kernel; integrate as an
# append-only region on top of 2C's file. Both targets read the immutable official fixture ip_official_fixture.tsv,
# located via the ALAYA_IP_FIXTURE_DIR compile-def (source tree, so it is committable and read-only in CI). Core-space
# only -- no LASER-internal link is needed; the dormant include/index/graph/laser/space/ip.hpp is header-only over core
# space.

# W0 oracle base for the LASER IP kernel: pins ground truth from the immutable official RaBitQ-Library METRIC_IP fixture
# (generated offline by space/tools/gen_ip_official_fixture.cpp), the in-repo memqg RaBitQCore/RaBitQSpace path, and
# brute-force exact, plus the one-bit error bound and degenerate-input contract. The fixture is read-only here; a
# missing/bad fixture is a hard failure.
alaya_cc_target(
  laser_ip_oracle_test
  GTEST
  SRCS space/ip_oracle_test.cpp
  DEFS ALAYA_IP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/space"
  OPTS -O3 -DNDEBUG -Wall -Wextra
)
alaya_add_test(
  NAME laser_test_ip_oracle
  TARGET laser_ip_oracle_test
  LABELS laser rabitq space
)

# W1 unit test for the dormant LASER IP kernel (include/index/graph/laser/space/ ip.hpp): full-sign factor estimate
# equivalence, the factor_dq == official f_rescale/2 lock (incl. the 1-D counterexample) against the fixture, factor_vq
# mapping, the NaN-free zero-residual policy, and generic/AVX2/AVX-512 coverage.
alaya_cc_target(
  laser_ip_kernel_test
  GTEST
  SRCS space/ip_kernel_test.cpp
  DEFS ALAYA_IP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/space"
  OPTS -O3 -DNDEBUG -Wall -Wextra
)
alaya_add_test(
  NAME laser_test_ip_kernel
  TARGET laser_ip_kernel_test
  LABELS laser rabitq space
)
