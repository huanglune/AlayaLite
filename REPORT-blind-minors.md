# Wave 3 blind-review MINOR fixes report

## Conclusion

All three reported MINOR findings were confirmed and fixed within the assigned file
territory. Each finding has an independent commit. The final build used both
`BUILD_PYTHON=ON` and `BUILD_TESTING=ON`; all 94 registered tests passed, and the two
frozen `SEGMENT_OP` wire-format golden tests passed byte-for-byte.

## M1: AVX-512-VL dispatch predicate

Commit: `5392130 fix(simd): gate RaBitQ AVX-512 on full target features`

- Added explicit AVX512DQ, AVX512VL, and OS opmask/ZMM-state capabilities to
  `CpuFeatures`.
- GNU/Clang detection continues to use `__builtin_cpu_supports`; these builtins already
  incorporate OS extended-state enablement, so a supported AVX512F result records the
  OS-state capability. The MSVC path now checks CPUID OSXSAVE and requires
  `(XCR0 & 0xE6) == 0xE6` via `_xgetbv(0)`.
- RaBitQ's shared AVX-512 tier now requires F + BW + DQ + VL + OS state, exactly covering
  the strongest `ALAYA_TARGET_AVX512_VL` target used by that tier.
- Added a narrow synthetic-feature selector test. F+BW without DQ/VL falls back to AVX2,
  and a complete ISA set without OS state also falls back to AVX2.

Evidence:

- `simd_test_cpu_features`: passed.
- `space_test_rabitq_dispatch`: passed, including
  `RabitqDispatchTest.Avx512RequiresFullVlTargetFeatureSetAndOsState` and the existing
  full differential dispatch checks.

## M2: `auto_seal_rows` capacity wraparound

Commit: `1284b83 fix(collection): validate auto-seal capacity arithmetic`

- Replaced the unchecked `size_t(auto_seal_rows) * 2 + 4096` computation with checked
  conversion, multiplication, addition, and a final `capacity > threshold` invariant.
- Invalid values return `StatusCode::invalid_argument` with
  `StatusDetail::arithmetic_overflow` during create validation.
- Persisted `auto_seal_rows` is revalidated immediately after loading the control state,
  before control-state normalization, segment open, or WAL recovery.
- Boundary tests cover `max_safe` through create and reopen, plus `max_safe + 1` and
  `UINT64_MAX` rejection through both create and persisted reopen paths. Reopen tests
  compare the entire logical WAL before and after rejection; the bytes remain identical,
  proving the error precedes any WAL COMMIT or other WAL write.

Evidence:

- `collection_facade_test`: passed, including
  `AutoSealRowsCapacityArithmeticAcceptsMaximumSafeValue`,
  `AutoSealRowsCapacityOverflowIsRejectedBeforeWalCreation`, and
  `PersistedAutoSealRowsOverflowIsRejectedBeforeWalRecovery`.

Judgment call: the review's additional suggestion to trigger rotation from physical
capacity before logical COMMIT was intentionally not implemented. It changes rotation/seal
policy and is outside this task's validation/checked-arithmetic boundary; the validated
configuration invariant closes the reported wraparound path without changing that design.

## M3: unified WAL vocabulary drift

Commit: `18608f2 docs(wal): align vocabulary with shipped frame codec`

- Marked record type 5 (`SEGMENT_OP`) as shipped with 2C.
- Replaced references to nonexistent detail namespaces with the delivered shared
  primitives in `include/wal/frame.hpp`, namespace `alaya::wal`, including `Decoder`.
- Updated the `SegmentOp::pid_generation` field comment: generation is zero without reuse,
  while reuse records carry the current non-zero generation.
- Only documentation/comments changed. Codec code, wire constants, and golden bytes were
  not modified.

Evidence:

- `SegmentOpCodec.Kind1Through6GoldenBytesAreFrozen`: passed.
- `SegmentOpCodec.Kind7And8GoldenBytesAreFrozen`: passed.
- Golden result: 2/2 passed, `golden_exit=0`.

## Full validation and scope audit

Configuration:

```text
BUILD_PYTHON=ON
BUILD_TESTING=ON
Python 3.13.6 (.venv/bin/python)
GNU C++ 11.4.0
```

Results:

- Full build: 242/242 Ninja steps completed.
- Full CTest: 94/94 passed, 0 failed, `ctest_exit=0`.
- The known `laser_test_threadpool_file_reader` and active-LASER teardown flakes did not
  occur; no retry was needed.
- `git diff --check` passed.
- Full pre-commit was not run, per the handoff instruction that the main integration
  session runs it once.

The final change list is confined to the assigned SIMD/RaBitQ, Collection validation,
WAL contract/comment, and corresponding test territories. No prohibited file, SIMD kernel,
WAL codec/wire format, golden byte expectation, rotation/seal design, or CI file changed.
