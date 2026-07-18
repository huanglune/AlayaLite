# U4-preflight delivery report

Branch `feat/u4-preflight`, based on `main@c231b7d`. Two independent deliverables per the
execution manifest (`/home/huangliang/.claude/jobs/e6b5e510/tmp/u4-manifest.md`, including its
"修正案 v2" amendment): (B) runtime SIMD dispatch for the memory RaBitQ fastscan/rotator/lut hot
path, and (A) an inner-product norm-parity audit of `RaBitQCore::memory_factors`. Not pushed, main
untouched, no PR opened.

## Commits

```
405db8f feat(space): runtime-dispatch rabitq fastscan/rotator/lut SIMD tiers
55f8fc8 perf(space): add rabitq dispatch A/B timing harness
7d7fe50 test(space,collection): lock rabitq IP branch formula, characterize non-unit-norm recall
f338bb3 test(collection): metric-aware exact GT + HNSW/QG recall parity baseline
```

(a 5th `docs` commit adding this report follows). No commit carries a `Co-Authored-By` trailer.
Each commit was built (`cmake --preset release -DBUILD_PYTHON=OFF && cmake --build build/Release
-j 64`) and tested (`ctest --test-dir build/Release -LE performance -j8`) green before the next
one was made.

## Change list

| Commit | Files |
|---|---|
| ① dispatch | `include/space/quant/rabitq/dispatch.hpp` (new), `fastscan.hpp`, `rotator.hpp`, `lut.hpp`, `tests/space/rabitq_dispatch_test.cpp` (new), `tests/space/rabitq_utils/lut_test.cpp`, `tests/space/CMakeLists.txt` |
| ② perf harness | `benchmarks/rabitq/rabitq_dispatch_benchmark.cpp` (new), `benchmarks/rabitq/CMakeLists.txt` |
| ③ IP audit tests | `include/space/quant/rabitq_core.hpp` (comment only), `tests/space/rabitq_space_test.cpp`, `tests/collection/collection_qg_ip_norm_test.cpp` (new), `tests/collection/CMakeLists.txt` |
| ④ parity infra | `tests/include/utils/evaluate.hpp`, `tests/collection/collection_hnsw_qg_parity_test.cpp` (new), `tests/collection/CMakeLists.txt` |

Red-line audit against `c231b7d`: `include/index/**`, `include/simd/laser_dispatch.hpp`,
`tests/laser/simd_dispatch_test.cpp`, all of `cmake/`, `pyproject.toml`, and
`tests/collection/collection_qg_seal_test.cpp` (the cosine-fallback lock) — zero diff against all
of them (verified with `git diff --stat c231b7d..HEAD -- <path>`, empty output). `include/simd/
fastscan.hpp` and `include/simd/cpu_features.hpp` were read-only-reused, never edited.

## Test numbers (final committed state)

- `ctest --test-dir build/Release -LE performance -j8`: **100% passed, 0 failed, 81/81**.
- `ctest -L performance -N`: 1 test (`index_test_rabitq_performance`), correctly excluded.
- `laser_segment_test`'s `LaserSegment.DifferentialRankOnlyManifestGateCollectionRejectionAndPerformance`
  confirmed **SKIPPED** (not failed) under `BUILD_PYTHON=OFF`, as the manifest predicted (LASER
  fixture unavailable) — does not count against the green gate.
- `uvx pre-commit run --all-files` (proxied): **all 18 hooks passed** on the first run, zero files
  auto-modified (ruff format/check, pylint, yamlfmt, trailing-whitespace, end-of-file-fixer,
  check-added-large-files, check-merge-conflict, check-yaml, detect-secrets, typos, cmake-format,
  cmake-lint, clang-format, cpplint, no-chinese-characters, layer-boundaries, reuse). No
  `.secrets.baseline` refresh was needed.
- Full clean rebuild (`cmake --build build/Release -j 64`) on the final committed tree: 0 errors
  (only pre-existing, unrelated warnings in `include/index/graph/laser/qg/qg.hpp` and
  `tests/laser/rabitq_factor_equivalence_test.cpp`, both outside my red-lined/touched files).

New test counts: `space_test_rabitq_dispatch` (7 cases, dispatch differential + factory
selection), `RaBitQCoreTest`/`RaBitQSpaceIpNormTest` inside `rabitq_space_test` (2 new cases),
`collection_qg_ip_norm_test` (1 case), `collection_hnsw_qg_parity_test` (4 parametrized cases).

## Dispatch A/B numbers (件 B, item 6)

This machine is AVX-512-capable, but the repo's release preset compiles with an `-mavx2 -mfma`
baseline (`X86_AVX2_BASE=ON`) — confirmed by reading the configure output. That means **every one
of the six `#if defined(__AVX512F__)` branches was dead code in a normal release build before this
change**, regardless of host CPU: the macro was never defined for this translation unit. After the
runtime-dispatch conversion, the AVX-512 code is compiled in unconditionally (via function-level
`ALAYA_TARGET_*` attributes) and actually selected and run on this host at process start.

Standalone op-latency harness (`benchmarks/rabitq/rabitq_dispatch_benchmark`, build-only, not
registered with ctest), 3 repeated runs, stable to within ~1%:

| kernel | generic (ns/op) | dispatched (ns/op) | speedup |
|---|---:|---:|---:|
| `accumulate` (dim=128) | 419-436 | 15.0 | **~28x** |
| `estimate_distances` (standalone, 32-wide) | 3.2 | 3.7 | 0.86-0.87x |
| `accumulate_and_estimate_distances` (dim=128) | 421-463 | 18.0 | **~23-26x** |
| `flip_sign` (dim=256) | 109.1-109.4 | 9.6-9.7 | ~11.3x |
| `kacs_walk` (len=256) | 13.7 | 9.1 | ~1.5x |
| `scalar_quantize_optimized` (len=512) | 62.1-62.5 | 22.2-22.8 | ~2.8x |

`accumulate_and_estimate_distances` is the one that matters in production: it is the fused kernel
memqg calls per candidate node (`rabitq_space.hpp` `QueryComputer::batch_est_dist`, wired through
`accumulate`/`estimate_distances` as its scalar/AVX2 fallback path). Its **~23-26x** speedup is the
headline number. `estimate_distances` alone is never called standalone in production (only inside
the fused kernel above) and is dominated by fixed dispatch/load overhead at this tiny (32-element,
128-byte) size -- not representative of its contribution inside the fused kernel, which does show a
strong speedup as a whole. The repository's background "17-24%" figure for this class of change
has no traceable source in this repo; the numbers above are freshly measured and reproducible
(`./build/Release/benchmarks/rabitq/rabitq_dispatch_benchmark`).

## IP landmine verdict: **falsified** (formula unchanged)

**Original working hypothesis** (manifest, pre-amendment): `RaBitQCore::memory_factors`'
inner-product branch base term carries a bare literal `1` (unlike the fully data-driven L2
branch), suspected of an implicit `||o||=1` assumption -- Collection does not L2-normalize for
`inner_product`, and every pre-existing rabitq/QG test only ever fed unit vectors, so this had
never actually been exercised.

**Verdict, per "修正案 v2" (a parallel codex math review, cross-checked by both the review and by
me independently): falsified.** Substituting the same one-factor estimator the (already-trusted)
L2 branch above it relies on through the algebra shows the branch computes **exactly**
`1 - <q,o>`, where the leading `1` is a constant that does not depend on the candidate `o` (or on
`q`) at all. Nearest-neighbor ranking only ever compares estimates against each other for one
fixed query, so a candidate-independent constant is invisible to the ranking -- minimizing
`1-<q,o>` is exactly equivalent to maximizing `<q,o>`, for **any** `||o||`, not just unit vectors.
This is *not* an implicit `||o||=1` assumption; it just visually resembles half the squared L2
distance between unit vectors (`1-<q,o> = ||q-o||^2/2` when `||q||=||o||=1`), which is exactly why
cosine can reuse this same branch.

**Evidence chain:**
1. **Upstream provenance**: `baselines/RaBitQ-Library/include/rabitqlib/quantization/
   rabitq_impl.hpp`'s `one_bit_code_with_factor()`, `METRIC_IP` branch, computes the
   byte-for-byte identical `f_add = 1 - dot_product(residual, centroid) + (l2_sqr * ip_cent_xucb /
   ip_resi_xucb)`. `K=1` is the upstream RaBitQ-Library convention, not something this port
   introduced.
2. **Exact algebraic lock** (`tests/space/rabitq_space_test.cpp`,
   `RaBitQCoreTest.InnerProductBranchLocksToOneMinusDot`): calling `memory_factors` directly with
   `q_rot = centroid` makes the K-estimator's own approximation error algebraically **exact rather
   than approximate** (no fastscan/LUT quantization involved -- this test bypasses that pipeline
   entirely), so the assembled estimate collapses to *exactly* `1 - <c,o>` to float precision, for
   deliberately non-unit-norm `data`/`centroid` (`||.||^2 ~ 341`, asserted far from 1 before
   running). **Passes.** This test would fail immediately if someone "fixed" the literal `1` into
   a norm term, or if the real bug had been a `||o||=1` assumption instead of a harmless constant
   -- the two are exactly distinguishable at this construction.
3. **Characterization + regression lock, space level**
   (`RaBitQSpaceIpNormTest.NonUnitNormRecallDoesNotCollapse`, single fastscan batch-of-32 estimate
   vs. exact `-dot` oracle, no graph re-ranking, averaged over 8 trials x 30 queries): unit-norm
   baseline recall@10 = **0.636**, non-unit-norm (per-point/per-query scale in [0.5, 3.0]) =
   **0.710** -- no decline.
4. **Characterization + regression lock, Collection level**
   (`collection_qg_ip_norm_test.cpp`, full QG seal+search pipeline): unit-norm recall@10 =
   **0.980**, non-unit-norm (hnsw_seal's `make_cosine_dataset` style, 0.25x-28.25x per-row scale)
   = **0.960** -- a 2-point decline, comfortably inside quantization-noise territory, not a
   collapse.
5. **HNSW/QG parity baseline** (`collection_hnsw_qg_parity_test.cpp`, 4 cases): `ip_unit`
   hnsw=1.00/qg=0.995, `ip_nonunit` hnsw=1.00/qg=0.965 -- QG tracks HNSW (unquantized, exact
   per-edge distances) closely in both inner_product cases, no norm-related cliff.

**Action taken**: the formula in `rabitq_core.hpp` is **unchanged, one line** -- only a derivation
comment was added to the inner-product branch, citing both the algebra above and the upstream
reference, explicitly warning against replacing the literal `1` with a data-driven norm term
(would target `||o||^2-<q,o>`, which is *not* order-preserving across differently-normed
candidates and would corrupt MIPS ranking -- worked example in the comment: candidate at true
`-dot=-10` vs. current `1-dot=-9`, same rank order; norm-"fixed" `||o||^2-dot` could flip to
`+90`, reversed order). `cosine`-for-QG wiring remains untouched (out of scope;
`collection_qg_seal_test.cpp:509-523`'s cosine-rejects-to-flat test still passes unmodified).

## Judgment calls outside the manifest checklist

Items the manifest didn't dictate explicitly, decided during execution -- listed with rationale so
the coordinator can override any of them:

1. **Two-tier kernels fall back to generic (not AVX2) on AVX2-only hosts.** `estimate_distances`,
   `accumulate_and_estimate_distances`, `flip_sign`, `kacs_walk`, `scalar_quantize_optimized`
   never had a real AVX2 branch pre-refactor (only AVX-512 vs. portable `#else`). Added a
   `select_rabitq_simd_avx512_or_generic` helper (distinct from the 3-tier `select_rabitq_simd`
   used only by `accumulate`) that treats AVX2-only hosts as generic for these five, matching what
   they actually did before (scope control: "不给现状没有 AVX2 档的函数补 AVX2").
2. **`accumulate_and_estimate_distances`'s generic tier composes the *live-dispatched*
   `get_accumulate_func()`/`get_estimate_distances_func()`**, not the pure-generic primitives --
   this exactly reproduces the pre-refactor behavior where its `#else` branch called `accumulate()`
   /`estimate_distances()` (which, pre-refactor, picked up whatever tier the *TU's own* baseline
   flags provided). Consequence: on an AVX2-only host, this fused kernel's "generic" tier actually
   runs AVX2 `accumulate` internally -- a deliberately preserved quirk, not a new one.
3. **Target-attribute precision found by compiling, not guessed.** `flip_sign_avx512` needed
   `ALAYA_TARGET_AVX512` (the DQ-inclusive macro, for `_mm512_mask_xor_ps`) rather than `_BW`
   (matches `distance_ip.ipp`'s own `ip_sqr_avx512` precedent of using DQ while gating dispatch on
   just `avx512f_`/`avx512bw_`). `scalar_quantize_optimized_avx512` needed an explicit
   `avx512vl` addition beyond *both* named macros (`_mm_storeu_epi8`) -- the compiler's own
   "inlining failed... target specific option mismatch" error caught this, not manual ISA-table
   lookup.
4. **Consolidated per-function fallback logging into one dispatch-resolution log.** The old
   per-call `log_scalar_fastscan_fallback()`/`log_rotator_fallback_once()` calls inside the
   CPU-dispatch branches were replaced by a single `LOG_INFO("rabitq_simd={}", ...)` in
   `get_rabitq_simd_level()`, logged once at first resolution -- mirrors LASER's own
   `get_laser_simd_level()` pattern exactly, and avoids a circular include (dispatch.hpp would
   otherwise need to depend back on the files that include it). The *unrelated* dim-misalignment
   fallback logging inside `accumulate()`/`accumulate_and_estimate_distances()` (nothing to do
   with CPU tier) was left untouched.
5. **`scalar_quantize_optimized`'s non-float instantiations always take the safe generic path**
   (`if constexpr`), rather than reproducing the pre-refactor quirk where a `T!=float`
   instantiation would (if the TU happened to have `__AVX512F__` defined) silently narrow into the
   AVX-512 float-only intrinsics. Verified via grep that `Lut<T>`/`scalar_quantize_optimized<T>` is
   never actually instantiated with `T!=float` anywhere in the codebase today, so this is inert in
   practice but strictly safer.
6. **`dispatch.hpp`'s generic `scalar_quantize_optimized` duplicates `scalar_quantize_normal`'s
   body** (same Eigen expression) instead of calling it, because `lut.hpp` includes `dispatch.hpp`
   (not the reverse) -- calling back would be circular. Documented in a comment that the two bodies
   must stay numerically identical.
7. **Cross-referenced `baselines/RaBitQ-Library`'s own dispatch split**
   (`include/rabitqlib/simd/{fastscan,rotator}_dispatch.hpp`) for the rotator/lut function-boundary
   decisions, per the coordinator's suggestion. Their mechanism differs (separate translation units
   compiled with different `-m` flags per ISA, vs. this repo's single-TU target-attribute
   multi-versioning) -- kept this repo's LASER pattern as instructed -- but their function-level
   slicing (`flip_sign`/`kacs_walk` as independent dispatch points, matching what I'd already
   designed) cross-validated the boundary choice.
8. **`rabitq_dispatch_test.cpp`'s `AccumulateAndEstimateDistancesDifferentialFuzz` oracle is a
   manually-composed true-scalar reference** (`simd::fastscan::accumulate_generic` +
   `detail::estimate_distances_generic`, called directly, bypassing dispatch), not the "generic"
   detail function itself -- the latter internally calls the live-dispatched sub-functions
   (judgment call #2 above) and would trivially match the AVX-512 result on this AVX-512 host,
   making the differential test tautological.
9. **`lut_test.cpp`'s `FastScanFallbackAccumulatesAndEstimatesDistances`** first attempt directly
   called `detail::accumulate_and_estimate_distances_avx512` with `dim=8` for an AVX-512
   differential check -- this violates that kernel's `dim % 16 == 0` precondition (caught by the
   test itself, not by inspection: results were nonsense). Fixed by recognizing `dim=8` exercises
   only the dim-misalignment fallback (unrelated to CPU-tier dispatch, and unreachable by the fused
   AVX-512 kernel by construction); real AVX-512-tier differential coverage for this kernel at
   valid (16-aligned) dims lives in `rabitq_dispatch_test.cpp`'s fuzz test instead.
10. **IP recall-test thresholds were calibrated from measurement, not assumed.** First attempt at
    the space-level characterization test assumed an 0.85 unit-norm floor (guessing it should
    resemble production end-to-end recall); actual measurement was 0.636, because this test
    exercises a single fastscan batch-of-32 estimate with no graph-search-level re-ranking, a much
    harsher measurement than full QG recall. Thresholds were rewritten to match observed reality
    with a documented margin, and the test comment now explicitly states what is and isn't being
    measured, to prevent future misreading of the number as end-to-end recall.
11. **`q=c` discriminating test operates on `RaBitQCore::memory_factors` directly**, bypassing
    `RaBitQSpace`/rotation/the fastscan LUT entirely -- going through the full production pipeline
    would still carry LUT-quantization noise even at `q=c` (the `Lut<T>` constructor's own uint8
    quantization is a separate, unrelated approximation), making an *exact* assertion impossible.
12. **Collection-level IP-norm characterization is a new file**
    (`collection_qg_ip_norm_test.cpp`), not an addition to `collection_qg_seal_test.cpp` -- the red
    line explicitly grants "`tests/collection/` 新文件" (new files); modifying the existing file
    wasn't explicitly granted, and keeping the diff isolated avoided any risk of colliding with the
    parallel U2-c wave even though that file wasn't on its own red-line list.
13. **Parity test dimension bumped from an initial 32 to 64.** `RaBitQSpace`'s `FhtKacRotator`
    requires `floor_log2(dim)` in `[6, 11]`, i.e. `dim >= 64`; a first draft at `kDim=32` (matching
    `collection_hnsw_seal_test.cpp`'s own HNSW-only convention) failed QG's `seal()` with
    "dimension of vector is too big or too small". Bumped to 64 (matching
    `collection_qg_seal_test.cpp`'s own convention) since this test must satisfy both engines.
14. **`evaluate.hpp`'s new `metric` parameter sits between the existing required params and the
    existing defaulted `deleted` param** (`dim, topk, metric=l2, deleted=nullptr`) -- preserves
    100% source compatibility for every existing positional call site
    (`tests/utils/evaluate_test.cpp`, `dataset_utils.hpp`'s `random_dataset`), verified by
    rebuilding and rerunning both unmodified.
15. **A heredoc/Python-string-escaping bug** (unrelated to the pre-warned Write/Edit worktree
    pinning, a different flavor of tool pitfall) corrupted two `'\n'` char literals into literal
    newlines inside `rabitq_space_test.cpp` when first written via a Python triple-quoted string
    (Python's own escape processing consumed the `\n` meant for the C++ source). Caught immediately
    by the compiler, fixed, and all other touched files were grepped for the same pattern (none
    found).

## Notes

- `baselines/RaBitQ-Library` (official RaBitQ reference implementation, read-only) is now cloned
  at `/home/huangliang/workspace/alaya-dev/baselines/RaBitQ-Library`, used above for the upstream
  IP-formula citation and the dispatch cross-reference. Untouched by this wave; noted here since
  it's a new resource for future n-bit RaBitQ work.
- Tool pitfall confirmed as predicted: this session's Write/Edit tools were pinned to the launch
  worktree (`laser-fullcache`), not this one. All file creation/edits in this worktree went
  through Bash (heredoc for new files, Python/perl exact-match replacement for edits), with
  `git diff`/rebuilds after every change.
