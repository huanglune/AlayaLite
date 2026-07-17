# REPORT-allocator-merge.md

D-lite wave: allocator-merge completion (W1) + LASER recall floor lock (W2) + fallback-text parity (W3).
Worktree `feat/allocator-merge`, based on `wave1-integration@d14cae6`. Manifest:
`/home/huangliang/.claude/jobs/e6b5e510/tmp/allocator-merge-manifest.md`.

## Commits

| W  | Hash      | Summary |
|----|-----------|---------|
| W1 | `b88b60f` | Merge `laser::memory::AlignedAllocator`/`Allocator<T>` into `alaya::AlignedAlloc<T>`; shrink `laser/utils/memory.hpp` to a 41-line `align_allocate`/`align_free` shim; fix a latent `ceil_log2` transitive-include gap the shrink exposed. |
| W2 | `dd2a678` | Add `tests/collection/collection_laser_recall_floor_test.cpp`: l2 unit/nonunit + resident_arena recall-floor lock, each asserting engine identity against the persisted sealed-segment manifest. |
| W3 | *(this commit)* | `collection.hpp::resolve_build_algorithm()` gets laser-specific fallback_reason branches (row count / metric / dim), matching qg's granularity. |

## Acceptance checklist

| # | Item | Status |
|---|------|--------|
| 1 | Full-tree `laser::memory::AlignedAllocator` call sites zeroed (or thin delegation, exceptions audited) | **PASS** -- 0 remaining (`grep -rn 'AlignedAllocator\|laser::memory::Allocator\b\|memory::Allocator<' include/ tests/ benchmarks/` -> no matches). `align_allocate`/`align_free` kept as documented exception (see W1 table below). |
| 2 | Full laser/disk/collection ctest green | **PASS** -- 83/83 (full suite, not just laser-labeled subset), confirmed after each of W1/W2/W3, plus 7 extra ad-hoc runs of the new recall-floor test and 3 extra ctest-driven runs for flake-checking. |
| 3 | `collection_laser_recall_floor_test` 3 tiers green incl. engine identity | **PASS** -- l2_unit, l2_nonunit, resident_arena all green; each calls `expect_laser_manifest()` (on-disk `ArtifactManifestV2` check, not the live `target_implementation_key()` accessor -- see judgment call #6). |
| 4 | Fallback text parity + REPORT complete | **PASS** -- 3 laser branches added (row count/metric/dim); this file. |

## W1 -- allocator merge: call-site inventory

Full-tree grep (`include/ tests/ benchmarks/`, `src/` does not exist in this repo) was authoritative per manifest; the
known list in the manifest undercounted -- it missed `laser::memory::Allocator<T>` (unaligned, unused default template
arg) and `align_allocate`/`align_free` (raw fixed-alignment allocation, a distinct construct from the STL-allocator
`AlignedAllocator<T,Alignment,HugePage>` class the manifest named).

### Replaced -> `::alaya::AlignedAlloc<T>` (14 new call sites this wave, +2 pre-existing from bbc1d51)

| File:line | Field | Original | New behavior | Audit conclusion |
|---|---|---|---|---|
| qg_query.hpp:39,91 | `lut_` (+accessor) | `AlignedAllocator<uint8_t,64>` | `AlignedAlloc<uint8_t>` (64B/2MB+THP by size) | **Behaviorally identical, not just compatible.** `lut_` size = `padded_dim*4` bytes; `padded_dim` is FHT-rotated dim, hard-capped at 2048 by `select_fht_float()`'s `throw` beyond `log_b=11` (rotator.hpp). Max possible size = 8192B, always < AlignedAlloc's 16KB threshold -> always the 64B tier, exactly matching the old `Alignment=64` default. |
| qg_query.hpp:70 | `rd_query` (thread_local) | `AlignedAllocator<float>` (default 64B) | same | Same FHT cap applies (padded_dim<=2048 -> <=8192B). Identical. |
| qg_query.hpp:75 | `byte_query` (thread_local) | `AlignedAllocator<uint8_t,64>` | same | Same FHT cap (<=2048B). Identical. |
| rotator.hpp:119 | `FHTRotator::mat_` | `AlignedAllocator<float>` (default 64B) | same | `mat_` dims = `{1,padded_dim_}`, same FHT cap (<=8192B). Identical. |
| buffer.hpp:35,97 | `SearchBuffer::data_` | `AlignedAllocator<Candidate<float>>` (default 64B) | same | `Candidate<float>`=8B; size=`(capacity_+1)*8`. Not hard-capped (capacity_ is caller-controlled ef/beam width) -- large `capacity_` could cross 16KB and pick up 2MB+THP, which is a superset of 64B alignment (correctness-safe) and arguably beneficial for a hot search-beam buffer, same rationale as bbc1d51's cache_nodes_. Not a regression: AlignedAlloc's tiering is pre-existing, unchanged behavior (red line). |
| buffer.hpp:123,128 | `ResultBuffer::ids_` (+accessor) | `AlignedAllocator<PID>` (default 64B) | same | `PID`=uint32_t; same capacity-dependent analysis as above. Safe. |
| buffer.hpp:129 | `ResultBuffer::distances_` | `AlignedAllocator<float>` (default 64B) | same | Same as above. Safe. |
| hashset.hpp:34,81 | `HashBasedBooleanSet::table_` | `AlignedAllocator<PID>` (default 64B) | same | Table size derived from caller's `size` (bucket-sized power of 2); can legitimately be large for full-graph visited-tracking. Same "safe, possibly beneficial" analysis; this is exactly the cache/random-access-heavy pattern bbc1d51 optimized cache_nodes_ for. |
| qg.hpp:256 | `QuantizedGraph::data_` | `AlignedAllocator<float,1<<22,true>` (4MB+hugepage always) | `AlignedAlloc<float>` (2MB+THP once >=16KB) | **Dead field.** `grep -rn '\.data_\.\|->data_\b' include/index/graph/laser/` (excluding `thread_data_` false-positive matches) found zero references to this field anywhere else in the class or file. It is declared, default-constructed (empty `Array()`), and never resized/read/written. 4MB-vs-2MB alignment was therefore moot either way; replaced anyway for parity with every other call site rather than being the one holdout. Also confirms no code depends on the specific 4MB value (grepped `1 << 22`/`4194304`/`0x400000` tree-wide -- only hits are in `third_party/ffht/fht_avx.hpp`'s unrelated FFT butterfly-network constants, coincidentally the same numeric value for algorithmic reasons, not alignment). |
| array.hpp:31 | `data::Array`'s default `Alloc` template param | `memory::Allocator<T>` (unaligned, `::operator new`) | `::alaya::AlignedAlloc<T>` | `memory::Allocator<T>` was **dead code**: `grep` found it used nowhere as an explicit allocator, only as this unused default (both real instantiations of `data::Array`, in rotator.hpp and qg.hpp, always pass an explicit `AlignedAllocator<...>`). Changed the default to `AlignedAlloc<T>` rather than leaving a dangling reference to a deleted type; zero behavior change for any real caller. |

### Kept as documented exception: `align_allocate<Alignment>`/`align_free` (16 call sites, untouched)

`AlignedAlloc<T>` is a runtime-size-driven STL allocator with exactly two tiers (64B small / 2MB+THP large, threshold
16KB) and **no explicit-alignment knob**. `align_allocate<Alignment>(nbytes, huge_page)` is a different construct: a raw
allocation function with a **compile-time-fixed** `Alignment`, used for O_DIRECT scratch buffers that need
`kSectorLen`=4096B alignment regardless of size. Concretely, `qg_builder.hpp:316`'s
`align_allocate<kSectorLen>(qg_.page_size_)` can be called with `page_size_` as small as one sector (4096B, well under
AlignedAlloc's 16KB huge-page threshold) -- AlignedAlloc would only guarantee 64B there, which fails O_DIRECT's
alignment contract. This is exactly the manifest's foreseen "<16KB but needs >64B alignment" case.

Resolution: shrunk `laser/utils/memory.hpp` from 143 -> 41 lines, keeping only `align_allocate`/`align_free`, and
switched their internals to reuse the exact same shared kernel `alaya::AlignedAlloc<T>` itself calls
(`alaya_aligned_alloc_impl`/`alaya_aligned_free_impl` in `platform/detect.hpp`; `math::round_up_pow2` in `utils/math.hpp`,
replacing the file's own `round_up_to_multiple` from `laser/utils/tools.hpp`, which is now used nowhere in this file --
both round-up functions were equivalent for the power-of-2 alignments actually used here, `kSectorLen=4096`,
`kPageSize=4096` in the one test that calls this template directly). Net effect: the tree has one malloc primitive and
one round-up-to-power-of-2 primitive, not two; `align_allocate`'s policy (fixed alignment, zero-fill, optional madvise)
stays distinct from `AlignedAlloc`'s policy (size-tiered, no zero-fill) because they serve genuinely different
contracts, not because of duplicated logic.

Call sites (all zero-diff -- same function names/signatures, only the implementation moved):
`qg_builder.hpp` (6: 3x `align_allocate`, 3x `align_free`), `qg.hpp` (3: 2x `align_allocate`, 1x `align_free`),
`tests/laser/utils/test_threadpool_file_reader.cpp` (7: 1x `align_allocate`, 6x `align_free`).

### `laser/utils/memory.hpp` final state

143 -> 41 lines (71% reduction). Not strictly <=30 as the manifest's stretch target suggested; the remaining ~10-line
gap is the file-level doc comment explaining the one retained exception (kept because the manifest itself asked for
this reasoning to be documented, and because a future reader hitting a "why does this file still exist" question
deserves the answer inline, not just in this REPORT) plus the license header boilerplate every file in this repo
carries. No further logic to cut without deleting that explanation.

## W1 -- build/test verification

- `cmake --preset release -DBUILD_PYTHON=OFF` (fresh configure, `ALAYA_ENABLE_LASER=ON` on this Linux x86_64 host) +
  `cmake --build --preset release -j 32`: clean after two fix-forward iterations (see judgment calls #1, #2 below).
  Zero new warnings from the 7 edited files beyond the pre-existing `-Walloc-size-larger-than=` GCC false-positive
  class (see judgment call #3).
- `ctest -j 16` (full suite): **82/82** immediately after W1 (before W2 added test #83). Re-run **83/83** after W2 and
  again after W3.
- No CMake-registered pure-C++ laser benchmark target exists in this repo (`benchmarks/CMakeLists.txt` only
  `add_subdirectory(simd)`/`add_subdirectory(rabitq)`; the only laser benchmark, `benchmarks/laser/disk_laser_smoke.py`,
  shells out to `python -m alayalite.bench.disk_collection`, requiring `BUILD_PYTHON=ON` -- contradicting this wave's
  mandated `-DBUILD_PYTHON=OFF`). Substituted: the 82/83-test ctest suite already builds+searches+reopens+rotates real
  LASER indices end-to-end (`collection_laser_target_test`, `laser_segment_test`, `segmented_collection_laser_filter_test`,
  `test_unified_laser_admission`, etc.) far more thoroughly than a benchmark smoke run would; `rabitq_benchmark`
  (memqg-focused, shares the `AlignedAlloc<T>` core) confirmed built clean as bonus evidence. Manifest explicitly
  authorized skipping the numeric comparison ("不要求数字对比"); judged the ctest suite as satisfying the intent.

## W2 -- LASER recall floor lock: measured values

7 back-to-back runs of `collection_laser_recall_floor_test` (direct binary invocation) + 3 further runs via `ctest -R`
(flake-check), all green:

| Tier | dim | rows | Measured recall@10 | Floor set | Margin vs. worst observed |
|---|---|---|---|---|---|
| l2_unit | 128 | 400 | **1.0000** every run (7/7) | 0.85 | 15pp |
| l2_nonunit | 128 | 400 | **0.9700-0.9800** (paged-pool scheduling noise across runs) | 0.85 | 12pp (vs. 0.9700 worst) |
| resident_arena | 128 | 400 (l2_unit dataset, reused) | **1.0000** every run (7/7); paged_reference **1.0000** every run | 0.85 (== l2_unit) | 15pp |

Engine identity: every tier's `expect_laser_manifest()` call passed every run (`factory_key=="laser"`,
`reader_compatibility.required_features=={"disk_laser_segment"}`). resident_arena additionally confirmed the native
per-segment manifest's `x_extras["x_laser_residency"]=="resident_arena"` on every run.

## Judgment calls (numbered)

1. **W1 build failure #1** (transitive-include break): `rotator.hpp` and `qg.hpp` called `tools.hpp`'s unqualified
   `ceil_log2()` without including `tools.hpp` directly -- they relied on getting it transitively via
   `laser/utils/memory.hpp -> tools.hpp`, a path that only existed because the old (pre-shrink) `memory.hpp` included
   `tools.hpp` for its own `round_up_to_multiple`. Shrinking `memory.hpp` (and switching its own round-up call to
   `math::round_up_pow2`) severed that accidental chain. Fixed by adding `#include "index/graph/laser/utils/tools.hpp"`
   directly to both files (the principled fix: make the real dependency explicit) rather than keeping a spurious
   `tools.hpp` include in the shrunk `memory.hpp` just to preserve someone else's implicit transitive path. Also
   restored `rotator.hpp`'s include-block alphabetical order, which my first-pass include-swap had disturbed.
2. **W2 build failure**: `ASSERT_NE(...)` inside `measure_laser_recall()` (returns `double`) does not compile --
   `ASSERT_*` macros expand to a bare `return;` on failure, illegal in a non-`void` function. Replaced with a manual
   `if (...) { ADD_FAILURE() << ...; } else { EXPECT_EQ(...); }`, matching the guard style already used earlier in the
   same function for `created.ok()`/`sealed.ok()`/`response.ok()`.
3. **Pre-existing `-Walloc-size-larger-than=` warning family**: GCC emits a false-positive
   "argument 2 value 9223372036854775808 exceeds maximum object size" warning wherever `std::vector<T, AlignedAlloc<T>>`
   goes through `_M_default_append` (its static bound-checker can't prove the doubling growth stays under `PTRDIFF_MAX`).
   Confirmed pre-existing for `T=char` (bbc1d51's `cache_nodes_`, unchanged by this wave) by elimination -- none of my 7
   edited files touch a `char`-typed allocator. W1 additionally surfaces it for `T=unsigned char` (the new
   `qg_query.hpp` `lut_`/`byte_query` migration) since `AlignedAlloc<T>` is now instantiated for more `T`s; same root
   cause, not a new problem class. Out of scope to silence (red line: "不改 AlignedAlloc 本身的行为").
4. **`laser/utils/memory.hpp` line count**: landed at 41 lines, not <=30. See "final state" note above -- judged that
   keeping the doc-comment explaining the one retained exception (which the manifest itself asked to have documented)
   outweighs shaving ~10 more lines. Recorded per manifest's "发现它需要改=停下写 REPORT"-style instruction to flag
   deviations rather than silently either violate the target or silently drop the explanation.
5. **No pure-C++ laser benchmark target** (see W1 verification section above) -- substituted the ctest suite,
   authorized by the manifest's own "不要求数字对比" clause.
6. **Engine-identity assertion mechanism** (W2): chose to read back the persisted `ArtifactManifestV2` on disk
   (`expect_laser_manifest()`, copied from `collection_laser_target_test.cpp`'s precedent) rather than
   `Collection::target_implementation_key()`, despite the latter's name being the more obvious grep hit for
   "implementation_key" (the manifest's literal wording). Traced `target_implementation_key()` (collection.hpp:650) and
   found it resolves `options_.target_algorithm` -- the *configured* target -- through a static registration table; it
   does not inspect what actually got built, so it would still report `"disk_laser_segment"` under a silent
   flat-fallback. This is exactly the "silent fallback flat false green" failure mode the manifest warned about, just
   one level removed (in the *assertion* itself, not merely in tier selection). Documented at length in the test file's
   header comment per the manifest's "写清注释供后续仿照" instruction, since this is genuinely non-obvious from the
   field name alone.
7. **W3 branch scope**: added exactly 3 laser-specific `fallback_reason` branches (row count, metric, dim), matching
   the manifest's explicit enumeration ("行数下限/metric/维度各一句"). `laser_target_support()`
   (`collection_target_builder.hpp`, read-only reference) also gates on `scalar_type==float32`, same as qg -- qg's own
   fallback branches special-case that condition (its second branch) but the manifest did not list scalar_type for
   laser, so a laser scalar_type mismatch falls through to the existing generic catch-all message (which still names
   the factory_key and is not wrong, just less specific). Followed the manifest's explicit scope rather than
   over-extending by analogy to qg.
8. **Recall floor value (0.85) uniform across all 3 W2 tiers**: rather than tuning three different floors tightly to
   each tier's measured ceiling, used one conservative 0.85 for all three (12-15pp margin below each tier's own
   worst-of-7-runs observation). Simpler to maintain, still well within the manifest's "measured-10~15pp" instruction
   for every tier, and (coincidentally) matches `collection_qg_recall_floor_test.cpp`'s own l2_unit floor value.
9. **`AlignedAlloc<T>`'s tier-boundary size dependence for `buffer.hpp`/`hashset.hpp` call sites**: these buffers'
   sizes depend on caller-supplied capacity/ef/table-size and are not hard-capped the way `qg_query.hpp`/`rotator.hpp`'s
   FHT-bound buffers are, so an unusually large capacity could cross the 16KB threshold and trigger 2MB-granularity
   rounding + THP where the old code never did (old code always used a fixed 64B alignment regardless of size).
   Judged this is (a) correctness-safe (2MB alignment is a superset of a 64B requirement), (b) plausibly beneficial
   for hot random-access structures (same rationale bbc1d51 used for `cache_nodes_`), and (c) not a change I'm
   authorized to alter regardless, since it's `AlignedAlloc`'s own pre-existing, size-driven behavior (red line:
   "不改 AlignedAlloc 本身的行为"). Recorded here rather than treated as a blocker.

## Red-line compliance

Not touched: `mutable_laser_segment.hpp`, `segment_op_wal.hpp`, `qg_updater.hpp`, `include/index/collection/**` (except
the `collection.hpp` `resolve_build_algorithm()` fallback-text lines, W3's explicit exception), `include/space/**`,
`include/simd/**`, `include/storage/**`. `qg.hpp` edits confined to the `data_` field's allocator template parameter
(line 256); no superblock/WAL/arena logic touched. `AlignedAlloc`'s own behavior (thresholds/alignment tiers) left
unmodified throughout -- every finding above that touches on its size-tiering behavior is an audit note about
*inherited* behavior at newly-migrated call sites, not a change to the allocator itself.

No `git stash` used. No pushes. No PR opened. No `Co-Authored-By` trailers.
