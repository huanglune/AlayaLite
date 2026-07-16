<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Segment admission contract: one predicate seam for user filters and tombstone visibility

Status: **contract** (implementation lands with the U2 lifecycle wave).
Extends `filter-seal-gc.md` (Gate 10 planner) and the `SegmentFilterView`
ABI in `core/value_types.hpp`. Evolves the LASER kernel's `result_filter_`
seam (`index/graph/laser/qg/qg.hpp`).

## Why now

Three facts, all already in the tree:

1. The Gate 10 planner routes mid-selectivity requests
   (`0.15 < s <= 0.60`) through **traversal filtering** via
   `SegmentFilterView` — but the LASER segment searchers do not execute
   it. `DiskSearchOptions` carries no filter and the legacy
   `LaserSegmentSearcher` ignores the view, so LASER segments can only be
   served by pre/postfilter today.
2. The traversal kernel already performs an admission test per candidate:
   `result_filter_` (a `const std::unordered_set<PID> *` exclude set),
   checked at the two result-admit points of both search paths. Its only
   producer is `QGUpdater` tombstone visibility.
3. U2 adds user filters to the same kernel. Two parallel admission
   mechanisms (exclude set for tombstones, something else for filters)
   would put two tests and two data shapes in the hottest loop.

The contract: **one admission test, two sources, composed once per query.**

## Contract

### 1. Admission point

Admission is a single per-candidate predicate evaluated at the existing
result-admit points of the traversal kernel (both the paged/pool path and
the resident-arena path — the same two sites `result_filter_` is checked
at today). Neighbor *expansion* is not gated: filtered-out rows still
route the walk; only result admission is filtered. Changing that (true
filtered traversal) is an explicit non-goal of v1.

### 2. RowAdmission v1 = include-semantics bitmap

```
RowAdmission = word-aligned bitmap over the segment's PID capacity
               bit set   -> admissible
               absent    -> all rows admissible (nullptr fast path)
               popcount cached at build time (density for the planner)
```

One load + one bit test per candidate. This replaces the per-candidate
hash probe of the exclude set (Milvus/Knowhere-convention bitset; the
hash probe is measurably hostile at pop rates of the arena kernel).

### 3. Composition (the two sources)

```
admission = user_bitmap AND live_bitmap        (composed at query setup)
```

- **System source (visibility):** `QGUpdater` maintains a live bitmap
  (bit cleared on tombstone). Replaces the `result_filter_` exclude set;
  a thin adapter keeps the exclude-set constructor during migration.
- **User source:** compiled from the request's `SegmentFilterView`:
  - `bitmap` — wrap the payload directly (zero copy when word-aligned);
  - `sorted_rows` — materialize into a bitmap at setup;
  - `predicate` / `composite` — the Collection pre-compiles to a bitmap
    against its logical registry (v1). Segment-local predicate execution
    is out of scope until segments own scalar storage.

The AND is O(capacity/64) at setup, never per candidate. If only one
source is active it is used directly (no copy).

### 4. Concurrency and staleness

Per-query admission has snapshot semantics inherited from the updater's
phase-separation contract: tombstone/consolidate mutation phases do not
run concurrently with searches on the same segment view. The bitmap
itself therefore needs no seqlock. G1 (durable op-WAL) does not change
this contract; if a future wave interleaves mutation and search phases,
it must revisit this section explicitly.

### 5. API surface

- `DiskSearchOptions` gains `SegmentFilterView filter{}` (default
  `kind=none` → behavior byte-identical to today).
- The unified segment searcher compiles the view + live bitmap into a
  `RowAdmission` and hands the kernel one pointer.
- `ResultFlag::filtered` is set on hits produced under an active
  admission, so Gate 10 accounting keeps working unchanged.

### 6. Planner interaction

Gate 10 thresholds are unchanged. This contract only makes the mid-band
(traversal filtering) executable on LASER segments — for both residency
modes, from the same kernel seam. `RowAdmission.popcount` gives the
planner the observed density when it wants it.

## Non-goals (v1)

- Filter-aware traversal (adaptive ef growth inside the kernel,
  filtered-greedy expansion): deferred until selectivity data from real
  workloads exists. The seam does not preclude it.
- Segment-local predicate/composite evaluation.
- Any change to recall or scoring semantics.

## Acceptance (checked when U2 lands the implementation)

1. `kind=none` end-to-end is byte-identical to the pre-contract searcher
   on both residency modes.
2. Tombstone parity: updater visibility via live bitmap produces the same
   result set as the legacy exclude set on the updater test corpus.
3. A bitmap-filter recall test on the unified segment, both residency
   modes, exercised through the Gate 10 planner's mid-band route.
4. Admission overhead ≤ the current exclude-set check at equal live
   ratios (section-profile evidence, `ALAYA_KERNEL_SECTION_PROFILE`).
