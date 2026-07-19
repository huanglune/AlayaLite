<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Gate 10 filter execution, successor seal, compact, and GC

> **Historical Gate 10 baseline (updated 2026-07-19).** The control-plane
> contracts remain relevant, but the Flat-only target descriptions predate
> the `qg` to LASER implementation swap. They record the Gate 10 rollout
> state rather than the current target inventory.

Gate 10 completes Collection-owned filter execution and resource accounting,
then adds successor-first rotation, immutable Flat targets, epoch-delayed
reclamation, and Flat-to-Flat compaction. DiskANN and LASER graph sources, all
Segment engine bodies, the legacy wrappers, and the existing core statistics
structures remain unchanged.

## Public and internal contract

Users continue to choose only `FilterPolicy::automatic`, `strict`, or
`allow_partial`. `FilterExecution::{prefilter,traversal,postfilter}` records the
Collection-internal plan and is not another user policy. The only Gate-10
change under `include/core` is that enum in `core/value_types.hpp`.

The C++ `alaya::Collection` surface adds `seal()`, `compact()`, and `gc()`.
`search()` and `batch_search()` accept a `CollectionFilter` and return
`CollectionSearchStatistics`. Python exposes the same operations as
`Collection.seal()`, `compact()`, and `gc()`; canonical search accepts
`metadata_filter`, `filter_policy`, an optional selectivity estimate, and
scratch/I/O budgets. `auto_seal_rows=0` disables automatic rotation; a positive
value rotates after the active generation reaches the configured physical-row
threshold.

Both Collection statistics structures use a versioned struct header:

| owner | incremental fields |
|---|---|
| `CollectionSearchStats` / Python `search_stats` | `filter_execution`, `filter_active`, `filter_examined`, `filter_passed`, `nan_discarded`, `overfetch_rounds`, `budget_consumed`, `lease_acquired`, `lease_released`, `lease_peak_bytes`, `io_requests_consumed`, `io_bytes_consumed` |
| `CollectionStatistics` / Python `stats()` | `sealed_segments_count`, `gc_pending_count`, `active_segment_algorithm`, `compacted_bytes` |

`core::SearchStats` and `core::SegmentStats` are deliberately unchanged.

## Filter plan and accounting

The auto planner uses the caller's selectivity estimate when present. Otherwise
it samples at most 256 visible logical rows. The stable thresholds are:

| estimated selectivity | execution |
|---:|---|
| `0.00 .. 0.15` | prefilter with exact Collection-owned vectors |
| `(0.15 .. 0.60]` | traversal filtering through `SegmentFilterView`, with Collection-side verification |
| `(0.60 .. 1.00]` | postfilter after score normalization and version suppression |

Strict always selects exact prefiltering. Every passing row is scored from its
Collection-owned vector, so strict needs no overfetch. If any live row lacks
that exact fallback vector, the whole request returns
`resource_exhausted/budget_denied` with no partial response.

Auto and allow-partial fanout start with `top_k` candidates per segment. If a
non-exhaustive round produces fewer than `top_k` eligible hits, the limit
doubles and the Collection re-queries. `maximum_overfetch_rounds` defaults to
four and `overfetch_rounds` counts only re-queries, not the initial round.
Allow-partial converts an individual segment/query failure into a best-effort
subset with `strategy_incomplete`; auto and strict propagate that failure.

Every numeric score is checked before it can enter normalization, sorting, or
tie-breaking. NaN hits, including NaN from exact reranking, are discarded and
increment `nan_discarded`. Rank-only LASER hits still require an exact rerank
source; Gate 10 does not make rank-only scores numerically mergeable.

Scratch bytes and worst-case disk I/O requests/bytes are overflow-checked and
preflighted before any segment writes results. A denial is
`resource_exhausted/budget_denied`, retryable with backoff, and has no partial
results. An RAII search lease increments the Collection reference/byte counters
only after successful preflight and releases them on every return path. Debug
destruction asserts that both counters are zero. `lease_acquired ==
lease_released` and a subsequent request using restored credits prove immediate
reuse.

## Successor-first seal protocol

The durable control record is checksummed and atomically replaced. Its phases
are `idle`, `cut_pending`, `successor_active`, `building`, and
`manifest_published`. One Collection control mutex serializes seal, compact,
checkpoint, and GC.

The seal sequence is:

1. Allocate the empty successor and Flat target identities, record the WAL-cut
   intent as `cut_pending`, and fsync the control state.
2. Enter a short admission gate, drain all previously admitted operations and
   their mutation/metadata/ID-map stages, checkpoint the old active and empty
   successor, write the immutable Collection checkpoint, and reset the logical
   WAL to its checkpoint frame.
3. Durably record `successor_active`, atomically publish a routing snapshot
   where the old active is read-only and the successor is writable, then
   release the short gate. New mutations now route only to the successor.
4. Pin the old snapshot plus successor, export the source rows and durable
   replacement map, then build the default immutable Flat target through the
   existing DiskFlat BuildFactory and build/snapshot leases. Searches and
   writes remain admitted throughout this potentially long build.
5. Publish the target through the manifest-v2 five-step artifact transaction,
   patch it to lifecycle `sealed`, mark sources `gc_pending`, atomically install
   the replacement routing snapshot, and take a full checkpoint. The sealed
   source op IDs are therefore covered by the checkpoint cut and are absent
   from replay work.

Vamana remains only a future L2 target candidate. Gate 10 always builds Flat
and does not expose DiskANN mutable internals.

### Four real-kill recovery cuts

`CollectionFacade.SealFourPointSigkillRecoveryRollsBackOrForward` forks one
child per row below on `/home/huangliang/md1/tmp`, delivers real `SIGKILL`, and
opens the same directory twice after recovery.

| injected cut | durable state at kill | recovery rule and assertion |
|---|---|---|
| cut recorded, successor not yet created | `cut_pending` | remove replacement intent, roll back the cut, restore the old active, then permit a fresh seal |
| successor created and admission switched | `successor_active` | reopen old read-only source plus writable successor and resume the seal |
| export/build in progress | `building` plus a real DiskFlat `after_staging_write` orphan | old snapshot and successor remain usable; restart observes and removes `.alaya_staging`, clears the partial map, then retries build |
| manifest atomically published | `manifest_published` | open the sealed target, roll the replacement forward, leave the source `gc_pending`, and continue with the successor |

At every cut the three pre-kill rows survive, a post-recovery successor write
is searchable, the resumed state has exactly one sealed generation, and a
second reopen is stable.

### Concurrent successor evidence

`CollectionFacadeStress.ConcurrentSealSearchAndWritesRouteToSuccessor` pauses
the seal after the successor routing switch and before the Flat build. While
paused, concurrent searches retain valid old snapshots and concurrent writes
complete on the successor. The test then releases the builder and verifies all
65 rows and zero pending operations after reopen. The focused TSan lane is run
with address randomization disabled:

```bash
setarch x86_64 -R env TSAN_OPTIONS=halt_on_error=1:history_size=7 \
  build/TSan/tests/collection/collection_facade_stress_test \
  --gtest_filter=CollectionFacadeStress.ConcurrentSealSearchAndWritesRouteToSuccessor
```

## Epoch GC and retention

Each search request owns a shared routing snapshot. A pending-GC record keeps a
weak reference to the source `SegmentEntry`; physical deletion is eligible only
after every older snapshot/search reference has released. GC never closes the
admission gate and never waits for a reader: it reports that source as deferred
and can be retried.

The manifest transition is durable and ordered:

1. publish GC phase `reclaimable` through atomic manifest-v2 replacement;
2. remove only artifacts belonging to eligible `gc_pending` entries;
3. fsync the segments directory;
4. remove reclaimed entries and IDs from the manifest, set phase to `idle` or
   `pending`, and atomically publish again.

At least the most recently published sealed generation is retained as the
recovery marker. It is listed in `gc.retained_sources` and cannot be reclaimed
even if accidentally present in the pending set. Artifacts removed after the
reclaimable publication are roll-forward-only and are never reconstructed by
an older reader.

The compact/GC test explicitly pins an old routing epoch: the first `gc()`
returns `reclaimed=0,deferred=2` and both source directories remain. After the
pin is released, the second call returns `reclaimed=2`, removes both directories,
and leaves the newest sealed target and manifest entry intact.

## Flat-to-Flat compact

Compact accepts at least two manifest-v2 entries whose algorithm is Flat and
lifecycle is `sealed`. It consumes each source through the existing
`export_rows` operation, validates every exported row byte-for-byte against the
Collection-owned vector, rebuilds one larger Flat generation, persists a
replacement map, and atomically swaps routing and manifest state. Other engine
types return `not_supported` and are not silently mixed.

`CollectionFacade.FlatCompactPreservesRowsAndGcDeletesOnlyReleasedSources`
pins all per-LogicalId vector bytes and SHA-256 values before compact, then
proves byte and hash equality afterward. It also requires bit-identical ordered
IDs/distances for the same query, source lifecycle `gc_pending`, deferred GC
under a held epoch, and eventual physical deletion after release.

## Design §9.12 acceptance matrix

| §9.12 requirement | executable evidence |
|---|---|
| strict at 0%, 50%, and 100%; missing vector denial | Python `test_gate10_filter_policies_overfetch_stats_and_budget_reuse`; C++ `StrictMissingVectorAndBudgetDenialHaveZeroEffectAndReleaseLease` |
| auto plan selection and result correctness | `FilterPoliciesSelectAllExecutionsAndBoundedOverfetch` asserts all three thresholds and exact IDs |
| allow-partial subset and bounded iterative overfetch | same C++ test asserts `strategy_incomplete` with zero allowed retries and exactly two retries for the 50% auto case |
| NaN is never a tie | `RejectsIncomparableScoreDomainsAndDiscardsNaN` asserts an empty hit set and `nan_discarded=1` |
| budget denial has zero partial results and leases are reusable | C++ strict/accounting test plus Python versioned `CollectionResourceExhaustedError(partial=False)` and successful retry |
| successor-first explicit and automatic seal | `SealRotatesToSuccessorPublishesFlatAndReopens` and `AutoSealRotatesAtConfiguredRowThreshold` |
| four crash cuts | real-kill battery and orphan assertions in the crash table above |
| concurrent search/write during build | focused release and TSan successor stress |
| manifest lifecycle and WAL cut | seal/reopen test asserts Flat, `sealed`, and exact receipt `wal_cut`; repeated checkpoint/reopen proves replay convergence |
| epoch GC and retention | compact/GC held-epoch test asserts deferred then physical deletion while retaining the newest sealed target |
| Flat compact row/search identity | compact test asserts all vector bytes and per-row SHA-256 plus bit-identical ordered search results |
| C++ and Python facade/statistics | Collection facade tests and `test_gate10_python_seal_compact_gc_and_collection_stats` |
| frozen lines and generated surfaces | final diff checks, full CTest/Python identity/golden lanes, codegen drift check, and size map |

## Final local validation record

The final 2026-07-13 validation used the Release AVX2+FMA lane with Python and
LASER enabled.

| verification lane | result |
|---|---|
| Release configure/build | complete with `BUILD_PYTHON=ON`; all targets and new header closures built |
| C++ release preset | final run `117/117` passed; Collection label `11/11` passed |
| Python default suite | `456 passed, 8 skipped, 51 deselected`, two passes above the Gate-9 `454/8` reference |
| legacy algorithm identity | `33 passed, 0 failed` |
| filter/accounting evidence | strict 0/50/100, all three auto executions, allow-partial, exactly two overfetch retries, NaN discard, budget denial, and lease reuse passed in C++ and Python |
| seal crash battery | all four forked children died by `SIGKILL`; every cut recovered and converged on second reopen; the build cut left then cleaned a real staging orphan |
| successor concurrency | focused Release stress passed; focused TSan with `setarch x86_64 -R` passed `1/1` with no report |
| compact and GC | ordered search bits, per-row bytes and SHA-256 matched; held epoch deferred both sources, release reclaimed both; latest sealed marker remained |
| artifact golden | generator matched all 14 checked families; structural inventory test passed |
| code generation | rerun left both generated files byte-identical (`afb6aa07...` dispatch header, `ef158da2...` Python matrix) |
| wheel and size map | wheel `8,851,204` bytes, SHA-256 `162679bef96592ceef9236a6285b16afa168e5e153c4ce3b086f269234e9d214`; `+135,028` bytes from Gate 9-B |
| binary attribution | extension `33,323,064` bytes; index-factory object `23,655,592` bytes / `.text` `7,457,383` bytes; all 33 dispatch rows retained and the factory figures are unchanged |
| frozen graph/engine lines | graph DiskANN/LASER diff empty; all five memory, three disk, and DiskANN Segment-body diffs empty; legacy-wrapper source diff empty |
| core boundary | only `include/core/value_types.hpp`, `+5/-0`, containing `FilterExecution`; core SearchStats and SegmentStats are byte-unchanged |

One immediately preceding full run observed the pre-existing frozen DiskANN
performance sentinel at 6.34% adapter overhead against its 5% noise threshold.
An isolated retry measured 3.29%, and the final unmodified full preset passed
117/117. No frozen source or threshold was changed.

## Rollback policy

| code rollback | runtime rollback | on-disk roll-forward |
|---|---|---|
| Filter planner/accounting, lifecycle control plane, facade bindings, and tests are separate commits | auto-seal can be disabled with `auto_seal_rows=0`; callers may stop invoking seal/compact/GC while canonical mutation/search continues | retain the Gate-10 control-state reader, manifest-v2 reader, Flat target opener, replacement-map recovery, and GC recovery logic |
| Flat compact can be reverted independently because it composes `export_rows` and does not alter a Segment body | the active target remains Flat and no runtime fallback opens DiskANN mutable or rank-only LASER as a numeric merge source | a published sealed/compacted target always rolls forward; physically GC'd artifacts are not recoverable and must never be selected by rollback |
| the sole core delta is the additive `FilterExecution` enum | legacy wrapper behavior and warning/telemetry remain Gate-9-pinned | API rollback does not delete format readers, WAL checkpoints, recovery markers, or retained-source policy |
