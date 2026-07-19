<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# DiskANN readonly Segment contract

> **Superseded / historical (2026-07-19).** The DiskANN index and Segment
> family were retired. The identities and APIs below are preserved only as a
> migration record.

`alaya::disk::DiskAnnSegment` is the Gate 8 readonly adapter for the retained
`alaya::diskann::DiskANNIndex`. The native index continues to own graph
traversal, cached beam scheduling, page reads, PQ rerank and its bounded
`ThreadData` pool. The adapter only validates contract-v3 values, accounts for
resources and translates native label/distance rows into `SearchResponse`.

The adapter registers a native coroutine operation table for contract-v3
`start_search`. `DiskANNIndex::search_pipelined` retains the native beam/page
loop on a readonly load, while the stable synchronous `search()` remains a
start-then-wait wrapper. Runtime capabilities therefore report
`native_async=true` and `cooperative_cancel=true`.

## Identity, format and capabilities

| item | readonly contract |
|---|---|
| descriptor | `algorithm_id=diskann` (8), native meta format version 1 or 2, factory version 1, float32/L2/disk |
| current registry identity | `diskann_segment / diskann` |
| retained direct identity | `diskann_index / diskann` |
| runtime feature bit | `DiskEngineFeatureFlags::diskann_segment`; disabling it returns `not_supported` only from `DiskAnnSegmentFactory` |
| capabilities in this checkpoint | synchronous/native-async search, batch search and stats through `AnySegment`; readonly and reentrant |
| deliberately absent | mutation bundle, save/export, checkpoint, freeze and public facade routing |

The logical artifact names map without conversion to the native family:
`meta -> meta.bin`, `index -> diskann.index`, `ids -> ids.bin`,
`cache_ids -> cache_ids.bin`, `cache_nodes -> cache_nodes.bin`, plus
`pq_pivots -> pq_pivots.bin` and `pq_compressed -> pq_compressed.bin` when the
native metadata enables PQ. Artifact paths must name one coherent directory.
The adapter opens the existing bytes in place and writes nothing.

A static readonly load rejects metadata with `live_count != max_slot_id`.
`DiskANNIndex::load(updatable=false)` does not apply a persisted tombstone
bitmap, so accepting such an artifact would silently resurrect rows. Reading
that mutable state belongs to G8-B.

## Typed search and resources

Queries are float32 `TypedTensorView` rows. There is no implicit conversion and
no engine-local metadata filter. DiskANN knobs live in the algorithm-keyed
`DiskAnnSegmentSearchExtension`: search-list size, PQ choice, rerank choice and
count, and deterministic scheduling. Requests beyond the retained pool's
search-list capacity are rejected with `resource_exhausted`; the adapter never
creates an unbounded scratch or worker pool.

Open admission checks the resident artifacts, cache artifacts, a conservative
four-slot native scratch-pool estimate, and I/O file/byte credits. Search
admission checks temporary label/distance/frontier memory and per-query graph
I/O credits. Native `SearchStats` counters are translated to the optional v3
stats sink.

Results use compact offsets and counts; native sentinel padding never becomes
a valid `SearchHit`. Scores are numeric L2 distances, results are marked
`approximate`, and PQ rerank also marks `exact_reranked`. A row with K hits is
`complete_k`; a proven short live set is `eligible_exhausted`; other short
approximate output is `strategy_incomplete`.

Cancellation and deadline are combined into one non-owning native probe. The
beam calls it only after the seed or an expansion wave has completely drained
and all returned pages have been consumed; the per-page and per-node hot loops
contain no cancellation branch. A stop never abandons an awaitable or its
`ThreadData` buffer. With partial disabled, the response is invalidated. With
partial enabled, compact completed slices are retained and every affected row
is marked `cancelled_partial`.

## §3.6 native-async contract

| §3.6 requirement | implementation | contract test |
|---|---|---|
| `start_search -> OperationHandle` and completion exactly once | an owning `NativeOperationState` has an atomic finish gate and an idempotent atomic cancel bit | `CompletionIsExactlyOnceUnderCancelStress` |
| requested lane and no inline reentrancy | work starts on a detached execution trampoline; terminal delivery is dispatched through the copied `RuntimeLane` | `NativeCompletionUsesRequestedLaneExactlyOnceAndAllowsReentry` |
| pin input, sink, routing/artifact, leases and credits | operation state owns the segment, request, copied context, internal native output/stats and `lifetime_pin` until callback return | both `*AtDrainedWavePinsBuffers*` tests |
| cooperative cancel during a beam | `BeamSearchCancelProbe` is optional/non-owning and is inspected only at drained-wave boundaries | `CancelAtDrainedWavePinsBuffersAndAppliesBothPartialPolicies` |
| timeout and safe I/O drain | the deadline is observed by the same probe after the page awaitable has returned; coroutine-owned `ThreadData` survives through result translation | `TimeoutAtDrainedWavePinsBuffersAndAppliesBothPartialPolicies` under ASan/UBSan |
| discard/retain terminal mapping | discard invalidates all metadata; retain compacts available hits and marks `cancelled_partial` | cancel and timeout dual-policy tests |
| fan-out child completion | each child handle receives cancellation and routing remains pinned until every child callback | `FanoutCancellationPropagatesAndWaitsForEveryChild` |
| callback re-entry and same-lane sync wait | callback re-entry is queued; the synchronous wrapper replaces its wait lane to avoid self-deadlock | lane/re-entry test above |
| mixed concurrency | the pipeline cache is mutex-serialized while sync/async operations retain independent operation state | `ConcurrentNativeAsyncAndSyncMixedStress` under TSan with ASLR disabled |

## Approved kernel hooks and compatibility

The kernel delta is restricted to `beam_search_async.hpp` and
`diskann_index.hpp`:

1. Both async beams accept an optional `BeamSearchCancelProbe`. The original
   reactor-reference overload remains, and a null probe preserves the old
   traversal, output and PQ-rerank rejection behavior.
2. Probe calls occur after a fully consumed seed/wave and at the all-waves-
   drained query boundary. No page/node hot loop was changed.
3. `search_pipelined` accepts readonly loads through their owned `PageReader`;
   the former updatable-blocking rejection remains unchanged.
4. Optional `per_query_counts` are a pure additive output. A null pointer keeps
   the sentinel-padded legacy surface unchanged.
5. The probed readonly PQ path can apply the same rerank candidate limit using
   exact distances already collected during traversal; it submits no new
   rerank read after cancellation.

On the fixed 32K-row sentinel index, 240 AB/BA-interleaved paired samples
measured +0.206% for original versus modified direct with no probe, +0.557%
for modified direct versus the Segment synchronous wrapper, and -0.268% for a
present-never-cancelled probe versus no probe. All are below the 2% Gate-8
stop threshold.

## Release-note material

- Added an independently gated, immutable DiskANN Segment adapter that opens
  the native file family in place and provides typed sync/native-async search,
  batch search, compact numeric results, resource admission, stats and an
  explicit direct-index rollback identity.
- Native cancellation and deadlines stop cooperatively only at drained beam
  waves, keep all request/I/O buffers pinned to exactly-once completion, and
  implement both discard and retained-partial terminal policies. No mutation
  or checkpoint capability is advertised by the readonly adapter.
