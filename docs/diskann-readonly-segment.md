<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# DiskANN readonly Segment contract

`alaya::disk::DiskAnnSegment` is the Gate 8 readonly adapter for the retained
`alaya::diskann::DiskANNIndex`. The native index continues to own graph
traversal, cached beam scheduling, page reads, PQ rerank and its bounded
`ThreadData` pool. The adapter only validates contract-v3 values, accounts for
resources and translates native label/distance rows into `SearchResponse`.

This checkpoint deliberately does not claim the design §3.6 native-async
proof. The current DiskANN coroutine has the cancellation gap described below;
the checked-in `into_any()` path therefore uses the frozen synchronous runtime
adapter and reports `native_async=false` and `cooperative_cancel=false`.

## Identity, format and capabilities

| item | readonly contract |
|---|---|
| descriptor | `algorithm_id=diskann` (8), native meta format version 1 or 2, factory version 1, float32/L2/disk |
| current registry identity | `diskann_segment / diskann` |
| retained direct identity | `diskann_index / diskann` |
| runtime feature bit | `DiskEngineFeatureFlags::diskann_segment`; disabling it returns `not_supported` only from `DiskAnnSegmentFactory` |
| capabilities in this checkpoint | synchronous search, batch search and stats through `AnySegment`; readonly and reentrant |
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

Cancellation and deadline checks currently occur before and after each native
single-query call and between batch rows. With partial disabled, the response
is invalidated. With partial enabled, completed slices are retained and marked
`cancelled_partial`. This is sufficient for the synchronous half but is not a
claim that an in-progress native beam cooperatively stops.

## §3.6 native-async audit

| §3.6 requirement | current evidence | Gate 8 status |
|---|---|---|
| `start_search -> OperationHandle` and completion exactly once | the frozen sync adapter supplies a tested slot | interim only; not native async |
| completion on the requested lane and no inline reentrancy | the common adapter dispatches after a detached execution trampoline | inherited sync-adapter proof only |
| input, sink, routing/artifact, leases and credits pinned to completion | `SearchRequest::lifetime_pin` and adapter state retain the request and segment | inherited sync-adapter proof only |
| idempotent cooperative cancel during a beam | `beam_search_async` exposes only `coro::task<vector<...>>`; its page awaitable owns a cancellable `BatchHandle`, but the beam does not expose it or inspect a cancel probe at wave boundaries | **blocked** |
| safe I/O drain after cancel/timeout | page buffers live in `ThreadData` until the coroutine returns, but no public operation owns and drains that coroutine under the v3 handle | **blocked** |
| partial/discard terminal mapping | implemented for synchronous pre/post and batch-row safe points | native async proof pending |
| fan-out cancel waits for every child completion | requires a real native child handle | pending |
| callback re-entry and same-lane sync deadlock | common sync adapter changes the wait lane and queues completion | native async proof pending |

No edits were made under `include/index/graph/diskann/**` to conceal this gap.

## Minimal native hook proposal (awaiting owner approval)

The smallest behavior-preserving kernel change is limited to
`beam_search_async.hpp` and `diskann_index.hpp`:

1. Add an optional non-owning cancel probe to both async beam functions. Check
   it only after an awaited page wave has drained and before submitting the
   next wave/expansion. A null probe keeps every existing call byte-for-byte on
   its current path.
2. Remove the unused `UringReactor&` and `fd` parameters from the async beams,
   updating their internal call sites mechanically. Reads already go through
   the owned `PageReader` awaitable.
3. Permit `DiskANNIndex::search_pipelined` on a static readonly load (it needs
   `reader_`, not mutable `page_io_`) and thread the optional cancel probe into
   each query. Return per-query counts so the wrapper never infers validity from
   sentinels.
4. The wrapper can then own one bounded operation state, copy/pin request
   references and internal output, invoke this native coroutine path, wait for
   all page waves to drain, apply partial/discard semantics, and post exactly
   one completion to `SearchContext::lane`.

The null-probe/direct-search path retains all algorithm and I/O semantics. The
new stop path changes only an explicitly cancelled operation after its current
wave is safe. Tests must cover exact-once/lane/no-inline, double and late
cancel, ASan buffer lifetime, both partial policies, Collection fan-out and
TSan mixed sync/async search before `native_async` is set true.

## Release-note material

- Added an independently gated, immutable DiskANN Segment adapter that opens
  the native file family in place and provides typed synchronous search, batch
  search, compact numeric results, resource admission, stats and an explicit
  direct-index rollback identity.
- Kept native async disabled pending a cooperative-cancel hook in the retained
  beam coroutine; no mutation or checkpoint capability is advertised by the
  readonly adapter.
