<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Canonical Collection facade audit

## Audit status

This document is the audit-first deliverable for Gate 9-A.  The audit was
performed on `eda0e53` before adding a public C++ facade or changing a Python
binding.  Implementation was initially **stopped pending a ruling** because existing
`python/tests/client/test_collection.py` assertions require the canonical
Python `Collection` to keep using and exposing the legacy `PyIndex`/RocksDB
backend.  That requirement conflicts with the binding and ownership decisions
for Gate 9-A, and changing those assertions is not covered by the narrowly
allowed §7.4 target-column test updates.  Option (a) was subsequently approved
under the conditions recorded in [Ruling result](#ruling-result-2026-07-13),
so implementation is now authorized to continue.

No frozen `include/core/` contract or graph/disk Segment implementation needs
to change to implement the intended facade.  A facade-private exact mutable
Flat operation table can compose `SegmentedCollection`; this remains a
composition-layer implementation and does not make DiskANN mutation public.

## Pinned release lifecycle

| item | pinned value | Gate 9-A placement |
|---|---|---|
| canonical public version (`V_public`) | `1.1.0` | Python package constant and release metadata |
| legacy removal version (`V_remove`) | `1.2.0` | Python package constant and this contract |

The warning/wrapper conversion itself belongs to Gate 9-B.  Gate 9-A must not
change behavior of `Index`, `DiskCollection`, `alayalite.laser.Index`, the
Vamana builder, or Client's index methods.

## Target canonical surface

The intended public C++ entry is `alaya::Collection`, implemented as a facade
over `internal::collection::SegmentedCollection`.  Its public lifecycle and
operations are `create`, `open`, `add`, `upsert`, `remove`, batch mutation in
`per_row_independent` and `all_or_nothing` modes, `search`, `batch_search`,
`get_by_id`, `checkpoint`, `stats`, and `close`.

The facade owns the only logical WAL and defaults writes to `wal_fsync`.
`open` must select the already-switched importer target or invoke
`LegacyImporter` directly on a legacy layout; it must never instantiate the
legacy recovery reader first.  Scalar item ID, document, and metadata live in
the Collection checkpoint/version map, not in a Python-owned RocksDB database.

Python `alayalite.Collection` is the canonical name.  It may retain convenient
projection helpers, but its storage and mutation calls must all use the new
native `Collection` binding.  The canonical response is one logical schema for
one or many queries: paired flat `ids`/`distances`, `offsets`, `valid_count`,
per-query status and completeness.  Every row satisfies
`offsets[i+1] - offsets[i] == valid_count[i]` and
`0 <= valid_count[i] <= top_k`; no valid hit is a sentinel.

Canonical creation accepts the legal non-RaBitQ rows under their real
algorithm identity.  `quantization_type="rabitq"` is legal only with an
explicit `index_type="qg"`.  The three legacy declarations that silently map
`{hnsw,nsg,fusion} + rabitq` to QG remain behavior of legacy `Index` only.

## Existing Python `Collection` inventory

| current public method | target semantics | disposition |
|---|---|---|
| `Collection(name, index_params)` | lazy schema capture is allowed; native create occurs once dimension/dtype are known; Collection WAL defaults to `wal_fsync` | canonical incorporation |
| `load(url, name)` | canonical open; legacy layout goes directly through the non-destructive importer | canonical incorporation |
| `insert(items)` | ordered `insert_only` batch; stable row receipts; successful mutable adds are searchable before return | canonical incorporation (`add`/batch) |
| `upsert(items)` | ordered LogicalId upsert with Collection-owned versions and metadata | canonical incorporation |
| `delete_by_id(ids)` | LogicalId remove with stable per-row `not_found`/`deleted` status | canonical incorporation |
| `batch_query(vectors, limit, ...)` | must be backed by canonical batch search and short rows, never sentinel padding | canonical convenience projection; canonical response must also be exposed |
| `hybrid_query(...)` | existing exact/filter convenience cannot open a second owner | retain only if implemented over the same native Collection; no Gate 10 pushdown work |
| `filter_query(...)` | read Collection-owned metadata at a pinned routing snapshot | retain only as a convenience over the same native Collection |
| `delete_by_filter(...)` | deterministic LogicalId expansion followed by coordinator mutations | retain only over the same native Collection |
| `get_by_id(ids)` | project Collection checkpoint data by LogicalId | canonical incorporation |
| `build_filter(...)` | Python filter compiler; no persistence ownership | convenience helper, no new Gate 10 execution work |
| `reindex(...)` | export/checkpoint, create with new build parameters, re-add, atomic swap; seal/compact remains Gate 10 | canonical incorporation under the ruling |
| `save(url)` | full Collection checkpoint/manifest publication, never legacy `Index.save` plus RocksDB | canonical incorporation (`checkpoint`) |
| `set_metric(metric)` | schema mutation is legal only before native create | canonical incorporation |
| `get_index_params()` | return canonical creation schema (legacy-only RocksDB path cannot be an owner) | canonical incorporation |
| `get_index()` | cannot return a second legacy owner | compatibility alias to the same read-only native view |
| `get_cpp_index()` | eventual deprecated read-only view; must not expose coordinator-bypassing mutation | canonical read-only view; warning remains Gate 9-B |
| `close()` / `__del__()` | close and drain the native Collection idempotently | canonical incorporation |

## Legacy `Index` inventory

All methods below remain on their current path in Gate 9-A.  Gate 9-B will
turn the type into a warning/telemetry wrapper; no behavior change is
authorized here.

| current public method | target/compatibility meaning | disposition |
|---|---|---|
| constructor, `fit`, `load`, `save`, `close` | legacy memory-index lifecycle and artifact reader | Gate 9-B wrapper; unchanged in G9-A |
| `search` | retains exact ID-dtype-max padding when `topk > count` | Gate 9-B wrapper; unchanged in G9-A |
| `batch_search`, `batch_search_with_distance` | retain legacy dense return and validation behavior | Gate 9-B wrapper; unchanged in G9-A |
| `insert`, `remove` | retain legacy mutation behavior for this entry during G9-A | Gate 9-B wrapper; unchanged in G9-A |
| `get_data_by_id`, `get_dim`, `get_dtype`, `get_params` | legacy inspection | Gate 9-B wrapper; unchanged in G9-A |
| `get_cpp_index` | legacy native escape hatch | Gate 9-B deprecation work; unchanged in G9-A |

## Legacy `DiskCollection` inventory

| current public method | target/compatibility meaning | disposition |
|---|---|---|
| constructor and `open` | legacy DiskCollection-v1 lifecycle | Gate 9-B wrapper; unchanged in G9-A |
| `add` | pending-only copy; not searchable and not counted by legacy `size()` | Gate 9-B wrapper; unchanged in G9-A |
| `flush` | publish pending immutable segment | Gate 9-B wrapper; unchanged in G9-A |
| `import_laser_segment` | legacy LASER import path | Gate 9-B wrapper; unchanged in G9-A |
| `search` | list of `(label, distance)` and truncation | Gate 9-B wrapper; unchanged in G9-A |
| `batch_search` | dense labels ndarray with `UINT64_MAX` padding | Gate 9-B wrapper; unchanged in G9-A |
| `batch_search_with_distance` | tuple of dense ndarrays with ID/NaN padding | Gate 9-B wrapper; unchanged in G9-A |
| `size`, `dim` | published searchable rows only; pending excluded | Gate 9-B wrapper; unchanged in G9-A |

Sentinel creation remains solely in this outer legacy binding.  Neither
canonical Collection nor an internal Segment/merge buffer may use those
values as hits.

## `Client` inventory

| current public method | target semantics | disposition |
|---|---|---|
| constructor | may discover canonical collections plus legacy indexes | collection routing becomes canonical; index routing unchanged |
| `list_collections`, `get_collection`, `create_collection`, `get_or_create_collection` | manage canonical `alayalite.Collection` objects | canonical incorporation |
| `save_collection`, `delete_collection` | canonical checkpoint/close and optional directory removal | canonical incorporation |
| `list_indices`, `get_index`, `create_index`, `get_or_create_index` | legacy index manager | Gate 9-B wrapper; unchanged in G9-A |
| `save_index`, `delete_index` | legacy index persistence/lifecycle | Gate 9-B wrapper; unchanged in G9-A |
| `reset` | close both maps; each object retains its own routed semantics | mixed manager; index half unchanged |

## §7.4 target-to-test map

| target requirement | required canonical assertion |
|---|---|
| truncate instead of ID-max padding | empty and short collections return only real hits for both single and batch; `valid_count <= top_k` |
| one single/batch logical schema | single has one offset interval; batch uses N intervals; flat IDs and distances have identical length |
| C++ Collection is the only WAL owner | canonical create/add/checkpoint emits only `.alaya_internal/collection_wal_v1`; no legacy recovery WAL or RocksDB owner is opened |
| searchable `size()` plus explicit accepted/pending/bytes | concurrency failpoint observes pending counters; successful default add receipt is searchable and `size()` advances before return |

Additional required tests are parity of C++ and Python result/status bit
patterns, both batch mutation modes, canonical algorithm identity, explicit-QG
negative cases, and canonical-open importer drills with unchanged source
SHA-256.

## Stop-report: blocking contradictions

The following are not implementation gaps; they are mutually exclusive
requirements in the current task scope.

1. `python/tests/client/test_collection.py::test_insert_uses_explicit_build_threads`
   patches `Index.fit` and requires `Collection.insert` to call that legacy
   Python backend.  Gate 9-A explicitly requires replacing that backend with
   the C++ `Collection` binding.  The assertion is not one of the four §7.4
   target-column behaviors, so the task's test-edit rule does not authorize
   changing it.
2. `test_cpp_batch_get_scalar_data_by_internal_ids` and
   `test_cpp_internal_id_bridge_supports_uint64` require `get_cpp_index()` to
   return the old `PyIndexInterface`, including direct mutable `remove()` and
   RocksDB scalar/internal-row methods.  That handle can mutate outside the
   Collection coordinator and makes PyIndex/RocksDB a second state owner,
   violating the sole-WAL-owner and scalar-checkpoint decisions.  A read-only
   view cannot satisfy these tests.
3. The materialized-view tests call
   `get_cpp_index().get_materialized_view_partition_count()` and pin legacy
   PyIndex partition invalidation.  Preserving this requires keeping the old
   backend or adding filter/materialized-view behavior explicitly excluded
   until Gate 10.  These assertions are also outside §7.4.
4. `test_reindex_large_scale` requires the existing public `reindex()` graph
   rebuild behavior, while Gate 9-A excludes seal/compact and forbids retaining
   a second legacy storage owner.  A no-op chosen merely to satisfy recall
   would not preserve the asserted API semantics.
5. `python/tests/golden/test_python_api_golden.py` describes its Collection
   case as legacy behavior and pins `batch_query`'s dict-of-nested-lists shape,
   while Gate 9-A calls `alayalite.Collection` canonical and requires a NumPy
   offsets/valid-count response.  The requirements need a ruling on whether
   `batch_query` remains a compatibility projection beside new canonical
   `search`/`batch_search`, or whether this golden assertion may change.

Required ruling: either (a) authorize updating/removing the listed
backend-introspection, materialized-view, reindex, and Collection golden
assertions as part of the canonical cutover; or (b) move the Python
`Collection` backend cutover to Gate 9-B and limit G9-A to the new C++/pybind
facade plus an additional Python canonical entry.  Option (b) conflicts with
the pinned decision that the existing `alayalite.Collection` name is the
canonical entry, so option (a) is the direct resolution.

## Ruling result (2026-07-13)

Option (a) was approved and the stop report is resolved under these binding
conditions:

1. `build_threads` remains a tested public creation/build parameter, but the
   test follows it into the native canonical path instead of patching
   `Index.fit`.
2. `get_cpp_index()` becomes a read-only native Collection view in Gate 9-A.
   Direct mutation and RocksDB/internal-row ownership are removed; scalar
   lookup, uint64 LogicalId bridging, and dtype preservation remain covered
   through canonical `get_by_id` projections.
3. Materialized-view partition-count assertions may be removed as legacy
   implementation introspection.  Filter, hybrid-query, and delete-by-filter
   value assertions remain and run as pinned-snapshot/client-filter
   projections over the same native Collection, without Gate 10 pushdown.
4. `reindex()` is reimplemented as export through canonical records, create a
   replacement native Collection with the requested parameters, re-add,
   atomically swap the Python binding, and checkpoint.  It is neither a no-op
   nor the Gate 10 rotate/seal state machine.
5. `batch_query` retains its existing dict-of-nested-lists public shape and
   values as a convenience projection over canonical batch search.  New
   canonical `search`/`batch_search` methods expose NumPy flat arrays plus
   offsets and valid counts alongside it.
6. Every modified pre-existing test is reported as old assertion to new
   canonical assertion, or as removed by an already-pinned decision.  Public
   values, return shapes, recall, and error semantics remain unchanged.

Implementation may therefore continue; this section supersedes the audit
status sentence saying that work is stopped, while retaining the original
conflict record for traceability.

## Canonical cutover result

The public C++ entry is `#include <alaya/collection.hpp>` and
`alaya::Collection`.  It composes the coordinator, a facade-private exact Flat
active generation, and any sealed imported segments.  The requested target
algorithm/factory identity is persisted separately from that active mutation
generation; Gate 9-A does not add Gate 10 seal/rotation.  The facade schema
also persists whether a legacy sealed segment must be registered, so every
later open continues through the importer marker instead of dropping the
dual-read path.

Python `alayalite.Collection` owns only this native object.  `add`, `upsert`,
`remove`, and `mutate_batch` expose the coordinator receipts and both batch
modes.  `search` and `batch_search` expose the common flat NumPy response;
`batch_query` remains the approved compatibility projection.  The read-only
view returned by `get_cpp_index()` has no mutation or internal-row API.
Filter convenience methods materialize one native record snapshot and filter
that owned projection client-side; no filter pushdown or seal machine was
introduced.

The package exports `COLLECTION_V_PUBLIC="1.1.0"`,
`LEGACY_API_V_REMOVE="1.2.0"`, and the versioned Collection status exception
hierarchy.  The project/wheel release version is `1.1.0`.

## Pre-existing test ruling map

| pre-existing test | old assertion | canonical assertion or pinned decision |
|---|---|---|
| `test_insert_uses_explicit_build_threads` | patch `Index.fit` and inspect `num_threads` | native read-only options report `build_threads=7`, proving the value reached canonical creation |
| `test_cpp_batch_get_scalar_data_by_internal_ids` | query RocksDB scalar rows by internal IDs, including an empty placeholder | query live records by LogicalId; preserve documents/metadata and float32 vector dtype; no internal-ID placeholder contract |
| `test_cpp_internal_id_bridge_supports_uint64` | mutate the old `PyIndexInterface` by uint64 internal row | assert the native view is read-only, retain uint64 configuration and vector dtype, and remove through Collection LogicalId |
| `test_rabitq_collection_build_with_scalar_data_keeps_space_accessible` | implicit `hnsw+rabitq`, optionally skip, then read an internal row | explicit `qg+rabitq` per decision #18; build must run and the same vector value/shape remains readable by LogicalId |
| `test_hybrid_query_materialized_view_merges_multiple_partitions` | require three legacy materialized-view partitions | partition count removed by the ruling; exact filtered result remains `c1,c2,a2` |
| `test_hybrid_query_materialized_view_partition_only_eq_filter` | require two legacy materialized-view partitions | partition count removed by the ruling; exact filtered result remains `a1,a2` |
| `test_materialized_view_invalidates_after_incremental_insert` | require partition count `2 -> 0` | assert the view cannot mutate; incremental item `a3` remains immediately visible to hybrid filtering |
| `test_rabitq_batch_hybrid_query_uses_materialized_view_without_crashing` | implicit RaBitQ plus ten partition introspection, with optional skip | explicit QG; introspection removed and all five filtered result rows remain value-checked |
| `test_reindex_large_scale` | rebuild through the legacy graph owner | checkpoint/export, create with `ef_construction=211` and three build threads, re-add, atomic swap, then preserve the existing recall assertions |
| `test_legacy_reader_recovers_checked_in_corpus` (Collection cases) | call old `get_data_num`, run the destructive reader, and delete its WAL | canonical `size`/stats plus value checks; direct importer marker is present and every pre-import source byte remains SHA-identical; Index cases keep the old reader assertions |
| `test_collection_recovers_after_unclean_exit` | inspect Python/PyIndex `recovery/CURRENT`, snapshots, and `wal.bin` | inspect the sole C++ `collection_wal_v1` checkpoint/WAL and retain the same post-crash document/metadata values |

`test_python_api_golden.py` was not edited: its Collection value, key order,
and nested-list shape assertions continue to pass through the compatibility
projection.

## Rollback policy

| code rollback | runtime routing rollback | on-disk policy |
|---|---|---|
| facade and Python binding may be reverted | entry routing may temporarily return to a legacy wrapper | never delete the new reader, coordinator, importer audit, or replay code |
| internal Segment bodies remain independently revertible because this gate only composes them | no runtime fallback may reopen a PyIndex WAL as a second owner | continue dual-read/import; a switched marker, new WAL, ID map, or manifest rolls forward only |
