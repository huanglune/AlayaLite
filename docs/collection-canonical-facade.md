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

## Gate 9-B legacy wrapper closeout

Gate 9-B retains the pinned lifecycle from Gate 9-A:
`V_public="1.1.0"` and `V_remove="1.2.0"`.  Every row below shares one
process-once entry token across all methods named in its trigger column.  The
warning is emitted by the outer public wrapper with a literal
`warnings.warn(..., stacklevel=2)`, so its filename and line number identify
the caller rather than `_legacy.py` or an implementation module.

| public surface | first-use trigger sharing one token | warning category | exact message | telemetry category |
|---|---|---|---|---|
| `alayalite.Index` | constructor or `load`; whichever is first | `AlayaLiteLegacyApiWarning` | `alayalite.Index is a legacy API and will be removed in AlayaLite 1.2.0; use alayalite.Collection instead.` | `index` |
| `alayalite.DiskCollection` | constructor or `open`; whichever is first | `AlayaLiteLegacyApiWarning` | `alayalite.DiskCollection is a legacy API and will be removed in AlayaLite 1.2.0; use alayalite.Collection instead.` | `disk_collection` |
| `alayalite.laser.Index` | first `fit`, `from_prefix`, `search`, `batch_search`, or `set_params` call | `AlayaLiteLegacyApiWarning` | `alayalite.laser.Index is a legacy API and will be removed in AlayaLite 1.2.0; use alayalite.Collection instead.` | `laser` |
| `alayalite.vamana.build_index` | first builder call | `AlayaLiteLegacyApiWarning` | `alayalite.vamana.build_index is a legacy API and will be removed in AlayaLite 1.2.0; use alayalite.Collection instead.` | `vamana` |
| `Client` index methods | first `list_indices`, `get_index`, `create_index`, `get_or_create_index`, `delete_index`, or `save_index` call | `AlayaLiteLegacyApiWarning` | `alayalite.Client index methods is a legacy API and will be removed in AlayaLite 1.2.0; use collection methods instead.` | `client_index` |
| `Collection.get_cpp_index()` | first call, independent of the `Index` token | `DeprecationWarning` | `Collection.get_cpp_index() is deprecated and will be removed in AlayaLite 1.2.0; use canonical Collection methods instead. The returned native view is read-only.` | `index` |

Nested implementation calls do not consume or misattribute a second public
entry.  `Client.create_index` suppresses the nested `Index` boundary after the
`client_index` warning, and LASER build suppresses the nested Vamana warning
while retaining the original monkeypatchable public builder seam.  A later
direct Vamana call still emits its own first-use event.

### Telemetry schema and sink

Each first-use warning atomically creates one schema-v1 record:

| field | value/constraint |
|---|---|
| `schema_version` | integer `1` |
| `event` | `legacy_api_used` |
| `category` | `index`, `disk_collection`, `laser`, `vamana`, or `client_index` |
| `caller_file` | warning callsite filename |
| `caller_line` | warning callsite line number |
| `removal_version` | `1.2.0` |

The record is retained in process memory and emitted at INFO to the
`alayalite.legacy` logger with the same structured fields.  It contains no API
arguments, IDs, vectors, paths passed as data, metadata, documents, or query
results, and no network sink exists.  The warning/record token is claimed
under one lock, so concurrent first use still emits once.

### Wrapper routing and compatibility-gate separation

`Index` continues to compute and store through the existing `PyIndex`; the
outer wrapper does not open another WAL.  `DiskCollection` owns one outer
Python forwarding object and the same native DiskCollection-v1 instance, so
all three search conversions and pending semantics remain in their original
native boundary.  LASER and Vamana forward to their existing implementations.
Client collection methods remain canonical, while only its index-method set is
wrapped.  `get_cpp_index()` returns the same cached read-only native view and
does not gain mutation, RocksDB, or internal-row methods.

| compatibility class | Gate 9-B action | independent retirement rule |
|---|---|---|
| API legacy | warning/telemetry wrappers above, removed at `1.2.0` | semver removal after the one-version window |
| source bridge | native pybind classes, old builders, and model adapters remain | delete only after source consumers migrate |
| format reader | legacy recovery/import and DiskCollection-v1 readers remain | delete only after inventory, converter, corpus, and reader telemetry gates |

No API-wrapper rollback may reopen a second PyIndex WAL for canonical
Collection.  Source bridges and format readers are not tied to the API
removal clock.

### Four-quirk bidirectional golden map

| §7.4 quirk | legacy wrapper assertion | canonical target assertion |
|---|---|---|
| memory ID-max padding versus Disk single truncation | `test_index_return_shapes_and_boundaries` locks `uint32` max padding; `test_disk_collection_public_shape_visibility_and_errors` locks list truncation | `test_target_search_truncates_without_id_max_or_any_sentinel` locks paired short hits and valid count |
| three Disk search shapes and dense sentinels | the Disk golden locks list / `uint64` ndarray / tuple-of-ndarrays, `(N,k)` shapes, `UINT64_MAX` and `NaN` tails, dtypes, and method-specific error text | `test_target_single_and_batch_share_one_short_row_response_schema` locks one flat response schema, offsets, valid counts, paired IDs/distances, and dtypes |
| WAL owner in PyIndex | the Index golden observes its sole `recovery/wal.bin` after mutation and absence of a canonical WAL | `test_target_cpp_collection_is_the_only_wal_and_scalar_owner` observes only `collection_wal_v1` and no PyIndex WAL/RocksDB owner |
| Disk pending is unsearchable and excluded from size | the Disk golden locks `add -> size()==0/search==[] -> flush -> size()==3` | `test_target_mutable_add_is_searchable_before_return_and_size_is_live` locks searchable receipts, live size, accepted/pending counts, and bytes |

### §9.11 verification reconciliation

| §9.11 requirement | executable evidence |
|---|---|
| Python/C++ parity | `test_collection_canonical.py` direct-native/reopen parity plus the C++ Collection parity tests in the release preset |
| warning category, stacklevel, once-only, telemetry | `test_legacy_warnings.py` covers all six public surfaces and the structured logger sink |
| legacy parameters, returns, exceptions | `test_python_api_golden.py`, `test_index_types.py`, the dispatch matrix, and the DiskCollection suites |
| canonical truncation, response, size/accounting | the four target-half tests in `test_collection_target_golden.py` |
| algorithm matrix and identity | `test_dispatch_matrix.py`, `test_index_algorithm_identity.py`, and `test_collection_canonical_matrix.py` |
| wheel/package size | CPython 3.13 Linux wheel is `8,716,176` bytes, `+3,496` from the Gate 9-A `8,712,680`-byte reference; SHA-256 is `4f93d1fe2f40bca3d1f6d5f46b091c52349456347633a8e730bdb7e40a94f88c` |
| read-only `get_cpp_index()` | existing Collection tests assert `mutable is False` and missing mutation methods; Gate 9-B adds warning callsite/category/once-only assertions |

The final 2026-07-13 execution record is:

| verification lane | result |
|---|---|
| release configure/build | `BUILD_PYTHON=ON`, Release build completed |
| C++ release preset | `117/117` passed |
| Python default suite | `454 passed, 8 skipped, 51 deselected`; exactly seven passes above the Gate 9-A `447/8` reference |
| warning/telemetry contract | `7 passed` |
| legacy algorithm identity | `33 passed, 0 failed` |
| legacy recovery corpus | `7 passed, 0 failed` |
| canonical cutover set | `42 passed, 0 failed` |
| DiskCollection family | `135 passed, 5 skipped` |
| artifact golden | all 14 families match `artifact-baseline.json` |
| code generation | generator rerun left the source worktree unchanged |
| wheel/size map | `build/g9b-wheel/size-map.json` records the `8,716,176`-byte wheel |

### Gate 9-B rollback policy

| code rollback | runtime routing rollback | on-disk policy |
|---|---|---|
| the Python wrapper/telemetry and tests are separable commits | an entry may temporarily route to the retained legacy implementation, but canonical Collection must keep its coordinator | retain canonical WAL/importer/read-only-view code and all legacy format readers |
| no frozen core, graph/disk Segment, or segment-body source changes are part of this gate | warning emission can be disabled only with the whole wrapper rollback, not by opening another owner | switched/imported directories remain roll-forward-only; API removal never implies format-reader removal |

## Gate 11 API removal at 1.2.0

AlayaLite 1.2.0 removes the one-release compatibility window. `Index`,
`DiskCollection`, `alayalite.laser.Index`/`RawIndex`,
`alayalite.vamana.build_index`, the six `Client` index methods, and the
`Collection.get_cpp_index()`/`get_index()` escape hatches are no longer in a
public `__all__` or class dictionary. A direct access through an old import
spelling raises `AlayaLiteLegacyApiWarning` with a stable “removed in 1.2.0”
message instead of executing the retired path.

This API removal does not retire a persisted-format reader. All six checked-in
legacy recovery cases—including the four historical `type=index` layouts—now
open as canonical `Collection` objects through `LegacyImporter`. The importer
continues to fingerprint and preserve every source byte. DiskCollection-v1,
old memory snapshot, LASER, manifest, WAL, and RocksDB checkpoint decoding are
tracked independently under the Gate-11 reader inventory.
