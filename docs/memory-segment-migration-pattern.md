<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Memory graph to Segment migration pattern

HNSW is the reference migration for Gate 0 and the first producer adapted to
the frozen [core contract v3](contract-v3.md) in Gate 2. The pattern is a direct
C++ API replacement: a public builder plus public `Graph` is not kept in
parallel with the Segment API.

## HNSW result

`HnswSegment<SearchSpace, BuildSpace>` now owns the searchable graph, both
spaces, and its search executor. Its public lifecycle is:

```cpp
static std::unique_ptr<HnswSegment> build(
    BuildInput, const HnswBuildOptions&, core::BuildContext&);
static std::unique_ptr<HnswSegment> open(
    core::ArtifactView, const core::OpenOptions&, core::OpenContext&);

core::Status search(const core::SearchRequest&) const;
core::Status batch_search(const core::SearchRequest&) const;
core::Status save(core::ArtifactWriter&, const core::SaveOptions&,
                  core::ArtifactManifest&) const;
core::Status stats(core::SegmentStats&) const noexcept;
core::Descriptor descriptor() const noexcept;
static core::Result<core::AnySegment> into_any(
    std::unique_ptr<HnswSegment>);
```

The type satisfies `Searchable`, `BatchSearchable`, `Saveable`, and
`StatsProvider`. It does not satisfy `Mutable`: the complete
prepare/stage/publish/abort/replay bundle is absent. `Descriptor` reports only
static identity; row counts and health are returned by `SegmentStats`.

`TypedTensorView` requires the native float32/int8/uint8 scalar type of the
instantiation. HNSW no longer silently converts float queries into byte
queries. Search writes flat `SearchHit` values and per-query
offset/count/status/completeness metadata. The AnySegment registration uses the
sync runtime adapter, exposes search/batch/save/stats slots, and derives
`mutable=false` from the absent bundle plus readonly instance configuration.

Artifact views and writers use logical-name mappings (`graph`, `data`, and,
for a distinct search space, `quant`) instead of an HNSW-specific tuple of
paths. `save` delegates to the pre-existing graph and space codecs, so the old
graph/data/quant bytes are unchanged. The returned manifest adds schema,
format, algorithm, logical names, and sizes. The design does not define a
standalone memory-segment manifest file yet, so this migration deliberately
does not add one; old artifacts remain directly openable.

The former public `hnsw_builder.hpp` is deleted. Construction policy now lives
in `index/graph/hnsw/detail/hnsw_builder_kernel.hpp`. Generic graph search,
hybrid search, materialized-view, and legacy Python mutation plumbing still
need graph access until their own abstraction steps; their access is isolated
in `detail/hnsw_segment_bridge.hpp`. Neither detail type is a registry entry or
a supported user API.

## NSG and Fusion Gate 5 result

`NsgSegment<SearchSpace, BuildSpace>` and
`FusionSegment<SearchSpace, BuildSpace>` expose the same contract-v3 lifecycle
as `HnswSegment`: typed-tensor `build`, `open`, `search`, `batch_search`,
`save`, `stats`, `descriptor() noexcept`, and `into_any`. Both are immutable,
reentrant for concurrent search, and satisfy `Searchable`, `BatchSearchable`,
`Saveable`, and `StatsProvider`; neither satisfies `Mutable`.

Each segment consumes `BuildContext`, `OpenContext`, and `SearchContext` and
uses logical artifact names `graph`, `data`, and, when the search and build
spaces differ, `quant`. Its returned manifest has schema version 1 and format
version 1. NSG format v1 is the retained NSG `Graph` codec plus the retained
space codecs; Fusion format v1 is the retained overlay-graph `Graph` codec plus
the same space codecs. No persisted bytes or standalone manifest file were
added, so artifacts written by the former C++ builders remain openable.

Fusion remains a composition of the detail HNSW and NSG construction kernels;
it does not copy either graph-building implementation. The former public
`nsg_builder.hpp` and `fusion_graph.hpp` signatures are deleted. Their retained
construction code now lives under `index/graph/nsg/detail` and
`index/graph/fusion/detail` for remaining internal consumers; neither detail
kernel is restored as a public builder API.

The Python dispatch matrix now sends all 20 non-RaBitQ NSG and Fusion rows to
`nsg_segment/nsg` and `fusion_segment/fusion`. `NONE` builds and searches one
raw space, including its scalar storage when enabled. `SQ4` and `SQ8` build the
graph in a scalar-free raw build space and search in the selected quantized
space; scalar-enabled rows attach scalar storage to that search space. The
three RaBitQ rows are governed by the separate pinned QG mapping below.

This is an intentional compatibility change: `index_type="nsg"` and
`index_type="fusion"` now build the declared algorithm instead of silently
building HNSW. The independent `nsg_segment` and `fusion_segment` feature bits
default on. Disabling either bit selects that row's recorded
`hnsw_segment/hnsw` legacy factory and therefore reproduces the former behavior
without changing the other engine. The retained NSG kernel seeds a fixed
64-neighbor NN-Descent graph, so both segments reject builds with fewer than 65
vectors instead of entering its out-of-range path.

## Memory QG Gate 5 result

`QgSegment<RaBitQSpace<...>>` owns the space containing QG adjacency, raw
vectors, quantized neighbor codes, factors, entry point, and its search
executor. It exposes typed-tensor `build`, `open`, `search`, `batch_search`,
`save`, `stats`, `descriptor() noexcept`, and `into_any`. It satisfies
`Searchable`, `BatchSearchable`, `Saveable`, and `StatsProvider`, is reentrant
for concurrent search, and explicitly does not satisfy `Mutable`.

The segment consumes `BuildContext`, `OpenContext`, and `SearchContext`.
`Descriptor.algorithm_id` and `engine_factory_id` are the stable core `qg`
identity (`5`), and preprocessing is reported as engine-quantized. `save` uses
the logical artifact name `qg` and returns schema version 1 / format version 1;
`open` also accepts the former logical name `quant`. Both operations delegate
to the retained `RaBitQSpace` codec, so existing one-file memory QG artifacts
remain openable and save/open/save is byte-stable.

The former public `index/graph/qg/qg_builder.hpp` signature is deleted. The
retained construction implementation is
`index/graph/qg/detail/qg_builder_kernel.hpp`; legacy fallback,
materialized-view, executor, benchmark, and characterization consumers use
that detail-only kernel until their own abstraction steps.

The three legacy Python rows
`{hnsw,nsg,fusion} x quant=rabitq` deliberately retain their historical
behavior of building QG. Their current identity is `qg_segment/qg`; disabling
the independent `qg_segment` feature bit selects
`legacy_qg_model/qg`. Scalar-off and scalar-on variants use the same handoff.
This is an introspection correction, not a request-type behavior change, and
does not invent HNSW-over-RaBitQ. The complete pinned table and Gate 9 rule are
in [Memory QG legacy dispatch contract](memory-qg-legacy-dispatch.md).

Production QG construction retains historical random sources, so independent
builds are not byte-deterministic. Differential coverage therefore uses the
documented fallback: a legacy artifact is opened by `QgSegment`, save/open/save
bytes are compared, and Segment versus direct search results are checked bit
for bit. A fixed-rotator/fixed-neighbor QG v1 golden separately provides a
reproducible format hash without changing the quantization implementation.

## Vamana-memory Gate 5 result

`VamanaMemSegment` is the public float32/L2 memory model for the retained
Vamana kernels. It owns its vector buffer, a validated `VamanaReader`, and the
`VamanaGreedySearch` executor, and exposes typed-tensor `build`, `open`,
`search`, `batch_search`, `save`, `stats`, `descriptor() noexcept`, and
`into_any`. It satisfies `Searchable`, `BatchSearchable`, `Saveable`, and
`StatsProvider`, is reentrant for concurrent search, and does not satisfy
`Mutable`.

The implementation composes `vamana_builder.hpp`, `vamana_writer.hpp`,
`vamana_reader.hpp`, and `vamana_greedy_search.hpp`; it does not copy a build,
codec, or greedy-search loop. The retained builder and reader have different
in-memory shapes, so build performs one temporary writer-to-reader transfer,
deletes the temporary file after validation, and thereafter remains entirely
memory resident. All Vamana kernel headers stay at their existing paths for
the disk, DiskANN, LASER, compatibility, and later-gate consumers.

Vamana-memory format v1 is exactly the retained Vamana `graph.index` encoding
under logical name `graph` plus the existing DiskANN float32 `.fbin` encoding
under logical name `data`; `save` returns schema version 1 / format version 1
without inventing a standalone manifest file. Consequently direct
`save_graph` + `.fbin` artifacts open in the Segment, and Segment output opens
with `VamanaReader` + `VamanaGreedySearch`. Independent fixed-seed builds are
byte deterministic under the existing Vamana `thread_count=1` contract; v1
rejects other build thread counts rather than promising deterministic parallel
inter-insert scheduling.

`Descriptor.algorithm_id` and `engine_factory_id` are the frozen `vamana`
identity (`6`). The standalone C++ factory uses
`EngineFeature::vamana_memory` (the source-compatible enum alias for the
pre-provisioned `vamana_memory_segment` bit). Enabled constructs
`VamanaMemSegment`; disabled returns `not_supported`. There was no public
legacy memory-Vamana model, so the registration records legacy identity as
`none`/not applicable and never falls back to the disk adapter or another
graph engine. Vamana is intentionally absent from the 33-row Python dispatch
matrix.

## Kernel-only graph variant

Not every graph-building algorithm is a user-searchable Segment. NN-Descent is
the Gate 5 example: it produces the initial KNNG consumed by NSG, but that
build-time output and the ability of a test to pass it to a generic graph
search job do not prove a supported user index lifecycle, persistence family,
or search quality contract.

Its implementation therefore lives at
`index/graph/knng/detail/nndescent_kernel.hpp` and is registered with role
`build_kernel`. It has no `AlgorithmId`, Descriptor, Segment, or searchable
capability. Per design §7.1, a future user-facing KNNG must independently prove
those properties before gaining them. `EngineFeature::knng` is only a kernel
ownership marker: toggling it does not select a different implementation or
change NSG behavior.

## Gate 5 five-graph final state

| engine | implementation / factory key | proven capabilities | feature-off / rollback semantics |
|---|---|---|---|
| KNNG / NN-Descent | `nndescent_kernel / knng` | build kernel only; no Segment or searchable capability | `knng` bit records ownership only; no behavior switch |
| NSG | `nsg_segment / nsg` | search, batch, save/open, stats; immutable | independent bit selects recorded `hnsw_segment / hnsw` legacy behavior |
| Fusion | `fusion_segment / fusion` | search, batch, save/open, stats; immutable | independent bit selects recorded `hnsw_segment / hnsw` legacy behavior |
| memory QG | `qg_segment / qg` | search, batch, save/open, stats; immutable | independent bit selects `legacy_qg_model / qg` |
| Vamana-memory | `vamana_mem_segment / vamana` | search, batch, save/open, stats; immutable | no legacy memory factory; disabled is `not_supported` |

KNNG and Vamana-memory have no Python dispatch row; the generated 33-row
allowlist remains unchanged.

## Space physical-ownership audit

This is a read-only fact table for the design §9.7 follow-up. “Segment owns”
below means the Segment is the lifetime owner; the parenthetical names the
object that physically stores the state. No responsibility was moved as part
of this audit.

| Segment | adjacency owner | entry-point owner | vector-storage owner | §9.7 physical peel status |
|---|---|---|---|---|
| HNSW | Segment (`Graph::data_storage_` plus `OverlayGraph::lists_`) | Segment (`OverlayGraph::ep_`) | `SearchSpace` / `BuildSpace` owned by Segment | 2/3: graph and entry point are outside Space; vectors remain in Space |
| NSG | Segment (`Graph::data_storage_`) | Segment (`Graph::eps_`) | `SearchSpace` / `BuildSpace` owned by Segment | 2/3: graph and entry point are outside Space; vectors remain in Space |
| Fusion | Segment (`Graph::data_storage_` plus retained overlay when HNSW supplies it) | Segment (`OverlayGraph::ep_`, otherwise `Graph::eps_`) | `SearchSpace` / `BuildSpace` owned by Segment | 2/3: graph and entry point are outside Space; vectors remain in Space |
| memory QG | Segment transitively owns one `RaBitQSpace`; adjacency is physically in `RaBitQSpace::storage_` | physically `RaBitQSpace::ep_` | physically the same `RaBitQSpace::storage_` (plus quantized neighbor data) | 0/3 physically peeled: all three responsibilities remain co-located in Space |
| Vamana-memory | Segment (`VamanaReader::graph_`) | Segment (`VamanaReader::start_`) | Segment (`VamanaMemSegment::vectors_`), with no Space object | 3/3 for the audited responsibilities |

## Per-row registry handoff

Gate 5 migrations switch one explicit dispatch row at a time. For each row:

1. Keep the row's old factory registered as its legacy fallback and implement
   the new Segment factory behind that engine's independent feature bit.
2. Change `implementation_key` and `engine_factory_key` in
   `tools/codegen/dispatch.yaml` to the implementation that really runs. Do not
   copy the requested `index_type` when the artifact proves a different engine.
3. Run `python tools/codegen/gen.py` and review both generated files. The C++
   registration and Python identity expectation must change together.
4. Run `python/tests/client/test_index_algorithm_identity.py`. The migrated row
   must report the generated runtime keys, persist the matching algorithm
   fingerprint for scalar off and on, and turn from its documented xfail into a
   pass. Unrelated rows must not change status.
5. Disable only that engine's feature bit and run the new/legacy differential
   tests; the row must use its recorded legacy factory while other engines stay
   on their selected implementations.

HNSW is the deliberate exception to step 5. Its public legacy builder was
removed before Gate 5, so `hnsw_segment` has no runtime fallback bit and rolls
back as the Gate 0 source/git revert unit. Feature-bit rollback applies only to
rows migrated from Gate 5 onward.

## Mutation gate outcome

The executable characterization in `tests/index/hnsw_test.cpp` checks the
candidate legacy `GraphUpdateJob` before a capability can be declared.

| Operation | Connectivity | HNSW layers | In/out constraints | Search reachability |
|---|---|---|---|---|
| Build | pass | pass | pass | pass |
| Generic insert | pass | fail: the overlay and entry point never change, while a native deterministic build assigns upper levels to the same suffix | fail: reciprocal repair creates duplicate neighbors and self-edges | pass |
| Generic erase of the entry point | fail: no live entry remains | fail: the overlay retains a deleted entry point | fail: active nodes retain edges to the deleted node | active vectors remain reachable and the deleted ID is omitted |

Passing search behavior alone is insufficient. Native HNSW mutation must make
all four columns pass, including concurrent search/mutation under sanitizers,
before `insert`, `erase`, or `core::Mutable` is added.

## Standard per-graph checklist

Use these steps for searchable NSG, Fusion, memory QG, and Vamana-memory.
KNNG applies the inventory and format-freeze checks plus the kernel-only
variant above; it deliberately skips Segment lifecycle/capability steps until
independent searchability evidence exists.

1. Inventory the public builder, runtime graph/search object, persistence
   codecs, registry/codegen rows, Python binding, factory, executor tests,
   hybrid/materialized-view consumers, and benchmarks for that graph only.
2. Freeze public behavior and every existing artifact family before editing.
   Keep the checked-in golden as the comparison target; never regenerate it to
   hide a format change.
3. Introduce one `*Segment` that owns every object required to search. Put all
   algorithm parameters in a typed build-options value. `build` must return a
   fully initialized, immediately searchable segment.
4. Implement `open` as a factory returning a fully initialized segment. Keep
   the old codec as a versioned reader; do not expose default-construct plus
   `load` as the new lifecycle.
5. Implement `search` and `batch_search` against `TypedTensorView` and the
   caller-owned response. Convert internal node IDs to `core::SegmentRowId`,
   fill offsets/count/status/completeness without valid sentinels, and never
   expose graph storage in the public signature. Batch orchestration belongs in
   C++, not in a Python loop.
6. Implement `save(ArtifactWriter&, SaveOptions)` with logical artifact names
   and return a versioned `ArtifactManifest`. Reuse the old codecs byte for
   byte. Add a manifest file only when the design specifies its durable format.
7. Add static `descriptor() noexcept`, dynamic `stats()`, an AnySegment
   producer, and positive assertions for each real capability. Add negative
   assertions for tempting but unproven capabilities; never add methods that
   throw “not supported” merely to satisfy a concept.
8. Write build/open/search/batch/save contract tests plus legacy-reader and
   legacy-open tests for every raw/quantized artifact family of the graph.
9. Before declaring mutation, test insertion and deletion for live-entry
   connectivity, layer/algorithm invariants, inbound and outbound constraints,
   and search visibility. If any check fails, omit `insert`/`erase`, record the
   exact failures, and skip the mutation stress that applies only to a real
   `Mutable` segment.
10. Change the registry source and regenerate its output. Adapt the C++ index
    factory, Python factory/binding registration, high-level compatibility
    wrapper, hybrid/materialized-view code, executor tests, and benchmarks.
    If a later abstraction still needs the old representation, grant access
    through one detail-only bridge rather than a second public API.
11. Delete the old public builder/header as soon as that graph is registered as
    a Segment. Keep an implementation kernel only when another unmigrated graph
    (for example Fusion) genuinely composes it.
12. Run release build and full CTest, Python golden tests using the newly built
    extension, artifact-byte verification, old-artifact open tests, relevant
    performance sanity, and the size map. If mutation is declared, also run a
    concurrent mutation/search stress under ASan/UBSan. Commit these phases
    separately so interface, consumers, tests, and documentation remain easy
    to review and revert.
