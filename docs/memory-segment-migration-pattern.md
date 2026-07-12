<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Memory graph to Segment migration pattern

HNSW is the reference migration for abstraction step 4. The pattern is a
direct C++ API replacement: a public builder plus public `Graph` is not kept in
parallel with the Segment API.

## HNSW result

`HnswSegment<SearchSpace, BuildSpace>` now owns the searchable graph, both
spaces, and its search executor. Its public lifecycle is:

```cpp
static std::unique_ptr<HnswSegment> build(
    BuildInput, const HnswBuildOptions&, core::BuildContext&);
static std::unique_ptr<HnswSegment> open(
    core::ArtifactView, const core::OpenOptions&, core::OpenContext&);

core::SearchResult search(
    core::QueryView, const core::SearchOptions&, core::SearchSink) const;
core::BatchSearchResult batch_search(
    core::QueryBatchView, const core::SearchOptions&, core::SearchSink) const;
core::ArtifactManifest save(
    core::ArtifactWriter&, const core::SaveOptions&) const;
core::Descriptor descriptor() const noexcept;
```

The type satisfies `Searchable`, `BatchSearchable`, and `Persistable`. It does
not satisfy `Mutable`: `insert` and `erase` are absent rather than implemented
as unsupported operations. The descriptor therefore reports a sealed memory
segment.

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

Use these steps for KNNG/NSG, Fusion, memory QG, and Vamana-memory:

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
5. Implement `search` and `batch_search` against caller-owned sinks. Convert
   internal node IDs to `core::ExternalId`; never expose graph storage or node
   IDs in the public signature. Batch orchestration belongs in C++, not in a
   Python loop.
6. Implement `save(ArtifactWriter&, SaveOptions)` with logical artifact names
   and return a versioned `ArtifactManifest`. Reuse the old codecs byte for
   byte. Add a manifest file only when the design specifies its durable format.
7. Add `descriptor() noexcept` and positive static assertions for each real
   capability. Add negative assertions for tempting but unproven capabilities;
   never add methods that throw “not supported” merely to satisfy a concept.
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
