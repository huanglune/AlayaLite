# Collection / index semantics baseline

This document freezes the observable state of the public index APIs before the
index/segment abstraction migration.  It is a description of the current tree,
not the target contract.  A check mark means that the operation is exposed by
the named public object; “adapter” means that another object supplies it.

## Capability matrix

| Public type | insert | erase | save | open/load | filter | search | concurrency today |
|---|---:|---:|---:|---:|---:|---:|---|
| `Index` backed by HNSW | yes | yes | yes | yes | native only when scalar storage is enabled | yes | batch build/search accept thread counts; mutation/search safety is not promised by the Python API |
| `Index` backed by NSG | adapter: generic `GraphUpdateJob` | adapter | yes | yes | as above | yes | same facade behavior; NSG does not publish an algorithm-specific online-update protocol |
| `Index` backed by Fusion | adapter: generic `GraphUpdateJob` | adapter | yes | yes | as above | yes | same facade behavior; Fusion is a builder composition, not a distinct persisted graph format |
| KNNG / `NndescentImpl` | no stable public mutation API | no | graph result can be saved | graph result can be loaded | no | no direct SDK object | build accepts a thread count; it is an NSG build tool |
| memory QG (`index/graph/qg`) | no | no | yes, one logical `qg` artifact | yes | legacy adapter when scalar storage is enabled | yes | immutable `QgSegment`; reentrant search-only profile |
| Vamana-memory `VamanaMemSegment` | no | no | yes, logical `graph` + `data` | yes | no | single/batch | immutable, reentrant search-only profile |
| `DiskANNIndex` static | no | no | `build` creates a file family | `load` | no | single/batch/pipeline | concurrent searches use a bounded scratch pool |
| `DiskANNIndex` updatable | yes | yes | `flush_pages` / `flush` | `load(updatable=true)` | no | single/batch/pipeline | dark-until-publish inserts, tombstone snapshots and internal locks define visibility; no SDK-wide guarantee |
| `DiskCollection(disk_flat)` segment | append to pending, then new segment | no | `flush` publishes | `open` | no | yes | published segments are immutable; collection lock is process-exclusive; searches may run concurrently |
| `DiskCollection(disk_vamana)` segment | append to pending, then new segment | no | `flush` publishes | `open` | no | yes | same immutable publication model |
| `DiskCollection(disk_laser)` segment | no (`add` is unsupported) | no | import only | `open` | no | yes | imported segment is immutable; searcher is intended for concurrent search |
| high-level `Collection` | yes / upsert | yes by item id/filter | yes | yes | yes | vector, hybrid and scalar | delegates to one memory `Index`; no cross-operation transaction or thread-safety contract |

The 33 generated memory combinations retain HNSW/NSG/Fusion as the legacy
declared names. Three `quant=rabitq` rows actually select memory QG and report
`qg_segment/qg`; KNNG, Vamana, DiskANN and the three `DiskCollection` engines
do not go through that dispatch table.

## State and visibility

### Memory `Index` and high-level `Collection`

`python/include/index.hpp::PyIndex<Runtime, Space>` owns the Segment or legacy
runtime, vector/quant spaces, jobs, scalar RocksDB storage, and (when a
recovery root can be derived) `RecoveryManager`. Recovery is therefore
attached to the Python adapter, not to a C++ Collection abstraction.

For a successful memory mutation today:

| term | current observable point |
|---|---|
| acknowledged | the Python/native mutation call returns |
| searchable | the graph/vector mutation has been applied before that return; there is no separate receipt or routing publication state |
| durable | **not implied by return**. WAL PREPARE/COMMIT streams are flushed, but the public API exposes no durability level/receipt and filesystem durability is not unified with graph/RocksDB state |
| snapshot-recovered | a published recovery `CURRENT` points at a snapshot manifest containing graph/data/optional quant and optional RocksDB checkpoint; committed WAL records after its watermark are replayed on load |

Snapshots are made after fit, on manual save, after load, and after recovery
when needed.  The WAL represents logical insert/upsert/remove mutations.  This
mechanism does not cover `DiskCollection` or `DiskANNIndex`.

### `DiskCollection`

`add` only acknowledges copying vectors and labels into a pending buffer.  The
rows are neither included in `size()` nor searchable.  `flush` de-duplicates,
builds a new immutable segment, atomically renames/publishes it, then atomically
updates `collection_manifest.txt`; directory fsync is attempted.  A failed
flush retains pending input until commit and may leave a classified orphan.
There is no operation WAL, delete/update replay, or searchable/durable receipt.
LASER has no pending-build path: `import_laser_segment` is its only publication
operation.

### `DiskANNIndex`

DiskANN has an independent build/load lifecycle and is not a
`SegmentSearcher`.  In update mode, a new slot remains dark until its vector,
edges and label mapping are ready; deletion uses tombstones and reconnect work.
`flush_pages`/`flush` persist engine pages/metadata.  It has no Collection WAL,
idempotent logical replay protocol, seal/export protocol, or shared definition
of acknowledged/searchable/durable.

## Persistence inventory

| owner | current artifact family |
|---|---|
| memory facade | graph file, data file, optional quant file; memory QG retains its one-file `rabitq.data` family; Vamana-memory retains `graph.index` plus DiskANN `.fbin` vectors; recovery additionally has `CURRENT`, snapshot manifest/files, RocksDB checkpoint and WAL |
| generic `Graph` | graph header/body plus optional overlay graph data |
| DiskFlat segment | `manifest.txt`, `ids.u64.bin`, `vectors.f32.bin` |
| Vamana segment | Flat files plus `graph.index` |
| LASER segment | imported LASER index and sidecars, external ids file, segment manifest |
| DiskCollection | `collection_manifest.txt`, `.lock`, `segments/seg_NNNNNNNN/**` |
| DiskANN | `meta.bin`, `diskann.index`, `ids.bin`, cache files, optional PQ pivots/codes, and update metadata such as tombstones/slots when applicable |

The machine-readable format hashes and parsed fields are maintained by
`scripts/golden/generate_artifact_baseline.py`; hashes deliberately freeze byte
format, while parsed fields make failures diagnosable.

## Golden coverage map

“Shared” means the concrete algorithm reaches the same generic Graph/Space
serializer through `PyIndex`; a representative byte corpus is used rather than
checking in 33 redundant copies.

| type | Python behavior | C++ compile surface | persistence format |
|---|---:|---:|---:|
| HNSW memory | yes (`Index`) | yes | yes, raw and SQ8 graph/data/quant representatives |
| NSG memory | facade behavior + 33-row dispatch assertion | header/codegen construction list | shared generic Graph/Space format |
| Fusion memory | facade behavior + 33-row dispatch assertion | header/codegen construction list | shared generic Graph/Space format |
| KNNG | documented (not a public Python Index) | detail NN-Descent build kernel | no independent user-index artifact family |
| memory QG | three pinned legacy declarations, scalar off/on, and feature fallback | `QgSegment` lifecycle/capability/TSan tests | independent reproducible memory QG v1 family |
| Vamana-memory | no Python dispatch row | `VamanaMemSegment` lifecycle/capability/TSan tests | independent `graph.index` + `vectors.fbin` family |
| DiskANN | no public Python entry | direct `DiskANNIndex` instantiation | complete small DiskANN directory |
| DiskCollection Flat | yes | direct collection/searcher types | collection + Flat segment |
| DiskCollection Vamana | common facade behavior; engine-specific existing tests | direct collection/Vamana surface | collection + Vamana segment |
| DiskCollection LASER | common facade behavior; engine-specific existing tests | LASER build closure remains in existing CTest | deterministic LASER fixture/sidecars |
| high-level Collection | yes | `PyIndex` is Python-layer C++, documented as such | memory artifact + recovery inventory |

Regenerate and compare artifacts with:

```bash
cmake --build build/Release --target artifact_diskann_generator \
  artifact_memory_qg_generator artifact_memory_vamana_generator
PYTHONPATH=python/src:build/Release/python \
  python scripts/golden/generate_artifact_baseline.py
```

Use `--write` only when intentionally accepting a format change.  The size map
is regenerated with `python scripts/size_map/generate_size_map.py`; pass
`--wheel PATH` to include the packaged wheel.  During steps 4–11, run behavior
golden before/after compatibility-wrapper changes, require old artifact open
and hash/field review for serializer changes, compile this header surface, and
compare the same-toolchain size map for added/removed template instances.
