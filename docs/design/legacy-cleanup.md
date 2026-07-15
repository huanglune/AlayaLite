# Gate 11 legacy cleanup and reader inventory

The legacy cleanup ships in AlayaLite **1.1.0** as a single hard cutover. The
architecture/engineering rewrite lands on top of the 1.0 line and does **not**
preserve backward compatibility with pre-1.1 artifacts. There is no
deprecate-at-1.1 / remove-at-1.2 window: the legacy Python/native API **and**
the pre-rewrite on-disk import path are both removed in 1.1.0.

> Historical note: earlier drafts of this document staged the API removal at a
> separate `V_remove=1.2.0` and pledged to retain every format reader. Both
> decisions were superseded. The version collapsed to a single 1.1.0 release,
> and the post-cutover engineering wave physically removed the pre-rewrite
> backward-compat readers listed under "Removed backward-compatibility readers"
> below.

## Retired surfaces

- Python no longer exports `Index`, `DiskCollection`, `laser.Index`,
  `laser.RawIndex`, `vamana.build_index`, or the six `Client` index methods.
  Their import locations are tombstones that raise
  `AlayaLiteLegacyApiWarning` and point users to `Collection`.
- The native module no longer registers `PyIndexInterface`, `Client`,
  `DiskCollection`, `_CollectionReadView`, the raw LASER/Vamana builders, or
  the old memory-engine feature object.
- The public source bridges `core/compat.hpp`, `index/compat.hpp`, and
  `index/disk/disk_collection.hpp` are gone. The internal DiskCollection-v1
  reader/writer (`index/disk/detail/disk_collection_v1.hpp`) was also removed;
  the canonical `Collection` opens only its own manifest-v2 layout.
- The old Python dispatch factory, its 33 template instantiations, runtime
  templates, feature rollback fields, and legacy factory identities are gone.
  Code generation now owns only the canonical identity test matrix.

## Removed backward-compatibility readers

The pre-rewrite import path is gone. 1.1.0 cannot open artifacts produced by
the pre-1.1 Python/native indexes, and this is intentional — there is no
compatibility promise across the rewrite.

| Removed reader | Path (deleted) | Note |
|---|---|---|
| Collection-level legacy PyIndex importer | `index/collection/legacy_importer.hpp` | Decoded schema/CURRENT/raw snapshots, read-only RocksDB checkpoints, and complete-COMMIT WAL prefixes into a sealed DiskFlat oracle. |
| DiskCollection-v1 reader/writer | `index/disk/detail/disk_collection_v1.hpp` | Old on-disk collection manifest/segment layout. |
| RocksDB scalar payload + dependency | `storage/rocksdb_storage.hpp`, `storage/scalar_data.hpp` | RocksDB is no longer a dependency; scalar state lives in the Collection checkpoint. |
| Legacy recovery corpus + matrix test | `tests/collection/legacy_importer_test.cpp` and the checked-in `legacy_recovery_corpus` fixtures | The pinned recovery variants were retired with the importer. |

The collection manifest reader is now v2-only and `Collection::open` has no
importer fallback: a directory without a canonical manifest-v2 layout returns
`not_found`.

## Retained current-format readers

These are not backward-compatibility shims. They read formats the current build
still produces and are pinned by the golden families.

| Inventory | Reader | Why retained |
|---|---|---|
| Memory graph/space segment layouts (HNSW, NSG, Fusion; raw and quantized) | the graph/Space loaders in `index/graph/detail/memory_graph_segment.hpp` | Current canonical memory-segment serialization; covered by the memory golden families. |
| Memory RaBitQ layout | the memory-QG/RaBitQ v1 loader in `index/graph/qg/qg_segment.hpp` | A distinct memory wire format with its own golden family; never shares a reader with LASER (see `rabitq-formats.md`). |
| LASER segment layout | `index/disk/laser_segment_importer.hpp` | `disk_laser_qg` is a live supported wire format with a deterministic fixture. |
| Disk segment manifest / factory | `index/disk/segment_manifest.hpp`, `index/disk/segment_factory.hpp` | Read the current manifest-v2 disk-segment inventory. |
| Canonical collection WAL v1 and manifest v2 | canonical collection recovery + manifest reader | Current formats. |

Memory RaBitQ and LASER are mathematically related but never share a reader.
The independent `rabitq_format_separation_test` wraps a real memory-QG file in
a complete LASER segment and verifies LASER open rejects its header; it also
feeds the checked-in LASER fixture index to memory `QgSegment::open` and
verifies the RaBitQ v1 header gate rejects it before allocation or rotator
construction.

## Utils and lint scope

The only pure forwarding RaBitQ utility header,
`rabitq_utils/search_utils/epoch_visited.hpp`, was removed. The dead legacy
`quantization_type.hpp` was removed with the native Index factory. Blanket
`NOLINTBEGIN`/`NOLINTEND` regions were removed from retained utils; narrow
line-level suppressions with an actual platform or intrinsic reason remain.

Production implementations such as `math.hpp`, `locks.hpp`, `random.hpp`,
`thread_pool.hpp`, metadata/scalar utilities, and the RaBitQ kernels stay at
their current paths. Moving those owners is a separate physical-layout
project. `test_legacy_cleanup_lint.py` prevents the deleted source paths,
native registrations, old codegen schema, and blanket utils suppressions from
returning.

## Golden generation after API removal

The 12 live current-format artifact families are generated and compared
byte-for-byte. The dead DiskCollection-v1 `disk_flat` and `disk_vamana`
collection layouts are not part of the 1.1.0 baseline because their only
writer was removed. A focused test-only native generator exercises the current
HNSW/NSG/Fusion segment `build` + `save` paths, and the LASER fixture uses its
standalone native builder. Neither helper is linked into the wheel or exposes a
legacy API.
