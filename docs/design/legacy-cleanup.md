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
| Retired memory RaBitQ v1 | no artifact reader; Collection recognizes `qg_segment` only to return `not_supported` plus `re-seal` | Not a current format or golden family; never shares a reader with LASER (see `rabitq-formats.md`). |
| LASER segment layout | `index/disk/laser_segment_importer.hpp` | `disk_laser_qg` is a live supported wire format with a deterministic fixture. |
| DiskFlat segment layout | `index/disk/disk_flat_segment.hpp` | The exact sealed fallback remains a current format with a deterministic golden. |
| Disk segment manifest / factory | `index/disk/segment_manifest.hpp`, `index/disk/segment_factory.hpp` | Read the current manifest-v2 disk-segment inventory. |
| Canonical collection WAL v1 and manifest v2 | canonical collection recovery + manifest reader | Current formats. |

Retired memory RaBitQ and LASER are mathematically related but never share a
reader. `rabitq_format_separation_test` mints the old bytes directly as hostile
input, wraps them in a complete LASER segment, and verifies LASER rejects the
header. There is no inverse test because the memory reader no longer exists.

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

The baseline now contains three live artifact families and compares them
byte-for-byte: `disk_flat_segment`, the standalone `laser_fixture`, and the
Collection-owned `collection_qg_laser` form. The retired `memory_qg` family has
neither a generator nor a reader. The helpers are test-only and are not linked
into the wheel.
