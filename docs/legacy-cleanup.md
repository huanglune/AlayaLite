# Gate 11 legacy cleanup and reader inventory

Gate 11 reaches the `V_remove=1.2.0` boundary without changing canonical
`Collection` semantics or deleting any reader needed by an existing artifact.
API legacy, source compatibility, and persisted formats remain independent
decisions.

## Retired surfaces

- Python no longer exports `Index`, `DiskCollection`, `laser.Index`,
  `laser.RawIndex`, `vamana.build_index`, or the six `Client` index methods.
  Their import locations are tombstones that raise
  `AlayaLiteLegacyApiWarning` and point users to `Collection`.
- The native module no longer registers `PyIndexInterface`, `Client`,
  `DiskCollection`, `_CollectionReadView`, the raw LASER/Vamana builders, or
  the old memory-engine feature object.
- The public source bridges `core/compat.hpp`, `index/compat.hpp`, and
  `index/disk/disk_collection.hpp` are gone. The DiskCollection-v1 reader and
  writer remain internal at `index/disk/detail/disk_collection_v1.hpp`.
- The old Python dispatch factory, its 33 template instantiations, runtime
  templates, feature rollback fields, and legacy factory identities are gone.
  Code generation now owns only the canonical identity test matrix.

## Retained format readers

| Inventory | Reader | Why retained |
|---|---|---|
| Legacy memory graph/space layouts (HNSW, NSG, Fusion; raw and quantized variants) | `LegacyImporter` plus the v1 graph and Space loaders | Checked-in legacy corpus and user `type=index` artifacts still require canonical open/import. |
| Legacy memory RaBitQ layout | the memory-QG/RaBitQ v1 loader | It is a distinct memory wire format and remains covered by its own golden family. |
| DiskCollection-v1 Flat/Vamana/LASER layouts | internal `disk_collection_v1.hpp`, segment manifest/factory readers, and the LASER importer | Existing disk manifests and native segment payloads must continue to open through canonical migration. |
| Legacy PyIndex recovery snapshot, RocksDB scalar payload, and WAL frames | bounded readers in `LegacyImporter` | The recovery corpus includes committed records and torn-tail cases; import must preserve op-id, tombstone, and WAL-cut semantics. |
| Canonical collection WAL v1 and manifest v1/v2 | canonical collection recovery and dual-manifest readers | These are current formats, not legacy retirement candidates. |

No format reader is retired in Gate 11. The canonical importer is the retained
converter: it validates a source fingerprint, emits a canonical checkpoint,
and records an audit/activation marker without rewriting the source artifact.

Memory RaBitQ and LASER are mathematically related but never share a reader.
Their reciprocal rejection tests live in the independent Gate 11 format test.

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

The 14 artifact families are still generated and compared byte-for-byte. A
test-only native retained-v1 generator replaces the removed Python wrappers;
the LASER fixture uses a standalone native builder. Neither helper is linked
into the wheel or re-exposes a public legacy API.
