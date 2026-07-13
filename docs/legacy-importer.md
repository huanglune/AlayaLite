<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Legacy PyIndex importer

## Scope and non-destructive rule

`LegacyImporter` is the Gate 7-B, Collection-internal importer required by
design §6.4. Its source is the legacy PyIndex layout and its target engine is a
sealed `DiskFlatSegment`. The importer never constructs `RecoveryManager`,
`WriteAheadLog`, `PyIndex`, or another legacy reader. In particular, it never
lets the old recovery path publish a `post_recovery` snapshot or delete
`recovery/wal.bin` before import.

The source-side operations are limited to regular-file reads and RocksDB
`DB::OpenForReadOnly` on the checkpoint named by the snapshot manifest. The
importer reads the original `schema.json`, `recovery/CURRENT`, snapshot
`manifest.txt`, `data.snapshot`, RocksDB checkpoint, and raw `wal.bin` itself.
It fingerprints every legacy file before decoding and again before marker
publication. A mismatch prevents the switch. The corpus test also checks each
file against its checked-in size and SHA-256 after every successful,
interrupted, and repeated import.

This contract is the implementation of design §6.4 and extends the
[Collection WAL coordinator contract](collection-wal-coordinator.md#gate-7-b-importer-seam).

## State machine

The only writer sequence is:

```text
discover legacy layout
  -> read schema + CURRENT and validate snapshot framing
  -> validate/read the named RocksDB checkpoint read-only
  -> scan the dtype-specific raw WAL to its verified prefix
  -> reserve minimum_next_op_id above every complete source frame
  -> fsync and atomically publish migration intent
  -> build the sealed DiskFlat payload
  -> write the Collection checkpoint and final manifest v2
  -> write the import audit
  -> fsync and atomically replace the ACTIVE marker
```

WAL scanning implements the legacy framing locally: little-endian `HEAD`,
version/type/mutation, op ID, bounded payload, and `TAIL`. PREPARE payloads are
held until a complete, matching COMMIT is seen. A short header/payload/trailer,
bad magic/version/type, or implausible length ends the verified prefix. No
resynchronization is attempted, and a complete PREPARE without a complete
COMMIT is not imported. The largest complete frame still reserves its source
op ID even when it is uncommitted, which is why the torn-tail corpus has
`wal_cut=1` and `minimum_next_op_id=3`.

Snapshot rows keep their native float32/int8/uint8 byte representation in the
Collection checkpoint. int8 and uint8 values are exactly promoted to float32
only for the sealed Flat artifact. A read-only adapter converts source-typed
queries to the Flat query type; `get_by_id` therefore returns the original
bytes while both direct Flat search and `SegmentedCollection` search use the
deterministic exact oracle. Numeric legacy IDs remain canonical legacy-u64
logical IDs. Collection item IDs, documents, and all metadata variants are
read from the checkpoint/WAL and stored in the ID map/version graph.

The old snapshot has only an aggregate applied-through ID. For an index,
sequential trailing inserts make their source sequence recoverable. A scalar
checkpoint can contain a committed deletion after RocksDB has removed the only
external-ID mapping. `minimum_visibility_watermark` preserves that committed
cut without inventing an ID or tombstone. WAL-carried remove/upsert operations
retain exact source op IDs and produce the expected tombstone/latest-version
suppression.

## Files, intent, audit, and marker

The target layout is:

```text
<target>/
  collection_manifest.txt
  segments/seg_00000001/             # DiskFlat native + v2 ownership files
  .alaya_internal/collection_wal_v1/ # checkpoint + CURRENT + one CHECKPOINT frame
  .alaya_internal/legacy_import_v1/
    intent.v1
    audit.v1
    ACTIVE
```

The old WAL is decoded into versions; it is never copied into `logical.wal`.
The new logical WAL contains only the G7-A CHECKPOINT frame, so enabling the
importer cannot reopen a second PyIndex mutation log.

`intent.v1` binds the source fingerprint/schema/snapshot, maximum observed
source op ID, and reserved next op ID before target artifacts are written.
`audit.v1` is retained after the switch and records:

- source fingerprint, file count, and byte count;
- source kind, dtype, ID width, dimension, and scalar mode;
- snapshot identity/cut, maximum seen and committed op IDs, and next-op floor;
- allocated/live/tombstone counts, committed WAL record count, torn-tail bit,
  and verified WAL byte count;
- RocksDB access mode, checkpoint name/WAL cut, Flat segment identity, and
  final manifest SHA-256.

`ACTIVE` is the routing decision. It contains the source fingerprint, audit
SHA-256, and a checksum over its complete prefix. It is written to a temporary
file, fsynced, atomically replaced, and followed by a parent-directory fsync.
A missing, truncated, extra-field, checksum-invalid, or audit-invalid marker
is unswitched and selects the legacy route. A valid marker selects only the
new route. After that decision, opening needs only the target checkpoint,
manifest, audit, and Flat artifact; loss or unavailability of the old source
does not request rollback.

## Idempotence, failure, and gates

`CollectionFeatureFlags::legacy_importer` independently controls new discovery
and pre-marker continuation. It does not control `wal_coordinator`, does not
enable the legacy PyIndex WAL, and does not disable readers for an already
valid marker. Calling import after a valid marker is an open-only idempotent
operation.

Before the marker, restart may remove only importer-owned target files and
rebuild them from the unchanged source. An intent is required before existing
target artifacts can be treated as importer-owned, preventing accidental
cleanup of another collection. After the marker, cleanup is forbidden and
recovery is roll-forward only.

The injected restart matrix covers RocksDB validation, mid-WAL scan, op-ID
reservation, intent publication, checkpoint publication, immediately before
the marker, and immediately after it. Every pre-marker cut selects the intact
legacy route and completes on restart. The post-marker cut opens the new route
and retains the audit. A separately written partial final marker is also
treated as unswitched.

## Supported legacy variants

Gate 7-B deliberately accepts only the pinned raw, little-endian, L2,
quantization-none format represented by the six recovery corpus cases:
float32/int8/uint8, uint32/uint64, and scalar off/on. Unknown dtype, ID width,
metric, quantization, unsafe path, snapshot version, malformed storage header,
or inconsistent scalar checkpoint stops with corruption/not-supported
evidence. It does not fall through to the destructive old reader.

## Corpus evidence

| Case | Terminal state | WAL result | Exact query | Source SHA |
|---|---:|---|---|---|
| `f32_u32_insert_clean` | 8 live | snapshot cut 2 | pass | unchanged |
| `i8_u64_insert_remove_wal` | 6 live, 1 tombstone | 2 commits | pass | unchanged |
| `u8_u32_snapshot_insert_wal` | 8 live | snapshot cut 1 + commit 2 | pass | unchanged |
| `f32_u64_torn_tail` | 7 live | commit 1 only; op 2 reserved | pass | unchanged |
| `i8_u32_collection_upsert_wal` | 4 live, 5 allocated | upsert commit 1 | pass | unchanged |
| `u8_u64_collection_delete_clean` | 4 live, 5 allocated | snapshot cut 1 | pass | unchanged |

The test fixes every live vector's SHA-256, all live logical IDs, metadata and
documents, source version/tombstone state, exact top-1 results through both
Flat and Collection, checkpoint/manifest linkage, the single new WAL frame,
two consecutive imports, and a mid-state restart for each case.
