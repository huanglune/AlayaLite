<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Manifest v2 control plane and DiskFlat segment

Gate 6 introduces a Collection-owned manifest without changing
`DiskCollection` v1 or the three native DiskFlat files. The implementation is
internal C++ infrastructure; Python dispatch and the five memory-segment
families are unchanged.

## Owned schema

`ArtifactManifestV2` lives in `index/collection/artifact_manifest_v2.hpp`.
Every string, feature name, path, extension and digest is owned by the
manifest. It does not retain a `string_view`, `span`, or pointer into an engine
buffer. SHA-256 is represented by `Sha256Digest`, exactly 32 bytes, and every
v2 artifact records `checksum_algorithm=sha256`; the legacy core
`Artifact::checksum` field is not reused or reinterpreted.

The deterministic line-oriented encoding has `version=2`. Strings are
hex-encoded so delimiters, empty strings and non-ASCII logical values round
trip without quoting rules. Unknown unnamespaced keys are rejected. The owned
model records:

| area | durable fields |
|---|---|
| collection schema | schema name/version, dimension, metric, scalar type, LogicalId encoding/version, metadata checkpoint and epoch |
| publication | generation, publication parent, next-segment hint |
| recovery cuts | collection and per-segment WAL cut, row-version range and ID-map checkpoint |
| segment identity | segment ID/generation, role, lifecycle (`active`, `successor`, `sealed`, `retired`, `gc_pending`), algorithm ID, native format version and factory key |
| capability snapshot | operation bits and the complete concurrency profile |
| artifact inventory | logical name, relative path, required/optional bit, size, SHA-256, READY bit and reader compatibility |
| retention and GC | per-segment source retention plus GC phase/generation/pending IDs/retained sources |
| compatibility | min/max reader version and owned required-feature names at segment and artifact granularity |

The extension maps preserve explicitly namespaced legacy data during a v1
mapping and allow forward additions without moving required fields.

## Five-step publication transaction

`ArtifactControlPlaneTransaction` is the Collection wrapper around the frozen
core `ArtifactWriter` path map. It implements the control-plane sequence:

1. validate the `BuildContext` deadline/cancellation, reserve staging memory
   and I/O credits, and create `.alaya_staging/<transaction>`;
2. resolve all logical artifact names before the engine writes, or adopt the
   directory atomically produced by a retained disk builder;
3. fsync every required file, compute its 32-byte SHA-256, and build the owned
   artifact inventory;
4. when the v2 writer gate is on, write and fsync `ARTIFACTS.v2`, then write a
   `READY` marker bound to that owned-manifest digest and fsync the staging
   directories;
5. rename the payload directory without overwrite, fsync its parent, and only
   then atomically replace `collection_manifest.txt`. Routing changes at the
   final manifest replacement, never at payload creation.

Failures before payload rename leave only staging state. A failure after
payload rename but before manifest replacement leaves a READY, transaction-
owned orphan that is not routable. Normal unwinding uses eager cleanup;
crash/error-injection tests can retain state to model process death.
`cleanup_orphans` is called while the Collection holds exclusive control-plane
ownership: it removes interrupted staging directories and removes only
unreferenced final directories that contain both `ARTIFACTS.v2` and `READY`.
It never selects a legacy directory. Once the manifest replacement succeeds,
the transaction is no longer abortable; a later directory-fsync warning must
not delete a now-routed payload.

The manifest-v2 writer feature is independent and defaults off. With it off,
the same staging/rename discipline publishes native files but creates no
`ARTIFACTS.v2`, `READY`, or v2 collection manifest. Existing memory saves and
their artifact bytes are not routed through this writer.

## Dual read and roll-forward

`CollectionManifestDualReader::open` is the unified entry point.

- A v1 `collection_manifest.txt` is parsed by the retained
  `CollectionManifest`/`SegmentManifest` readers and mapped into the v2-shaped
  view. Missing metadata epoch/checkpoint, publication generation/parent,
  recovery cuts, capability snapshot, READY/SHA, reader compatibility and GC
  fields receive explicit values and an `ExplicitManifestDefault` entry. A v1
  artifact is never falsely marked READY or assigned a fabricated digest.
- A v2 manifest reconstructs every owned field, checks reader-version/features,
  validates `READY` against its recorded SHA-256 and owned-manifest digest, and
  checks every artifact's existence, size and SHA-256 before returning it for
  routing.

Disabling the writer only blocks new v2 publications. The v2 reader remains
available. Once v2 has been published, the supported policy is roll-forward
with this reader or a converter; old binaries are not promised downgrade
access.

## DiskFlat exact oracle

`DiskFlatSegment` composes `DiskFlatBuilder` and
`DiskFlatSegmentSearcher`. It exposes typed float32 `build`, `open`, exact
single/batch `search`, byte-preserving `save`, `stats`, `descriptor() noexcept`,
`into_any()` and `export_rows`. Its descriptor uses core algorithm `flat` (1),
native format 1, disk medium and the native metric. Its AnySegment capability
set is search, batch, save, export and stats; it has no mutation method or
mutation capability.

The export request supplies a lifetime owner and batch size. The stable cursor
keeps the mmap searcher alive and yields u64 logical labels, a float32 typed
tensor view over stored row bytes, and one empty metadata reference per row.
Collection remains the metadata owner. For cosine artifacts the exported
vectors are the native, L2-normalized stored representation; L2/IP export bytes
are the builder input bytes.

Build invokes the retained builder inside transaction staging. Therefore
`manifest.txt`, `ids.u64.bin` and `vectors.f32.bin` are byte-identical to a
direct builder invocation. Save copies and fsyncs those native files without
rewriting them. Legacy searchers open Segment output, and the Segment opens
both legacy v1 collections and v2 collections through the dual reader.

The disk registry records `disk_flat_segment/flat` as the new identity and
`disk_flat_legacy/disk_flat` as the explicit compatibility factory. The
`disk_flat_segment` feature defaults on; turning it off makes the new factory
return `not_supported` and does not redirect or modify `DiskCollection` v1.

