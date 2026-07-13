<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Gate 6 disk-segment final state

Gate 6 closes the internal disk-engine migration without switching the public
Python facade or adding a mutable disk protocol. The three new producers are
immutable `AnySegment` implementations owned by their existing disk codecs.

## Three-engine final state

| engine | implementation / factory key | proven capabilities | feature-off / rollback semantics | format contract |
|---|---|---|---|---|
| Flat | `disk_flat_segment / flat`; legacy `disk_flat_legacy / disk_flat` | exact search, batch, transactional save, export, stats; immutable and reentrant | `disk_flat_segment=false` makes only the new factory return `not_supported`; source can be reverted independently; legacy `DiskCollection` is unchanged | native format 1: `manifest.txt`, `ids.u64.bin`, `vectors.f32.bin`; optional v2 control files wrap but do not rewrite native bytes |
| Vamana | `disk_vamana_segment / vamana`; legacy `disk_vamana_legacy / disk_vamana` | approximate L2 search, batch, transactional save, stats; immutable and reentrant | `disk_vamana_segment=false` removes only the new factory surface; commit `7e2aaaa` plus `766ddff`/`91d077a` is the tested source rollback unit; legacy Vamana remains | native format 1: Flat files plus `graph.index`; L2 only; optional v2 control files preserve every native byte |
| LASER | `disk_laser_segment / laser`; legacy `disk_laser_legacy / disk_laser` | open, rank-only search, batch, stats, read-only reference publication; immutable and reentrant | `disk_laser_segment=false` removes only the new factory surface; a build without LASER is also `not_supported`; legacy direct search remains | `disk_laser_qg` format 1: importer manifest, ids, index and sidecars; v2 publication checksums the existing payload without copying or rewriting it |

Manifest v2 final state: `CollectionManifestDualReader` remains enabled for v1
and v2, while every `manifest_v2_writer` feature bit defaults off and gates
only new v2 publications.

## Release-note material

- Added immutable internal Segment adapters for DiskFlat, disk Vamana and
  imported LASER, each with an independent runtime factory gate and explicit
  legacy factory identity.
- Added Collection manifest v2 with owned SHA-256 artifact inventories,
  READY-bound five-step publication and permanent dual-read compatibility;
  the v2 writer remains disabled by default.
- Added real heterogeneous Collection routing. HNSW, DiskFlat and DiskVamana
  numeric hits merge in one snapshot against a standalone DiskFlat exact
  oracle; LASER rank-only output requires an exact rerank source or is rejected
  when mixed with numeric scores.
- Added host-native and fixed AVX2+FMA release lanes with same-lane artifact
  reproducibility and strict cross-lane Flat-family equality.

There is no public API cutover in this gate. Disk mutation coordination,
DiskANN mutable routing, filter pushdown and the Python Collection entry point
remain later-gate work.

## Design §9.8 closure checklist

| §9.8 item | result | evidence |
|---|---|---|
| AnySegment/Collection stable; staging writer and owned SHA available | complete | contract v3 and Gate 4 Collection tests; manifest/control-plane commits `d12b6bd`, `82f5c20`, `c31c67f` |
| Flat immutable Segment and exact-oracle/export contract | complete | `575de71`, `e491d74`; `DiskFlatSegment.DifferentialBytesSearchExportAndLegacyBidirectionalOpen` and `ManifestV2GatePublishesAndReaderSurvivesRuntimeDisable` |
| Disk Vamana immutable Segment and explicit L2 gate | complete | `7e2aaaa`, `766ddff`, `91d077a`, `98869de`; differential bytes/search, crash cuts, checksum damage and recall tests |
| LASER open-only/rank-only Segment, build on/off and factory gate | complete | `932d5d9`, `e57ddf8`, `cc927b2`, `ad9ff36`, `9dc97d8`; direct differential search, standalone exact-reranked route and numeric-mix rejection |
| Legacy v1 manifest/artifact open | complete | manifest v1 mapping plus bidirectional Flat/Vamana open and LASER legacy differential open |
| New/legacy differential behavior and native bytes | complete | Flat, Vamana and LASER Segment suites; 14-family checked golden inventory |
| READY/checksum and build/open/save crash cuts | complete | `manifest_v2_test`, Flat/Vamana durability tests and the three-engine roll-forward drill |
| Numeric/rank-only mixing rule | complete | numeric three-engine Flat-oracle test; LASER mixed request returns `not_supported`; LASER-only route succeeds with exact rerank |
| Cross-segment version suppression | complete | heterogeneous test registers stale and tombstoned disk rows; no duplicate or resurrected LogicalId reaches top-k |
| Batch short-result and partial failure contract | complete | all three numeric engines satisfy offsets/count/status/completeness under `top_k > rows`; injected DiskFlat per-query I/O failure marks only that query failed |
| Runtime disable | complete | all three new factories off with writer off leaves all legacy factories usable; each one-engine-off row leaves the other two new factory surfaces enabled |
| Code rollback | complete | disposable branch reverted the Vamana source/test/golden unit while retaining the v2 reader; clean Release build and 106/106 remaining CTests passed; branch was deleted and never pushed |
| On-disk roll-forward | complete | one v2 collection manifest contains Flat, Vamana and LASER entries; all reopen with writer default off; one flipped Flat payload byte is rejected as corruption by SHA-256 |
| Cross-platform build/artifact matrix | complete | host-native and fixed AVX2+FMA full Release builds; 12/12 disk-segment subset per lane; two generations per lane identical; Flat and disk Vamana families matched across these lanes |

The local native lane additionally produced stable but different DiskANN,
LASER-fixture and memory-QG bytes from the fixed AVX2 checked baseline. Those
non-Gate-6 families are reported by the matrix tool and are not normalized or
silently accepted as new golden data.
