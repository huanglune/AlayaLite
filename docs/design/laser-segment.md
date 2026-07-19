<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# LASER immutable disk segment

> **Superseded / historical (2026-07-19).** This document freezes the former
> imported, rank-only adapter. Current Collection `qg`/LASER segments return
> numerically comparable scores, and Linux builds also have a separate
> writable LASER active-engine surface.

`LaserSegment` is the open-only `AnySegment` producer for an imported LASER
disk quantized graph. It composes `LaserSegmentSearcher`; the LASER beam, page
I/O, scanner, and parameter-cache loops are unchanged. The segment boundary is
entered once per query row and does not introduce per-hit calls into the
kernel.

The public segment surface is `open`, typed float32 `search` and
`batch_search`, `stats`, `descriptor() noexcept`, and `into_any()`. Its erased
operation table contains search, batch, and stats. Build, save, export,
checkpoint, freeze, close/drain, and every mutation slot are absent. Imported
native artifacts remain immutable; this adapter does not expose the existing
importer as a v3 build operation.

The descriptor records algorithm `laser` (stable ID 7), format version 1, L2,
float32, and disk medium. The format name `disk_laser_qg` follows the naming
facade and the separate wire-format contract in
[RaBitQ format contracts](rabitq-formats.md). The retired memory-QG format has
no reader and these artifacts must only reach LASER decoders.

## Rank-only result contract

The retained LASER searcher returns ordered labels and a NaN placeholder in
its legacy distance field. `LaserSegment` copies both the labels and every
score bit unchanged, declares `score_kind=rank_only`, and marks the results
approximate. It does not reinterpret, normalize, or synthesize a numeric
distance. Numeric NaN remains invalid for numeric score domains; the
Collection NaN check therefore applies only when `score_kind` is distance or
similarity.

A LASER-only segment query preserves the engine's returned rank order. A
Collection query that combines rank-only LASER hits with a numeric segment
requires an exact rerank source for those LASER rows. Without one, score
normalization returns `not_supported`; NaN is never used as a tie value or
allowed into numeric sorting.

## Registry, gate, and rollback

The disk registry current identity is `disk_laser_segment / laser`. The
explicit compatibility identity is `disk_laser_legacy / disk_laser`, which is
the existing direct `LaserSegmentSearcher` path. The independent
`disk_laser_segment` runtime feature defaults on. Turning it off makes the new
factory return `not_supported`; it does not silently select the legacy factory
and does not affect `DiskCollection` v1. A build without LASER also reports
`not_supported` at the new factory boundary.

## Manifest v2 reference publication

An importer-created segment directory can be enrolled in manifest v2 through
`publish_reference`. With the manifest-v2 writer gate off, this operation is a
no-op and creates no staging path, `ARTIFACTS.v2`, `READY`, or v2 collection
manifest. With the gate on, the transaction computes SHA-256 over the existing
native files and stages only `ARTIFACTS.v2` and `READY`; it never copies,
renames, or rewrites LASER payload files. The collection entry records
algorithm 7, factory `laser`, search/batch/stats capabilities, sealed lifecycle,
the `disk_laser_segment` reader feature, and explicit rank-only/read-only
extensions.

The dual reader remains available after the writer gate is turned off. Crash
cleanup recognizes reference-mode sidecars and removes only uncommitted
control files, never the referenced native LASER directory.
