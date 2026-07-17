<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Unified WAL vocabulary: one envelope, two op families

Status: **contract** (binding on U2 lifecycle code and the G1 durable
update gate). Extends `collection-wal-coordinator.md` — Physical WAL v1
framing is normative and is not restated here.

## Why now

Two write-ahead concerns exist on the roadmap:

1. **Collection logical WAL** (shipped): row-level logical mutations
   (`write`/`erase`), receipts, publication markers, checkpoints —
   record types 1–4 in the WAL7 frame format.
2. **Segment op-WAL** (G1 gate): segment-physical, in-place operations on
   sealed LASER segments (`QGUpdater`: row patches, tombstones,
   consolidate, publish watermarks, superblock flips). Today these have
   CRC'd A/B superblocks but **no WAL and no torn-write recovery**, which
   is exactly why G1 blocks durable in-place claims.

If these two families grow independent framings/codecs, the divergence is
permanent. This contract fixes the vocabulary before U2/G1 write the code.

## Contract

### 1. One envelope

Every durable log record — both families — uses Physical WAL v1 framing:
36-byte header (`WAL7` magic, format version, record type, flags, lengths,
op ids, CRC-32 over the whole frame), `END7` trailer, prefix-safe scan,
truncate-to-verified-boundary recovery. **No second framing format may be
introduced.** Segment identity lives in the payload, not the header; the
header's id fields keep their v1 meanings (op id, batch op id).

### 2. Record-type space

| type | meaning | status |
|---:|---|---|
| 1–4 | `PREPARE`/`COMMIT`/`PUBLISH_MARKER`/`CHECKPOINT` | shipped |
| **5** | **`SEGMENT_OP`** — segment-physical operation | **reserved by this contract** |
| 6–15 | future collection-level records | reserved |
| 16+ | unassigned | — |

### 3. `SEGMENT_OP` payload

Versioned independently (leading `u16` payload version, exactly like the
mutation payload codec), then:

```
u64 segment_id     u64 segment_generation     u8 op kind     op body
```

| kind | name | body | replay rule |
|---:|---|---|---|
| 1 | `row_patch` | pid, absolute byte offset, length, bytes | absolute-offset rewrite (idempotent). Whole-row granularity is required whenever a resident-arena mirror is active (`arena_mirror_write` contract). |
| 2 | `tombstone` | pid | set-only; "resurrect" is a new insert, never an un-tombstone |
| 3 | `consolidate_begin` | consolidate epoch | phase barrier |
| 4 | `consolidate_end` | consolidate epoch | an unmatched `begin` at recovery ⇒ the consolidate replays or is discarded **as a unit** |
| 5 | `publish` | visibility watermark | monotone max on replay |
| 6 | `superblock_flip` | target A/B slot, superblock CRC | commit point of a segment checkpoint; CRC-guarded |

Gardening is not an op kind: a garden pass is a sequence of `row_patch`
records (plus `publish`), nothing more.

### 4. Idempotency invariant

Every `SEGMENT_OP` must replay idempotently. Recovery = scan forward from
the last `CHECKPOINT`, apply every complete frame in log order, stop at
the torn tail (v1 rule, unchanged). Log order is apply order; there is no
cross-family reordering — a frame's effects are visible to every later
frame regardless of family.

### 5. Codec reuse

The `SEGMENT_OP` payload codec uses `logical_wal_detail` primitives
(`put_u*/get_u*/crc32`). The byte `Decoder` currently in
`mutation_wal_codec_detail` is hoisted into `logical_wal_detail` when the
segment codec first needs it (mechanical move, no format impact). **No
third primitive set.**

### 6. What is *not* a WAL op

Rotation, seal, compact, and GC are manifest/control-record transactions
(successor-first seal protocol; durable checksummed control record).
Encoding them in the WAL as well would create a second source of truth —
forbidden. The WAL may only *reference* control state (e.g. a
`CHECKPOINT` cut after a rotation), never restate it.

### 6a. Physical file boundaries and cross-family ordering (clause J, U2-a)

The collection logical WAL and each segment op-WAL are **physically separate
files** (`collection_wal_v1/logical.wal`; `<index>.opwal`, one per segment).
The single *envelope* is shared, but there is deliberately **no cross-file
total order**: a byte offset only orders frames *within one file*. Causality
between the two families is mediated by durable control records and
checkpoints (a collection `CHECKPOINT`/manifest cut references a segment state;
a segment `superblock_flip` is a durable base), never by a global WAL sequence.
A single append-only stream that interleaves both families is therefore a
framework-scan property, not a runtime layout — the runtime never writes both
families into one file.

### 7. G1 gate, restated

No durable in-place mutation claim ships until `SEGMENT_OP` WAL replay
**and** a crash matrix (kill-point table per op kind, in the style of the
Gate 10 "four real-kill recovery cuts") land together. The A/B superblock
alone is a consistency device, not a durability claim.

## Acceptance (checked at G1)

1. Crash matrix: for each op kind × kill point, reopen converges to a
   state reachable by some prefix of the log, and a subsequent full replay
   is byte-stable (double-replay test).
2. Superblock/WAL equivalence: recovery via WAL replay and recovery via
   the surviving A/B superblock agree wherever both are defined.
3. One envelope, two families (revised per clause J, U2-a — the two families
   live in separate files, so there is no single cross-file replay order to
   assert): (a) the framework scan reads an interleaved type-1..4 / type-5
   stream back in byte order without interpreting the type, and (b) the
   collection layer is fail-closed on a foreign (`SEGMENT_OP`) record type — it
   raises a typed semantic error and leaves the WAL byte-identical, never
   silently truncating it.
