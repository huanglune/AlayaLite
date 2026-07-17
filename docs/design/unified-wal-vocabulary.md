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
| 7 | `label_bind` | txid, row_op_id, pid, pid_generation, label | (2A) stages one explicit pid→label binding under a txid; buffered, promoted atomically by the matching `tx_publish`. Idempotent de-dup on `(txid, row_op_id)` for a torn large-bundle retry. |
| 8 | `tx_publish` | txid, new_committed_watermark, row_count, applied_collection_op_id | (2A) the single durable (fsync) commit point of a label bundle: advances `committed`, promotes the bundle's staged `label_bind` frames, and carries the collection op watermark that is 2B's idempotency basis. `batch_id == tx_id`; strict-increasing txid; non-regressing applied op. |

Gardening is not an op kind: a garden pass is a sequence of `row_patch`
records (plus `publish`), nothing more.

#### 3a. `consolidate` as a maintenance transaction (2C / W1)

Under `enable_wal`, `consolidate()` is a single maintenance transaction bracketed
by a durable `consolidate_begin(epoch)` … `consolidate_end(epoch)` pair (kinds 3/4).
Its body is whole-page `row_patch` (kind 1) after-images produced in a **private
overlay** that may spill to the *same* op-WAL when it exceeds the page-cache cap;
nothing touches the index or the resident arena before the transaction commits.

- **No-steal.** Between the durable `begin` and the durable `end` no index/arena
  write may happen — the only durable maintenance store is the overlay → op-WAL.
  A last-line guard in the index write path poisons on any violation.
- **Commit point.** The `end` fsync is the single commit point. Only after it does
  the transaction install every touched page's final image into the index (under a
  per-page seqlock so a concurrent search never copies a half-installed page).
- **Incomplete epoch ⇒ rollback (S_old).** No durable `end` ⇒ recovery discards the
  epoch and semantically truncates the op-WAL back to the `begin` boundary; the index
  is untouched, so the segment is exactly its pre-consolidate state.
- **Complete epoch ⇒ roll-forward (S_new).** A durable `end` makes `begin` + every
  after-image + `end` a durable prefix; recovery redoes all latest-per-page images
  over an index that may hold any old / new / half / partial subset of the pages
  (whole-page redo repairs it). The install index writes are themselves unforced.
- **Free-list.** Reclaimed rows carry `FREE | TOMBSTONE` trailers inside the kind-1
  after-images. Recovery re-derives the free set from the final trailers and rebuilds
  the canonical (ascending-PID) free chain once, in the single post-replay convergence
  point — byte-stable across repeated recovery, and identical whether or not a reopen
  intervened.
- **Epoch state machine.** A `consolidate_begin`/`consolidate_end` epoch and a
  canonical label bundle form mutually-exclusive replay lanes; any other op kind
  inside an open epoch poisons.

`garden()` still throws under `enable_wal` (it is *not* a kind 3/4 epoch — see §5.1 of
the 2C design). PID reuse is likewise gated: the `label_bind.pid_generation` field is
reserved and is currently always written 0 (every WAL insert appends a fresh PID);
bundle-only reuse over a v3 `pid_generation` base is designed but not yet enabled.

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

### 8. Active engine — the mutable LASER segment as a Collection generation (2B)

A Collection's **active (writable) generation** may be a durable on-disk mutable
LASER segment (`active_engine=laser`) instead of the in-memory flat table. The
label-transaction op-WAL (kinds 7/8) is then the physical durability layer; the
Collection logical WAL (types 1–4) remains the source of truth. The bridge is the
facade-private `MutableLaserCollectionAdapter`, and the wiring obeys:

- **Physical txid = the real logical-WAL transaction id.** A single or per-row
  transaction uses the row's own `op_id`; an atomic batch uses its shared WAL frame
  id (`batch_op_id`). It is **never** the per-row batch's shared `batch_op_id` — a
  per-row batch is N independent transactions, so sharing it would satisfy
  `txid <= last_committed` for every row after the first and drop the write. The txid
  and the transaction's max row `op_id` are passed *typed* through `MutationContext`
  (set by Collection at its three dispatch sites), never re-derived by the adapter.
- **Idempotency basis.** A write is skipped iff `max_row_op_id <=
  applied_collection_op_id` (persisted in the segment superblock, `[40..56)`);
  otherwise it requires `txid > last_committed_txid`, else it is WAL corruption
  (poison). Erase/previous-tombstone is convergent by construction (a missing target
  is an idempotent hit). A checkpoint image is replayed in `op_id`-ascending order so
  an empty physical segment rebuilt from it never commits a high txid before a low one.
- **Two-layer fail-closed.** The Collection latches a *recovery-required* state if the
  post-COMMIT window (L:COMMIT durable → `publish_snapshot`) is exited abnormally
  (a committed-but-unpublished transaction must not be dropped by a later checkpoint);
  the adapter independently latches on any failure once physical state may have
  advanced (e.g. a partial tombstone), gating search/checkpoint until reopen.
- **Superseded versions.** An upsert's publish plan tombstones every row's
  same-segment `previous` (deduped) after the write bundle, so a stale version cannot
  stay live in the graph and shadow the current one.
- **Config persistence.** `active_engine` rides the facade schema: flat keeps the
  14-field layout (byte-compatible with pre-2B readers), laser widens it to 15 so an
  old binary's strict field count fails closed rather than reverting to flat.

Rotation/seal is unchanged control-plane vocabulary (§6): sealing an active LASER
generation builds an immutable sealed LASER segment into the manifest; the retired
active directory is reclaimed by an idempotent open-time sweep (no WAL restatement).

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
