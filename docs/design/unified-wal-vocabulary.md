<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Unified WAL vocabulary: one envelope, two op families

Status: **contract** (binding on U2 lifecycle code and the G1 durable
update gate). Extends `collection-wal-coordinator.md` — Physical WAL v1
framing is normative and is not restated here.

## Why now

Two write-ahead concerns share one framing contract:

1. **Collection logical WAL** (shipped): row-level logical mutations
   (`write`/`erase`), receipts, publication markers, checkpoints —
   record types 1–4 in the WAL7 frame format.
2. **Segment op-WAL** (shipped through 2C): segment-physical operations on
   mutable LASER segments (`QGUpdater`: row patches, tombstones, maintenance
   transactions, publish watermarks, label bundles, and superblock flips).
   Whole-page redo, A/B superblocks, and prefix-safe replay jointly cover
   torn writes; the Collection logical WAL remains the logical source of truth.

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
| **5** | **`SEGMENT_OP`** — segment-physical operation | **shipped with 2C** |
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
| 1 | `row_patch` | pid, absolute byte offset, length, bytes | One unchanged wire shape has three roles: ordinary legacy after-image; maintenance whole-page after-image; canonical bundle FREE preimage and latest final page. Replay validates the role from its state-machine lane. |
| 2 | `tombstone` | pid | PID-only, set-only wire. Runtime ABA protection is carried by `PidToken {pid, pid_generation}` at the mutable-segment boundary, not by this record. |
| 3 | `consolidate_begin` | consolidate epoch | Opens the maintenance lane; BEGIN alone is not a commit. |
| 4 | `consolidate_end` | consolidate epoch | Its fsync is the maintenance transaction's only commit point. An unmatched BEGIN is discarded as a unit. |
| 5 | `publish` | visibility watermark | Legacy append visibility only; monotone on replay and never a PID-reuse commit. |
| 6 | `superblock_flip` | target A/B slot, superblock CRC | Commit point of a segment checkpoint/base absorption; it is not the maintenance commit point. |
| 7 | `label_bind` | txid, row_op_id, pid, pid_generation, label | Legacy 2A binding or canonical prebind. A canonical writer emits every bind before its page frames; `pid_generation > 0` identifies reuse. |
| 8 | `tx_publish` | txid, new_committed_watermark, row_count, applied_collection_op_id | Single durable commit point of a label bundle. Canonical replay validates generation/label/final-live evidence and `new_hwm = old_hwm + count(generation == 0)` before applying. `batch_id == tx_id`; strict-increasing txid; non-regressing applied op. |

Kinds 1–6 retain `batch_id == 0`; kinds 7/8 use the physical transaction id.
The payload layout and the kind 1–8 numeric values are unchanged.

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

`garden()` still throws under `enable_wal`; it is not encoded as a maintenance
epoch or a new op kind. Legacy `insert`/`add_batch`/kind-5 publication continues
to append fresh PIDs. Reuse is limited to a canonical physical bundle.

#### 3b. Canonical prebind and bundle-only PID reuse (2C / W2)

Once PID-generation support is activated, every physical label bundle uses the
canonical lane, including append-only, mixed, and all-reuse bundles:

1. reserve all `PidToken`s, popping the canonical free-chain prefix before dense
   append allocation;
2. append all kind-7 prebinds;
3. for every reused page, record its committed `FREE | TOMBSTONE` kind-1 preimage;
4. build in a writer-private overlay and record each dirty page's latest final
   kind-1 image;
5. validate every bound PID's final trailer as live, then fsync one kind 8;
6. publish in the fixed order label snapshot → routing → reused-hidden clear →
   committed HWM.

The high-water mark advances only for generation-0 rows. Therefore an all-reuse
bundle has `new_hwm == old_hwm` while still carrying a positive binding count.
Generation must advance exactly by one for a reused PID; `UINT32_MAX` is a
permanent tombstone and can never become FREE again. Public query reads bypass the
private overlay and cannot observe a reserved row before publication.

#### 3c. Mutually exclusive replay lanes

```text
Idle
  kind3 -> Maintenance
  canonical kind7 -> CanonicalBundle

Maintenance
  kind1* -> Maintenance
  matching kind4 -> Idle
  EOF -> discard + truncate to begin
  other -> poison

CanonicalBundle
  kind7* -> prebind
  kind1* -> preimage/latest-page stage
  matching kind8 -> validate/apply -> Idle
  EOF -> discard + truncate to first kind7
  other -> poison
```

Standalone and legacy effects use a separate commit unit after activation. It
stores WAL frame locations plus row-level evidence, attaches an unambiguous legacy
owner, and re-reads/re-validates frames at the consuming kind 5, kind 8, or flip.
This prevents a current-generation page image from escaping through the wrong lane.

#### 3d. v2/v3 activation and reader admission

Immutable builders and `create_empty` continue to produce v2. The first WAL
maintenance operation checkpoints a v3 base requiring maintenance transaction and
post-redo free-list support. The first reuse-enabled bundle checkpoints the same v3
outer layout with the maintenance pair plus the PID-generation, canonical-prebind,
and mutable-label-slot triple. A reader must understand every required bit; it fails
closed on the highest valid but unsupported copy instead of falling back to an older
v2 copy.

During activation one A/B copy can temporarily be v3 while the older copy is v2.
The mutable updater understands this mixed-slot state, but role handoff to a sealed
reader is allowed only after a checkpoint leaves both copies as supported v3 images.
The Collection segment descriptor intentionally remains format 2: that descriptor is
the capability/adapter contract, while v3 is the nested QG physical-base version.

### 4. Idempotency invariant

Every `SEGMENT_OP` must replay idempotently. Recovery = scan forward from
the last `CHECKPOINT`, apply every complete frame in log order, stop at
the torn tail (v1 rule, unchanged). Log order is apply order; there is no
cross-family reordering — a frame's effects are visible to every later
frame regardless of family.

### 5. Codec reuse

The `SEGMENT_OP` payload codec uses the shared primitives in
`include/wal/frame.hpp`, under `alaya::wal` (`put_u*`, `get_u*`, `crc32`, and
`Decoder`). The collection logical codec delegates to the same delivered
primitive set. **No third primitive set.**

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

The complete mutable-LASER persistence set is:

```text
index                       QG pages + A/B superblocks
<index>.opwal               segment kinds 1–8
<index>.labels.slot0        double-buffered explicit bindings
<index>.labels.slot1        double-buffered explicit bindings
collection_wal_v1/logical.wal   independent Collection mutations
```

A pure consolidate writes kinds 3/1/4 and installs index pages; it does not
modify either label slot and it emits no Collection logical-WAL record. A reuse
bundle writes kinds 7/1/8; a later checkpoint may serialize its bindings to the
inactive label slot, append kind 6, install the alternate A/B copy, and reset the
op-WAL to that flip. Ordering is defined within each file and by those explicit
commit points, never by a synthetic cross-file sequence.

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

### 8a. Collection maintenance admission (2C / W3)

`Collection::consolidate` is an explicit control call routed only to the current
active mutable LASER generation. Its lock/admission order is fixed:

```text
Collection checkpoint mutex
  -> mutation mutex
  -> recovery-required recheck
  -> resolve current routing snapshot and active identity
  -> adapter maintenance admission (no pending mutation)
  -> QG maintenance transaction
```

The active identity is resolved after waiting, so a queued request cannot maintain
a source that became sealed during rotation. The hook is private routing metadata;
it is cleared on sealed/replacement entries and does not extend the public segment
capability ABI. Physical poison is queried explicitly and is promoted to the
Collection recovery-required latch; only destroying and reopening the handle clears
that state.

Search does not take the checkpoint mutex, mutation mutex, a unique operation lock,
or `ControlPlaneGate`. It may run throughout consolidate and relies on the QG page
seqlock to retry an odd install version. Writes, checkpoint, rotate, seal, compact,
and GC remain excluded or recovery-gated as appropriate. Active LASER filtering is
still Collection postfilter, not QG pushdown.

No kind 9 and no `.maint`, `.shadow`, or `.pidgen` file is introduced. Gardening
remains gated under the WAL.

## Acceptance

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
4. Maintenance/reuse: incomplete maintenance and canonical lanes roll back;
   durable END/kind 8 roll forward; a second reopen is byte/semantic stable.
5. Collection integration: active identity is revalidated under the mutation
   lock, physical poison gates checkpoints and control operations, and public
   search remains available across seqlock-protected page installation.
