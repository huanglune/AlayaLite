<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Collection logical WAL and mutation coordinator

## Scope and namespace

`SegmentedCollection` has one owner for logical mutation ordering, redo,
recovery, visibility, retry receipts, and checkpoints. Engines contribute only
the mutation bundle and idempotent replay hooks already present in
`AnySegment`; they do not create another SDK-visible WAL or watermark.

The durable writer is feature-gated by
`CollectionFeatureFlags::wal_coordinator`. When enabled it accepts only this
new namespace:

```text
<collection-root>/.alaya_internal/collection_wal_v1/
  logical.wal
  CURRENT                         # present after the first checkpoint
  checkpoint_<wal-cut>.bin        # immutable checkpoint image
```

It never reads, writes, renames, or removes a legacy PyIndex WAL, recovery
directory, or artifact layout. Disabling the feature prevents new WAL-backed
mutations/checkpoints. The reader and recovery code must remain available once
this namespace has been written (on-disk roll-forward).

## Physical WAL v1

`logical.wal` is an append-only sequence of independently checksummed frames.
All integers use little-endian encoding. The fixed header is 36 bytes:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | magic `WAL7` (`0x374c4157`) |
| 4 | 2 | format version, currently `1` |
| 6 | 1 | record type |
| 7 | 1 | flags |
| 8 | 4 | complete frame length, including header and trailer |
| 12 | 4 | payload length |
| 16 | 8 | transaction/op id |
| 24 | 8 | batch op id |
| 32 | 4 | IEEE CRC-32 |

The payload follows the header and a four-byte `END7` trailer terminates the
frame. CRC-32 covers the complete frame with the checksum field treated as
zero, including lengths, identifiers, payload, and trailer. Payloads are
bounded at 64 MiB.

Record types are `PREPARE=1`, `COMMIT=2`, `PUBLISH_MARKER=3`, and
`CHECKPOINT=4`. Flag bit `0x01` records a `wal_fsync` outcome. A
`PUBLISH_MARKER` with bit `0x80` contains the completed per-row batch receipt;
this keeps invalid/conflict/aborted statuses and the batch retry token stable
across reopen.

Scanning is prefix-safe. EOF at a frame boundary is clean. A short header,
impossible length, unknown version/type, missing trailer, or checksum mismatch
marks the rest of the file as a corrupt/torn tail. Recovery stops at the last
complete frame and never searches past damage for a later COMMIT. Before new
appends, open truncates the bad suffix to that verified boundary and fsyncs the
file.

The mutation payload codec owns strings, vector bytes, metadata variants,
documents, LogicalId kind/canonical bytes, row addresses, previous addresses,
per-row op ids/statuses, and retry tokens. No persisted payload retains a
caller or engine buffer view.

## Redo and publication state machine

For a WAL-enabled mutation the only order is:

```text
admission + monotonic op-id
  -> append PREPARE (flush for wal_fsync)
  -> engine prepare/stage + private RoutingSnapshot/ID-map/metadata stage
  -> append COMMIT (fsync for wal_fsync)
  -> engine publish
  -> atomic RoutingSnapshot pointer exchange
  -> append publish marker
  -> cache/deliver MutationReceipt
```

The private snapshot is constructed before engine stage. An engine-stage
failure discards that metadata/ID-map view and aborts the engine token. A
metadata-stage failure after engine stage also aborts the engine token. Neither
side is externally reachable until COMMIT and the routing pointer exchange.
Searches admitted earlier retain their pinned snapshot/watermark and filter out
dark or newer engine rows.

Once COMMIT is written, it is authoritative. A publish failure is not converted
to abort: reopen invokes the engine's idempotent replay hook and publishes the
logical view. Repeating the same committed frame or replay operation does not
allocate another version.

`WriteOptions` defaults to `WriteDurability::wal_fsync`. In this mode COMMIT
fsync also makes its preceding WAL prefix stable and the receipt reports
`DurabilityState::wal_fsync` plus the real durable watermark. The explicit
`WriteDurability::searchable` mode buffers WAL frames without COMMIT fsync,
publishes them for current-process search, and reports
`DurabilityState::searchable_not_durable`. It can lose the acknowledged
mutation on process or system crash and is not a durable mode. With the WAL
feature disabled, the compatibility shell still uses the same coordinator
ordering in memory and reports `memory_only`.

## Crash and recovery table

| Crash boundary | WAL/recovery behavior |
|---|---|
| before PREPARE | no record and no operation |
| after PREPARE, before stage/COMMIT | unmatched PREPARE is not replayed; its engine token is aborted/reclaimed |
| after stage, before COMMIT | staged data remains dark; unmatched PREPARE is aborted or idempotently overwritten |
| after COMMIT, before publish | committed payload is replayed through the engine and atomically added to the routing view |
| after publish, before receipt | replay is idempotent; row/batch retry token returns the original op id and status |
| after receipt | visibility and durable watermarks recover to the acknowledged cut; searchable-only receipts may be lost as declared |

Recovery first loads `CURRENT` if present, then scans the verified WAL prefix.
PREPARE records are paired with exactly one matching COMMIT in file order.
Unmatched PREPARE records are aborted. Committed transactions newer than the
checkpoint cut are replayed in WAL order. A missing publish marker is repaired
after replay. Duplicate frames are ignored after the first committed
transaction identity.

The crash battery includes injected tests for all six boundaries, corrupt and
truncated tails, duplicate replay, and a real child-process `SIGKILL`: a
`wal_fsync` COMMIT is recovered after kill, while a searchable-only mutation
killed before buffered frames leave the process is absent after reopen.

## Receipts and batch behavior

Each `MutationReceipt` carries the compatibility `op_id`, explicit
`batch_op_id` and `row_op_id`, visibility and durable watermarks, searchable
state, durability state, stable row status, and retry token. A completed retry
token lookup occurs before validation or op-id allocation and returns the
stored receipt. Checkpoints persist both row and batch retry ledgers.

The stable row-status set is:

```text
inserted updated replaced deleted already_exists not_found
conflict invalid_argument aborted
```

`per_row_independent` is the default. It processes input order directly (never
hash iteration), assigns a row op id even to rejected rows, and commits valid
rows separately. Thus duplicate LogicalIds observe earlier rows in the same
input. A durable batch-completion marker preserves the complete ordered status
vector and batch retry token.

`all_or_nothing` validates every row and rejects duplicate LogicalIds before
staging. A validation/duplicate failure marks the failing row with its stable
status and every other row `aborted`. A valid batch is sent as one opaque
engine bundle, staged once, represented by one PREPARE/COMMIT pair, and
published with one routing pointer exchange. Engine stage failure aborts the
bundle and marks every row `aborted`. The Collection-internal
`atomic_mutation_bundle` capability must be true; otherwise the call returns
`not_supported` and never falls back to per-row behavior.

## Checkpoint, WAL cut, and manifest v2

Checkpoint serializes control-plane calls. It closes normal operation
admission, drains every previously admitted search/mutation, and then acquires
the mutation lane. The active mutable engine must expose its existing
`Checkpointable` slot; absence returns `not_supported`.

After the engine checkpoint, Collection writes and fsyncs an immutable image
containing the latest version map, vectors/metadata/documents needed by the
test engine replay seam, visibility/durable watermarks, metadata epoch, and
retry ledgers. It atomically publishes and fsyncs `CURRENT`, then atomically
replaces the WAL with one fsynced CHECKPOINT frame. These orderings make either
the old WAL or the new checkpoint sufficient at every interruption point.

`CheckpointReceipt` reports `durable_watermark`, `wal_cut`, `metadata_epoch`,
and the checkpoint logical name. Call
`SegmentedCollection::apply_checkpoint_to_manifest` before a manifest-v2
control-plane publication; it sets the collection `wal_cut`, metadata
checkpoint/epoch, ID-map checkpoint, and row-version maximum. The manifest's
own artifact transaction remains responsible for its atomic publication.

## Gate 7-B importer seam

The concrete state machine, non-destructive source reader, marker/audit
formats, independent gate, and corpus evidence belong to the retired legacy
PyIndex importer contract, which is no longer part of this documentation tree.

An importer must write only this new namespace. It preserves source ordering
without fabricating a second WAL as follows:

1. Register imported rows with their original `RegisteredRow::upsert_sequence`.
2. Set `CollectionRecoveryOptions::minimum_next_op_id` to one past the maximum
   source op id (or the next reserved source id). Collection takes the maximum
   of this floor, registered rows, checkpoint state, and all WAL frames.
3. If a committed snapshot cut has no surviving logical row (for example a
   scalar delete whose RocksDB checkpoint no longer retains the external ID),
   set `minimum_visibility_watermark` to that committed cut. This preserves the
   cut without fabricating a row or advancing to an uncommitted reserved ID.
4. Open the WAL-enabled Collection and call full `checkpoint`.
5. Apply the returned cut to manifest v2 with
   `apply_checkpoint_to_manifest`, then publish that manifest through the
   existing artifact transaction.

The importer must not copy a legacy PyIndex WAL into `logical.wal`, reuse a
legacy recovery directory, renumber source versions below the configured
floor, or publish a migration marker before the new checkpoint and manifest
are durable.
