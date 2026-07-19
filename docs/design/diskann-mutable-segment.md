<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# DiskANN internal mutable Segment contract

> **Superseded / historical (2026-07-19).** The DiskANN index and Segment
> family were retired. This document records the former Gate 8-B contract and
> does not describe a current C++ or Python surface.

## Scope and gate

Gate 8-B adds a Collection-internal mutation bundle to `DiskAnnSegment`. It is
not an SDK or Python surface. `DiskEngineFeatureFlags::diskann_mutable_segment`
is independent of the readonly `diskann_segment` flag and defaults off. Only
`DiskAnnMutableSegmentFactory` can create a writer operation table, and a
disabled writer factory returns `not_supported` before opening or modifying an
artifact.

The existing readonly factory, direct `DiskANNIndex`, legacy PyIndex and
DiskCollection-v1 paths are unchanged. The writer uses only the public native
`load(..., updatable=true)`, `insert`, `remove`, `flush` and search APIs. Gate
8-B does not add a kernel hook or change the DiskANN, LASER, or core contract
trees.

The runtime identities are:

| instance | feature | factory version | advertised operations |
|---|---|---:|---|
| Gate 8-A readonly | `diskann_segment` | 1 | search, batch search, stats |
| Gate 8-B writer | `diskann_mutable_segment` | 2 | search, batch search, mutation bundle, checkpoint, stats, close, drain |
| roll-forward reader | always available to the new reader | 2 | search, batch search, stats, close, drain |

The roll-forward reader never advertises mutation or checkpoint. It loads a
published mutable checkpoint in tombstone-aware mode and consumes any complete
committed Collection-WAL tail into a private working copy before erasing the
writer slots. A sealed `SegmentedCollection` can then rebuild its logical
snapshot from the same WAL while remaining non-mutable. This is an on-disk
roll-forward guarantee, not compatibility with an older binary.

## Redo ordering and dark stage

The Gate 7 Collection coordinator remains the sole owner of WAL ordering,
op-id allocation, logical version maps, retry ledgers and receipts. A durable
mutation has this one order:

```text
Collection admission + op-id
  -> WAL PREPARE containing the owned mutation payload
  -> DiskAnnSegment prepare/stage in private transaction storage
  -> WAL COMMIT (fsync for wal_fsync)
  -> DiskAnnSegment publish through native insert/remove
  -> atomic Segment visibility-map and Collection routing-snapshot publish
  -> publish marker and MutationReceipt
```

Prepare copies every vector and row address. Stage changes only the private
transaction state; it does not call the native update API. A staged insert is
therefore absent from both the native graph and the pinned logical visibility
map. Abort drops that transaction and updates pending/in-flight accounting.

Publish holds the native visibility lock, applies rows in op-id order, updates
the label-to-slot map, per-slot version/tombstone state and applied watermark,
then atomically replaces the final-filter visibility map. Search captures that
map at admission. Graph traversal may observe a weakly consistent native
generation, but a result is returned only if its label is live in the pinned
map. Old upsert versions and tombstones therefore cannot reappear.

COMMIT is authoritative before the publish slot is called. A publish error
fail-closes new admission; recovery reopens the last immutable checkpoint and
replays the committed WAL. It is never converted to an abort.

## Mapping, idempotence and batches

Native external labels are Collection physical row IDs. The Segment maintains:

- one live `label -> internal slot` mapping;
- one record per allocated slot containing label, last applied op-id and
  tombstone state;
- a monotonic applied-op-id watermark and persisted minimum-next-op-id floor.

Insert calls native `insert(vector, label)`. Delete and upsert resolve the
previous label to its internal slot before calling native `remove(slot)`.
Replay skips every row at or below the applied watermark, so a repeated COMMIT
does not allocate another slot or tombstone. Normal prepare rejects an op-id
below the persisted next-op floor; only the replay seam may consume an old id.

`per_row_independent` remains the default Collection behavior. Each valid row
has its own PREPARE/COMMIT/publish and stable status. Duplicate, missing,
insert-only, replace and upsert status selection stays in the ordered
Collection coordinator.

For `all_or_nothing`, the registration explicitly sets
`atomic_mutation_bundle=true`. The Collection validates every row and duplicate
before PREPARE. DiskANN applies a valid mixed bundle to an ephemeral clone of
the current working generation and swaps that clone only after every native
operation succeeds. Failure discards the clone. This is deliberately more
expensive than per-row mode and is charged to the stage/I/O reservation; it is
never silently downgraded.

Retry tokens are persisted by the Collection WAL/checkpoint. A denied
admission creates no PREPARE or engine transaction, so retrying the same token
with sufficient resources performs the operation once.

## Concurrency, lifecycle and resources

The writer reports reentrant search, `search_with_stage=true`,
`search_with_publish=false`, serial mutation, no checkpoint/search overlap,
native async/cooperative cancellation, and explicit drain. The native public
index serializes its own updates; the Segment additionally uses a shared
search/exclusive publish lock for the strict visibility boundary.

Close stops new search and mutation admission. Drain waits for admitted native
searches through callback return, as well as prepared/publishing mutations.
The request `lifetime_pin`, response buffers, cancellation state and Segment
owner remain alive through exactly-once lane delivery. Native `flush()` is the
reconnect/dirty-page barrier used before checkpoint.

Open accounts for both reading the immutable generation and copying it to the
private work directory, plus the public update pools/gates. Mutation admission
checks pending and stage reservations, conservative full-graph I/O credits,
and the extra memory/artifact copy needed by an atomic shadow bundle. The
Collection preflights the complete batch and its WAL frames before the first
row. A denied lease or credit returns `resource_exhausted` with no WAL frame,
native update or logical publish.

`SegmentStats` reports the applied watermark, native live/allocated/tombstone
counts, pending transactions, in-flight search/mutation, accounted dirty bytes,
health and last error. The internal typed counters additionally expose
prepared, staged, committed, applied, replayed and aborted totals.

## Checkpoint and mutable state v1

The mutable writer never updates the immutable source checkpoint. It works in:

```text
<collection-root>/.alaya_internal/diskann_mutable_v1/work/<segment_pid_seq>/
```

After Collection closes mutation admission and drains admitted work, the
Segment checkpoint:

1. verifies that no private transaction remains;
2. calls native `flush()` to drain reconnect work and persist graph/meta/IDs,
   cache, PQ data and `slots.bin`;
3. fsyncs `diskann_mutable_state.bin`;
4. copies the complete native family into an artifact-transaction staging
   generation and writes checksummed `ARTIFACTS.v2` plus `READY`;
5. durably installs the generation, then atomically replaces manifest v2.

The immutable target is
`segments/<segment-id>_g<segment-generation>_c<checkpoint-generation>`. A crash
at any transaction step leaves the old manifest and generation usable.
Restart cleanup removes only owned staging/READY orphans not referenced by the
manifest.

`diskann_mutable_state.bin` is little-endian, versioned and CRC-protected. Its
header stores applied watermark, minimum-next-op-id, checkpoint generation and
slot count. Each fixed-width slot record stores label, last op-id, tombstone and
zeroed reserved bytes. Open verifies its length/CRC, `ids.bin` labels, native
deleted bitmap, duplicate live labels and live/tombstone totals before making
the instance searchable.

The Collection checkpoint follows the Segment checkpoint and then cuts its
logical WAL. If the process dies after the Segment manifest publish but before
the Collection checkpoint/WAL cut, the newer physical watermark makes old
checkpoint rows no-ops and the remaining WAL reconstructs the missing logical
tail. If the process dies earlier, the old Segment generation plus WAL replays
to the same state.

## Crash matrix

| kill boundary | recovered result |
|---|---|
| before PREPARE | no operation |
| after PREPARE, before stage | unmatched payload is ignored/aborted; no native row |
| after stage, before COMMIT | private stage is lost and remains invisible |
| after COMMIT, before publish | reader/writer replay applies the native row and logical version |
| after publish, before receipt | watermark suppresses duplicate native apply; retry token returns the original receipt |
| after receipt | acknowledged visibility, durability, status and op-id recover |

The battery uses real `fork`/`SIGKILL` and a local filesystem. Every case is
opened twice; committed cases retain one physical allocation and pre-COMMIT
cases retain none.

## Gate 8-B acceptance map

| requirement | contract test |
|---|---|
| dark stage under concurrent search; immediate publish | `MutableGateKeepsStageDarkAndPublishesThroughNativeApi` |
| callback pin, cancel, concurrent publish and close/drain | `MutableCloseDrainsPublishAndCancelledSearchBufferLifetime` |
| search/mutation concurrency | `ConcurrentMutableCollectionSearchMutationStress` under TSan with ASLR disabled |
| checkpointed mapping/tombstones/watermark and gate-off roll-forward, including WAL tail | `CollectionWalCheckpointReopenUsesAppliedWatermark` |
| per-row/AON statuses, retry and zero-effect budget denial | `MutableCollectionBatchModesAndBudgetDenialAreStable` |
| all six real crash cuts and repeated recovery | `MutableWalSixPointSigkillBatteryAndRepeatedReplayConverge` |
| mixed insert/remove/bundle native bitwise differential | `MutableDifferentialSequenceAndCheckpointCutsMatchDirectKernel` |
| randomized publish/checkpoint/reopen cuts | `MutableRandomizedPublishCheckpointCutsRemainDifferential` |
| manifest transaction crash steps | `MutableCheckpointFiveStepFailuresPreserveOldGeneration` |

## Release-note material

- Added an internal, default-off DiskANN mutable Segment bundle using the
  existing Collection WAL coordinator. It provides dark staging, durable
  COMMIT-before-publish ordering, strict live-version/tombstone filtering,
  idempotent WAL replay, both batch modes and resource-governed admission.
- Added atomic manifest-v2 DiskANN checkpoints carrying label/slot/tombstone
  state and the applied/next-op watermarks. The new readonly recovery path can
  roll a committed WAL tail forward while exposing no mutation API. The
  canonical SDK and Python API remain readonly for DiskANN mutation.
