<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# U2-a report: LASER mutable-segment lifecycle + SEGMENT_OP op-WAL (G1)

Branch `feat/u2a-segment-lifecycle` off `main@866331e`. All work follows the
manifest **and its Amendment v2** (codec adversarial review: 5 BLOCKER +
6 MAJOR, all adopted). The G1 crash matrix is green, so the "durable in-place
updates (op-WAL)" claim is authorized and recorded.

## Commits (each builds + tests green)

| hash | one line |
|---|---|
| `e424ef2` | refactor(wal): hoist Physical WAL v1 framing into a shared `wal/` layer (W1) |
| `3036e24` | refactor(wal): W1 fixup per review — primitives-only Decoder, fail-closed byte-invariance proof, contract doc |
| `a640fe1` | feat(laser): SEGMENT_OP op-WAL payload codec, record type 5 (W2-1) |
| `32092c8` | feat(laser): QGUpdater op-WAL integration + dedicated recovery (W2-2) |
| `23da083` | test(laser): G1 crash matrix — SIGKILL + power-loss persistence model (W2-3) |
| `2a6ce14` | feat(disk): MutableLaserSegment — durable single-writer mutable handle (W3) |
| trailing  | style: apply repo clang-format/cmake-format (tooling; see judgment call 7) |

Diffstat vs main: 18 files, +2876 / -216. New: `include/wal/frame.hpp`,
`include/index/graph/laser/qg/segment_op_wal.hpp`,
`include/index/disk/mutable_laser_segment.hpp`, 5 test files + `tests/wal/`.

## Test results

Full `ctest -LE performance -j8`: **74/74 pass** (the environmentally-flaky
`DifferentialRankOnlyManifestGate` passed here too). `uvx pre-commit run
--all-files`: **all hooks pass** (cpplint, clang-format, cmake-format,
layer-boundaries, no-chinese, reuse, detect-secrets, ...).

New suites:

- `wal_frame_test` (8): golden byte layout, structural scan, one-envelope /
  two-family mixed scan, torn-tail heal, atomic reset marker, primitive+take
  decoder, collection cross-family **byte-invariant** fail-closed, torn-tail
  still truncates.
- `test_segment_op_wal` (7): round-trip all six op kinds, type-5 framing
  integration, version/kind/size/truncation/trailing rejection.
- `test_qg_updater_wal` (6): fresh lineage, published inserts+tombstones survive
  reopen with and without a checkpoint, byte-stable double replay, foreign
  lineage rejected, reclaim/consolidate/garden rejected under WAL.
- `test_mutable_laser_segment` (4): add/search/tombstone/flush/checkpoint/reopen
  convergence in both residency modes, label mapping, single-writer flock, batch.

## Crash matrix (`test_segment_op_wal_crash`, 12 cases, all green)

Two layers, per Amendment clause G (SIGKILL does not drop the kernel page cache,
so the persistence model is the real G1 gate).

**Layer 1 — SIGKILL / failpoint (6 cells).** A forked child self-SIGKILLs at each
lifecycle cut; the parent reopens (recovery), asserts the committed count is a
reachable log prefix, then reopens again for byte- and state-stable double replay.

| kill point | expected | result |
|---|---|---|
| after_wal_append_before_apply | inserts not committed (unforced) | prefix=base, stable — pass |
| after_apply_before_publish_fsync | inserts not committed | prefix=base, stable — pass |
| after_publish_fsync | inserts committed (forced) | base+N live, stable — pass |
| flip_before_superblock | checkpoint rolls forward via flip frame | base+N, stable — pass |
| superblock_before_reset | new superblock durable, marker skipped | base+N, stable — pass |
| after_wal_reset | checkpoint complete | base+N, stable — pass |

**Layer 2 — power-loss persistence model (3 scenarios x 4 retain/drop states).**
A `SegmentIoObserver` snapshots the forced content of the index and the op-WAL at
every fsync; the harness materializes each retain/drop combination of the two
streams' unforced tails and recovers each.

| scenario | claim | result |
|---|---|---|
| published inserts | durable via the WAL even when unforced index pages are dropped | all 4 states -> base+N, stable — pass |
| unpublished inserts | never visible (no publish record) even though writeback forced the row after-images | all 4 states -> base, stable — pass |
| checkpoint | atomic: flip rolls forward or old base rolls back, never a torn superblock | all 4 states prefix-reachable, stable — pass |

**Layer extras (3).** Torn WAL tail and a corrupt earlier frame each stop at the
verified prefix and never resynchronize (no committed batch applied);
superblock/WAL replay equivalence (acceptance 2) holds — recovery via the
surviving new superblock and via replaying the pre-checkpoint WAL over the old
superblock agree (base+N, same live count).

## Out-of-manifest judgment calls

1. **All file I/O via Bash (tooling).** The harness pinned this subagent to the
   launch worktree (`laser-fullcache`); the Write/Edit tools stayed locked there
   even after `EnterWorktree` moved cwd + Bash write-access to `u2a-lifecycle`
   (verified: Bash writes land correctly, Write/Edit refuse). So new files were
   written with `cat` heredocs and every surgical edit went through a perl
   literal-exact-replace helper that reproduces the Edit tool's
   uniqueness-or-fail contract. Every change was `git diff`-reviewed and
   build/test-verified; no functional impact.
2. **`qg.hpp` touched (recovery_mode).** Clause C's recovery open must skip the
   exact `file_size` superblock check, which lives in
   `QuantizedGraph::load_disk_index` (qg.hpp), not qg_updater.hpp. Added a
   defaulted `bool recovery_mode = false` param — existing callers unchanged.
   qg.hpp is not on the red-line forbidden list.
3. **segment_uid in `reserved[0..8)`, not a new field.** Clause F says "reserved
   fixed offset"; I memcpy the uid into the existing `QGSuperblockV2::reserved`
   array, so the 512-byte superblock layout + static_assert are untouched.
4. **Flip replay sets the file length once, in the post-replay rebuild.** Clause
   E specifies the flip apply sets the length from the image; I do the single
   authoritative `ftruncate` in `rebuild_state_after_replay` (which provably
   equals `flip.file_size == committed-pages`) and keep the superblock pwrite in
   the flip path. Functionally equivalent, one truncate.
5. **Writer lock is `<index>.writer.lock`, not the .opwal.** Clause H says flock
   the .opwal, but checkpoint atomically renames the .opwal (reset), silently
   orphaning an flock on the old inode (caught by the single-writer test). A
   dedicated never-renamed lock file gives the same guarantee, robustly.
6. **Mutable search uses `QGUpdater::search`.** For the standalone handle, search
   goes through the updater's committed/deleted-coherent read path in both
   residency modes; residency still drives arena materialization + the
   `write_at` arena mirror. A v1 simplification (a graph-path arena search would
   need the AIO/workspace setup and arena freshness at publish).
7. **Trailing `style:` commit.** Edit tool blocked + interactive rebase
   unavailable, so cosmetic clang-format/cmake-format changes were folded into
   one trailing commit rather than amended per originating commit. Intermediate
   commits are build+test green; the final tree is pre-commit clean.
8. **Handle `tombstone()` forces via `publish(num_points())`.** Same-watermark
   publish is valid and appends a publish frame + fsync, making the tombstone
   durable on return.

## G1 achievement + residual boundaries

**G1 is met for the minimal safe scope** (insert / tombstone / publish /
checkpoint lifecycle): durable in-place updates with a no-steal
force-before-writeback WAL, group-committed publish, atomic checkpoint flip,
lineage-bound + fail-closed recovery, and a two-layer crash matrix (SIGKILL +
power-loss) that is green. Recorded in CHANGELOG, the qg_updater.hpp header, and
`unified-wal-vocabulary.md` (§6a + acceptance 3 revised per clause J).

Residual / next wave (all deliberate):

- **Out of G1 scope, throws under `enable_wal`:** PID reuse/reclaim, consolidate,
  garden, bloom. Their WAL transaction formats are the next wave — the six-kind
  codec (incl. consolidate_begin/end) is already implemented format-first.
- **Collection wiring is OUT OF SCOPE** (red line with the U2-b branch touching
  `segmented_collection.hpp`). `MutableLaserSegment` is a standalone handle; the
  next wave wires it into the `collection_target_builder` lazy-laser slot with
  the logical<->row map that owns non-identity labels.
- **Labels sidecar is not in the op-WAL** (v1: appended row label == PID; base
  rows map through the immutable ids sidecar).
- **Phase separation** (tombstone/consolidate vs inserts) remains a caller
  contract, as before.
- **Crash-matrix cells for a later wave:** a failpoint *inside* the recovery
  replay (today covered indirectly by double-replay byte-stability) and explicit
  ENOSPC/short-write/fsync-error fault injection into the poison path (the poison
  path exists and is reachable via lineage rejection, but is not fault-injected).
