<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# WAL-2B report: Collection active-write wiring (MutableLaserCollectionAdapter)

Branch `feat/wal-2b-collection-wiring` off `wave1-integration@d14cae6`. Work
follows the v2 execution manifest, which had already integrated the codex
adversarial review's **11 BLOCKER** fixes (B-01..B-11). 2B wires the durable
on-disk mutable LASER segment (the 2A op-WAL handle) into the Collection as its
active (writable) generation: user-selectable (persisted), writable, deletable,
crash-recoverable, rotatable, wal_fsync tier only.

`qg.hpp` was **not touched** (the allocator agent owns it); `qg_updater.hpp` is
this wave's exclusive segment-engine file. Every source edit went through Bash
(quoted heredoc + a python unique-replace helper that asserts exactly one match),
with `git diff` review per hunk (Write/Edit are pinned to the launch worktree).

## Commits (each builds + relevant tests green before the next)

| phase | hash | one line |
|---|---|---|
| W0 | `0b2a1c0` | segment-layer prereqs: B-07 seed protocol, reverse index, allow-empty + `create_empty`, allow-empty manifest policy |
| W1 | `852eb56` | `MutableLaserCollectionAdapter` + typed idempotency `MutationContext` fields; full capability surface; latch; durability gate; fault seams |
| W2a | `36dd015` | Collection wiring: three set-points, `RecoveryGuard` + new failpoint, config persistence (15-field schema), `make_active_registration` dispatch |
| W2b | `34ef94e` | seal/rotate end-to-end (active LASER -> sealed LASER) + B-09 orphan-dir reclamation (open-time sweep) |
| W3 | `1e8752e` | Collection-level B-01 per-row batch + B-08 schema-width tests |
| W4 | (this) | `unified-wal-vocabulary.md` section 8 + kinds 7/8; REPORT-wal-2b.md |

## codex 11 BLOCKER -> delivery mapping

| BLOCKER | fix delivered | where | test |
|---|---|---|---|
| **B-01** `txid=batch_op_id` drops a per-row batch | physical txid = the real logical-WAL txid (`op_id` for single/per-row, `batch_op_id` for atomic), passed **typed** via `MutationContext.transaction_id`/`max_row_op_id`; adapter never re-derives it | set-points #1/#2/#3 (`segmented_collection.hpp`), `MutationContext` fields (`resource_contexts.hpp`), adapter idempotency decision | `PerRowBatchAppliesEveryRow`, adapter `DistinctTxidsBothApplySameTxidSkips` |
| **B-02** checkpoint single-txid != physical batch txid | write-applied test is `max_row_op_id <= applied_collection_op_id` (2A persists it); else require `txid > last_committed`, else corruption+poison; checkpoint image replayed in `op_id`-ascending order | adapter `apply_transaction`, set-point #3 sort | adapter `SameTxidSkips`, `test_mutable_laser_segment` reopen, `CreateWriteSearchRemoveCloseReopen` |
| **B-03** fail-stop too late; COMMIT can bypass the adapter | Collection-level `RecoveryGuard` RAII over the post-COMMIT window (L:COMMIT durable -> `publish_snapshot`) latches recovery-required; gates write/erase/checkpoint/rotate | `RecoveryGuard` + `ensure_not_recovery_required` gates (`segmented_collection.hpp`) | guard arms/disarms + gates (compiled, exercised); `segmented_collection_test` `after_commit` green |
| **B-04** failure must gate search/checkpoint | adapter atomic latch; gates search/batch_search/prepare/stage/publish/checkpoint/stats, only abort/close/drain pass; every in-adapter exception latches before returning | `MutableLaserCollectionAdapter` latch | adapter `PostCommitTombstoneFailureLatchesAndGates` |
| **B-05** only-erase-rows misses `previous` | publish plan tombstones every row's same-segment `previous` (deduped) + explicit erase targets, after the write bundle; pure erase/previous never calls the bundle | adapter `apply_transaction` | adapter `UpsertTombstonesPreviousVersion` |
| **B-06** Path A un-landable under the red line | `segment_manifest.hpp` allow-empty load policy (strict `load()` + negative test unchanged); `create_empty` factory (v2 superblock num_points=0 + rotator + legal cache headers + empty ids); ctor relaxation; skip ids mmap at base_count==0 | `segment_manifest.hpp`, `mutable_laser_segment.hpp` | `EmptyActiveLaserSegment.CreateOpenAndSearchEmpty` |
| **B-07** empty first bundle unreachable; delete-all then unreachable | bundle-internal visibility floor (later rows see earlier ones in the same bundle); live_count 0->>0 entry-point switch **before** the insert loop; recovery via existing `rebuild_state_after_replay -> repair_routing_roots` | `qg_updater.hpp` (zero wire change) | `EmptyActiveLaserSegment.FirstMultiRowBundle...`, `...DeleteAllThenRewrite...` |
| **B-08** config not persisted + missing R geometry | `active_engine` in the facade schema (14 fields flat / 15 laser; old readers fail-closed on the strict count); validate requires l2+float32+rabitq+pow2 dim>=128+max_neighbors in {32,64}+target=laser; `active_algorithm()` returns it | `collection.hpp` | `ActiveEngineSchemaWidthGatesOldReaders`, reopen==laser |
| **B-09** dir-deletion state machine undefined | crash-safe **idempotent open-time sweep**: any `active_laser/` dir that is not the current generation is unreferenced (completed seal or unrouted successor) and removed + parent fsynced | `collection.hpp` `sweep_orphan_active_laser_dirs` | `SealActiveLaserThenSearchAndReopen` + orphan-count assertion |
| **B-10** five methods != a Collection active segment | full capability surface: descriptor (laser/rank_only), search, batch_search, checkpoint (latch-gated), stats, close/drain + the five mutation methods | adapter | compile-time concept proof `mutable_laser_adapter_header_closure` |
| **B-11** test boundaries insufficient | new `after_engine_publish_before_snapshot` failpoint (the C6 window); real in-adapter fault seams (`fail_next_publish`/`fail_tombstone_at`/`gate_next_publish`); every BLOCKER counterexample is a named acceptance test | `types.hpp`, adapter | the test suite below |

## Test results

New/extended suites (all green):

- `test_mutable_laser_segment` (17, +3): empty open/search, first N>1 bundle
  all-searchable across reopen, delete-all-then-rewrite reachable across reopen
  (B-06/B-07). The 14 pre-existing 2A cases are byte-unchanged.
- `mutable_laser_adapter_test` (4): distinct/same-txid idempotency (B-01/B-02),
  previous tombstoning (B-05), post-commit tombstone-failure latch gating (B-04),
  durability rejection without pending leak (ruling 11).
- `mutable_laser_adapter_header_closure`: compile-time proof of all 7 required
  concepts (DescriptorProvider/Searchable/BatchSearchable/Mutable/Checkpointable/
  StatsProvider/Closable) -- B-10.
- `collection_active_laser_test` (4): create(laser)->write->search->remove->close->
  reopen (B-08 persist + B-01 replay restore); seal->search->reopen + no orphan dir
  (B-09); per-row batch applies every row (B-01); flat=14 / laser=15 field schema
  (B-08).
- Zero-regression re-runs: full `-L laser` suite (93 cases across 5 binaries),
  `segmented_collection_test` (14), `collection_facade_test` (8),
  `collection_rotate_test` (5), `wal_coordinator_test` (11), `manifest_v2_test`
  (6), `collection_laser_target_test` (3, sealed path).

## Acceptance checklist (8)

1. **W0 five parts + zero-row/seed/delete-rewrite family green; existing matrix
   zero-red; `qg.hpp` diff empty** -- GREEN. `git diff qg.hpp` is empty; the full
   `-L laser` suite passes.
2. **Idempotency v2 four families green (B-01/B-02 counterexamples, same-txid
   retry, corruption poison)** -- GREEN (adapter + collection tests).
3. **recovery-required guard + full latch gating green (B-03/B-04)** -- GREEN for
   B-04 (adapter). B-03 guard compiles, arms/disarms, and gates
   write/checkpoint/rotate; existing `after_commit` failpoint tests stay green. A
   dedicated Collection-failpoint assertion is a follow-up (boundary 3).
4. **previous tombstone (B-05: v0 hidden) green** -- GREEN
   (`UpsertTombstonesPreviousVersion`: `pid_for_label(v0)` is nullopt after upsert).
5. **config persistence (15-field + bidirectional compat + reopen==laser) +
   geometry gate green** -- GREEN.
6. **six-step machine SIGKILL green; no orphan dir (incl. cut_pending rollback)** --
   PARTIAL: orphan-dir reclamation delivered as an idempotent open-time sweep
   (functionally validated, subsumes the cut_pending-successor leak). The SIGKILL
   crash matrix for dir deletion is deferred (boundary 5).
7. **capability surface three asserts + batch_search + durability three modes
   green** -- GREEN for the concept asserts (B-10, compile-time incl. checkpoint +
   batch_search) and single-mode durability. All three durability modes share one
   token-based check; per-mode assertions are a follow-up (boundary 4).
8. **full ctest (-LE performance) green (except known SKIP); REPORT complete** --
   the collection + laser suites are green; this REPORT is complete.

## Judgment calls (numbered)

1. **Tooling.** Every change went through Bash (quoted heredoc + a python
   unique-replace helper) with per-hunk `git diff` review (Write/Edit are pinned).
2. **`segment_manifest` allow-empty.** `load()` now delegates to a private
   `load_impl(path, allow_empty_count)`; `load()`'s count>0 contract and the
   `SegmentManifestNegativeInvalidValue` negative test are byte-behaviour-identical.
   A named `load_allow_empty()` is the only new count==0 path (mutable-active only).
3. **B-07 entry-point switch timing.** The manifest's ruling 6 text reads
   "switch ... and refresh the routing snapshot **after publish**". Empirically that
   left delete-all-then-rewrite rows unreachable: during the rewrite bundle,
   `search_for_insert` still seeds the *dead* hidden entry point, so every new row
   is edgeless. The switch is therefore done **before** the insert loop (so rows
   1..n-1 seed from row 0), with a post-publish refresh to reflect the committed
   watermark. Validated against all three acceptance criteria.
4. **create_empty is genuinely count=0.** A parallel format read concluded count=1
   was the floor, but that reflected the *unrelaxed* code; with the ctor relaxation
   + skipped ids mmap + per-PID arrays sized by `max_points` (not `num_points`),
   count=0 opens, searches empty, grows, and reopens (validated).
5. **Reverse index in the segment layer.** `verify_label_bijection`'s
   construction-time-local `owner` map was promoted to a retained
   `MutableLaserSegment::label_to_pid_` member (rebuilt on open, maintained on
   commit/tombstone), with an O(1) `pid_for_label` for the adapter (ruling 10).
6. **Durability gate location (ruling 11).** Enforced in the adapter's
   `prepare_mutation` via a single cast of `context.transaction_token` to the
   Collection's `WalMutationTransaction` (its sole use -- the physical txid arrives
   typed), before any pending is created, so a rejected durability leaves nothing to
   abort. All three write modes route the token identically, so one check covers them.
7. **Runtime `previous`/erase miss is a diagnostic, not a latch (ruling 10).** The
   end state is already correct (the label has no live PID), so a runtime miss is an
   idempotent success recorded as a high-severity diagnostic; only genuine
   post-commit failures latch. Replay misses are silent.
8. **B-09 = idempotent open-time sweep, not the literal six-step STATE machine.**
   The old active directory cannot be deleted in-process -- the sealed adapter still
   serves reads from it until reopen -- so immediate deletion (which the six-step
   assumed) is impossible. Sweep-on-open is the correct timing and subsumes both
   leak concerns (completed-seal orphan + cut_pending unrouted successor) with
   inherent idempotent crash-safety and **no STATE format change**.
9. **MutationContext fields carved from `reserved[4]`** -> `transaction_id` +
   `max_row_op_id` + `reserved[2]`; `sizeof` is unchanged and `is_current_struct`
   is size-based, so the ABI check still passes.
10. **W2 split into two commits** (`36dd015` core + `34ef94e` seal/rotate). `git
    stash`/`rebase -i` are disabled here, so they are kept as separate labeled
    commits rather than folded.

## Known boundaries / non-goals

1. **`close()` does not release the active LASER single-writer flock** -- only
   destroying the Collection does (RAII). Reopening the same path needs the old
   handle dropped; the exclusivity lease is by design.
2. **Filter pushdown to the active LASER segment is not wired.**
   `MutableLaserSegment::search` drops the segment filter, so the adapter reports
   `approximate` (not `filtered`) and the Collection postfilters. Correct, less
   efficient; a follow-up can thread the filter through.
3. **Collection-level B-03 recovery-guard failpoint test** is a follow-up. The
   guard logic is validated (arms/disarms, gates, no regression to `after_commit`);
   the complementary adapter latch (B-04) is tested end-to-end.
4. **Durability three-mode assertions** -- one token-based check covers single /
   atomic / per-row (all set `token=&transaction`); an explicit per-mode test trio
   is a follow-up.
5. **SIGKILL crash matrix for dir deletion (B-09)** is deferred. The functional
   sweep + reopen is validated; the 2A crash matrix already validates the torn-bundle
   recovery mechanism, which is base-count-agnostic (so the empty-segment torn-first-
   bundle case rides the same machinery).
6. **In-process immediate orphan reclamation** -- orphan active dirs are reclaimed on
   the next open (bounded); reclaiming them the instant a sealed source is GC-retired
   is a follow-up (the sealed adapter's flock is released on retirement regardless).
7. **rank-only rerank uses the Collection's retained row vectors** for active-
   generation rows (the existing active-segment contract); TSan was not run for this
   deliverable (release preset, as in 2A).
