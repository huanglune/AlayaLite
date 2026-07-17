<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# WAL-2A report: op-WAL label physical transaction base

Branch `feat/wal-2a-physical` off `main@5dfc5a5`. Work follows the v2 execution
manifest, which had already integrated the codex adversarial review's **7 BLOCKER**
fixes (B-01..B-07). 2A is the **pure physical layer**: it does not implement the
`core::Mutable` concept and does not wire into Collection (that is 2B). A **second**
codex adversarial pass was run on the finished W3 code (max effort) and its findings
were adopted (see the review section). The crash matrix — SIGKILL + a power-loss
persistence model, including the new label-transaction cuts and a four-file
time-ordered boundary test — is green, so the "durable label transaction" claim is
authorized.

## Commits (each builds + relevant tests green before the next)

| phase | hash | one line |
|---|---|---|
| W0 | `fab2d1a` | single-writer handle mutex + lock-free poison read gate (`ensure_readable`) |
| W1 | `3a9793e` | wire layer: kind=7 `label_bind` + kind=8 `tx_publish`, superblock sub-layout, golden bytes |
| W2 | `74b35f4` | appended-label runtime snapshot, durable double-buffered slots, checkpoint integration, bijection |
| W3 | `3e6dec9` | replay staging/promotion (a)/(b) + full B-04 validation set + divergence tests |
| W4 | `a1a3094` | crash-matrix extension (B-07): 4 new failpoints, bundle/label power-loss, boundary states |
| W3.1 | `14a7e8f` | hardening: idempotent bind de-dup for torn large-bundle retry (codex B1, self-found) |
| review | `68e7848` | hardening: no-steal guard, staged-backlink guard, mid-tx poison, fresh-enable crash window (codex B2-B5) |
| W5 | (this) | REPORT-wal-2a.md |

Diffstat vs `main@5dfc5a5`: **8 files, +1796 / -39**. Changed: `qg_updater.hpp`,
`segment_op_wal.hpp`, `qg.hpp` (comment only), `mutable_laser_segment.hpp`, and 4
test files under `tests/laser/qg/`. **No red-line file touched** (`wal/frame.hpp`,
`index/collection/**`, `core/capabilities.hpp`, `laser_segment*.hpp`,
`laser_segment_importer.hpp`, `space/**`, `simd/**`, `index/graph/hnsw/**`). kind=1..6
wire bytes, the 12-cell matrix cases, and the kind=5 emit path are unchanged.

Note on commit count: the manifest specified six commits (W0-W5). The two extra
hardening commits (`14a7e8f`, `68e7848`) are correctness fixes surfaced by
self-review and the codex pass; folding them into W3 was impossible here because the
environment forbids `git stash` and interactive rebase (`-i`). They are kept as
separate, clearly-labeled commits (judgment call 7).

## Test results

Full `ctest --preset release -L laser -j 8`: **24/24 pass, 0 fail**
(`DifferentialRankOnlyManifestGate` SKIPs under `-DBUILD_PYTHON=OFF`, as expected).
The full project build (`cmake --build --preset release`) is clean.

Label-transaction coverage:
- `test_segment_op_wal` (11): + kind=7/8 round-trip & truncation/trailing, unknown-kind
  moved 7->9, **kind=1..6 golden bytes** frozen.
- `test_qg_updater_wal` (20): + multi-bundle pure-WAL-replay recovery, torn-bundle
  retry with the same txid, orphan-bind de-dup, checkpointed txid/applied-op
  preconditions (B-03), staged-backlink poison, mid-tx-failure poison, and the
  **B-04 divergence family** (8 crafted-WAL cases each poison on replay).
- `test_mutable_laser_segment` (14): + concurrent add/checkpoint//search smoke,
  explicit & mixed(3-domain) label translation across reopen, slot-load validation
  family (checksum/truncate/trailing/missing), canonical-empty tolerates missing
  slots, and both **B-06** bijection poison cases.
- `test_segment_op_wal_crash` (20): the original **12-cell matrix unchanged**, +4
  bundle SIGKILL cuts, +bundle power-loss atomicity, +label-checkpoint 5-state
  time-ordered boundary atomicity, +fresh-enable crash-window recovery, +residual
  `.reset.tmp` regression; equivalence now asserts label + tx watermark.

## codex 7 BLOCKER -> delivery mapping (manifest B-01..B-07)

| BLOCKER | fix delivered | where |
|---|---|---|
| **B-01** tx must not include tombstone | `commit_physical_bundle` takes only rows+labels (no tombstone param); `binding_count==0` illegal; standalone non-tx `tombstone()` unchanged | W2; wire contract W1 |
| **B-02** atomic committed/label visibility + poison-gated search | immutable snapshot published (release) **before** committed release-store; search acquires committed **then** snapshot; `ensure_readable` poison gate at search entry+exit | W0 (gate) + W2 (snapshot swap in `publish_common`/`commit_physical_bundle`) |
| **B-03** tx watermark survives checkpoint | `last_committed_txid` + `applied_collection_op_id` in superblock `[40..56)`; **every** checkpoint writes them; recovery adopts + max on replay | W1 + W2 + W3/W4 tests |
| **B-04** promotion can't diverge committed/label | `replay_tx_publish` full set: strict-inc txid, non-regress applied-op, `new==old+count`, staged count, kPidMax/capacity, row_op_id=={0..count-1} unique, pid==[old,new) unique, batch_id==tx_id | W3 (`replay_tx_publish` + 8 divergence tests) |
| **B-05** slot dir persistence root-cause | pre-create both slots + fsync files + fsync parent dir before any content flip; content checkpoint overwrites the inactive slot in place; canonical-empty tolerates absent files | W2 + W4 (boundary states) |
| **B-06** bijection over base + implicit identity | segment-layer `verify_label_bijection` over **live** PIDs across base sidecar U legacy identity U explicit binding; two poison tests | W2 |
| **B-07** matrix models 3+ file checkpoint | label-slot fsync observer hook; 4-file (index/WAL/slot0/slot1) forced capture; time-ordered boundary states; +4 Layer1 cuts + 2 Layer2 scenarios | W1 + W4 |

## Second codex adversarial pass (on finished W3, max effort) -> disposition

codex found 6 BLOCKERs + 4 OK-notes. The OK-notes confirm the core design
(two-rows-one-bind cannot silently promote; the (a)/(b) boundary + separate local
watermark are safe for legal flip logs; the mutex snapshot visibility ordering is
sound; the B-05 content-checkpoint cuts are ordered safely; row-patch replay
materializes pages before rebuild; double replay is stable).

| codex BLOCKER | disposition | fix / commit |
|---|---|---|
| B1: torn-retry orphan binds poison a legal same-txid retry | **fixed** | idempotent `(tx_id,row_op_id)` de-dup at staging (`14a7e8f`, self-found first) |
| B2: `write_cache=false` steal / row-patch has no tx membership | **guard + documented** | ctor rejects `write_cache=false` under enable_wal (`68e7848`); residual = recall artifact (below) |
| B3: `stage_backlinks=true` commits rows before reverse edges exist | **fixed** | `has_staged_edges()` guard in `commit_physical_bundle` (`68e7848`) |
| B4: internal exception mid-bundle doesn't poison | **fixed** | try/catch poisons on any exception (`68e7848`) |
| B5: fresh-enable crash window is unrecoverable (poisons) | **fixed** | discard an orphan-flip-only WAL + re-stamp (`68e7848`) |

## Acceptance checklist (8/8)

1. **codec kind=7/8 full + kind=1..6 golden regression** — GREEN (`test_segment_op_wal`, 11).
2. **existing 12-cell matrix zero-change, zero-red** — GREEN (cases byte-identical; 20 crash tests total).
3. **new matrix: Layer1 4 cuts, Layer2 2 scenarios, time-ordered 5 states, equivalence incl. label+tx** — GREEN.
4. **B-04 divergence family each -> test, green** — GREEN (8 cases + orphan de-dup + staged/mid-tx poison).
5. **B-03: tx->checkpoint->reopen->old txid rejected; torn retry same txid accepted** — GREEN.
6. **B-06: bijection two poison + mixed 3-domain** — GREEN.
7. **slot load validation family + legacy empty + missing-file poison; first-open dir fsync visible in code** — GREEN.
8. **full `ctest -L laser` green (except known SKIP); REPORT complete** — GREEN (24/24).

## Judgment calls (deviations, numbered)

1. **Tooling.** Write/Edit are pinned to the launch worktree; every file change went
   through Bash quoted-heredoc + a python unique-replace helper (asserts one match),
   with `git diff` review per hunk. (Per manifest.)
2. **W2/W3 boundary.** The `commit_physical_bundle` write path + the `publish_common`
   refactor moved from W3 into W2 (explicit bindings are only producible via the
   commit path, so W2's slot/reload/bijection tests are otherwise untestable —
   circular dependency). W3 keeps the replay staging/promotion, the B-04 set, poison
   paths, and the recovery functional tests.
3. **Sub-layout location.** `kLabelStateReservedOffset` / `kTxStateReservedOffset` +
   accessors live in `qg_updater.hpp` next to `kUidReservedOffset` (precedent), with
   a documenting comment in `qg.hpp`. 512B superblock layout unchanged.
4. **kind=1..6 batch_id on replay.** Did not add a `batch_id==0` assertion to
   kind=1..6 replay (always 0 already) to avoid perturbing the 12-cell matrix; only
   kind=7/8 get `frame.batch_id==tx_id` validation.
5. **Snapshot concurrency.** GCC 11 lacks `std::atomic<std::shared_ptr>`; switched to
   a plain `shared_ptr` under a dedicated `label_snapshot_mutex_` (never the handle
   mutex). B-02 ordering holds via mutex release/acquire + the committed acquire.
6. **adopt unification.** Recovery-start and flip-adoption label loads are identical,
   so they share one `adopt_label_state(sb)`.
7. **Two extra hardening commits** (`14a7e8f`, `68e7848`) instead of six exactly —
   correctness fixes from self-review + the codex pass; `git stash` and `-i` are
   disabled here, so clean folding into W3 was impossible. Kept as labeled commits.

## Known boundaries / non-goals

- **Aborted-bundle stale edge (codex B2 residual) is a recall artifact, not a
  correctness bug.** After a GRACEFULLY-aborted bundle (mid-tx exception) whose
  backlink after-image was flushed by `~WalFile`, replay restores an edge `u->p`
  where `p` is an uncommitted tentative PID. When a later transaction reuses `p`, the
  edge carries the old neighbor's FastScan code — but exact rerank reads `p`'s actual
  row, so results stay correct; only frontier-expansion recall can dip. The general
  fix (transaction membership for physical row-patch after-images) is the same
  deferred class as B-01 and belongs to 2B/2C. The no-steal guard closes the worse
  durable-steal form; SIGKILL / power-loss never expose the residual (buffered
  after-images are not forced).
- `LabelBindings` is an ordered `std::map` copied whole per bundle (COW snapshot).
  O(n) per commit; fine for the 2A physical base, a 2B/2C optimization target.
- `commit_physical_bundle` does not check label uniqueness at runtime (per manifest);
  logical uniqueness is 2B's job. The construction-time bijection (B-06) is
  defense-in-depth and catches a conflicting binding on the next reopen.
- **Transactional tombstone is out of 2A** (B-01): `commit_physical_bundle` carries
  no tombstone; the standalone immediate-publish `tombstone()` is unchanged and its
  crash semantics stay covered by the original 12-cell matrix.
- **TSan not run** for this deliverable (the manifest gate is the release preset; the
  W0 concurrency smoke runs under release). A TSan pass over the concurrent
  add/checkpoint//search path is a reasonable follow-up.
