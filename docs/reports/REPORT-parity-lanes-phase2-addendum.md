<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI
SPDX-License-Identifier: AGPL-3.0-only
-->

# Parity Lanes Phase 2 Addendum: Tight-Cache Row And Full E2E Table

Date: 2026-07-19. This addendum closes the two formal measurements that
[REPORT-parity-lanes-phase2.md](REPORT-parity-lanes-phase2.md) deferred
because both were blocked by production defects its forensic legs exposed:

- The **tight-cache paged spectrum row** was rejected fail-closed because the
  paged kernel returned nondeterministic results under cache misses. Fixed on
  main by `63dde8c` (issue-order FIFO processing).
- The **full 10,000-query Collection E2E table** was truncated to a 100-query
  forensic leg because `RoutingSnapshot::known_rows_for()` was O(N) per call.
  Fixed on main by `6ddec5c` (cached per-segment row counts).

Both legs below were measured after the qg→LASER same-id swap landed
(`main@1eb32ed`), so the E2E table also gets a **qg-face** arm: the same
physical LASER segment exposed under the public `qg` algorithm id, which is
the route production qg traffic takes after the swap.

Host, dataset, CPU pinning, warmup, paired forward/reverse orders, and the
three-repeat protocol are unchanged from phase 2. All measurements passed the
fail-closed checksum contract: one unique checksum per `(arm, ef)` group,
zero `RESULT_MISMATCH` events across every leg.

## 1. Tight-cache paged spectrum row

Protocol identical to the phase-2 `sift1m-paged-full` leg (10,000 queries,
lanes 1/4/16, CPUs 32-47, three repeats per paired order, warmup 1 round,
0.5 s minimum measure) with `--laser-residency paged_pool` on the retained
tight import (`indices/laser/sift1m-tight`, `online_cache_num: 150000` = 15%).
LASER ef 50 (the phase-2 paged calibration point), MemQG ef 76 rides along as
the in-run comparator. Balanced QPS = sqrt(median fwd × median rev).

| Arm | ef | recall@10 | C=1 | C=4 | C=16 |
|---|---:|---:|---:|---:|---:|
| MemQG (in-run) | 76 | 0.95518 | 23,972 | 97,924 | 350,707 |
| LASER paged, tight 15% | 50 | 0.96112 | **1,110** | **1,954** | **1,304** |
| vs phase-2 arena row | | | 3.95% | 1.79% | 0.36% |

Readings:

- The row that phase 2 had to reject is now protocol-valid: 36/36
  measurements checksum-stable. The determinism fix, not a protocol change,
  is what admitted it.
- C=16 is *slower* than C=4. This is the block-C libaio I/O scaling ceiling
  (t8/t1 = 1.95x) surfacing in the spectrum: sixteen lanes of ef-50 misses
  oversubscribe the same AIO path that four lanes nearly saturate. The
  all-cache rows never see this ceiling; the tight row is the honest floor of
  the residency spectrum.
- Recall 0.96112 differs from the phase-2 paged-full 0.95631 at the same
  ef 50. The cache set changes which nodes are served from cache versus I/O,
  and the fixed kernel is deterministic *per cache state*, not across cache
  states; the tight leg is also a separate import with its own 150k sidecar.

## 2. Collection E2E, full 10,000-query protocol

All E2E legs: resident arena, `--collection-e2e --return-distances`, lanes
1/16, CPUs 32-47, three repeats per paired order. The Collection face wraps
the identical physical segment the direct arm searches; construction and the
million-row registration are excluded from timing, exactly as in the phase-2
forensic leg. Recall and per-arm checksums are invariant in every group, and
recall is **identical between the direct and Collection arms** in every leg.

### 2.1 Laser face at the .95 operating point (ef 73)

| C | segment direct QPS | Collection QPS | Collection / direct | adapter + routing tax |
|---:|---:|---:|---:|---:|
| 1 | 25,062 | 14,588 | 58.20% | 41.80% |
| 16 | 388,472 | 245,289 | 63.14% | 36.86% |

Against the phase-2 forensic table (tax 99.98%), the routing fix moved the
Collection face from 5.8 QPS to 14.6k QPS at C=1. The remaining 37-42% is
real adapter-layer work (AnySegment dispatch, fanout bookkeeping, logical-id
mapping, response assembly) and is now the honest optimization target.

The 100-query forensic leg reported a higher direct QPS (35k) than this
full-protocol leg (25k): 100 looping queries stay hotter in cache than 10,000
distinct ones. The full-protocol numbers are the citable ones.

### 2.2 Matched-depth triangle at ef 100: laser face vs qg face

The public qg route synthesizes its search effort as
`max(QgSearchExtension default = 100, caller effort, candidate limit)`
(`segmented_collection.hpp` fanout, legacy-faithful and preserved by the
swap), so it cannot express ef 73. Both faces were therefore measured at
ef 100, where user effort equals the floor and all arms run the same depth.
Recall is 0.97404 for every arm of every leg — the swap route returns the
same results as the laser face and the direct segment.

| Leg | C | direct QPS | Collection QPS | Collection / direct |
|---|---:|---:|---:|---:|
| laser face, ef 100 | 1 | 19,604 | 13,594 | 69.35% |
| laser face, ef 100 | 16 | 302,853 | 209,119 | 69.05% |
| **qg face (swap route), ef 100** | 1 | 18,398 | 11,010 | 59.84% |
| **qg face (swap route), ef 100** | 16 | 296,507 | 214,055 | 72.19% |

Cross-face, Collection arm to Collection arm: **C=16 = 102.4%** (the qg
face is at parity; the extension-translation route costs nothing visible at
the concurrent operating point) and C=1 = 81.0%, which this addendum does
*not* attribute to the translation path: the qg face necessarily opens a
second arena instance of the segment (different page placement/warmth), the
two legs' direct arms themselves drift 6.2% at C=1, and per-query the gap is
~17 µs where the translation work is a per-request validation loop plus one
64-byte extension rebuild. A dedicated same-instance A/B would be needed to
split placement noise from route cost; C=16 parity bounds the route cost at
noise level where it matters.

### Findings recorded in passing

- **Public qg effort floor.** Through Collection, qg search effort has a
  floor of 100 (the extension default participates in a max). Callers can
  raise but not lower it. Pre-existing memqg semantics, faithfully carried by
  the swap; surfaced here because the harness asked for ef 73 and measured
  recall said otherwise.

## Raw data and provenance

Raw root:
`/home/huangliang/workspace/alaya-dev/data/laser-update/parity-lanes-phase2-addendum-20260719/`

| File | SHA-256 |
|---|---|
| `raw/sift1m-paged-tight.csv` | `ababa56d821432bc70541c9f406fcdafb6b8cc9c77372ef09bcecbdf09e51571` |
| `raw/sift1m-paged-tight.json` | `ec730520cc4ec644c462fa64b72b5c6caec6d7f48dfe60ecd1066b6cf519055c` |
| `raw/sift1m-e2e-full.csv` | `47295153b1ac144369b9e89c08a3462d0fb7fd865297396b7bf0ce9e6364f702` |
| `raw/sift1m-e2e-full.json` | `04fe3be8ed202c5624bbec2d3183862dfd83c76ea7fc7aba1863dca54ce0844f` |
| `raw/sift1m-e2e-laserface-ef100.csv` | `f133afa921fc6ed76f0302d7960f7b157d025fe9c2112f80b1ab39922f07aed2` |
| `raw/sift1m-e2e-laserface-ef100.json` | `c95de6d1037b674305f3c3a110e0d24d13b96240807a3c60af00693d0ffb3ed4` |
| `raw/sift1m-e2e-qgface.csv` | `f1c493c9ccafcc63d3c94b41af0109b2b432dd8bceb1dc1bcf9e991535b246e9` |
| `raw/sift1m-e2e-qgface.json` | `d13e19aca88759f06350c0257149b8b984377e8829907ead72a8effa060841c5` |

Engine base: `main@1eb32ed` (qg→LASER swap + Windows WAL flush fix). The
tight row and the ef-73 E2E leg ran the benchmark built at `main@1eb32ed`;
the ef-100 triangle ran the benchmark built at `34fc7d5`
(`bench/e2e-qg-face`, benchmarks-only change adding `--collection-face qg`;
its default laser-face path is byte-identical in behavior to the base
binary). Artifacts: phase-1/phase-2 retained indices as recorded in the
phase-2 report; the tight leg reuses `indices/laser/sift1m-tight`, the E2E
legs reuse the phase-1 arena segment and MemQG artifact.
