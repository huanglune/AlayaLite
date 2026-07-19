<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# Documentation map

The documentation is split by purpose so that current contracts are not
confused with delivery records or exploratory work. This directory currently
contains 18 design documents, 18 reports, 6 research notes, 2 user guides, and
9 image assets.

## Start here

- **Current specifications:** start with the frozen
  [core segment contract](design/contract-v3.md), then follow the
  [segment admission contract](design/segment-admission-contract.md) and the
  relevant Collection, WAL, RaBitQ, or LASER design below.
- **Historical delivery reports:** use [reports/](reports/) to trace what a
  completed engineering wave changed and how it was verified.
- **Research notes:** use [research/](research/) for experiments, design
  critiques, and paper-planning material that is informative but not normative.
- **User documentation:** start with the [build guide](user/BUILDING.md) or the
  [client user guide](user/CLIENT_USER_MANUAL.md).

## Directory map

| Directory | Responsibility | Reading entry |
| --- | --- | --- |
| [`design/`](design/) | Current architecture contracts and explicitly labelled historical design records. A historical document is evidence of an earlier decision, not the current product contract. | [Current specifications and guides](#current-specifications-and-guides) |
| [`reports/`](reports/) | Historical engineering delivery, review, performance, and verification reports. | [Report index](#historical-delivery-reports) |
| [`research/`](research/) | Exploratory analysis, experiments, design reviews, and publication planning. | [Research index](#research-notes) |
| [`user/`](user/) | Supported build and client-facing usage instructions. | [User guides](#user-guides) |
| [`images/`](images/) | README branding, community QR codes, and benchmark/result figures referenced by documentation. | [Image assets](#image-assets) |

## Design documents

### Current specifications and guides

- [Core segment contract v3](design/contract-v3.md) — frozen engine boundary
  and compatibility contract.
- [Segment admission contract](design/segment-admission-contract.md) — binding
  predicate seam for filters and tombstone visibility.
- [Unified WAL vocabulary](design/unified-wal-vocabulary.md) — binding envelope
  and operation-family vocabulary.
- [Collection logical WAL and mutation coordinator](design/collection-wal-coordinator.md) —
  Collection mutation ordering, recovery, visibility, and checkpoints.
- [RaBitQ build and format contracts](design/rabitq-formats.md) — current LASER
  format contract, including explicitly separated retired memory-QG bytes.
- [LASER implementation guide](design/LASER.md) — native LASER build, I/O, and
  implementation guidance.
- [Gate 11 legacy cleanup and reader inventory](design/legacy-cleanup.md) —
  current post-cutover API, dependency, and artifact-reader inventory.

### Historical design records

The following files remain in place for traceability. Each is marked historical
here only because its own heading or opening paragraph says that it is
historical, superseded, or retained solely as migration evidence.

- **Historical:** [ANN sealed-segment build cutover](design/ann-sealed-target.md)
  — its opening notice says the design was superseded after HNSW retirement and
  the `target_algorithm` default change.
- **Historical:** [Canonical Collection facade audit](design/collection-canonical-facade.md)
  — its opening notice identifies the Gate 9-A/9-B audit as superseded.
- **Historical:** [Collection / index semantics baseline](design/collection-semantics.md)
  — its opening paragraph retains the removed public-index behavior only as
  migration evidence.
- **Historical:** [DiskANN internal mutable Segment contract](design/diskann-mutable-segment.md)
  — its opening notice says the retired DiskANN family is not a current surface.
- **Historical:** [DiskANN readonly Segment contract](design/diskann-readonly-segment.md)
  — its opening notice preserves retired DiskANN identities as a migration record.
- **Historical:** [Gate 10 filter, seal, compact, and GC baseline](design/filter-seal-gc.md)
  — its opening notice says the Flat-only target descriptions predate current state.
- **Historical:** [Gate 6 disk-segment final state](design/gate6-disk-segments.md)
  — its opening notice calls the checkpoint superseded and retains its tables as
  migration evidence.
- **Historical:** [LASER immutable disk segment](design/laser-segment.md) — its
  opening notice says the former imported, rank-only adapter was superseded.
- **Historical:** [Manifest v2 control plane and DiskFlat segment](design/manifest-v2-disk-flat.md)
  — its opening notice labels it a historical Gate 6 baseline and rollout record.
- **Historical:** [Memory QG legacy dispatch contract](design/memory-qg-legacy-dispatch.md)
  — its opening notice preserves only the pre-retirement mapping.
- **Historical:** [Memory graph to Segment migration pattern](design/memory-segment-migration-pattern.md)
  — its opening notice retains the retired graph-service migration as evidence.

## Historical delivery reports

Reports describe completed work and point-in-time verification; they do not
override current contracts in `design/`.

- [Allocator merge](reports/REPORT-allocator-merge.md)
- [Blind-review fail-closed fixes](reports/REPORT-blind-failclosed.md)
- [Blind-review minor fixes](reports/REPORT-blind-minors.md)
- [Final review #25 accounting fixes](reports/REPORT-final-review-25.md)
- [HNSW retirement](reports/REPORT-hnsw-retirement.md)
- [LASER importer dimension gate](reports/REPORT-laser-importer-dim-gate.md)
- [LASER inner-product kernel](reports/REPORT-laser-ip.md)
- [Parity lanes phase 1](reports/REPORT-parity-lanes.md)
- [Parity lanes phase 2](reports/REPORT-parity-lanes-phase2.md)
- [Parity lanes phase 2 addendum](reports/REPORT-parity-lanes-phase2-addendum.md)
- [Topology-preserving seal](reports/REPORT-topology-seal.md)
- [U2-a: mutable-segment lifecycle and op-WAL](reports/REPORT-u2a.md)
- [U2-b: filter pushdown and admission predicate](reports/REPORT-u2b.md)
- [U2-c: LASER Collection target wiring](reports/REPORT-u2c.md)
- [U4 preflight](reports/REPORT-u4-preflight.md)
- [WAL-2A: label physical transaction base](reports/REPORT-wal-2a.md)
- [WAL-2B: Collection active-write wiring](reports/REPORT-wal-2b.md)
- [WAL-2C: maintenance transactions](reports/REPORT-wal-2c.md)

## Research notes

- [O_DIRECT flush-path analysis](research/laser_dio_flush_analysis.md) — I/O
  diagnosis and flush-mode selection.
- [LASER update design review](research/laser_update_design_review.md) —
  independent critique and experiment plan.
- [LASER update exploration](research/LASER_UPDATE_EXPLORATION.md) — dynamic
  update prototype, experiments, and production gaps.
- [LASER update plan from the Yi comparison](research/LASER_UPDATE_NEXT_PLAN_YI.md)
  — next-step architecture and quality plan.
- [LASER update paper storyline review](research/LASER_UPDATE_PAPER_STORYLINE.md)
  — publication positioning and adversarial review.
- [LASER update research storyline](research/LASER_UPDATE_STORY.md) — research
  narrative, evidence gaps, and paper outline.

## User guides

- [Building AlayaLite](user/BUILDING.md) — prerequisites, configure/build/test
  commands, platform notes, and troubleshooting.
- [AlayaLite Client User Guide](user/CLIENT_USER_MANUAL.md) — canonical
  Collection usage plus clearly identified pre-1.1 migration material.

## Image assets

- Branding and community assets: [banner](images/banner.jpg),
  [user portrait](images/user-portrait.svg),
  [WeChat group QR code](images/wechat-group-qr.png), and
  [legacy WeChat QR code](images/wechat_QR_code.jpg).
- Benchmark and result figures:
  [Fashion-MNIST](images/fashion-mnist-784-euclidean.png),
  [GIST](images/gist-960-euclidean.png),
  [integer-filter QPS](images/int-0p1p_qps_c1_c80.png),
  [string-equality-filter QPS](images/strequ-0p1p_qps_c1_c80.png), and
  [LASER versus disk ANN systems](images/laser-vs-disk-anns.png).
