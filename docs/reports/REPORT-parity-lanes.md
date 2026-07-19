# E-parity phase 1: C-lanes SIFT1M anchor

Date: 2026-07-18. Code base: `main@b295189`, branch `feat/parity-lanes-harness`.

## Verdict

Resident-arena LASER passes the phase-1 parity gate. The geometric mean of the nine
`arena / memqg` iso-recall QPS ratios is **106.47%**, above the 85% gate; the C=16 geometric
mean is **110.80%**. The former 9-10% 16-lane deficit is absent, so the phase-1 verdict is
**"lane repetition + queue storm attribution closed"** (`lane 重复+队列风暴归因闭环`).

No search-result inconsistency was observed. For every `(arm, ef)`, recall and the complete
top-10 checksum were identical across C=1/4/16, both A/B orders, and all repetitions.

## Harness contract

`benchmarks/parity_lanes_benchmark.cpp` implements the primary metric as follows:

- Both engines are constructed as `core::AnySegment`. Their construction/opening code is the only
  engine-specific part of the measured arms.
- Every request carries both the LASER and QG effort extensions with `ignore_safe`. Therefore the
  measured `dispatch_query()` has no engine branch and always makes one synchronous,
  single-row `segment.search()` call with fanout one.
- One persistent worker pool is constructed per C. Each worker is pinned with
  `pthread_setaffinity_np`, then repeatedly takes one query from a shared atomic cursor.
- A full 10,000-query warmup pass precedes every point and is excluded from timing. A measurement
  repeats full passes until at least 0.5 seconds of search time has accumulated; CSV/JSON record the
  exact pass and call counts. Forward and reverse arm blocks each have three independent repeats.
- The harness fails immediately on the first result difference from the initial serial snapshot and
  reports the first changed query/hit.

The CSV is flushed after every record. JSON includes the protocol, inputs, grid, machine state,
selected CPUs, actual runtime SIMD paths, allocator contract, and all measurements.

## Preparation and machine

- Dataset: SIFT1M, 1,000,000 x 128 base vectors; 10,000 queries; exact top-100 GT, evaluated at
  recall@10.
- Memory QG: rebuilt with the current `QgSegment` builder because the old probe did not retain a
  loadable current-format artifact. Parameters were fixed degree R=32 and `ef_build=100`;
  build 8.48 s and save 0.90 s were excluded from measurement.
- LASER: the retained R32 native index and full sidecars were accepted by the current importer and
  hard-linked into a current `resident_arena` segment. `ResidentArenaProvider::prepare()`
  materialized the full row store before measurement.
- Host: dual AMD EPYC 9554; CPUs 32-47 (one socket, one hardware thread from each core) were used.
  SMT was enabled, but their siblings 160-175 were unused. Governor was `schedutil`; THP was
  `madvise`.
- Runtime ISA: LASER integer/fastscan path `avx512`; memory RaBitQ path `avx512`; the common stable
  FP32 exact-distance policy selected `AVX2+FMA` on Zen 4.
- Allocator parity: memory QG `StaticStorage` and LASER resident `cache_nodes_`/materialized arena use
  `alaya::AlignedAlloc`, whose large tier is 2 MiB aligned plus `MADV_HUGEPAGE`. The harness also
  verified its query buffer was 2 MiB aligned.

Input SHA-256:

| Input | SHA-256 |
|---|---|
| `sift_base.fbin` | `8c7b3d999ba3133f865af72df078f77c2d248fdb80571d7ea1f1bb8e1750658e` |
| `sift_query.fbin` | `9b0082b67d0ac55b4c7d42216560344567ad87ce3e75a9d5214a0762f1c15d65` |
| `sift1m_gt100_exact.ibin` | `65a84807779dac6e0c9a6e3a1ae89d22d1b8a0170737b0aeb9994b47312d7a52` |

## Measured ef curves

QPS below is balanced per point as
`sqrt(median(forward x3) * median(reverse x3))`. Recall was invariant across C and repeats.

| Arm | ef | recall@10 | C=1 QPS | C=4 QPS | C=16 QPS |
|---|---:|---:|---:|---:|---:|
| MemQG | 40 | 0.88979 | 42,193 | 163,338 | 503,740 |
| MemQG | 60 | 0.93624 | 32,417 | 126,450 | 400,037 |
| MemQG | 100 | 0.97145 | 21,642 | 84,250 | 282,001 |
| MemQG | 200 | 0.99157 | 12,094 | 47,308 | 156,616 |
| LASER arena | 40 | 0.88461 | 43,007 | 168,426 | 605,061 |
| LASER arena | 60 | 0.93816 | 33,201 | 130,108 | 444,110 |
| LASER arena | 100 | 0.97404 | 21,744 | 85,314 | 279,355 |
| LASER arena | 200 | 0.99360 | 12,039 | 47,089 | 153,780 |

## Iso-recall grid

Historical linear interpolation is applied to QPS on the recall axis. Targets 0.90, 0.95, and
0.99 use ef brackets 40-60, 60-100, and 100-200 respectively. `ef*` is the corresponding
interpolated effort coordinate, not an additional measured point.

| C | target | MemQG ef* / QPS | LASER ef* / QPS | arena / memqg |
|---:|---:|---:|---:|---:|
| 1 | 0.90 | 44.40 / 40,044 | 45.75 / 40,189 | **100.36%** |
| 1 | 0.95 | 75.63 / 28,206 | 73.20 / 29,420 | **104.31%** |
| 1 | 0.99 | 192.20 / 12,839 | 181.60 / 13,825 | **107.68%** |
| 4 | 0.90 | 44.40 / 155,230 | 45.75 / 157,414 | **101.41%** |
| 4 | 0.95 | 75.63 / 109,959 | 73.20 / 115,326 | **104.88%** |
| 4 | 0.99 | 192.20 / 50,191 | 181.60 / 54,124 | **107.84%** |
| 16 | 0.90 | 44.40 / 480,945 | 45.75 / 558,804 | **116.19%** |
| 16 | 0.95 | 75.63 / 353,908 | 73.20 / 389,742 | **110.13%** |
| 16 | 0.99 | 192.20 / 166,400 | 181.60 / 176,892 | **106.31%** |

| Aggregate | Geometric mean |
|---|---:|
| C=1, three targets | **104.07%** |
| C=4, three targets | **104.68%** |
| C=16, three targets | **110.80%** |
| All nine cells | **106.47%** |

At C=16, scaling efficiency relative to 16 x C=1 was 75.07/78.42/81.01% for MemQG and
86.90/82.80/79.97% for LASER at targets 0.90/0.95/0.99. Thus the old cross-engine residual did not
survive the production-style lane protocol. Absolute 16-lane scaling is still sublinear for both
engines, but that is a separate capacity/profile question rather than a LASER parity failure.

## Protocol deviations and qualifications

- No semantic redline was intentionally changed: same machine, core budget, synchronous single-row
  AnySegment dispatch, per-call effort, sealed immutable data, no deadline/cancellation, paired arm
  order, three repeats per order, and equal warmup were all retained.
- `numactl` was unavailable. Instead, `taskset` constrained the process to node-0 physical CPUs and
  each worker pinned itself inside that set; resident allocation used first-touch on that socket.
  There was no explicit `membind`. This environmental difference is recorded rather than hidden.
- The governor remained `schedutil`; it was observed and recorded but not changed.
- The current MemQG artifact had to be rebuilt as allowed by the protocol. LASER native artifacts
  did not require rebuilding, only current-importer wrapping.
- Two short-window forensic attempts are retained under `attempts/` but excluded from every number
  above. They motivated the 0.5-second minimum and are not cherry-picked into the final grid.
- Deep10M/GTE768 cross-machine cells, paged residency, public multi-row `batch_search`, and the E2E
  Collection leg are the explicitly deferred next wave, not phase-1 omissions silently treated as
  passing data.

## Raw data and verification

Raw root:
`/home/huangliang/workspace/alaya-dev/data/laser-update/parity-lanes-20260718/`

Final files:

- `raw/sift1m-parity-lanes.csv` (SHA-256
  `654a7df06005e506aa5da6eed752480a4e97cd1d0732b5bdd2712844314a8d72`)
- `raw/sift1m-parity-lanes.json` (SHA-256
  `774be91281f884a266c650192310a7ce8b832acbfb613b5ba6957c74f241d365`)
- `logs/formal.stdout.log`, `logs/formal.stderr.log`, preparation logs, smoke grids, and retained
  forensic attempts.

Verification completed:

- Release configure with `BUILD_TESTING=ON`, Python off, runtime-dispatch build (`NATIVE_ARCH=OFF`).
- `cmake --build build/Release --target parity_lanes_benchmark -j16`.
- 100-query same-C smoke, 1,000-query C=1/4/16 checksum smoke, and minimum-duration reuse smoke.
- Final CSV: 144 records, 48 `(arm,C,ef,order)` cells, multiplicity exactly three; JSON schema 2
  parsed successfully and contained the same 144 measurements.
- `clang-format --dry-run --Werror` and `git diff --check`.
- `uvx pre-commit run -a` passed all hooks.

## Next-wave handoff

Reuse this harness unchanged for the Deep10M and topology-sealed GTE768 resident-arena grid, with
host-local CPU lists and explicit machine metadata. Add paged residency only after block C lands;
keep it out of this anchor. The public multi-row batch secondary metric and Collection E2E leg
should remain separate tables so neither can overwrite the C-lanes primary claim.
