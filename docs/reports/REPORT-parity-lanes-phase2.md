# E-parity phase 2: Deep10M/GTE768 grid and secondary spectra

Date: 2026-07-19. Engine base: `main@e8e2462`. Branch:
`feat/parity-lanes-phase2`.

## Verdict

The resident-arena C-lanes claim generalizes beyond the SIFT1M anchor.

- Deep10M's nine-cell `LASER arena / MemQG` iso-recall geometric mean is
  **162.09%**; its C=16 mean is **161.01%**.
- Topology-sealed GTE768's nine-cell mean is **102.68%**; its C=16 mean is
  **112.38%**. The sealed 768-dimensional leg therefore does not reproduce
  the old high-dimensional deficit.
- Across the two equally sized nine-cell grids, the combined geometric mean
  is **129.01%**, and the combined C=16 mean is **134.52%**. Both datasets pass
  the pre-registered 85% arena gate, and the former 16-lane residual remains
  closed.

The secondary findings are mixed but actionable:

- Public multi-row batch calls are within **-1.05% to +1.77%** of the
  single-row query-lane baseline. This passes the 5% forensic threshold and
  supports the post-D interpretation of batch as per-query sugar.
- The all-row-cache paged kernel is deterministic, but achieves only
  **42.32/43.39/39.92%** of resident-arena throughput at C=1/4/16. The 15%
  tight-cache paged leg cannot produce protocol-valid performance data:
  identical serial calls changed top-10 IDs, so the harness failed closed.
- The Collection E2E forensic leg exposes an unrelated O(N) routing defect.
  On a 100-query subset, Collection delivers only **5.762 QPS at C=1** and
  **74.839 QPS at C=16**, versus **35,049** and **431,685** through the segment
  face. The measured tax is **99.9836%/99.9827%**. The full 10,000-query leg
  was not run because each query scans the one-million-entry reverse map at
  least twice before graph search.

## Protocol and machine

The two primary grids used the phase-1 harness unchanged: one persistent
external fixed worker pool per C, one synchronous single-row `AnySegment`
call per query, fanout one, both ignore-safe QG and LASER effort extensions,
sealed immutable data, and no deadline or cancellation. Every point received
one full-query warmup pass. Measurement repeated full passes until at least
0.5 seconds of search time accumulated. Forward and reverse arm blocks each
had three repeats. The CSV was flushed per record; a changed valid count or
top-10 ID aborted the run.

Point QPS is balanced as
`sqrt(median(forward x3) * median(reverse x3))`. Historical linear
interpolation is applied to QPS and ef on the recall axis. `ef*` below is an
interpolated coordinate, not another measurement.

- Host: dual AMD EPYC 9554, Linux 6.8.0-59; CPUs 32-47 only. These are node-0
  physical cores; SMT siblings 160-175 were not selected.
- The process was constrained with `taskset`, and each worker pinned itself.
  `numactl` was unavailable, so there was no explicit `membind`; first touch
  occurred inside the node-0 CPU mask.
- Pre-run `top` snapshots reported 99.9-100.0% idle CPU and no competing
  compute process. The load averages still reflected excluded preparation in
  some snapshots.
- Governor: `schedutil`; SMT active; THP: `madvise`.
- Release build, `BUILD_TESTING=ON`, Python off, runtime dispatch. LASER and
  memory RaBitQ selected `avx512`; the common stable FP32 distance policy
  selected `AVX2+FMA`.
- Large MemQG storage, LASER cache/arena storage, and query matrices use
  `alaya::AlignedAlloc`, whose large tier is 2 MiB aligned with
  `MADV_HUGEPAGE`. JSON verified 2 MiB query-buffer alignment.

Machine/ISA/allocator fields are embedded in every completed JSON file. The
pre-run load snapshots were observed immediately before launch and are
summarized above.

## Ground truth and preparation

### Deep10M

The supplied `sift_gt.ibin` has header 10,000 x 100 but is not a canonical
ibin payload. It contains a contiguous uint32 ID matrix followed by a
contiguous float32 squared-distance matrix, hence its 8,000,008-byte size.
The harness correctly rejected it by exact-size validation. The ID block was
losslessly extracted to `raw/deep10m_gt100_exact.ibin`.

The source has 10,000/10,000 monotonically ordered distance rows and every ID
is in `[0, 10,000,000)`. An independent audit selected queries
`0,17,113,997,2027,4099,6007,7993,9001,9999`, scanned all 10M vectors using
float64 squared L2, ordered by `(distance,id)`, and matched all 1,000 top-100
positions. The audit took 3.45 seconds and is retained in
`logs/deep10m-gt-audit.log`.

The missing native LASER artifact was rebuilt with the retained probe tool:

```text
bench_laser_update_sift build \
  --base .../deep10m-fbin/sift_base.fbin \
  --prefix .../indices/native/deep10m/deep10m-r32 \
  --n 10000000 --R 32 --main_dim 96 --threads 48
```

Vamana topology took 174.0 seconds; native packing completed in 208.0
seconds, 387.4 seconds total. The resulting geometry is `main_dim=96`,
`residual_dimension=0`. A 10M-row identity full-cache sidecar was generated
with node length 1,408 bytes, then the current importer hard-linked the native
artifacts into a resident-arena segment. The current MemQG builder used R=32,
`ef_build=100`: build 92.24 seconds, save 10.25 seconds. All preparation time
is excluded.

### GTE768 seal

`gt.ibin` is canonical 1,000 x 100. Its adjacent manifest records deterministic
float64 blocked exhaustive L2, `(distance,base-id)` ordering, a near-tie pass,
and an independent ten-query direct float64 verification. It was retained.

The LASER source is the topology-faithful QG seal
`gte-m768qgfull_R32_MD768.index`, with cache IDs/nodes and rotator. The current
importer wrapped it without rebuilding. Its load reports `main_dim=768`,
`dimension=768`, and `residual_dimension=0`. The missing `_pca.bin` warning is
therefore expected and has no query-transform effect. The cache sidecars cover
all 1,006,717 rows. MemQG was rebuilt with R=32 and `ef_build=100`: build 26.08
seconds, save 4.07 seconds.

Input hashes:

| Input | SHA-256 |
|---|---|
| Deep10M base | `a8653d77b20e63434f4ff9ac5607a3f24c98e5a3ceb7fc6e90abeb5bcf8b5c7e` |
| Deep10M query | `8438fc763f14e0f9741fda15b3e11215aef089ce17d1ad47d53b52a7c9fda5bb` |
| Deep10M source GT (IDs + distances) | `1143a372b9fb64170aa032e084d1659d9c090c57601f0aefdb3e2c2f79f51369` |
| Deep10M canonical GT | `5994e55a274c30cfc339bcf6462847fa77f9157ecfcb2105ae65cd3781d82ffb` |
| GTE768 base | `70aa6e7e7d6f3423e303fa4fa9d31137a3c4f2b296eab1a98c86ad9d370b02fa` |
| GTE768 query | `0f6bae398109b4bb51b1c427bbca8f9b6054d404466ef0130de58e90032568d4` |
| GTE768 exact GT | `501d4ba5b7c680ac12051c4ca5e9fdad684c58a845aca8afbb1edb84b6ca06d0` |

Key native-index hashes:

| Index | SHA-256 |
|---|---|
| Deep10M R32/MD96 | `f4524fd2eb8285460edf114266161dea367d6d885e282fe8c6ad8872f9696d8a` |
| GTE768 QG-seal R32/MD768 | `1fbc9cd0c342697b58f2ad4e3d5c1e95b66b7a223af7e4cfd23c8e080cefd6c8` |

## ef calibration

Calibration used C=1 and a 1,000-query pass. GTE768's 1,000 rows are the full
query set; Deep10M used the first 1,000 rows. Recall was identical in both arm
orders.

| Dataset / arm | ef40 | ef80 | ef160 | ef320 | ef640 | ef1280 |
|---|---:|---:|---:|---:|---:|---:|
| Deep10M MemQG | 0.7754 | 0.8737 | 0.9367 | 0.9679 | 0.9855 | 0.9926 |
| Deep10M LASER | 0.8083 | 0.9069 | 0.9606 | 0.9842 | 0.9931 | 0.9966 |
| GTE768 MemQG | 0.8871 | 0.9268 | 0.9495 | 0.9690 | 0.9836 | 0.9933 |
| GTE768 LASER | 0.8863 | 0.9229 | 0.9528 | 0.9729 | 0.9852 | 0.9919 |

The formal Deep10M points were refined to MemQG
`100,200,400,1280` and LASER `60,120,320,800`; their full 10,000-query curves
bracket all targets. GTE768 used `40,160,640,1280` for both arms. Calibration
CSV/JSON and logs are under `smoke/` and `logs/`.

## Deep10M primary grid

Measured curves:

| Arm | ef | recall@10 | C=1 QPS | C=4 QPS | C=16 QPS |
|---|---:|---:|---:|---:|---:|
| MemQG | 100 | 0.893530 | 18,577 | 72,352 | 239,500 |
| MemQG | 200 | 0.947530 | 10,562 | 41,341 | 140,389 |
| MemQG | 400 | 0.975860 | 5,484 | 21,472 | 78,323 |
| MemQG | 1280 | 0.994010 | 1,615 | 6,382 | 23,883 |
| LASER arena | 60 | 0.869600 | 29,275 | 112,607 | 385,455 |
| LASER arena | 120 | 0.941740 | 17,058 | 66,297 | 218,178 |
| LASER arena | 320 | 0.985230 | 7,129 | 27,889 | 101,218 |
| LASER arena | 800 | 0.995970 | 2,845 | 11,212 | 41,326 |

Iso-recall grid:

| C | target | MemQG ef* / QPS | LASER ef* / QPS | arena / memqg |
|---:|---:|---:|---:|---:|
| 1 | 0.90 | 111.98 / 17,617 | 85.28 / 24,127 | **136.95%** |
| 1 | 0.95 | 217.44 / 10,120 | 157.99 / 15,172 | **149.93%** |
| 1 | 0.99 | 1085.58 / 2,470 | 533.18 / 5,227 | **211.63%** |
| 4 | 0.90 | 111.98 / 68,637 | 85.28 / 93,092 | **135.63%** |
| 4 | 0.95 | 217.44 / 39,609 | 157.99 / 59,002 | **148.96%** |
| 4 | 0.99 | 1085.58 / 9,716 | 533.18 / 20,482 | **210.80%** |
| 16 | 0.90 | 111.98 / 227,625 | 85.28 / 314,964 | **138.37%** |
| 16 | 0.95 | 217.44 / 134,978 | 157.99 / 195,964 | **145.18%** |
| 16 | 0.99 | 1085.58 / 35,911 | 533.18 / 74,618 | **207.79%** |

| Aggregate | Geometric mean |
|---|---:|
| C=1 | **163.18%** |
| C=4 | **162.09%** |
| C=16 | **161.01%** |
| All nine cells | **162.09%** |

Deep10M used a freshly built Vamana R32 LASER topology and a separately built
MemQG topology, as specified. Iso-recall removes recall-level differences but
does not claim byte-identical topology; this is why its advantage is much
larger than the topology-sealed GTE leg.

## GTE768-seal primary grid

Measured curves:

| Arm | ef | recall@10 | C=1 QPS | C=4 QPS | C=16 QPS |
|---|---:|---:|---:|---:|---:|
| MemQG | 40 | 0.887100 | 35,021 | 132,888 | 242,599 |
| MemQG | 160 | 0.949500 | 14,479 | 54,601 | 139,549 |
| MemQG | 640 | 0.983600 | 4,510 | 17,167 | 50,034 |
| MemQG | 1280 | 0.993300 | 2,255 | 8,661 | 24,848 |
| LASER arena | 40 | 0.886300 | 34,520 | 130,416 | 363,205 |
| LASER arena | 160 | 0.952800 | 13,971 | 52,379 | 139,823 |
| LASER arena | 640 | 0.985200 | 4,391 | 16,601 | 45,760 |
| LASER arena | 1280 | 0.991900 | 2,200 | 8,461 | 24,671 |

Iso-recall grid:

| C | target | MemQG ef* / QPS | LASER ef* / QPS | arena / memqg |
|---:|---:|---:|---:|---:|
| 1 | 0.90 | 64.81 / 30,774 | 64.72 / 30,287 | **98.42%** |
| 1 | 0.95 | 167.04 / 14,333 | 154.95 / 14,836 | **103.51%** |
| 1 | 0.99 | 1062.27 / 3,022 | 1098.51 / 2,821 | **93.36%** |
| 4 | 0.90 | 64.81 / 116,704 | 64.72 / 114,339 | **97.97%** |
| 4 | 0.95 | 167.04 / 54,052 | 154.95 / 55,665 | **102.98%** |
| 4 | 0.99 | 1062.27 / 11,555 | 1098.51 / 10,769 | **93.20%** |
| 16 | 0.90 | 64.81 / 221,295 | 64.72 / 317,185 | **143.33%** |
| 16 | 0.95 | 167.04 / 138,236 | 154.95 / 149,229 | **107.95%** |
| 16 | 0.99 | 1062.27 / 33,416 | 1098.51 / 30,652 | **91.73%** |

| Aggregate | Geometric mean |
|---|---:|
| C=1 | **98.34%** |
| C=4 | **97.97%** |
| C=16 | **112.38%** |
| All nine cells | **102.68%** |

GTE768 is above the gate and near parity at C=1/4, while C=16 retains the
phase-1 reversal in LASER's favor. The `.99` cells are lower than the other
targets but remain above 91%. Exact GT, full cache coverage, full-dimensional
geometry, zero residual dimension, and the absence of an active PCA transform
were all checked before accepting this result.

## Paged residency spectrum

The .95 operating points were measured in a fresh same-host run. MemQG ef76
has recall 0.95518; arena ef73 has 0.95502. A paged-kernel calibration selected
ef50, recall 0.95631.

| Arm / residency | Cache rows | ef | recall@10 | C=1 QPS | C=4 QPS | C=16 QPS | vs arena C1/4/16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MemQG | n/a | 76 | 0.95518 | 26,461 | 103,160 | 339,092 | 94.09/94.38/94.68% |
| LASER resident arena | 1,000,000 | 73 | 0.95502 | 28,124 | 109,309 | 358,132 | 100/100/100% |
| LASER paged, full cache | 1,000,000 | 50 | 0.95631 | 11,903 | 47,426 | 142,967 | **42.32/43.39/39.92%** |
| LASER paged, tight cache | 150,000 (15%) | 50 | invalid | blocked | blocked | blocked | checksum fail-closed |

The normal paged loader calculates its budget and then caps it at
`kCacheRatio * N = 15%`, even if a full sidecar and larger DRAM budget are
provided. Tight mode uses the first 150,000 high-degree cache IDs from the
retained sidecar; the open log confirms `online_cache_num: 150000`.

To realize the required all-row-cache/paged-kernel point without changing
engine headers, the harness constructs the public `LaserSegmentSearcher`,
calls the public graph materialization seam once outside measurement, and
continues to invoke `LaserSegmentSearcher::search`, which selects
`QuantizedGraph::search` (the paged kernel). Thus every row can hit cache while
the paged algorithm remains active. This benchmark-only adapter mirrors the
request/result validation used by `LaserSegment`; it is not a production open
configuration because the production loader's 15% cap makes that state
unexpressible.

The tight-cache minimal reproduction is stronger than a noisy throughput
qualification. With one lane, 100 queries, ef50, and a full warmup, its first
forward pass had recall 0.971 on that subset. The reverse call changed query
10 hit 2 from ID 992617 to 297239. A separate ef73 smoke changed query 13 hit
6 from 522080 to 868867. Neither QPS is admitted into the spectrum.

Paged results must also be read against block C's measured I/O scaling ceiling:
the t8/t1 backend ratios were 1.95x for libaio and 4.70x for the thread pool.
This build uses libaio. The all-cache row above performs no misses and therefore
does not measure that ceiling; the tight-cache leg that would expose it is the
one rejected for result nondeterminism.

## Public multi-row batch secondary metric

Both arms use their .95 point. Batch mode issues 64 rows per public multi-row
operation; the same external pool supplies C=1 or C=16 concurrent calls.
Throughput is still counted in queries per second.

| Arm | C | single-row QPS | batch QPS | batch / single | Verdict |
|---|---:|---:|---:|---:|---|
| MemQG | 1 | 26,461 | 26,581 | **1.0046x** | pass |
| MemQG | 16 | 339,092 | 335,530 | **0.9895x** | pass |
| LASER arena | 1 | 28,124 | 28,623 | **1.0177x** | pass |
| LASER arena | 16 | 358,132 | 362,227 | **1.0114x** | pass |

Recall and the full top-10 checksum are identical between single and batch for
each `(arm,ef)`. No cell crosses the pre-registered 5% forensic threshold.

## Collection E2E forensic leg

The first E2E smoke correctly rejected rank-only LASER hits because the
benchmark registration did not retain a million base vectors for exact rerank.
The measured E2E request therefore enables numeric LASER distances on both the
direct and Collection arms. This isolates adapter/routing work instead of
adding a separate exact-distance callback.

The full 10,000-query run was infeasible after the smoke exposed an O(N)
routing path. A reduced first-100-query leg retained CPUs, affinity, warmup,
0.5-second minimum duration, paired forward/reverse order, and three repeats
per order. Construction and the million-row registration are excluded.

| C | segment direct QPS | Collection QPS | Collection / direct | adapter + routing tax |
|---:|---:|---:|---:|---:|
| 1 | 35,049.290 | 5.762 | 0.01644% | **99.98356%** |
| 16 | 431,685.246 | 74.839 | 0.01734% | **99.98266%** |

Both arms have stable per-arm checksums and recall 0.969 on the subset. The
Collection checksum is stable but not byte-equal to the direct sequence; this
harness does not distinguish an order change from a membership change across
the two faces.

The cause is explicit in production code. `RoutingSnapshot::known_rows_for()`
iterates the complete `reverse` map. `fanout_search()` calls it once while
computing `maximum_known_rows` and again for each segment before search. With a
one-million-row, one-segment Collection, every query therefore performs at
least two million ordered-map visits before the LASER call. This is not the F
AnySegment adapter tax and is not fixed here because engine changes are out of
scope.

## Deviations and qualifications

- No primary-grid semantic redline changed. Deep10M and GTE768 primary data
  were collected before the secondary CLI extensions were added, using the
  phase-1 harness source unchanged on engine base `e8e2462`.
- As in phase 1, `numactl` was unavailable. CPU affinity plus first touch was
  used, without explicit memory binding.
- Deep10M GT required a format-only extraction. Its exactness is supported by
  the retained source distances and a ten-query full-corpus float64 audit; a
  new all-query brute-force run was unnecessary.
- GTE768 emits a missing PCA-sidecar warning. Full main dimension and zero
  residual dimension make the sidecar dormant; this was checked rather than
  hidden.
- The tight-cache paged formal table is absent by design: two serial
  reproductions violated checksum stability, so reporting their QPS would
  violate the fail-closed contract.
- The all-cache paged state uses a benchmark-only adapter because engine open
  hard-caps paged cache residency at 15%. Search still enters the paged kernel;
  construction/materialization is excluded.
- E2E uses only the first 100 queries and numeric-distance return. It keeps the
  timing/repetition protocol but is a forensic table, not a replacement for
  the 10,000-query main metric. The exact reason for truncation is the O(N)
  production routing behavior above.
- The Deep10M native builder's verbose console stream was not redirected into
  the raw root. Its exact command, geometry, final timings, output artifacts,
  and hashes are recorded here; sidecar and importer/MemQG logs are retained.
- Preparation logs for the first GTE importer/MemQG invocation were observed
  live but not redirected. The artifacts, measured build/save times, formal
  logs, and hashes are retained.

## Raw data, artifacts, and reproducibility

Raw root:
`/home/huangliang/workspace/alaya-dev/data/laser-update/parity-lanes-phase2-20260719/`

Primary and secondary files:

| File | SHA-256 |
|---|---|
| `raw/deep10m-parity-lanes.csv` | `66301d718bd0d930b0053eb3cd8043a29ad356d355281835d6ececcfd018f357` |
| `raw/deep10m-parity-lanes.json` | `230a54113969aa8982bafa944a25ad4caa911b9c508696f3801a51851ab8701b` |
| `raw/gte768-parity-lanes.csv` | `49579503ead6ed84f76fce5c271bcebb4bdb584017249913392e34057e5f3971` |
| `raw/gte768-parity-lanes.json` | `8684ca240bd6af6a1e9fd913117cfe39ea78e12e89d9b0ebb16ac735d18ceede` |
| `raw/sift1m-secondary-single.csv` | `48bf9fa7dd39b64d8e98acf099113ac4543ceb94efa0a332d2027b5471065797` |
| `raw/sift1m-secondary-single.json` | `94d28b0afe0760a8909d5ab1cf1145de61366314895a40d70a07a7e4340cc14a` |
| `raw/sift1m-paged-full.csv` | `360d7c9fdeb81741ad46f9534769d745e5bf376a05fb7a6da2e05197f86ecb14` |
| `raw/sift1m-paged-full.json` | `73eb62dd1f18207fa9f644bce40ce146ac2a561a9d78818063352ce304910330` |
| `raw/sift1m-secondary-batch.csv` | `c58525554cc5ae12aa700606b21cefb8258b61e0e07a9a71f3d81664f8c98bd2` |
| `raw/sift1m-secondary-batch.json` | `390d52c6bcca0b2035c8e9a1b701646e7b679bbebb0ef4578d89f25c2baf9296` |
| `raw/sift1m-e2e-100q.csv` | `33423e09adebb682737d39ce6868ad61cadfd85939cf857832eb8d7e791ca71d` |
| `raw/sift1m-e2e-100q.json` | `94ad88cabe5dfef8d182cf8ba4c45753f0ad7c7733d0f5353aeb8a55e85b37ed` |

All six completed CSV/JSON pairs passed JSON parsing and CSV/JSON record-count
agreement. The main files each contain 144 records, 48
`(arm,C,ef,order)` cells, and multiplicity three. Secondary single/paged,
batch, and E2E contain 36, 36, 24, and 24 records respectively, again with
three records per arm/C/ef/order cell. Recall and checksum are invariant inside
every completed `(arm,ef)` group.

Artifact paths:

- Deep native prefix:
  `indices/native/deep10m/deep10m-r32`; full-cache prefix:
  `indices/native/deep10m/deep10m-r32full`.
- Deep imported segment:
  `indices/laser/deep10m/segments/seg_00000001`; MemQG:
  `indices/memqg-deep10m-r32-efb100.qg`.
- GTE source seal:
  `/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715/gte-stage/gte-m768qgfull_R32_MD768.index`;
  imported segment: `indices/laser/gte768/segments/seg_00000001`; MemQG:
  `indices/memqg-gte768-r32-efb100.qg`.
- SIFT tight-cache imported segment:
  `indices/laser/sift1m-tight/segments/seg_00000001`. The phase-1 arena and
  MemQG artifacts are reused from
  `/home/huangliang/workspace/alaya-dev/data/laser-update/parity-lanes-20260718/indices/`.

`scripts/fullcache-probe/summarize_parity_lanes.py` validates multiplicity,
recall/checksum invariance, paired-order medians, interpolation coverage, and
prints the measured/iso-recall Markdown tables. Its outputs are retained as
`logs/deep10m-summary.md` and `logs/gte768-summary.md`.

## Verification and remaining work

Verification completed:

- Release rebuild of `parity_lanes_benchmark` with `BUILD_TESTING=ON`.
- Extended single, batch, paged-full, paged fail-closed, and Collection E2E
  smokes.
- JSON/CSV structure, multiplicity, recall, checksum, and summary-script
  audits described above.
- Focused Release ctest:
  `any_segment_test`, `test_laser_arena_reentrancy`,
  `test_unified_laser_admission`, and `collection_facade_test`; 4/4 passed.
- `clang-format --dry-run --Werror`, Python byte compilation,
  `git diff --check`, and `uvx pre-commit run -a`; every hook passed.

Two engine issues remain for main-session adjudication:

1. paged AIO search can return different top-10 IDs for identical serial
   requests when cache misses occur; and
2. Collection routing derives known row counts by scanning the full reverse
   map on every query.

Neither issue changes the positive resident-arena primary verdict. Both block
turning their respective secondary stories into publication-grade full-grid
claims until fixed and remeasured.
