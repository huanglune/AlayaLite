# QPatch phase 1: lock-free RaBitQ preencode

Date: 2026-07-13 (America/Los_Angeles)

The control and QPatch runs used independent copies of the same 990k SIFT
R32/MD128 baseline.  Baseline index SHA-256:

```text
0d83d8e4ac748fc52455037b5e973f8097c4bc54c49bd190b2e0941c69464dee
```

The churn command matches the requested 10-round configuration, with only
`--qpatch 0` versus `--qpatch 1` changed.  Host: dual-socket AMD EPYC 9554,
256 logical CPUs, Linux 6.8.0-59-generic.

## Final paired run

| Round | Legacy QPS | Legacy seconds | QPatch QPS | QPatch seconds | Delta |
|---:|---:|---:|---:|---:|---:|
| 1 | 9,885.90 | 0.1001 | 9,859.74 | 0.1004 | -0.26% |
| 2 | 9,211.16 | 0.1075 | 9,228.03 | 0.1073 | +0.18% |
| 3 | 9,621.50 | 0.1029 | 9,311.69 | 0.1063 | -3.22% |
| 4 | 9,780.64 | 0.1012 | 9,415.91 | 0.1051 | -3.73% |
| 5 | 9,559.94 | 0.1036 | 9,737.48 | 0.1017 | +1.86% |
| 6 | 9,696.61 | 0.1021 | 9,828.21 | 0.1007 | +1.36% |
| 7 | 10,111.60 | 0.0979 | 10,581.70 | 0.0936 | +4.65% |
| 8 | 9,600.30 | 0.1031 | 10,137.30 | 0.0977 | +5.59% |
| 9 | 8,982.03 | 0.1102 | 8,733.68 | 0.1134 | -2.76% |
| 10 | 3,153.91 | 0.3139 | 3,150.76 | 0.3142 | -0.10% |

Round 10 triggers the configured cache high-to-low writeback/eviction.  For
rounds 1-9, aggregate QPS (total inserts divided by total insertion-phase
time) is 9,594.68 legacy versus 9,620.78 QPatch: **+0.27%**.  Across all ten
rounds it is 7,967.58 versus 7,981.75: **+0.18%**.  This is below the 3%
material-improvement threshold.

QPatch prepared 306,914 intents and installed 227,568, with zero generation
fallbacks.  Thus 25.85% of preencodes were discarded by alpha/admission and
QPatch executed 1.349 encodes per installed edge.  The remaining locked work
still includes admission FastScan, block unpack/repack, PID/factor mutation,
and page/cache synchronization.  The result indicates RaBitQ encoding alone
is not the dominant throughput limiter at this concurrency; batching patches
per page is the more promising next experiment.

## Recall/correctness

Mean recall over post-update rounds 1-10 was 0.977764 legacy and 0.977732
QPatch, a delta of **-0.0032 percentage points**, within the requested
plus/minus 0.01pp bound.  Individual asynchronous beam evaluations are noisy:
round 0, before any update path ran, differed by 0.041pp between the paired
runs.  The deterministic unit test therefore also compares complete terminal
index bytes; legacy and QPatch are byte-identical in both inline and staged
single-thread modes.

Raw logs:

```text
/home/huangliang/workspace/alaya-dev/data/laser-update/qpatch-20260713/final_legacy.log
/home/huangliang/workspace/alaya-dev/data/laser-update/qpatch-20260713/final_qpatch.log
```
