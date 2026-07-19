# Delayed Bloom consolidation results

Date: 2026-07-14 (America/Los_Angeles)

Both arms started from independent copies of the same 990k SIFT R32/MD128
static checkpoint. Each ran 10 rounds of 990 delete/insert pairs with 32
insert/maintenance threads, `r_target=28`, Bloom consolidation, splice enabled,
a 65,536-page write cache, and a checkpoint after every round. The only control
variable was `--consolidate_every 1` versus `--consolidate_every 5`.

## Bloom bottleneck breakdown

For `consolidate_every=1`, the new per-call instrumentation reported these
10-round means:

| Phase | Mean | Share of internal total |
|---|---:|---:|
| Bloom construction | 1.96 ms | 0.69% |
| PID scan + Bloom/exact check | 81.06 ms | 28.50% |
| Candidate-row repair | 188.70 ms | 66.33% |
| Other finalization/accounting | 12.74 ms | 4.48% |
| Total | 284.47 ms | 100.00% |

The scan passed 40,511 of 994,455 rows per call on average (4.074%) and
skipped 953,944 (95.926%). Repair, not Bloom construction, is now the dominant
cost. Eliminating the measured scan completely would cap the direct
consolidation speedup at about 1.40x; a PID-only small-read experiment therefore
also needs to account for syscall and page-cache granularity rather than assume
that a 128-byte `pread` costs 1/32 of a 4 KiB page access.

## Delayed consolidation

| Metric | Every round | Every 5 rounds | Change |
|---|---:|---:|---:|
| Consolidate calls | 10 | 2 | -80.00% |
| Internal consolidation total | 2.845 s | 0.683 s | -75.98% |
| Bench `consolidate_s` total | 2.989 s | 0.765 s | -74.41% |
| Mean post-round recall | 0.986338 | 0.986433 | +0.0095 pp |
| Final online recall | 0.986250 | 0.986210 | -0.0040 pp |
| Reopened 3-run recall | 0.986420 | 0.986240 | -0.0180 pp |

The two delayed calls repaired 195,223 rows at round 5 and 158,610 rows at
round 10. Their mean internal duration was 341.72 ms, versus 284.47 ms for the
per-round arm, but the lower call frequency reduced aggregate internal time by
75.98%. Online mean, final online, and independently reopened deltas all remain
within 0.02 percentage points, with no consistent directional recall signal.

Raw logs:

```text
/home/huangliang/workspace/alaya-dev/data/laser-update/consolidate-every-20260714/every1.log
/home/huangliang/workspace/alaya-dev/data/laser-update/consolidate-every-20260714/every5.log
/home/huangliang/workspace/alaya-dev/data/laser-update/consolidate-every-20260714/every1_eval.log
/home/huangliang/workspace/alaya-dev/data/laser-update/consolidate-every-20260714/every5_eval.log
```
