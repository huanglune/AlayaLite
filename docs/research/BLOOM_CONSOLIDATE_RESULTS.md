# Bloom/PID-only consolidation results

Date: 2026-07-14 (America/Los_Angeles)

The control and Bloom runs used independent copies of the same 990k SIFT
R32/MD128 index. Baseline index SHA-256:

```text
0d83d8e4ac748fc52455037b5e973f8097c4bc54c49bd190b2e0941c69464dee
```

Both runs used 10 churn rounds of 990 delete/insert pairs, 32 insertion and
maintenance threads, `r_target=28`, splice enabled, a 65,536-page write cache,
and checkpointed every round. The only control variable was
`--bloom_consolidate 0` versus `--bloom_consolidate 1`.

## Consolidation time

| Round | Control (s) | Bloom (s) | Change |
|---:|---:|---:|---:|
| 1 | 1.733210 | 0.248157 | -85.68% |
| 2 | 1.097560 | 0.407595 | -62.86% |
| 3 | 1.189140 | 0.358562 | -69.85% |
| 4 | 1.049210 | 0.265128 | -74.73% |
| 5 | 1.086000 | 0.271903 | -74.96% |
| 6 | 0.994549 | 0.279930 | -71.85% |
| 7 | 0.953698 | 0.316162 | -66.85% |
| 8 | 0.949940 | 0.260371 | -72.59% |
| 9 | 0.982231 | 0.252685 | -74.27% |
| 10 | 1.050570 | 0.252627 | -75.95% |

All-round mean fell from **1.108611 s** to **0.291312 s**, a **73.72%**
reduction. The medians were 1.049890 s and 0.268516 s. Excluding the first
round, the means were 1.039211 s and 0.296107 s, a 71.51% reduction.

The Bloom path selected 40,517 rows per round on average (4.093% of 990k;
range 33,519--46,379). Its mean phase breakdown was 72.78 ms for PID scanning,
191.96 ms for full processing of selected rows, and 11.67 ms for final overlay
merge/writeback bookkeeping.

## Effective update throughput

Define effective throughput as:

```text
pairs/s = 9900 / sum_round(consolidate_s + 990 / insert_qps)
```

Across all ten rounds, throughput rose from **806.63 to 1733.41 pairs/s**,
or **+114.90%**. Excluding round 1, it rose from 853.19 to 1663.61 pairs/s,
or +94.99%.

## Recall and persistence

Mean post-update recall over rounds 1--10 was 0.977757 for control and
0.977802 for Bloom: **+0.0045 percentage points**, inside the requested
plus/minus 0.01 pp bound. Individual asynchronous beam-search rounds are
noisy, so both terminal indexes were also reopened in independent processes.
The three-run `ef=100` eval means were 0.977527 and 0.977607, a **+0.0080 pp**
difference. This also verifies that MAP_SHARED cache-miss row updates survived
checkpoint and independent reload.

The Release updater suite passes all 28 tests. Its maintenance equivalence
case runs the Bloom path with four threads and compares terminal page hashes,
consolidated-row counts, and cache-watermark behavior against the legacy path.
The same maintenance case also passes the GCC ThreadSanitizer build (with the
existing sanitizer-only serialization used to avoid libgomp barrier-modeling
false positives).

Raw logs:

```text
/home/huangliang/workspace/alaya-dev/data/laser-update/bloom-20260714/final_control.log
/home/huangliang/workspace/alaya-dev/data/laser-update/bloom-20260714/final_bloom.log
/home/huangliang/workspace/alaya-dev/data/laser-update/bloom-20260714/final_control_eval.log
/home/huangliang/workspace/alaya-dev/data/laser-update/bloom-20260714/final_bloom_eval.log
```
