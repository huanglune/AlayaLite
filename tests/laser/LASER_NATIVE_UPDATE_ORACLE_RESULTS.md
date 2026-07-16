# LASER native-update oracle experiments (SIFT 1M, R=32)

Date: 2026-07-13

This report records the three read-only oracle experiments used to evaluate the
"one row is a distance oracle" premise. The implementation adds
`fastscan_oracle` and `twohop_oracle` modes to
`bench_laser_update_sift`; it does not modify `qg_updater.hpp`.

## Common setup

- Base/query/GT: `data/sift-fbin/{sift_base.fbin,sift_query.fbin,sift_gt.ibin}`
- Static population: first 990,000 SIFT vectors
- Graph: R=32, L=200, `ef_indexing=200`, alpha=1.2, rotator seed 42
- Search measurement: 32 threads, beam=16, one warm-up plus three timed/scored runs
- Artifact prefix: `data/laser-update/oracle-native-20260713/idx/static990k`
- All data-intensive commands were serialized with
  `flock data/.aio.lock`.

## Oracle 1: static-rebuild recall ceiling

The build completed in 74.3 s. The Vamana input graph contained 30,524,303
out-edges (mean degree 30.83).

| ef | recall@10 | best QPS | scored recall runs |
|---:|---:|---:|:---|
| 40 | 0.944703 | 14,425.2 | 0.94467, 0.94403, 0.94541 |
| 60 | 0.970783 | 12,633.6 | 0.97128, 0.97061, 0.97046 |
| 80 | 0.980893 | 11,642.8 | 0.98094, 0.98056, 0.98118 |
| 100 | **0.986423** | 10,419.5 | 0.98664, 0.98626, 0.98637 |
| 120 | 0.989933 | 9,635.2 | 0.99018, 0.98959, 0.99003 |
| 200 | 0.995857 | 7,600.9 | 0.99584, 0.99583, 0.99590 |

Conclusion: at the churn protocol's ef=100, the static-rebuild reference is
about 0.9864, consistent with the expected 0.987 neighborhood. Increasing ef
still exposes substantial graph quality: 0.9899 at ef=120 and 0.9959 at
ef=200.

The ten-round churn run below ended at recall 0.98649. That is inside the
static run-to-run interval and is not a significant exceedance. Also, the
round-10 live source set is `[9900, 999900)`, whereas this static index is
`[0, 990000)`, so the initial static number is a protocol reference rather
than a mathematical upper bound for the changed final population. A strict
round-10 ceiling would require rebuilding the final live source set.

## Oracle 2: cross-row FastScan ranking

Sampling uses directed graph relations:

- 1-hop: choose `a` from `N(u)`.
- strict 2-hop: choose `u -> b -> a` with `a` absent from `N(u)`.
- Scan row `a` with vector `u` as the query. The query PID itself is removed
  from `N(a)`, matching maintenance filtering and avoiding a trivial
  zero-distance hit.

Each row's FastScan order is compared with exact squared-L2 order over the same
candidates. Results are macro averages over 1,000 accepted samples per
relation.

| relation | mean candidates | top-1 | top-4 | top-8 | Spearman | median relative error | normalized MAE | invalid estimates | mean `||u-a||` |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1-hop | 30.452 | 0.556 | 0.70425 | **0.789** | 0.89443 | 5.712% | 6.761% | 0.0788% | 235.837 |
| strict 2-hop | 31.258 | 0.709 | 0.73900 | **0.797** | 0.89451 | 5.742% | 6.759% | 0.0736% | 290.934 |

The ordinary mean relative error for 1-hop is not useful: 0.0197% of its exact
distances are zero/near-zero, making percentage error singular. Median error
and `sum(abs(error))/sum(exact distance)` are stable and agree across both
relations.

Conclusion: the cross-row estimator is suitable for a shortlist (top-4 or,
preferably, top-8), but not as a reliable top-1 oracle. Moving the anchor from
one hop to a strict two-hop relation increases mean anchor distance by 23%, yet
does not degrade top-8 overlap, rank correlation, or normalized distance
error. The core FastScan premise therefore passes.

## Oracle 3: two-hop candidates versus ef=200 beam candidates

The static artifact was copied and subjected to ten sliding-window churn
rounds (990 deletes/inserts per round), using alpha backlinks, consolidation,
`r_target=28`, no garden, and no PID reuse. The resulting v2 index has 999,900
allocated rows and 990,000 live rows. Its round-10 recall@10 at ef=100 was
0.98649.

The oracle recomputed indegrees from the v2 graph. Among live rows, p10 was 16;
86,188 rows satisfied the strict condition `indegree < 16`. It sampled 1,000
of those rows with seed 42.

For each `u`:

1. The beam arm mirrors `garden_row`'s `search_for_insert`: ef=200 FastScan
   frontier, exact distance for expanded rows, and pool cap 300.
2. The two-hop arm ranks the live outgoing anchors using row `u`, scans each
   anchor row with query `u`, and unions each row's top-T candidates.
3. The exact-control arm uses the same anchors but exact distances to rank
   each `N(a)`. This isolates FastScan ranking loss from graph-topology loss.
4. RobustPrune uses alpha=1.2 and target 28. As in `garden_row`, both prune
   pools also include `u`'s current live outgoing neighbors.

`A=32` means all live outgoing anchors; low-indegree samples had only 27.011
such anchors on average. Candidate coverage is the macro mean of
`|C_2hop intersect C_beam| / |C_beam|`. The beam candidate set contained
135.777 rows on average.

| A | T | mean two-hop size | FastScan candidate coverage | exact-control coverage | final edge recall vs beam | exact-control edge recall | edge Jaccard vs beam |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 13.575 | 9.970% | 9.722% | 55.910% | 56.381% | 42.408% |
| 4 | 8 | 27.791 | 17.671% | 17.690% | 61.857% | 62.369% | 47.820% |
| 8 | 4 | 25.532 | 17.344% | 16.900% | 61.771% | 62.713% | 47.885% |
| 8 | 8 | 52.487 | 28.262% | 28.423% | 68.512% | 69.188% | 54.296% |
| all (mean 27.011) | 4 | 82.891 | 34.933% | 34.690% | 73.171% | 74.378% | 59.394% |
| all (mean 27.011) | 8 | 169.402 | **46.912%** | **47.286%** | **77.533%** | **78.046%** | **64.518%** |

For the all-anchor/T=8 arm, median per-row candidate coverage was 45.833% and
micro coverage was 44.494%. FastScan's candidate set retained 80.392% of the
same-anchor exact two-hop set.

Conclusion:

- Two-hop locality is the limiting factor. At all anchors/T=8, replacing
  FastScan ranking with exact ranking changes beam-candidate coverage by only
  +0.374 percentage points and final edge recall by +0.513 points.
- Cardinality is not the problem: the FastScan two-hop union has 169.4 unique
  candidates versus 135.8 beam candidates, yet covers only 46.9% of them.
  Beam search reaches candidates outside the local two-hop neighborhoods.
- A two-hop-only implementation is not a fidelity-equivalent replacement for
  `garden_row`: even the largest tested arm changes about 22.5% of the final
  beam-selected edges.
- It is still viable as a cheap first-stage or partial repair. The recommended
  starting point is all live anchors with T=8, followed by beam fallback when
  the candidate union or post-prune gain is insufficient. A=4/8 is too lossy
  for an unconditional replacement.

## Reproduction commands

The relevant mode invocations (with the common `flock` prefix omitted here for
readability) are:

```text
bench_laser_update_sift build --base sift_base.fbin --n 990000 \
  --prefix static990k --R 32 --L 200 --ef_indexing 200 --threads 32 --seed 42

bench_laser_update_sift eval --prefix static990k --n 990000 --live_max 990000 \
  --query sift_query.fbin --gt sift_gt.ibin --R 32 --threads 32 --beam 16 \
  --topk 10 --runs 3 --efs 40,60,80,100,120,200

bench_laser_update_sift fastscan_oracle --prefix static990k \
  --base sift_base.fbin --R 32 --samples 1000 --seed 42

bench_laser_update_sift twohop_oracle --prefix churn990k \
  --base sift_base.fbin --R 32 --samples 1000 --seed 42 \
  --ef_maintenance 200 --prune_cap 300 --r_target 28 --alpha 1.2
```
