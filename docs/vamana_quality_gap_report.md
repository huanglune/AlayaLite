# ShardVamanaBuilder Recall Gap Analysis Report

**Date**: 2026-04-07
**Dataset**: GIST-1M (1,000,000 vectors, 960 dimensions)
**Build Parameters**: R=64, L=200, alpha=1.2

---

## 1. Problem Statement

AlayaLite's LASER build pipeline produces indexes with ~6% lower recall compared to the original Laser pipeline (which uses DiskANN's official `build_memory_index` to build the Vamana graph).

| Build Method | Recall@10 (ef=80) | Recall@10 (ef=200) |
|---|---|---|
| DiskANN Vamana + AlayaLite QGBuilder | **91.50%** | **97.19%** |
| External Build (gpu04 pre-built) | 91.58% | 97.40% |
| AlayaLite ShardVamana (single shard + backfill) | 85.36% | 93.83% |
| AlayaLite ShardVamana (single shard, no backfill) | 82.40% | 91.84% |
| AlayaLite ShardVamana (3 shards, no backfill) | 83.10% | 91.95% |

## 2. Root Cause Isolation

Through controlled experiments, we confirmed:

- **PCA transform**: No issue (DiskANN Vamana + our PCA = 91.50%)
- **Medoid generation**: No issue
- **QGBuilder (quantized graph)**: No issue
- **Partitioning/merge strategy**: Minor impact (<0.5%)
- **Neighbor backfill**: +3% recall improvement, but non-standard behavior

**Conclusion: 100% of the recall gap originates from `ShardVamanaBuilder`'s Vamana graph quality.**

## 3. Algorithmic Differences (Ranked by Impact)

### 3.1 [CRITICAL] Pruning: Single-pass vs Multi-pass Alpha Escalation

**DiskANN** (`src/index.cpp`, `occlude_list`): Uses a graduated alpha-escalation strategy. Starts with `cur_alpha = 1.0`, selects the strictest (non-diverse) neighbors first, then relaxes to `cur_alpha *= 1.2` in successive passes until the target alpha is reached or the degree is filled.

```cpp
// DiskANN: Multi-pass alpha escalation
float cur_alpha = 1;
while (cur_alpha <= alpha && result.size() < degree) {
    for (auto iter = pool.begin(); iter != pool.end(); ++iter) {
        if (occlude_factor[t] > cur_alpha) continue;
        result.push_back(*iter);
        // Update occlude factors for remaining candidates
        occlude_factor[t] = max(occlude_factor[t], iter2->distance / djk);
    }
    cur_alpha *= 1.2f;
}
```

**AlayaLite** (`shard_vamana_builder.hpp:510-541`, `robust_prune`): Single-pass with full alpha=1.2 applied uniformly. A single dominator causes immediate rejection.

```cpp
// AlayaLite: Single-pass, full alpha
for (const auto &candidate : candidates) {
    bool dominated = false;
    for (const auto &chosen : selected) {
        if (alpha * distance(candidate, chosen) < candidate.distance) {
            dominated = true;
            break;
        }
    }
    if (!dominated) selected.push_back(candidate);
}
```

**Impact**: DiskANN's approach guarantees that the closest, most directly useful neighbors are always selected first (at alpha=1.0). Only then does it add diversity-expanding neighbors (at higher alpha). AlayaLite's single-pass applies alpha=1.2 from the start, which can aggressively reject nearby candidates in favor of distant diverse ones, degrading navigation quality. **This is the primary cause of the ~6% recall loss.**

Additionally, DiskANN tracks a per-candidate `occlude_factor` (the maximum occlusion ratio across all selected neighbors), whereas AlayaLite exits on the first dominator — meaning a candidate dominated by one selected neighbor is immediately rejected, even if it would survive a more nuanced multi-factor analysis.

### 3.2 [SIGNIFICANT] Candidate Pool Size for Pruning

**DiskANN**: The candidate pool passed to `prune_neighbors` includes **all expanded nodes** during the greedy search (not just the top-L results). Pool is reserved at `3*L + R` and can contain significantly more candidates.

**AlayaLite**: The `greedy_search` returns only the top `min(size, ef)` candidates (line 501), which are then merged with existing neighbors. The CandidateList capacity is `ef + max_degree`.

**Impact**: DiskANN's pruning has a much richer candidate set (typically 600+ candidates for L=200, R=64) vs AlayaLite's ~264. More candidates means the alpha-escalation pruning can find better diversity-quality tradeoffs.

### 3.3 [SIGNIFICANT] GRAPH_SLACK_FACTOR Over-provisioning

**DiskANN**: Uses `GRAPH_SLACK_FACTOR = 1.3`, allowing neighbor lists to grow to ~83 edges (1.3 * 64) during construction before triggering pruning.

**AlayaLite**: Prunes immediately at `max_degree = 64` in `add_reverse_edge`.

**Impact**: Over-provisioning during construction improves graph connectivity and search quality during the build phase. When pruning is finally triggered, it has more edges to select from, producing better final neighborhoods. A post-build cleanup pass then ensures the final graph respects the R=64 degree constraint.

### 3.4 [MODERATE] Post-Build Cleanup Pass

**DiskANN**: After the main link loop, runs a final pass to prune any nodes still exceeding the degree bound (due to GRAPH_SLACK_FACTOR).

**AlayaLite**: No equivalent cleanup pass.

**Impact**: Ensures the final graph is strictly degree-bounded while retaining the benefits of over-provisioned construction.

### 3.5 [MODERATE] Build Pass Structure

**DiskANN**: Runs **one pass** with sequential visit order. The multi-pass alpha escalation within `occlude_list` handles both strict and relaxed pruning internally.

**AlayaLite**: Runs **two passes** — first with alpha=1.0, second with alpha=1.2. The processing order is a random permutation.

**Impact**: AlayaLite's two-pass approach is a reasonable approximation of DiskANN's internal alpha escalation, but the lack of per-candidate `occlude_factor` tracking in `robust_prune` undermines its effectiveness.

### 3.6 [MINOR] Greedy Search Termination

**DiskANN**: Exhaustively expands all unexpanded nodes in the priority queue.

**AlayaLite**: Has an early termination condition (line 483-485) that stops when the next unexpanded candidate is worse than the worst in a full list.

**Impact**: Could cause premature search termination in some cases, though the practical frequency is unclear.

### 3.7 [MINOR] No Duplicate ID Check in CandidateList

**DiskANN**: `NeighborPriorityQueue::insert()` deduplicates by ID during binary search.

**AlayaLite**: `CandidateList::insert()` does not check for duplicate IDs. Mitigated by the external `VisitedList` during search, but not during merge operations.

## 4. Experimental Evidence

### Build Statistics

| Metric | DiskANN | AlayaLite (no backfill) | AlayaLite (with backfill) |
|---|---|---|---|
| Vamana file size | 168 MB | 164 MB | 248 MB |
| Average degree | 42.8 / 64 (67%) | ~42 / 64 (est.) | 64 / 64 (100%) |
| Build time (96 threads) | 324 s | ~800 s (Vamana phase) | ~900 s |

### Recall Progression (all on gpu04, same ground truth)

| Experiment | Change | Recall@10 (ef=80) |
|---|---|---|
| Baseline (3 shards, no backfill) | - | 83.10% |
| Single shard, no backfill | Removed shard/merge overhead | 82.40% (-0.7%) |
| Single shard + backfill | Filled all 64 neighbor slots | 85.36% (+2.96%) |
| DiskANN Vamana + our QGBuilder | Replaced graph builder | **91.50% (+9.1%)** |

**Conclusion**: Backfill recovers ~3%, but the remaining ~6% requires matching DiskANN's pruning quality.

Note: DiskANN's average degree is 42.8/64 — it deliberately does **not** fill all neighbor slots. The unfilled slots are a feature of robust pruning, not a bug. Our backfill approach actually degrades navigability by inserting low-quality distant neighbors.

## 5. Recommended Fixes (Priority Order)

### Fix 1: Implement Multi-pass Alpha Escalation in `robust_prune`

Replace the single-pass domination check with DiskANN's `occlude_list` algorithm:

```
function occlude_prune(node, candidates, max_degree, alpha):
    sort candidates by distance
    truncate to MaxC (750)
    occlude_factor = [0.0] * len(candidates)
    result = []
    
    cur_alpha = 1.0
    while cur_alpha <= alpha and len(result) < max_degree:
        for each candidate c (in distance order):
            if occlude_factor[c] > cur_alpha: skip
            result.append(c)
            for each remaining candidate t:
                djk = dist(c, t)
                occlude_factor[t] = max(occlude_factor[t], dist_to_query[t] / djk)
        cur_alpha *= 1.2
    
    return result
```

**Expected impact**: +4-5% recall

### Fix 2: Expand Candidate Pool for Pruning

Pass all expanded (visited) nodes from `greedy_search` to `robust_prune`, not just the top-L results. Reserve pool at `3*L + R`.

**Expected impact**: +1-2% recall (compounds with Fix 1)

### Fix 3: Implement GRAPH_SLACK_FACTOR

Allow neighbor lists to grow to `1.3 * R` during construction. Add a post-build cleanup pass to enforce the degree constraint.

**Expected impact**: +0.5-1% recall

### Fix 4: Remove Backfill

The backfill logic we added is non-standard and counter-productive for high-quality graphs. Once Fixes 1-3 are implemented, remove the backfill code from `robust_prune`.

### Fix 5 (Optional): Remove Early Termination in Greedy Search

Match DiskANN's exhaustive expansion for maximum search quality during construction.

## 6. Verification Plan

1. Implement Fix 1 (alpha escalation) first, measure recall
2. Add Fix 2 (expanded pool), measure incremental gain
3. Add Fix 3 (slack factor), measure incremental gain
4. Remove backfill, confirm recall is maintained
5. Final benchmark: build + search on GIST-1M, transfer to gpu04, compare with external build

**Target**: Recall within 0.5% of DiskANN at all ef values.
