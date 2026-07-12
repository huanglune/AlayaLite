# Hot-path performance baseline

This directory locks the g03 baseline for SIMD distance/FHT kernels, the in-memory RaBitQ QG path, and the LASER disk QG build/search path. The runners perform benchmark work through SSH; dataset inputs and generated indexes remain under `/md1` on g03.

## Re-run

From the repository root:

```bash
scripts/perf_baseline/run_baseline.py \
  --host g03 \
  --data-root /md1/huangliang/data \
  --remote-tmp /md1/huangliang/tmp/perf-baseline \
  --output /tmp/candidate-g03.json
```

Use `--collect-existing` only to rebuild JSON from a completed raw run. The runner recreates the deep1M QG index in the remote tmp directory, executes five external rounds, pins CPU and memory to NUMA node 0, and emits structured JSON. It never writes generated artifacts into the dataset root.

Requirements are passwordless SSH, `numactl`, `/usr/bin/time`, the Release binaries at the same NFS path on the target, and deep1M under the selected data root.

The LASER baseline additionally requires the uv-managed worktree venv and a
`BUILD_PYTHON=ON` build. Re-run it with:

```bash
.venv/bin/python scripts/perf_baseline/run_laser_baseline.py \
  --host g03 \
  --data-root /md1/huangliang/data \
  --tmp-root /md1/huangliang/tmp/perf-baseline/14ad5369-laser \
  --output /tmp/candidate-g03-laser.json
```

The LASER runner converts deep1M fvecs/ivecs inputs to fbin/ibin under the
temporary directory, creates a fresh R64/MD64 index, then performs five
external warm-process rounds at EF 100/200/300. `--collect-existing` only
re-parses a completed raw run. Build and search are pinned to NUMA node 0;
search is single-threaded with beam width 16 and one full-query warmup.

## Gate contract

Compare identical names and parameters on the same host, NUMA binding, Release toolchain/flags, dataset, and thread setting. Use medians, not the best run:

- SIMD: compare each `name`'s `median` (`ns_per_call`, lower is better). Each value is the median of five process runs; each process performs 1,000 warmups and 100,000 timed calls per implementation/dimension. A candidate fails when `(candidate / baseline - 1) * 100 > 3`.
- QG search: compare `qps_median` at every matching ef (higher is better), while requiring recall not to decrease. A candidate fails when `(baseline / candidate - 1) * 100 > 3`. Each external process reports the mean of its three built-in query rounds; the locked value is the median of five process results.
- QG build: compare `estimated_build_duration_s` (lower is better). It is cold total wall time minus median warm/search-only wall time. Peak RSS must be reviewed separately and cannot be silently omitted.
- Index correctness: the SHA-256 is provenance, not a performance metric. A changed hash is expected only when index serialization or construction changes and must be explained.
- LASER build: compare `build_duration_s` (lower is better) and review `build_peak_rss_kb`. The measurement covers the complete unified `Index.fit` pipeline, including PCA, medoids, Vamana, and disk QG construction.
- LASER search: compare every EF's `qps_median` while requiring `recall_at_10_median` not to decrease. Each locked value is the median of five external processes; each process warms once and times one complete 1,000-query pass. Review `search_peak_rss_kb` separately.
- LASER artifacts: `artifact_manifest_sha256` covers the sorted `(filename, size, SHA-256)` manifest. The main disk QG and every supporting artifact also have individual hashes. Hash changes require explanation but are not themselves performance failures.

p50/p95/p99 in SIMD are percentiles across the five process-level `ns_per_call` values. The current QG binary exposes only aggregate QPS/recall, not per-query latency, so QG p50/p95/p99 are unavailable rather than fabricated.

Treat a regression over 3% in any gated row as rollback-worthy unless a reviewed exception documents measurement noise or an intentional tradeoff. Re-run both baseline and candidate when g03 pre-run load average exceeds 10; the captured load is in `environment`.

## Coverage decisions

- `rabitq_performance_test` hard-codes deep1M L2. It has no gist1m mode, so gist1m is recorded as unsupported by this target.
- LASER uses the repository's unified Python CLI because no LASER C++ perf target exists. deep1M is mandatory and locked; optional gist1m was omitted because deep1M completed the required coverage and a second 1M-point build was not needed within the two-hour bench budget.
- DiskANN update benchmarking is deferred until wave3, after delete-repair lands.

The committed snapshots are `baseline-14ad5369-g03.json` and
`baseline-14ad5369-g03-laser.json`. Raw LASER logs and its fresh artifacts
remain under `/md1/huangliang/tmp/perf-baseline/14ad5369-laser/` on g03.
