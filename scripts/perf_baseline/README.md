# Hot-path performance baseline

This directory locks the g03 baseline for SIMD distance/FHT kernels and the in-memory RaBitQ QG path. The runner performs all benchmark work through SSH; dataset inputs and generated indexes remain under `/md1` on g03.

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

## Gate contract

Compare identical names and parameters on the same host, NUMA binding, Release toolchain/flags, dataset, and thread setting. Use medians, not the best run:

- SIMD: compare each `name`'s `median` (`ns_per_call`, lower is better). Each value is the median of five process runs; each process performs 1,000 warmups and 100,000 timed calls per implementation/dimension. A candidate fails when `(candidate / baseline - 1) * 100 > 3`.
- QG search: compare `qps_median` at every matching ef (higher is better), while requiring recall not to decrease. A candidate fails when `(baseline / candidate - 1) * 100 > 3`. Each external process reports the mean of its three built-in query rounds; the locked value is the median of five process results.
- QG build: compare `estimated_build_duration_s` (lower is better). It is cold total wall time minus median warm/search-only wall time. Peak RSS must be reviewed separately and cannot be silently omitted.
- Index correctness: the SHA-256 is provenance, not a performance metric. A changed hash is expected only when index serialization or construction changes and must be explained.

p50/p95/p99 in SIMD are percentiles across the five process-level `ns_per_call` values. The current QG binary exposes only aggregate QPS/recall, not per-query latency, so QG p50/p95/p99 are unavailable rather than fabricated.

Treat a regression over 3% in any gated row as rollback-worthy unless a reviewed exception documents measurement noise or an intentional tradeoff. Re-run both baseline and candidate when g03 pre-run load average exceeds 10; the captured load is in `environment`.

## Coverage decisions

- `rabitq_performance_test` hard-codes deep1M L2. It has no gist1m mode, so gist1m is recorded as unsupported by this target.
- No LASER build/search perf C++ target exists in this checkout. `laser_cross_platform_perf.py` requires the `alayalite` Python binding and this worktree has no `.venv`; enabling `BUILD_PYTHON` solely for this baseline is intentionally out of scope.
- DiskANN update benchmarking is deferred until wave3, after delete-repair lands.

The committed snapshot is `baseline-14ad5369-g03.json`; raw logs and the 2.27 GiB QG artifact remain in `/md1/huangliang/tmp/perf-baseline/` on g03.
