# Hot-path performance baseline

Baseline JSON files in this directory are **machine-local artifacts** and
gitignored. They capture SIMD/QG/LASER performance numbers on a specific
host (CPU, NUMA, toolchain) and are meaningless on a different machine.

## Generate a baseline

```bash
benchmarks/perf/run_baseline.py \
  --host <hostname> \
  --data-root /path/to/data \
  --remote-tmp /tmp/perf-baseline \
  --output benchmarks/perf/baselines/baseline-<commit>-<host>.json
```

LASER baseline:

```bash
.venv/bin/python benchmarks/perf/run_laser_baseline.py \
  --host <hostname> \
  --data-root /path/to/data \
  --tmp-root /tmp/perf-baseline \
  --output benchmarks/perf/baselines/baseline-<commit>-<host>-laser.json
```

## Compare against a candidate

After a code change, re-run on the same host and compare:

```bash
# diff the JSON fields manually or use jq
diff <(jq '.simd' baselines/baseline-old.json) <(jq '.simd' baselines/baseline-new.json)
```

## Gate contract

A regression over 3% in any gated row is rollback-worthy unless a reviewed
exception documents measurement noise or an intentional tradeoff.

- **SIMD**: compare `median` ns_per_call (lower is better)
- **QG search**: compare `qps_median` at every ef (higher is better), recall must not decrease
- **QG build**: compare `estimated_build_duration_s` (lower is better)
- **LASER search**: compare `qps_median` per EF, recall must not decrease
- **LASER build**: compare `build_duration_s` (lower is better)

Use medians, not best run. Pin to same NUMA node. Re-run both sides if
pre-run load average exceeds 10.
