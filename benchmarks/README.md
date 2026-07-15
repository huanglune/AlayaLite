# Benchmarks

```
benchmarks/
├── perf/                 # Hot-path performance baseline (SIMD/QG/LASER)
│   ├── run_baseline.py       # QG + SIMD baseline runner (SSH to target host)
│   ├── run_laser_baseline.py # LASER baseline runner
│   └── baselines/            # Machine-local baseline JSONs (gitignored)
├── laser/                # LASER-specific benchmarks and alignment harnesses
│   ├── disk_laser_smoke.py        # Legacy stdout compatibility wrapper
│   ├── laser_unified_fit_bench.py # Unified fit benchmark
│   ├── laser_unified_fit_plot.py  # Recall-QPS curve renderer
│   └── alignment/                 # Port-LASER tier A/B alignment verification
├── size_map/             # Binary size tracking
├── adapters/             # External benchmark framework adapters
│   └── annbenchmark/         # ANN-benchmarks integration
└── README.md
```

Golden artifact generators live in `tests/golden/` (test infrastructure,
not benchmarks).

## Python benchmark CLIs

- `python -m alayalite.bench.disk_collection` — unified DiskCollection
  sweep harness (engines: `disk_flat`, `disk_vamana`, `disk_laser`).
- `python -m alayalite.bench.laser_compare` — paired native vs
  DiskCollection LASER adapter overhead harness.

Large external datasets are off by default. CI exercises only the
`laser_compare` smoke test on a deterministic 256-row fixture.
