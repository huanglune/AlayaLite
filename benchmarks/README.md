# Benchmarks

```
benchmarks/
├── CMakeLists.txt                 # Native benchmark targets
├── any_segment_sync_benchmark.cpp
├── parity_lanes_benchmark.cpp
├── result_contract_benchmark.cpp
├── simd/                          # SIMD microbenchmarks
├── rabitq/                        # RaBitQ dispatch benchmark only
├── laser/                         # Native LASER benchmarks and manual research tools
│   ├── alignment/                 # Offline artifact comparators and data generators
│   └── tools/                     # Manual alignment and dataset-preparation runners
├── perf/baselines/                # Historical, machine-specific captures
├── size_map/                      # Native module/wheel size tracking
└── README.md
```

Golden artifact generators live in `tests/golden/` (test infrastructure,
not benchmarks).

Repository benchmark support is intentionally separate from the installed SDK.
The retired Python CLIs and ANN adapter are not part of the current benchmark
surface; use the native targets and scripts shown above.
