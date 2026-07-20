# Benchmark taxonomy

`benchmarks/` contains manual measurement and research assets. Native targets
are build-only: none is registered with CTest, and a successful `ctest` run
does not execute them. Golden artifact generators belong in `tests/golden/`,
not here.

## Categories

| Category | Contract |
| --- | --- |
| **microbench** | A small, repeatable timing loop for a current implementation hot path. Results are local to the host, toolchain, build flags, and arguments; they are not a checked-in cross-machine baseline. |
| **evidence-harness** | An A/B or parity program kept to reproduce the evidence behind a specific delivery or research report. Its experiment contract matters more than general benchmark coverage. |
| **research-tool** | A dataset or artifact workflow used by an active research line. It may prepare data, compare artifacts, run oracles, or measure a research arm; it is not a supported SDK surface. |
| **tracking** | A generated measurement used to watch a repository property such as binary size. Regenerate it with the named tool and compare only under the same measurement environment. |

## Authoritative map

| Asset | Category | Registration and disposition | Evidence owner or reference |
| --- | --- | --- | --- |
| `simd/` (seven native targets) | microbench | Kept in place; `simd/CMakeLists.txt`, build-only | Current SIMD kernel timing; no report owns a portable baseline |
| `any_segment_sync_benchmark.cpp` | microbench | Kept in place; top-level `CMakeLists.txt`, build-only | AnySegment synchronous-adapter overhead; no downstream path reference |
| `parity_lanes_benchmark.cpp` | evidence-harness | Kept in place; top-level `CMakeLists.txt` when `ALAYA_ENABLE_LASER=ON`, build-only | `docs/reports/REPORT-parity-lanes*.md` and `docs/design/adr-qg-laser-boundary.md` |
| `result_contract_benchmark.cpp` | evidence-harness | Kept in place; top-level `CMakeLists.txt` when `ALAYA_ENABLE_LASER=ON`, build-only | U-line rank-only versus numeric-distance result-contract evidence surface |
| `rabitq/rabitq_dispatch_benchmark.cpp` | evidence-harness | Kept in place; `rabitq/CMakeLists.txt`, build-only | Dispatch-refactor A/B numbers in `docs/reports/REPORT-u4-preflight.md` |
| `laser/` including `alignment/` and `tools/` | research-tool | Kept in place with target and path names unchanged; native target is gated as described below | Active LASER research, `docs/research/`, `docs/design/LASER.md`, and `scripts/fullcache-probe/` |
| `size_map/` | tracking | Kept in place; Python generator, not a CMake or CTest target and not called directly by CI | `docs/design/collection-semantics.md` and the HNSW-retirement record |
| former `perf/baselines/` | tracking | Retired from the working tree in `e2adf51`; recover the machine-specific JSON and its warning README from Git history | No runner and no reference from `docs/`, `scripts/`, or `.github/`; the captures were not valid cross-machine comparisons |

The two RaBitQ families are deliberately not a consolidation queue. The
memory-QG builder scratch format and LASER v1 artifact format coexist under
different contracts, and their rotators are not equivalent; see
`docs/design/rabitq-formats.md`.

## Build once

From the repository root:

```bash
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DALAYA_NATIVE_ARCH=OFF
cmake --build build/Release -j32
```

The commands below assume that build. Build an individual executable with
`cmake --build build/Release --target <target> -j32` before running it.

### `simd/`: kernel microbenchmarks

The directory contains L2 and inner-product loops for full, SQ8, and SQ4 data,
plus the FHT loop. `benchmarks/CMakeLists.txt` always adds `simd/`; the seven
targets are build-only and have no repository consumer beyond that
registration. Run, for example:

```bash
./build/Release/benchmarks/simd/l2_sqr_full_benchmark 128 256 768
./build/Release/benchmarks/simd/fht_benchmark 7 8 9
```

The optional numbers are dimensions for distance/IP targets and log2 sizes
for `fht_benchmark`; omitting them uses each program's defaults.

### Root microbench: AnySegment synchronous adaptation

`any_segment_sync_benchmark` isolates fixed synchronous-adapter overhead with
deliberately small engine work. Only the top-level benchmark CMake file refers
to it. It is always built and takes optional iteration, repeat, and warm-up
counts:

```bash
./build/Release/benchmarks/any_segment_sync_benchmark 20000 5 2000
```

### Root evidence harnesses: U-line LASER parity and result contract

`parity_lanes_benchmark` is the primary LASER A/B harness behind the U-line
parity reports; `result_contract_benchmark` is its companion rank-only versus
numeric-distance result-contract harness. These are evidence sources, not
general performance tests. Their source semantics and root paths are kept
stable for the paper trail. Both are created only when
`ALAYA_ENABLE_LASER=ON` and neither enters CTest.

The parity report contains the canonical experiment commands; inspect the
current option surface with:

```bash
./build/Release/benchmarks/parity_lanes_benchmark --help
./build/Release/benchmarks/result_contract_benchmark \
  --root /tmp/result-contract --dim 128 --rebuild
```

The report chain is `docs/reports/REPORT-parity-lanes.md`,
`REPORT-parity-lanes-phase2.md`, and its addendum. Scripts in
`scripts/fullcache-probe/` also post-process parity output.

### `rabitq/`: dispatch-refactor evidence

`rabitq_dispatch_benchmark` compares forced-generic and runtime-dispatched
implementations for the six memory-RaBitQ kernels. It was created for the U4
refactor delivery and its numbers are recorded in
`docs/reports/REPORT-u4-preflight.md`. The harness remains useful because the
current `include/space/quant/rabitq/{fastscan,lut,rotator}.hpp` paths still call
`dispatch.hpp`; this is an active architecture surface even though the
program itself is a one-shot evidence harness. `rabitq/CMakeLists.txt` always
registers it as build-only. Run it without arguments:

```bash
./build/Release/benchmarks/rabitq/rabitq_dispatch_benchmark
```

### `laser/`: active research tools

Everything under `laser/` is part of the active LASER research evidence
surface and stays at its current path. `bench_laser_update_sift`, its oracle
sources, `alignment/`, and `tools/` are referenced by `docs/research/`,
`docs/design/LASER.md`, and the fleet runbooks in `scripts/fullcache-probe/`.
The runbooks hard-code both the directory and `bench_laser_update_sift` target
name.

The native target is build-only and `laser/CMakeLists.txt` returns early unless
both `ALAYA_ENABLE_LASER=ON` and `ALAYA_ENABLE_MUTABLE_LASER_TESTS=ON`. A
no-LASER configuration therefore intentionally creates no LASER research
target. Use the experiment modes and dataset-specific options documented in
the source and research reports, for example:

```bash
./build/Release/benchmarks/laser/bench_laser_update_sift eval \
  --prefix /data/laser/sift900k --n 1000000 \
  --query sift_query.fbin --gt sift_gt.ibin --efs 60,80,100,150,200
python benchmarks/laser/alignment/tier_a_byte_equal.py --help
python benchmarks/laser/alignment/tier_b_statistical.py --help
python benchmarks/laser/tools/test_laser_alignment.py --help
python benchmarks/laser/tools/drift_prep.py --help
```

The `insert` mode mutates index artifacts; clone the artifact directory before
comparing research arms.

### `size_map/`: native binding size tracking

`size_map/` contains the checked-in same-toolchain reference JSON and the
generator that measures the Release pybind object/module, optionally a wheel,
and retired-symbol absence. It has no CMake registration and no direct CI
caller; `benchmarks/**` merely participates in broad build/test path filters.
After building the Release Python module, regenerate with:

```bash
python benchmarks/size_map/generate_size_map.py
python benchmarks/size_map/generate_size_map.py --wheel dist/alayalite-....whl
```

Treat changes as tracking evidence only when compiler, linker, flags, and
packaging inputs match. See `size_map/README.md` for the tracked fields.

## Retired historical captures

`perf/baselines/` held machine-specific JSON from a benchmark surface whose
unified runner, QG test target, and public LASER Python builder had already
been removed. No `docs/`, `scripts/`, or `.github/` path referenced the
directory. Keeping those files in the working tree invited invalid comparisons,
so commit `e2adf51` retires them into Git history. This retirement does not
define a replacement unified baseline runner; current measurements use the
manual assets above under their category-specific contracts.
