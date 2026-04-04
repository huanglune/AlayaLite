# Scripts

This directory contains helper scripts for dependency setup, data conversion, lint support, coverage reporting, and benchmark automation.

## Conventions

- Run Python scripts from the repository root unless noted otherwise.
- Prefer `uv run ...` or `uvx ...` so tool versions come from the project environment.
- Some scripts depend on external tools such as `conan`, `wget` or `curl`, `lcov`, or LLVM coverage utilities.

## Layout

| Path | Purpose |
| --- | --- |
| `scripts/conan_build/conan_install.py` | Installs Conan dependencies with the platform-specific profile |
| `scripts/benchmark/run_diskann.py` | Legacy DiskANN benchmark wrapper that parses benchmark output into structured artifacts |
| `scripts/hdf5_to_fvecs.py` | Converts ANN-Benchmarks `.hdf5` datasets into `fvecs` and `ivecs` files |
| `scripts/check_chinese.py` | Fails if source files contain Chinese characters |
| `scripts/license_check/license_script.py` | Checks or adds Apache license headers |
| `scripts/test/gnu_codecoverage.sh` | Generates GNU `gcov` and `lcov` HTML coverage output |
| `scripts/test/llvm_codecoverage.sh` | Generates LLVM coverage output |
| `scripts/ci/codecov/gnu_codecoverage.sh` | CI-oriented GNU coverage upload helper |

## Common Commands

### Install Conan dependencies

```bash
uv run scripts/conan_build/conan_install.py --project-dir .
```

Optional flags:

- `--build-type Debug`
- `--build-type Release`

### Convert `.hdf5` benchmark datasets

```bash
uv run scripts/hdf5_to_fvecs.py \
  --input http://ann-benchmarks.com/deep-image-96-angular.hdf5 \
  --output data/deep10M \
  --name deep10M
```

The script writes:

- `<name>_base.fvecs`
- `<name>_query.fvecs`
- `<name>_groundtruth.ivecs`

### Check for Chinese characters

```bash
python3 scripts/check_chinese.py path/to/file1 path/to/file2
```

This helper is used by the lint workflow.

### Check or add license headers

Check only:

```bash
python3 scripts/license_check/license_script.py --check-only
```

Modify files in place:

```bash
python3 scripts/license_check/license_script.py
```

### Generate coverage reports

GNU toolchain:

```bash
bash scripts/test/gnu_codecoverage.sh
```

LLVM toolchain:

```bash
bash scripts/test/llvm_codecoverage.sh
```

These scripts assume coverage artifacts already exist in `build/`.

## Benchmarks

The benchmark sources now live under `benchmark/`, and the CMake targets produced today are:

- `build/benchmark/index/graph_search_bench`
- `build/benchmark/index/diskann_update_bench`

Sample configs live in `benchmark/index/configs/`.

### Run the graph benchmark directly

```bash
./build/benchmark/index/graph_search_bench benchmark/index/configs/bench_hnsw_gist.toml
```

### Run the DiskANN update benchmark directly

```bash
./build/benchmark/index/diskann_update_bench benchmark/index/configs/bench_deep1m.toml
```

### About `run_diskann.py`

`scripts/benchmark/run_diskann.py` still targets an older benchmark binary name, `diskann_searcher_benchmark`, and writes summarized results under `benchmark/results/diskann_searcher/runs/<run_id>/`.

Use it only if your local build still provides that legacy benchmark binary, or adapt the script before relying on it in a new workflow.
