# Scripts

This directory contains helper scripts for dependency setup, data conversion, benchmarking, lint support, and coverage reporting.

## Conventions

- Run Python scripts from the repository root unless noted otherwise.
- Prefer `uv run ...` or `uvx ...` so tool versions come from the project environment.
- Some scripts depend on external tools such as `conan`, `wget`/`curl`, `lcov`, or LLVM coverage utilities.

## Layout

| Path | Purpose |
| --- | --- |
| `scripts/conan_build/conan_install.py` | Installs Conan dependencies with the platform-specific profile |
| `scripts/benchmark/run_diskann.py` | Runs the DiskANN searcher benchmark and writes structured results |
| `scripts/hdf5_to_fvecs.py` | Converts ANN-Benchmarks `.hdf5` datasets into `fvecs`/`ivecs` files |
| `scripts/check_chinese.py` | Fails if source files contain Chinese characters |
| `scripts/license_check/license_script.py` | Checks or adds Apache license headers |
| `scripts/test/gnu_codecoverage.sh` | Generates GNU `gcov`/`lcov` HTML coverage output |
| `scripts/test/llvm_codecoverage.sh` | Generates LLVM coverage output |

## Common Commands

### Install Conan dependencies

```bash
uv run scripts/conan_build/conan_install.py --project-dir .
```

Optional flags:

- `--build-type Debug`
- `--build-type Release`

### Run the DiskANN searcher benchmark

The benchmark binary is expected at `build/benchmark/index/diskann_searcher_benchmark`.
Build the project first, then run:

```bash
uv run scripts/benchmark/run_diskann.py
```

Useful options:

```bash
uv run scripts/benchmark/run_diskann.py \
  --skip-build \
  --ef-values 16,32,64 \
  --cache-values 20,50,100 \
  --thread-values 1,4,16 \
  --json
```

Results are written under `benchmark/results/diskann_searcher/runs/<run_id>/`.

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
