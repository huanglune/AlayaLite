# CLAUDE.md

## Build & Test

```bash
make build           # Release build (tests + benchmarks ON)
make build-debug     # Debug build
make test            # All tests
make test-cpp        # C++ only
make test-py         # Python tests via pytest
make lint            # pre-commit checks (clang-format, ruff, typos)
make dev-install     # Python editable install with dev deps (run first for Python work)
EXTRA_CMAKE_FLAGS="-DBUILD_PYTHON=ON" make build  # Extra flags
```

Single test: `./build/tests/index/diskann_builder_test`
GTest filter: `./build/tests/storage/buffer_pool_test --gtest_filter="BufferPoolTest.BasicReadWrite"`
Pytest: `uv run pytest python/tests/ -v -k "test_search"`

- Direct CMake defaults differ from Makefile: `BUILD_TESTING=OFF`, `BUILD_BENCHMARKS=OFF`, `BUILD_PYTHON=OFF` unless explicitly enabled.
- Commit messages: Conventional Commits `<type>: <description>`, type = `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`.

## Project Constraints

- Header-only C++20 library — do not create `.cpp` source files under `include/`.
- C++20 coroutine-based concurrency (`include/executor/`). Use coroutine primitives, not raw threads.
- `MetricType` and `DiskDataType` enums live in `include/utils/types.hpp` — reuse them, do not duplicate.
- Python bindings via pybind11, glue code in `python/adapters/`.

## Architecture (quick reference)

```
include/
├── index/graph/     # ANN indices: HNSW, NSG, DiskANN, KNNG, QG, Fusion (templated on Space)
├── space/           # Distance computation: RawSpace, PQSpace, SQ4/SQ8Space, RaBitQSpace
├── storage/         # BufferPool, DiskANN files (Meta/Data/PQ), DirectFileIO, io_uring
├── executor/        # Coroutine scheduler, workers, search/update jobs
├── simd/            # SIMD distance kernels (AVX2, AVX-512, NEON)
└── utils/           # Bitsets, candidate lists, types, logging
```

DiskANN storage uses a three-file architecture:
- `MetaFile`: metadata + validity bitmap
- `DataFile`: graph adjacency + raw vectors (Direct I/O)
- `PQFile`: PQ data (mmap)

## Testing Expectations

- Add/update tests with every behavior change. C++ tests in `tests/` as `*_test.cpp`.
- Prefer deterministic fixtures and reopen-on-disk validation for persistence changes.
- For search changes, validate uniqueness, ID range, ordering, non-negative L2 distances, and recall.
- Preserve fail-fast behavior (e.g. `reserve_capacity()` on read-only, unimplemented `insert()`).

## DiskANN & Storage Constraints

- Two search modes: with PQ rerank (in-memory PQ + disk rerank) and without (disk-heavy), but now we only need to consider pure disk.
- `DiskANNSearcher` uses `BufferPool` with `ClockReplacer` by default (not LRU).
- `reserve_capacity()` is writable-only, unsupported for PQ indexes.
- `insert()` fails fast with `std::logic_error`; do not add silent fallback.
- `BufferPool`: page lifetime via RAII `PageHandle` pin/unpin; dirty pages need `mark_dirty()`.

## FastAPI Service

- All endpoints are `POST` routes including listing. `ALAYALITE_DATA_DIR` controls storage dir (default `./data`).

## Naming & Lint

- All C++ code must conform to `.clang-format` and `.clang-tidy` configs at the repo root. Run `make lint` before committing and ensure it passes.
- Chinese characters in code/config will fail lint (`scripts/check_chinese.py`).
- C++: `CamelCase` classes, `lower_case` functions/variables, trailing `_` members, `kConstantName` constants.
- Do not hand-edit generated output in `build/`, `Testing/`, `.venv/`, `dist/`, or `*.egg-info`.
