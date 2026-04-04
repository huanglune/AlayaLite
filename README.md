<p align="center">
  <a href="https://github.com/AlayaDB-AI"><img src="https://github.com/AlayaDB-AI/AlayaLite/blob/main/.assets/banner.jpg?raw=true" width="300" alt="AlayaLite banner"></a>
</p>

<p align="center">
  <b>AlayaLite</b> is a lightweight vector database toolkit with a header-only C++20 core,
  Python bindings, an optional FastAPI service, and example RAG applications.
</p>

<div class="column" align="middle">
  <a href="https://github.com/AlayaDB-AI/AlayaLite/releases"><img height="20" src="https://img.shields.io/badge/alayalite-blue" alt="release"></a>
  <a href="https://pypi.org/project/alayalite/"><img src="https://img.shields.io/pypi/v/alayalite" alt="PyPI"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE"><img height="20" src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="LICENSE"></a>
  <a href="https://codecov.io/github/AlayaDB-AI/AlayaLite"><img height="20" src="https://codecov.io/github/AlayaDB-AI/AlayaLite/graph/badge.svg?token=KA6V0DHHUU" alt="codecov"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml"><img height="20" src="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml/badge.svg?branch=main" alt="CI"></a>
</div>

## Highlights

- Header-only C++20 ANN components under `include/`.
- In-memory graph indices for `hnsw`, `nsg`, and `fusion`.
- DiskANN-oriented C++ modules for build, search, storage, and async I/O.
- Python SDK with `Client`, `Collection`, and `Index` abstractions.
- Quantization options including `sq8`, `sq4`, and `rabitq`.
- Optional FastAPI service in `app/`.
- Streamlit RAG demo in `examples/rag/`.

## Repository Guide

| Path | What it contains |
| --- | --- |
| [`python/README.md`](python/README.md) | Python SDK usage, persistence, and supported parameters |
| [`app/README.md`](app/README.md) | FastAPI standalone service and HTTP examples |
| [`examples/rag/README.md`](examples/rag/README.md) | End-to-end Streamlit RAG demo |

## Install

### Repository install

```bash
make install
```

This installs the package from the repository without development dependencies.

### Local development

```bash
make dev-install
make build
```

Notes:

- `make dev-install` installs the editable Python package plus development dependencies.
- `make build` builds the C++ tests and benchmarks in release mode.
- To build the pybind target through CMake as well, run `EXTRA_CMAKE_FLAGS="-DBUILD_PYTHON=ON" make build`.

## Quick Start

### Collection workflow

```python
from alayalite import Client
import numpy as np

client = Client(url="./data")
collection = client.create_collection("docs", metric="cosine")

items = [
    (1, "first document", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"source": "guide"}),
    (2, "second document", np.array([0.2, 0.1, 0.4], dtype=np.float32), {"source": "faq"}),
]

collection.insert(items)

results = collection.batch_query(
    [[0.1, 0.2, 0.3]],
    limit=2,
    ef_search=10,
    num_threads=1,
)

print(results["document"][0])
client.save_collection("docs")
```

### Lower-level index workflow

```python
from alayalite import Client
import numpy as np

vectors = np.random.rand(1000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)

client = Client()
index = client.create_index(
    "default",
    index_type="hnsw",
    metric="l2",
    quantization_type="none",
)
index.fit(vectors, ef_construction=100, num_threads=1)

neighbors = index.batch_search(queries, topk=10, ef_search=100, num_threads=1)
print(neighbors.shape)
```

## Build and Test

```bash
make build          # release build with tests + benchmarks
make build-debug    # debug build
make test           # C++ + Python tests
make test-cpp       # C++ tests only
make test-py        # Python + API tests
make lint           # pre-commit checks
make lint-tidy      # clang-tidy rebuild
```

Direct CMake defaults are intentionally leaner than the Makefile:

- `BUILD_TESTING=OFF`
- `BUILD_BENCHMARKS=OFF`
- `BUILD_PYTHON=OFF`

## Benchmarks

Benchmark sources live in `benchmark/`, with dataset/config samples under `benchmark/index/configs/`.

Current CMake benchmark targets:

- `graph_search_bench`: graph build and search benchmark driven by a TOML config
- `diskann_update_bench`: DiskANN delete/insert/search benchmark driven by a TOML config

Example:

```bash
./build/benchmark/index/graph_search_bench benchmark/index/configs/bench_hnsw_gist.toml
```

The helper scripts and coverage utilities are documented in [`scripts/README.md`](scripts/README.md).

## Standalone Service

The FastAPI app lives in `app/` and exposes collection-oriented endpoints under `/api/v1`.

```bash
uv sync --group api --group test
uv run uvicorn app.main:app --reload
```

See [`app/README.md`](app/README.md) for endpoint examples and Docker usage.

## Contributing

Contributions are welcome. Please include tests for behavior changes and keep documentation in sync with the relevant module.

## Contact

Questions or ideas: `dev@alayadb.ai`

## License

[Apache 2.0](LICENSE)
