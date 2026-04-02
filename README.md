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

- Header-only C++20 vector indexing components under `include/`.
- Python SDK with `Client`, `Collection`, and `Index` abstractions.
- Multiple index and quantization options, including `hnsw`, `nsg`, `fusion`, `sq8`, `sq4`, and `rabitq`.
- Optional standalone FastAPI service in `app/`.
- Example Streamlit RAG app in `examples/rag/`.

## Repository Guide

| Path | What it contains |
| --- | --- |
| [`python/README.md`](python/README.md) | Python SDK usage and API reference |
| [`app/README.md`](app/README.md) | FastAPI standalone service |
| [`examples/rag/README.md`](examples/rag/README.md) | End-to-end RAG demo |
| [`include/simd/README.md`](include/simd/README.md) | SIMD kernels and dispatch overview |
| [`scripts/README.md`](scripts/README.md) | Helper scripts for builds, data prep, and benchmarks |

## Install

### Python package

```bash
pip install alayalite
```

### Local development

```bash
uv sync
make build
```

`make dev-install` is also available if you want the standard development setup in one step.

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
index = client.create_index("default", index_type="hnsw", metric="l2")
index.fit(vectors, ef_construction=100, num_threads=1)

neighbors = index.batch_search(queries, topk=10, ef_search=100, num_threads=1)
print(neighbors.shape)
```

## Standalone Service

The FastAPI app lives in `app/` and exposes collection-oriented endpoints under `/api/v1`.

```bash
uv sync --group api
uv run uvicorn app.main:app --reload
```

See [`app/README.md`](app/README.md) for endpoint examples and Docker usage.

## Build and Test

```bash
make build          # release build
make build-debug    # debug build
make test           # C++ + Python tests
make test-cpp       # C++ tests only
make test-py        # Python + API tests
make lint           # pre-commit checks
```

Additional project scripts are documented in [`scripts/README.md`](scripts/README.md).

## Benchmarks

Benchmark sources and runners live in `benchmark/` and `scripts/benchmark/`.
The SIMD helpers used by distance kernels are documented in [`include/simd/README.md`](include/simd/README.md).

## Contributing

Contributions are welcome. Please open an issue or pull request and include tests for behavior changes.

## Contact

Questions or ideas: `dev@alayadb.ai`

## License

[Apache 2.0](LICENSE)
