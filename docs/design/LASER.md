# LASER implementation and offline tools

LASER is the physical on-disk implementation used by eligible sealed `qg` Collection generations. It is not a Python
index family: Python exposes `flat` and `qg`, while `alayalite.capabilities()` reports whether the installed wheel can
serve QG and which LASER SIMD backend was selected.

## Ownership boundary

- Public Python schema: `CollectionConfig(index=QGIndexConfig(...))`.
- Public C++ algorithm identity: `core::algorithm::qg`.
- Physical sealed implementation: `qg_laser_segment` and the LASER reader/builder stack.
- Repository-only preprocessing: `tools/laser/`.
- Native benchmarks and manual alignment tools: `benchmarks/laser/`.

The accepted ownership boundary is recorded in [the QG/LASER ADR](adr-qg-laser-boundary.md). Native LASER source paths
remain under `include/index/graph/laser/`; path identity is deliberate and must not be flattened into the Collection
layer.

## Python use

```python
import numpy as np

import alayalite
from alayalite.config import CollectionConfig, QGIndexConfig

if "qg" not in alayalite.capabilities().index_types:
    raise RuntimeError("This wheel does not provide QG")

with alayalite.connect("./qg-data") as database:
    collection = database.create_collection(
        "vectors",
        config=CollectionConfig(
            dimension=128,
            dtype="float32",
            metric="l2",
            index=QGIndexConfig(max_neighbors=32, construction_effort=400),
        ),
    )
    vectors = np.random.default_rng(42).normal(size=(128, 128)).astype(np.float32)
    collection.add(ids=[str(row) for row in range(len(vectors))], vectors=vectors)
    collection.seal()
    result = collection.search(vectors[0], limit=10, effort=200)
    print(result[0].ids.tolist())
    collection.close()
```

QG creation is float32-only and currently accepts dimensions from 33 through 2048. Supported graph degree values are
32 and 64; construction effort must be at least the degree. Unsupported platforms fail during collection creation,
before filesystem mutation, with a typed diagnostic that explicitly says Flat fallback is disabled.

## Platform and I/O backends

| Platform | Sealed QG | I/O backend | SIMD report |
| --- | ---: | --- | --- |
| Linux x86_64 | yes | libaio by default | generic/AVX2/AVX-512 |
| macOS x86_64/arm64 | yes | portable thread pool | architecture-selected |
| Linux aarch64 | no in published wheel | n/a | `None` |
| Windows x64 | no in published wheel | n/a | `None` |

The writable LASER active engine is a separate C++-only, Linux-only capability. It does not create another Python
index kind.

## Repository-only Python tools

`tools/laser/` contains preprocessing utilities used to create or inspect native fixtures:

- `_io.py`: fbin/ibin readers and writers;
- `_pca.py`: deterministic incremental PCA fitting, serialization, and transformed-data generation;
- `_medoid.py`: deterministic medoid generation through FAISS;
- `tests/`: focused deterministic PCA tests.

Install their dependencies through the repository group, not an SDK runtime extra:

```bash
uv sync --group laser-tools
uv run pytest tools/laser/tests -q
```

Import them only from repository tooling:

```python
from tools.laser._io import read_fbin
from tools.laser._pca import fit_incremental_pca
```

These modules are intentionally excluded from both wheel and sdist. Applications should not use them at runtime.

## Deterministic fixture path

`tests/disk/fixtures/build_laser_fixture.py` loads the requested build-tree extension directly and imports only the
repository tool package. It no longer stages or copies the installed SDK package. The fixture pipeline pins numerical
worker pools, uses deterministic seeds, writes into a temporary sibling, validates every required artifact, and then
publishes atomically.

The generated family includes input fbin data, PCA parameters/base vectors, medoid files, a Vamana graph, the LASER
index, rotation/cache sidecars, and provenance hashes. CTest builds it through the `laser_segment_fixture` target.

## Build and verification

```bash
uv sync --group laser-tools
cmake --preset release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release --target laser_segment_fixture -j2
ctest --test-dir build/Release --output-on-failure
```

Alignment scripts live under `benchmarks/laser/alignment/`; manual dataset preparation and the migrated alignment test
live under `benchmarks/laser/tools/`. These paths validate artifacts and native behavior, not a wheel API.

## Search semantics

The Collection layer owns logical IDs, document/metadata state, the logical WAL, routing snapshots, filter admission,
and maintenance publication. LASER owns physical candidate search and persisted graph artifacts. Search responses are
translated into the Collection CSR contract with stable IDs, numeric distances, per-query status/completeness, and no
sentinel padding.

QG effort defaults to `max(100, limit)`. An explicit value below that floor is rejected rather than silently raised.
Filtering remains a Collection-owned contract; the physical search implementation does not define a competing Python
filter language.
