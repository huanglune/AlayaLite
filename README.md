<p align="center">
  <a href="https://github.com/AlayaDB-AI"><img src="https://github.com/AlayaDB-AI/AlayaLite/blob/main/docs/images/banner.jpg?raw=true" width="300" alt="AlayaDB logo"></a>
</p>

<p align="center">
  <b>AlayaLite — an embedded vector database with a typed Python SDK.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/alayalite/"><img src="https://img.shields.io/pypi/v/alayalite" alt="PyPI"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License"></a>
  <a href="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml"><img src="https://github.com/AlayaDB-AI/AlayaLite/actions/workflows/code-checker.yaml/badge.svg?branch=main" alt="CI"></a>
</p>

## What it provides

- A local `connect → Database → Collection` lifecycle with context managers.
- Exact `flat` search on every supported wheel and platform-gated `qg` search backed by LASER.
- Columnar add, replace, upsert, delete, metadata filtering, aligned reads, and CSR search results.
- Atomic-by-default writes, explicit durability, typed receipts, typed exceptions, and true read-only opens.
- A lean runtime wheel whose only Python dependency is NumPy.

## Install

```bash
pip install alayalite
# or
uv add alayalite
```

The SDK v2 design ships as package version `1.1.0`; “v2” names the API generation, not the distribution version.

## Quick start

```python
from pathlib import Path

import numpy as np

import alayalite
from alayalite.config import CollectionConfig, FlatIndexConfig

root = Path("./alaya-data")
config = CollectionConfig(
    dimension=3,
    dtype="float32",
    metric="cosine",
    index=FlatIndexConfig(),
)

with alayalite.connect(root) as database:
    collection = database.create_collection("docs", config=config)
    collection.add(
        ids=["a", "b"],
        vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
        documents=["first document", "second document"],
        metadata=[{"lang": "en"}, {"lang": "zh"}],
    )

    result = collection.search(
        np.asarray([1, 0, 0], dtype=np.float32),
        limit=2,
        where={"lang": {"$in": ["en", "zh"]}},
    )
    records = collection.get(result[0].ids.tolist())
    print([(record.id, record.document) for record in records if record is not None])
    collection.checkpoint()
    collection.close()

with alayalite.connect(root, read_only=True) as database:
    with database.open_collection("docs") as collection:
        assert collection.count() == 2
```

The example selects Flat explicitly so it works on every wheel. `CollectionConfig` defaults to QG; call
`alayalite.capabilities()` before relying on that default in cross-platform applications.

## Platform matrix

| Wheel/build | Flat | QG sealed target | Writable native LASER active engine |
| --- | ---: | ---: | ---: |
| Linux x86_64 | yes | yes | C++ only |
| macOS x86_64/arm64 | yes | yes | no |
| Linux aarch64 | yes | no | no |
| Windows x64 | yes | no | no |

Unsupported QG creation fails before a collection directory is created and includes `Flat fallback is disabled` in
the typed diagnostic. AlayaLite never silently changes the requested index family.

## Documentation

- [Python package guide](python/README.md)
- [Embedded Python SDK guide](docs/user/CLIENT_USER_MANUAL.md)
- [Build guide](docs/user/BUILDING.md)
- [Documentation map](docs/README.md)
- [LASER implementation and offline-tool guide](docs/design/LASER.md)
- [FastAPI example](app/README.md)
- [RAG example](examples/rag/README.md)

## Performance context

The following figures are historical measurements retained for research context. Current executable benchmark targets
live under `benchmarks/`; they are not SDK APIs.

| Fashion-MNIST | GIST |
| :---: | :---: |
| ![Fashion-MNIST](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/fashion-mnist-784-euclidean.png) | ![GIST](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/gist-960-euclidean.png) |

![LASER versus disk ANN systems](https://raw.githubusercontent.com/AlayaDB-AI/AlayaLite/main/docs/images/laser-vs-disk-anns.png)

See the [AlayaLaser paper](https://arxiv.org/abs/2602.23342) for algorithm and experiment details.

## Contributing and license

Open an issue before substantial work, include tests with changes, and run the repository checks described in the build
guide. AlayaLite is licensed under [AGPL-3.0](LICENSE).
