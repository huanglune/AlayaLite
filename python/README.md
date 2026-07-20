# AlayaLite Python SDK

AlayaLite is an embedded database. A filesystem directory owns a `Database`; the database catalog owns named
collections; each `Collection` owns its native storage, WAL, search snapshots, and maintenance lifecycle.

## Install

```bash
pip install alayalite
# or
uv add alayalite
```

The base runtime dependency is NumPy. RAG helpers and offline LASER artifact tools live in this repository under
`examples/rag/support/` and `tools/laser/`; they are not installed by the SDK wheel.

## Complete example

```python
from pathlib import Path

import numpy as np

import alayalite
from alayalite.config import CollectionConfig, FlatIndexConfig

database_path = Path("./demo-db")

with alayalite.connect(database_path) as database:
    collection = database.create_collection(
        "articles",
        config=CollectionConfig(
            dimension=3,
            dtype="float32",
            metric="l2",
            index=FlatIndexConfig(),
        ),
    )
    receipt = collection.add(
        ids=["one", "two"],
        vectors=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        documents=["first", "second"],
        metadata=[{"topic": "database"}, {"topic": "search"}],
    )
    assert [row.status.value for row in receipt.rows] == ["inserted", "inserted"]

    result = collection.search(
        np.asarray([[0, 0, 0]], dtype=np.float32),
        limit=2,
        where={"topic": {"$in": ["database", "search"]}},
    )
    assert result.offsets.tolist() == [0, 2]
    records = collection.get(result[0].ids.tolist(), include_vector=False)
    print([record.document for record in records if record is not None])
    collection.checkpoint()
    collection.close()
```

## Public surface

The root package exports 23 names: `connect`, `capabilities`, `Database`, `Collection`, three configuration models,
four common result models, and twelve exception classes. Less common receipts, statistics, enums, and type aliases
live in `alayalite.models`; configuration aliases live in `alayalite.config`.

### Database lifecycle

```python
with alayalite.connect("./data") as database:
    names = database.list_collections()
    collection = database.open_collection(names[0])
    collection.close()
```

`connect()` is lazy: it does not eagerly open every catalog entry. Creation and opening are separate operations.
`drop_collection()` permanently removes a collection and rejects an active handle. Database and Collection `close()`
methods are idempotent.

### Configuration

```python
from alayalite.config import CollectionConfig, FlatIndexConfig, QGIndexConfig

portable = CollectionConfig(dimension=384, metric="cosine", index=FlatIndexConfig())
graph = CollectionConfig(
    dimension=384,
    metric="cosine",
    index=QGIndexConfig(max_neighbors=32, construction_effort=400),
)
```

Configuration models are frozen. QG is the default and is float32-only with a current dimension envelope of 33–2048.
On a wheel without QG support, creation fails before writing. Check `alayalite.capabilities().index_types` when an
application needs to choose a portable configuration.

### Writes

`add`, `replace`, and `upsert` share one keyword-only columnar signature. IDs are strict strings; every supplied column
must have the same row count; empty batches are rejected. The default mode is `atomic` and the default durability is
`fsync`.

```python
result = collection.upsert(
    ids=["one"],
    vectors=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float32),
    documents=["updated"],
    metadata=[{"revision": 2}],
    mode="atomic",
    durability="fsync",
    idempotency_key="request-42",
)
```

Row conflicts are reported in `MutationResult.rows`; call-level I/O, lifecycle, platform, and corruption failures raise
typed exceptions.

### Search, get, and scan

`search()` accepts one vector or a query matrix and always returns a read-only CSR `SearchResult`. `result[i]` is a
shared-memory read-only row view. Search returns IDs and distances only; fetch payloads explicitly with `get()`.

```python
result = collection.search(queries, limit=10, where={"revision": {"$ge": 2}})
first_row = result[0]
payload = collection.get(first_row.ids.tolist())
```

`get()` preserves input order and length, returning `None` for a missing ID. `scan()` performs a native filtered,
limited projection; vectors are included only when requested.

Supported filter operators are `$eq`, `$gt`, `$ge`, `$lt`, `$le`, `$in`, `$and`, and `$or` over flat metadata.

### QG effort

QG uses `effort`; omitting it chooses `max(100, limit)`. An explicit value below that floor is rejected. Flat search
rejects an explicit effort because the setting would have no effect.

### Maintenance and read-only access

Collections expose typed `checkpoint`, `seal`, `compact`, `collect_garbage`, `rebuild_index`, and `stats` operations.
Successful writes are already searchable on return, so there is no separate flush state.

```python
with alayalite.connect("./data", read_only=True) as database:
    with database.open_collection("articles") as collection:
        print(collection.count())
```

A read-only open does not create, repair, truncate, or rewrite files. Mutation and maintenance calls fail with
`CollectionNotSupportedError`.

## Development checks

From the repository root:

```bash
uv sync
cmake --preset release -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build --preset release -j2
ctest --test-dir build/Release --output-on-failure
uv run --locked pytest python/tests -q
MYPYPATH=python/src uv run --with mypy mypy --strict python/src/alayalite
uvx pre-commit run -a
```

See the [embedded SDK guide](../docs/user/CLIENT_USER_MANUAL.md) for detailed contracts and the
[build guide](../docs/user/BUILDING.md) for native prerequisites.
