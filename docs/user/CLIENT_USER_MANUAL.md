# Embedded Python SDK guide

This guide documents the supported AlayaLite Python API shipped as version `1.1.0`. AlayaLite is a local embedded
database: `connect(path)` opens a directory catalog, and that catalog creates or opens named collections.

## Installation

```bash
pip install alayalite
# or
uv add alayalite
```

NumPy is the only runtime dependency of the base wheel.

## Executable quickstart

```python
from pathlib import Path

import numpy as np

import alayalite
from alayalite.config import CollectionConfig, FlatIndexConfig

path = Path("./quickstart-db")

with alayalite.connect(path) as database:
    collection = database.create_collection(
        "docs",
        config=CollectionConfig(
            dimension=3,
            dtype="float32",
            metric="cosine",
            index=FlatIndexConfig(),
        ),
    )
    added = collection.add(
        ids=["a", "b", "c"],
        vectors=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.2, 0.0]],
            dtype=np.float32,
        ),
        documents=["alpha", "beta", "gamma"],
        metadata=[{"lang": "en"}, {"lang": "zh"}, {"lang": "en"}],
    )
    assert [row.status.value for row in added.rows] == ["inserted"] * 3

    result = collection.search(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        limit=2,
        where={"lang": "en"},
    )
    assert result.valid_counts.tolist() == [2]
    records = collection.get(result[0].ids.tolist())
    print([record.document for record in records if record is not None])

    collection.checkpoint()
    collection.close()

with alayalite.connect(path, read_only=True) as database:
    with database.open_collection("docs") as collection:
        assert collection.count() == 3
```

Flat is selected explicitly above so the example runs on every wheel.

## Database lifecycle

```python
database = alayalite.connect("./data")
print(database.path)
print(database.list_collections())
database.close()
database.close()  # idempotent
```

`path=None` and `path=":memory:"` create a temporary filesystem-backed database whose lifetime is scoped to its owner.
Remote and file URI spellings are rejected. A read-write connection creates a missing root; a read-only connection
requires the root to exist and never writes it.

Collection names are safe single path components. Creation, opening, and permanent removal are deliberately separate:

```python
with alayalite.connect("./data") as database:
    collection = database.create_collection("items", config=config)
    collection.close()
    reopened = database.open_collection("items")
    reopened.close()
    database.drop_collection("items")
```

Opening a missing name raises `CollectionNotFoundError`. Creating an existing name or removing a collection with an
active handle raises `CollectionConflictError`.

## Configuration and capabilities

```python
from alayalite.config import CollectionConfig, FlatIndexConfig, QGIndexConfig

flat = CollectionConfig(
    dimension=768,
    dtype="float32",
    metric="cosine",
    index=FlatIndexConfig(),
)

qg = CollectionConfig(
    dimension=768,
    dtype="float32",
    metric="cosine",
    index=QGIndexConfig(
        max_neighbors=32,
        construction_effort=400,
        build_threads=None,
    ),
)
```

The accepted dtype values are `float32`, `int8`, and `uint8`; metrics are `l2`, `ip`, and `cosine`. Configuration is
frozen after construction. Flat is exact and portable. QG is the default, accepts float32 dimensions from 33 through
2048, and is available only when reported by the wheel:

```python
features = alayalite.capabilities()
print(features.index_types, features.laser_enabled, features.laser_simd)
```

Unsupported QG creation fails before any collection directory is created. The diagnostic states that Flat fallback is
disabled; choosing Flat is always an explicit application decision.

## Columnar writes

`add`, `replace`, and `upsert` share one keyword-only signature:

```python
result = collection.upsert(
    ids=["a", "b"],
    vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    documents=["new a", "new b"],
    metadata=[{"revision": 2}, {"revision": 2}],
    mode="atomic",
    durability="fsync",
    idempotency_key="batch-42",
)
```

- IDs must be `str`; no integer, byte, UUID, or NumPy-integer coercion occurs.
- All supplied columns must have the vector row count. Empty and ragged batches fail before native execution.
- `add` inserts only, `replace` requires an existing ID and replaces the entire row, and `upsert` inserts or replaces.
- `mode="atomic"` is the safe default. `mode="partial"` reports each independent row outcome.
- `durability="fsync"` is the default. `durability="buffered"` is searchable on return but not crash-durable.
- Reusing an idempotency key for the same completed batch returns the same operation receipt.

Row conflicts are `MutationResult.rows[*].status` values. Call-level failures use typed exceptions.

## Delete operations

```python
receipt = collection.delete(["a", "missing"])
print([row.status.value for row in receipt.rows])  # deleted, not_found

summary = collection.delete_where({"revision": {"$lt": 2}}, batch_size=1000)
print(summary.matched, summary.deleted, summary.not_found, summary.batches)
```

An empty filter is rejected by `delete_where`. Each expanded batch is atomic, but the complete matched set is not one
cross-batch transaction.

## Search and CSR results

```python
from alayalite.models import SearchBudget

result = collection.search(
    queries,
    limit=10,
    where={"lang": "en"},
    filter_policy="auto",
    selectivity_hint=0.2,
    budget=SearchBudget(scratch_bytes=64 << 20),
)
```

One vector and a query matrix return the same `SearchResult` type. Its `ids` and `distances` are flat hit columns;
`offsets` bounds each query row; status and completeness have one value per query. There is no sentinel padding.

```python
for index in range(len(result)):
    row = result[index]
    print(row.ids.tolist(), row.distances.tolist(), row.status.value, row.completeness.value)
```

Parent arrays and row views are read-only. A row shares memory with its parent, and result buffers remain valid after
the Collection closes.

For QG, omitting `effort` selects `max(100, limit)`. An explicit value below that floor is rejected. Flat rejects an
explicit effort because it has no effect there.

## Metadata filters

Metadata is a flat mapping from string keys to boolean, signed integer, finite float, or string values. Supported filter
forms include:

```python
{"kind": "paper"}
{"score": {"$ge": 0.8, "$lt": 1.0}}
{"kind": {"$in": ["paper", "note"]}}
{"$and": [{"kind": "paper"}, {"year": {"$gt": 2024}}]}
{"$or": [{"lang": "zh"}, {"lang": "en"}]}
```

The complete operator set is `$eq`, `$gt`, `$ge`, `$lt`, `$le`, `$in`, `$and`, and `$or`. Filters are validated in
Python and evaluated by the native implementation; there is no second Python evaluator.

## Aligned reads and scans

```python
records = collection.get(["a", "missing", "a"], include_vector=True)
assert records[1] is None
assert records[0] is not None and records[0].vector is not None

selected = collection.scan(where={"lang": "en"}, limit=100, include_vector=False)
```

`get()` preserves input order, duplicates, and missing positions. Projected vectors are owned read-only arrays.
`scan()` applies the filter and limit natively, then materializes only the requested projection.

Search intentionally does not attach payloads: search and a subsequent get are two snapshots. Applications such as RAG
perform those two explicit calls instead of implying snapshot atomicity that the engine does not provide.

## Maintenance and statistics

```python
checkpoint = collection.checkpoint()
seal = collection.seal()
compaction = collection.compact()
garbage = collection.collect_garbage()
stats = collection.stats()
print(stats.size, collection.count(), len(collection))
```

Writes are searchable before they return, so there is no extra flush state. A checkpoint creates a durable recovery
point and truncates eligible WAL. `rebuild_index(index=...)` can replace the index configuration while preserving live
records; it coordinates a staged same-filesystem swap.

## Read-only behavior

```python
with alayalite.connect("./data", read_only=True) as database:
    with database.open_collection("docs") as collection:
        print(collection.stats())
```

Read-only opening does not create files, repair state, truncate a torn WAL, or publish recovery markers. All catalog,
mutation, and maintenance operations fail with `CollectionNotSupportedError`. A read-write Database may narrow one
opened Collection to read-only; a read-only Database cannot upgrade a handle.

## Exception taxonomy

All SDK status failures derive from `CollectionStatusError` and preserve native metadata such as `status_code`,
`operation_stage`, `status_detail`, `retryability`, `partial`, and `status_version`. Specific classes distinguish invalid
arguments, unsupported features, conflicts, missing collections, resource exhaustion, deadlines, cancellation, I/O,
corruption, closed handles, and internal errors.

Do not parse exception message strings to choose control flow; catch the specific exception type.
