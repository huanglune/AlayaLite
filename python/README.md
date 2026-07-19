# AlayaLite Python interface

The supported Python entry point is the canonical `Collection`, optionally
managed by `Client`. The former `Index`, `DiskCollection`, LASER builder, and
Vamana builder APIs were removed; importing those names raises the documented
legacy-API error instead of selecting a compatibility implementation.

See the [Client User Guide](../docs/user/CLIENT_USER_MANUAL.md) for the full
filter, durability, checkpoint, and recovery contracts.

## Install

```bash
pip install alayalite
# or
uv add alayalite
```

## Create and query a collection

`flat` is the portable exact-search family and works on every wheel:

```python
import numpy as np
from alayalite import Client

client = Client()
collection = client.create_collection(
    "docs",
    index_type="flat",
    metric="l2",
    indexed_fields=["category"],
)

vectors = np.random.default_rng(42).random((100, 64), dtype=np.float32)
collection.insert(
    [
        (f"doc-{i}", f"Document {i}", vector, {"category": "a" if i % 2 == 0 else "b"})
        for i, vector in enumerate(vectors)
    ]
)

result = collection.batch_query(vectors[:2], limit=5, ef_search=20)
print(result["id"])

filtered = collection.hybrid_query(
    vectors[:2],
    limit=5,
    metadata_filter={"category": "a"},
    ef_search=20,
)
print(filtered["id"])
```

The public index-family values are `flat` and `qg`. On LASER-enabled Linux
x86_64 and macOS builds, eligible sealed `qg` generations use LASER. Linux
aarch64 and Windows wheels are flat-only: sealing `qg` raises
`CollectionNotSupportedError` and never falls back silently.

## Mutations

Items are `(id, document, embedding, metadata)` tuples. IDs are normalized to
strings, vectors in one collection must share a dtype and dimension, and
metadata may be an empty dictionary.

```python
collection.upsert([
    ("doc-0", "Updated document", vectors[0], {"category": "a"}),
])
collection.delete_by_id(["doc-1"])
collection.delete_by_filter({"category": "b"})
```

For strict native result metadata, use `search` or `batch_search`. The
compatibility projections `batch_query` and `hybrid_query` return document and
metadata-oriented dictionaries. Their `num_threads` argument is deprecated;
issue independent queries from external threads when concurrency is needed.

## Persistence through Client

Give `Client` a root directory when collections must survive process restarts:

```python
client = Client("./alaya-data")
collection = client.get_or_create_collection("docs", index_type="flat")

# Insert or mutate rows, then persist discovery metadata and a checkpoint.
client.save_collection("docs")

reopened = Client("./alaya-data").get_collection("docs")
print(reopened.stats())
```

`Client` supports collection management only:
`list_collections`, `get_collection`, `create_collection`,
`get_or_create_collection`, `save_collection`, `delete_collection`, and
`reset`.
