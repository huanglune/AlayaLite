# AlayaLite Python SDK

The Python package exposes three main objects:

- `Client` for lifecycle and persistence
- `Collection` for document + metadata + vector workflows
- `Index` for lower-level vector indexing operations

Utilities such as `calc_gt`, `calc_recall`, `load_fvecs`, and `load_ivecs` are also available from `alayalite`.

## Install

```bash
pip install alayalite
```

For local development from the repository root:

```bash
uv sync
```

## Core Concepts

### `Client`

`Client` manages named collections and indices. Passing `url=...` enables persistence and auto-loads saved objects from that directory at startup.

```python
from alayalite import Client

memory_client = Client()
disk_client = Client(url="./data")
```

Useful methods:

- `list_collections()`
- `list_indices()`
- `get_collection(name="default")`
- `get_index(name="default")`
- `create_collection(name="default", **kwargs)`
- `create_index(name="default", **kwargs)`
- `get_or_create_collection(name, **kwargs)`
- `get_or_create_index(name, **kwargs)`
- `delete_collection(collection_name, delete_on_disk=False)`
- `delete_index(index_name, delete_on_disk=False)`
- `save_collection(collection_name)`
- `save_index(index_name)`
- `reset(delete_on_disk=False)`

Persistence notes:

- `save_*()` requires `Client(url=...)`.
- Saved collections and indices are reloaded automatically when a new client points at the same directory.
- `delete_on_disk=True` removes the persisted directory as well as the in-memory handle.

## Collection API

`Collection` stores raw documents and metadata alongside an internal vector index.

### Create and insert

```python
from alayalite import Client
import numpy as np

client = Client()
collection = client.create_collection("docs", metric="cosine")

items = [
    (1, "Document 1", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"}),
    (2, "Document 2", np.array([0.2, 0.1, 0.4], dtype=np.float32), {"category": "B"}),
]

collection.insert(items)
```

Item format:

```python
(id, document, embedding, metadata)
```

Behavior notes:

- The first insert initializes the backing index and fixes vector dimensionality.
- Call `set_metric(metric)` before the first insert if you want something other than the default `l2`.
- `insert([])` and `upsert([])` are treated as no-ops.

### Query

```python
result = collection.batch_query(
    [[0.1, 0.2, 0.3]],
    limit=2,
    ef_search=10,
    num_threads=1,
)
```

`batch_query(...)` returns a dictionary shaped like:

```python
{
    "id": [[1, 2]],
    "document": [["Document 1", "Document 2"]],
    "metadata": [[{"category": "A"}, {"category": "B"}]],
    "distance": [[0.0, 0.03]],
}
```

### Other collection operations

```python
collection.upsert([
    (1, "Updated document", np.array([0.1, 0.2, 0.3], dtype=np.float32), {"category": "A"})
])

collection.get_by_id([1, 2])
collection.filter_query({"category": "A"})
collection.delete_by_id([2])
collection.delete_by_filter({"category": "A"})
collection.reindex()
```

Notes:

- `filter_query(...)` performs exact-match filtering over metadata keys.
- `reindex()` is useful after large delete-heavy workloads because it rebuilds the underlying index and ID maps.

## Index API

`Index` is the lower-level vector interface when you want to manage embeddings only.

### Create and fit

```python
from alayalite import Client
import numpy as np

vectors = np.random.rand(1000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)

client = Client()
index = client.create_index(
    "ann",
    index_type="hnsw",
    metric="l2",
    quantization_type="none",
)

index.fit(vectors, ef_construction=100, num_threads=1)
```

An `Index` can only be fitted once.

### Search and update

```python
neighbors = index.search(queries[0], topk=10, ef_search=100)
batch_neighbors = index.batch_search(queries, topk=10, ef_search=100, num_threads=1)
ids, distances = index.batch_search_with_distance(queries, topk=10, ef_search=100, num_threads=1)

inserted_id = index.insert(np.random.rand(128).astype(np.float32), ef=100)
index.remove(inserted_id)
vector = index.get_data_by_id(0)
dim = index.get_dim()
dtype = index.get_dtype()
```

### Persistence

```python
disk_client = Client(url="./data")
index = disk_client.create_index("saved_index", metric="l2")
index.fit(vectors)
disk_client.save_index("saved_index")

loaded = disk_client.get_index("saved_index")
```

You can also load directly:

```python
from alayalite import Index

loaded = Index.load("./data", "saved_index")
```

## Supported Parameters

`create_collection(...)` and `create_index(...)` accept the same index-related keyword arguments.

| Parameter | Allowed values | Default |
| --- | --- | --- |
| `index_type` | `hnsw`, `nsg`, `fusion` | `hnsw` |
| `metric` | `l2`, `euclidean`, `ip`, `cosine`, `cos` | `l2` |
| `quantization_type` | `none`, `sq8`, `sq4`, `rabitq` | `none` |
| `data_type` | `numpy.float32`, `numpy.float64`, `numpy.int8`, `numpy.uint8`, `numpy.int32`, `numpy.uint32` | inferred / `float32` |
| `id_type` | `numpy.uint32`, `numpy.uint64` | `uint32` |
| `capacity` | positive integer | `100000` |
| `max_nbrs` | integer in `(0, 1000)` | `32` |

Notes:

- `rabitq` support is intended for x86_64 environments with AVX-512.
- The Python SDK currently exposes graph-style indices; DiskANN-specific components are C++-side today.

If you need direct access to the parameter object, import `IndexParams` from `alayalite.schema`.

## Utilities

```python
from alayalite import calc_gt, calc_recall, load_fvecs, load_ivecs
```

- `calc_gt(data, query, topk)` builds an exact L2 ground-truth matrix.
- `calc_recall(result, gt_data)` computes recall for a result matrix.
- `load_fvecs(path)` loads `fvecs` files into `numpy.float32`.
- `load_ivecs(path)` loads `ivecs` files into `numpy.int32`.

## RAG Helpers

The package also ships reusable RAG helpers under `alayalite.rag.*`, including chunkers and embedders for higher-level retrieval pipelines.

## End-to-End Example

```python
from alayalite import Client, calc_gt, calc_recall
import numpy as np

vectors = np.random.rand(1000, 128).astype(np.float32)
queries = np.random.rand(10, 128).astype(np.float32)

client = Client()
index = client.create_index("default", metric="l2")
index.fit(vectors)

result = index.batch_search(queries, topk=10)
gt = calc_gt(vectors, queries, 10)
print(calc_recall(result, gt))
```
