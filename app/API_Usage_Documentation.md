# FastAPI v2 usage

Base path: `/api/v2`. Requests and responses use JSON. This adapter keeps the SDK’s strict string IDs, immutable
collection schema, typed mutation receipts, and CSR search shape.

## 1. Create a collection

`POST /collections/create`

```json
{
  "collection_name": "docs",
  "dimension": 3,
  "dtype": "float32",
  "metric": "l2",
  "index": {"kind": "flat"},
  "auto_seal_rows": null
}
```

For QG, use:

```json
{
  "kind": "qg",
  "max_neighbors": 32,
  "construction_effort": 400,
  "build_threads": null
}
```

QG creation is platform-gated. The service returns HTTP 400 with the typed SDK diagnostic when the installed wheel
does not support it.

## 2. List collections

`GET /collections`

```json
["docs", "images"]
```

## 3. Permanently drop a collection

`POST /collections/drop`

```json
{"collection_name": "docs", "missing_ok": false}
```

## 4. Add records

`POST /collections/add`

```json
{
  "collection_name": "docs",
  "ids": ["a", "b"],
  "vectors": [[1, 0, 0], [0, 1, 0]],
  "documents": ["first", "second"],
  "metadata": [{"kind": "paper"}, {"kind": "note"}],
  "mode": "atomic",
  "durability": "fsync",
  "idempotency_key": "request-42"
}
```

The response contains batch watermarks and one row receipt per input ID:

```json
{
  "batch_op_id": 1,
  "visibility_watermark": 2,
  "durable_watermark": 2,
  "searchable": true,
  "durable": true,
  "durability": "fsync",
  "idempotency_key": "request-42",
  "rows": [{"id": "a", "status": "inserted"}]
}
```

The actual row objects also include operation IDs, row watermarks, searchability, durability, and idempotency fields.

## 5. Upsert records

`POST /collections/upsert` uses the same columnar payload as add. Existing IDs are replaced and missing IDs are
inserted; row receipts distinguish the outcomes.

## 6. Get aligned records

`POST /collections/get`

```json
{
  "collection_name": "docs",
  "ids": ["a", "missing", "b"],
  "include_vector": false
}
```

The response has the same length and order. A missing position is JSON `null`.

## 7. Search

`POST /collections/search`

```json
{
  "collection_name": "docs",
  "queries": [[1, 0, 0]],
  "limit": 10,
  "where": {"kind": {"$in": ["paper", "note"]}},
  "effort": null,
  "filter_policy": "auto",
  "selectivity_hint": null
}
```

Response:

```json
{
  "ids": ["a", "b"],
  "distances": [0.0, 2.0],
  "offsets": [0, 2],
  "valid_counts": [2],
  "statuses": ["ok"],
  "completeness": ["eligible_exhausted"],
  "visibility_watermark": 2,
  "metadata_epoch": 1,
  "stats": {"effective_effort": null}
}
```

`ids` and `distances` are flat CSR columns; `offsets` delimit query rows. The full stats object includes filter,
resource, I/O, lease, and rerank accounting.

## 8. Delete IDs

`POST /collections/delete`

```json
{
  "collection_name": "docs",
  "ids": ["a", "missing"],
  "mode": "atomic",
  "durability": "fsync",
  "idempotency_key": null
}
```

The response is the same mutation-receipt shape as add/upsert, with `deleted` and `not_found` row statuses.

## 9. Delete by filter

`POST /collections/delete-where`

```json
{
  "collection_name": "docs",
  "where": {"kind": "note"},
  "batch_size": 1000,
  "durability": "fsync"
}
```

```json
{"matched": 1, "deleted": 1, "not_found": 0, "batches": 1}
```

An empty filter is rejected.

## 10. Checkpoint

`POST /collections/checkpoint`

```json
{"collection_name": "docs"}
```

The typed receipt reports the durable watermark, WAL cut, metadata epoch, and checkpoint name.

## HTTP error mapping

| SDK category | HTTP status |
| --- | ---: |
| Missing collection | 404 |
| Catalog or handle conflict | 409 |
| Closed database/collection | 503 |
| Invalid input or unsupported capability | 400 |
| Other typed status failure | 500 |

Pydantic request-shape failures use FastAPI’s standard 422 response.
