# AlayaLite Standalone API

This directory contains the FastAPI service that exposes collection-oriented AlayaLite operations over HTTP.

## What It Provides

- FastAPI app entrypoint at `app.main:app`
- Collection management endpoints under `/api/v1/collection/*`
- Persistent storage controlled by `ALAYALITE_DATA_DIR`
- Interactive API docs at `/docs`

## Local Development

Run these commands from the repository root:

```bash
uv sync --group api --group test
uv run uvicorn app.main:app --reload
```

The service starts on `http://127.0.0.1:8000`.

### Run tests

```bash
uv run pytest app/tests -v
```

## Storage

- Default storage directory: `./data`
- Override with `ALAYALITE_DATA_DIR`
- The directory is created automatically on startup (the parent path must be writable)

Example:

```bash
ALAYALITE_DATA_DIR=./tmp-data uv run uvicorn app.main:app --reload
```

## Docker

Build from the `app/` directory:

```bash
docker build -t alayalite-standalone .
```

Run the container:

```bash
docker run -d \
  --name alayalite-standalone \
  -p 8000:8000 \
  -v "$(pwd)/data:/data" \
  -e ALAYALITE_DATA_DIR=/data \
  alayalite-standalone
```

## Endpoints

All collection endpoints use `POST`.

| Endpoint | Purpose |
| --- | --- |
| `/api/v1/collection/create` | Create a collection |
| `/api/v1/collection/set_metric` | Set the metric before the first insert |
| `/api/v1/collection/list` | List collection names |
| `/api/v1/collection/delete` | Delete a collection |
| `/api/v1/collection/reset` | Clear all in-memory collections |
| `/api/v1/collection/insert` | Insert items |
| `/api/v1/collection/query` | Search by query vectors |
| `/api/v1/collection/upsert` | Insert or update items |
| `/api/v1/collection/delete_by_id` | Delete items by id |
| `/api/v1/collection/delete_by_filter` | Delete items by metadata filter |
| `/api/v1/collection/save` | Persist a collection to disk |

The root route `GET /` returns a simple readiness message.

## Example Requests

### Create a collection

```bash
curl -X POST http://127.0.0.1:8000/api/v1/collection/create \
  -H "Content-Type: application/json" \
  -d '{"collection_name":"demo"}'
```

### Set the metric

Call this before the first insert if you do not want the default `l2` metric.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/collection/set_metric \
  -H "Content-Type: application/json" \
  -d '{"collection_name":"demo","metric":"cosine"}'
```

### Insert items

Each item is `[id, document, embedding, metadata]`.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/collection/insert \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "demo",
    "items": [
      [1, "Document 1", [0.1, 0.2, 0.3], {"category": "A"}],
      [2, "Document 2", [0.2, 0.1, 0.4], {"category": "B"}]
    ]
  }'
```

### Query

```bash
curl -X POST http://127.0.0.1:8000/api/v1/collection/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "demo",
    "query_vector": [[0.1, 0.2, 0.3]],
    "limit": 2,
    "ef_search": 10,
    "num_threads": 1
  }'
```

The response shape matches `Collection.batch_query(...)`:

```json
{
  "id": [[1, 2]],
  "document": [["Document 1", "Document 2"]],
  "metadata": [[{"category": "A"}, {"category": "B"}]],
  "distance": [[0.0, 0.03]]
}
```

### Save a collection

```bash
curl -X POST http://127.0.0.1:8000/api/v1/collection/save \
  -H "Content-Type: application/json" \
  -d '{"collection_name":"demo"}'
```

For a fuller endpoint walkthrough, see `app/API_Usage_Documentation.md`.
