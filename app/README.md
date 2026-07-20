# AlayaLite FastAPI example

This directory is an example HTTP adapter around the SDK v2 surface. It is not a separate network mode of the Python
SDK. FastAPI lifespan owns one `Database`, collection handles are scoped to requests, and typed SDK errors map to HTTP
status codes without parsing message strings.

## Development setup

Run from the repository root:

```bash
uv sync
uv pip install -r app/requirements.txt
uv run uvicorn app.main:app --reload
```

The API is available under `/api/v2`; interactive OpenAPI documentation is at `/docs`.

## Test and smoke

```bash
uv run pytest app/tests -q
```

The tests start the real application lifespan and exercise create, add, get, search, upsert, delete, filtered delete,
checkpoint, drop, and close/reopen persistence.

## Docker

Build with the repository root as context:

```bash
docker build -f app/Dockerfile -t alayalite-fastapi .
docker run --rm -p 8000:8000 -v "$PWD/alaya-data:/data" alayalite-fastapi
```

Set `ALAYALITE_DATA_DIR` to choose the database directory. Successful default writes are already searchable and fsync
durable; the checkpoint endpoint creates a recovery point and truncates eligible WAL rather than “saving” volatile
application state.

## Minimal flow

Create a portable Flat collection:

```bash
curl -sS -X POST http://localhost:8000/api/v2/collections/create \
  -H 'content-type: application/json' \
  -d '{
    "collection_name": "docs",
    "dimension": 3,
    "dtype": "float32",
    "metric": "cosine",
    "index": {"kind": "flat"}
  }'
```

Add columnar records:

```bash
curl -sS -X POST http://localhost:8000/api/v2/collections/add \
  -H 'content-type: application/json' \
  -d '{
    "collection_name": "docs",
    "ids": ["a", "b"],
    "vectors": [[1, 0, 0], [0, 1, 0]],
    "documents": ["first", "second"],
    "metadata": [{"lang": "en"}, {"lang": "zh"}]
  }'
```

Search and receive JSON-encoded CSR columns:

```bash
curl -sS -X POST http://localhost:8000/api/v2/collections/search \
  -H 'content-type: application/json' \
  -d '{
    "collection_name": "docs",
    "queries": [[1, 0, 0]],
    "limit": 2,
    "where": {"lang": {"$in": ["en", "zh"]}}
  }'
```

See [API usage](API_Usage_Documentation.md) for every request shape and response field.
