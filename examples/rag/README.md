# RAG example with AlayaLite

This repository-only example demonstrates the SDK v2 retrieval flow:

1. split and embed documents with code under `examples/rag/support/`;
2. create or open an explicitly configured Flat collection;
3. add strict string IDs and columnar vectors/documents/metadata;
4. search for IDs, then call `get()` to fetch documents;
5. close the Database at the end of the UI or CLI lifecycle.

The model wrappers are example support code and are not included in the AlayaLite wheel.

## Local setup

Run from the repository root:

```bash
python -m venv .rag-venv
source .rag-venv/bin/activate
python -m pip install -r examples/rag/requirements.txt "alayalite==1.1.0"
export ALAYALITE_RAG_DATA_DIR="$PWD/rag-data"
python -m streamlit run examples/rag/ui.py
```

The first request downloads the selected embedding model. Open `http://localhost:8501`, configure an LLM endpoint,
upload a document, and ask a question. `test_docs.txt` is included as a small input.

## Docker

The Dockerfile expects the repository root as build context:

```bash
docker build -f examples/rag/Dockerfile -t alayalite-rag .
docker run --rm -p 8501:8501 -v "$PWD/rag-data:/data" alayalite-rag
```

Override `ALAYALITE_VERSION` at build time only when testing another published wheel:

```bash
docker build -f examples/rag/Dockerfile \
  --build-arg ALAYALITE_VERSION=1.1.0 \
  -t alayalite-rag .
```

## Download-free smoke test

```bash
uv run pytest examples/rag/tests -q
```

The smoke replaces splitting and embedding with deterministic functions, then verifies add, search, explicit payload
fetch, close/reopen persistence, and explicit catalog clearing.

## Files

- `db.py`: Database ownership, collection creation/opening, writes, search, and payload fetch.
- `ui.py`: Streamlit lifecycle and interaction flow.
- `llm.py`: external LLM request adapter.
- `utils.py`: small adapters over the example-local support package.
- `support/`: chunker and embedding implementations moved out of the runtime wheel.

![Overview](https://github.com/AlayaDB-AI/AlayaLite/blob/main/examples/rag/figures/overview.png?raw=true)
