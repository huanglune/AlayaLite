# RAG Example

This example shows how to build a small document Q&A app with AlayaLite, Streamlit, and an OpenAI-compatible chat completion endpoint.

## What Is Included

| File | Purpose |
| --- | --- |
| `ui.py` | Streamlit interface for upload, chat, and export |
| `db.py` | AlayaLite-backed insert and retrieval helpers |
| `llm.py` | Chat completion request helper |
| `utils.py` | Text splitting and embedding helpers |
| `test_docs.txt` | Sample input document |

The app accepts `txt`, `md`, `pdf`, and `docx` files.

## Quick Start With Docker

From `examples/rag/`:

```bash
docker build -t alayalite-rag .
docker run -it --rm -p 8501:8501 -v "$(pwd):/app" alayalite-rag /bin/bash
```

Inside the container:

```bash
streamlit run ui.py
```

Then open `http://127.0.0.1:8501`.

## Local Run

If you prefer to run it without Docker, install the required Python packages first:

```bash
pip install alayalite streamlit langchain_text_splitters FlagEmbedding python-docx pypdf requests torch
cd examples/rag
streamlit run ui.py
```

## How It Works

1. Upload documents in the sidebar.
2. The app normalizes file content and splits it into chunks.
3. `FlagEmbedding` generates embeddings for each chunk.
4. Chunks are inserted into an in-memory AlayaLite collection named `rag_collection`.
5. Queries are embedded, retrieved from AlayaLite, and passed to the LLM request helper.

## UI Settings

The sidebar lets you configure:

- `LLM Base URL`
- `LLM API Key`
- `LLM Model`
- `Embedding Model`

The default chat endpoint logic appends `/completions` to the base URL. For example, a base URL ending in `/v1/chat` becomes `/v1/chat/completions`.

## Notes

- Uploading a new batch of files resets the in-memory database before processing.
- The first embedding request may take time because the model can be downloaded on demand.
- The Docker image sets `HF_ENDPOINT=https://hf-mirror.com`; set or remove that environment variable based on your environment.
- When running in Codespaces or a similar environment, forward port `8501` to access the Streamlit UI.
