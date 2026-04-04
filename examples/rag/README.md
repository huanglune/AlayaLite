# RAG Example

This example shows how to build a small document Q&A app with AlayaLite, Streamlit, FlagEmbedding, and an OpenAI-compatible chat completion endpoint.

## What Is Included

| File | Purpose |
| --- | --- |
| `ui.py` | Streamlit interface for upload, chat, and export |
| `db.py` | AlayaLite-backed insert, reset, and retrieval helpers |
| `llm.py` | OpenAI-compatible chat completion request helper |
| `utils.py` | Text splitting and embedding helpers |
| `test_docs.txt` | Sample input document |
| `figures/` | Screenshots used for walkthroughs |

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

Notes:

- The Docker image installs Python 3.11, `alayalite`, Streamlit, `FlagEmbedding`, `pypdf`, and `python-docx`.
- The image appends `HF_ENDPOINT=https://hf-mirror.com` to the shell profile; keep it, change it, or remove it based on your network environment.

## Local Run

Install the required Python packages first:

```bash
pip install alayalite streamlit langchain_text_splitters FlagEmbedding python-docx pypdf requests torch
cd examples/rag
streamlit run ui.py
```

## How It Works

1. Upload documents from the sidebar.
2. The app normalizes file content and splits it into chunks.
3. `FlagEmbedding` generates embeddings for each chunk.
4. Chunks are inserted into an in-memory AlayaLite collection named `rag_collection`.
5. Queries are expanded with recent chat history, retrieved from AlayaLite, and passed to the LLM helper.

## UI Settings

The sidebar lets you configure:

- `LLM Base URL`
- `LLM API Key`
- `LLM Model`
- `Embedding Model`

Behavior notes:

- The current `llm.py` appends `/completions` to the base URL you enter.
- The default embedding model path is `BAAI/bge-small-zh-v1.5`.
- Uploading a new batch resets the in-memory database before processing.

## Current Limitations

- The example currently supports BGE-style embedding models only.
- Retrieval state is in memory only; restarting the app clears the collection.
- The first embedding request may take time because the model can be downloaded on demand.

## Tips

- Try `test_docs.txt` for a quick smoke test.
- When running in Codespaces or a similar environment, forward port `8501` to access the Streamlit UI.
