# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Small RAG workflow built on the SDK v2 Database/Collection lifecycle."""

from __future__ import annotations

import atexit
import os
import traceback
import uuid
from pathlib import Path

import numpy as np
from alayalite import (
    CollectionConfig,
    CollectionNotFoundError,
    Database,
    FlatIndexConfig,
    connect,
)

from examples.rag.utils import embed_texts, split_text

_database: Database | None = None


def open_database(path: str | os.PathLike[str] | None = None) -> Database:
    """Open the example database once and return its owner."""
    global _database  # pylint: disable=global-statement
    if _database is None:
        location = path or os.environ.get("ALAYALITE_RAG_DATA_DIR", "./rag-data")
        _database = connect(location)
    return _database


def close_database() -> None:
    """Close the example database; safe to call repeatedly."""
    global _database  # pylint: disable=global-statement
    if _database is not None:
        _database.close()
        _database = None


def clear_database() -> None:
    """Explicitly drop every collection owned by this example."""
    database = open_database()
    for name in database.list_collections():
        database.drop_collection(name)


def insert_text(
    collection_name: str,
    embed_model_path: str,
    docs: str,
    chunk_size: int = 256,
    overlap: int = 25,
) -> bool:
    """Split, embed, and add text chunks to a collection."""
    chunks = split_text(docs, chunk_size, overlap)
    embeddings = embed_texts(chunks, embed_model_path)
    if not chunks or len(embeddings) != len(chunks):
        print("Embedding failed; no chunks were added")
        return False

    try:
        vectors = np.asarray(embeddings, dtype=np.float32)
        database = open_database()
        if collection_name in database.list_collections():
            collection = database.open_collection(collection_name)
        else:
            collection = database.create_collection(
                collection_name,
                config=CollectionConfig(
                    dimension=int(vectors.shape[1]),
                    dtype="float32",
                    metric="cosine",
                    index=FlatIndexConfig(),
                ),
            )
        with collection:
            collection.add(
                ids=[uuid.uuid4().hex for _ in chunks],
                vectors=vectors,
                documents=chunks,
                metadata=[{"chunk": index} for index in range(len(chunks))],
            )
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"Error while adding chunks: {error}")
        traceback.print_exc()
        return False
    return True


def query_text(collection_name: str, embed_model_path: str, query: str, top_k: int = 5) -> str:
    """Search for relevant chunks, then explicitly fetch their documents."""
    try:
        database = open_database()
        with database.open_collection(collection_name) as collection:
            queries = np.asarray(embed_texts([query], embed_model_path), dtype=np.float32)
            result = collection.search(queries, limit=top_k)
            records = collection.get(result[0].ids.tolist())
        return "\n\n".join(record.document for record in records if record is not None)
    except CollectionNotFoundError:
        return ""
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"Error during retrieval: {error}")
        traceback.print_exc()
        return ""


atexit.register(close_database)


if __name__ == "__main__":
    from examples.rag.llm import ask_llm

    sample_text = Path(__file__).with_name("test_docs.txt").read_text(encoding="utf-8")
    model_path = "BAAI/bge-small-zh-v1.5"
    try:
        insert_text("test", model_path, sample_text, chunk_size=128)
        retrieved = query_text("test", model_path, "What are higher-order chunking techniques?")
        answer = ask_llm(
            "Your LLM service base URL here",
            "Your API key here",  # pragma: allowlist secret
            "deepseek-v3",
            query="What are higher-order chunking techniques?",
            retrieved_docs=retrieved,
            is_stream=False,
        )
        print(f"=== Response ===\n{answer}")
    finally:
        close_database()


__all__ = ["clear_database", "close_database", "insert_text", "open_database", "query_text"]
