# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Example-local text splitting and embedding adapters."""

from __future__ import annotations

from examples.rag.support.chunker import get_chunker
from examples.rag.support.embedder import get_embedder


def split_text(text: str, chunk_size: int = 256, overlap: int = 25) -> list[str]:
    """Split text with the repository-local fixed-size chunker."""
    chunker = get_chunker("fix_size", chunksize=chunk_size, overlap=overlap)
    return list(chunker.chunking(text))


def embed_texts(texts: list[str], model_path: str) -> list[list[float]]:
    """Embed text using a model implementation from ``examples.rag.support``."""
    model_name = model_path.rsplit("/", maxsplit=1)[-1]
    embedder = get_embedder(model_name=model_name, model_path=model_path)
    embeddings, _dimension = embedder.get_embeddings(texts)
    return embeddings


__all__ = ["embed_texts", "split_text"]
