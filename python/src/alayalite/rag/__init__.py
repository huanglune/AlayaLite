# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Optional RAG helpers: text chunkers and embedding-model wrappers.

Importing this package is dependency-free; the model-backed classes import their heavy dependencies
(sentence-transformers, FlagEmbedding, transformers, langchain-text-splitters) lazily on first use and
raise a descriptive ImportError otherwise. Install them with: pip install 'alayalite[rag]'.
"""

from . import chunker, embedder

__all__ = ["chunker", "embedder"]
