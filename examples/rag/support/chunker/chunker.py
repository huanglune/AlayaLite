# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides factory functions to create and use various text chunkers.
"""

from .fix_size_chunker import FixSizeChunker
from .semantic_chunker import SemanticChunker
from .sentence_chunker import SentenceChunker


def get_chunker(chunk_model="fix_size", chunksize=3, overlap=0, semantic_model="all-MiniLM-L6-v2"):
    """
    Factory function to get a chunker instance based on the specified model.
    """
    if chunk_model == "fix_size":
        chunker_instance = FixSizeChunker(chunk_size=chunksize, chunk_overlap=overlap)
    elif chunk_model == "semantic":
        chunker_instance = SemanticChunker(window_size=chunksize, model_name=semantic_model)
    elif chunk_model == "sentence":
        chunker_instance = SentenceChunker(chunk_size=chunksize, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unsupported chunk model: {chunk_model}")
    return chunker_instance


def chunker(docs, chunk_model="fix_size", chunksize=3, overlap=0, semantic_model="all-MiniLM-L6-v2"):
    """
    Creates a chunker and applies it to the given documents.
    """
    chunker_instance = get_chunker(chunk_model, chunksize, overlap, semantic_model)
    return chunker_instance.chunking(docs)


# Example usage:
#
# sample_text = "Your sample text goes here..."
# chunks = chunker(sample_text, 'fix_size', 512, 50)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
#
# chunks = chunker(sample_text, 'semantic', 3)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
#
# chunks = chunker(sample_text, 'sentence', 10, 2)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i+1}:\n{chunk}\n{'-'*40}")
