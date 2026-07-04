# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the SemanticChunker class, which splits text based on
semantic similarity using sentence embeddings.
"""

from typing import List

import numpy as np

from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """
    A dynamic text chunker based on semantic similarity.

    Args:
        model_name (str): Name of the semantic encoding model. Defaults to 'all-MiniLM-L6-v2'.
        threshold (float): Similarity threshold (0-1). Defaults to 0.8.
        window_size (int): Sliding window size in sentences. Defaults to 3.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.8, window_size: int = 3):
        """Initializes the Semantic Chunker."""
        try:
            from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "SemanticChunker requires the optional RAG dependencies; install with: pip install 'alayalite[rag]'"
            ) from exc

        # This class does not use the BaseChunker's __init__ as its logic is different.
        super().__init__(chunk_size=0, chunk_overlap=0)
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.window_size = window_size

    def chunking(self, docs: str) -> List[str]:
        """
        Implements the semantic-aware chunking logic.

        Args:
            docs (str): The input text.

        Returns:
            List[str]: A list of chunked text blocks.
        """
        # Step 1: Basic sentence splitting
        sentences = self._split_into_sentences(docs)

        if not sentences:
            return []

        # Step 2: Calculate sentence embeddings
        embeddings = self._encode_sentences(sentences)

        # Step 3: Sliding window analysis
        chunks = []
        current_chunk_sentences = []
        for i, sentence in enumerate(sentences):
            current_chunk_sentences.append(sentence)

            # Start checking when enough sentences have accumulated
            if len(current_chunk_sentences) >= self.window_size:
                # Compare the similarity of the current window with the next one
                if self._should_split(embeddings, i):
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []

        # Add any remaining content as the last chunk
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        return chunks

    def _should_split(self, embeddings: np.ndarray, current_index: int) -> bool:
        """
        Determines whether to split by calculating semantic similarity using a sliding window.

        Calculation logic:
        previous_window = [sent_{k-n}, ..., sent_k]
        next_window = [sent_{k+1}, ..., sent_{k+n+1}]
        similarity = cos_sim(mean(prev_window), mean(next_window))
        """
        window_size = self.window_size

        # Get the index ranges for the current and next windows
        prev_start = max(0, current_index - window_size + 1)
        next_start = current_index + 1
        next_end = min(len(embeddings), current_index + window_size + 1)

        # If there's no next window to compare, don't split
        if next_start >= len(embeddings):
            return False

        # Calculate mean embeddings
        prev_emb = np.mean(embeddings[prev_start : current_index + 1], axis=0)
        next_emb = np.mean(embeddings[next_start:next_end], axis=0)

        # Calculate cosine similarity (epsilon guards the degenerate all-zero embedding)
        denominator = np.linalg.norm(prev_emb) * np.linalg.norm(next_emb) + 1e-12
        similarity = float(np.dot(prev_emb, next_emb) / denominator)
        return similarity < self.threshold

    def _encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """Batch encode sentences into embedding vectors."""
        return self.model.encode(sentences, convert_to_numpy=True)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Basic sentence splitting (can be replaced with more complex logic if needed)."""
        # A simple split by period, can be improved with regex for more delimiters.
        return [s.strip() for s in text.split(".") if s.strip()]
