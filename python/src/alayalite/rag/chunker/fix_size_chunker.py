# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module provides the FixSizeChunker class for splitting text into fixed-size chunks.
"""

from .base import BaseChunker


class FixSizeChunker(BaseChunker):
    """
    A class for chunking text into fixed-size chunks with optional overlap.

    Attributes:
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The number of overlapping tokens between chunks.
        separator (str): The separator string to use for chunking (default is "\\n\\n").
        length_function (function): Function to calculate the length of a chunk (default is len).
    """

    def chunking(self, docs):
        """
        Splits a document into chunks of a fixed size.

        Args:
            docs (str): The document text to be chunked.

        Returns:
            list: A list of text chunks.
        """
        try:
            from langchain_text_splitters import CharacterTextSplitter  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError(
                "FixSizeChunker requires the optional RAG dependencies; install with: pip install 'alayalite[rag]'"
            ) from exc

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=False,
        )
        return text_splitter.split_text(docs)
