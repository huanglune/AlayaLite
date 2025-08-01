# Copyright 2025 AlayaDB.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides the base class for all text embedding model implementations.
"""

from typing import List, Tuple


class BaseEmbedding:
    """Abstract base class for embedding models."""

    def __init__(self, path: str) -> None:
        """
        Initializes the base embedding model.

        Args:
            path (str): The path or name of the embedding model.
        """
        self.path = path

    def get_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generates embeddings for a list of texts. This method must be implemented by subclasses.
        """
        raise NotImplementedError
