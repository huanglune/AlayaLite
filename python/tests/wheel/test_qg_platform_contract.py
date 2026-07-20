# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Installed-wheel smoke for the complete SDK v2 contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import alayalite
import numpy as np
import pytest
from alayalite import (
    CollectionConfig,
    CollectionNotSupportedError,
    FlatIndexConfig,
    QGIndexConfig,
    connect,
)


def test_wheel_exports_only_the_v2_root_surface() -> None:
    assert alayalite.__version__ == "1.1.0"
    assert "Client" not in alayalite.__all__
    assert "Index" not in alayalite.__all__
    assert "DiskCollection" not in alayalite.__all__
    package = Path(alayalite.__file__).parent
    assert (package / "py.typed").is_file()
    assert (package / "_alayalitepy.pyi").is_file()
    for module in ("client", "schema", "laser", "rag", "vamana"):
        assert importlib.util.find_spec(f"alayalite.{module}") is None


def test_flat_wheel_crud_checkpoint_reopen_and_read_only(tmp_path: Path) -> None:
    root = tmp_path / "database"
    config = CollectionConfig(dimension=3, metric="cosine", index=FlatIndexConfig())
    with connect(root) as database:
        collection = database.create_collection("flat", config=config)
        collection.add(
            ids=["a", "b"],
            vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            documents=["A", "B"],
        )
        assert collection.search(np.asarray([1, 0, 0], dtype=np.float32), limit=2)[0].ids.tolist() == ["a", "b"]
        collection.checkpoint()
        collection.close()

    with connect(root, read_only=True) as database:
        with database.open_collection("flat") as collection:
            assert collection.get(["b"])[0].document == "B"


def test_qg_wheel_succeeds_or_fails_at_create_by_capability(tmp_path: Path) -> None:
    root = tmp_path / "qg-database"
    config = CollectionConfig(dimension=64, index=QGIndexConfig())
    with connect(root) as database:
        if "qg" not in alayalite.capabilities().index_types:
            with pytest.raises(CollectionNotSupportedError, match="Flat fallback is disabled"):
                database.create_collection("qg", config=config)
            assert database.list_collections() == []
            return

        collection = database.create_collection("qg", config=config)
        vectors = np.arange(40 * 64, dtype=np.float32).reshape(40, 64) / np.float32(257.0)
        collection.add(ids=[str(row) for row in range(len(vectors))], vectors=vectors)
        collection.seal()
        result = collection.search(vectors[0], limit=5)
        assert result[0].ids.size == 5
        collection.close()
