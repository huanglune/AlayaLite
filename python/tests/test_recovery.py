# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Crash-recovery tests for SDK v2 collection workflows."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

from alayalite import connect


def _run_crashing_child(root: Path, body: str, exit_code: int = 91) -> subprocess.CompletedProcess[str]:
    script = "\n".join(
        [
            "import os",
            "import numpy as np",
            "from alayalite import CollectionConfig, FlatIndexConfig, connect",
            f"database = connect({str(root)!r})",
            textwrap.dedent(body).strip(),
            f"os._exit({exit_code})",
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_collection_recovers_after_unclean_exit(tmp_path: Path) -> None:
    result = _run_crashing_child(
        tmp_path,
        """
        collection = database.create_collection(
            "recovering",
            config=CollectionConfig(dimension=3, index=FlatIndexConfig()),
        )
        collection.add(
            ids=["a", "b"],
            vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            documents=["Document A", "Document B"],
            metadata=[{"group": "keep"}, {"group": "drop"}],
        )
        collection.upsert(
            ids=["a", "c"],
            vectors=np.asarray([[1, 0.1, 0], [0, 0, 1]], dtype=np.float32),
            documents=["Document A v2", "Document C"],
            metadata=[{"group": "keep", "v": 2}, {"group": "keep"}],
        )
        collection.delete(["b"])
        """,
    )
    assert result.returncode == 91, result.stdout + result.stderr

    collection_root = tmp_path / "recovering"
    recovery_dir = collection_root / ".alaya_internal" / "collection_wal_v1"
    assert (collection_root / "schema.json").is_file()
    assert (recovery_dir / "CURRENT").is_file()
    assert (recovery_dir / "checkpoint_0.bin").is_file()
    assert (recovery_dir / "logical.wal").is_file()

    with connect(tmp_path) as database:
        with database.open_collection("recovering") as collection:
            records = collection.get(["a", "b", "c"])
            assert [record.id if record is not None else None for record in records] == ["a", None, "c"]
            assert records[0] is not None
            assert records[0].document == "Document A v2"
            assert records[0].metadata["v"] == 2


def test_recovery_is_idempotent_across_restarts(tmp_path: Path) -> None:
    result = _run_crashing_child(
        tmp_path,
        """
        collection = database.create_collection(
            "idempotent",
            config=CollectionConfig(dimension=3, index=FlatIndexConfig()),
        )
        collection.add(
            ids=["a", "b"],
            vectors=np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            documents=["A", "B"],
        )
        collection.upsert(
            ids=["a"],
            vectors=np.asarray([[1, 0.2, 0]], dtype=np.float32),
            documents=["A v2"],
            metadata=[{"kind": "updated"}],
        )
        collection.delete(["b"])
        """,
    )
    assert result.returncode == 91, result.stdout + result.stderr

    snapshots = []
    for _ in range(2):
        with connect(tmp_path) as database:
            with database.open_collection("idempotent") as collection:
                record = collection.get(["a", "b"])
                snapshots.append(
                    [
                        None if item is None else (item.id, item.document, dict(item.metadata), item.version)
                        for item in record
                    ]
                )
    assert snapshots[0] == snapshots[1]
