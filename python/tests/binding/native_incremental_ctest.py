# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""CTest entry that validates the built extension without an installed wheel."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np


def _load_extension(path: Path):
    spec = importlib.util.spec_from_file_location("_alayalitepy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = _load_extension(Path(sys.argv[1]))
    capabilities = module.capabilities()
    assert capabilities.index_types[0] == "flat"
    assert capabilities.laser_enabled is ("qg" in capabilities.index_types)

    with tempfile.TemporaryDirectory(prefix="alayalite-binding-ctest-") as directory:
        collection_type = module._Collection  # pylint: disable=protected-access
        collection = collection_type.create(
            str(Path(directory) / "collection"),
            3,
            "l2",
            np.dtype(np.float32),
            "flat",
            "none",
        )
        receipt = collection.mutate_typed(
            ["a", "b"],
            ["A", "B"],
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            [{"kind": "keep"}, {"kind": "drop"}],
            "add",
        )
        assert type(receipt).__name__ == "_MutationResponse"
        assert [row.row_status for row in receipt.rows] == [0, 0]
        records = collection.scan(metadata_filter={"kind": "keep"}, limit=1)
        assert [(record.id, record.vector) for record in records] == [("a", None)]
        response = collection.search_typed(np.zeros(3, dtype=np.float32), 2)
        assert response.ids.tolist() == ["a", "b"]
        assert response.search_stats.effective_effort is None
        assert isinstance(collection.search(np.zeros(3, dtype=np.float32), 2), dict)
        collection.close()

        root = Path(directory) / "collection"
        before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        reader = module._Collection.open(str(root), True)  # pylint: disable=protected-access
        assert reader.read_only is True
        try:
            reader.remove_typed(["a"])
        except module.CollectionNotSupportedError as error:
            assert error.status_detail == 15
        else:
            raise AssertionError("read-only mutation unexpectedly succeeded")
        reader.close()
        after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
        assert after == before


if __name__ == "__main__":
    main()
