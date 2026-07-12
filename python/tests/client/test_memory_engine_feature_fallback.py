# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Runtime rollback checks for independently migrated memory graph engines."""

from pathlib import Path

import numpy as np
from alayalite._alayalitepy import (
    PyIndexInterface as _NativeIndex,
)
from alayalite._alayalitepy import (
    _MemoryEngineFeatureFlags,
)
from alayalite.schema import IndexParams

from .test_index_algorithm_identity import (
    CAPACITY,
    MAX_NBRS,
    _persisted_identity,
    _vectors,
)


def _build_with_features(index_type: str, features, case_dir: Path) -> tuple[str, tuple[str, str, str]]:
    case_dir.mkdir(parents=True)
    params = IndexParams(
        index_type=index_type,
        data_type=np.float32,
        id_type=np.uint32,
        quantization_type="none",
        metric="l2",
        capacity=CAPACITY,
        max_nbrs=MAX_NBRS,
    )
    params.fill_none_values()
    vectors = _vectors(np.float32, "none")
    native = _NativeIndex(params.to_cpp_params(), features)
    try:
        native.fit(vectors, 64, 1)
        runtime_identity = (
            native.get_declared_index_type(),
            native.get_implementation_key(),
            native.get_engine_factory_key(),
        )
        graph_path = case_dir / f"{index_type}.index"
        data_path = case_dir / "raw.data"
        native.save(str(graph_path), str(data_path), "")
    finally:
        native.close_db()

    reopened = _NativeIndex(params.to_cpp_params(), features)
    try:
        reopened.load(str(graph_path), str(data_path), "")
        assert reopened.search(vectors[0], 5, 32).shape == (5,)
    finally:
        reopened.close_db()
    return _persisted_identity(case_dir, np.uint32, "none"), runtime_identity


def test_nsg_fusion_feature_bits_restore_legacy_hnsw_independently(tmp_path):
    for disabled, enabled in (("nsg", "fusion"), ("fusion", "nsg")):
        features = _MemoryEngineFeatureFlags()
        setattr(features, f"{disabled}_segment", False)

        disabled_artifact, disabled_runtime = _build_with_features(
            disabled, features, tmp_path / f"{disabled}-legacy"
        )
        assert disabled_runtime == (disabled, "hnsw_segment", "hnsw")
        assert disabled_artifact == "hnsw"

        enabled_artifact, enabled_runtime = _build_with_features(
            enabled, features, tmp_path / f"{enabled}-current"
        )
        assert enabled_runtime == (enabled, f"{enabled}_segment", enabled)
        assert enabled_artifact == enabled
