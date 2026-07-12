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


def _build_with_features(
    index_type: str, features, case_dir: Path, quant: str = "none"
) -> tuple[str, tuple[str, str, str]]:
    case_dir.mkdir(parents=True)
    params = IndexParams(
        index_type=index_type,
        data_type=np.float32,
        id_type=np.uint32,
        quantization_type=quant,
        metric="l2",
        capacity=CAPACITY,
        max_nbrs=MAX_NBRS,
    )
    params.fill_none_values()
    vectors = _vectors(np.float32, quant)
    native = _NativeIndex(params.to_cpp_params(), features)
    try:
        native.fit(vectors, 64, 1)
        runtime_identity = (
            native.get_declared_index_type(),
            native.get_implementation_key(),
            native.get_engine_factory_key(),
        )
        if quant == "rabitq":
            graph_path = ""
            data_path = ""
            quant_path = case_dir / "rabitq.data"
        else:
            graph_path = case_dir / f"{index_type}.index"
            data_path = case_dir / "raw.data"
            quant_path = ""
        native.save(str(graph_path), str(data_path), str(quant_path))
    finally:
        native.close_db()

    reopened = _NativeIndex(params.to_cpp_params(), features)
    try:
        reopened.load(str(graph_path), str(data_path), str(quant_path))
        assert reopened.search(vectors[0], 5, 32).shape == (5,)
    finally:
        reopened.close_db()
    return _persisted_identity(case_dir, np.uint32, quant), runtime_identity


def test_nsg_fusion_feature_bits_restore_legacy_hnsw_independently(tmp_path):
    for disabled, enabled in (("nsg", "fusion"), ("fusion", "nsg")):
        features = _MemoryEngineFeatureFlags()
        setattr(features, f"{disabled}_segment", False)

        disabled_artifact, disabled_runtime = _build_with_features(disabled, features, tmp_path / f"{disabled}-legacy")
        assert disabled_runtime == (disabled, "hnsw_segment", "hnsw")
        assert disabled_artifact == "hnsw"

        enabled_artifact, enabled_runtime = _build_with_features(enabled, features, tmp_path / f"{enabled}-current")
        assert enabled_runtime == (enabled, f"{enabled}_segment", enabled)
        assert enabled_artifact == enabled


def test_qg_feature_bit_restores_legacy_model_without_changing_qg_behavior(tmp_path):
    features = _MemoryEngineFeatureFlags()
    features.qg_segment = False

    legacy_artifact, legacy_runtime = _build_with_features("hnsw", features, tmp_path / "qg-legacy", quant="rabitq")
    assert legacy_runtime == ("hnsw", "legacy_qg_model", "qg")
    assert legacy_artifact == "qg"

    current_features = _MemoryEngineFeatureFlags()
    current_artifact, current_runtime = _build_with_features(
        "fusion", current_features, tmp_path / "qg-current", quant="rabitq"
    )
    assert current_runtime == ("fusion", "qg_segment", "qg")
    assert current_artifact == "qg"

    # The QG switch is row-scoped and does not disable Fusion's non-RaBitQ segment.
    fusion_artifact, fusion_runtime = _build_with_features("fusion", features, tmp_path / "fusion-current")
    assert fusion_runtime == ("fusion", "fusion_segment", "fusion")
    assert fusion_artifact == "fusion"
