# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Live internal-core counterparts for the 23 config and effort goldens."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from alayalite._capabilities import capabilities
from alayalite._database import connect
from alayalite.config import CollectionConfig, FlatIndexConfig, QGIndexConfig
from alayalite.models import Capabilities


def test_collection_config_defaults_to_qg():
    config = CollectionConfig(dimension=64)
    assert isinstance(config.index, QGIndexConfig)
    assert config.index.kind == "qg"
    assert config.dtype == "float32"
    assert config.metric == "l2"
    assert config.auto_seal_rows is None


def test_index_configs_form_a_frozen_discriminated_union():
    flat = FlatIndexConfig()
    qg = QGIndexConfig(max_neighbors=64, construction_effort=500, build_threads=2)

    assert flat.kind == "flat"
    assert qg.kind == "qg"
    assert qg.max_neighbors == 64
    assert qg.construction_effort == 500
    with pytest.raises(FrozenInstanceError):
        qg.max_neighbors = 32
    with pytest.raises(TypeError):
        FlatIndexConfig(max_neighbors=32)  # pylint: disable=unexpected-keyword-arg


@pytest.mark.parametrize("metric", ["euclidean", "cos", "dot", "L2", ""])
def test_only_canonical_metric_spellings_are_accepted(metric):
    with pytest.raises(ValueError):
        CollectionConfig(dimension=3, metric=metric, index=FlatIndexConfig())


@pytest.mark.parametrize("dtype", ["float64", "float16", "i8", "", np.float32])
def test_only_canonical_dtype_strings_are_accepted(dtype):
    with pytest.raises((TypeError, ValueError)):
        CollectionConfig(dimension=3, dtype=dtype, index=FlatIndexConfig())


@pytest.mark.parametrize("dimension", [0, -1])
def test_dimension_must_be_positive(dimension):
    with pytest.raises(ValueError):
        CollectionConfig(dimension=dimension, index=FlatIndexConfig())


@pytest.mark.parametrize("dimension", [32, 2049])
def test_qg_dimension_envelope_is_validated_before_creation(sdk, dimension, tmp_path):
    config = CollectionConfig(dimension=dimension, index=QGIndexConfig())
    with connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionInvalidArgumentError):
            database.create_collection("bad-qg", config=config)
        assert database.list_collections() == []


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_qg_is_float32_only_and_fails_before_writing(sdk, dtype, tmp_path):
    config = CollectionConfig(dimension=64, dtype=dtype, index=QGIndexConfig())
    with connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionInvalidArgumentError):
            database.create_collection("bad-qg", config=config)
        assert database.list_collections() == []


def test_capabilities_are_typed_and_self_consistent():
    result = capabilities()

    assert isinstance(result, Capabilities)
    assert isinstance(result.index_types, frozenset)
    assert "flat" in result.index_types
    assert result.index_types <= {"flat", "qg"}
    assert result.laser_enabled is ("qg" in result.index_types)
    if result.laser_enabled:
        assert result.laser_simd in {"generic", "avx2", "avx512"}
    else:
        assert result.laser_simd is None


def test_default_qg_create_succeeds_or_fails_fast_by_capability(sdk, tmp_path):
    root = tmp_path / "database"
    config = CollectionConfig(dimension=64)
    with connect(root) as database:
        if "qg" in capabilities().index_types:
            collection = database.create_collection("default", config=config)
            assert isinstance(collection.config.index, QGIndexConfig)
            collection.close()
        else:
            with pytest.raises(sdk.CollectionNotSupportedError, match="Flat fallback is disabled"):
                database.create_collection("default", config=config)
            assert database.list_collections() == []


def test_explicit_qg_uses_the_same_create_time_platform_gate(sdk, tmp_path):
    if "qg" in capabilities().index_types:
        pytest.skip("unsupported-platform diagnostic is not reachable on this wheel")
    config = CollectionConfig(dimension=64, index=QGIndexConfig())
    with connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionNotSupportedError, match="Flat fallback is disabled"):
            database.create_collection("qg", config=config)
        assert database.list_collections() == []


def test_flat_rejects_the_effort_pseudo_knob(flat_collection):
    query = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="effort"):
        flat_collection.search(query, effort=100)


def test_qg_effort_default_floor_and_explicit_validation(tmp_path):
    if "qg" not in capabilities().index_types:
        pytest.skip("QG is unavailable on this wheel")

    config = CollectionConfig(dimension=64)
    with connect(tmp_path / "database") as database:
        collection = database.create_collection("qg", config=config)
        rng = np.random.default_rng(20260719)
        vectors = rng.normal(size=(128, 64)).astype(np.float32)
        collection.add(ids=[str(row) for row in range(len(vectors))], vectors=vectors)
        collection.seal()

        default = collection.search(vectors[0], limit=10)
        raised_floor = collection.search(vectors[0], limit=125)
        explicit = collection.search(vectors[0], limit=10, effort=200)
        assert default.stats.effective_effort == 100
        assert raised_floor.stats.effective_effort == 125
        assert explicit.stats.effective_effort == 200

        for limit, effort in ((10, 99), (125, 124)):
            with pytest.raises(ValueError, match="effort"):
                collection.search(vectors[0], limit=limit, effort=effort)
        collection.close()
