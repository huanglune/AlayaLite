# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Goldens for discriminated config, effort, and platform capabilities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import alayalite._collection as collection_core
import numpy as np
import pytest


def test_collection_config_defaults_to_qg(sdk):
    config = sdk.CollectionConfig(dimension=64)
    assert isinstance(config.index, sdk.QGIndexConfig)
    assert config.index.kind == "qg"
    assert config.dtype == "float32"
    assert config.metric == "l2"
    assert config.auto_seal_rows is None


def test_index_configs_form_a_frozen_discriminated_union(sdk):
    flat = sdk.FlatIndexConfig()
    qg = sdk.QGIndexConfig(max_neighbors=64, construction_effort=500, build_threads=2)

    assert flat.kind == "flat"
    assert qg.kind == "qg"
    assert qg.max_neighbors == 64
    assert qg.construction_effort == 500
    with pytest.raises(FrozenInstanceError):
        qg.max_neighbors = 32
    with pytest.raises(TypeError):
        sdk.FlatIndexConfig(max_neighbors=32)


@pytest.mark.parametrize("metric", ["euclidean", "cos", "dot", "L2", ""])
def test_only_canonical_metric_spellings_are_accepted(sdk, metric):
    with pytest.raises(ValueError):
        sdk.CollectionConfig(dimension=3, metric=metric, index=sdk.FlatIndexConfig())


@pytest.mark.parametrize("dtype", ["float64", "float16", "i8", "", np.float32])
def test_only_canonical_dtype_strings_are_accepted(sdk, dtype):
    with pytest.raises((TypeError, ValueError)):
        sdk.CollectionConfig(dimension=3, dtype=dtype, index=sdk.FlatIndexConfig())


@pytest.mark.parametrize("dimension", [0, -1])
def test_dimension_must_be_positive(sdk, dimension):
    with pytest.raises(ValueError):
        sdk.CollectionConfig(dimension=dimension, index=sdk.FlatIndexConfig())


@pytest.mark.parametrize("dimension", [32, 2049])
def test_qg_dimension_envelope_is_validated_before_creation(sdk, dimension, tmp_path):
    config = sdk.CollectionConfig(dimension=dimension, index=sdk.QGIndexConfig())
    with sdk.connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionInvalidArgumentError):
            database.create_collection("bad-qg", config=config)
        assert database.list_collections() == []


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_qg_is_float32_only_and_fails_before_writing(sdk, dtype, tmp_path):
    config = sdk.CollectionConfig(dimension=64, dtype=dtype, index=sdk.QGIndexConfig())
    with sdk.connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionInvalidArgumentError):
            database.create_collection("bad-qg", config=config)
        assert database.list_collections() == []


def test_capabilities_are_typed_and_self_consistent(sdk):
    capabilities = sdk.capabilities()

    assert isinstance(capabilities, sdk.Capabilities)
    assert isinstance(capabilities.index_types, frozenset)
    assert "flat" in capabilities.index_types
    assert capabilities.index_types <= {"flat", "qg"}
    assert capabilities.laser_enabled is ("qg" in capabilities.index_types)
    if capabilities.laser_enabled:
        assert capabilities.laser_simd in {"generic", "avx2", "avx512"}
    else:
        assert capabilities.laser_simd is None


def test_default_qg_create_succeeds_or_fails_fast_by_capability(sdk, tmp_path):
    root = tmp_path / "database"
    config = sdk.CollectionConfig(dimension=64)
    with sdk.connect(root) as database:
        if "qg" in sdk.capabilities().index_types:
            collection = database.create_collection("default", config=config)
            assert isinstance(collection.config.index, sdk.QGIndexConfig)
            collection.close()
        else:
            with pytest.raises(sdk.CollectionNotSupportedError, match="Flat fallback is disabled"):
                database.create_collection("default", config=config)
            assert database.list_collections() == []


def test_explicit_qg_uses_the_same_create_time_platform_gate(sdk, monkeypatch, tmp_path):
    monkeypatch.setattr(
        collection_core,
        "capabilities",
        lambda: sdk.Capabilities(index_types=frozenset({"flat"}), laser_enabled=False, laser_simd=None),
    )
    config = sdk.CollectionConfig(dimension=64, index=sdk.QGIndexConfig())
    with sdk.connect(tmp_path / "database") as database:
        with pytest.raises(sdk.CollectionNotSupportedError, match="Flat fallback is disabled"):
            database.create_collection("qg", config=config)
        assert database.list_collections() == []


def test_flat_rejects_the_effort_pseudo_knob(flat_collection):
    query = np.zeros(3, dtype=np.float32)
    with pytest.raises(ValueError, match="effort"):
        flat_collection.search(query, effort=100)


def test_qg_effort_default_floor_and_explicit_validation(sdk, tmp_path):
    if "qg" not in sdk.capabilities().index_types:
        pytest.skip("QG is unavailable on this wheel")

    config = sdk.CollectionConfig(dimension=64)
    with sdk.connect(tmp_path / "database") as database:
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
