# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Persisted-artifact identity checks for every generated Python dispatch row.

The configured index type is not an identity signal: filenames are derived from
``IndexParams.index_type`` even when a different builder ran.  These tests parse
the native ``Graph::save`` layout instead.  An HNSW graph has no flat entry-point
array and appends an overlay graph; NSG has a flat entry point and no overlay;
Fusion carries the HNSW overlay but has a wider fused underlay.  RaBitQ's QG path
does not write graph/raw artifacts at all and persists only ``rabitq.data``.

``_FULL_PARAMS`` currently records ``has_scalar_data=False`` for its 33 rows.
Each row is deliberately executed twice here so both generated scalar branches
are covered without turning the public 33-row matrix into 66 generated rows.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
from alayalite import Index
from alayalite._alayalitepy import PyIndexInterface as _NativeIndex
from alayalite.schema import IndexParams

from ._dispatch_matrix_params import _FULL_PARAMS, _IMPLEMENTATION_IDENTITIES

DIM = 16
RABITQ_DIM = 64
N_VECTORS = 256
CAPACITY = 512
MAX_NBRS = 16

ARTIFACT_IDENTITY_BY_IMPLEMENTATION_KEY = {
    "hnsw_segment": "hnsw",
    "nsg_segment": "nsg",
    "fusion_segment": "fusion",
    "qg_segment": "qg",
    "legacy_qg_model": "qg",
}


def _case_id(case) -> str:
    data_type, id_type, quant, index_type, _ = case
    return f"{np.dtype(data_type).name}-{np.dtype(id_type).name}-{quant}-{index_type}"


def _vectors(data_type, quant: str) -> np.ndarray:
    rng = np.random.default_rng(0)
    dim = RABITQ_DIM if quant == "rabitq" else DIM
    if np.issubdtype(data_type, np.floating):
        return rng.standard_normal((N_VECTORS, dim)).astype(data_type)
    info = np.iinfo(data_type)
    return rng.integers(
        max(info.min, -100),
        min(info.max, 100),
        size=(N_VECTORS, dim),
        dtype=data_type,
    )


def _read_native_word(raw: bytes, offset: int, width: int) -> int:
    return int.from_bytes(raw[offset : offset + width], byteorder="little", signed=False)


def _graph_identity(graph_path: Path, id_type, configured_max_nbrs: int) -> str:
    """Classify a legacy Graph::save artifact by its persisted structure."""
    raw = graph_path.read_bytes()
    id_width = np.dtype(id_type).itemsize
    word_width = struct.calcsize("P")

    # Graph::save: int nep, IDType eps[nep], IDType max_nodes,
    # sizeof(IDType) bytes starting at max_nbrs, then SequentialStorage.
    nep = _read_native_word(raw, 0, 4)
    storage_offset = 4 + (nep * id_width) + (2 * id_width)
    max_nbrs_offset = 4 + (nep * id_width) + id_width
    max_nbrs = _read_native_word(raw, max_nbrs_offset, 4)

    # SequentialStorage::save writes five size_t values, its aligned capacity,
    # and an unaligned validity bitmap.  Any remaining bytes are OverlayGraph.
    fields = [_read_native_word(raw, storage_offset + (i * word_width), word_width) for i in range(5)]
    _, aligned_item_size, capacity, _, _ = fields
    flat_graph_size = storage_offset + (5 * word_width) + (aligned_item_size * capacity) + ((capacity + 7) // 8)
    overlay_size = len(raw) - flat_graph_size
    assert overlay_size >= 0, "legacy Graph artifact is shorter than its declared storage layout"

    if nep == 0 and max_nbrs == configured_max_nbrs and overlay_size > 0:
        return "hnsw"
    if nep > 0 and max_nbrs == configured_max_nbrs and overlay_size == 0:
        return "nsg"
    if nep == 0 and max_nbrs > configured_max_nbrs and overlay_size > 0:
        return "fusion"
    return "unknown"


def _persisted_identity(save_dir: Path, id_type, quant: str) -> str:
    graph_files = list(save_dir.glob("*.index"))
    if graph_files:
        assert len(graph_files) == 1
        assert (save_dir / "raw.data").is_file()
        return _graph_identity(graph_files[0], id_type, MAX_NBRS)

    # The legacy RaBitQ fit path builds QG inside RaBitQSpace and serializes it
    # as one quant artifact; no Graph or RawSpace file is emitted.
    if quant == "rabitq" and (save_dir / "rabitq.data").is_file():
        assert sorted(path.name for path in save_dir.iterdir()) == ["rabitq.data"]
        return "qg"
    return "unknown"


def _build_and_observe(case, has_scalar_data: bool, case_dir: Path) -> tuple[str, tuple[str, str, str]]:
    data_type, id_type, quant, requested_index_type, _ = case
    rocksdb_path = str(case_dir / "active" / "rocksdb") if has_scalar_data else ""
    params = IndexParams(
        index_type=requested_index_type,
        data_type=data_type,
        id_type=id_type,
        quantization_type=quant,
        metric="l2",
        capacity=CAPACITY,
        max_nbrs=MAX_NBRS,
        rocksdb_path=rocksdb_path,
        has_scalar_data=has_scalar_data,
    )
    params.fill_none_values()
    vectors = _vectors(data_type, quant)
    index = Index(f"identity_{_case_id(case)}_sd{int(has_scalar_data)}", params)
    try:
        fit_kwargs = {}
        if has_scalar_data:
            fit_kwargs = {
                "item_ids": [f"item-{i}" for i in range(N_VECTORS)],
                "documents": [f"document-{i}" for i in range(N_VECTORS)],
                "metadata_list": [{"row": i} for i in range(N_VECTORS)],
            }
        index.fit(vectors, ef_construction=64, num_threads=1, **fit_kwargs)
        cpp_index = index.get_cpp_index()
        runtime_identity = (
            cpp_index.get_declared_index_type(),
            cpp_index.get_implementation_key(),
            cpp_index.get_engine_factory_key(),
        )
        assert index.search(vectors[0], topk=5, ef_search=64).shape == (5,)
        save_dir = case_dir / "saved"
        index.save(save_dir)
        persisted_identity = _persisted_identity(save_dir, id_type, quant)
        index.close()

        reopened = _NativeIndex(params.to_cpp_params())
        try:
            reopened.load(
                params.index_path(save_dir),
                params.data_path(save_dir),
                params.quant_path(save_dir),
            )
            assert reopened.search(vectors[0], 5, 64).shape == (5,)
        finally:
            reopened.close_db()
        return persisted_identity, runtime_identity
    finally:
        index.close()


@pytest.mark.parametrize(
    "case,declared_identity",
    list(zip(_FULL_PARAMS, _IMPLEMENTATION_IDENTITIES, strict=True)),
    ids=[_case_id(case) for case in _FULL_PARAMS],
)
def test_full_dispatch_matrix_persisted_algorithm_identity(case, declared_identity, tmp_path):
    """Pin generated/runtime keys and the artifact algorithm in both directions."""
    _, _, quant, requested_index_type, _ = case
    observations = [
        _build_and_observe(case, has_scalar_data, tmp_path / f"sd{int(has_scalar_data)}")
        for has_scalar_data in (False, True)
    ]
    observed_artifacts = [artifact for artifact, _ in observations]
    runtime_identities = [identity for _, identity in observations]

    implementation_key, engine_factory_key, declared_artifact_identity = declared_identity
    expected_runtime_identity = (
        requested_index_type,
        implementation_key,
        engine_factory_key,
    )
    assert runtime_identities == [expected_runtime_identity, expected_runtime_identity]
    assert ARTIFACT_IDENTITY_BY_IMPLEMENTATION_KEY[implementation_key] == declared_artifact_identity
    assert engine_factory_key == declared_artifact_identity
    assert observed_artifacts == [declared_artifact_identity, declared_artifact_identity]

    # Z2 pins the legacy-entry compatibility mapping: any declared memory graph
    # plus quant=RaBitQ builds QG. Gate 9's Collection entry will require an
    # explicit QG declaration, while this legacy quirk remains compatible.
    if quant == "rabitq":
        assert declared_artifact_identity == "qg"
        assert observed_artifacts == ["qg", "qg"]
    else:
        assert declared_artifact_identity == requested_index_type
        assert observed_artifacts == [requested_index_type, requested_index_type]
