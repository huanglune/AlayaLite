# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Canonical Collection identity coverage over the frozen 33-row legacy matrix."""

import numpy as np
import pytest
from alayalite import Collection, CollectionInvalidArgumentError
from alayalite.schema import IndexParams

from ._dispatch_matrix_params import _FULL_PARAMS, _IMPLEMENTATION_IDENTITIES


def _case_id(case) -> str:
    data_type, id_type, quantization, index_type, _ = case
    return f"{np.dtype(data_type).name}-{np.dtype(id_type).name}-{quantization}-{index_type}"


LEGAL_ROWS = [
    (case, identity) for case, identity in zip(_FULL_PARAMS, _IMPLEMENTATION_IDENTITIES) if case[2] != "rabitq"
]
RABITQ_ROWS = [case for case in _FULL_PARAMS if case[2] == "rabitq"]


@pytest.mark.parametrize("case,identity", LEGAL_ROWS, ids=lambda value: _case_id(value) if len(value) == 5 else None)
def test_canonical_legal_33_matrix_subset_preserves_declared_identity(case, identity, tmp_path):
    data_type, id_type, quantization, index_type, _ = case
    implementation_key, engine_factory_key, declared_artifact_identity = identity
    root = tmp_path / _case_id(case)
    collection = Collection(
        root.name,
        IndexParams(
            index_type=index_type,
            data_type=data_type,
            id_type=id_type,
            quantization_type=quantization,
            metric="l2",
            rocksdb_path=str(root / "rocksdb"),
        ),
    )
    vector = np.arange(4, dtype=data_type)
    collection.add([("row", "document", vector, {"axis": np.dtype(id_type).name})])

    options = collection.get_cpp_index().options()
    assert options["index_type"] == index_type
    assert options["implementation_key"] == implementation_key
    assert options["engine_factory_key"] == engine_factory_key == declared_artifact_identity
    assert options["active_algorithm"] == "flat"
    assert options["quantization_type"] == quantization
    collection.checkpoint()
    collection.close()

    reopened = Collection.load(tmp_path, root.name)
    reopened_options = reopened.get_cpp_index().options()
    assert reopened_options["implementation_key"] == implementation_key
    assert reopened_options["engine_factory_key"] == engine_factory_key
    assert reopened.get_records(["row"])[0]["vector"].dtype == np.dtype(data_type)
    reopened.close()


@pytest.mark.parametrize("case", RABITQ_ROWS, ids=_case_id)
def test_canonical_rabitq_legacy_spellings_require_explicit_qg(case, tmp_path):
    data_type, id_type, quantization, index_type, _ = case
    collection = Collection(
        _case_id(case),
        IndexParams(
            index_type=index_type,
            data_type=data_type,
            id_type=id_type,
            quantization_type=quantization,
            metric="l2",
            rocksdb_path=str(tmp_path / _case_id(case) / "rocksdb"),
        ),
    )

    with pytest.raises(CollectionInvalidArgumentError, match="explicit index_type=qg") as captured:
        collection.add([("row", "document", np.zeros(4, dtype=data_type), {})])

    assert captured.value.status_version == "1"
    assert captured.value.status_code != 0


@pytest.mark.parametrize("id_type", [np.uint32, np.uint64])
def test_canonical_explicit_qg_identity_is_legal(id_type, tmp_path):
    root = tmp_path / np.dtype(id_type).name
    collection = Collection(
        "qg",
        IndexParams(
            index_type="qg",
            data_type=np.float32,
            id_type=id_type,
            quantization_type="rabitq",
            metric="l2",
            rocksdb_path=str(root / "rocksdb"),
        ),
    )
    collection.add([("qg", "document", np.zeros(4, dtype=np.float32), {})])

    assert collection.get_cpp_index().options()["implementation_key"] == "qg_segment"
    assert collection.get_cpp_index().options()["engine_factory_key"] == "qg"
