# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Byte-stable QG codec checks across the three pinned legacy declarations."""

import numpy as np
from alayalite._alayalitepy import PyIndexInterface as _NativeIndex
from alayalite.schema import IndexParams

from .test_index_algorithm_identity import CAPACITY, MAX_NBRS, _vectors


def test_rabitq_declared_rows_round_trip_one_qg_artifact_byte_for_byte(tmp_path):
    """The pinned declaration quirk is independent of the legacy index_type spelling.

    QG construction itself uses legacy random sources, so independently built
    artifacts are not byte deterministic. This checks the stronger available
    codec invariant: one built QG artifact opened and saved through each of the
    three declared rows is byte-identical and searches identically.
    """
    vectors = _vectors(np.float32, "rabitq")

    def params_for(index_type: str) -> IndexParams:
        params = IndexParams(
            index_type=index_type,
            data_type=np.float32,
            id_type=np.uint32,
            quantization_type="rabitq",
            metric="l2",
            capacity=CAPACITY,
            max_nbrs=MAX_NBRS,
        )
        params.fill_none_values()
        return params

    source_path = tmp_path / "source-rabitq.data"
    source = _NativeIndex(params_for("hnsw").to_cpp_params())
    try:
        source.fit(vectors, 64, 1)
        source.save("", "", str(source_path))
        expected_ids = source.search(vectors[0], 8, 64)
    finally:
        source.close_db()

    round_trips = []
    for declared in ("hnsw", "nsg", "fusion"):
        native = _NativeIndex(params_for(declared).to_cpp_params())
        output = tmp_path / f"{declared}-rabitq.data"
        try:
            assert (
                native.get_declared_index_type(),
                native.get_implementation_key(),
                native.get_engine_factory_key(),
            ) == (declared, "qg_segment", "qg")
            native.load("", "", str(source_path))
            np.testing.assert_array_equal(native.search(vectors[0], 8, 64), expected_ids)
            native.save("", "", str(output))
        finally:
            native.close_db()
        round_trips.append(output.read_bytes())

    assert round_trips == [source_path.read_bytes()] * 3
