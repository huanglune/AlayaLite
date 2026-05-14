# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Coverage tests for the C++ dispatch tree in `python/include/dispatch.hpp`.

The dispatch macros translate runtime IndexParams into compile-time template
specializations of ``PyIndex<GraphBuilderType, SearchSpaceType>``. These tests
exist as a safety net for refactors of that translation: every leaf of the
dispatch tree must remain reachable from the SDK.

The tests assert behavioural smoke (operations do not raise) rather than
numerical correctness; recall/precision are validated in dedicated tests
elsewhere.

Two layers:

* **Smoke (default)** -- each dispatch axis (DataType / IDType / Quantization /
  IndexType / has_scalar_data) is varied independently. Roughly 17 cases.
* **Extended (opt-in via ``-m extended``)** -- full Cartesian product of legal
  combinations. Verifies that every reachable PyIndex specialization can be
  constructed and fitted.
"""

# pylint: disable=redefined-outer-name

import itertools
import os
import tempfile

import numpy as np
import pytest
from alayalite import Index
from alayalite.schema import IndexParams

# ---------------------------------------------------------------------------
# Dispatch axes
# ---------------------------------------------------------------------------

DATA_TYPES = [np.float32, np.int8, np.uint8, np.int32, np.uint32, np.float64]
ID_TYPES = [np.uint32, np.uint64]
QUANT_TYPES = ["none", "sq8", "sq4", "rabitq"]
INDEX_TYPES = ["hnsw", "nsg", "fusion"]

# RaBitQ's FhtKacRotator requires floor_log2(dim) in [6, 11], i.e. dim >= 64.
DIM = 64
N_VECTORS = 256


def _is_valid_combo(data_type, id_type, quant) -> bool:
    """Filter out combinations rejected at construct/fit time.

    Constraints come from the C++ layer:

    * RaBitQ space only supports float32 vectors
      (``DISPATCH_BUILD_SPACE_TYPE`` / ``DISPATCH_SEARCH_SPACE_TYPE`` in
      ``python/include/dispatch.hpp``).
    * RaBitQ space requires a 32-bit ID type
      (``include/space/rabitq_space.hpp::RaBitQSpace::fit``).
    """
    if quant == "rabitq" and data_type is not np.float32:
        return False
    if quant == "rabitq" and id_type is not np.uint32:
        return False
    return True


def _random_vectors(rng, n, dim, dtype):
    if np.issubdtype(dtype, np.floating):
        return rng.standard_normal((n, dim)).astype(dtype)
    info = np.iinfo(dtype)
    low = max(info.min, -100)
    high = min(info.max, 100)
    return rng.integers(low, high, size=(n, dim), dtype=dtype)


def _build_index(name, *, data_type, id_type, quant, index_type, has_scalar_data, rocksdb_path=""):
    params = IndexParams(
        index_type=index_type,
        data_type=data_type,
        id_type=id_type,
        quantization_type=quant,
        metric="l2",
        capacity=2048,
        max_nbrs=16,
        rocksdb_path=rocksdb_path,
        has_scalar_data=has_scalar_data,
    )
    params.fill_none_values()
    return Index(name, params)


# ---------------------------------------------------------------------------
# Layer 1: Smoke tests (default-on, one axis at a time)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("data_type", DATA_TYPES)
def test_axis_data_type(data_type):
    """Each VectorDType must be constructible, fittable, and support insert/remove."""
    rng = np.random.default_rng(0)
    index = _build_index(
        f"axis_dt_{np.dtype(data_type).name}",
        data_type=data_type,
        id_type=np.uint32,
        quant="none",
        index_type="hnsw",
        has_scalar_data=False,
    )
    vectors = _random_vectors(rng, N_VECTORS, DIM, data_type)
    index.fit(vectors)
    assert index.get_dim() == DIM

    # Insert one more vector; should return a valid ID.
    extra = _random_vectors(rng, 1, DIM, data_type)[0]
    new_id = index.insert(extra)
    assert new_id is not None

    # Remove the inserted vector.
    index.remove(new_id)


@pytest.mark.parametrize("id_type", ID_TYPES)
def test_axis_id_type(id_type):
    rng = np.random.default_rng(0)
    index = _build_index(
        f"axis_id_{np.dtype(id_type).name}",
        data_type=np.float32,
        id_type=id_type,
        quant="none",
        index_type="hnsw",
        has_scalar_data=False,
    )
    vectors = _random_vectors(rng, N_VECTORS, DIM, np.float32)
    index.fit(vectors)
    result = index.search(vectors[0], topk=5)
    assert len(result) == 5


@pytest.mark.parametrize("quant", QUANT_TYPES)
def test_axis_quantization(quant):
    rng = np.random.default_rng(0)
    index = _build_index(
        f"axis_q_{quant}",
        data_type=np.float32,
        id_type=np.uint32,
        quant=quant,
        index_type="hnsw",
        has_scalar_data=False,
    )
    vectors = _random_vectors(rng, N_VECTORS, DIM, np.float32)
    index.fit(vectors)
    result = index.search(vectors[0], topk=5)
    assert len(result) == 5

    # Save + load round-trip exercises DISPATCH_AND_CAST for save/load.
    with tempfile.TemporaryDirectory() as tmp:
        index.save(tmp)
        # Load currently requires an existing Index with matching params;
        # the dispatch-level coverage comes from save() alone.


@pytest.mark.parametrize("index_type", INDEX_TYPES)
def test_axis_index_type(index_type):
    rng = np.random.default_rng(0)
    index = _build_index(
        f"axis_ix_{index_type}",
        data_type=np.float32,
        id_type=np.uint32,
        quant="none",
        index_type=index_type,
        has_scalar_data=False,
    )
    vectors = _random_vectors(rng, N_VECTORS, DIM, np.float32)
    index.fit(vectors)
    result = index.search(vectors[0], topk=5)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Layer 2: Full Cartesian matrix (opt-in via @pytest.mark.extended)
# ---------------------------------------------------------------------------


_FULL_PARAMS = [
    (dt, idt, q, ixt, sd)
    for dt, idt, q, ixt, sd in itertools.product(DATA_TYPES, ID_TYPES, QUANT_TYPES, INDEX_TYPES, [False])
    if _is_valid_combo(dt, idt, q)
]


def _full_param_id(case) -> str:
    dt, idt, q, ixt, sd = case
    return f"{np.dtype(dt).name}-{np.dtype(idt).name}-{q}-{ixt}-sd{int(sd)}"


@pytest.mark.extended
@pytest.mark.parametrize("case", _FULL_PARAMS, ids=_full_param_id)
def test_full_matrix_construct_and_fit(case):
    """Every legal (data_type, id_type, quant, index_type) leaf must construct + fit."""
    data_type, id_type, quant, index_type, has_scalar_data = case
    rng = np.random.default_rng(0)
    index = _build_index(
        f"full_{_full_param_id(case)}",
        data_type=data_type,
        id_type=id_type,
        quant=quant,
        index_type=index_type,
        has_scalar_data=has_scalar_data,
    )
    vectors = _random_vectors(rng, N_VECTORS, DIM, data_type)
    index.fit(vectors)
    assert index.get_dim() == DIM


# ---------------------------------------------------------------------------
# has_scalar_data branch (extended; needs RocksDB temp dir)
# ---------------------------------------------------------------------------


@pytest.mark.extended
def test_axis_has_scalar_data(tmp_path):
    rocksdb_dir = tmp_path / "rocksdb"
    original = os.environ.get("ALAYALITE_ROCKSDB_DIR")
    os.environ["ALAYALITE_ROCKSDB_DIR"] = str(rocksdb_dir)
    try:
        rng = np.random.default_rng(0)
        index = _build_index(
            "axis_sd_true",
            data_type=np.float32,
            id_type=np.uint32,
            quant="none",
            index_type="hnsw",
            has_scalar_data=True,
            rocksdb_path=str(rocksdb_dir / "db"),
        )
        vectors = _random_vectors(rng, N_VECTORS, DIM, np.float32)
        item_ids = [f"item_{i}" for i in range(N_VECTORS)]
        documents = [f"doc {i}" for i in range(N_VECTORS)]
        metadata = [{"i": i} for i in range(N_VECTORS)]
        index.fit(vectors, item_ids=item_ids, documents=documents, metadata_list=metadata)
        assert index.get_dim() == DIM
    finally:
        if original is None:
            os.environ.pop("ALAYALITE_ROCKSDB_DIR", None)
        else:
            os.environ["ALAYALITE_ROCKSDB_DIR"] = original
