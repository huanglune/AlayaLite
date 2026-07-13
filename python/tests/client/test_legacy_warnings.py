# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Gate 9-B warning, stacklevel, once-only, and telemetry contracts."""

import inspect
import logging
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from alayalite import (
    LEGACY_API_V_REMOVE,
    AlayaLiteLegacyApiWarning,
    Client,
    Collection,
    DiskCollection,
    Index,
    MetricType,
    vamana,
)
from alayalite._legacy import _legacy_telemetry_snapshot, _reset_legacy_runtime_state_for_tests
from alayalite.laser import Index as LaserIndex
from alayalite.schema import IndexParams


@pytest.fixture(autouse=True)
def _fresh_legacy_runtime_state():
    _reset_legacy_runtime_state_for_tests()


def _assert_one_telemetry(category: str, caller_line: int) -> None:
    records = _legacy_telemetry_snapshot()
    assert len(records) == 1
    assert records[0].schema_version == 1
    assert records[0].event == "legacy_api_used"
    assert records[0].category == category
    assert records[0].caller_file == __file__
    assert records[0].caller_line == caller_line
    assert records[0].removal_version == LEGACY_API_V_REMOVE == "1.2.0"


def _assert_one_warning(captured, warning_category, caller_line: int) -> None:
    assert len(captured) == 1
    assert captured[0].category is warning_category
    assert captured[0].filename == __file__
    assert captured[0].lineno == caller_line
    assert LEGACY_API_V_REMOVE in str(captured[0].message)


def test_index_warns_at_constructor_callsite_once():
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        caller_line = inspect.currentframe().f_lineno + 1
        first = Index("first")
        second = Index("second")

    _assert_one_warning(captured, AlayaLiteLegacyApiWarning, caller_line)
    _assert_one_telemetry("index", caller_line)
    first.close()
    second.close()


def test_disk_collection_warns_at_constructor_callsite_once(tmp_path):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        caller_line = inspect.currentframe().f_lineno + 1
        first = DiskCollection(str(tmp_path / "first"), 2, MetricType.L2, "disk_flat")
        second = DiskCollection(str(tmp_path / "second"), 2, MetricType.L2, "disk_flat")

    _assert_one_warning(captured, AlayaLiteLegacyApiWarning, caller_line)
    _assert_one_telemetry("disk_collection", caller_line)
    assert first.size() == second.size() == 0


def test_laser_index_warns_at_first_method_callsite_once():
    class RawIndexDouble:
        def set_params(self, **_kwargs):
            return None

    index = LaserIndex(
        RawIndexDouble(),
        "unused",
        SimpleNamespace(raw_dim=2),
        loaded=True,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        caller_line = inspect.currentframe().f_lineno + 1
        index.set_params(num_threads=1)
        index.set_params(num_threads=1)

    _assert_one_warning(captured, AlayaLiteLegacyApiWarning, caller_line)
    _assert_one_telemetry("laser", caller_line)


def test_vamana_builder_warns_at_callsite_once(tmp_path):
    missing = tmp_path / "missing.fbin"
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(OSError, match="data_path does not exist"):
            caller_line = inspect.currentframe().f_lineno + 1
            vamana.build_index(str(missing), str(tmp_path / "out.index"), 8)
        with pytest.raises(OSError, match="data_path does not exist"):
            vamana.build_index(str(missing), str(tmp_path / "out-again.index"), 8)

    _assert_one_warning(captured, AlayaLiteLegacyApiWarning, caller_line)
    _assert_one_telemetry("vamana", caller_line)


def test_client_index_surface_warns_at_callsite_once():
    client = Client()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        caller_line = inspect.currentframe().f_lineno + 1
        created = client.create_index("legacy")
        assert client.get_index("legacy") is created

    _assert_one_warning(captured, AlayaLiteLegacyApiWarning, caller_line)
    _assert_one_telemetry("client_index", caller_line)


def test_get_cpp_index_warns_as_deprecation_at_callsite_once(tmp_path):
    collection = Collection("view", IndexParams(rocksdb_path=str(tmp_path / "view" / "rocksdb")))
    collection.add([("a", "A", np.asarray([0.0, 0.0], dtype=np.float32), {})])

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        caller_line = inspect.currentframe().f_lineno + 1
        first = collection.get_cpp_index()
        second = collection.get_cpp_index()

    _assert_one_warning(captured, DeprecationWarning, caller_line)
    _assert_one_telemetry("index", caller_line)
    assert first is second
    assert first.mutable is False


def test_legacy_telemetry_is_observable_as_structured_log(caplog):
    caplog.set_level(logging.INFO, logger="alayalite.legacy")
    client = Client()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AlayaLiteLegacyApiWarning)
        caller_line = inspect.currentframe().f_lineno + 1
        client.list_indices()

    records = [record for record in caplog.records if record.name == "alayalite.legacy"]
    assert len(records) == 1
    record = records[0]
    assert record.legacy_schema_version == 1
    assert record.legacy_event == "legacy_api_used"
    assert record.legacy_category == "client_index"
    assert record.legacy_caller_file == __file__
    assert record.legacy_caller_line == caller_line
    assert record.legacy_removal_version == "1.2.0"
