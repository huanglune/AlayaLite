# SPDX-License-Identifier: AGPL-3.0-only
"""Mechanical Gate 11 guards for retired source surfaces."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

REMOVED_PATHS = (
    "include/core/compat.hpp",
    "include/index/compat.hpp",
    "include/index/disk/disk_collection.hpp",
    "include/index/index_type.hpp",
    "include/utils/quantization_type.hpp",
    "include/utils/rabitq_utils/search_utils/epoch_visited.hpp",
    "python/include/base_py_index.hpp",
    "python/include/disk_collection.hpp",
    "python/include/dispatch_generated.hpp",
    "python/include/index.hpp",
    "python/src/index_factory.cpp",
    "python/src/alayalite/laser/_bindings.hpp",
    "python/src/alayalite/vamana/_bindings.hpp",
    "tools/codegen/templates/dispatch_factory.hpp.j2",
    # HNSW retirement wave: the whole hnsw-keyed dispatch codegen chain.
    "tools/codegen/dispatch.yaml",
    "tools/codegen/gen.py",
    "tools/codegen/templates/test_matrix_params.py.j2",
    "python/tests/client/_dispatch_matrix_params.py",
)


def test_retired_source_paths_stay_absent():
    assert not [path for path in REMOVED_PATHS if (ROOT / path).exists()]


def test_native_module_has_no_legacy_api_registration():
    binding = (ROOT / "python/src/pybind.cpp").read_text(encoding="utf-8")
    forbidden = (
        "PyIndexInterface",
        "_CollectionReadView",
        "register_disk_collection",
        "register_vamana_module",
        "register_laser_module",
    )
    assert not [name for name in forbidden if name in binding]


def test_utils_have_no_blanket_nolint_regions():
    offenders = []
    for header in (ROOT / "include/utils").rglob("*.hpp"):
        text = header.read_text(encoding="utf-8")
        if "NOLINTBEGIN" in text or "NOLINTEND" in text:
            offenders.append(header.relative_to(ROOT).as_posix())
    assert not offenders
