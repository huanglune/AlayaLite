# SPDX-License-Identifier: AGPL-3.0-only
"""Mechanical Gate 11 guards for retired source surfaces."""

from pathlib import Path

import yaml

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


def test_codegen_schema_has_no_runtime_or_rollback_fields():
    config = yaml.safe_load((ROOT / "tools/codegen/dispatch.yaml").read_text(encoding="utf-8"))
    assert set(config) == {"implementation_registry", "combinations"}
    obsolete = {
        "runtime_template",
        "feature_flag",
        "legacy_implementation_key",
        "legacy_engine_factory_key",
        "rollback",
    }
    registrations = config["implementation_registry"].values()
    assert not any(obsolete.intersection(registration) for registration in registrations)


def test_utils_have_no_blanket_nolint_regions():
    offenders = []
    for header in (ROOT / "include/utils").rglob("*.hpp"):
        text = header.read_text(encoding="utf-8")
        if "NOLINTBEGIN" in text or "NOLINTEND" in text:
            offenders.append(header.relative_to(ROOT).as_posix())
    assert not offenders
