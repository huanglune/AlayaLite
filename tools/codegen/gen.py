# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Generate dispatch code from tools/codegen/dispatch.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
CONFIG_PATH = ROOT / "dispatch.yaml"
TEMPLATE_DIR = ROOT / "templates"
OUTPUT_PATH = REPO / "python/include/dispatch_generated.hpp"
MATRIX_PARAMS_OUTPUT_PATH = REPO / "python/tests/client/_dispatch_matrix_params.py"

NUMPY_DTYPE_MAP = {
    "float": "np.float32",
    "double": "np.float64",
    "int8_t": "np.int8",
    "uint8_t": "np.uint8",
    "int32_t": "np.int32",
    "uint32_t": "np.uint32",
    "uint64_t": "np.uint64",
}

VALID_DATA_TYPES = frozenset(NUMPY_DTYPE_MAP.keys())
VALID_ID_TYPES = frozenset({"uint32_t", "uint64_t"})
VALID_ENGINE_FEATURES = frozenset(
    {
        "none",
        "knng_segment",
        "nsg_segment",
        "fusion_segment",
        "qg_segment",
        "vamana_memory_segment",
    }
)
VALID_ROLLBACK_MODES = frozenset({"feature_flag", "source_revert"})

QUANTIZATION_MAP = {
    "NONE": "none",
    "SQ8": "sq8",
    "SQ4": "sq4",
    "RABITQ": "rabitq",
}

INDEX_TYPE_MAP = {
    "HNSW": "hnsw",
    "NSG": "nsg",
    "FUSION": "fusion",
}


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config root in {CONFIG_PATH}: expected mapping")
    return config


def _validate(config: dict[str, Any]) -> None:
    combinations = config.get("combinations")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("'combinations' must be a non-empty list")

    implementation_registry = config.get("implementation_registry")
    if not isinstance(implementation_registry, dict) or not implementation_registry:
        raise ValueError("'implementation_registry' must be a non-empty mapping")
    engine_factories = config.get("engine_factories")
    if not isinstance(engine_factories, dict) or not engine_factories:
        raise ValueError("'engine_factories' must be a non-empty mapping")

    for implementation_key, registration in implementation_registry.items():
        if not isinstance(implementation_key, str) or not implementation_key:
            raise ValueError(f"Invalid implementation key: {implementation_key!r}")
        if not isinstance(registration, dict):
            raise ValueError(f"Implementation registration must be a mapping: {implementation_key}")
        required = {"engine_factory_key", "artifact_identity", "runtime_template"}
        missing = required - set(registration)
        if missing:
            raise ValueError(f"Implementation '{implementation_key}' missing keys {sorted(missing)}")
        if not isinstance(registration["engine_factory_key"], str) or not registration["engine_factory_key"]:
            raise ValueError(f"Implementation '{implementation_key}' has an invalid engine key")
        if not isinstance(registration["artifact_identity"], str) or not registration["artifact_identity"]:
            raise ValueError(f"Implementation '{implementation_key}' has an invalid artifact identity")
        if not isinstance(registration["runtime_template"], str) or not registration["runtime_template"]:
            raise ValueError(f"Implementation '{implementation_key}' has an invalid runtime template")

    for engine_factory_key, registration in engine_factories.items():
        if not isinstance(engine_factory_key, str) or not engine_factory_key:
            raise ValueError(f"Invalid engine factory key: {engine_factory_key!r}")
        if not isinstance(registration, dict):
            raise ValueError(f"Engine factory registration must be a mapping: {engine_factory_key}")
        required = {
            "feature_flag",
            "legacy_implementation_key",
            "legacy_engine_factory_key",
            "rollback",
        }
        missing = required - set(registration)
        if missing:
            raise ValueError(f"Engine factory '{engine_factory_key}' missing keys {sorted(missing)}")

        feature_flag = str(registration["feature_flag"])
        rollback = str(registration["rollback"])
        legacy_implementation_key = str(registration["legacy_implementation_key"])
        legacy_engine_factory_key = str(registration["legacy_engine_factory_key"])
        if feature_flag not in VALID_ENGINE_FEATURES:
            raise ValueError(f"Engine factory '{engine_factory_key}' has unknown feature flag '{feature_flag}'")
        if rollback not in VALID_ROLLBACK_MODES:
            raise ValueError(f"Engine factory '{engine_factory_key}' has unknown rollback mode '{rollback}'")
        if rollback == "source_revert" and feature_flag != "none":
            raise ValueError(f"Source-revert engine '{engine_factory_key}' cannot have a runtime feature flag")
        if rollback == "feature_flag" and feature_flag == "none":
            raise ValueError(f"Runtime-rollback engine '{engine_factory_key}' must have an independent feature flag")
        if legacy_implementation_key not in implementation_registry:
            raise ValueError(
                f"Engine factory '{engine_factory_key}' names unknown legacy implementation "
                f"'{legacy_implementation_key}'"
            )
        if legacy_engine_factory_key not in engine_factories:
            raise ValueError(
                f"Engine factory '{engine_factory_key}' names unknown legacy engine '{legacy_engine_factory_key}'"
            )
        registered_legacy_engine = str(implementation_registry[legacy_implementation_key]["engine_factory_key"])
        if registered_legacy_engine != legacy_engine_factory_key:
            raise ValueError(
                f"Engine factory '{engine_factory_key}' legacy pair disagrees: "
                f"implementation '{legacy_implementation_key}' belongs to "
                f"'{registered_legacy_engine}', not '{legacy_engine_factory_key}'"
            )

    seen: set[tuple[str, str, str, str]] = set()
    required_keys = {
        "data",
        "id",
        "quant",
        "index",
        "implementation_key",
        "engine_factory_key",
    }
    valid_quants = set(config.get("search_spaces", {}).keys())
    valid_indexes = set(config.get("builders", {}).keys())

    for combo in combinations:
        if not isinstance(combo, dict):
            raise ValueError(f"Combination must be a mapping: {combo!r}")
        missing = required_keys - set(combo.keys())
        if missing:
            raise ValueError(f"Combination missing keys {sorted(missing)}: {combo}")

        data = str(combo["data"])
        id_type = str(combo["id"])
        quant = str(combo["quant"])
        index = str(combo["index"])
        implementation_key = str(combo["implementation_key"])
        engine_factory_key = str(combo["engine_factory_key"])

        key = (data, id_type, quant, index)
        if key in seen:
            raise ValueError(f"Duplicate combination: {combo}")
        seen.add(key)

        if data not in VALID_DATA_TYPES:
            raise ValueError(f"Unknown data type '{data}' in {combo}; valid: {sorted(VALID_DATA_TYPES)}")
        if id_type not in VALID_ID_TYPES:
            raise ValueError(f"Unknown id type '{id_type}' in {combo}; valid: {sorted(VALID_ID_TYPES)}")
        if quant not in valid_quants:
            raise ValueError(f"Unknown quantization type '{quant}' in {combo}")
        if quant not in config.get("build_spaces", {}):
            raise ValueError(f"Missing build_space template for quantization type '{quant}'")
        if index not in valid_indexes:
            raise ValueError(f"Unknown index type '{index}' in {combo}")
        if implementation_key not in implementation_registry:
            raise ValueError(f"Unknown implementation key '{implementation_key}' in {combo}")
        if engine_factory_key not in engine_factories:
            raise ValueError(f"Unknown engine factory key '{engine_factory_key}' in {combo}")
        registered_engine = str(implementation_registry[implementation_key]["engine_factory_key"])
        if registered_engine != engine_factory_key:
            raise ValueError(
                f"Dispatch row implementation/engine mismatch: '{implementation_key}' belongs to "
                f"'{registered_engine}', not '{engine_factory_key}' in {combo}"
            )

        if quant == "RABITQ":
            if data != "float":
                raise ValueError(f"RaBitQ only supports float data: {combo}")
            if id_type != "uint32_t":
                raise ValueError(f"RaBitQ only supports uint32_t id: {combo}")


def _expand_search_space(config: dict[str, Any], combo: dict[str, str], scalar: str) -> str:
    template = config["search_spaces"][combo["quant"]]
    return template.format(
        data=combo["data"],
        id=combo["id"],
        distance=config["distance_type"],
        scalar=scalar,
        build_scalar=config["build_scalar_type"],
    )


def _expand_build_space(config: dict[str, Any], combo: dict[str, str], scalar: str) -> str:
    template = config["build_spaces"][combo["quant"]]
    return template.format(
        data=combo["data"],
        id=combo["id"],
        distance=config["distance_type"],
        scalar=scalar,
        build_scalar=config["build_scalar_type"],
    )


def _expand_builder(config: dict[str, Any], combo: dict[str, str], space: str) -> str:
    template = config["builders"][combo["index"]]
    return template.format(space=space)


def _expand_runtime(
    config: dict[str, Any],
    combo: dict[str, str],
    implementation_key: str,
    search_space: str,
    build_space: str,
) -> str:
    registration = config["implementation_registry"][implementation_key]
    declared_runtime = _expand_builder(config, combo, build_space)
    return str(registration["runtime_template"]).format(
        search_space=search_space,
        build_space=build_space,
        declared_runtime=declared_runtime,
    )


def _to_numpy_dtype(cpp_dtype: str) -> str:
    try:
        return NUMPY_DTYPE_MAP[cpp_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype mapping for generated tests: {cpp_dtype}") from exc


def _to_python_quantization(quant: str) -> str:
    try:
        return QUANTIZATION_MAP[quant]
    except KeyError as exc:
        raise ValueError(f"Unsupported quantization mapping for generated tests: {quant}") from exc


def _to_python_index_type(index: str) -> str:
    try:
        return INDEX_TYPE_MAP[index]
    except KeyError as exc:
        raise ValueError(f"Unsupported index mapping for generated tests: {index}") from exc


def _implementation_registration(config: dict[str, Any], combo: dict[str, str]) -> dict[str, str]:
    return config["implementation_registry"][combo["implementation_key"]]


def _engine_registration(config: dict[str, Any], combo: dict[str, str]) -> dict[str, str]:
    return config["engine_factories"][combo["engine_factory_key"]]


def _render(config: dict[str, Any]) -> tuple[str, str]:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["expand_search_space"] = lambda combo, scalar: _expand_search_space(config, combo, scalar)
    env.globals["expand_build_space"] = lambda combo, scalar: _expand_build_space(config, combo, scalar)
    env.globals["expand_current_runtime"] = lambda combo, search_space, build_space: _expand_runtime(
        config, combo, combo["implementation_key"], search_space, build_space
    )
    env.globals["expand_legacy_runtime"] = lambda combo, search_space, build_space: _expand_runtime(
        config,
        combo,
        _engine_registration(config, combo)["legacy_implementation_key"],
        search_space,
        build_space,
    )
    env.globals["scalar_with"] = config["search_scalar_types"]["with_scalar"]
    env.globals["scalar_without"] = config["search_scalar_types"]["without_scalar"]
    env.globals["to_numpy_dtype"] = _to_numpy_dtype
    env.globals["to_python_quantization"] = _to_python_quantization
    env.globals["to_python_index_type"] = _to_python_index_type
    env.globals["artifact_identity"] = lambda combo: _implementation_registration(config, combo)["artifact_identity"]
    env.globals["engine_feature"] = lambda combo: _engine_registration(config, combo)["feature_flag"]
    env.globals["legacy_implementation_key"] = lambda combo: _engine_registration(config, combo)[
        "legacy_implementation_key"
    ]
    env.globals["legacy_engine_factory_key"] = lambda combo: _engine_registration(config, combo)[
        "legacy_engine_factory_key"
    ]
    env.globals["source_revert_only"] = lambda combo: (
        _engine_registration(config, combo)["rollback"] == "source_revert"
    )

    dispatch_template = env.get_template("dispatch_factory.hpp.j2")
    dispatch_rendered = dispatch_template.render(combinations=config["combinations"]).rstrip() + "\n"

    matrix_template = env.get_template("test_matrix_params.py.j2")
    matrix_rendered = matrix_template.render(combinations=config["combinations"]).rstrip() + "\n"

    return dispatch_rendered, matrix_rendered


def main() -> None:
    config = _load_config()
    _validate(config)

    dispatch_rendered, matrix_rendered = _render(config)
    OUTPUT_PATH.write_text(dispatch_rendered, encoding="utf-8")
    MATRIX_PARAMS_OUTPUT_PATH.write_text(matrix_rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
