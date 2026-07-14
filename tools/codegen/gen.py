# SPDX-FileCopyrightText: 2025 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

"""Generate the canonical identity test matrix from dispatch.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
CONFIG_PATH = ROOT / "dispatch.yaml"
TEMPLATE_DIR = ROOT / "templates"
MATRIX_PARAMS_OUTPUT_PATH = REPO / "python/tests/client/_dispatch_matrix_params.py"

NUMPY_DTYPE_MAP = {
    "float": "np.float32",
    "int8_t": "np.int8",
    "uint8_t": "np.uint8",
    "uint32_t": "np.uint32",
    "uint64_t": "np.uint64",
}
VALID_DATA_TYPES = frozenset({"float", "int8_t", "uint8_t"})
VALID_ID_TYPES = frozenset({"uint32_t", "uint64_t"})
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
    if set(config) != {"implementation_registry", "combinations"}:
        raise ValueError("dispatch.yaml may only contain implementation_registry and combinations")
    registry = config["implementation_registry"]
    combinations = config["combinations"]
    if not isinstance(registry, dict) or not registry:
        raise ValueError("'implementation_registry' must be a non-empty mapping")
    if not isinstance(combinations, list) or not combinations:
        raise ValueError("'combinations' must be a non-empty list")

    for key, registration in registry.items():
        if not isinstance(key, str) or not key or not isinstance(registration, dict):
            raise ValueError(f"Invalid implementation registration: {key!r}")
        if set(registration) != {"engine_factory_key", "artifact_identity"}:
            raise ValueError(f"Implementation '{key}' has obsolete or missing fields")
        if not all(isinstance(registration[field], str) and registration[field] for field in registration):
            raise ValueError(f"Implementation '{key}' has an invalid identity")

    required = {
        "data",
        "id",
        "quant",
        "index",
        "implementation_key",
        "engine_factory_key",
    }
    seen: set[tuple[str, str, str, str]] = set()
    for combo in combinations:
        if not isinstance(combo, dict) or set(combo) != required:
            raise ValueError(f"Combination has obsolete or missing fields: {combo!r}")
        data = str(combo["data"])
        id_type = str(combo["id"])
        quant = str(combo["quant"])
        index = str(combo["index"])
        implementation_key = str(combo["implementation_key"])
        engine_factory_key = str(combo["engine_factory_key"])
        identity = (data, id_type, quant, index)
        if identity in seen:
            raise ValueError(f"Duplicate combination: {combo}")
        seen.add(identity)
        if data not in VALID_DATA_TYPES:
            raise ValueError(f"Unknown data type '{data}'")
        if id_type not in VALID_ID_TYPES:
            raise ValueError(f"Unknown id type '{id_type}'")
        if quant not in QUANTIZATION_MAP:
            raise ValueError(f"Unknown quantization type '{quant}'")
        if index not in INDEX_TYPE_MAP:
            raise ValueError(f"Unknown index type '{index}'")
        if implementation_key not in registry:
            raise ValueError(f"Unknown implementation key '{implementation_key}'")
        registered_engine = str(registry[implementation_key]["engine_factory_key"])
        if registered_engine != engine_factory_key:
            raise ValueError(
                f"Implementation '{implementation_key}' belongs to '{registered_engine}', not '{engine_factory_key}'"
            )
        if quant == "RABITQ" and (data != "float" or id_type != "uint32_t"):
            raise ValueError(f"RaBitQ only supports float/uint32_t: {combo}")


def _render(config: dict[str, Any]) -> str:
    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.globals["to_numpy_dtype"] = lambda value: NUMPY_DTYPE_MAP[value]
    environment.globals["to_python_quantization"] = lambda value: QUANTIZATION_MAP[value]
    environment.globals["to_python_index_type"] = lambda value: INDEX_TYPE_MAP[value]
    environment.globals["artifact_identity"] = lambda combo: config["implementation_registry"][
        combo["implementation_key"]
    ]["artifact_identity"]
    template = environment.get_template("test_matrix_params.py.j2")
    return template.render(combinations=config["combinations"]).rstrip() + "\n"


def main() -> None:
    config = _load_config()
    _validate(config)
    MATRIX_PARAMS_OUTPUT_PATH.write_text(_render(config), encoding="utf-8")


if __name__ == "__main__":
    main()
