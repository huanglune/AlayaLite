# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Generate dispatch code from tools/codegen/dispatch.yaml."""

from __future__ import annotations

import shutil
import subprocess
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

    seen: set[tuple[str, str, str, str]] = set()
    required_keys = {"data", "id", "quant", "index"}
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


def _render(config: dict[str, Any]) -> tuple[str, str]:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["expand_search_space"] = lambda combo, scalar: _expand_search_space(config, combo, scalar)
    env.globals["expand_build_space"] = lambda combo, scalar: _expand_build_space(config, combo, scalar)
    env.globals["expand_builder"] = lambda combo, space: _expand_builder(config, combo, space)
    env.globals["scalar_with"] = config["search_scalar_types"]["with_scalar"]
    env.globals["scalar_without"] = config["search_scalar_types"]["without_scalar"]
    env.globals["to_numpy_dtype"] = _to_numpy_dtype
    env.globals["to_python_quantization"] = _to_python_quantization
    env.globals["to_python_index_type"] = _to_python_index_type

    dispatch_template = env.get_template("dispatch_factory.hpp.j2")
    dispatch_rendered = dispatch_template.render(combinations=config["combinations"]).rstrip() + "\n"

    matrix_template = env.get_template("test_matrix_params.py.j2")
    matrix_rendered = matrix_template.render(combinations=config["combinations"]).rstrip() + "\n"

    return dispatch_rendered, matrix_rendered


def _format_cpp_in_place(path: Path) -> None:
    """Run clang-format in-place so generator output matches what pre-commit will produce.

    Without this, dispatch_generated.hpp committed via pre-commit (clang-format applied)
    diverges from raw gen.py output, breaking the CI drift check.
    """
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        raise RuntimeError(
            "clang-format not found on PATH. Install it (apt: clang-format, brew: clang-format) "
            "so generated headers match the style pre-commit enforces."
        )
    subprocess.run([clang_format, "-i", "--style=file", str(path)], check=True)


def main() -> None:
    config = _load_config()
    _validate(config)

    dispatch_rendered, matrix_rendered = _render(config)
    OUTPUT_PATH.write_text(dispatch_rendered, encoding="utf-8")
    MATRIX_PARAMS_OUTPUT_PATH.write_text(matrix_rendered, encoding="utf-8")
    _format_cpp_in_place(OUTPUT_PATH)


if __name__ == "__main__":
    main()
