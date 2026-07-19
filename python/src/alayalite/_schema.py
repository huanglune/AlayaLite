# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Private catalog-schema codec for the v2 Python core."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from .config import CollectionConfig, FlatIndexConfig, IndexConfig, Metric, QGIndexConfig, VectorDType

_SCHEMA_NAME = "schema.json"
_STORAGE_FORMAT = "canonical_collection_v1"
_PUBLIC_VERSION = "1.1.0"


def write_collection_schema(root: Path, config: CollectionConfig) -> None:
    """Atomically persist a catalog-visible v2 schema."""
    schema_path = root / _SCHEMA_NAME
    temporary_path = root / f"{_SCHEMA_NAME}.tmp"
    payload: dict[str, object] = {
        "type": "collection",
        "format": _STORAGE_FORMAT,
        "public_version": _PUBLIC_VERSION,
        "config": _encode_config(config),
    }
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary_path, schema_path)


def is_catalog_collection(root: Path) -> bool:
    """Return whether a child directory has a supported collection header."""
    if root.is_symlink() or not root.is_dir():
        return False
    try:
        payload = _read_payload(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return payload.get("type") == "collection" and payload.get("format") == _STORAGE_FORMAT


def load_collection_schema(root: Path) -> tuple[CollectionConfig | None, str | None]:
    """Load a v2 config or a naturally readable pre-v2 schema declaration."""
    try:
        payload = _read_payload(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, None
    if payload.get("type") not in {"collection", "index"}:
        return None, None
    config_payload = payload.get("config")
    if isinstance(config_payload, Mapping):
        try:
            return _decode_config(config_payload), None
        except (KeyError, TypeError, ValueError):
            return None, None
    legacy_payload = payload.get("index")
    if not isinstance(legacy_payload, Mapping):
        return None, None
    try:
        return _decode_legacy_config(legacy_payload), _legacy_quantization(legacy_payload)
    except (KeyError, TypeError, ValueError):
        return None, None


def storage_format() -> str:
    """Return the canonical format identifier used in collection info."""
    return _STORAGE_FORMAT


def _read_payload(root: Path) -> dict[str, object]:
    """Read and type-check one JSON object."""
    with (root / _SCHEMA_NAME).open(encoding="utf-8") as source:
        value: object = json.load(source)
    if not isinstance(value, dict):
        raise TypeError("collection schema must be a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise TypeError("collection schema keys must be strings")
    return cast(dict[str, object], value)


def _encode_config(config: CollectionConfig) -> dict[str, object]:
    """Encode the immutable public configuration."""
    if isinstance(config.index, FlatIndexConfig):
        index: dict[str, object] = {"kind": "flat"}
    else:
        index = {
            "kind": "qg",
            "max_neighbors": config.index.max_neighbors,
            "construction_effort": config.index.construction_effort,
            "build_threads": config.index.build_threads,
        }
    return {
        "dimension": config.dimension,
        "dtype": config.dtype,
        "metric": config.metric,
        "index": index,
        "auto_seal_rows": config.auto_seal_rows,
    }


def _decode_config(payload: Mapping[object, object]) -> CollectionConfig:
    """Decode the current v2 schema shape."""
    index_payload = payload["index"]
    if not isinstance(index_payload, Mapping):
        raise TypeError("collection index config must be an object")
    kind = index_payload.get("kind")
    if kind == "flat":
        index: IndexConfig = FlatIndexConfig()
    elif kind == "qg":
        index = QGIndexConfig(
            max_neighbors=_required_int(index_payload, "max_neighbors"),
            construction_effort=_required_int(index_payload, "construction_effort"),
            build_threads=_optional_int(index_payload.get("build_threads")),
        )
    else:
        raise ValueError("unknown collection index kind")
    return CollectionConfig(
        dimension=_required_int(payload, "dimension"),
        dtype=_vector_dtype(payload, "dtype"),
        metric=_metric(payload, "metric"),
        index=index,
        auto_seal_rows=_optional_int(payload.get("auto_seal_rows")),
    )


def _decode_legacy_config(payload: Mapping[object, object]) -> CollectionConfig:
    """Decode the naturally compatible 1.1 discovery schema."""
    kind = _required_str(payload, "index_type")
    if kind == "qg":
        index: IndexConfig = QGIndexConfig(
            max_neighbors=_int_with_default(payload.get("max_nbrs"), 32),
            construction_effort=_int_with_default(payload.get("ef_construction"), 400),
            build_threads=_optional_int(payload.get("build_threads")),
        )
    elif kind == "flat":
        index = FlatIndexConfig()
    else:
        raise ValueError("unknown legacy index kind")
    dimension = payload.get("dimension")
    if dimension is None:
        # Legacy discovery schemas did not persist dimension. Native options
        # remain authoritative, so the caller will derive the complete config.
        raise KeyError("dimension")
    return CollectionConfig(
        dimension=_required_int(payload, "dimension"),
        dtype=_vector_dtype(payload, "data_type"),
        metric=_metric(payload, "metric"),
        index=index,
    )


def _legacy_quantization(payload: Mapping[object, object]) -> str | None:
    """Return an ignored old quantization declaration for diagnostics."""
    value = payload.get("quantization_type")
    return value if isinstance(value, str) and value not in {"none", "rabitq"} else None


def _required_int(payload: Mapping[object, object], key: str) -> int:
    """Read one required non-boolean integer."""
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an int")
    return value


def _required_str(payload: Mapping[object, object], key: str) -> str:
    """Read one required string."""
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _vector_dtype(payload: Mapping[object, object], key: str) -> VectorDType:
    """Read and narrow one canonical vector dtype."""
    value = _required_str(payload, key)
    if value not in {"float32", "int8", "uint8"}:
        raise ValueError(f"{key} is not a supported vector dtype")
    return cast(VectorDType, value)


def _metric(payload: Mapping[object, object], key: str) -> Metric:
    """Read and narrow one canonical metric."""
    value = _required_str(payload, key)
    if value not in {"l2", "ip", "cosine"}:
        raise ValueError(f"{key} is not a supported metric")
    return cast(Metric, value)


def _optional_int(value: object) -> int | None:
    """Read a nullable non-boolean integer."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("optional integer field is invalid")
    return value


def _int_with_default(value: object, default: int) -> int:
    """Read an optional legacy integer with a concrete default."""
    resolved = _optional_int(value)
    return default if resolved is None else resolved
