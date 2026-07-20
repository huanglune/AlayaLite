# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Runtime validation shared by the v2 database and collection wrappers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from .models import BatchMode, Filter, MetadataScalar, WriteDurability

_FILTER_OPERATORS = frozenset({"$eq", "$gt", "$ge", "$lt", "$le", "$in"})
_LOGICAL_OPERATORS = frozenset({"$and", "$or"})
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def positive_int(value: object, name: str) -> int:
    """Validate a positive Python integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def strict_ids(ids: Sequence[str], *, allow_empty: bool) -> list[str]:
    """Materialize a finite sequence of strict string logical IDs."""
    if isinstance(ids, str | bytes) or not isinstance(ids, Sequence):
        raise TypeError("ids must be a finite sequence of strings")
    result = list(ids)
    if not allow_empty and not result:
        raise ValueError("ids must not be empty")
    for item_id in result:
        if not isinstance(item_id, str):
            raise TypeError("every id must be a str")
        try:
            item_id.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("ids must be valid UTF-8 strings") from error
    return result


def document_column(documents: Sequence[str] | None, rows: int) -> list[str]:
    """Validate and materialize the document column."""
    if documents is None:
        return [""] * rows
    if isinstance(documents, str | bytes) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a finite sequence of strings or None")
    result = list(documents)
    if len(result) != rows:
        raise ValueError("documents and ids must have equal lengths")
    if any(not isinstance(document, str) for document in result):
        raise TypeError("every document must be a str")
    return result


def metadata_column(
    metadata: Sequence[Mapping[str, MetadataScalar] | None] | None,
    rows: int,
) -> list[dict[str, MetadataScalar] | None]:
    """Validate and materialize the flat metadata column."""
    if metadata is None:
        return [None] * rows
    if isinstance(metadata, str | bytes) or not isinstance(metadata, Sequence):
        raise TypeError("metadata must be a finite sequence of mappings or None")
    materialized = list(metadata)
    if len(materialized) != rows:
        raise ValueError("metadata and ids must have equal lengths")
    result: list[dict[str, MetadataScalar] | None] = []
    for row in materialized:
        if row is None:
            result.append(None)
            continue
        if not isinstance(row, Mapping):
            raise TypeError("every metadata row must be a mapping or None")
        normalized: dict[str, MetadataScalar] = {}
        for key, value in row.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
            normalized[key] = metadata_scalar(value, "metadata value")
        result.append(normalized)
    return result


def metadata_scalar(value: object, name: str) -> MetadataScalar:
    """Validate one native flat-metadata scalar."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < _INT64_MIN or value > _INT64_MAX:
            raise ValueError(f"{name} must fit in signed 64 bits")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be bool, int, float, or str")


def filter_expression(where: Filter | None) -> dict[str, object] | None:
    """Validate and normalize the fixed native filter DSL without evaluating it."""
    if where is None:
        return None
    if not isinstance(where, Mapping):
        raise TypeError("where must be a mapping or None")
    return _filter_mapping(where)


def _filter_mapping(expression: Mapping[str, object]) -> dict[str, object]:
    """Normalize one recursive filter mapping."""
    result: dict[str, object] = {}
    for key, value in expression.items():
        if not isinstance(key, str):
            raise TypeError("filter keys must be strings")
        if key in _LOGICAL_OPERATORS:
            if isinstance(value, str | bytes) or not isinstance(value, Sequence):
                raise TypeError(f"{key} expects a finite sequence of filters")
            children = list(value)
            if not children:
                raise ValueError(f"{key} expects at least one filter")
            normalized_children: list[dict[str, object]] = []
            for child in children:
                if not isinstance(child, Mapping):
                    raise TypeError(f"{key} children must be filter mappings")
                normalized_children.append(_filter_mapping(child))
            result[key] = normalized_children
            continue
        if key.startswith("$"):
            raise ValueError(f"Unsupported operator: {key}")
        if isinstance(value, Mapping):
            result[key] = _filter_field(value)
        else:
            result[key] = metadata_scalar(value, "filter operand")
    return result


def _filter_field(expression: Mapping[str, object]) -> dict[str, object]:
    """Normalize comparison operators for one metadata field."""
    if not expression:
        raise ValueError("field filter expressions must not be empty")
    result: dict[str, object] = {}
    for operator, operand in expression.items():
        if not isinstance(operator, str):
            raise TypeError("filter operators must be strings")
        if operator not in _FILTER_OPERATORS:
            raise ValueError(f"Unsupported operator: {operator}")
        if operator == "$in":
            if isinstance(operand, str | bytes) or not isinstance(operand, Sequence):
                raise TypeError("$in expects a finite sequence of metadata scalars")
            choices = list(operand)
            if not choices:
                raise ValueError("$in expects at least one operand")
            result[operator] = [metadata_scalar(choice, "$in operand") for choice in choices]
        else:
            result[operator] = metadata_scalar(operand, f"{operator} operand")
    return result


def native_batch_mode(mode: BatchMode) -> str:
    """Translate the two public batch-mode spellings."""
    if mode == "atomic":
        return "all_or_nothing"
    if mode == "partial":
        return "per_row_independent"
    raise ValueError("mode must be atomic or partial")


def native_durability(durability: WriteDurability) -> str:
    """Translate the two public durability spellings."""
    if durability == "fsync":
        return "wal_fsync"
    if durability == "buffered":
        return "searchable"
    raise ValueError("durability must be fsync or buffered")
