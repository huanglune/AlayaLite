# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Public exception taxonomy for the AlayaLite v2 Python core."""

from __future__ import annotations

from typing import TypeVar

from ._alayalitepy import (
    CollectionCancelledError,
    CollectionClosedError,
    CollectionConflictError,
    CollectionCorruptionError,
    CollectionDeadlineExceededError,
    CollectionInternalError,
    CollectionInvalidArgumentError,
    CollectionIoError,
    CollectionNotFoundError,
    CollectionNotSupportedError,
    CollectionResourceExhaustedError,
    CollectionStatusError,
)

_StatusErrorT = TypeVar("_StatusErrorT", bound=CollectionStatusError)


def _status_error(
    error_type: type[_StatusErrorT],
    message: str,
    *,
    status_code: int,
    operation_stage: int,
    status_detail: int = 0,
    retryability: int = 0,
    partial: bool = False,
) -> _StatusErrorT:
    """Construct a Python-originated error with native protocol metadata."""
    error = error_type(message)
    error.status_code = status_code
    error.operation_stage = operation_stage
    error.status_detail = status_detail
    error.retryability = retryability
    error.partial = partial
    error.status_version = "1"
    return error


__all__ = [
    "CollectionStatusError",
    "CollectionInvalidArgumentError",
    "CollectionNotSupportedError",
    "CollectionConflictError",
    "CollectionNotFoundError",
    "CollectionResourceExhaustedError",
    "CollectionDeadlineExceededError",
    "CollectionCancelledError",
    "CollectionIoError",
    "CollectionCorruptionError",
    "CollectionClosedError",
    "CollectionInternalError",
]
