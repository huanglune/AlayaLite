# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Gate 9 legacy API warning and local telemetry support.

The telemetry sink is deliberately process-local.  It emits a structured
``alayalite.legacy`` log record and retains the same non-user-data fields in
memory for diagnostics and tests; it never performs network I/O.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, TypeVar

LEGACY_API_V_REMOVE = "1.2.0"


class AlayaLiteLegacyApiWarning(DeprecationWarning):
    """Warning emitted on first use of a Gate 9 legacy Python entry."""


@dataclass(frozen=True)
class _LegacyTelemetryRecord:
    schema_version: int
    event: str
    category: str
    caller_file: str
    caller_line: int
    removal_version: str


_ALLOWED_CATEGORIES = frozenset({"index", "disk_collection", "laser", "vamana", "client_index"})
_LOGGER = logging.getLogger("alayalite.legacy")
_LOCK = threading.Lock()
_EMITTED_ENTRIES: set[str] = set()
_TELEMETRY: list[_LegacyTelemetryRecord] = []
_SUPPRESSED_ENTRIES = ContextVar("alayalite_suppressed_legacy_entries", default=frozenset())
_F = TypeVar("_F", bound=Callable)


def _legacy_message(api_name: str, replacement: str) -> str:
    return (
        f"{api_name} is a legacy API and will be removed in AlayaLite {LEGACY_API_V_REMOVE}; use {replacement} instead."
    )


def _claim_entry(entry: str) -> bool:
    with _LOCK:
        if entry in _EMITTED_ENTRIES:
            return False
        _EMITTED_ENTRIES.add(entry)
        return True


def _record_telemetry(category: str, caller_file: str, caller_line: int) -> None:
    record = _LegacyTelemetryRecord(
        schema_version=1,
        event="legacy_api_used",
        category=category,
        caller_file=caller_file,
        caller_line=caller_line,
        removal_version=LEGACY_API_V_REMOVE,
    )
    with _LOCK:
        _TELEMETRY.append(record)
    _LOGGER.info(
        "legacy_api_used category=%s caller_file=%s caller_line=%d removal_version=%s",
        record.category,
        record.caller_file,
        record.caller_line,
        record.removal_version,
        extra={
            "legacy_schema_version": record.schema_version,
            "legacy_event": record.event,
            "legacy_category": record.category,
            "legacy_caller_file": record.caller_file,
            "legacy_caller_line": record.caller_line,
            "legacy_removal_version": record.removal_version,
        },
    )


@contextmanager
def _suppress_legacy_warning(entry: str):
    """Suppress one nested wrapper without consuming its process-once event."""

    suppressed = _SUPPRESSED_ENTRIES.get()
    token = _SUPPRESSED_ENTRIES.set(suppressed | {entry})
    try:
        yield
    finally:
        _SUPPRESSED_ENTRIES.reset(token)


def legacy_api(
    entry: str,
    category: str,
    api_name: str,
    replacement: str,
    *,
    warning_category: type[Warning] = AlayaLiteLegacyApiWarning,
    message: str | None = None,
) -> Callable[[_F], _F]:
    """Decorate a public legacy boundary with a process-once warning.

    ``warnings.warn`` intentionally lives in the returned public wrapper, so
    its literal ``stacklevel=2`` resolves to the user's calling file and line.
    Multiple methods can share ``entry`` to form one legacy surface.
    """

    if category not in _ALLOWED_CATEGORIES:
        raise ValueError(f"unknown legacy telemetry category: {category}")
    warning_message = message or _legacy_message(api_name, replacement)

    def decorate(function: _F) -> _F:
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            if entry not in _SUPPRESSED_ENTRIES.get() and _claim_entry(entry):
                frame = inspect.currentframe()
                caller = frame.f_back if frame is not None else None
                caller_file = caller.f_code.co_filename if caller is not None else "<unknown>"
                caller_line = caller.f_lineno if caller is not None else 0
                try:
                    _record_telemetry(category, caller_file, caller_line)
                    warnings.warn(warning_message, warning_category, stacklevel=2)
                finally:
                    del caller
                    del frame
            return function(*args, **kwargs)

        return wrapped  # type: ignore[return-value]

    return decorate


def _legacy_telemetry_snapshot() -> tuple[_LegacyTelemetryRecord, ...]:
    """Return a stable in-memory snapshot without exposing mutable state."""

    with _LOCK:
        return tuple(_TELEMETRY)


def _reset_legacy_runtime_state_for_tests() -> None:
    """Reset process-once state for isolated warning contract tests."""

    with _LOCK:
        _EMITTED_ENTRIES.clear()
        _TELEMETRY.clear()
