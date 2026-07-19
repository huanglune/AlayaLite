# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Database catalog and connect entry point for the dormant SDK v2 core."""

# The Database/Collection pair deliberately shares a small private lifecycle
# seam while neither class is part of the old public entry point in this wave.
# pylint: disable=protected-access

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import warnings
import weakref
from pathlib import Path
from typing import final

from ._alayalitepy import _Collection as _NativeCollection
from ._collection import (
    Collection,
    config_from_native,
    config_matches_native,
    create_native_collection,
    validate_creation_config,
)
from ._schema import is_catalog_collection, load_collection_schema, write_collection_schema
from .config import CollectionConfig
from .exceptions import (
    CollectionClosedError,
    CollectionConflictError,
    CollectionIoError,
    CollectionNotFoundError,
    CollectionNotSupportedError,
    _status_error,
)

_COLLECTION_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


@final
class Database:
    """A local directory catalog of lazily opened collections.

    Notes
    -----
    A Database context manager owns resource lifetime only; it is not a
    transaction. Closing it closes all collection handles still reachable by
    this catalog and is idempotent.
    """

    __slots__ = (
        "__weakref__",
        "_closed",
        "_handles",
        "_lock",
        "_path",
        "_read_only",
        "_temporary",
    )

    def __init__(
        self,
        path: Path,
        *,
        read_only: bool,
        temporary: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self._path = path
        self._read_only = read_only
        self._temporary = temporary
        self._closed = False
        self._handles: dict[tuple[str, bool], weakref.ReferenceType[Collection]] = {}
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Return the resolved database directory."""
        return self._path

    @property
    def read_only(self) -> bool:
        """Return whether catalog mutation and permission upgrades are denied."""
        return self._read_only

    def list_collections(self) -> list[str]:
        """Return sorted catalog names without opening native owners.

        Unknown, malformed, or unrelated child directories are ignored so one
        bad entry cannot prevent connecting to the database.
        """
        self._require_open()
        try:
            children = tuple(self._path.iterdir())
        except OSError as error:
            raise _status_error(
                CollectionIoError,
                f"failed to read database catalog: {error}",
                status_code=8,
                operation_stage=3,
            ) from error
        return sorted(child.name for child in children if child.is_dir() and is_catalog_collection(child))

    def create_collection(
        self,
        name: str,
        *,
        config: CollectionConfig,
    ) -> Collection:
        """Create and immediately materialize an empty native collection.

        Parameters
        ----------
        name
            Safe single path component of 1 through 128 ASCII characters.
        config
            Immutable schema and discriminated index configuration.

        Returns
        -------
        Collection
            Open read-write collection handle.

        Raises
        ------
        CollectionNotSupportedError
            If this database is read-only or QG is unavailable.
        CollectionConflictError
            If the catalog name already exists.
        """
        self._require_writable()
        normalized_name = _collection_name(name)
        validate_creation_config(config)
        root = self._path / normalized_name
        if os.path.lexists(root):
            raise _status_error(
                CollectionConflictError,
                f"collection {normalized_name!r} already exists",
                status_code=3,
                operation_stage=2,
                status_detail=14,
            )
        native: _NativeCollection | None = None
        try:
            native = create_native_collection(root, config)
            write_collection_schema(root, config)
        except OSError as error:
            if native is not None:
                native.close()
                shutil.rmtree(root, ignore_errors=True)
            raise _status_error(
                CollectionIoError,
                f"failed to persist collection catalog schema: {error}",
                status_code=8,
                operation_stage=2,
            ) from error
        collection = Collection._create_handle(
            database=self,
            name=normalized_name,
            path=root,
            config=config,
            native=native,
        )
        self._register(collection)
        return collection

    def open_collection(
        self,
        name: str,
        *,
        read_only: bool | None = None,
    ) -> Collection:
        """Lazily open a collection and optionally narrow its permissions.

        Parameters
        ----------
        name
            Safe single path component.
        read_only
            ``None`` inherits the Database mode. A read-write Database may
            narrow one handle; a read-only Database cannot upgrade one.
        """
        self._require_open()
        normalized_name = _collection_name(name)
        if read_only is not None and not isinstance(read_only, bool):
            raise TypeError("read_only must be a bool or None")
        if self._read_only and read_only is False:
            raise _status_error(
                CollectionNotSupportedError,
                "a read-only Database cannot open a read-write Collection handle",
                status_code=2,
                operation_stage=3,
                status_detail=15,
            )
        resolved_read_only = self._read_only if read_only is None else read_only
        cached = self._cached(normalized_name, resolved_read_only)
        if cached is not None:
            return cached
        root = self._path / normalized_name
        native = _NativeCollection.open(os.fspath(root), resolved_read_only)
        try:
            options = native.options_typed()
            discovered_config, legacy_quantization = load_collection_schema(root)
            config = (
                discovered_config
                if discovered_config is not None and config_matches_native(discovered_config, options)
                else config_from_native(options)
            )
            collection = Collection._create_handle(
                database=self,
                name=normalized_name,
                path=root,
                config=config,
                native=native,
                legacy_quantization=legacy_quantization,
            )
        except Exception:
            native.close()
            raise
        self._register(collection)
        return collection

    def drop_collection(self, name: str, *, missing_ok: bool = False) -> None:
        """Permanently remove a closed collection directory.

        Parameters
        ----------
        name
            Catalog collection name.
        missing_ok
            Suppress the typed not-found error when true.
        """
        self._require_writable()
        normalized_name = _collection_name(name)
        if not isinstance(missing_ok, bool):
            raise TypeError("missing_ok must be a bool")
        if self._has_live_handle(normalized_name):
            raise _status_error(
                CollectionConflictError,
                f"collection {normalized_name!r} has an active handle",
                status_code=3,
                operation_stage=1,
                status_detail=14,
            )
        root = self._path / normalized_name
        if not is_catalog_collection(root):
            if missing_ok:
                return
            raise _status_error(
                CollectionNotFoundError,
                f"collection {normalized_name!r} does not exist",
                status_code=4,
                operation_stage=3,
            )
        try:
            shutil.rmtree(root)
        except FileNotFoundError as error:
            if not missing_ok:
                raise _status_error(
                    CollectionNotFoundError,
                    f"collection {normalized_name!r} does not exist",
                    status_code=4,
                    operation_stage=3,
                ) from error
        except OSError as error:
            raise _status_error(
                CollectionIoError,
                f"failed to drop collection {normalized_name!r}: {error}",
                status_code=8,
                operation_stage=18,
            ) from error

    def close(self) -> None:
        """Idempotently close tracked collection handles and this catalog."""
        with self._lock:
            if self._closed:
                return
            handles = tuple(
                collection for reference in self._handles.values() if (collection := reference()) is not None
            )
            first_error: RuntimeError | None = None
            for collection in handles:
                try:
                    collection.close()
                except RuntimeError as error:
                    if first_error is None:
                        first_error = error
            self._handles.clear()
            self._closed = True
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            if first_error is not None:
                raise first_error

    def __enter__(self) -> Database:
        """Return this open database handle."""
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Close resources without transaction or rollback semantics."""
        del exc_type, exc_value, traceback
        self.close()

    def __del__(self) -> None:
        """Warn and best-effort close an unclosed database handle."""
        try:
            if not self._closed:
                warnings.warn(
                    f"unclosed AlayaLite Database at {self._path}",
                    ResourceWarning,
                    stacklevel=2,
                )
                self.close()
        except (AttributeError, RuntimeError):
            pass

    def _register(self, collection: Collection) -> None:
        """Track a handle weakly for conflict checks and Database.close."""
        key = (collection.name, collection.read_only)
        database_ref = weakref.ref(self)

        def discard(reference: weakref.ReferenceType[Collection]) -> None:
            database = database_ref()
            if database is not None:
                with database._lock:
                    if database._handles.get(key) is reference:
                        database._handles.pop(key, None)

        with self._lock:
            self._handles[key] = weakref.ref(collection, discard)

    def _cached(self, name: str, read_only: bool) -> Collection | None:
        """Return a still-open cached handle for one name and mode."""
        with self._lock:
            reference = self._handles.get((name, read_only))
            collection = None if reference is None else reference()
            if collection is None or collection._closed:
                self._handles.pop((name, read_only), None)
                return None
            return collection

    def _has_live_handle(self, name: str) -> bool:
        """Return whether any mode has a still-open handle for this name."""
        with self._lock:
            for key, reference in tuple(self._handles.items()):
                collection = reference()
                if collection is None or collection._closed:
                    self._handles.pop(key, None)
                    continue
                if key[0] == name:
                    return True
            return False

    def _collection_closed(self, collection: Collection) -> None:
        """Forget a collection that explicitly closed itself."""
        with self._lock:
            for key, reference in tuple(self._handles.items()):
                if reference() is collection:
                    self._handles.pop(key, None)

    def _require_open(self) -> None:
        """Raise the status-protocol closed error after Database.close."""
        if self._closed:
            raise _status_error(
                CollectionClosedError,
                "Database handle is closed",
                status_code=10,
                operation_stage=1,
            )

    def _require_writable(self) -> None:
        """Reject catalog mutation on a read-only Database."""
        self._require_open()
        if self._read_only:
            raise _status_error(
                CollectionNotSupportedError,
                "catalog mutation is unavailable on a read-only Database",
                status_code=2,
                operation_stage=1,
                status_detail=15,
            )


def connect(
    path: str | os.PathLike[str] | None = None,
    *,
    read_only: bool = False,
) -> Database:
    """Connect to a local embedded database directory.

    Parameters
    ----------
    path
        Local filesystem directory. ``None`` and ``":memory:"`` create a
        temporary process-scoped database.
    read_only
        Require an existing directory and prohibit catalog mutation.

    Returns
    -------
    Database
        Open local catalog handle.

    Raises
    ------
    ValueError
        If ``path`` uses a remote or URI spelling.
    CollectionNotFoundError
        If a read-only path does not exist.
    """
    if not isinstance(read_only, bool):
        raise TypeError("read_only must be a bool")
    if path is None or path == ":memory:":
        if read_only:
            raise _status_error(
                CollectionNotSupportedError,
                "an in-memory Database cannot be opened read-only",
                status_code=2,
                operation_stage=3,
                status_detail=15,
            )
        temporary = tempfile.TemporaryDirectory(prefix="alayalite-")  # pylint: disable=consider-using-with
        return Database(Path(temporary.name).resolve(), read_only=False, temporary=temporary)
    raw_path = os.fspath(path)
    if not isinstance(raw_path, str):
        raise TypeError("path must resolve to a string filesystem path")
    if _URI.match(raw_path):
        raise ValueError("connect accepts local filesystem paths, not URI spellings")
    resolved = Path(raw_path).expanduser().resolve()
    if read_only:
        if not resolved.is_dir():
            raise _status_error(
                CollectionNotFoundError,
                f"database directory does not exist: {resolved}",
                status_code=4,
                operation_stage=3,
            )
    else:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise _status_error(
                CollectionIoError,
                f"failed to create database directory: {error}",
                status_code=8,
                operation_stage=3,
            ) from error
        if not resolved.is_dir():
            raise ValueError("database path must be a directory")
    return Database(resolved, read_only=read_only)


def _collection_name(name: str) -> str:
    """Validate a safe single catalog path component."""
    if not isinstance(name, str):
        raise TypeError("collection name must be a str")
    if _COLLECTION_NAME.fullmatch(name) is None:
        raise ValueError("collection name must be a safe 1-128 character path component")
    return name


__all__ = ["Database", "connect"]
