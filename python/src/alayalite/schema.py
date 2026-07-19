# SPDX-FileCopyrightText: 2025 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
This module defines the schema for indices and collections, including
parameter classes and functions for saving and loading schema files.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .common import (
    IDType,
    VectorDType,
    assert_valid_index_type,
    assert_valid_metric_type,
    assert_valid_quantization_type,
    valid_capacity_type,
    valid_dtype,
    valid_id_type,
    valid_max_nbrs,
    valid_thread_count,
)

__all__ = ["IndexParams", "load_schema", "save_schema"]


@dataclass
class IndexParams:
    """Parameters for configuring vector index creation and management."""

    index_type: str = None
    data_type: VectorDType = None
    id_type: IDType = None
    quantization_type: str = None
    metric: str = None
    capacity: np.uint32 = None
    max_nbrs: int = None
    ef_construction: Optional[int] = None
    build_threads: Optional[int] = None
    materialized_view_build_threads: Optional[int] = None
    storage_path: str = ""  # Collection storage directory
    has_scalar_data: bool = False  # Whether to enable scalar data storage
    indexed_fields: list = None  # Fields to create secondary indexes for (for fast filtering)

    def index_path(self, folder_uri):
        return os.path.join(folder_uri, f"{self.index_type}_{self.metric}_{self.max_nbrs}.index")

    def data_path(self, folder_uri):
        return os.path.join(folder_uri, "raw.data")

    def quant_path(self, folder_uri):
        if self.quantization_type == "none":
            return ""
        else:
            return os.path.join(folder_uri, f"{self.quantization_type}.data")

    def fill_none_values(self):
        if self.data_type is None:
            self.data_type = np.float32
        if self.index_type is None:
            is_float32 = np.dtype(self.data_type) == np.dtype(np.float32)
            if is_float32 and self.quantization_type in (None, "rabitq"):
                # qg is the stable RaBitQ graph identity. Eligible sealed
                # generations use LASER on supported builds.
                self.index_type = "qg"
            elif is_float32:
                # An explicit sq8/sq4/none quantization_type has no ANN
                # engine left that can honor it: qg is RaBitQ-only and flat
                # never quantizes (it silently ignores a non-none
                # quantization_type and always searches exactly) -- so this
                # honestly downgrades to exact flat search rather than a
                # quantized ANN engine.
                self.index_type = "flat"
            else:
                # qg is physically float32-only; anything non-float32
                # honestly falls back to exact flat search rather than
                # silently downgrading accuracy on an unsupported engine.
                self.index_type = "flat"
        if self.id_type is None:
            self.id_type = np.uint32
        if self.quantization_type is None:
            self.quantization_type = "rabitq" if self.index_type == "qg" else "none"
        if self.metric is None:
            self.metric = "l2"
        if self.capacity is None:
            self.capacity = 100000
        if self.max_nbrs is None:
            self.max_nbrs = 32
        if self.ef_construction is None:
            self.ef_construction = 400

    def to_json_dict(self) -> dict:
        return {
            "index_type": self.index_type,
            "data_type": np.dtype(self.data_type).name,  # Convert dtype to string
            "id_type": np.dtype(self.id_type).name,  # Convert dtype to string
            "quantization_type": self.quantization_type,
            "metric": self.metric,
            "capacity": self.capacity,
            "max_nbrs": self.max_nbrs,
            "ef_construction": self.ef_construction,
            "build_threads": self.build_threads,
            "materialized_view_build_threads": self.materialized_view_build_threads,
            "storage_path": self.storage_path,
            "has_scalar_data": self.has_scalar_data,
            "indexed_fields": self.indexed_fields if self.indexed_fields else [],
        }

    @classmethod
    def from_str_dict(cls, data: dict) -> "IndexParams":
        """Deserialize from a JSON-compatible dict."""
        return cls(
            index_type=data["index_type"],
            data_type=np.dtype(data["data_type"]).type,  # Convert back to dtype
            id_type=np.dtype(data["id_type"]).type,  # Convert back to dtype
            quantization_type=data["quantization_type"],
            metric=data["metric"],
            capacity=data["capacity"],
            max_nbrs=data["max_nbrs"],
            ef_construction=data.get("ef_construction", 400),
            build_threads=data.get("build_threads") or None,
            materialized_view_build_threads=data.get("materialized_view_build_threads") or None,
            storage_path=data.get("storage_path", ""),
            has_scalar_data=data.get("has_scalar_data", False),  # Default to False for backward compatibility
            indexed_fields=data.get("indexed_fields", []),  # Default to empty list for backward compatibility
        )

    @classmethod
    def from_kwargs(cls, **kwargs) -> "IndexParams":
        index_type = None
        data_type = None
        id_type = None
        quantization_type = None
        metric = None
        capacity = None
        max_nbrs = None
        ef_construction = None
        build_threads = None
        materialized_view_build_threads = None
        storage_path = ""
        indexed_fields = None

        if kwargs.get("index_type") is not None:
            ind_type = kwargs.get("index_type")
            assert_valid_index_type(ind_type)
            index_type = ind_type
        if kwargs.get("data_type") is not None:
            data_type = valid_dtype(kwargs.get("data_type"))
        if kwargs.get("id_type") is not None:
            id_type = valid_id_type(kwargs.get("id_type"))
        if kwargs.get("quantization_type") is not None:
            qt = kwargs.get("quantization_type")
            assert_valid_quantization_type(qt)
            quantization_type = qt
        if kwargs.get("metric") is not None:
            mt = kwargs.get("metric")
            assert_valid_metric_type(mt)
            metric = mt
        if kwargs.get("capacity") is not None:
            capacity = valid_capacity_type(kwargs.get("capacity"))
        if kwargs.get("max_nbrs") is not None:
            max_nbrs = valid_max_nbrs(kwargs.get("max_nbrs"))
        if kwargs.get("ef_construction") is not None:
            ef_construction = int(kwargs.get("ef_construction"))
            if ef_construction <= 0:
                raise ValueError("ef_construction must be greater than 0")
        if kwargs.get("build_threads") is not None:
            build_threads = valid_thread_count(kwargs.get("build_threads"))
        if kwargs.get("materialized_view_build_threads") is not None:
            materialized_view_build_threads = valid_thread_count(kwargs.get("materialized_view_build_threads"))
        if kwargs.get("storage_path") is not None:
            storage_path = str(kwargs.get("storage_path"))
        if kwargs.get("indexed_fields") is not None:
            indexed_fields = list(kwargs.get("indexed_fields"))
        return cls(
            index_type=index_type,
            data_type=data_type,
            id_type=id_type,
            quantization_type=quantization_type,
            metric=metric,
            capacity=capacity,
            max_nbrs=max_nbrs,
            ef_construction=ef_construction,
            build_threads=build_threads,
            materialized_view_build_threads=materialized_view_build_threads,
            storage_path=storage_path,
            indexed_fields=indexed_fields,
        )


def load_schema(url) -> dict:
    if not os.path.exists(url):
        raise FileNotFoundError("The schema file does not exist!")
    with open(url, encoding="utf-8") as f:
        return json.load(f)


def save_schema(schema_url, schema_map):
    schema_dir = os.path.dirname(schema_url)
    if schema_dir:
        os.makedirs(schema_dir, exist_ok=True)
    tmp_schema_url = schema_url + ".tmp"
    with open(tmp_schema_url, "w", encoding="utf-8") as f:
        json.dump(schema_map, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_schema_url, schema_url)


def is_index_url(url):
    schema_url = os.path.join(url, "schema.json")
    if not os.path.exists(schema_url):
        return False
    else:
        schema_map = load_schema(schema_url)
        return schema_map["type"] == "index"


def is_collection_url(url):
    schema_url = os.path.join(url, "schema.json")
    if not os.path.exists(schema_url):
        return False
    else:
        schema_map = load_schema(schema_url)
        return schema_map["type"] == "collection"
