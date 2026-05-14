// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <string>
#include <variant>

#include "utils/metadata_filter.hpp"

namespace py = pybind11;

namespace alaya {

class BasePyIndex {
 public:
  virtual ~BasePyIndex() = default;

  virtual auto to_string() const -> std::string = 0;
  virtual auto has_scalar_data() const -> bool = 0;

  virtual auto fit(py::array vectors,
                   uint32_t ef_construction,
                   uint32_t num_threads,
                   const py::object &item_ids,
                   const py::object &documents,
                   const py::object &metadata_list) -> void = 0;

  virtual auto search(py::array query, uint32_t topk, uint32_t ef) -> py::array = 0;
  virtual auto get_data_by_id(const py::object &id_obj) -> py::array = 0;

  virtual auto insert(py::array insert_data,
                      uint32_t ef,
                      const py::object &item_id_obj,
                      const std::string &document,
                      const py::dict &metadata) -> std::variant<uint32_t, uint64_t> = 0;

  virtual auto upsert(py::array insert_data,
                      uint32_t ef,
                      const py::object &item_id_obj,
                      const std::string &document,
                      const py::dict &metadata) -> std::variant<uint32_t, uint64_t> = 0;

  virtual auto remove(const py::object &id_obj) -> void = 0;
  virtual auto remove_by_item_id(const py::object &item_id_obj) -> void = 0;

  virtual auto contains(const py::object &item_id_obj) -> bool = 0;

  virtual auto get_scalar_data_by_item_id(const py::object &item_id_obj) -> py::dict = 0;
  virtual auto get_scalar_data_by_internal_id(const py::object &internal_id_obj) -> py::dict = 0;
  virtual auto batch_get_scalar_data_by_internal_ids(py::array internal_ids) -> py::list = 0;
  virtual auto batch_get_item_ids_by_internal_ids(py::array internal_ids) -> py::list = 0;

  virtual auto filter_query(const MetadataFilter &filter, uint32_t limit) -> py::object = 0;
  virtual auto get_data_num() -> std::variant<uint32_t, uint64_t> = 0;
  virtual auto get_materialized_view_partition_count() const -> uint32_t = 0;

  virtual auto batch_search(py::array queries, uint32_t topk, uint32_t ef, uint32_t num_threads)
      -> py::array = 0;
  virtual auto batch_search_with_distance(py::array queries,
                                          uint32_t topk,
                                          uint32_t ef,
                                          uint32_t num_threads) -> py::object = 0;

  virtual auto load(const std::string &index_path,
                    const std::string &data_path,
                    const std::string &quant_path) -> void = 0;
  virtual auto save(const std::string &index_path,
                    const std::string &data_path,
                    const std::string &quant_path) -> void = 0;

  virtual auto get_data_dim() -> uint32_t = 0;

  virtual auto hybrid_search(py::array query,
                             uint32_t topk,
                             uint32_t ef,
                             const MetadataFilter &filter,
                             bool bf,
                             const std::string &filter_exec_hint) -> py::object = 0;
  virtual auto batch_hybrid_search(py::array queries,
                                   uint32_t topk,
                                   uint32_t ef,
                                   const MetadataFilter &filter,
                                   uint32_t num_threads,
                                   bool bf,
                                   const std::string &filter_exec_hint) -> py::object = 0;

  virtual auto close_db() -> void = 0;
};

}  // namespace alaya
