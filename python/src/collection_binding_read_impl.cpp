// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

[[nodiscard]] auto PyCollection::get_by_id(const std::string &id, bool include_vector)
    -> py::object {
  auto record = [&] {
    py::gil_scoped_release release;
    return collection_->get_by_id(logical_id(id), record_projection(include_vector));
  }();
  if (!record.ok() && record.status().code() == core::StatusCode::not_found) {
    return py::none();
  }
  return py::cast(record_to_response(unwrap(std::move(record))));
}

[[nodiscard]] auto PyCollection::get_by_ids(const py::list &ids, bool include_vector) -> py::list {
  py::list result;
  for (const auto &id : ids) {
    result.append(get_by_id(py::cast<std::string>(id), include_vector));
  }
  return result;
}

[[nodiscard]] auto PyCollection::records() -> std::vector<PyRecordResponse> {
  auto records = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->records());
  }();
  std::vector<PyRecordResponse> result;
  result.reserve(records.size());
  for (const auto &record : records) {
    result.push_back(record_to_response(record));
  }
  return result;
}

[[nodiscard]] auto PyCollection::scan(const py::object &metadata_filter,
                                      std::size_t limit,
                                      bool include_vector) -> std::vector<PyRecordResponse> {
  if (limit == 0) {
    throw py::value_error("canonical Collection scan limit must be positive");
  }
  const auto filter = collection_filter(metadata_filter, py::none());
  auto records = [&] {
    py::gil_scoped_release release;
    return unwrap(collection_->scan(filter, limit, record_projection(include_vector)));
  }();
  std::vector<PyRecordResponse> result;
  result.reserve(records.size());
  for (const auto &record : records) {
    result.push_back(record_to_response(record));
  }
  return result;
}

}  // namespace alaya::python::collection_binding
