// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

void register_collection_mutation(PyCollectionClass &collection) {
  collection
      .def("mutate",
           &PyCollection::mutate,
           py::arg("ids"),
           py::arg("documents"),
           py::arg("vectors"),
           py::arg("metadata"),
           py::arg("action"),
           py::kw_only(),
           py::arg("mode") = "per_row_independent",
           py::arg("durability") = "wal_fsync",
           py::arg("retry_token") = "")
      .def("remove",
           &PyCollection::remove,
           py::arg("ids"),
           py::kw_only(),
           py::arg("mode") = "per_row_independent",
           py::arg("durability") = "wal_fsync",
           py::arg("retry_token") = "");
}

void register_collection_read(PyCollectionClass &collection) {
  collection
      .def("get_by_id",
           &PyCollection::get_by_id,
           py::arg("id"),
           py::kw_only(),
           py::arg("include_vector") = true)
      .def("get_by_ids",
           &PyCollection::get_by_ids,
           py::arg("ids"),
           py::kw_only(),
           py::arg("include_vector") = true)
      .def("records", &PyCollection::records)
      .def("scan",
           &PyCollection::scan,
           py::kw_only(),
           py::arg("metadata_filter") = py::none(),
           py::arg("limit") = 100,
           py::arg("include_vector") = false);
}

}  // namespace alaya::python::collection_binding
