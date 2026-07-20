// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

void register_collection_search(PyCollectionClass &collection) {
  collection
      .def("search",
           &PyCollection::search,
           py::arg("query"),
           py::arg("top_k"),
           py::kw_only(),
           py::arg("effort") = 100,
           py::arg("metadata_filter") = py::none(),
           py::arg("filter_policy") = "auto",
           py::arg("filter_selectivity") = py::none(),
           py::arg("scratch_budget_bytes") = core::kUnlimitedResource,
           py::arg("io_budget_requests") = core::kUnlimitedResource,
           py::arg("io_budget_bytes") = core::kUnlimitedResource)
      .def("batch_search",
           &PyCollection::batch_search,
           py::arg("queries"),
           py::arg("top_k"),
           py::kw_only(),
           py::arg("effort") = 100,
           py::arg("metadata_filter") = py::none(),
           py::arg("filter_policy") = "auto",
           py::arg("filter_selectivity") = py::none(),
           py::arg("scratch_budget_bytes") = core::kUnlimitedResource,
           py::arg("io_budget_requests") = core::kUnlimitedResource,
           py::arg("io_budget_bytes") = core::kUnlimitedResource);
}

}  // namespace alaya::python::collection_binding
