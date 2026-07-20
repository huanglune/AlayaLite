// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

void register_collection_factory(PyCollectionClass &collection) {
  collection
      .def_static("create",
                  &PyCollection::create,
                  py::arg("root"),
                  py::arg("dim"),
                  py::arg("metric"),
                  py::arg("dtype"),
                  py::arg("index_type"),
                  py::arg("quantization_type"),
                  py::arg("build_threads") = 1,
                  py::arg("max_neighbors") = 32,
                  py::arg("ef_construction") = 400,
                  py::arg("auto_seal_rows") = 0)
      .def_static("open", &PyCollection::open, py::arg("root"), py::arg("read_only") = false);
}

void register_collection_management(PyCollectionClass &collection) {
  collection.def("checkpoint", &PyCollection::checkpoint)
      .def("seal", &PyCollection::seal)
      .def("compact", &PyCollection::compact)
      .def("gc", &PyCollection::gc)
      .def("stats", &PyCollection::stats)
      .def("options", &PyCollection::options)
      .def_property_readonly("read_only", &PyCollection::read_only)
      .def("close", &PyCollection::close);
}

}  // namespace alaya::python::collection_binding
