// SPDX-FileCopyrightText: 2025 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <pybind11/pybind11.h>

#include <optional>
#include <string>

#ifdef ALAYA_ENABLE_LASER
  #include "simd/laser_dispatch.hpp"
#endif

#include "collection_binding.hpp"
#include "core/value_types.hpp"

namespace py = pybind11;
namespace collection_binding = alaya::python::collection_binding;

PYBIND11_MODULE(_alayalitepy, module) {
  module.doc() = "AlayaLite canonical Collection bindings";

#ifdef VERSION_INFO
  module.attr("__version__") = VERSION_INFO;
#else
  module.attr("__version__") = "dev";
#endif

  py::enum_<alaya::core::Metric>(module, "MetricType")
      .value("L2", alaya::core::Metric::l2)
      .value("IP", alaya::core::Metric::inner_product)
      .value("COS", alaya::core::Metric::cosine)
      .export_values();

  collection_binding::register_exceptions(module);
  collection_binding::register_response_types(module);
  collection_binding::PyCollectionClass collection(module, "_Collection");
  collection_binding::register_collection_factory(collection);
  collection_binding::register_collection_mutation(collection);
  collection_binding::register_collection_search(collection);
  collection_binding::register_collection_read(collection);
  collection_binding::register_collection_management(collection);
#ifdef ALAYA_ENABLE_LASER
  collection_binding::register_capabilities(module,
                                            true,
                                            std::string(alaya::laser::simd::get_laser_simd_name()));
#else
  collection_binding::register_capabilities(module, false, std::nullopt);
#endif
}
