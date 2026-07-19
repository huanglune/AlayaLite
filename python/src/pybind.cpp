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

PYBIND11_MODULE(_alayalitepy, module) {
  module.doc() = "AlayaLite canonical Collection bindings";

#ifdef ALAYA_ENABLE_LASER
  auto laser = module.def_submodule("laser", "LASER diagnostics");
  laser.def("selected_simd", [] {
    return std::string(alaya::laser::simd::get_laser_simd_name());
  });
#endif

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

  alaya::python::collection_binding::register_collection(module);
#ifdef ALAYA_ENABLE_LASER
  alaya::python::collection_binding::
      register_capabilities(module, true, std::string(alaya::laser::simd::get_laser_simd_name()));
#else
  alaya::python::collection_binding::register_capabilities(module, false, std::nullopt);
#endif
}
