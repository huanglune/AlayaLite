// SPDX-FileCopyrightText: 2025 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <pybind11/pybind11.h>

#include <string>

#ifdef ALAYA_ENABLE_LASER
  #include "simd/laser_dispatch.hpp"
#endif

#include "collection_binding.hpp"
#include "utils/metric_type.hpp"

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

  py::enum_<alaya::MetricType>(module, "MetricType")
      .value("L2", alaya::MetricType::L2)
      .value("IP", alaya::MetricType::IP)
      .value("COS", alaya::MetricType::COS)
      .export_values();

  alaya::python::collection_binding::register_collection(module);
}
