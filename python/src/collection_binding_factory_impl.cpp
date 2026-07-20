// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

PyCollection::PyCollection(std::shared_ptr<Collection> collection)
    : collection_(std::move(collection)) {}

[[nodiscard]] auto PyCollection::create(const std::string &root,
                                        std::uint32_t dim,
                                        const std::string &metric_value,
                                        const py::dtype &dtype,
                                        const std::string &index_type,
                                        const std::string &quantization_type,
                                        std::uint32_t build_threads,
                                        std::uint32_t max_neighbors,
                                        std::uint32_t ef_construction,
                                        std::uint64_t auto_seal_rows)
    -> std::shared_ptr<PyCollection> {
  CollectionOptions options;
  options.root = root;
  options.dim = dim;
  options.metric = metric(metric_value);
  options.scalar_type = scalar_type(dtype);
  options.target_algorithm = algorithm(index_type);
  options.quantization = quantization(quantization_type);
  options.build_threads = build_threads;
  options.max_neighbors = max_neighbors;
  options.ef_construction = ef_construction;
  options.auto_seal_rows = auto_seal_rows;
  auto collection = [&] {
    py::gil_scoped_release release;
    return unwrap(Collection::create(std::move(options)));
  }();
  return std::make_shared<PyCollection>(std::move(collection));
}

[[nodiscard]] auto PyCollection::open(const std::string &root, bool read_only)
    -> std::shared_ptr<PyCollection> {
  CollectionOpenOptions options;
  options.read_only = read_only;
  auto collection = [&] {
    py::gil_scoped_release release;
    return unwrap(Collection::open(root, options));
  }();
  return std::make_shared<PyCollection>(std::move(collection));
}

}  // namespace alaya::python::collection_binding
