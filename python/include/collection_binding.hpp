// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "alaya/collection.hpp"

namespace alaya::python::collection_binding {

namespace py = pybind11;

class CollectionStatusException : public std::runtime_error {
 public:
  explicit CollectionStatusException(core::Status status)
      : std::runtime_error(status.diagnostic().empty() ? "canonical Collection operation failed"
                                                       : status.diagnostic()),
        status_(std::move(status)) {}

  [[nodiscard]] auto status() const noexcept -> const core::Status & { return status_; }

 private:
  core::Status status_{};
};

inline PyObject *g_status_exceptions[12]{};

[[nodiscard]] inline auto status_exception_index(core::StatusCode code) -> std::size_t {
  const auto value = static_cast<std::size_t>(code);
  return value < std::size(g_status_exceptions)
             ? value
             : static_cast<std::size_t>(core::StatusCode::internal);
}

inline void translate_collection_status(std::exception_ptr pointer) {
  try {
    if (pointer != nullptr) {
      std::rethrow_exception(pointer);
    }
  } catch (const CollectionStatusException &exception) {
    const auto &status = exception.status();
    auto *type = g_status_exceptions[status_exception_index(status.code())];
    if (type == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, exception.what());
      return;
    }
    auto *message = PyUnicode_FromString(exception.what());
    auto *instance = PyObject_CallFunctionObjArgs(type, message, nullptr);
    Py_DECREF(message);
    if (instance == nullptr) {
      return;
    }
    auto set_integer = [&](const char *name, std::uint64_t value) {
      auto *number = PyLong_FromUnsignedLongLong(value);
      PyObject_SetAttrString(instance, name, number);
      Py_DECREF(number);
    };
    set_integer("status_code", static_cast<std::uint8_t>(status.code()));
    set_integer("operation_stage", static_cast<std::uint8_t>(status.stage()));
    set_integer("status_detail", static_cast<std::uint16_t>(status.detail()));
    set_integer("retryability", static_cast<std::uint8_t>(status.retryability()));
    auto *partial = PyBool_FromLong(status.partial() ? 1 : 0);
    PyObject_SetAttrString(instance, "partial", partial);
    Py_DECREF(partial);
    auto *version = PyUnicode_FromString("1");
    PyObject_SetAttrString(instance, "status_version", version);
    Py_DECREF(version);
    PyErr_SetObject(type, instance);
    Py_DECREF(instance);
  }
}

inline void throw_status(const core::Status &status) {
  if (!status.ok()) {
    throw CollectionStatusException(status);
  }
}

template <class T>
[[nodiscard]] auto unwrap(core::Result<T> result) -> T {
  if (!result.ok()) {
    throw CollectionStatusException(result.status());
  }
  return std::move(result).value();
}

[[nodiscard]] inline auto exception_base_for(core::StatusCode code) -> PyObject * {
  switch (code) {
    case core::StatusCode::invalid_argument:
      return PyExc_ValueError;
    case core::StatusCode::not_supported:
      return PyExc_NotImplementedError;
    case core::StatusCode::not_found:
      return PyExc_KeyError;
    default:
      return PyExc_RuntimeError;
  }
}

inline void register_exceptions(py::module_ &module) {
  auto *base = PyErr_NewException("alayalite._alayalitepy.CollectionStatusError",
                                  PyExc_RuntimeError,
                                  nullptr);
  if (base == nullptr) {
    throw py::error_already_set();
  }
  module.attr("CollectionStatusError") = py::reinterpret_borrow<py::object>(base);
  g_status_exceptions[0] = base;
  const std::array<std::pair<core::StatusCode, const char *>, 11> names{{
      {core::StatusCode::invalid_argument, "CollectionInvalidArgumentError"},
      {core::StatusCode::not_supported, "CollectionNotSupportedError"},
      {core::StatusCode::conflict, "CollectionConflictError"},
      {core::StatusCode::not_found, "CollectionNotFoundError"},
      {core::StatusCode::resource_exhausted, "CollectionResourceExhaustedError"},
      {core::StatusCode::deadline_exceeded, "CollectionDeadlineExceededError"},
      {core::StatusCode::cancelled, "CollectionCancelledError"},
      {core::StatusCode::io_error, "CollectionIoError"},
      {core::StatusCode::corruption, "CollectionCorruptionError"},
      {core::StatusCode::closed, "CollectionClosedError"},
      {core::StatusCode::internal, "CollectionInternalError"},
  }};
  for (const auto &[code, name] : names) {
    auto *bases = PyTuple_Pack(2, base, exception_base_for(code));
    auto *type =
        PyErr_NewException(("alayalite._alayalitepy." + std::string(name)).c_str(), bases, nullptr);
    Py_DECREF(bases);
    if (type == nullptr) {
      throw py::error_already_set();
    }
    module.attr(name) = py::reinterpret_borrow<py::object>(type);
    g_status_exceptions[status_exception_index(code)] = type;
  }
  py::register_exception_translator(&translate_collection_status);
}

[[nodiscard]] inline auto scalar_type(const py::dtype &dtype) -> core::ScalarType {
  if (dtype.is(py::dtype::of<float>())) {
    return core::ScalarType::float32;
  }
  if (dtype.is(py::dtype::of<std::int8_t>())) {
    return core::ScalarType::int8;
  }
  if (dtype.is(py::dtype::of<std::uint8_t>())) {
    return core::ScalarType::uint8;
  }
  throw py::type_error("canonical Collection dtype must be float32, int8, or uint8");
}

[[nodiscard]] inline auto scalar_dtype(core::ScalarType scalar) -> py::dtype {
  switch (scalar) {
    case core::ScalarType::float32:
      return py::dtype::of<float>();
    case core::ScalarType::int8:
      return py::dtype::of<std::int8_t>();
    case core::ScalarType::uint8:
      return py::dtype::of<std::uint8_t>();
  }
  throw py::type_error("canonical Collection scalar type is unsupported");
}

[[nodiscard]] inline auto metric(std::string_view value) -> core::Metric {
  if (value == "l2" || value == "euclidean") {
    return core::Metric::l2;
  }
  if (value == "ip") {
    return core::Metric::inner_product;
  }
  if (value == "cos" || value == "cosine") {
    return core::Metric::cosine;
  }
  throw py::value_error("canonical Collection metric must be l2, ip, or cosine");
}

[[nodiscard]] inline auto metric_name(core::Metric value) -> std::string {
  switch (value) {
    case core::Metric::l2:
      return "l2";
    case core::Metric::inner_product:
      return "ip";
    case core::Metric::cosine:
      return "cosine";
  }
  return "unknown";
}

[[nodiscard]] inline auto algorithm(std::string_view value) -> core::AlgorithmId {
  if (value == "flat") {
    return core::algorithm::flat;
  }
  if (value == "hnsw") {
    return core::algorithm::hnsw;
  }
  if (value == "nsg") {
    return core::algorithm::nsg;
  }
  if (value == "fusion") {
    return core::algorithm::fusion;
  }
  if (value == "qg") {
    return core::algorithm::qg;
  }
  throw py::value_error("canonical Collection index_type must be flat, hnsw, nsg, fusion, or qg");
}

[[nodiscard]] inline auto algorithm_name(core::AlgorithmId value) -> std::string {
  switch (value) {
    case core::algorithm::flat:
      return "flat";
    case core::algorithm::hnsw:
      return "hnsw";
    case core::algorithm::nsg:
      return "nsg";
    case core::algorithm::fusion:
      return "fusion";
    case core::algorithm::qg:
      return "qg";
    default:
      return "unknown";
  }
}

[[nodiscard]] inline auto quantization(std::string_view value) -> CollectionQuantization {
  if (value.empty() || value == "none") {
    return CollectionQuantization::none;
  }
  if (value == "sq8") {
    return CollectionQuantization::sq8;
  }
  if (value == "sq4") {
    return CollectionQuantization::sq4;
  }
  if (value == "rabitq") {
    return CollectionQuantization::rabitq;
  }
  throw py::value_error("canonical Collection quantization_type is unsupported");
}

[[nodiscard]] inline auto quantization_name(CollectionQuantization value) -> std::string {
  switch (value) {
    case CollectionQuantization::none:
      return "none";
    case CollectionQuantization::sq8:
      return "sq8";
    case CollectionQuantization::sq4:
      return "sq4";
    case CollectionQuantization::rabitq:
      return "rabitq";
  }
  return "unknown";
}

[[nodiscard]] inline auto metadata_value(const py::handle &value) -> CollectionScalarValue {
  if (py::isinstance<py::bool_>(value)) {
    return value.cast<bool>();
  }
  if (py::isinstance<py::int_>(value)) {
    return value.cast<std::int64_t>();
  }
  if (py::isinstance<py::float_>(value)) {
    return value.cast<double>();
  }
  if (py::isinstance<py::str>(value)) {
    return value.cast<std::string>();
  }
  throw py::type_error("canonical Collection metadata values must be bool, int64, float, or str");
}

[[nodiscard]] inline auto metadata_from_python(const py::handle &object) -> CollectionMetadata {
  if (object.is_none()) {
    return {};
  }
  const auto dictionary = py::reinterpret_borrow<py::dict>(object);
  CollectionMetadata result;
  for (const auto &[key, value] : dictionary) {
    result.emplace(py::cast<std::string>(key), metadata_value(value));
  }
  return result;
}

[[nodiscard]] inline auto metadata_to_python(const CollectionMetadata &metadata) -> py::dict {
  py::dict result;
  for (const auto &[key, value] : metadata) {
    std::visit(
        [&](const auto &item) {
          result[py::str(key)] = py::cast(item);
        },
        value);
  }
  return result;
}

[[nodiscard]] inline auto logical_id(std::string_view value) -> core::LogicalId {
  return core::LogicalId::from_utf8(value);
}

[[nodiscard]] inline auto logical_id_to_python(const core::LogicalId &id) -> py::object {
  const auto bytes = id.canonical_bytes();
  if (id.kind() == core::LogicalIdKind::utf8) {
    return py::str(std::string(reinterpret_cast<const char *>(bytes.data()), bytes.size()));
  }
  if (bytes.size() != sizeof(std::uint64_t)) {
    throw std::runtime_error("canonical Collection legacy LogicalId width is invalid");
  }
  std::uint64_t value{};
  for (const auto byte : bytes) {
    value = (value << 8U) | std::to_integer<std::uint8_t>(byte);
  }
  return py::int_(value);
}

[[nodiscard]] inline auto tensor_view(const py::array &vectors,
                                      std::uint32_t expected_dim,
                                      bool require_two_dimensions = true) -> core::TypedTensorView {
  if (require_two_dimensions && vectors.ndim() != 2) {
    throw py::value_error("canonical Collection vectors must be a 2D array");
  }
  if (!require_two_dimensions && vectors.ndim() != 1 && vectors.ndim() != 2) {
    throw py::value_error("canonical Collection query must be a 1D or 2D array");
  }
  const auto rows = vectors.ndim() == 1 ? 1 : static_cast<std::uint64_t>(vectors.shape(0));
  const auto dim = vectors.ndim() == 1 ? static_cast<std::uint64_t>(vectors.shape(0))
                                       : static_cast<std::uint64_t>(vectors.shape(1));
  if (dim != expected_dim) {
    throw py::value_error("Vector dimension must match the index dimension.");
  }
  if ((vectors.flags() & py::array::c_style) == 0) {
    throw py::type_error("canonical Collection vectors must be C-contiguous");
  }
  const auto stride = vectors.ndim() == 1
                          ? vectors.itemsize() * static_cast<py::ssize_t>(expected_dim)
                          : vectors.strides(0);
  return {vectors.data(),
          scalar_type(vectors.dtype()),
          rows,
          expected_dim,
          static_cast<std::uint64_t>(stride)};
}

[[nodiscard]] inline auto owned_vector_to_array(const internal::collection::OwnedVector &vector)
    -> py::array {
  py::array result(scalar_dtype(vector.scalar_type()),
                   py::array::ShapeContainer{static_cast<py::ssize_t>(vector.dim())});
  std::memcpy(result.mutable_data(), vector.bytes().data(), vector.bytes().size());
  return result;
}

[[nodiscard]] inline auto record_to_dict(const CollectionRecord &record) -> py::dict {
  py::dict result;
  result["id"] = logical_id_to_python(record.logical_id);
  result["upsert_sequence"] = record.upsert_sequence;
  result["document"] = record.document;
  result["metadata"] = metadata_to_python(record.metadata);
  result["vector"] =
      record.vector.has_value() ? py::object(owned_vector_to_array(*record.vector)) : py::none();
  return result;
}

[[nodiscard]] inline auto receipt_to_dict(const CollectionMutationReceipt &receipt) -> py::dict {
  py::dict result;
  result["op_id"] = receipt.op_id;
  result["batch_op_id"] = receipt.batch_op_id;
  result["row_op_id"] = receipt.row_op_id;
  result["visibility_watermark"] = receipt.visibility_watermark;
  result["durable_watermark"] = receipt.durable_watermark;
  result["searchable"] = receipt.searchable;
  result["durability"] = static_cast<std::uint8_t>(receipt.durability);
  result["row_status"] = static_cast<std::uint8_t>(receipt.row_status);
  result["retry_token"] = receipt.retry_token;
  return result;
}

[[nodiscard]] inline auto batch_receipt_to_dict(const CollectionBatchMutationReceipt &receipt)
    -> py::dict {
  py::dict result;
  result["batch_op_id"] = receipt.batch_op_id;
  result["visibility_watermark"] = receipt.visibility_watermark;
  result["durable_watermark"] = receipt.durable_watermark;
  result["searchable"] = receipt.searchable;
  result["durability"] = static_cast<std::uint8_t>(receipt.durability);
  result["retry_token"] = receipt.retry_token;
  py::list rows;
  for (const auto &row : receipt.rows) {
    rows.append(receipt_to_dict(row));
  }
  result["rows"] = std::move(rows);
  return result;
}

template <class T>
[[nodiscard]] inline auto copy_array(const std::vector<T> &values) -> py::array_t<T> {
  py::array_t<T> result(values.size());
  if (!values.empty()) {
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(T));
  }
  return result;
}

[[nodiscard]] inline auto search_response_to_dict(const CollectionSearchResponse &response)
    -> py::dict {
  py::list id_list;
  for (const auto &id : response.ids) {
    id_list.append(logical_id_to_python(id));
  }
  auto numpy = py::module_::import("numpy");
  py::object ids = numpy.attr("asarray")(id_list, py::arg("dtype") = numpy.attr("object_"));
  std::vector<std::uint8_t> statuses;
  std::vector<std::uint8_t> completeness;
  statuses.reserve(response.statuses.size());
  completeness.reserve(response.completeness.size());
  for (const auto &status : response.statuses) {
    statuses.push_back(static_cast<std::uint8_t>(status.code()));
  }
  for (const auto value : response.completeness) {
    completeness.push_back(static_cast<std::uint8_t>(value));
  }
  py::dict result;
  result["ids"] = std::move(ids);
  result["distances"] = copy_array(response.distances);
  result["offsets"] = copy_array(response.offsets);
  result["valid_counts"] = copy_array(response.valid_counts);
  result["statuses"] = copy_array(statuses);
  result["completeness"] = copy_array(completeness);
  result["visibility_watermark"] = response.visibility_watermark;
  result["metadata_epoch"] = response.metadata_epoch;
  return result;
}

[[nodiscard]] inline auto write_options(std::string_view durability, const std::string &retry_token)
    -> CollectionWriteOptions {
  CollectionWriteOptions result;
  if (durability == "wal_fsync") {
    result.durability = CollectionWriteDurability::wal_fsync;
  } else if (durability == "searchable") {
    result.durability = CollectionWriteDurability::searchable;
  } else {
    throw py::value_error("canonical Collection durability must be wal_fsync or searchable");
  }
  result.retry_token = retry_token;
  return result;
}

[[nodiscard]] inline auto batch_mode(std::string_view mode) -> CollectionBatchMutationMode {
  if (mode == "per_row_independent") {
    return CollectionBatchMutationMode::per_row_independent;
  }
  if (mode == "all_or_nothing") {
    return CollectionBatchMutationMode::all_or_nothing;
  }
  throw py::value_error(
      "canonical Collection batch mode must be per_row_independent or all_or_nothing");
}

class PyCollection {
 public:
  explicit PyCollection(std::shared_ptr<Collection> collection)
      : collection_(std::move(collection)) {}

  [[nodiscard]] static auto create(const std::string &root,
                                   std::uint32_t dim,
                                   const std::string &metric_value,
                                   const py::dtype &dtype,
                                   const std::string &index_type,
                                   const std::string &quantization_type,
                                   std::uint32_t build_threads,
                                   std::uint32_t ef_construction) -> std::shared_ptr<PyCollection> {
    CollectionOptions options;
    options.root = root;
    options.dim = dim;
    options.metric = metric(metric_value);
    options.scalar_type = scalar_type(dtype);
    options.target_algorithm = algorithm(index_type);
    options.quantization = quantization(quantization_type);
    options.build_threads = build_threads;
    options.ef_construction = ef_construction;
    return std::make_shared<PyCollection>(unwrap(Collection::create(std::move(options))));
  }

  [[nodiscard]] static auto open(const std::string &root) -> std::shared_ptr<PyCollection> {
    return std::make_shared<PyCollection>(unwrap(Collection::open(root)));
  }

  [[nodiscard]] auto mutate(const py::list &ids,
                            const py::list &documents,
                            const py::array &vectors,
                            const py::list &metadata,
                            const std::string &action,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> py::dict {
    const auto view = tensor_view(vectors, collection_->options().dim);
    const auto rows = static_cast<std::size_t>(view.rows);
    if (ids.size() != rows || documents.size() != rows || metadata.size() != rows) {
      throw py::value_error("canonical Collection item columns must have equal row counts");
    }
    CollectionMutationAction mutation_action;
    if (action == "add") {
      mutation_action = CollectionMutationAction::add;
    } else if (action == "upsert") {
      mutation_action = CollectionMutationAction::upsert;
    } else if (action == "replace") {
      mutation_action = CollectionMutationAction::replace;
    } else {
      throw py::value_error("canonical Collection mutation action is invalid");
    }
    std::vector<CollectionBatchRow> native;
    native.reserve(rows);
    const auto row_bytes = static_cast<std::uint64_t>(collection_->options().dim) *
                           core::scalar_type_size(collection_->options().scalar_type);
    for (std::size_t index = 0; index < rows; ++index) {
      const auto *row = static_cast<const std::byte *>(view.data) + index * view.row_stride;
      native.push_back(CollectionBatchRow{
          mutation_action,
          logical_id(py::cast<std::string>(ids[index])),
          core::TypedTensorView(row, view.scalar_type, 1, view.dim, row_bytes),
          metadata_from_python(metadata[index]),
          py::cast<std::string>(documents[index]),
          {},
      });
    }
    auto receipt = unwrap(collection_->mutate_batch(native,
                                                    batch_mode(mode),
                                                    write_options(durability, retry_token)));
    return batch_receipt_to_dict(receipt);
  }

  [[nodiscard]] auto remove(const py::list &ids,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> py::dict {
    std::vector<CollectionBatchRow> native;
    native.reserve(ids.size());
    for (const auto &id : ids) {
      CollectionBatchRow row;
      row.action = CollectionMutationAction::remove;
      row.logical_id = logical_id(py::cast<std::string>(id));
      native.push_back(std::move(row));
    }
    return batch_receipt_to_dict(
        unwrap(collection_->mutate_batch(native,
                                         batch_mode(mode),
                                         write_options(durability, retry_token))));
  }

  [[nodiscard]] auto search(const py::array &queries, std::uint64_t top_k) -> py::dict {
    const auto view = tensor_view(queries, collection_->options().dim, false);
    if (view.scalar_type != collection_->options().scalar_type) {
      throw py::type_error("canonical Collection query dtype must match the collection dtype");
    }
    if (queries.ndim() == 1) {
      return search_response_to_dict(unwrap(collection_->search(view, top_k)));
    }
    return search_response_to_dict(unwrap(collection_->batch_search(view, top_k)));
  }

  [[nodiscard]] auto batch_search(const py::array &queries, std::uint64_t top_k) -> py::dict {
    const auto view = tensor_view(queries, collection_->options().dim);
    if (view.scalar_type != collection_->options().scalar_type) {
      throw py::type_error("canonical Collection query dtype must match the collection dtype");
    }
    return search_response_to_dict(unwrap(collection_->batch_search(view, top_k)));
  }

  [[nodiscard]] auto get_by_id(const std::string &id) -> py::object {
    auto record = collection_->get_by_id(logical_id(id));
    if (!record.ok() && record.status().code() == core::StatusCode::not_found) {
      return py::none();
    }
    return record_to_dict(unwrap(std::move(record)));
  }

  [[nodiscard]] auto get_by_ids(const py::list &ids) -> py::list {
    py::list result;
    for (const auto &id : ids) {
      result.append(get_by_id(py::cast<std::string>(id)));
    }
    return result;
  }

  [[nodiscard]] auto records() -> py::list {
    py::list result;
    for (const auto &record : unwrap(collection_->records())) {
      result.append(record_to_dict(record));
    }
    return result;
  }

  [[nodiscard]] auto checkpoint() -> py::dict {
    const auto receipt = unwrap(collection_->checkpoint());
    py::dict result;
    result["durable_watermark"] = receipt.durable_watermark;
    result["wal_cut"] = receipt.wal_cut;
    result["metadata_epoch"] = receipt.metadata_epoch;
    result["checkpoint_name"] = receipt.checkpoint_name;
    return result;
  }

  [[nodiscard]] auto stats() const -> py::dict {
    const auto stats = collection_->stats();
    py::dict result;
    result["size"] = stats.size;
    result["accepted_count"] = stats.accepted_count;
    result["pending_count"] = stats.pending_count;
    result["searchable_bytes"] = stats.searchable_bytes;
    result["accepted_bytes"] = stats.accepted_bytes;
    result["searchable_vector_bytes"] = stats.searchable_vector_bytes;
    result["accepted_vector_bytes"] = stats.accepted_vector_bytes;
    result["pending_bytes"] = stats.pending_bytes;
    result["allocated_count"] = stats.allocated_count;
    result["tombstone_count"] = stats.tombstone_count;
    result["routing_generation"] = stats.routing_generation;
    result["visibility_watermark"] = stats.visibility_watermark;
    result["durable_watermark"] = stats.durable_watermark;
    result["metadata_epoch"] = stats.metadata_epoch;
    result["lifecycle"] = static_cast<std::uint8_t>(stats.lifecycle);
    return result;
  }

  [[nodiscard]] auto options() const -> py::dict {
    const auto &options = collection_->options();
    py::dict result;
    result["root"] = options.root.string();
    result["dim"] = options.dim;
    result["metric"] = metric_name(options.metric);
    result["dtype"] = scalar_dtype(options.scalar_type);
    result["index_type"] = algorithm_name(options.target_algorithm);
    result["quantization_type"] = quantization_name(options.quantization);
    result["build_threads"] = options.build_threads;
    result["ef_construction"] = options.ef_construction;
    result["implementation_key"] = collection_->target_implementation_key();
    result["engine_factory_key"] = collection_->target_engine_factory_key();
    result["active_algorithm"] = algorithm_name(collection_->active_algorithm());
    result["imported_legacy_layout"] = collection_->imported_legacy_layout();
    return result;
  }

  void close() { throw_status(collection_->close()); }

  [[nodiscard]] auto collection() const -> const std::shared_ptr<Collection> & {
    return collection_;
  }

 private:
  std::shared_ptr<Collection> collection_{};
};

class PyCollectionReadView {
 public:
  explicit PyCollectionReadView(std::shared_ptr<PyCollection> owner) : owner_(std::move(owner)) {}

  [[nodiscard]] auto get_by_id(const std::string &id) -> py::object {
    return owner_->get_by_id(id);
  }
  [[nodiscard]] auto get_by_ids(const py::list &ids) -> py::list { return owner_->get_by_ids(ids); }
  [[nodiscard]] auto records() -> py::list { return owner_->records(); }
  [[nodiscard]] auto search(const py::array &queries, std::uint64_t top_k) -> py::dict {
    return owner_->search(queries, top_k);
  }
  [[nodiscard]] auto batch_search(const py::array &queries, std::uint64_t top_k) -> py::dict {
    return owner_->batch_search(queries, top_k);
  }
  [[nodiscard]] auto stats() const -> py::dict { return owner_->stats(); }
  [[nodiscard]] auto options() const -> py::dict { return owner_->options(); }
  [[nodiscard]] auto mutable_access() const noexcept -> bool { return false; }

 private:
  std::shared_ptr<PyCollection> owner_{};
};

inline void register_collection(py::module_ &module) {
  register_exceptions(module);
  py::class_<PyCollection, std::shared_ptr<PyCollection>>(module, "_Collection")
      .def_static("create",
                  &PyCollection::create,
                  py::arg("root"),
                  py::arg("dim"),
                  py::arg("metric"),
                  py::arg("dtype"),
                  py::arg("index_type"),
                  py::arg("quantization_type"),
                  py::arg("build_threads") = 1,
                  py::arg("ef_construction") = 400)
      .def_static("open", &PyCollection::open, py::arg("root"))
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
           py::arg("retry_token") = "")
      .def("search", &PyCollection::search, py::arg("query"), py::arg("top_k"))
      .def("batch_search", &PyCollection::batch_search, py::arg("queries"), py::arg("top_k"))
      .def("get_by_id", &PyCollection::get_by_id, py::arg("id"))
      .def("get_by_ids", &PyCollection::get_by_ids, py::arg("ids"))
      .def("records", &PyCollection::records)
      .def("checkpoint", &PyCollection::checkpoint)
      .def("stats", &PyCollection::stats)
      .def("options", &PyCollection::options)
      .def("close", &PyCollection::close);

  py::class_<PyCollectionReadView, std::shared_ptr<PyCollectionReadView>>(module,
                                                                          "_CollectionReadView")
      .def(py::init<std::shared_ptr<PyCollection>>(), py::arg("collection"))
      .def("get_by_id", &PyCollectionReadView::get_by_id, py::arg("id"))
      .def("get_by_ids", &PyCollectionReadView::get_by_ids, py::arg("ids"))
      .def("records", &PyCollectionReadView::records)
      .def("search", &PyCollectionReadView::search, py::arg("query"), py::arg("top_k"))
      .def("batch_search",
           &PyCollectionReadView::batch_search,
           py::arg("queries"),
           py::arg("top_k"))
      .def("stats", &PyCollectionReadView::stats)
      .def("options", &PyCollectionReadView::options)
      .def_property_readonly("mutable", &PyCollectionReadView::mutable_access);
}

}  // namespace alaya::python::collection_binding
