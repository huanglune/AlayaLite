// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "alaya/collection.hpp"
#include "index/graph/qg/qg_search_extension.hpp"

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
  if (value == "qg") {
    return core::algorithm::qg;
  }
  throw py::value_error("canonical Collection index_type must be flat or qg");
}

[[nodiscard]] inline auto algorithm_name(core::AlgorithmId value) -> std::string {
  switch (value) {
    case core::algorithm::flat:
      return "flat";
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

[[nodiscard]] inline auto scalar_number(const CollectionScalarValue &value)
    -> std::optional<long double> {
  return std::visit(
      [](const auto &item) -> std::optional<long double> {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, std::int64_t> ||
                      std::is_same_v<T, double>) {
          return static_cast<long double>(item);
        }
        return std::nullopt;
      },
      value);
}

[[nodiscard]] inline auto scalar_equal(const CollectionScalarValue &lhs,
                                       const CollectionScalarValue &rhs) -> bool {
  const auto left_number = scalar_number(lhs);
  const auto right_number = scalar_number(rhs);
  if (left_number.has_value() && right_number.has_value()) {
    return *left_number == *right_number;
  }
  return lhs == rhs;
}

[[nodiscard]] inline auto scalar_less(const CollectionScalarValue &lhs,
                                      const CollectionScalarValue &rhs) -> bool {
  const auto left_number = scalar_number(lhs);
  const auto right_number = scalar_number(rhs);
  if (left_number.has_value() && right_number.has_value()) {
    return *left_number < *right_number;
  }
  if (const auto *left = std::get_if<std::string>(&lhs)) {
    if (const auto *right = std::get_if<std::string>(&rhs)) {
      return *left < *right;
    }
  }
  return false;
}

using MetadataPredicate = std::function<bool(const CollectionMetadata &)>;

[[nodiscard]] inline auto compile_metadata_filter(const py::handle &object) -> MetadataPredicate {
  if (!py::isinstance<py::dict>(object)) {
    throw py::type_error("metadata_filter must be a dict");
  }
  const auto dictionary = py::reinterpret_borrow<py::dict>(object);
  std::vector<MetadataPredicate> clauses;
  clauses.reserve(dictionary.size());
  for (const auto &[raw_key, raw_expected] : dictionary) {
    const auto key = py::cast<std::string>(raw_key);
    if (key == "$and" || key == "$or") {
      if (!py::isinstance<py::sequence>(raw_expected) || py::isinstance<py::str>(raw_expected)) {
        throw py::type_error(key + " expects a list of filter expressions");
      }
      std::vector<MetadataPredicate> children;
      for (const auto &child : py::reinterpret_borrow<py::sequence>(raw_expected)) {
        children.push_back(compile_metadata_filter(child));
      }
      clauses.push_back(
          [children = std::move(children), any = key == "$or"](const CollectionMetadata &metadata) {
            if (any) {
              return std::ranges::any_of(children, [&](const auto &child) {
                return child(metadata);
              });
            }
            return std::ranges::all_of(children, [&](const auto &child) {
              return child(metadata);
            });
          });
      continue;
    }

    if (!py::isinstance<py::dict>(raw_expected)) {
      auto expected = metadata_value(raw_expected);
      clauses.push_back([key, expected = std::move(expected)](const CollectionMetadata &metadata) {
        const auto found = metadata.find(key);
        return found != metadata.end() && scalar_equal(found->second, expected);
      });
      continue;
    }

    const auto operations = py::reinterpret_borrow<py::dict>(raw_expected);
    for (const auto &[raw_operation, raw_operand] : operations) {
      const auto operation = py::cast<std::string>(raw_operation);
      if (operation == "$in") {
        if (!py::isinstance<py::sequence>(raw_operand) || py::isinstance<py::str>(raw_operand)) {
          throw py::type_error("$in expects a list-like operand");
        }
        std::vector<CollectionScalarValue> values;
        for (const auto &value : py::reinterpret_borrow<py::sequence>(raw_operand)) {
          values.push_back(metadata_value(value));
        }
        clauses.push_back([key, values = std::move(values)](const CollectionMetadata &metadata) {
          const auto found = metadata.find(key);
          return found != metadata.end() && std::ranges::any_of(values, [&](const auto &value) {
                   return scalar_equal(found->second, value);
                 });
        });
        continue;
      }
      if (operation != "$eq" && operation != "$gt" && operation != "$ge" && operation != "$lt" &&
          operation != "$le") {
        throw py::value_error("Unsupported operator: " + operation);
      }
      auto operand = metadata_value(raw_operand);
      clauses.push_back(
          [key, operation, operand = std::move(operand)](const CollectionMetadata &metadata) {
            const auto found = metadata.find(key);
            if (found == metadata.end()) {
              return false;
            }
            if (operation == "$eq") {
              return scalar_equal(found->second, operand);
            }
            if (operation == "$gt") {
              return scalar_less(operand, found->second);
            }
            if (operation == "$ge") {
              return !scalar_less(found->second, operand);
            }
            if (operation == "$lt") {
              return scalar_less(found->second, operand);
            }
            return !scalar_less(operand, found->second);
          });
    }
  }
  return [clauses = std::move(clauses)](const CollectionMetadata &metadata) {
    return std::ranges::all_of(clauses, [&](const auto &clause) {
      return clause(metadata);
    });
  };
}

[[nodiscard]] inline auto collection_filter(const py::object &expression,
                                            const py::object &selectivity) -> CollectionFilter {
  if (expression.is_none()) {
    return {};
  }
  auto predicate = compile_metadata_filter(expression);
  std::optional<double> estimate;
  if (!selectivity.is_none()) {
    estimate = py::cast<double>(selectivity);
  }
  return CollectionFilter(
      [predicate = std::move(predicate)](const core::LogicalId &,
                                         const CollectionMetadata &metadata,
                                         std::string_view) {
        return predicate(metadata);
      },
      estimate);
}

[[nodiscard]] inline auto filter_policy(std::string_view value) -> core::FilterPolicy {
  if (value == "auto") {
    return core::FilterPolicy::automatic;
  }
  if (value == "strict") {
    return core::FilterPolicy::strict;
  }
  if (value == "allow_partial") {
    return core::FilterPolicy::allow_partial;
  }
  throw py::value_error("filter_policy must be auto, strict, or allow_partial");
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

// Python owns the storage behind py::array.  Keep native operations independent
// of both that owner and concurrent Python-side mutation while the GIL is
// released by copying the validated, C-contiguous tensor into C++ storage.
struct OwnedTensor {
  std::vector<std::byte> storage{};
  core::ScalarType scalar_type{};
  std::uint64_t rows{};
  std::uint32_t dim{};
  std::uint64_t row_stride{};

  [[nodiscard]] auto view() const noexcept -> core::TypedTensorView {
    return {storage.data(), scalar_type, rows, dim, row_stride};
  }
};

[[nodiscard]] inline auto owned_tensor(const py::array &vectors,
                                       std::uint32_t expected_dim,
                                       bool require_two_dimensions = true) -> OwnedTensor {
  const auto borrowed = tensor_view(vectors, expected_dim, require_two_dimensions);
  std::uint64_t row_bytes{};
  std::uint64_t total_bytes{};
  if (!core::checked_multiply(borrowed.dim,
                              core::scalar_type_size(borrowed.scalar_type),
                              row_bytes) ||
      !core::checked_multiply(borrowed.rows, row_bytes, total_bytes) ||
      total_bytes > std::numeric_limits<std::size_t>::max()) {
    throw py::value_error("canonical Collection tensor byte size is not representable");
  }
  OwnedTensor result;
  result.storage.resize(static_cast<std::size_t>(total_bytes));
  if (total_bytes != 0) {
    std::memcpy(result.storage.data(), borrowed.data, static_cast<std::size_t>(total_bytes));
  }
  result.scalar_type = borrowed.scalar_type;
  result.rows = borrowed.rows;
  result.dim = borrowed.dim;
  result.row_stride = row_bytes;
  return result;
}

// These response carriers are intentionally private binding types.  Wave A keeps
// the dictionary-returning methods alive for the existing Python facade while
// giving the v2 core a field-checked path that cannot drift through string keys.
// TODO(sdk-v2 wave C): make these the canonical method responses and remove the
// dictionary compatibility shims in the same cut as the old Python facade.
struct PyRecordResponse {
  py::object id{};
  std::uint64_t upsert_sequence{};
  std::string document{};
  py::dict metadata{};
  py::object vector{};
};

struct PyMutationRowResponse {
  std::uint64_t op_id{};
  std::uint64_t batch_op_id{};
  std::uint64_t row_op_id{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  bool searchable{};
  std::uint8_t durability{};
  std::uint8_t row_status{};
  std::string retry_token{};
};

struct PyMutationResponse {
  std::uint64_t batch_op_id{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  bool searchable{};
  std::uint8_t durability{};
  std::string retry_token{};
  std::vector<PyMutationRowResponse> rows{};
};

struct PySearchStatsResponse {
  bool filter_active{};
  std::string filter_execution{};
  std::uint64_t filter_examined{};
  std::uint64_t filter_passed{};
  std::uint64_t nan_discarded{};
  std::uint64_t overfetch_rounds{};
  std::uint64_t budget_consumed{};
  std::uint64_t lease_acquired{};
  std::uint64_t lease_released{};
  std::uint64_t lease_peak_bytes{};
  std::uint64_t io_requests_consumed{};
  std::uint64_t io_bytes_consumed{};
  std::uint64_t rerank_nanoseconds{};
  std::optional<std::uint32_t> effective_effort{};
};

struct PySearchResponse {
  py::array ids{};
  py::array distances{};
  py::array offsets{};
  py::array valid_counts{};
  py::array status_codes{};
  py::array completeness_codes{};
  std::uint64_t visibility_watermark{};
  std::uint64_t metadata_epoch{};
  PySearchStatsResponse search_stats{};
};

struct PyCheckpointResponse {
  std::uint64_t durable_watermark{};
  std::uint64_t wal_cut{};
  std::uint64_t metadata_epoch{};
  std::string checkpoint_name{};
};

struct PySealResponse {
  std::uint64_t source_segment_id{};
  std::uint64_t successor_segment_id{};
  std::uint64_t sealed_segment_id{};
  std::uint64_t wal_cut{};
  core::RowCount sealed_rows{};
  std::uint64_t sealed_bytes{};
  std::uint64_t manifest_generation{};
};

struct PyCompactResponse {
  std::vector<std::uint64_t> source_segment_ids{};
  std::uint64_t compacted_segment_id{};
  core::RowCount compacted_rows{};
  std::uint64_t input_bytes{};
  std::uint64_t output_bytes{};
  std::uint64_t manifest_generation{};
};

struct PyGcResponse {
  core::RowCount pending{};
  core::RowCount reclaimed{};
  core::RowCount deferred{};
  std::uint64_t reclaimed_bytes{};
  std::uint64_t manifest_generation{};
};

struct PyStatsResponse {
  core::RowCount size{};
  core::RowCount accepted_count{};
  core::RowCount pending_count{};
  std::uint64_t searchable_bytes{};
  std::uint64_t accepted_bytes{};
  std::uint64_t searchable_vector_bytes{};
  std::uint64_t accepted_vector_bytes{};
  std::uint64_t pending_bytes{};
  core::RowCount allocated_count{};
  core::RowCount tombstone_count{};
  std::uint64_t routing_generation{};
  std::uint64_t visibility_watermark{};
  std::uint64_t durable_watermark{};
  std::uint64_t metadata_epoch{};
  core::RowCount sealed_segments_count{};
  core::RowCount gc_pending_count{};
  std::string active_segment_algorithm{};
  std::uint64_t compacted_bytes{};
  std::uint8_t lifecycle{};
};

struct PyOptionsResponse {
  std::string root{};
  bool read_only{};
  std::uint32_t dim{};
  std::string metric{};
  py::dtype dtype{};
  std::string index_type{};
  std::string quantization_type{};
  std::uint32_t build_threads{};
  std::uint32_t max_neighbors{};
  std::uint32_t ef_construction{};
  std::string implementation_key{};
  std::string engine_factory_key{};
  std::string active_algorithm{};
  std::uint64_t auto_seal_rows{};
};

struct PyCapabilitiesResponse {
  std::vector<std::string> index_types{};
  bool laser_enabled{};
  std::optional<std::string> laser_simd{};
};

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

[[nodiscard]] inline auto record_to_response(const CollectionRecord &record) -> PyRecordResponse {
  PyRecordResponse result;
  result.id = logical_id_to_python(record.logical_id);
  result.upsert_sequence = record.upsert_sequence;
  result.document = record.document;
  result.metadata = metadata_to_python(record.metadata);
  result.vector =
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

[[nodiscard]] inline auto receipt_to_response(const CollectionMutationReceipt &receipt)
    -> PyMutationRowResponse {
  return {receipt.op_id,
          receipt.batch_op_id,
          receipt.row_op_id,
          receipt.visibility_watermark,
          receipt.durable_watermark,
          receipt.searchable,
          static_cast<std::uint8_t>(receipt.durability),
          static_cast<std::uint8_t>(receipt.row_status),
          receipt.retry_token};
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

[[nodiscard]] inline auto batch_receipt_to_response(const CollectionBatchMutationReceipt &receipt)
    -> PyMutationResponse {
  PyMutationResponse result;
  result.batch_op_id = receipt.batch_op_id;
  result.visibility_watermark = receipt.visibility_watermark;
  result.durable_watermark = receipt.durable_watermark;
  result.searchable = receipt.searchable;
  result.durability = static_cast<std::uint8_t>(receipt.durability);
  result.retry_token = receipt.retry_token;
  result.rows.reserve(receipt.rows.size());
  for (const auto &row : receipt.rows) {
    result.rows.push_back(receipt_to_response(row));
  }
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

[[nodiscard]] inline auto filter_execution_name(core::FilterExecution execution)
    -> std::string_view {
  switch (execution) {
    case core::FilterExecution::prefilter:
      return "prefilter";
    case core::FilterExecution::traversal:
      return "traversal";
    case core::FilterExecution::postfilter:
      return "postfilter";
  }
  return "postfilter";
}

[[nodiscard]] inline auto search_stats_to_dict(const CollectionSearchStatistics &stats)
    -> py::dict {
  py::dict result;
  result["filter_active"] = stats.filter_active;
  result["filter_execution"] = filter_execution_name(stats.filter_execution);
  result["filter_examined"] = stats.filter_examined;
  result["filter_passed"] = stats.filter_passed;
  result["nan_discarded"] = stats.nan_discarded;
  result["overfetch_rounds"] = stats.overfetch_rounds;
  result["budget_consumed"] = stats.budget_consumed;
  result["lease_acquired"] = stats.lease_acquired;
  result["lease_released"] = stats.lease_released;
  result["lease_peak_bytes"] = stats.lease_peak_bytes;
  result["io_requests_consumed"] = stats.io_requests_consumed;
  result["io_bytes_consumed"] = stats.io_bytes_consumed;
  result["rerank_nanoseconds"] = stats.rerank_nanoseconds;
  return result;
}

[[nodiscard]] inline auto search_stats_to_response(
    const CollectionSearchStatistics &stats,
    std::optional<std::uint32_t> effective_effort = {}) -> PySearchStatsResponse {
  return {stats.filter_active,
          std::string(filter_execution_name(stats.filter_execution)),
          stats.filter_examined,
          stats.filter_passed,
          stats.nan_discarded,
          stats.overfetch_rounds,
          stats.budget_consumed,
          stats.lease_acquired,
          stats.lease_released,
          stats.lease_peak_bytes,
          stats.io_requests_consumed,
          stats.io_bytes_consumed,
          stats.rerank_nanoseconds,
          effective_effort};
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
  result["search_stats"] = search_stats_to_dict(response.search_stats);
  return result;
}

[[nodiscard]] inline auto search_response_to_response(
    const CollectionSearchResponse &response,
    std::optional<std::uint32_t> effective_effort = {}) -> PySearchResponse {
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
  PySearchResponse result;
  result.ids = py::cast<py::array>(std::move(ids));
  result.distances = copy_array(response.distances);
  result.offsets = copy_array(response.offsets);
  result.valid_counts = copy_array(response.valid_counts);
  result.status_codes = copy_array(statuses);
  result.completeness_codes = copy_array(completeness);
  result.visibility_watermark = response.visibility_watermark;
  result.metadata_epoch = response.metadata_epoch;
  result.search_stats = search_stats_to_response(response.search_stats, effective_effort);
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

[[nodiscard]] constexpr auto record_projection(bool include_vector) -> CollectionProjection {
  auto fields = static_cast<std::uint8_t>(CollectionProjection::metadata) |
                static_cast<std::uint8_t>(CollectionProjection::document);
  if (include_vector) {
    fields |= static_cast<std::uint8_t>(CollectionProjection::vector);
  }
  return static_cast<CollectionProjection>(fields);
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
                                   std::uint32_t max_neighbors,
                                   std::uint32_t ef_construction,
                                   std::uint64_t auto_seal_rows) -> std::shared_ptr<PyCollection> {
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

  [[nodiscard]] static auto open(const std::string &root, bool read_only = false)
      -> std::shared_ptr<PyCollection> {
    CollectionOpenOptions options;
    options.read_only = read_only;
    auto collection = [&] {
      py::gil_scoped_release release;
      return unwrap(Collection::open(root, options));
    }();
    return std::make_shared<PyCollection>(std::move(collection));
  }

  [[nodiscard]] auto mutate(const py::list &ids,
                            const py::list &documents,
                            const py::array &vectors,
                            const py::list &metadata,
                            const std::string &action,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> py::dict {
    return batch_receipt_to_dict(
        mutate_response(ids, documents, vectors, metadata, action, mode, durability, retry_token));
  }

  [[nodiscard]] auto mutate_typed(const py::list &ids,
                                  const py::list &documents,
                                  const py::array &vectors,
                                  const py::list &metadata,
                                  const std::string &action,
                                  const std::string &mode,
                                  const std::string &durability,
                                  const std::string &retry_token) -> PyMutationResponse {
    return batch_receipt_to_response(
        mutate_response(ids, documents, vectors, metadata, action, mode, durability, retry_token));
  }

  [[nodiscard]] auto mutate_response(const py::list &ids,
                                     const py::list &documents,
                                     const py::array &vectors,
                                     const py::list &metadata,
                                     const std::string &action,
                                     const std::string &mode,
                                     const std::string &durability,
                                     const std::string &retry_token)
      -> CollectionBatchMutationReceipt {
    const auto owned_vectors = owned_tensor(vectors, collection_->options().dim);
    const auto view = owned_vectors.view();
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
    const auto native_mode = batch_mode(mode);
    const auto native_options = write_options(durability, retry_token);
    return [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->mutate_batch(native, native_mode, native_options));
    }();
  }

  [[nodiscard]] auto remove(const py::list &ids,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> py::dict {
    return batch_receipt_to_dict(remove_response(ids, mode, durability, retry_token));
  }

  [[nodiscard]] auto remove_typed(const py::list &ids,
                                  const std::string &mode,
                                  const std::string &durability,
                                  const std::string &retry_token) -> PyMutationResponse {
    return batch_receipt_to_response(remove_response(ids, mode, durability, retry_token));
  }

  [[nodiscard]] auto remove_response(const py::list &ids,
                                     const std::string &mode,
                                     const std::string &durability,
                                     const std::string &retry_token)
      -> CollectionBatchMutationReceipt {
    std::vector<CollectionBatchRow> native;
    native.reserve(ids.size());
    for (const auto &id : ids) {
      CollectionBatchRow row;
      row.action = CollectionMutationAction::remove;
      row.logical_id = logical_id(py::cast<std::string>(id));
      native.push_back(std::move(row));
    }
    const auto native_mode = batch_mode(mode);
    const auto native_options = write_options(durability, retry_token);
    return [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->mutate_batch(native, native_mode, native_options));
    }();
  }

  [[nodiscard]] auto search(const py::array &queries,
                            std::uint64_t top_k,
                            std::uint32_t ef_search,
                            const py::object &metadata_filter,
                            const std::string &policy,
                            const py::object &selectivity,
                            std::uint64_t scratch_budget_bytes,
                            std::uint64_t io_budget_requests,
                            std::uint64_t io_budget_bytes) -> py::dict {
    return search_response_to_dict(search_response(queries,
                                                   top_k,
                                                   ef_search,
                                                   metadata_filter,
                                                   policy,
                                                   selectivity,
                                                   scratch_budget_bytes,
                                                   io_budget_requests,
                                                   io_budget_bytes));
  }

  [[nodiscard]] auto search_typed(const py::array &queries,
                                  std::uint64_t top_k,
                                  std::uint32_t ef_search,
                                  const py::object &metadata_filter,
                                  const std::string &policy,
                                  const py::object &selectivity,
                                  std::uint64_t scratch_budget_bytes,
                                  std::uint64_t io_budget_requests,
                                  std::uint64_t io_budget_bytes) -> PySearchResponse {
    const auto effective_effort = collection_->target_algorithm() == core::algorithm::qg
                                      ? std::optional<std::uint32_t>(ef_search)
                                      : std::nullopt;
    return search_response_to_response(search_response(queries,
                                                       top_k,
                                                       ef_search,
                                                       metadata_filter,
                                                       policy,
                                                       selectivity,
                                                       scratch_budget_bytes,
                                                       io_budget_requests,
                                                       io_budget_bytes),
                                       effective_effort);
  }

  [[nodiscard]] auto search_response(const py::array &queries,
                                     std::uint64_t top_k,
                                     std::uint32_t ef_search,
                                     const py::object &metadata_filter,
                                     const std::string &policy,
                                     const py::object &selectivity,
                                     std::uint64_t scratch_budget_bytes,
                                     std::uint64_t io_budget_requests,
                                     std::uint64_t io_budget_bytes) -> CollectionSearchResponse {
    const auto owned_queries = owned_tensor(queries, collection_->options().dim, false);
    const auto view = owned_queries.view();
    if (view.scalar_type != collection_->options().scalar_type) {
      throw py::type_error("canonical Collection query dtype must match the collection dtype");
    }
    core::SearchOptions options(top_k);
    QgSearchExtension qg_options;
    qg_options.effort = ef_search;
    const auto qg_extension = make_qg_search_extension(qg_options);
    options.extensions = std::span<const core::AlgorithmSearchExtension>(&qg_extension, 1);
    options.filter_policy = filter_policy(policy);
    core::SearchContext context;
    context.query_scratch_lease.available_bytes = scratch_budget_bytes;
    context.io_credits.available_requests = io_budget_requests;
    context.io_credits.available_bytes = io_budget_bytes;
    const auto filter = collection_filter(metadata_filter, selectivity);
    if (queries.ndim() == 1) {
      auto response = [&] {
        py::gil_scoped_release release;
        return unwrap(collection_->search(view, options, context, filter));
      }();
      return response;
    }
    auto response = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->batch_search(view, options, context, filter));
    }();
    return response;
  }

  [[nodiscard]] auto batch_search(const py::array &queries,
                                  std::uint64_t top_k,
                                  std::uint32_t ef_search,
                                  const py::object &metadata_filter,
                                  const std::string &policy,
                                  const py::object &selectivity,
                                  std::uint64_t scratch_budget_bytes,
                                  std::uint64_t io_budget_requests,
                                  std::uint64_t io_budget_bytes) -> py::dict {
    return search_response_to_dict(batch_search_response(queries,
                                                         top_k,
                                                         ef_search,
                                                         metadata_filter,
                                                         policy,
                                                         selectivity,
                                                         scratch_budget_bytes,
                                                         io_budget_requests,
                                                         io_budget_bytes));
  }

  [[nodiscard]] auto batch_search_typed(const py::array &queries,
                                        std::uint64_t top_k,
                                        std::uint32_t ef_search,
                                        const py::object &metadata_filter,
                                        const std::string &policy,
                                        const py::object &selectivity,
                                        std::uint64_t scratch_budget_bytes,
                                        std::uint64_t io_budget_requests,
                                        std::uint64_t io_budget_bytes) -> PySearchResponse {
    const auto effective_effort = collection_->target_algorithm() == core::algorithm::qg
                                      ? std::optional<std::uint32_t>(ef_search)
                                      : std::nullopt;
    return search_response_to_response(batch_search_response(queries,
                                                             top_k,
                                                             ef_search,
                                                             metadata_filter,
                                                             policy,
                                                             selectivity,
                                                             scratch_budget_bytes,
                                                             io_budget_requests,
                                                             io_budget_bytes),
                                       effective_effort);
  }

  [[nodiscard]] auto batch_search_response(const py::array &queries,
                                           std::uint64_t top_k,
                                           std::uint32_t ef_search,
                                           const py::object &metadata_filter,
                                           const std::string &policy,
                                           const py::object &selectivity,
                                           std::uint64_t scratch_budget_bytes,
                                           std::uint64_t io_budget_requests,
                                           std::uint64_t io_budget_bytes)
      -> CollectionSearchResponse {
    const auto owned_queries = owned_tensor(queries, collection_->options().dim);
    const auto view = owned_queries.view();
    if (view.scalar_type != collection_->options().scalar_type) {
      throw py::type_error("canonical Collection query dtype must match the collection dtype");
    }
    core::SearchOptions options(top_k);
    QgSearchExtension qg_options;
    qg_options.effort = ef_search;
    const auto qg_extension = make_qg_search_extension(qg_options);
    options.extensions = std::span<const core::AlgorithmSearchExtension>(&qg_extension, 1);
    options.filter_policy = filter_policy(policy);
    core::SearchContext context;
    context.query_scratch_lease.available_bytes = scratch_budget_bytes;
    context.io_credits.available_requests = io_budget_requests;
    context.io_credits.available_bytes = io_budget_bytes;
    const auto filter = collection_filter(metadata_filter, selectivity);
    return [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->batch_search(view, options, context, filter));
    }();
  }

  [[nodiscard]] auto get_by_id(const std::string &id) -> py::object {
    auto record = [&] {
      py::gil_scoped_release release;
      return collection_->get_by_id(logical_id(id));
    }();
    if (!record.ok() && record.status().code() == core::StatusCode::not_found) {
      return py::none();
    }
    return record_to_dict(unwrap(std::move(record)));
  }

  [[nodiscard]] auto get_by_id_typed(const std::string &id, bool include_vector = true)
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

  [[nodiscard]] auto get_by_ids(const py::list &ids) -> py::list {
    py::list result;
    for (const auto &id : ids) {
      result.append(get_by_id(py::cast<std::string>(id)));
    }
    return result;
  }

  [[nodiscard]] auto get_by_ids_typed(const py::list &ids, bool include_vector = true) -> py::list {
    py::list result;
    for (const auto &id : ids) {
      result.append(get_by_id_typed(py::cast<std::string>(id), include_vector));
    }
    return result;
  }

  [[nodiscard]] auto records() -> py::list {
    auto records = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->records());
    }();
    py::list result;
    for (const auto &record : records) {
      result.append(record_to_dict(record));
    }
    return result;
  }

  [[nodiscard]] auto records_typed() -> std::vector<PyRecordResponse> {
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

  [[nodiscard]] auto scan(const py::object &metadata_filter, std::size_t limit, bool include_vector)
      -> std::vector<PyRecordResponse> {
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

  [[nodiscard]] auto checkpoint() -> py::dict {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->checkpoint());
    }();
    py::dict result;
    result["durable_watermark"] = receipt.durable_watermark;
    result["wal_cut"] = receipt.wal_cut;
    result["metadata_epoch"] = receipt.metadata_epoch;
    result["checkpoint_name"] = receipt.checkpoint_name;
    return result;
  }

  [[nodiscard]] auto checkpoint_typed() -> PyCheckpointResponse {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->checkpoint());
    }();
    return {receipt.durable_watermark,
            receipt.wal_cut,
            receipt.metadata_epoch,
            receipt.checkpoint_name};
  }

  [[nodiscard]] auto seal() -> py::dict {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->seal());
    }();
    py::dict result;
    result["source_segment_id"] = receipt.source_segment_id;
    result["successor_segment_id"] = receipt.successor_segment_id;
    result["sealed_segment_id"] = receipt.sealed_segment_id;
    result["wal_cut"] = receipt.wal_cut;
    result["sealed_rows"] = receipt.sealed_rows;
    result["sealed_bytes"] = receipt.sealed_bytes;
    result["manifest_generation"] = receipt.manifest_generation;
    return result;
  }

  [[nodiscard]] auto seal_typed() -> PySealResponse {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->seal());
    }();
    return {receipt.source_segment_id,
            receipt.successor_segment_id,
            receipt.sealed_segment_id,
            receipt.wal_cut,
            receipt.sealed_rows,
            receipt.sealed_bytes,
            receipt.manifest_generation};
  }

  [[nodiscard]] auto compact() -> py::dict {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->compact());
    }();
    py::dict result;
    result["source_segment_ids"] = receipt.source_segment_ids;
    result["compacted_segment_id"] = receipt.compacted_segment_id;
    result["compacted_rows"] = receipt.compacted_rows;
    result["input_bytes"] = receipt.input_bytes;
    result["output_bytes"] = receipt.output_bytes;
    result["manifest_generation"] = receipt.manifest_generation;
    return result;
  }

  [[nodiscard]] auto compact_typed() -> PyCompactResponse {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->compact());
    }();
    return {receipt.source_segment_ids,
            receipt.compacted_segment_id,
            receipt.compacted_rows,
            receipt.input_bytes,
            receipt.output_bytes,
            receipt.manifest_generation};
  }

  [[nodiscard]] auto gc() -> py::dict {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->gc());
    }();
    py::dict result;
    result["pending"] = receipt.pending;
    result["reclaimed"] = receipt.reclaimed;
    result["deferred"] = receipt.deferred;
    result["reclaimed_bytes"] = receipt.reclaimed_bytes;
    result["manifest_generation"] = receipt.manifest_generation;
    return result;
  }

  [[nodiscard]] auto gc_typed() -> PyGcResponse {
    const auto receipt = [&] {
      py::gil_scoped_release release;
      return unwrap(collection_->gc());
    }();
    return {receipt.pending,
            receipt.reclaimed,
            receipt.deferred,
            receipt.reclaimed_bytes,
            receipt.manifest_generation};
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
    result["sealed_segments_count"] = stats.sealed_segments_count;
    result["gc_pending_count"] = stats.gc_pending_count;
    result["active_segment_algorithm"] = algorithm_name(stats.active_segment_algorithm);
    result["compacted_bytes"] = stats.compacted_bytes;
    result["lifecycle"] = static_cast<std::uint8_t>(stats.lifecycle);
    return result;
  }

  [[nodiscard]] auto stats_typed() const -> PyStatsResponse {
    const auto stats = collection_->stats();
    return {stats.size,
            stats.accepted_count,
            stats.pending_count,
            stats.searchable_bytes,
            stats.accepted_bytes,
            stats.searchable_vector_bytes,
            stats.accepted_vector_bytes,
            stats.pending_bytes,
            stats.allocated_count,
            stats.tombstone_count,
            stats.routing_generation,
            stats.visibility_watermark,
            stats.durable_watermark,
            stats.metadata_epoch,
            stats.sealed_segments_count,
            stats.gc_pending_count,
            algorithm_name(stats.active_segment_algorithm),
            stats.compacted_bytes,
            static_cast<std::uint8_t>(stats.lifecycle)};
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
    result["max_neighbors"] = options.max_neighbors;
    result["ef_construction"] = options.ef_construction;
    result["implementation_key"] = collection_->target_implementation_key();
    result["engine_factory_key"] = collection_->target_engine_factory_key();
    result["active_algorithm"] = algorithm_name(collection_->active_algorithm());
    result["auto_seal_rows"] = options.auto_seal_rows;
    return result;
  }

  [[nodiscard]] auto options_typed() const -> PyOptionsResponse {
    const auto &options = collection_->options();
    return {options.root.string(),
            collection_->read_only(),
            options.dim,
            metric_name(options.metric),
            scalar_dtype(options.scalar_type),
            algorithm_name(options.target_algorithm),
            quantization_name(options.quantization),
            options.build_threads,
            options.max_neighbors,
            options.ef_construction,
            std::string(collection_->target_implementation_key()),
            std::string(collection_->target_engine_factory_key()),
            algorithm_name(collection_->active_algorithm()),
            options.auto_seal_rows};
  }

  void close() {
    py::gil_scoped_release release;
    throw_status(collection_->close());
  }

  [[nodiscard]] auto read_only() const noexcept -> bool { return collection_->read_only(); }

  [[nodiscard]] auto collection() const -> const std::shared_ptr<Collection> & {
    return collection_;
  }

 private:
  std::shared_ptr<Collection> collection_{};
};

inline void register_response_types(py::module_ &module) {
  py::class_<PyRecordResponse>(module, "_RecordResponse")
      .def_readonly("id", &PyRecordResponse::id)
      .def_readonly("upsert_sequence", &PyRecordResponse::upsert_sequence)
      .def_readonly("document", &PyRecordResponse::document)
      .def_readonly("metadata", &PyRecordResponse::metadata)
      .def_readonly("vector", &PyRecordResponse::vector);

  py::class_<PyMutationRowResponse>(module, "_MutationRowResponse")
      .def_readonly("op_id", &PyMutationRowResponse::op_id)
      .def_readonly("batch_op_id", &PyMutationRowResponse::batch_op_id)
      .def_readonly("row_op_id", &PyMutationRowResponse::row_op_id)
      .def_readonly("visibility_watermark", &PyMutationRowResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyMutationRowResponse::durable_watermark)
      .def_readonly("searchable", &PyMutationRowResponse::searchable)
      .def_readonly("durability", &PyMutationRowResponse::durability)
      .def_readonly("row_status", &PyMutationRowResponse::row_status)
      .def_readonly("retry_token", &PyMutationRowResponse::retry_token);

  py::class_<PyMutationResponse>(module, "_MutationResponse")
      .def_readonly("batch_op_id", &PyMutationResponse::batch_op_id)
      .def_readonly("visibility_watermark", &PyMutationResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyMutationResponse::durable_watermark)
      .def_readonly("searchable", &PyMutationResponse::searchable)
      .def_readonly("durability", &PyMutationResponse::durability)
      .def_readonly("retry_token", &PyMutationResponse::retry_token)
      .def_readonly("rows", &PyMutationResponse::rows);

  py::class_<PySearchStatsResponse>(module, "_SearchStatsResponse")
      .def_readonly("filter_active", &PySearchStatsResponse::filter_active)
      .def_readonly("filter_execution", &PySearchStatsResponse::filter_execution)
      .def_readonly("filter_examined", &PySearchStatsResponse::filter_examined)
      .def_readonly("filter_passed", &PySearchStatsResponse::filter_passed)
      .def_readonly("nan_discarded", &PySearchStatsResponse::nan_discarded)
      .def_readonly("overfetch_rounds", &PySearchStatsResponse::overfetch_rounds)
      .def_readonly("budget_consumed", &PySearchStatsResponse::budget_consumed)
      .def_readonly("lease_acquired", &PySearchStatsResponse::lease_acquired)
      .def_readonly("lease_released", &PySearchStatsResponse::lease_released)
      .def_readonly("lease_peak_bytes", &PySearchStatsResponse::lease_peak_bytes)
      .def_readonly("io_requests_consumed", &PySearchStatsResponse::io_requests_consumed)
      .def_readonly("io_bytes_consumed", &PySearchStatsResponse::io_bytes_consumed)
      .def_readonly("rerank_nanoseconds", &PySearchStatsResponse::rerank_nanoseconds)
      .def_readonly("effective_effort", &PySearchStatsResponse::effective_effort);

  py::class_<PySearchResponse>(module, "_SearchResponse")
      .def_readonly("ids", &PySearchResponse::ids)
      .def_readonly("distances", &PySearchResponse::distances)
      .def_readonly("offsets", &PySearchResponse::offsets)
      .def_readonly("valid_counts", &PySearchResponse::valid_counts)
      .def_readonly("status_codes", &PySearchResponse::status_codes)
      .def_readonly("completeness_codes", &PySearchResponse::completeness_codes)
      .def_readonly("visibility_watermark", &PySearchResponse::visibility_watermark)
      .def_readonly("metadata_epoch", &PySearchResponse::metadata_epoch)
      .def_readonly("search_stats", &PySearchResponse::search_stats);

  py::class_<PyCheckpointResponse>(module, "_CheckpointResponse")
      .def_readonly("durable_watermark", &PyCheckpointResponse::durable_watermark)
      .def_readonly("wal_cut", &PyCheckpointResponse::wal_cut)
      .def_readonly("metadata_epoch", &PyCheckpointResponse::metadata_epoch)
      .def_readonly("checkpoint_name", &PyCheckpointResponse::checkpoint_name);

  py::class_<PySealResponse>(module, "_SealResponse")
      .def_readonly("source_segment_id", &PySealResponse::source_segment_id)
      .def_readonly("successor_segment_id", &PySealResponse::successor_segment_id)
      .def_readonly("sealed_segment_id", &PySealResponse::sealed_segment_id)
      .def_readonly("wal_cut", &PySealResponse::wal_cut)
      .def_readonly("sealed_rows", &PySealResponse::sealed_rows)
      .def_readonly("sealed_bytes", &PySealResponse::sealed_bytes)
      .def_readonly("manifest_generation", &PySealResponse::manifest_generation);

  py::class_<PyCompactResponse>(module, "_CompactResponse")
      .def_readonly("source_segment_ids", &PyCompactResponse::source_segment_ids)
      .def_readonly("compacted_segment_id", &PyCompactResponse::compacted_segment_id)
      .def_readonly("compacted_rows", &PyCompactResponse::compacted_rows)
      .def_readonly("input_bytes", &PyCompactResponse::input_bytes)
      .def_readonly("output_bytes", &PyCompactResponse::output_bytes)
      .def_readonly("manifest_generation", &PyCompactResponse::manifest_generation);

  py::class_<PyGcResponse>(module, "_GcResponse")
      .def_readonly("pending", &PyGcResponse::pending)
      .def_readonly("reclaimed", &PyGcResponse::reclaimed)
      .def_readonly("deferred", &PyGcResponse::deferred)
      .def_readonly("reclaimed_bytes", &PyGcResponse::reclaimed_bytes)
      .def_readonly("manifest_generation", &PyGcResponse::manifest_generation);

  py::class_<PyStatsResponse>(module, "_StatsResponse")
      .def_readonly("size", &PyStatsResponse::size)
      .def_readonly("accepted_count", &PyStatsResponse::accepted_count)
      .def_readonly("pending_count", &PyStatsResponse::pending_count)
      .def_readonly("searchable_bytes", &PyStatsResponse::searchable_bytes)
      .def_readonly("accepted_bytes", &PyStatsResponse::accepted_bytes)
      .def_readonly("searchable_vector_bytes", &PyStatsResponse::searchable_vector_bytes)
      .def_readonly("accepted_vector_bytes", &PyStatsResponse::accepted_vector_bytes)
      .def_readonly("pending_bytes", &PyStatsResponse::pending_bytes)
      .def_readonly("allocated_count", &PyStatsResponse::allocated_count)
      .def_readonly("tombstone_count", &PyStatsResponse::tombstone_count)
      .def_readonly("routing_generation", &PyStatsResponse::routing_generation)
      .def_readonly("visibility_watermark", &PyStatsResponse::visibility_watermark)
      .def_readonly("durable_watermark", &PyStatsResponse::durable_watermark)
      .def_readonly("metadata_epoch", &PyStatsResponse::metadata_epoch)
      .def_readonly("sealed_segments_count", &PyStatsResponse::sealed_segments_count)
      .def_readonly("gc_pending_count", &PyStatsResponse::gc_pending_count)
      .def_readonly("active_segment_algorithm", &PyStatsResponse::active_segment_algorithm)
      .def_readonly("compacted_bytes", &PyStatsResponse::compacted_bytes)
      .def_readonly("lifecycle", &PyStatsResponse::lifecycle);

  py::class_<PyOptionsResponse>(module, "_OptionsResponse")
      .def_readonly("root", &PyOptionsResponse::root)
      .def_readonly("read_only", &PyOptionsResponse::read_only)
      .def_readonly("dim", &PyOptionsResponse::dim)
      .def_readonly("metric", &PyOptionsResponse::metric)
      .def_readonly("dtype", &PyOptionsResponse::dtype)
      .def_readonly("index_type", &PyOptionsResponse::index_type)
      .def_readonly("quantization_type", &PyOptionsResponse::quantization_type)
      .def_readonly("build_threads", &PyOptionsResponse::build_threads)
      .def_readonly("max_neighbors", &PyOptionsResponse::max_neighbors)
      .def_readonly("ef_construction", &PyOptionsResponse::ef_construction)
      .def_readonly("implementation_key", &PyOptionsResponse::implementation_key)
      .def_readonly("engine_factory_key", &PyOptionsResponse::engine_factory_key)
      .def_readonly("active_algorithm", &PyOptionsResponse::active_algorithm)
      .def_readonly("auto_seal_rows", &PyOptionsResponse::auto_seal_rows);

  py::class_<PyCapabilitiesResponse>(module, "_CapabilitiesResponse")
      .def_readonly("index_types", &PyCapabilitiesResponse::index_types)
      .def_readonly("laser_enabled", &PyCapabilitiesResponse::laser_enabled)
      .def_readonly("laser_simd", &PyCapabilitiesResponse::laser_simd);
}

inline void register_capabilities(py::module_ &module,
                                  bool laser_enabled,
                                  std::optional<std::string> laser_simd) {
  module.def("capabilities", [laser_enabled, laser_simd = std::move(laser_simd)] {
    std::vector<std::string> index_types{"flat"};
    if (laser_enabled) {
      index_types.emplace_back("qg");
    }
    return PyCapabilitiesResponse{std::move(index_types), laser_enabled, laser_simd};
  });
}

inline void register_collection(py::module_ &module) {
  register_exceptions(module);
  register_response_types(module);
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
                  py::arg("max_neighbors") = 32,
                  py::arg("ef_construction") = 400,
                  py::arg("auto_seal_rows") = 0)
      .def_static("open", &PyCollection::open, py::arg("root"), py::arg("read_only") = false)
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
      .def("mutate_typed",
           &PyCollection::mutate_typed,
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
      .def("remove_typed",
           &PyCollection::remove_typed,
           py::arg("ids"),
           py::kw_only(),
           py::arg("mode") = "per_row_independent",
           py::arg("durability") = "wal_fsync",
           py::arg("retry_token") = "")
      .def("search",
           &PyCollection::search,
           py::arg("query"),
           py::arg("top_k"),
           py::kw_only(),
           py::arg("ef_search") = 100,
           py::arg("metadata_filter") = py::none(),
           py::arg("filter_policy") = "auto",
           py::arg("filter_selectivity") = py::none(),
           py::arg("scratch_budget_bytes") = core::kUnlimitedResource,
           py::arg("io_budget_requests") = core::kUnlimitedResource,
           py::arg("io_budget_bytes") = core::kUnlimitedResource)
      .def("search_typed",
           &PyCollection::search_typed,
           py::arg("query"),
           py::arg("top_k"),
           py::kw_only(),
           py::arg("ef_search") = 100,
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
           py::arg("ef_search") = 100,
           py::arg("metadata_filter") = py::none(),
           py::arg("filter_policy") = "auto",
           py::arg("filter_selectivity") = py::none(),
           py::arg("scratch_budget_bytes") = core::kUnlimitedResource,
           py::arg("io_budget_requests") = core::kUnlimitedResource,
           py::arg("io_budget_bytes") = core::kUnlimitedResource)
      .def("batch_search_typed",
           &PyCollection::batch_search_typed,
           py::arg("queries"),
           py::arg("top_k"),
           py::kw_only(),
           py::arg("ef_search") = 100,
           py::arg("metadata_filter") = py::none(),
           py::arg("filter_policy") = "auto",
           py::arg("filter_selectivity") = py::none(),
           py::arg("scratch_budget_bytes") = core::kUnlimitedResource,
           py::arg("io_budget_requests") = core::kUnlimitedResource,
           py::arg("io_budget_bytes") = core::kUnlimitedResource)
      .def("get_by_id", &PyCollection::get_by_id, py::arg("id"))
      .def("get_by_id_typed",
           &PyCollection::get_by_id_typed,
           py::arg("id"),
           py::kw_only(),
           py::arg("include_vector") = true)
      .def("get_by_ids", &PyCollection::get_by_ids, py::arg("ids"))
      .def("get_by_ids_typed",
           &PyCollection::get_by_ids_typed,
           py::arg("ids"),
           py::kw_only(),
           py::arg("include_vector") = true)
      .def("records", &PyCollection::records)
      .def("records_typed", &PyCollection::records_typed)
      .def("scan",
           &PyCollection::scan,
           py::kw_only(),
           py::arg("metadata_filter") = py::none(),
           py::arg("limit") = 100,
           py::arg("include_vector") = false)
      .def("checkpoint", &PyCollection::checkpoint)
      .def("checkpoint_typed", &PyCollection::checkpoint_typed)
      .def("seal", &PyCollection::seal)
      .def("seal_typed", &PyCollection::seal_typed)
      .def("compact", &PyCollection::compact)
      .def("compact_typed", &PyCollection::compact_typed)
      .def("gc", &PyCollection::gc)
      .def("gc_typed", &PyCollection::gc_typed)
      .def("stats", &PyCollection::stats)
      .def("stats_typed", &PyCollection::stats_typed)
      .def("options", &PyCollection::options)
      .def("options_typed", &PyCollection::options_typed)
      .def_property_readonly("read_only", &PyCollection::read_only)
      .def("close", &PyCollection::close);
}

}  // namespace alaya::python::collection_binding
