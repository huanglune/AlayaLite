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
#include "index/collection/qg_search_extension.hpp"

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

// These response carriers are intentionally private binding types.  The Python
// core consumes named fields so native response shape cannot drift through
// string-keyed dictionaries.
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
  explicit PyCollection(std::shared_ptr<Collection> collection);

  [[nodiscard]] static auto create(const std::string &root,
                                   std::uint32_t dim,
                                   const std::string &metric_value,
                                   const py::dtype &dtype,
                                   const std::string &index_type,
                                   const std::string &quantization_type,
                                   std::uint32_t build_threads,
                                   std::uint32_t max_neighbors,
                                   std::uint32_t ef_construction,
                                   std::uint64_t auto_seal_rows) -> std::shared_ptr<PyCollection>;

  [[nodiscard]] static auto open(const std::string &root, bool read_only = false)
      -> std::shared_ptr<PyCollection>;

  [[nodiscard]] auto mutate(const py::list &ids,
                            const py::list &documents,
                            const py::array &vectors,
                            const py::list &metadata,
                            const std::string &action,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> PyMutationResponse;

  [[nodiscard]] auto mutate_response(const py::list &ids,
                                     const py::list &documents,
                                     const py::array &vectors,
                                     const py::list &metadata,
                                     const std::string &action,
                                     const std::string &mode,
                                     const std::string &durability,
                                     const std::string &retry_token)
      -> CollectionBatchMutationReceipt;

  [[nodiscard]] auto remove(const py::list &ids,
                            const std::string &mode,
                            const std::string &durability,
                            const std::string &retry_token) -> PyMutationResponse;

  [[nodiscard]] auto remove_response(const py::list &ids,
                                     const std::string &mode,
                                     const std::string &durability,
                                     const std::string &retry_token)
      -> CollectionBatchMutationReceipt;

  [[nodiscard]] auto search(const py::array &queries,
                            std::uint64_t top_k,
                            std::uint32_t effort,
                            const py::object &metadata_filter,
                            const std::string &policy,
                            const py::object &selectivity,
                            std::uint64_t scratch_budget_bytes,
                            std::uint64_t io_budget_requests,
                            std::uint64_t io_budget_bytes) -> PySearchResponse;

  [[nodiscard]] auto search_response(const py::array &queries,
                                     std::uint64_t top_k,
                                     std::uint32_t effort,
                                     const py::object &metadata_filter,
                                     const std::string &policy,
                                     const py::object &selectivity,
                                     std::uint64_t scratch_budget_bytes,
                                     std::uint64_t io_budget_requests,
                                     std::uint64_t io_budget_bytes) -> CollectionSearchResponse;

  [[nodiscard]] auto batch_search(const py::array &queries,
                                  std::uint64_t top_k,
                                  std::uint32_t effort,
                                  const py::object &metadata_filter,
                                  const std::string &policy,
                                  const py::object &selectivity,
                                  std::uint64_t scratch_budget_bytes,
                                  std::uint64_t io_budget_requests,
                                  std::uint64_t io_budget_bytes) -> PySearchResponse;

  [[nodiscard]] auto batch_search_response(const py::array &queries,
                                           std::uint64_t top_k,
                                           std::uint32_t effort,
                                           const py::object &metadata_filter,
                                           const std::string &policy,
                                           const py::object &selectivity,
                                           std::uint64_t scratch_budget_bytes,
                                           std::uint64_t io_budget_requests,
                                           std::uint64_t io_budget_bytes)
      -> CollectionSearchResponse;

  [[nodiscard]] auto get_by_id(const std::string &id, bool include_vector = true) -> py::object;
  [[nodiscard]] auto get_by_ids(const py::list &ids, bool include_vector = true) -> py::list;
  [[nodiscard]] auto records() -> std::vector<PyRecordResponse>;
  [[nodiscard]] auto scan(const py::object &metadata_filter, std::size_t limit, bool include_vector)
      -> std::vector<PyRecordResponse>;

  [[nodiscard]] auto checkpoint() -> PyCheckpointResponse;
  [[nodiscard]] auto seal() -> PySealResponse;
  [[nodiscard]] auto compact() -> PyCompactResponse;
  [[nodiscard]] auto gc() -> PyGcResponse;
  [[nodiscard]] auto stats() const -> PyStatsResponse;
  [[nodiscard]] auto options() const -> PyOptionsResponse;
  void close();
  [[nodiscard]] auto read_only() const noexcept -> bool;
  [[nodiscard]] auto collection() const -> const std::shared_ptr<Collection> &;

 private:
  std::shared_ptr<Collection> collection_{};
};

using PyCollectionClass = py::class_<PyCollection, std::shared_ptr<PyCollection>>;

void register_response_types(py::module_ &module);
void register_capabilities(py::module_ &module,
                           bool laser_enabled,
                           std::optional<std::string> laser_simd);
void register_collection_factory(PyCollectionClass &collection);
void register_collection_mutation(PyCollectionClass &collection);
void register_collection_search(PyCollectionClass &collection);
void register_collection_read(PyCollectionClass &collection);
void register_collection_management(PyCollectionClass &collection);

}  // namespace alaya::python::collection_binding
