// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

[[nodiscard]] auto PyCollection::mutate(const py::list &ids,
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

[[nodiscard]] auto PyCollection::mutate_response(const py::list &ids,
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

[[nodiscard]] auto PyCollection::remove(const py::list &ids,
                                        const std::string &mode,
                                        const std::string &durability,
                                        const std::string &retry_token) -> PyMutationResponse {
  return batch_receipt_to_response(remove_response(ids, mode, durability, retry_token));
}

[[nodiscard]] auto PyCollection::remove_response(const py::list &ids,
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

}  // namespace alaya::python::collection_binding
