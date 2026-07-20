// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "collection_binding.hpp"

namespace alaya::python::collection_binding {

[[nodiscard]] auto PyCollection::search(const py::array &queries,
                                        std::uint64_t top_k,
                                        std::uint32_t effort,
                                        const py::object &metadata_filter,
                                        const std::string &policy,
                                        const py::object &selectivity,
                                        std::uint64_t scratch_budget_bytes,
                                        std::uint64_t io_budget_requests,
                                        std::uint64_t io_budget_bytes) -> PySearchResponse {
  const auto effective_effort = collection_->target_algorithm() == core::algorithm::qg
                                    ? std::optional<std::uint32_t>(effort)
                                    : std::nullopt;
  return search_response_to_response(search_response(queries,
                                                     top_k,
                                                     effort,
                                                     metadata_filter,
                                                     policy,
                                                     selectivity,
                                                     scratch_budget_bytes,
                                                     io_budget_requests,
                                                     io_budget_bytes),
                                     effective_effort);
}

[[nodiscard]] auto PyCollection::search_response(const py::array &queries,
                                                 std::uint64_t top_k,
                                                 std::uint32_t effort,
                                                 const py::object &metadata_filter,
                                                 const std::string &policy,
                                                 const py::object &selectivity,
                                                 std::uint64_t scratch_budget_bytes,
                                                 std::uint64_t io_budget_requests,
                                                 std::uint64_t io_budget_bytes)
    -> CollectionSearchResponse {
  const auto owned_queries = owned_tensor(queries, collection_->options().dim, false);
  const auto view = owned_queries.view();
  if (view.scalar_type != collection_->options().scalar_type) {
    throw py::type_error("canonical Collection query dtype must match the collection dtype");
  }
  core::SearchOptions options(top_k);
  QgSearchExtension qg_options;
  qg_options.effort = effort;
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

[[nodiscard]] auto PyCollection::batch_search(const py::array &queries,
                                              std::uint64_t top_k,
                                              std::uint32_t effort,
                                              const py::object &metadata_filter,
                                              const std::string &policy,
                                              const py::object &selectivity,
                                              std::uint64_t scratch_budget_bytes,
                                              std::uint64_t io_budget_requests,
                                              std::uint64_t io_budget_bytes) -> PySearchResponse {
  const auto effective_effort = collection_->target_algorithm() == core::algorithm::qg
                                    ? std::optional<std::uint32_t>(effort)
                                    : std::nullopt;
  return search_response_to_response(batch_search_response(queries,
                                                           top_k,
                                                           effort,
                                                           metadata_filter,
                                                           policy,
                                                           selectivity,
                                                           scratch_budget_bytes,
                                                           io_budget_requests,
                                                           io_budget_bytes),
                                     effective_effort);
}

[[nodiscard]] auto PyCollection::batch_search_response(const py::array &queries,
                                                       std::uint64_t top_k,
                                                       std::uint32_t effort,
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
  qg_options.effort = effort;
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

}  // namespace alaya::python::collection_binding
