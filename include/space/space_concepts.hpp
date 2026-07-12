// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "core/value_types.hpp"

namespace alaya {

inline constexpr std::uint32_t kAlignment = 64;

struct EmptyScalarData {};

template <typename DataType, typename DistanceType>
using DistFunc = DistanceType (*)(const DataType *, const DataType *, std::size_t);

template <typename DataType, typename DistanceType>
using DistFuncSQ =
    DistanceType (*)(const std::uint8_t *, const std::uint8_t *, std::size_t, const DataType *, const DataType *);

template <typename DataType, typename DistanceType>
using DistFuncRaBitQ = DistanceType (*)(const DataType *, const DataType *, std::size_t);

// A distance policy owns metric semantics and query preparation. It does not
// own metadata, persistence, mutation, or graph adjacency.
template <typename T>
concept DistanceSpace = requires(const T space,
                                 typename T::IDTypeAlias lhs,
                                 typename T::IDTypeAlias rhs) {
  typename T::DataTypeAlias;
  typename T::DistanceTypeAlias;
  { space.get_dim() } -> std::same_as<std::uint32_t>;
  { space.metric() } -> std::same_as<core::Metric>;
  { space.get_distance(lhs, rhs) } -> std::same_as<typename T::DistanceTypeAlias>;
};

// Quantized and exact spaces both expose a typed query computer. For an exact
// space this is the identity/no-code quantizer. SIMD dispatch remains private
// to the concrete query computer rather than part of this public contract.
template <typename T>
concept Quantizer = requires(const T quantizer, const typename T::DataTypeAlias *query) {
  { quantizer.get_query_computer(query) };
};

// Vector ownership is independent from external IDs. Counts cross this
// boundary as core::RowCount even while legacy implementations retain their
// internal ID-sized counters.
template <typename T>
concept VectorStore = requires(T store,
                               const T const_store,
                               const typename T::DataTypeAlias *data,
                               core::RowCount rows,
                               typename T::IDTypeAlias id) {
  { const_store.get_data_size() } -> std::convertible_to<std::size_t>;
  { const_store.get_capacity() } -> std::convertible_to<core::RowCount>;
  { const_store.get_data_num() } -> std::convertible_to<core::RowCount>;
  { const_store.get_data_by_id(id) };
  { store.fit(data, rows) } -> std::same_as<void>;
};

// GraphStore is intentionally orthogonal to Space. Algorithms that traverse
// adjacency require it explicitly instead of acquiring graph operations via a
// distance policy.
template <typename T>
concept GraphStore = requires(T graph,
                              const T const_graph,
                              typename T::NodeIDTypeAlias node,
                              typename T::EdgeIDType edge) {
  { const_graph.edges(node) } -> std::convertible_to<const typename T::NodeIDTypeAlias *>;
  { graph.edges(node) } -> std::convertible_to<typename T::NodeIDTypeAlias *>;
  { const_graph.at(node, edge) } -> std::same_as<typename T::NodeIDTypeAlias>;
};

// Breaking-step compatibility bridge. Existing graph consumers can keep the
// Space name while its contract is now only distance/quantization + vector
// storage. GraphStore is a separate requirement by design.
template <typename T>
concept Space = DistanceSpace<T> && Quantizer<T> && VectorStore<T>;

}  // namespace alaya
