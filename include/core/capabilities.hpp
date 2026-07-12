// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <concepts>
#include <span>

#include "core/value_types.hpp"

namespace alaya::core {

template <class T>
concept DescriptorProvider = requires(const T &segment) {
  { segment.descriptor() } noexcept -> std::same_as<Descriptor>;
};

template <class T>
concept Searchable = DescriptorProvider<T> &&
                     requires(const T &segment, QueryView query, const SearchOptions &options,
                              SearchSink output) {
                       { segment.search(query, options, output) } -> std::same_as<SearchResult>;
                     };

template <class T>
concept BatchSearchable = Searchable<T> &&
                          requires(const T &segment, QueryBatchView queries,
                                   const SearchOptions &options, SearchSink output) {
                            { segment.batch_search(queries, options, output) }
                                -> std::same_as<BatchSearchResult>;
                          };

template <class T>
concept Mutable = Searchable<T> &&
                  requires(T &segment, VectorBatchView batch,
                           std::span<const ExternalId> ids, MutationContext &context) {
                    { segment.insert(batch, context) } -> std::same_as<MutationResult>;
                    { segment.erase(ids, context) } -> std::same_as<MutationResult>;
                  };

template <class T>
concept Persistable = requires(T &segment, CheckpointContext &context) {
  { segment.checkpoint(context) } -> std::same_as<CheckpointToken>;
};

template <class T>
concept Sealable = Mutable<T> && requires(T &segment, SealContext &context) {
  { segment.seal(context) } -> std::same_as<SealedArtifact>;
};

template <class T>
concept Filterable = Searchable<T> && requires(const T &segment, const void *filter_plan) {
  { segment.supports_filter(filter_plan) } noexcept -> std::same_as<FilterSupport>;
};

template <class T>
concept SegmentBuilder = requires(T &builder, VectorBatchView batch, BuildContext &context) {
  { builder.build(batch, context) } -> std::same_as<SealedArtifact>;
};

}  // namespace alaya::core
