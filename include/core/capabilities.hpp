// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <concepts>

#include "core/value_types.hpp"

namespace alaya::core {

template <class T>
concept DescriptorProvider = requires(const T &segment) {
  { segment.descriptor() } noexcept -> std::same_as<Descriptor>;
};

template <class T>
concept Searchable =
    DescriptorProvider<T> && requires(const T &segment, const SearchRequest &request) {
      { segment.search(request) } -> std::same_as<Status>;
    };

template <class T>
concept BatchSearchable =
    Searchable<T> && requires(const T &segment, const SearchRequest &request) {
      { segment.batch_search(request) } -> std::same_as<Status>;
    };

template <class T>
concept AsyncSearchable =
    DescriptorProvider<T> && requires(const T &segment, SearchRequest request, void *completion) {
      { segment.start_search(std::move(request), completion) };
    };

template <class T>
concept Mutable = Searchable<T> && requires(T &segment,
                                            const OpaqueOperationRequest &request,
                                            MutationContext &context,
                                            MutationToken &token) {
  { segment.prepare_mutation(request, context, token) } -> std::same_as<Status>;
  { segment.stage_mutation(token, context) } -> std::same_as<Status>;
  { segment.publish_mutation(token, context) } -> std::same_as<Status>;
  { segment.abort_mutation(token, context) } -> std::same_as<Status>;
  { segment.replay_mutation(request, context) } -> std::same_as<Status>;
};

template <class T>
concept Saveable = requires(const T &segment,
                            ArtifactWriter &writer,
                            const SaveOptions &options,
                            ArtifactManifest &manifest) {
  { segment.save(writer, options, manifest) } -> std::same_as<Status>;
};

template <class T>
concept Exportable =
    requires(const T &segment, const OpaqueOperationRequest &request, ExportCursor &cursor) {
      { segment.export_rows(request, cursor) } -> std::same_as<Status>;
    };

template <class T>
concept Checkpointable = requires(T &segment, CheckpointContext &context, CheckpointToken &token) {
  { segment.checkpoint(context, token) } -> std::same_as<Status>;
};

template <class T>
concept Freezable = requires(T &segment, SealContext &context, FreezeToken &token) {
  { segment.freeze_snapshot(context, token) } -> std::same_as<Status>;
};

template <class T>
concept SnapshotExportable =
    Freezable<T> && requires(T &segment, const FreezeToken &token, ExportCursor &cursor) {
      { segment.export_snapshot(token, cursor) } -> std::same_as<Status>;
    };

template <class T>
concept StatsProvider = requires(const T &segment, SegmentStats &stats) {
  { segment.stats(stats) } noexcept -> std::same_as<Status>;
};

template <class T>
concept Closable = requires(T &segment, const Deadline &deadline) {
  { segment.close() } -> std::same_as<Status>;
  { segment.drain(deadline) } -> std::same_as<Status>;
};

template <class T>
concept BuildFactory = requires(T &factory,
                                const OpaqueOperationRequest &request,
                                BuildContext &context,
                                SealedArtifact &artifact) {
  { factory.build(request, context, artifact) } -> std::same_as<Status>;
};

}  // namespace alaya::core
