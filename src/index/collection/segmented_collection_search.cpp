// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/collection/qg_search_extension.hpp"
#include "index/collection/segmented_collection.hpp"
#include "index/disk/laser_segment.hpp"

namespace alaya::internal::collection {

[[nodiscard]] auto SegmentedCollection::search_at_snapshot(const RoutingSnapshotPtr &snapshot,
                                                           const CollectionSearchRequest &request,
                                                           bool prefer_exact)
    -> core::Result<CollectionSearchResult> {
  if (snapshot == nullptr || request.context == nullptr ||
      !core::is_current_struct(*request.context) || !core::is_current_struct(request.options) ||
      (request.stats != nullptr && !core::is_current_struct(*request.stats))) {
    return core::Status::error(core::StatusCode::invalid_argument,
                               core::OperationStage::validation,
                               core::StatusDetail::malformed_struct,
                               "collection search request is incomplete or incompatible");
  }
  if (request.stats != nullptr) {
    *request.stats = CollectionSearchStats{};
    request.stats->filter_active = request.filter.active();
  }
  auto status =
      core::validate_tensor(request.queries, schema_.dim, core::OperationStage::validation);
  if (!status.ok()) {
    return status;
  }
  if (request.queries.scalar_type != schema_.scalar_type) {
    return core::Status::error(core::StatusCode::not_supported,
                               core::OperationStage::validation,
                               core::StatusDetail::unsupported_scalar_type,
                               "collection does not implicitly convert query scalar types");
  }
  status = core::validate_runtime_control(request.context->deadline,
                                          request.context->cancellation,
                                          core::OperationStage::search);
  if (!status.ok()) {
    return status;
  }

  CollectionSearchResult result;
  result.visibility_watermark = snapshot->visibility_watermark;
  result.metadata_epoch = snapshot->metadata_epoch;
  result.queries.resize(static_cast<std::size_t>(request.queries.rows));
  if (request.options.top_k == 0 || request.queries.rows == 0) {
    for (auto &query : result.queries) {
      query.status = core::Status::success();
      query.completeness = core::SearchCompleteness::complete_k;
    }
    return result;
  }

  auto selected = select_filter_execution(*snapshot, request, prefer_exact);
  if (!selected.ok()) {
    return selected.status();
  }
  auto execution = std::move(selected).value();
  const auto all_vectors_owned =
      std::all_of(snapshot->versions.begin(), snapshot->versions.end(), [](const auto &item) {
        return item.second.state != VersionState::live || item.second.payload.vector.has_value();
      });
  if (execution == core::FilterExecution::prefilter && !all_vectors_owned) {
    if (request.options.filter_policy == core::FilterPolicy::strict) {
      return search_budget_denied("strict filtered search requires exact fallback vectors");
    }
    if (prefer_exact) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::search,
                                 core::StatusDetail::operation_slot_absent,
                                 "exact hybrid search requires readable live vectors");
    }
    execution = core::FilterExecution::traversal;
  }
  if (request.stats != nullptr) {
    request.stats->filter_execution = execution;
  }

  auto budget = preflight_search_budget(*snapshot, request);
  if (!budget.ok()) {
    return budget.status();
  }
  if (request.stats != nullptr) {
    request.stats->budget_consumed = budget.value().scratch_bytes;
  }
  SearchLeaseGuard lease(this, budget.value().scratch_bytes, request.stats);

  if (execution == core::FilterExecution::prefilter) {
    return exact_search(snapshot, request);
  }
  return fanout_search(snapshot, request, execution);
}

[[nodiscard]] auto SegmentedCollection::exact_search(const RoutingSnapshotPtr &snapshot,
                                                     const CollectionSearchRequest &request)
    -> core::Result<CollectionSearchResult> {
  CollectionSearchResult result;
  result.visibility_watermark = snapshot->visibility_watermark;
  result.metadata_epoch = snapshot->metadata_epoch;
  result.queries.resize(static_cast<std::size_t>(request.queries.rows));
  for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
    auto control = core::validate_runtime_control(request.context->deadline,
                                                  request.context->cancellation,
                                                  core::OperationStage::search);
    if (!control.ok()) {
      return control;
    }
    auto &query_result = result.queries[static_cast<std::size_t>(query_index)];
    for (const auto &[logical_id, version] : snapshot->versions) {
      if (version.state != VersionState::live ||
          version.upsert_sequence > snapshot->visibility_watermark) {
        continue;
      }
      if (request.filter.active()) {
        if (request.stats != nullptr) {
          ++request.stats->filter_examined;
        }
        if (!request.filter.matches(logical_id,
                                    version.payload.metadata,
                                    version.payload.document)) {
          continue;
        }
        if (request.stats != nullptr) {
          ++request.stats->filter_passed;
        }
      }
      if (!version.payload.vector.has_value()) {
        return search_budget_denied("exact fallback cannot read a live row vector");
      }
      auto score = exact_distance(query_row(request.queries, query_index),
                                  *version.payload.vector,
                                  schema_.metric);
      if (!score.ok()) {
        return score.status();
      }
      if (is_nan_score(score.value())) {
        if (request.stats != nullptr) {
          ++request.stats->nan_discarded;
        }
        continue;
      }
      auto flags = core::ResultFlag::exact_reranked | core::ResultFlag::version_checked;
      if (request.filter.active()) {
        flags = flags | core::ResultFlag::filtered;
      }
      query_result.hits.push_back(CollectionHit{logical_id,
                                                std::move(score).value(),
                                                core::ScoreKind::distance,
                                                schema_.metric,
                                                flags,
                                                version.upsert_sequence,
                                                version.address});
      if (request.context->stats != nullptr) {
        ++request.context->stats->rerank_count;
        if (request.filter.active()) {
          ++request.context->stats->filter_candidates;
        }
      }
    }
    sort_hits(query_result.hits);
    if (query_result.hits.size() > request.options.top_k) {
      query_result.hits.resize(static_cast<std::size_t>(request.options.top_k));
    }
    query_result.status = core::Status::success();
    query_result.completeness = query_result.hits.size() == request.options.top_k
                                    ? core::SearchCompleteness::complete_k
                                    : core::SearchCompleteness::eligible_exhausted;
  }
  return result;
}

[[nodiscard]] auto SegmentedCollection::fanout_search(const RoutingSnapshotPtr &snapshot,
                                                      const CollectionSearchRequest &request,
                                                      core::FilterExecution execution)
    -> core::Result<CollectionSearchResult> {
  auto request_limit = request.options.top_k;
  core::RowCount maximum_known_rows{};
  for (const auto &entry : snapshot->segments) {
    maximum_known_rows = std::max(maximum_known_rows, snapshot->known_rows_for(*entry));
  }

  for (std::uint32_t round = 0;; ++round) {
    std::vector<std::vector<Candidate>> candidates(static_cast<std::size_t>(request.queries.rows));
    std::vector<bool> exhaustive(static_cast<std::size_t>(request.queries.rows), true);
    std::vector<core::Status> per_query_status(static_cast<std::size_t>(request.queries.rows),
                                               core::Status::success());

    for (const auto &entry : snapshot->segments) {
      const auto known_rows = snapshot->known_rows_for(*entry);
      if (known_rows == 0) {
        continue;
      }
      const auto candidate_limit = std::min<core::RowCount>(known_rows, request_limit);
      std::uint64_t sink_count{};
      if (candidate_limit == 0 ||
          !core::checked_multiply(request.queries.rows, candidate_limit, sink_count) ||
          sink_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        return core::Status::error(core::StatusCode::invalid_argument,
                                   core::OperationStage::search,
                                   core::StatusDetail::arithmetic_overflow,
                                   "collection fanout sink size is not representable");
      }
      SegmentSearchStorage storage(request.queries.rows, candidate_limit);
      const auto descriptor = entry->segment.descriptor();
      const auto is_memory_graph = descriptor.algorithm_id == core::algorithm::qg;
      const auto is_qg_laser =
          is_memory_graph && descriptor.engine_factory_id == core::algorithm::laser;
      std::vector<core::AlgorithmSearchExtension> segment_extensions;
      segment_extensions.reserve(request.options.extensions.size() + 1);
      for (const auto &extension : request.options.extensions) {
        if (extension.algorithm_id == descriptor.algorithm_id) {
          segment_extensions.push_back(extension);
        }
      }
      // Declared at this scope (one per segment-loop iteration), not
      // inside the `if` below: make_qg_search_extension() stores
      // payload = std::addressof(its argument), and the resulting
      // AlgorithmSearchExtension is read through segment_extensions well
      // past that `if` block's closing brace (segment_request.options.
      // extensions below, and again inside entry->segment.search()). A
      // narrower scope here is a dangling-pointer bug -- caught here
      // because an unrelated local added a few lines down
      // (segment_filter_storage) perturbed the stack layout enough to
      // turn latent UB into a real failure (a memory graph segment
      // rejecting its own synthesized effort extension as corrupt).
      // Pre-existing, unrelated to the admission contract; fixed in
      // passing since it now reproduces deterministically.
      ::alaya::QgSearchExtension qg_effort;
      ::alaya::disk::LaserSegmentSearchExtension laser_effort;
      if (is_memory_graph) {
        if (candidate_limit > std::numeric_limits<std::uint32_t>::max()) {
          return core::Status::error(core::StatusCode::invalid_argument,
                                     core::OperationStage::search,
                                     core::StatusDetail::arithmetic_overflow,
                                     "Collection memory graph candidate limit exceeds uint32");
        }
        auto synthesize_effort = [&]<typename Extension>(Extension &effective, auto make) {
          effective.effort = std::max<std::uint32_t>(effective.effort,
                                                     static_cast<std::uint32_t>(candidate_limit));
          for (const auto &extension : segment_extensions) {
            if (extension.payload == nullptr || extension.payload_size < sizeof(Extension)) {
              continue;
            }
            Extension requested;
            std::memcpy(std::addressof(requested), extension.payload, sizeof(Extension));
            if (core::is_current_struct(requested)) {
              effective.effort = std::max(effective.effort, requested.effort);
            }
          }
          segment_extensions.push_back(make(effective));
        };  // NOLINT(readability/braces)
        synthesize_effort(qg_effort, [](const auto &extension) {
          return ::alaya::make_qg_search_extension(extension);
        });
        if (is_qg_laser) {
          // Preserve legacy qg validation before replacing its payload.
          // The final entry is the synthesized, known-good extension.
          for (std::size_t index = 0; index + 1 < segment_extensions.size(); ++index) {
            const auto &extension = segment_extensions[index];
            if (extension.payload == nullptr ||
                extension.payload_size < sizeof(::alaya::QgSearchExtension)) {
              return core::Status::error(core::StatusCode::invalid_argument,
                                         core::OperationStage::validation,
                                         core::StatusDetail::malformed_struct,
                                         "Collection qg search extension payload is truncated");
            }
            ::alaya::QgSearchExtension requested;
            std::memcpy(std::addressof(requested),
                        extension.payload,
                        sizeof(::alaya::QgSearchExtension));
            if (!core::is_current_struct(requested)) {
              return core::Status::
                  error(core::StatusCode::invalid_argument,
                        core::OperationStage::validation,
                        core::StatusDetail::malformed_struct,
                        "Collection qg search extension has an incompatible version");
            }
          }
          // The public qg extension remains the stable user contract. The
          // same-id LASER segment consumes its effort as native ef and opts
          // into the numeric-distance result contract. Do not forward the
          // qg payload to LASER: its default unknown-extension policy is
          // reject, while the translated extension is the one this physical
          // engine owns.
          segment_extensions.clear();
          laser_effort.effort = qg_effort.effort;
          laser_effort.return_distances = true;
          auto translated = ::alaya::disk::make_laser_segment_search_extension(laser_effort);
          translated.unknown_policy = core::UnknownExtensionPolicy::ignore_safe;
          segment_extensions.push_back(translated);
        }
      }
      core::SearchRequest segment_request;
      segment_request.queries = request.queries;
      segment_request.options = request.options;
      segment_request.options.top_k = candidate_limit;
      segment_request.options.extensions = segment_extensions;
      // segment_filter_storage must outlive the segment.search() call
      // below (segment_request.filter.payload points into it); declaring
      // it in this scope, alongside segment_request, guarantees that.
      std::vector<std::uint64_t> segment_filter_storage;
      if (execution == core::FilterExecution::traversal && request.filter.active()) {
        // No segment type evaluates a LogicalFilter itself (all four
        // reject any non-none/non-bitmap filter kind), so Collection
        // precompiles admission here against its own logical registry,
        // in this segment's row space, and sends kind=bitmap rather
        // than kind=predicate. Segment admission contract section 3
        // (docs/design/segment-admission-contract.md).
        segment_filter_storage.assign((known_rows + 63) / 64, std::uint64_t{0});
        for (const auto &[logical_id, version] : snapshot->versions) {
          if (version.address.segment_id != entry->segment_id ||
              version.address.generation != entry->generation ||
              version.state != VersionState::live ||
              version.upsert_sequence > snapshot->visibility_watermark) {
            continue;
          }
          const auto row = static_cast<std::uint64_t>(version.address.row_id);
          if (row >= known_rows) {
            continue;  // defensive: outside this bitmap's capacity
          }
          if (request.filter.matches(logical_id,
                                     version.payload.metadata,
                                     version.payload.document)) {
            segment_filter_storage[row >> 6U] |= (std::uint64_t{1} << (row & 63U));
          }
        }
        segment_request.filter.kind = core::SegmentFilterKind::bitmap;
        segment_request.filter.exact = false;
        segment_request.filter.metadata_epoch = snapshot->metadata_epoch;
        segment_request.filter.payload = segment_filter_storage.data();
        segment_request.filter.payload_size = segment_filter_storage.size() * sizeof(std::uint64_t);
        segment_request.filter.selectivity_hint =
            request.filter.selectivity_estimate().value_or(1.0);
      }
      segment_request.context = request.context;
      segment_request.response = &storage.response;
      segment_request.lifetime_pin = std::const_pointer_cast<RoutingSnapshot>(snapshot);

      const auto capabilities = entry->segment.capabilities();
      if (request.stats != nullptr && descriptor.medium == core::Medium::disk) {
        std::uint64_t requests{};
        std::uint64_t bytes{};
        std::uint64_t row_bytes{};
        const auto accounted =
            core::checked_multiply(request.queries.rows, std::uint64_t{1}, requests) &&
            core::checked_multiply(entry->segment.descriptor().dim, sizeof(float), row_bytes) &&
            core::checked_multiply(known_rows, row_bytes, bytes) &&
            core::checked_multiply(bytes, request.queries.rows, bytes) &&
            core::checked_add(request.stats->io_requests_consumed,
                              requests,
                              request.stats->io_requests_consumed) &&
            core::checked_add(request.stats->io_bytes_consumed,
                              bytes,
                              request.stats->io_bytes_consumed) &&
            core::checked_add(request.stats->budget_consumed,
                              bytes,
                              request.stats->budget_consumed);
        if (!accounted) {
          return search_budget_denied("collection search runtime accounting overflowed");
        }
      }
      core::Status segment_status;
      if (capabilities.concurrency.reentrant_search) {
        std::shared_lock operation_lock(entry->operation_mutex);
        segment_status = entry->segment.search(std::move(segment_request));
      } else {
        std::unique_lock operation_lock(entry->operation_mutex);
        segment_status = entry->segment.search(std::move(segment_request));
      }
      if (!segment_status.ok() && execution == core::FilterExecution::traversal &&
          request.filter.active() && segment_status.code() == core::StatusCode::not_supported &&
          request.options.filter_policy == core::FilterPolicy::automatic) {
        // This segment cannot execute the bitmap filter it was just
        // handed (qg/disk_flat all still reject any non-none
        // filter kind). Re-planning the whole query would cost a
        // second selectivity pass; instead retry just this segment
        // unfiltered -- the per-hit re-verify a few lines down (already
        // unconditional whenever execution == traversal &&
        // request.filter.active(), independent of whether *this*
        // segment's own request carried a filter) weeds out
        // non-matching rows, and the existing overfetch/incomplete
        // machinery covers any resulting shortfall. strict policy is
        // deliberately excluded: it keeps today's fail-fast semantics
        // (a strong-consistency request must not silently degrade).
        core::SearchRequest retry_request;
        retry_request.queries = request.queries;
        retry_request.options = request.options;
        retry_request.options.top_k = candidate_limit;
        retry_request.options.extensions = segment_extensions;
        retry_request.context = request.context;
        retry_request.response = &storage.response;
        retry_request.lifetime_pin = std::const_pointer_cast<RoutingSnapshot>(snapshot);
        if (capabilities.concurrency.reentrant_search) {
          std::shared_lock operation_lock(entry->operation_mutex);
          segment_status = entry->segment.search(std::move(retry_request));
        } else {
          std::unique_lock operation_lock(entry->operation_mutex);
          segment_status = entry->segment.search(std::move(retry_request));
        }
      }
      if (!segment_status.ok()) {
        if (request.options.filter_policy != core::FilterPolicy::allow_partial) {
          return segment_status;
        }
        std::fill(exhaustive.begin(), exhaustive.end(), false);
        continue;
      }
      auto response_status =
          validate_segment_response(storage.response, request.queries.rows, candidate_limit);
      if (!response_status.ok()) {
        return response_status;
      }

      for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
        const auto index = static_cast<std::size_t>(query_index);
        if (!storage.statuses[index].ok()) {
          if (request.options.filter_policy == core::FilterPolicy::allow_partial) {
            exhaustive[index] = false;
          } else {
            per_query_status[index] = storage.statuses[index];
            exhaustive[index] = false;
          }
          continue;
        }
        exhaustive[index] =
            exhaustive[index] &&
            (storage.completeness[index] == core::SearchCompleteness::eligible_exhausted ||
             storage.counts[index] >= known_rows);
        for (core::RowCount hit_index = storage.offsets[index];
             hit_index < storage.offsets[index + 1];
             ++hit_index) {
          const auto &hit = storage.hits[static_cast<std::size_t>(hit_index)];
          if (hit.score_kind != core::ScoreKind::rank_only && is_nan_score(hit.score)) {
            if (request.stats != nullptr) {
              ++request.stats->nan_discarded;
            }
            continue;
          }
          const RowAddress address{entry->segment_id, entry->generation, hit.row_id};
          const auto reverse = snapshot->reverse.find(address);
          if (reverse == snapshot->reverse.end() ||
              reverse->second.upsert_sequence > snapshot->visibility_watermark) {
            continue;
          }
          const auto version = snapshot->versions.find(reverse->second.logical_id);
          if (version == snapshot->versions.end() || version->second.state != VersionState::live ||
              version->second.upsert_sequence != reverse->second.upsert_sequence ||
              version->second.address != address ||
              version->second.upsert_sequence > snapshot->visibility_watermark) {
            continue;
          }
          if (execution == core::FilterExecution::traversal && request.filter.active()) {
            if (request.stats != nullptr) {
              ++request.stats->filter_examined;
            }
            if (!request.filter.matches(version->first,
                                        version->second.payload.metadata,
                                        version->second.payload.document)) {
              continue;
            }
            if (request.stats != nullptr) {
              ++request.stats->filter_passed;
            }
          }
          auto flags = hit.result_flags | core::ResultFlag::version_checked;
          if (request.filter.active()) {
            flags = flags | core::ResultFlag::filtered;
          }
          candidates[index].push_back(Candidate{CollectionHit{version->first,
                                                              hit.score,
                                                              hit.score_kind,
                                                              hit.comparable_metric,
                                                              flags,
                                                              version->second.upsert_sequence,
                                                              address},
                                                entry,
                                                &version->second.payload});
        }
      }
    }

    CollectionSearchResult result;
    result.visibility_watermark = snapshot->visibility_watermark;
    result.metadata_epoch = snapshot->metadata_epoch;
    result.queries.resize(static_cast<std::size_t>(request.queries.rows));
    bool needs_more{};
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      const auto index = static_cast<std::size_t>(query_index);
      auto &query_result = result.queries[index];
      if (!per_query_status[index].ok()) {
        query_result.status = per_query_status[index];
        query_result.completeness = core::SearchCompleteness::failed;
        continue;
      }
      auto normalized = normalize_scores(candidates[index],
                                         query_row(request.queries, query_index),
                                         request.context,
                                         request.stats);
      if (!normalized.ok()) {
        return normalized.status();
      }
      query_result.hits = std::move(normalized).value();
      sort_hits(query_result.hits);
      query_result.hits.erase(std::unique(query_result.hits.begin(),
                                          query_result.hits.end(),
                                          [](const CollectionHit &lhs, const CollectionHit &rhs) {
                                            return lhs.logical_id == rhs.logical_id;
                                          }),
                              query_result.hits.end());

      if (execution == core::FilterExecution::postfilter && request.filter.active()) {
        std::vector<CollectionHit> filtered;
        filtered.reserve(query_result.hits.size());
        for (auto &hit : query_result.hits) {
          const auto version = snapshot->versions.find(hit.logical_id);
          if (request.stats != nullptr) {
            ++request.stats->filter_examined;
          }
          if (version == snapshot->versions.end() ||
              !request.filter.matches(hit.logical_id,
                                      version->second.payload.metadata,
                                      version->second.payload.document)) {
            continue;
          }
          if (request.stats != nullptr) {
            ++request.stats->filter_passed;
          }
          filtered.push_back(std::move(hit));
        }
        query_result.hits = std::move(filtered);
      }
      if (query_result.hits.size() > request.options.top_k) {
        query_result.hits.resize(static_cast<std::size_t>(request.options.top_k));
      }
      query_result.status = core::Status::success();
      query_result.completeness =
          query_result.hits.size() == request.options.top_k ? core::SearchCompleteness::complete_k
          : exhaustive[index] ? core::SearchCompleteness::eligible_exhausted
                              : core::SearchCompleteness::strategy_incomplete;
      needs_more =
          needs_more || (query_result.hits.size() < request.options.top_k && !exhaustive[index]);
    }

    if (!needs_more || round >= request.maximum_overfetch_rounds ||
        request_limit >= maximum_known_rows) {
      return result;
    }
    const auto doubled = request_limit > std::numeric_limits<core::RowCount>::max() / 2
                             ? std::numeric_limits<core::RowCount>::max()
                             : request_limit * 2;
    const auto next_limit = std::min(maximum_known_rows, doubled);
    if (next_limit <= request_limit) {
      return result;
    }
    request_limit = next_limit;
    if (request.stats != nullptr) {
      ++request.stats->overfetch_rounds;
    }
  }
}

[[nodiscard]] auto SegmentedCollection::normalize_scores(const std::vector<Candidate> &candidates,
                                                         const core::TypedTensorView &query,
                                                         core::SearchContext *context,
                                                         CollectionSearchStats *stats)
    -> core::Result<std::vector<CollectionHit>> {
  std::vector<CollectionHit> result;
  result.reserve(candidates.size());
  std::optional<std::pair<core::ScoreKind, core::Metric>> domain;
  std::optional<std::chrono::steady_clock::time_point> rerank_started;
  for (const auto &candidate : candidates) {
    auto hit = candidate.hit;
    if (hit.score_kind == core::ScoreKind::rank_only) {
      if (stats != nullptr && !rerank_started.has_value()) {
        rerank_started = std::chrono::steady_clock::now();
      }
      core::Result<float> reranked =
          core::Status::error(core::StatusCode::not_supported,
                              core::OperationStage::search,
                              core::StatusDetail::operation_slot_absent,
                              "rank-only segment has no exact rerank source");
      if (candidate.segment->exact_rerank) {
        reranked = candidate.segment->exact_rerank(query, hit.source.row_id);
      } else if (candidate.payload != nullptr && candidate.payload->vector.has_value()) {
        reranked = exact_distance(query, *candidate.payload->vector, schema_.metric);
      }
      if (!reranked.ok()) {
        return reranked.status();
      }
      hit.score = std::move(reranked).value();
      if (is_nan_score(hit.score)) {
        if (stats != nullptr) {
          ++stats->nan_discarded;
        }
        continue;
      }
      hit.score_kind = core::ScoreKind::distance;
      hit.comparable_metric = schema_.metric;
      hit.result_flags = hit.result_flags | core::ResultFlag::exact_reranked;
      if (context != nullptr && context->stats != nullptr) {
        ++context->stats->rerank_count;
      }
    }
    const auto current_domain = std::pair{hit.score_kind, hit.comparable_metric};
    if (domain.has_value() && *domain != current_domain) {
      return core::Status::error(core::StatusCode::not_supported,
                                 core::OperationStage::search,
                                 core::StatusDetail::invalid_score,
                                 "segment score domains are not comparable");
    }
    domain = current_domain;
    result.push_back(std::move(hit));
  }
  if (stats != nullptr && rerank_started.has_value()) {
    stats->rerank_nanoseconds +=
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                       std::chrono::steady_clock::now() - *rerank_started)
                                       .count());
  }
  return result;
}
}  // namespace alaya::internal::collection
