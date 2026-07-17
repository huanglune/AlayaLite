// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fake_mutable_segment.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "space/rabitq_space.hpp"
#include "space/raw_space.hpp"

namespace alaya::internal::collection {
namespace {

using test::FakeMutableSegment;

class StaticSegment {
 public:
  using Rows = std::map<std::uint64_t, std::array<float, 2>>;

  explicit StaticSegment(Rows rows,
                         core::ScoreKind score_kind = core::ScoreKind::distance,
                         bool emit_nan = false,
                         std::shared_ptr<std::atomic_bool> destroyed = {})
      : rows_(std::move(rows)),
        score_kind_(score_kind),
        emit_nan_(emit_nan),
        destroyed_(std::move(destroyed)) {}

  ~StaticSegment() {
    if (destroyed_ != nullptr) {
      destroyed_->store(true, std::memory_order_release);
    }
  }

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor {
    core::Descriptor descriptor;
    descriptor.algorithm_id = score_kind_ == core::ScoreKind::similarity ? 7002 : 7001;
    descriptor.format_version = 1;
    descriptor.factory_version = 1;
    descriptor.dim = 2;
    descriptor.metric = core::Metric::l2;
    descriptor.stored_scalar_type = core::ScalarType::float32;
    descriptor.medium = core::Medium::memory;
    descriptor.engine_factory_id = descriptor.algorithm_id;
    return descriptor;
  }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    if (request.queries.rows != 1) {
      return core::Status::error(core::StatusCode::invalid_argument,
                                 core::OperationStage::validation,
                                 core::StatusDetail::malformed_struct,
                                 "static single search requires one query");
    }
    return execute(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const noexcept -> core::Status {
    stats = core::SegmentStats{};
    stats.live_rows = rows_.size();
    stats.allocated_rows = rows_.size();
    return core::Status::success();
  }

 private:
  [[nodiscard]] auto execute(const core::SearchRequest &request) const -> core::Status {
    auto &response = *request.response;
    response.query_count = request.queries.rows;
    response.score_kind = score_kind_;
    response.comparable_metric = core::Metric::l2;
    response.offsets[0] = 0;
    core::RowCount cursor{};
    for (core::RowCount query_index = 0; query_index < request.queries.rows; ++query_index) {
      const auto *query = request.queries.row<float>(query_index);
      struct Scored {
        std::uint64_t row{};
        float score{};
      };
      std::vector<Scored> scored;
      for (const auto &[row, vector] : rows_) {
        const auto first = query[0] - vector[0];
        const auto second = query[1] - vector[1];
        const auto distance = first * first + second * second;
        scored.push_back({row,
                          emit_nan_ ? std::numeric_limits<float>::quiet_NaN()
                          : score_kind_ == core::ScoreKind::similarity ? -distance
                                                                       : distance});
      }
      std::sort(scored.begin(), scored.end(), [&](const Scored &lhs, const Scored &rhs) {
        if (lhs.score != rhs.score) {
          return score_kind_ == core::ScoreKind::similarity ? lhs.score > rhs.score
                                                            : lhs.score < rhs.score;
        }
        return lhs.row < rhs.row;
      });
      const auto count = std::min<std::size_t>(scored.size(), request.options.top_k);
      for (std::size_t index = 0; index < count; ++index) {
        response.hits[static_cast<std::size_t>(cursor++)] =
            core::SearchHit(core::SegmentRowId(scored[index].row),
                            scored[index].score,
                            score_kind_,
                            core::Metric::l2,
                            core::ResultFlag::approximate);
      }
      response.offsets[static_cast<std::size_t>(query_index + 1)] = cursor;
      response.valid_counts[static_cast<std::size_t>(query_index)] = count;
      response.statuses[static_cast<std::size_t>(query_index)] = core::Status::success();
      response.completeness[static_cast<std::size_t>(query_index)] =
          count == request.options.top_k ? core::SearchCompleteness::complete_k
                                         : core::SearchCompleteness::eligible_exhausted;
    }
    return core::Status::success();
  }

  Rows rows_{};
  core::ScoreKind score_kind_{core::ScoreKind::distance};
  bool emit_nan_{};
  std::shared_ptr<std::atomic_bool> destroyed_{};
};

[[nodiscard]] auto owned_payload(const std::array<float, 2> &vector,
                                 Metadata metadata = {},
                                 std::string document = {}) -> RecordPayload {
  auto owned = OwnedVector::copy_row(core::TypedTensorView::contiguous(vector.data(), 1, 2), 0);
  EXPECT_TRUE(owned.ok());
  RecordPayload payload;
  if (owned.ok()) {
    payload.vector = std::move(owned).value();
  }
  payload.metadata = std::move(metadata);
  payload.document = std::move(document);
  return payload;
}

[[nodiscard]] auto readonly_any(const std::shared_ptr<StaticSegment> &producer)
    -> core::AnySegment {
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency.reentrant_search = true;
  config.concurrency.explicit_drain = false;
  auto erased = core::AnySegment::from_sync(producer, config);
  EXPECT_TRUE(erased.ok());
  return std::move(erased).value();
}

[[nodiscard]] auto fake_registration(const std::shared_ptr<FakeMutableSegment> &producer)
    -> SegmentRegistration {
  auto erased = test::make_fake_mutable_any(producer);
  EXPECT_TRUE(erased.ok());
  SegmentRegistration registration;
  registration.segment_id = FakeMutableSegment::kSegmentId;
  registration.generation = 1;
  registration.role = SegmentRole::active_mutable;
  registration.segment = std::move(erased).value();
  registration.atomic_mutation_bundle = true;
  return registration;
}

[[nodiscard]] auto open_fake_collection(std::shared_ptr<FakeMutableSegment> *producer_out = nullptr)
    -> std::shared_ptr<SegmentedCollection> {
  auto producer = std::make_shared<FakeMutableSegment>();
  std::vector<SegmentRegistration> registrations;
  registrations.push_back(fake_registration(producer));
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          std::move(registrations));
  EXPECT_TRUE(opened.ok());
  if (producer_out != nullptr) {
    *producer_out = std::move(producer);
  }
  return std::move(opened).value();
}

[[nodiscard]] auto make_search_request(const float *queries,
                                       core::RowCount rows,
                                       std::uint64_t top_k,
                                       core::SearchContext &context,
                                       LogicalFilter filter = {}) -> CollectionSearchRequest {
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(queries, rows, 2);
  request.options.top_k = top_k;
  request.filter = std::move(filter);
  request.context = &context;
  return request;
}

[[nodiscard]] auto write_request(const core::LogicalId &logical_id,
                                 const std::array<float, 2> &vector,
                                 Metadata metadata = {},
                                 std::string document = {},
                                 WriteMode mode = WriteMode::upsert) -> WriteRequest {
  WriteRequest request;
  request.logical_id = logical_id;
  request.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  request.metadata = std::move(metadata);
  request.document = std::move(document);
  request.mode = mode;
  return request;
}

TEST(SegmentedCollection, StringLogicalIdInsertGetUpsertDeleteVisibility) {
  const auto collection = open_fake_collection();
  core::MutationContext mutation_context;
  const auto id = core::LogicalId::from_utf8("alpha");
  const std::array<float, 2> first{0.0F, 0.0F};
  auto inserted =
      collection->write(write_request(id, first, {{"group", std::string("red")}}, "first"),
                        mutation_context);
  ASSERT_TRUE(inserted.ok());
  EXPECT_EQ(inserted.value().op_id, 1U);
  EXPECT_EQ(inserted.value().row_status, RowMutationStatus::inserted);

  auto record = collection->get_by_id(id);
  ASSERT_TRUE(record.ok());
  EXPECT_EQ(record.value().logical_id, id);
  EXPECT_EQ(record.value().document, "first");
  EXPECT_EQ(record.value().metadata.at("group"), ScalarValue(std::string("red")));

  const std::array<float, 2> second{5.0F, 5.0F};
  auto updated =
      collection->write(write_request(id, second, {{"group", std::string("blue")}}, "second"),
                        mutation_context);
  ASSERT_TRUE(updated.ok());
  EXPECT_EQ(updated.value().op_id, 2U);
  EXPECT_EQ(updated.value().row_status, RowMutationStatus::updated);
  record = collection->get_by_id(id);
  ASSERT_TRUE(record.ok());
  EXPECT_EQ(record.value().upsert_sequence, 2U);
  EXPECT_EQ(record.value().document, "second");

  core::SearchContext search_context;
  auto search = collection->search(make_search_request(second.data(), 1, 1, search_context));
  ASSERT_TRUE(search.ok());
  ASSERT_EQ(search.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(search.value().queries[0].hits[0].logical_id, id);
  EXPECT_EQ(search.value().queries[0].hits[0].upsert_sequence, 2U);

  auto deleted = collection->erase(id, mutation_context);
  ASSERT_TRUE(deleted.ok());
  EXPECT_EQ(deleted.value().op_id, 3U);
  EXPECT_EQ(deleted.value().row_status, RowMutationStatus::deleted);
  EXPECT_EQ(collection->get_by_id(id).status().code(), core::StatusCode::not_found);
  search = collection->search(make_search_request(second.data(), 1, 1, search_context));
  ASSERT_TRUE(search.ok());
  EXPECT_TRUE(search.value().queries[0].hits.empty());
  EXPECT_EQ(collection->stats().size, 0U);
  EXPECT_EQ(collection->stats().accepted_count, 0U);
  EXPECT_EQ(collection->stats().tombstone_count, 1U);
}

TEST(SegmentedCollection, ShellFlagAndWriteModesHaveExplicitStatus) {
  auto producer = std::make_shared<FakeMutableSegment>();
  CollectionConfig disabled;
  disabled.features.collection_shell = false;
  auto unavailable = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                               {fake_registration(producer)},
                                               disabled);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), core::StatusCode::not_supported);

  const auto collection = open_fake_collection();
  core::MutationContext context;
  const auto id = core::LogicalId::from_utf8("mode");
  const auto missing = core::LogicalId::from_utf8("missing");
  const std::array<float, 2> vector{1.0F, 2.0F};
  auto replace_missing =
      collection->write(write_request(missing, vector, {}, {}, WriteMode::replace), context);
  EXPECT_FALSE(replace_missing.ok());
  EXPECT_EQ(replace_missing.status().code(), core::StatusCode::not_found);
  ASSERT_TRUE(
      collection->write(write_request(id, vector, {}, {}, WriteMode::insert_only), context).ok());
  auto duplicate =
      collection->write(write_request(id, vector, {}, {}, WriteMode::insert_only), context);
  EXPECT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.status().code(), core::StatusCode::conflict);
  auto replaced = collection->write(write_request(id, vector, {}, {}, WriteMode::replace), context);
  ASSERT_TRUE(replaced.ok());
  EXPECT_EQ(replaced.value().row_status, RowMutationStatus::replaced);
  EXPECT_EQ(collection->stats().accepted_count, 1U);
}

TEST(SegmentedCollection, LegacyUint64IdentityMapsLabelsWithoutChangingCanonicalBytes) {
  auto producer = std::make_shared<StaticSegment>(
      StaticSegment::Rows{{42, {0.0F, 0.0F}}, {43, {10.0F, 10.0F}}});
  SegmentRegistration registration;
  registration.segment_id = 1;
  registration.role = SegmentRole::legacy_readonly;
  registration.segment = readonly_any(producer);
  registration.rows = {{core::LogicalId::from_legacy_uint64(42),
                        core::SegmentRowId(42),
                        0,
                        VersionState::live,
                        owned_payload({0.0F, 0.0F})},
                       {core::LogicalId::from_legacy_uint64(43),
                        core::SegmentRowId(43),
                        0,
                        VersionState::live,
                        owned_payload({10.0F, 10.0F})}};
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok());
  const auto collection = std::move(opened).value();

  const auto id = core::LogicalId::from_legacy_uint64(42);
  auto record = collection->get_by_id(id, Projection::identity);
  ASSERT_TRUE(record.ok());
  EXPECT_EQ(record.value().logical_id.kind(), core::LogicalIdKind::legacy_uint64);
  EXPECT_EQ(record.value().logical_id.canonical_bytes().size(), sizeof(std::uint64_t));
  EXPECT_EQ(std::to_integer<unsigned>(record.value().logical_id.canonical_bytes().back()), 42U);

  const std::array<float, 2> query{0.0F, 0.0F};
  core::SearchContext context;
  auto result = collection->search(make_search_request(query.data(), 1, 1, context));
  ASSERT_TRUE(result.ok());
  ASSERT_EQ(result.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(result.value().queries[0].hits[0].logical_id, id);
  EXPECT_EQ(result.value().queries[0].hits[0].source.row_id, core::SegmentRowId(42));
}

TEST(SegmentedCollection, MetadataFilterScalarHybridAndDeleteByFilterAreExact) {
  const auto collection = open_fake_collection();
  core::MutationContext mutation_context;
  const auto red = core::LogicalId::from_utf8("red");
  const auto blue = core::LogicalId::from_utf8("blue");
  const std::array<float, 2> red_vector{0.0F, 0.0F};
  const std::array<float, 2> blue_vector{10.0F, 10.0F};
  ASSERT_TRUE(
      collection
          ->write(write_request(red, red_vector, {{"color", std::string("red")}}, "red document"),
                  mutation_context)
          .ok());
  ASSERT_TRUE(collection
                  ->write(write_request(blue,
                                        blue_vector,
                                        {{"color", std::string("blue")}},
                                        "blue document"),
                          mutation_context)
                  .ok());

  const auto blue_filter =
      LogicalFilter::metadata_equals("color", ScalarValue(std::string("blue")));
  auto scalar = collection->scalar_query(blue_filter, 10);
  ASSERT_TRUE(scalar.ok());
  ASSERT_EQ(scalar.value().size(), 1U);
  EXPECT_EQ(scalar.value()[0].logical_id, blue);

  core::SearchContext search_context;
  auto request = make_search_request(blue_vector.data(), 1, 2, search_context, blue_filter);
  request.options.filter_policy = core::FilterPolicy::strict;
  auto hybrid = collection->hybrid_query(request);
  ASSERT_TRUE(hybrid.ok());
  ASSERT_EQ(hybrid.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(hybrid.value().queries[0].hits[0].logical_id, blue);
  EXPECT_EQ(hybrid.value().queries[0].completeness, core::SearchCompleteness::eligible_exhausted);

  const auto red_filter = LogicalFilter::metadata_equals("color", ScalarValue(std::string("red")));
  auto deleted = collection->delete_by_filter(red_filter, mutation_context);
  ASSERT_TRUE(deleted.ok());
  ASSERT_EQ(deleted.value().size(), 1U);
  EXPECT_EQ(collection->stats().size, 1U);
  EXPECT_EQ(collection->get_by_id(red).status().code(), core::StatusCode::not_found);
  EXPECT_TRUE(collection->get_by_id(blue).ok());
}

TEST(SegmentedCollection, DarkStageAbortAndPendingStatsNeverBecomeVisible) {
  std::shared_ptr<FakeMutableSegment> producer;
  const auto collection = open_fake_collection(&producer);
  producer->gate_next_stage();
  const auto id = core::LogicalId::from_utf8("pending");
  const std::array<float, 2> vector{1.0F, 1.0F};
  core::MutationContext context;
  const auto request = write_request(id, vector);
  auto pending = std::async(std::launch::async, [&] {
    return collection->write(request, context);
  });
  ASSERT_TRUE(producer->wait_for_stage());
  auto stats = collection->stats();
  EXPECT_EQ(stats.accepted_count, 1U);
  EXPECT_EQ(stats.pending_count, 1U);
  EXPECT_EQ(stats.pending_bytes, sizeof(float) * 2U);
  EXPECT_EQ(stats.size, 0U);
  EXPECT_EQ(collection->get_by_id(id).status().code(), core::StatusCode::not_found);
  producer->release_stage();
  ASSERT_TRUE(pending.get().ok());
  stats = collection->stats();
  EXPECT_EQ(stats.pending_count, 0U);
  EXPECT_EQ(stats.pending_bytes, 0U);
  EXPECT_EQ(stats.size, 1U);

  producer->fail_next_stage();
  const auto aborted_id = core::LogicalId::from_utf8("aborted");
  const auto aborted = collection->write(write_request(aborted_id, vector), context);
  EXPECT_FALSE(aborted.ok());
  EXPECT_EQ(aborted.status().stage(), core::OperationStage::mutation_stage);
  EXPECT_EQ(producer->abort_count(), 1U);
  EXPECT_EQ(collection->get_by_id(aborted_id).status().code(), core::StatusCode::not_found);
  EXPECT_EQ(collection->stats().size, 1U);
  EXPECT_EQ(collection->stats().accepted_count, 1U);
}

TEST(SegmentedCollection, WatermarkAtAdmissionSuppressesRowsPublishedDuringSearch) {
  std::shared_ptr<FakeMutableSegment> producer;
  const auto collection = open_fake_collection(&producer);
  core::MutationContext mutation_context;
  const auto id = core::LogicalId::from_utf8("versioned");
  const std::array<float, 2> old_vector{0.0F, 0.0F};
  const std::array<float, 2> new_vector{10.0F, 10.0F};
  auto inserted = collection->write(write_request(id, old_vector), mutation_context);
  ASSERT_TRUE(inserted.ok());

  producer->gate_next_search();
  core::SearchContext search_context;
  const auto request = make_search_request(new_vector.data(), 1, 1, search_context);
  auto admitted_search = std::async(std::launch::async, [&] {
    return collection->search(request);
  });
  ASSERT_TRUE(producer->wait_for_search());
  auto updated = collection->write(write_request(id, new_vector), mutation_context);
  ASSERT_TRUE(updated.ok());
  producer->release_search();
  auto old_view = admitted_search.get();
  ASSERT_TRUE(old_view.ok());
  EXPECT_EQ(old_view.value().visibility_watermark, inserted.value().visibility_watermark);
  for (const auto &hit : old_view.value().queries[0].hits) {
    EXPECT_LE(hit.upsert_sequence, old_view.value().visibility_watermark);
    EXPECT_NE(hit.upsert_sequence, updated.value().op_id);
  }

  auto new_view = collection->search(request);
  ASSERT_TRUE(new_view.ok());
  ASSERT_EQ(new_view.value().queries[0].hits.size(), 1U);
  EXPECT_EQ(new_view.value().visibility_watermark, updated.value().op_id);
  EXPECT_EQ(new_view.value().queries[0].hits[0].upsert_sequence, updated.value().op_id);
}

TEST(SegmentedCollection, MutationMutationIsSerializedAndOpIdsAreMonotonic) {
  std::shared_ptr<FakeMutableSegment> producer;
  const auto collection = open_fake_collection(&producer);
  constexpr std::size_t kWrites = 16;
  std::vector<std::thread> threads;
  std::vector<std::uint64_t> op_ids;
  std::mutex result_mutex;
  for (std::size_t index = 0; index < kWrites; ++index) {
    threads.emplace_back([&, index] {
      const auto id = core::LogicalId::from_utf8("id-" + std::to_string(index));
      const std::array<float, 2> vector{static_cast<float>(index), 0.0F};
      core::MutationContext context;
      auto receipt = collection->write(write_request(id, vector), context);
      ASSERT_TRUE(receipt.ok());
      if (receipt.ok()) {
        std::lock_guard lock(result_mutex);
        op_ids.push_back(receipt.value().op_id);
      }
    });
  }
  for (auto &thread : threads) {
    thread.join();
  }
  std::sort(op_ids.begin(), op_ids.end());
  ASSERT_EQ(op_ids.size(), kWrites);
  for (std::size_t index = 0; index < kWrites; ++index) {
    EXPECT_EQ(op_ids[index], index + 1);
  }
  EXPECT_EQ(producer->maximum_active_mutations(), 1U);
  EXPECT_EQ(collection->stats().size, kWrites);
}

TEST(SegmentedCollection, RejectsIncomparableScoreDomainsAndDiscardsNaN) {
  auto distance = std::make_shared<StaticSegment>(StaticSegment::Rows{{0, {0.0F, 0.0F}}},
                                                  core::ScoreKind::distance);
  auto similarity = std::make_shared<StaticSegment>(StaticSegment::Rows{{0, {1.0F, 1.0F}}},
                                                    core::ScoreKind::similarity);
  SegmentRegistration first;
  first.segment_id = 10;
  first.segment = readonly_any(distance);
  first.rows = {{core::LogicalId::from_utf8("distance"),
                 core::SegmentRowId(0),
                 0,
                 VersionState::live,
                 owned_payload({0.0F, 0.0F})}};
  SegmentRegistration second;
  second.segment_id = 11;
  second.segment = readonly_any(similarity);
  second.rows = {{core::LogicalId::from_utf8("similarity"),
                  core::SegmentRowId(0),
                  0,
                  VersionState::live,
                  owned_payload({1.0F, 1.0F})}};
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(first), std::move(second)});
  ASSERT_TRUE(opened.ok());
  const std::array<float, 2> query{};
  core::SearchContext context;
  auto mixed = std::move(opened).value()->search(make_search_request(query.data(), 1, 2, context));
  EXPECT_FALSE(mixed.ok());
  EXPECT_EQ(mixed.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(mixed.status().detail(), core::StatusDetail::invalid_score);

  auto nan = std::make_shared<StaticSegment>(StaticSegment::Rows{{0, {0.0F, 0.0F}}},
                                             core::ScoreKind::distance,
                                             true);
  SegmentRegistration nan_registration;
  nan_registration.segment_id = 12;
  nan_registration.segment = readonly_any(nan);
  nan_registration.rows = {{core::LogicalId::from_utf8("nan"),
                            core::SegmentRowId(0),
                            0,
                            VersionState::live,
                            owned_payload({0.0F, 0.0F})}};
  opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                     {std::move(nan_registration)});
  ASSERT_TRUE(opened.ok());
  const auto nan_collection = std::move(opened).value();
  CollectionSearchStats nan_stats;
  auto nan_request = make_search_request(query.data(), 1, 1, context);
  nan_request.stats = &nan_stats;
  auto discarded = nan_collection->search(nan_request);
  ASSERT_TRUE(discarded.ok()) << discarded.status().diagnostic();
  EXPECT_TRUE(discarded.value().queries[0].hits.empty());
  EXPECT_EQ(discarded.value().queries[0].completeness,
            core::SearchCompleteness::eligible_exhausted);
  EXPECT_EQ(nan_stats.nan_discarded, 1U);
  EXPECT_EQ(nan_collection->outstanding_search_leases(), 0U);
}

TEST(SegmentedCollection, FilterPoliciesSelectAllExecutionsAndBoundedOverfetch) {
  StaticSegment::Rows physical;
  SegmentRegistration registration;
  registration.segment_id = 41;
  registration.role = SegmentRole::sealed;
  for (std::uint64_t row = 0; row < 8; ++row) {
    const std::array<float, 2> vector{static_cast<float>(row), 0.0F};
    physical.emplace(row, vector);
    registration.rows.push_back({core::LogicalId::from_utf8("row-" + std::to_string(row)),
                                 core::SegmentRowId(row),
                                 row + 1,
                                 VersionState::live,
                                 owned_payload(vector, {{"selected", row >= 4}, {"all", true}})});
  }
  registration.segment = readonly_any(std::make_shared<StaticSegment>(std::move(physical)));
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();
  const std::array<float, 2> query{};
  core::SearchContext context;

  CollectionSearchStats low_stats;
  auto low = make_search_request(query.data(),
                                 1,
                                 2,
                                 context,
                                 LogicalFilter::metadata_equals("selected", true));
  low.filter = LogicalFilter(
      [](const core::LogicalId &, const Metadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("selected"));
      },
      0.05);
  low.stats = &low_stats;
  auto prefiltered = collection->search(low);
  ASSERT_TRUE(prefiltered.ok()) << prefiltered.status().diagnostic();
  EXPECT_EQ(low_stats.filter_execution, core::FilterExecution::prefilter);
  EXPECT_EQ(prefiltered.value().queries[0].hits.size(), 2U);
  EXPECT_EQ(low_stats.overfetch_rounds, 0U);

  CollectionSearchStats medium_stats;
  auto medium = low;
  medium.filter = LogicalFilter(
      [](const core::LogicalId &, const Metadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("selected"));
      },
      0.50);
  medium.stats = &medium_stats;
  auto traversed = collection->search(medium);
  ASSERT_TRUE(traversed.ok()) << traversed.status().diagnostic();
  EXPECT_EQ(medium_stats.filter_execution, core::FilterExecution::traversal);
  EXPECT_EQ(medium_stats.overfetch_rounds, 2U);
  ASSERT_EQ(traversed.value().queries[0].hits.size(), 2U);
  EXPECT_EQ(traversed.value().queries[0].hits[0].logical_id, core::LogicalId::from_utf8("row-4"));
  EXPECT_GT(medium_stats.filter_examined, medium_stats.filter_passed);

  CollectionSearchStats high_stats;
  auto high = low;
  high.filter = LogicalFilter(
      [](const core::LogicalId &, const Metadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("all"));
      },
      1.0);
  high.stats = &high_stats;
  auto postfiltered = collection->search(high);
  ASSERT_TRUE(postfiltered.ok()) << postfiltered.status().diagnostic();
  EXPECT_EQ(high_stats.filter_execution, core::FilterExecution::postfilter);
  EXPECT_EQ(postfiltered.value().queries[0].hits.size(), 2U);
  EXPECT_EQ(high_stats.overfetch_rounds, 0U);

  CollectionSearchStats partial_stats;
  auto partial = medium;
  partial.options.filter_policy = core::FilterPolicy::allow_partial;
  partial.maximum_overfetch_rounds = 0;
  partial.stats = &partial_stats;
  auto best_effort = collection->search(partial);
  ASSERT_TRUE(best_effort.ok()) << best_effort.status().diagnostic();
  EXPECT_TRUE(best_effort.value().queries[0].hits.empty());
  EXPECT_EQ(best_effort.value().queries[0].completeness,
            core::SearchCompleteness::strategy_incomplete);
}

TEST(SegmentedCollection, StrictMissingVectorAndBudgetDenialHaveZeroEffectAndReleaseLease) {
  auto producer =
      std::make_shared<StaticSegment>(StaticSegment::Rows{{0, {0.0F, 0.0F}}, {1, {1.0F, 0.0F}}});
  SegmentRegistration registration;
  registration.segment_id = 42;
  registration.role = SegmentRole::sealed;
  registration.segment = readonly_any(producer);
  registration.rows = {
      {core::LogicalId::from_utf8("with-vector"),
       core::SegmentRowId(0),
       1,
       VersionState::live,
       owned_payload({0.0F, 0.0F}, {{"keep", true}})},
      {core::LogicalId::from_utf8("missing-vector"),
       core::SegmentRowId(1),
       2,
       VersionState::live,
       RecordPayload{std::nullopt, {{"keep", true}}, {}}},
  };
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();
  const std::array<float, 2> query{};
  core::SearchContext context;
  CollectionSearchStats stats;
  auto request = make_search_request(query.data(),
                                     1,
                                     2,
                                     context,
                                     LogicalFilter::metadata_equals("keep", true));
  request.options.filter_policy = core::FilterPolicy::strict;
  request.stats = &stats;
  auto missing = collection->search(request);
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), core::StatusCode::resource_exhausted);
  EXPECT_EQ(missing.status().detail(), core::StatusDetail::budget_denied);
  EXPECT_EQ(collection->outstanding_search_leases(), 0U);

  // Best effort does not require the missing exact fallback vector. Denial is
  // preflighted before any segment writes a result, and the same context is
  // immediately reusable once its budget is restored.
  request.options.filter_policy = core::FilterPolicy::allow_partial;
  request.filter = LogicalFilter{};
  context.query_scratch_lease.available_bytes = 1;
  auto denied = collection->search(request);
  ASSERT_FALSE(denied.ok());
  EXPECT_EQ(denied.status().code(), core::StatusCode::resource_exhausted);
  EXPECT_EQ(collection->outstanding_search_leases(), 0U);

  context.query_scratch_lease.available_bytes = core::kUnlimitedResource;
  auto retried = collection->search(request);
  ASSERT_TRUE(retried.ok()) << retried.status().diagnostic();
  EXPECT_EQ(retried.value().queries[0].hits.size(), 2U);
  EXPECT_EQ(stats.lease_acquired, 1U);
  EXPECT_EQ(stats.lease_released, 1U);
  EXPECT_GT(stats.budget_consumed, 0U);
  EXPECT_EQ(collection->outstanding_search_leases(), 0U);
}

TEST(SegmentedCollection, CrossSegmentQgUpsertDeleteSuppressesOldVersions) {
  using Space = RaBitQSpace<>;
  using Segment = QgSegment<Space>;
  // QgSegment::build() requires more live rows than its fixed degree bound
  // (32); dim stays 2 to keep pairing with FakeMutableSegment, whose wire
  // format/distance math is hardcoded to dim=2 (see fake_mutable_segment.hpp)
  // -- MatrixRotator (unlike the default FhtKacRotator) has no dim floor, so
  // it can rotate a dim=2 space same as any other.
  constexpr std::uint32_t kRows = 40;
  std::vector<float> data(kRows * 2);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    data[row * 2] = static_cast<float>(row);
    data[row * 2 + 1] = static_cast<float>(row);
  }
  auto space = std::make_shared<Space>(kRows, 2, core::Metric::l2, RotatorType::MatrixRotator);
  space->fit(data.data(), kRows);
  core::BuildContext build_context;
  QgBuildOptions build_options;
  build_options.ef_build = 100;
  build_options.thread_count = 1;
  auto qg = Segment::build({core::TypedTensorView::contiguous(data.data(), kRows, 2), space},
                           build_options,
                           build_context);
  auto qg_any = Segment::into_any(std::move(qg));
  ASSERT_TRUE(qg_any.ok());

  SegmentRegistration sealed;
  sealed.segment_id = 1;
  sealed.role = SegmentRole::sealed;
  sealed.segment = std::move(qg_any).value();
  for (std::uint32_t row = 0; row < kRows; ++row) {
    const std::array<float, 2> vector{data[row * 2], data[row * 2 + 1]};
    sealed.rows.push_back({core::LogicalId::from_utf8("qg-" + std::to_string(row)),
                           core::SegmentRowId(row),
                           0,
                           VersionState::live,
                           owned_payload(vector)});
  }
  auto mutable_producer = std::make_shared<FakeMutableSegment>();
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(sealed), fake_registration(mutable_producer)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  const auto collection = std::move(opened).value();
  const auto id = core::LogicalId::from_utf8("qg-0");
  const std::array<float, 2> replacement{20.0F, 20.0F};
  core::MutationContext mutation_context;
  auto updated = collection->write(write_request(id, replacement), mutation_context);
  ASSERT_TRUE(updated.ok());

  core::SearchContext search_context;
  auto result =
      collection->search(make_search_request(replacement.data(), 1, kRows, search_context));
  ASSERT_TRUE(result.ok());
  const auto duplicates = std::count_if(result.value().queries[0].hits.begin(),
                                        result.value().queries[0].hits.end(),
                                        [&](const CollectionHit &hit) {
                                          return hit.logical_id == id;
                                        });
  EXPECT_EQ(duplicates, 1);
  ASSERT_FALSE(result.value().queries[0].hits.empty());
  EXPECT_EQ(result.value().queries[0].hits.front().logical_id, id);
  EXPECT_EQ(result.value().queries[0].hits.front().upsert_sequence, updated.value().op_id);

  ASSERT_TRUE(collection->erase(id, mutation_context).ok());
  result = collection->search(make_search_request(replacement.data(), 1, kRows, search_context));
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(std::none_of(result.value().queries[0].hits.begin(),
                           result.value().queries[0].hits.end(),
                           [&](const CollectionHit &hit) {
                             return hit.logical_id == id;
                           }));
  EXPECT_EQ(collection->stats().size, kRows - 1);
}

TEST(SegmentedCollection, SnapshotReferenceDelaysSegmentReclamation) {
  auto destroyed = std::make_shared<std::atomic_bool>(false);
  auto producer = std::make_shared<StaticSegment>(StaticSegment::Rows{},
                                                  core::ScoreKind::distance,
                                                  false,
                                                  destroyed);
  SegmentRegistration registration;
  registration.segment_id = 99;
  registration.segment = readonly_any(producer);
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok());
  const auto collection = std::move(opened).value();
  auto pinned = collection->pin_routing_snapshot();
  producer.reset();
  ASSERT_TRUE(collection->retire_segment(99, 1).ok());
  EXPECT_FALSE(destroyed->load(std::memory_order_acquire));
  pinned.reset();
  EXPECT_TRUE(destroyed->load(std::memory_order_acquire));
}

TEST(SegmentedCollection, CloseThenDrainRejectsAllNewAdmissions) {
  const auto collection = open_fake_collection();
  ASSERT_TRUE(collection->close().ok());
  const std::array<float, 2> query{};
  core::SearchContext search_context;
  auto search = collection->search(make_search_request(query.data(), 1, 1, search_context));
  EXPECT_FALSE(search.ok());
  EXPECT_EQ(search.status().code(), core::StatusCode::closed);
  core::MutationContext mutation_context;
  auto write = collection->write(write_request(core::LogicalId::from_utf8("closed"), query),
                                 mutation_context);
  EXPECT_FALSE(write.ok());
  EXPECT_EQ(write.status().code(), core::StatusCode::closed);
  ASSERT_TRUE(collection->drain().ok());
  EXPECT_EQ(collection->stats().lifecycle, LifecycleState::closed);
}

TEST(SegmentedCollection, ExperimentalWriterIsOffByDefaultAndConfinedToNewNamespace) {
  const auto root =
      std::filesystem::temp_directory_path() / "alaya-segmented-collection-writer-test";
  std::filesystem::remove_all(root);
  auto disabled = open_fake_collection();
  EXPECT_EQ(disabled->persist_experimental_snapshot().code(), core::StatusCode::not_supported);
  EXPECT_FALSE(std::filesystem::exists(root));

  auto producer = std::make_shared<FakeMutableSegment>();
  CollectionConfig config;
  config.features.experimental_persistence_writer = true;
  config.persistence.root = root;
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {fake_registration(producer)},
                                          config);
  ASSERT_TRUE(opened.ok());
  const auto collection = std::move(opened).value();
  EXPECT_TRUE(collection->persist_experimental_snapshot().ok());
  EXPECT_TRUE(std::filesystem::exists(root / ".alaya_internal" / "collection_shell_v1" /
                                      "version_map_1.snapshot"));
  EXPECT_FALSE(std::filesystem::exists(root / "collection_manifest.txt"));
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace alaya::internal::collection
