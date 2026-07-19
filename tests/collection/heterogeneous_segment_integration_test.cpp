// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/collection/segmented_collection.hpp"
#include "index/disk/disk_flat_segment.hpp"
#include "platform/detect.hpp"

namespace alaya::internal::collection {
namespace {

constexpr std::uint32_t kDim = 64;
constexpr std::uint64_t kRowsPerSegment = 40;
constexpr std::uint64_t kLiveDuplicateId = 4242;
constexpr std::uint64_t kDeletedId = 4343;

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-heterogeneous-segment-test-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct Rows {
  std::vector<float> vectors{};
  std::vector<std::uint64_t> physical_ids{};
};

[[nodiscard]] auto make_rows(std::uint64_t first_id, std::uint32_t seed) -> Rows {
  Rows rows;
  rows.vectors.resize(kRowsPerSegment * kDim);
  rows.physical_ids.resize(kRowsPerSegment);
  std::mt19937 random(seed);
  std::uniform_real_distribution<float> distribution(-3.0F, 3.0F);
  for (std::uint64_t row = 0; row < kRowsPerSegment; ++row) {
    rows.physical_ids[row] = first_id + row;
    for (std::uint32_t column = 0; column < kDim; ++column) {
      rows.vectors[row * kDim + column] = distribution(random) + static_cast<float>(row) * 0.03125F;
    }
  }
  return rows;
}

[[nodiscard]] auto build_flat(const Rows &rows,
                              const std::filesystem::path &root,
                              std::string segment_id)
    -> std::unique_ptr<::alaya::disk::DiskFlatSegment> {
  ::alaya::disk::DiskFlatPublicationOptions options;
  options.collection_root = root;
  options.segment_id = std::move(segment_id);
  core::BuildContext context;
  auto built = ::alaya::disk::DiskFlatSegmentFactory::
      build({core::TypedTensorView::contiguous(rows.vectors.data(), kRowsPerSegment, kDim),
             rows.physical_ids},
            core::Metric::l2,
            options,
            context);
  if (!built.ok()) {
    throw std::runtime_error(built.status().diagnostic());
  }
  return std::move(built).value();
}

struct SearchStorage {
  SearchStorage(const float *queries, core::RowCount query_count, core::RowCount top_k)
      : hits(static_cast<std::size_t>(query_count * top_k)),
        offsets(static_cast<std::size_t>(query_count + 1)),
        counts(static_cast<std::size_t>(query_count)),
        statuses(static_cast<std::size_t>(query_count)),
        completeness(static_cast<std::size_t>(query_count)) {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    request.queries = core::TypedTensorView::contiguous(queries, query_count, kDim);
    request.options.top_k = top_k;
    request.context = &context;
    request.response = &response;
  }

  core::SearchContext context{};
  std::vector<core::SearchHit> hits{};
  std::vector<core::RowCount> offsets{};
  std::vector<core::RowCount> counts{};
  std::vector<core::Status> statuses{};
  std::vector<core::SearchCompleteness> completeness{};
  core::SearchResponse response{};
  core::SearchRequest request{};
};

template <typename Segment>
void expect_truncated_batch_contract(const Segment &segment, const float *queries) {
  constexpr core::RowCount kQueries = 2;
  constexpr core::RowCount kTopK = kRowsPerSegment + 7;
  SearchStorage storage(queries, kQueries, kTopK);
  const auto status = segment.batch_search(storage.request);
  ASSERT_TRUE(status.ok()) << status.diagnostic();
  EXPECT_EQ(storage.response.query_count, kQueries);
  EXPECT_EQ(storage.offsets.front(), 0U);
  for (std::size_t query = 0; query < kQueries; ++query) {
    EXPECT_TRUE(storage.statuses[query].ok()) << storage.statuses[query].diagnostic();
    EXPECT_EQ(storage.offsets[query + 1] - storage.offsets[query], storage.counts[query]);
    EXPECT_EQ(storage.counts[query], kRowsPerSegment);
    EXPECT_LE(storage.counts[query], kTopK);
    EXPECT_EQ(storage.completeness[query], core::SearchCompleteness::eligible_exhausted);
  }
}

[[nodiscard]] auto normal_registration(std::uint64_t segment_id,
                                       core::AnySegment segment,
                                       const Rows &rows,
                                       bool dense_row_ids) -> SegmentRegistration {
  SegmentRegistration registration;
  registration.segment_id = segment_id;
  registration.segment = std::move(segment);
  for (std::uint64_t row = 0; row < kRowsPerSegment; ++row) {
    registration.rows.push_back({core::LogicalId::from_legacy_uint64(rows.physical_ids[row]),
                                 core::SegmentRowId(dense_row_ids ? row : rows.physical_ids[row]),
                                 1,
                                 VersionState::live,
                                 {}});
  }
  return registration;
}

[[nodiscard]] auto make_collection_request(const float *queries,
                                           core::RowCount query_count,
                                           core::RowCount top_k,
                                           core::SearchContext &context)
    -> CollectionSearchRequest {
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(queries, query_count, kDim);
  request.options.top_k = top_k;
  request.context = &context;
  return request;
}

class PerQueryFailureProxy {
 public:
  PerQueryFailureProxy(core::AnySegment inner, core::RowCount failed_query)
      : inner_(std::move(inner)), failed_query_(failed_query) {}

  [[nodiscard]] auto descriptor() const noexcept -> core::Descriptor { return inner_.descriptor(); }

  [[nodiscard]] auto search(const core::SearchRequest &request) const -> core::Status {
    return execute(request);
  }

  [[nodiscard]] auto batch_search(const core::SearchRequest &request) const -> core::Status {
    return execute(request);
  }

  [[nodiscard]] auto stats(core::SegmentStats &stats) const -> core::Status {
    return inner_.stats(stats);
  }

 private:
  [[nodiscard]] auto execute(const core::SearchRequest &request) const -> core::Status {
    auto status = inner_.search(request);
    if (!status.ok() || request.response == nullptr || failed_query_ >= request.queries.rows) {
      return status;
    }
    auto &response = *request.response;
    const std::vector<core::RowCount> original_offsets(response.offsets.begin(),
                                                       response.offsets.end());
    std::vector<core::SearchHit> retained;
    retained.reserve(response.hits.size());
    core::RowCount cursor{};
    for (core::RowCount query = 0; query < request.queries.rows; ++query) {
      const auto begin = original_offsets[static_cast<std::size_t>(query)];
      const auto end = original_offsets[static_cast<std::size_t>(query + 1)];
      response.offsets[static_cast<std::size_t>(query)] = cursor;
      if (query == failed_query_) {
        response.valid_counts[static_cast<std::size_t>(query)] = 0;
        response.statuses[static_cast<std::size_t>(query)] =
            core::Status::error(core::StatusCode::io_error,
                                core::OperationStage::search,
                                core::StatusDetail::engine_exception,
                                "injected per-query disk read failure");
        response.completeness[static_cast<std::size_t>(query)] = core::SearchCompleteness::failed;
      } else {
        for (core::RowCount hit = begin; hit < end; ++hit) {
          retained.push_back(response.hits[static_cast<std::size_t>(hit)]);
        }
        cursor += end - begin;
      }
      response.offsets[static_cast<std::size_t>(query + 1)] = cursor;
    }
    std::copy(retained.begin(), retained.end(), response.hits.begin());
    return core::Status::success();
  }

  core::AnySegment inner_{};
  core::RowCount failed_query_{};
};

[[nodiscard]] auto make_failure_proxy(core::AnySegment inner, core::RowCount failed_query)
    -> core::AnySegment {
  auto proxy = std::make_shared<PerQueryFailureProxy>(std::move(inner), failed_query);
  core::SegmentInstanceConfig config;
  config.readonly = true;
  config.concurrency.reentrant_search = true;
  auto erased = core::AnySegment::from_sync(std::move(proxy), std::move(config));
  if (!erased.ok()) {
    throw std::runtime_error(erased.status().diagnostic());
  }
  return std::move(erased).value();
}

TEST(HeterogeneousSegmentIntegration, FlatSegmentsMatchOracleAndSuppressStaleDiskVersions) {
  TemporaryDirectory temporary;
  auto first_rows = make_rows(1000, 101);
  auto second_rows = make_rows(2000, 202);
  second_rows.physical_ids[kRowsPerSegment - 2] = 91'001;
  second_rows.physical_ids[kRowsPerSegment - 1] = 91'002;

  // The contract under test is cross-segment version arbitration, not an
  // engine-specific approximation. Two independently published Flat segments
  // preserve distinct segment identities and provide an exact oracle.
  auto first = build_flat(first_rows, temporary.path() / "first", "seg_00000001");
  auto second = build_flat(second_rows, temporary.path() / "second", "seg_00000002");

  std::vector<float> queries;
  queries.insert(queries.end(), first_rows.vectors.begin(), first_rows.vectors.begin() + kDim);
  queries.insert(queries.end(),
                 second_rows.vectors.begin() + 7 * kDim,
                 second_rows.vectors.begin() + 8 * kDim);
  expect_truncated_batch_contract(*first, queries.data());
  expect_truncated_batch_contract(*second, queries.data());

  auto first_any = ::alaya::disk::DiskFlatSegment::into_any(std::move(first));
  auto second_any = ::alaya::disk::DiskFlatSegment::into_any(std::move(second));
  ASSERT_TRUE(first_any.ok()) << first_any.status().diagnostic();
  ASSERT_TRUE(second_any.ok()) << second_any.status().diagnostic();

  auto first_registration =
      normal_registration(1, std::move(first_any).value(), first_rows, false);
  auto second_registration =
      normal_registration(2, std::move(second_any).value(), second_rows, false);

  // The first segment claims the same two logical ids at a newer sequence, so
  // it wins the live duplicate and supersedes the second copy with a tombstone.
  second_registration.rows[kRowsPerSegment - 2].logical_id =
      core::LogicalId::from_legacy_uint64(kLiveDuplicateId);
  second_registration.rows[kRowsPerSegment - 2].upsert_sequence = 1;
  second_registration.rows[kRowsPerSegment - 1].logical_id =
      core::LogicalId::from_legacy_uint64(kDeletedId);
  second_registration.rows[kRowsPerSegment - 1].upsert_sequence = 1;
  first_registration.rows[kRowsPerSegment - 2].logical_id =
      core::LogicalId::from_legacy_uint64(kLiveDuplicateId);
  first_registration.rows[kRowsPerSegment - 2].upsert_sequence = 2;
  first_registration.rows[kRowsPerSegment - 1].logical_id =
      core::LogicalId::from_legacy_uint64(kDeletedId);
  first_registration.rows[kRowsPerSegment - 1].upsert_sequence = 2;
  first_registration.rows[kRowsPerSegment - 1].state = VersionState::tombstone;

  auto opened =
      SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                {std::move(first_registration), std::move(second_registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto collection = std::move(opened).value();
  EXPECT_EQ(collection->stats().size, 2 * kRowsPerSegment - 3);

  Rows oracle_rows;
  auto append = [&](const Rows &source, std::uint64_t rows) {
    oracle_rows.vectors.insert(oracle_rows.vectors.end(),
                               source.vectors.begin(),
                               source.vectors.begin() + static_cast<std::ptrdiff_t>(rows * kDim));
    oracle_rows.physical_ids.insert(oracle_rows.physical_ids.end(),
                                    source.physical_ids.begin(),
                                    source.physical_ids.begin() +
                                        static_cast<std::ptrdiff_t>(rows));
  };
  append(first_rows, kRowsPerSegment - 2);
  append(second_rows, kRowsPerSegment - 2);
  oracle_rows.vectors.insert(oracle_rows.vectors.end(),
                             first_rows.vectors.end() - 2 * kDim,
                             first_rows.vectors.end() - kDim);
  oracle_rows.physical_ids.push_back(kLiveDuplicateId);

  ::alaya::disk::DiskFlatPublicationOptions oracle_options;
  oracle_options.collection_root = temporary.path() / "oracle";
  oracle_options.segment_id = "seg_00000003";
  core::BuildContext oracle_build_context;
  auto oracle = ::alaya::disk::DiskFlatSegmentFactory::
      build({core::TypedTensorView::contiguous(oracle_rows.vectors.data(),
                                               oracle_rows.physical_ids.size(),
                                               kDim),
             oracle_rows.physical_ids},
            core::Metric::l2,
            oracle_options,
            oracle_build_context);
  ASSERT_TRUE(oracle.ok()) << oracle.status().diagnostic();

  constexpr core::RowCount kQueryCount = 2;
  constexpr core::RowCount kTopK = 15;
  SearchStorage exact(queries.data(), kQueryCount, kTopK);
  ASSERT_TRUE(oracle.value()->batch_search(exact.request).ok());
  core::SearchContext mixed_context;
  auto mixed = collection->search(
      make_collection_request(queries.data(), kQueryCount, kTopK, mixed_context));
  ASSERT_TRUE(mixed.ok()) << mixed.status().diagnostic();
  ASSERT_EQ(mixed.value().queries.size(), kQueryCount);
  for (std::size_t query = 0; query < kQueryCount; ++query) {
    const auto &actual = mixed.value().queries[query];
    ASSERT_TRUE(actual.status.ok()) << actual.status.diagnostic();
    ASSERT_EQ(actual.completeness, core::SearchCompleteness::complete_k);
    ASSERT_EQ(actual.hits.size(), exact.counts[query]);
    std::vector<core::LogicalId> seen;
    for (std::size_t index = 0; index < actual.hits.size(); ++index) {
      const auto &oracle_hit = exact.hits[exact.offsets[query] + index];
      EXPECT_EQ(actual.hits[index].logical_id,
                core::LogicalId::from_legacy_uint64(static_cast<std::uint64_t>(oracle_hit.row_id)));
      EXPECT_EQ(std::bit_cast<std::uint32_t>(actual.hits[index].score),
                std::bit_cast<std::uint32_t>(oracle_hit.score));
      EXPECT_EQ(actual.hits[index].score_kind, core::ScoreKind::distance);
      EXPECT_EQ(actual.hits[index].comparable_metric, core::Metric::l2);
      seen.push_back(actual.hits[index].logical_id);
    }
    std::sort(seen.begin(), seen.end(), LogicalIdLess{});
    EXPECT_EQ(std::adjacent_find(seen.begin(), seen.end()), seen.end());
    EXPECT_EQ(std::find(seen.begin(), seen.end(), core::LogicalId::from_legacy_uint64(kDeletedId)),
              seen.end());
  }

  auto snapshot = collection->pin_routing_snapshot();
  const auto duplicate =
      snapshot->versions.find(core::LogicalId::from_legacy_uint64(kLiveDuplicateId));
  ASSERT_NE(duplicate, snapshot->versions.end());
  EXPECT_EQ(duplicate->second.address.segment_id, 1U);
  EXPECT_EQ(duplicate->second.upsert_sequence, 2U);
  const auto deleted = snapshot->versions.find(core::LogicalId::from_legacy_uint64(kDeletedId));
  ASSERT_NE(deleted, snapshot->versions.end());
  EXPECT_EQ(deleted->second.state, VersionState::tombstone);
}

TEST(HeterogeneousSegmentIntegration,
     DiskPerQueryFailureDoesNotPromoteToGlobalFailureAndOffsetsStayConsistent) {
  TemporaryDirectory temporary;
  auto rows = make_rows(7000, 707);
  auto flat = build_flat(rows, temporary.path() / "failure", "seg_00000004");
  auto erased = ::alaya::disk::DiskFlatSegment::into_any(std::move(flat));
  ASSERT_TRUE(erased.ok()) << erased.status().diagnostic();
  auto registration =
      normal_registration(4, make_failure_proxy(std::move(erased).value(), 1), rows, false);
  auto opened = SegmentedCollection::open({kDim, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();

  std::vector<float> queries;
  for (const auto row : {0U, 5U, 11U}) {
    queries.insert(queries.end(),
                   rows.vectors.begin() + static_cast<std::ptrdiff_t>(row * kDim),
                   rows.vectors.begin() + static_cast<std::ptrdiff_t>((row + 1) * kDim));
  }
  core::SearchContext context;
  auto result =
      std::move(opened).value()->search(make_collection_request(queries.data(), 3, 7, context));
  ASSERT_TRUE(result.ok()) << result.status().diagnostic();
  ASSERT_EQ(result.value().queries.size(), 3U);

  std::array<core::RowCount, 4> offsets{};
  for (std::size_t query = 0; query < result.value().queries.size(); ++query) {
    const auto &row = result.value().queries[query];
    offsets[query + 1] = offsets[query] + row.hits.size();
    EXPECT_EQ(offsets[query + 1] - offsets[query], row.hits.size());
    EXPECT_LE(row.hits.size(), 7U);
  }
  EXPECT_TRUE(result.value().queries[0].status.ok());
  EXPECT_EQ(result.value().queries[0].completeness, core::SearchCompleteness::complete_k);
  EXPECT_EQ(result.value().queries[0].hits.size(), 7U);
  EXPECT_EQ(result.value().queries[1].status.code(), core::StatusCode::io_error);
  EXPECT_EQ(result.value().queries[1].completeness, core::SearchCompleteness::failed);
  EXPECT_TRUE(result.value().queries[1].hits.empty());
  EXPECT_TRUE(result.value().queries[2].status.ok());
  EXPECT_EQ(result.value().queries[2].completeness, core::SearchCompleteness::complete_k);
  EXPECT_EQ(result.value().queries[2].hits.size(), 7U);
}

}  // namespace
}  // namespace alaya::internal::collection
