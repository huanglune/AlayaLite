// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "index/graph/detail/search_runtime/graph_search_job.hpp"
#include "index/collection/legacy_disk_adapter.hpp"
#include "index/collection/legacy_memory_adapter.hpp"
#include "index/collection/segmented_collection.hpp"
#include "index/graph/hnsw/detail/hnsw_segment_bridge.hpp"
#include "index/graph/hnsw/hnsw_segment.hpp"
#include "space/raw_space.hpp"

namespace alaya::internal::collection {
namespace {

using Space = RawSpace<>;
using Segment = HnswSegment<Space>;
using SearchJob = GraphSearchJob<Space, Space>;

struct MemoryFixture {
  std::vector<float> data{};
  std::shared_ptr<Space> space{};
  std::shared_ptr<SearchJob> legacy_job{};
  std::unique_ptr<Segment> segment{};
};

[[nodiscard]] auto build_memory_fixture(std::uint32_t rows = 24) -> MemoryFixture {
  MemoryFixture fixture;
  fixture.data.resize(rows * 2);
  for (std::uint32_t row = 0; row < rows; ++row) {
    fixture.data[row * 2] = static_cast<float>(row);
    fixture.data[row * 2 + 1] = static_cast<float>(row % 5);
  }
  fixture.space = std::make_shared<Space>(rows + 8, 2, core::Metric::l2);
  fixture.space->fit(fixture.data.data(), rows);
  core::BuildContext context;
  fixture.segment = Segment::build({core::TypedTensorView::contiguous(fixture.data.data(), rows, 2),
                                    fixture.space,
                                    fixture.space},
                                   {.max_neighbors = 6, .ef_construction = 32, .thread_count = 1},
                                   context);
  fixture.legacy_job =
      std::make_shared<SearchJob>(detail::HnswSegmentBridge<Space, Space>::search_space(
                                      *fixture.segment),
                                  detail::HnswSegmentBridge<Space, Space>::graph(*fixture.segment),
                                  nullptr,
                                  detail::HnswSegmentBridge<Space, Space>::build_space(
                                      *fixture.segment));
  return fixture;
}

[[nodiscard]] auto payload_for(const float *vector) -> RecordPayload {
  auto owned = OwnedVector::copy_row(core::TypedTensorView::contiguous(vector, 1, 2), 0);
  EXPECT_TRUE(owned.ok());
  RecordPayload payload;
  if (owned.ok()) {
    payload.vector = std::move(owned).value();
  }
  return payload;
}

[[nodiscard]] auto search_request(const float *query,
                                  std::uint64_t top_k,
                                  core::SearchContext &context) -> CollectionSearchRequest {
  CollectionSearchRequest request;
  request.queries = core::TypedTensorView::contiguous(query, 1, 2);
  request.options.top_k = top_k;
  request.context = &context;
  return request;
}

[[nodiscard]] auto memory_descriptor() -> core::Descriptor {
  core::Descriptor descriptor;
  descriptor.algorithm_id = core::algorithm::hnsw;
  descriptor.format_version = 1;
  descriptor.factory_version = 1;
  descriptor.dim = 2;
  descriptor.metric = core::Metric::l2;
  descriptor.stored_scalar_type = core::ScalarType::float32;
  descriptor.medium = core::Medium::memory;
  descriptor.engine_factory_id = core::algorithm::hnsw;
  return descriptor;
}

[[nodiscard]] auto disk_descriptor() -> core::Descriptor {
  auto descriptor = memory_descriptor();
  descriptor.algorithm_id = core::algorithm::flat;
  descriptor.engine_factory_id = core::algorithm::flat;
  descriptor.medium = core::Medium::disk;
  return descriptor;
}

TEST(LegacyMemoryAdapter, MatchesRawGraphSearchJobAndHasIndependentRuntimeFlag) {
  auto fixture = build_memory_fixture();
  constexpr std::uint32_t kTopK = 5;
  const std::array<float, 2> query{7.25F, 2.0F};
  std::array<std::uint32_t, kTopK> direct_ids{};
  std::array<float, kTopK> direct_distances{};
  fixture.legacy_job->search_solo(const_cast<float *>(query.data()),
                                  direct_ids.data(),
                                  direct_distances.data(),
                                  kTopK,
                                  100);

  CollectionFeatureFlags disabled;
  disabled.legacy_memory_adapter = false;
  auto unavailable =
      make_legacy_memory_segment<float, std::uint32_t, float>(fixture.legacy_job,
                                                              memory_descriptor(),
                                                              fixture.data.size() / 2,
                                                              disabled);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), core::StatusCode::not_supported);
  std::array<std::uint32_t, kTopK> direct_after_disable{};
  std::array<float, kTopK> direct_distances_after_disable{};
  fixture.legacy_job->search_solo(const_cast<float *>(query.data()),
                                  direct_after_disable.data(),
                                  direct_distances_after_disable.data(),
                                  kTopK,
                                  100);
  EXPECT_EQ(direct_after_disable, direct_ids);

  CollectionFeatureFlags enabled;
  auto erased = make_legacy_memory_segment<float, std::uint32_t, float>(fixture.legacy_job,
                                                                        memory_descriptor(),
                                                                        fixture.data.size() / 2,
                                                                        enabled);
  ASSERT_TRUE(erased.ok());
  EXPECT_FALSE(erased.value().capabilities().supports(core::OperationCapability::mutation));
  SegmentRegistration registration;
  registration.segment_id = 20;
  registration.role = SegmentRole::legacy_readonly;
  registration.segment = std::move(erased).value();
  for (std::uint32_t row = 0; row < fixture.data.size() / 2; ++row) {
    registration.rows.push_back(
        {core::LogicalId::from_legacy_uint64(row),
         core::SegmentRowId(row),
         0,
         VersionState::live,
         payload_for(fixture.data.data() + static_cast<std::size_t>(row) * 2)});
  }
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok());
  core::SearchContext context;
  auto routed = std::move(opened).value()->search(search_request(query.data(), kTopK, context));
  ASSERT_TRUE(routed.ok());
  ASSERT_EQ(routed.value().queries[0].hits.size(), kTopK);
  for (std::size_t index = 0; index < kTopK; ++index) {
    EXPECT_EQ(routed.value().queries[0].hits[index].source.row_id,
              core::SegmentRowId(direct_ids[index]));
    EXPECT_FLOAT_EQ(routed.value().queries[0].hits[index].score, direct_distances[index]);
  }
}

TEST(LegacyDiskAdapter, MatchesDirectDiskCollectionAndNeverExposesMutation) {
  const auto root =
      std::filesystem::temp_directory_path() / "alaya-legacy-disk-collection-adapter-test";
  std::filesystem::remove_all(root);
  auto disk_collection =
      std::make_shared<disk::DiskCollection>(root, 2, core::Metric::l2, disk::DiskIndexType::Flat);
  const std::array<float, 8> vectors{0.0F, 0.0F, 2.0F, 2.0F, 5.0F, 5.0F, 9.0F, 9.0F};
  const std::array<std::uint64_t, 4> labels{100, 200, 300, 400};
  disk_collection->add_batch(vectors.data(), labels.data(), labels.size());
  disk_collection->flush();
  const std::array<float, 2> query{4.5F, 4.5F};
  disk::DiskSearchOptions options;
  options.top_k = 3;
  const auto direct = disk_collection->search(query.data(), options);

  CollectionFeatureFlags disabled;
  disabled.legacy_disk_adapter = false;
  auto unavailable = make_legacy_disk_segment(disk_collection, disk_descriptor(), disabled, false);
  ASSERT_FALSE(unavailable.ok());
  EXPECT_EQ(unavailable.status().code(), core::StatusCode::not_supported);
  EXPECT_EQ(disk_collection->search(query.data(), options).size(), direct.size());

  CollectionFeatureFlags enabled;
  auto erased = make_legacy_disk_segment(disk_collection, disk_descriptor(), enabled, false);
  ASSERT_TRUE(erased.ok());
  EXPECT_FALSE(erased.value().capabilities().supports(core::OperationCapability::mutation));
  SegmentRegistration registration;
  registration.segment_id = 30;
  registration.role = SegmentRole::legacy_readonly;
  registration.segment = std::move(erased).value();
  for (std::size_t row = 0; row < labels.size(); ++row) {
    registration.rows.push_back({core::LogicalId::from_legacy_uint64(labels[row]),
                                 core::SegmentRowId(labels[row]),
                                 0,
                                 VersionState::live,
                                 payload_for(vectors.data() + row * 2)});
  }
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(registration)});
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  core::SearchContext context;
  auto routed = collection->search(search_request(query.data(), options.top_k, context));
  ASSERT_TRUE(routed.ok());
  ASSERT_EQ(routed.value().queries[0].hits.size(), direct.size());
  for (std::size_t index = 0; index < direct.size(); ++index) {
    EXPECT_EQ(routed.value().queries[0].hits[index].source.row_id,
              core::SegmentRowId(direct[index].label));
    EXPECT_FLOAT_EQ(routed.value().queries[0].hits[index].score, direct[index].distance);
  }
  collection.reset();
  disk_collection.reset();
  std::filesystem::remove_all(root);
}

TEST(LegacyAdapters, MemoryAndDiskSlotsMixWithHnswProducerInOneSnapshot) {
  const auto root = std::filesystem::temp_directory_path() / "alaya-mixed-routing-adapter-test";
  std::filesystem::remove_all(root);
  auto fixture = build_memory_fixture(12);
  CollectionFeatureFlags features;
  auto memory_erased =
      make_legacy_memory_segment<float, std::uint32_t, float>(fixture.legacy_job,
                                                              memory_descriptor(),
                                                              fixture.data.size() / 2,
                                                              features);
  ASSERT_TRUE(memory_erased.ok());
  auto hnsw_erased = Segment::into_any(std::move(fixture.segment));
  ASSERT_TRUE(hnsw_erased.ok());

  auto disk_collection =
      std::make_shared<disk::DiskCollection>(root, 2, core::Metric::l2, disk::DiskIndexType::Flat);
  const std::array<float, 4> disk_vectors{50.0F, 50.0F, 60.0F, 60.0F};
  const std::array<std::uint64_t, 2> disk_labels{500, 600};
  disk_collection->add_batch(disk_vectors.data(), disk_labels.data(), disk_labels.size());
  disk_collection->flush();
  auto disk_erased = make_legacy_disk_segment(disk_collection, disk_descriptor(), features);
  ASSERT_TRUE(disk_erased.ok());

  SegmentRegistration hnsw_registration;
  hnsw_registration.segment_id = 40;
  hnsw_registration.segment = std::move(hnsw_erased).value();
  SegmentRegistration memory_registration;
  memory_registration.segment_id = 41;
  memory_registration.role = SegmentRole::legacy_readonly;
  memory_registration.segment = std::move(memory_erased).value();
  for (std::uint32_t row = 0; row < fixture.data.size() / 2; ++row) {
    const auto *vector = fixture.data.data() + static_cast<std::size_t>(row) * 2;
    hnsw_registration.rows.push_back({core::LogicalId::from_utf8("hnsw-" + std::to_string(row)),
                                      core::SegmentRowId(row),
                                      0,
                                      VersionState::live,
                                      payload_for(vector)});
    memory_registration.rows.push_back(
        {core::LogicalId::from_utf8("legacy-memory-" + std::to_string(row)),
         core::SegmentRowId(row),
         0,
         VersionState::live,
         payload_for(vector)});
  }
  SegmentRegistration disk_registration;
  disk_registration.segment_id = 42;
  disk_registration.role = SegmentRole::legacy_readonly;
  disk_registration.segment = std::move(disk_erased).value();
  for (std::size_t row = 0; row < disk_labels.size(); ++row) {
    disk_registration.rows.push_back({core::LogicalId::from_legacy_uint64(disk_labels[row]),
                                      core::SegmentRowId(disk_labels[row]),
                                      0,
                                      VersionState::live,
                                      payload_for(disk_vectors.data() + row * 2)});
  }
  auto opened = SegmentedCollection::open({2, core::Metric::l2, core::ScalarType::float32},
                                          {std::move(hnsw_registration),
                                           std::move(memory_registration),
                                           std::move(disk_registration)});
  ASSERT_TRUE(opened.ok());
  auto collection = std::move(opened).value();
  const std::array<float, 2> query{0.0F, 0.0F};
  core::SearchContext context;
  auto routed = collection->search(search_request(query.data(), 26, context));
  ASSERT_TRUE(routed.ok());
  std::set<std::uint64_t> sources;
  for (const auto &hit : routed.value().queries[0].hits) {
    sources.insert(hit.source.segment_id);
  }
  EXPECT_TRUE(sources.contains(40));
  EXPECT_TRUE(sources.contains(41));
  EXPECT_TRUE(sources.contains(42));
  collection.reset();
  disk_collection.reset();
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace alaya::internal::collection
