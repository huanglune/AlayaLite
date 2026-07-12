// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

#include <gtest/gtest.h>

#include "core/capabilities.hpp"
#include "core/compat.hpp"

namespace {

using namespace alaya::core;

struct SearchOnly {
  auto descriptor() const noexcept -> Descriptor { return {}; }
  auto search(QueryView, const SearchOptions &, SearchSink) const -> SearchResult { return {}; }
};

struct FullSegment : SearchOnly {
  auto batch_search(QueryBatchView, const SearchOptions &, SearchSink) const -> BatchSearchResult {
    return {};
  }
  auto insert(VectorBatchView, MutationContext &) -> MutationResult { return {}; }
  auto erase(std::span<const ExternalId>, MutationContext &) -> MutationResult { return {}; }
  auto checkpoint(CheckpointContext &) -> CheckpointToken { return {}; }
  auto seal(SealContext &) -> SealedArtifact { return {}; }
  auto supports_filter(const void *) const noexcept -> FilterSupport {
    return FilterSupport::traversal;
  }
};

struct WrongSearchReturn {
  auto descriptor() const noexcept -> Descriptor { return {}; }
  auto search(QueryView, const SearchOptions &, SearchSink) const -> std::size_t { return 0; }
};

struct MutableWithoutSearch {
  auto insert(VectorBatchView, MutationContext &) -> MutationResult { return {}; }
  auto erase(std::span<const ExternalId>, MutationContext &) -> MutationResult { return {}; }
};

struct Builder {
  auto build(VectorBatchView, BuildContext &) -> SealedArtifact { return {}; }
};

static_assert(std::same_as<ExternalId, std::uint64_t>);
static_assert(std::same_as<RowCount, std::uint64_t>);
static_assert(std::is_trivially_copyable_v<QueryView>);
static_assert(std::is_trivially_copyable_v<VectorBatchView>);
static_assert(std::is_trivially_copyable_v<SearchOptions>);
static_assert(std::is_trivially_copyable_v<SearchHit>);
static_assert(std::is_trivially_copyable_v<Descriptor>);
static_assert(std::is_trivially_copyable_v<ArtifactManifest>);

static_assert(sizeof(QueryView) == 16 && alignof(QueryView) == 8);
static_assert(sizeof(QueryBatchView) == 24 && alignof(QueryBatchView) == 8);
static_assert(sizeof(VectorBatchView) == 32 && alignof(VectorBatchView) == 8);
static_assert(sizeof(SearchOptions) == 32 && alignof(SearchOptions) == 8);
static_assert(sizeof(SearchHit) == 16 && alignof(SearchHit) == 8);
static_assert(sizeof(Descriptor) == 40 && alignof(Descriptor) == 8);
static_assert(sizeof(Artifact) == 32 && alignof(Artifact) == 8);
static_assert(sizeof(ArtifactManifest) == 32 && alignof(ArtifactManifest) == 8);

static_assert(DescriptorProvider<SearchOnly>);
static_assert(Searchable<SearchOnly>);
static_assert(!BatchSearchable<SearchOnly>);
static_assert(!Mutable<SearchOnly>);
static_assert(!Persistable<SearchOnly>);
static_assert(!Sealable<SearchOnly>);
static_assert(!Filterable<SearchOnly>);

static_assert(Searchable<FullSegment>);
static_assert(BatchSearchable<FullSegment>);
static_assert(Mutable<FullSegment>);
static_assert(Persistable<FullSegment>);
static_assert(Sealable<FullSegment>);
static_assert(Filterable<FullSegment>);

static_assert(!Searchable<WrongSearchReturn>);
static_assert(!Mutable<MutableWithoutSearch>);
static_assert(SegmentBuilder<Builder>);

TEST(CoreCompat, ConvertsLegacyDiskSearchValuesWithoutChangingSemantics) {
  const alaya::disk::DiskSearchOptions legacy{7, 81, 5, false};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  const auto options = alaya::core::compat::from_disk_search_options(legacy);
  const auto roundtrip = alaya::core::compat::to_disk_search_options(options);
  const alaya::disk::DiskSearchHit legacy_hit{std::numeric_limits<std::uint64_t>::max(), -2.5F};
  const auto hit = alaya::core::compat::from_disk_search_hit(legacy_hit);
  const auto roundtrip_hit = alaya::core::compat::to_disk_search_hit(hit);
#pragma GCC diagnostic pop

  EXPECT_EQ(options.top_k, 7U);
  EXPECT_EQ(options.effort, 81U);
  EXPECT_EQ(options.beam_width, 5U);
  EXPECT_FALSE(options.exact_rerank);
  EXPECT_EQ(roundtrip.top_k, legacy.top_k);
  EXPECT_EQ(roundtrip.ef, legacy.ef);
  EXPECT_EQ(roundtrip.beam_width, legacy.beam_width);
  EXPECT_EQ(roundtrip.exact_rerank, legacy.exact_rerank);
  EXPECT_EQ(hit.id, legacy_hit.label);
  EXPECT_FLOAT_EQ(hit.distance, legacy_hit.distance);
  EXPECT_EQ(roundtrip_hit.label, legacy_hit.label);
  EXPECT_FLOAT_EQ(roundtrip_hit.distance, legacy_hit.distance);
}

TEST(CoreCompat, ConvertsEveryValidLegacyMetricAndRejectsNone) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  EXPECT_EQ(alaya::core::compat::from_metric_type(alaya::MetricType::L2), Metric::l2);
  EXPECT_EQ(alaya::core::compat::from_metric_type(alaya::MetricType::IP), Metric::inner_product);
  EXPECT_EQ(alaya::core::compat::from_metric_type(alaya::MetricType::COS), Metric::cosine);
  EXPECT_EQ(alaya::core::compat::to_metric_type(Metric::l2), alaya::MetricType::L2);
  EXPECT_EQ(alaya::core::compat::from_index_type(alaya::IndexType::HNSW),
            alaya::core::compat::kAlgorithmHnsw);
  EXPECT_EQ(alaya::core::compat::from_disk_index_type(alaya::disk::DiskIndexType::Laser),
            alaya::core::compat::kAlgorithmLaser);
  EXPECT_THROW((void)alaya::core::compat::from_metric_type(alaya::MetricType::NONE),
               std::invalid_argument);
#pragma GCC diagnostic pop
}

}  // namespace
