// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "core/any_segment.hpp"

namespace {

using namespace alaya::core;

struct SearchOnly {
  auto descriptor() const noexcept -> Descriptor { return {}; }
  auto search(const SearchRequest &) const -> Status { return Status::success(); }
};

struct FullSegment : SearchOnly {
  auto batch_search(const SearchRequest &) const -> Status { return Status::success(); }
  auto prepare_mutation(const OpaqueOperationRequest &, MutationContext &, MutationToken &)
      -> Status {
    return Status::success();
  }
  auto stage_mutation(MutationToken &, MutationContext &) -> Status { return Status::success(); }
  auto publish_mutation(MutationToken &, MutationContext &) -> Status { return Status::success(); }
  auto abort_mutation(MutationToken &, MutationContext &) -> Status { return Status::success(); }
  auto replay_mutation(const OpaqueOperationRequest &, MutationContext &) -> Status {
    return Status::success();
  }
  auto save(ArtifactWriter &, const SaveOptions &, ArtifactManifest &) const -> Status {
    return Status::success();
  }
  auto checkpoint(CheckpointContext &, CheckpointToken &) -> Status { return Status::success(); }
  auto freeze_snapshot(SealContext &, FreezeToken &) -> Status { return Status::success(); }
  auto stats(SegmentStats &) const noexcept -> Status { return Status::success(); }
};

struct WrongSearchReturn {
  auto descriptor() const noexcept -> Descriptor { return {}; }
  auto search(const SearchRequest &) const -> std::size_t { return 0; }
};

struct MutableWithoutSearch {
  auto prepare_mutation(const OpaqueOperationRequest &, MutationContext &, MutationToken &)
      -> Status {
    return Status::success();
  }
};

struct Builder {
  auto build(const OpaqueOperationRequest &, BuildContext &, SealedArtifact &) -> Status {
    return Status::success();
  }
};

static_assert(std::is_same_v<RowCount, std::uint64_t>);
static_assert(sizeof(SegmentRowId) == sizeof(std::uint64_t));
static_assert(std::is_trivially_copyable_v<TypedTensorView>);
static_assert(std::is_trivially_copyable_v<SearchHit>);

// Same-toolchain layout regression canaries. These catch accidental field
// insertion; they are not a cross-compiler or cross-stdlib ABI promise.
static_assert(sizeof(VersionedStructHeader) == 8 && alignof(VersionedStructHeader) == 4);
static_assert(sizeof(Status) == 64 && alignof(Status) == 8);
static_assert(sizeof(LogicalIdView) == 56 && alignof(LogicalIdView) == 8);
static_assert(sizeof(TypedTensorView) == 80 && alignof(TypedTensorView) == 8);
static_assert(sizeof(AlgorithmSearchExtension) == 64 && alignof(AlgorithmSearchExtension) == 8);
static_assert(sizeof(SearchOptions) == 88 && alignof(SearchOptions) == 8);
static_assert(sizeof(SearchHit) == 64 && alignof(SearchHit) == 8);
static_assert(sizeof(SearchResponse) == 144 && alignof(SearchResponse) == 8);
static_assert(sizeof(SearchRequest) == 312 && alignof(SearchRequest) == 8);
static_assert(sizeof(Descriptor) == 80 && alignof(Descriptor) == 8);
static_assert(sizeof(MemoryReservation) == 96 && alignof(MemoryReservation) == 8);
static_assert(sizeof(OpenContext) == 384 && alignof(OpenContext) == 8);
static_assert(sizeof(BuildContext) == 352 && alignof(BuildContext) == 8);
static_assert(sizeof(MutationContext) == 496 && alignof(MutationContext) == 8);
static_assert(sizeof(SealContext) == 448 && alignof(SealContext) == 8);
static_assert(sizeof(SearchContext) == 296 && alignof(SearchContext) == 8);
static_assert(sizeof(CheckpointContext) == 304 && alignof(CheckpointContext) == 8);
static_assert(sizeof(SegmentStats) == 144 && alignof(SegmentStats) == 8);
static_assert(sizeof(AnySegmentOperationTable) == 184 && alignof(AnySegmentOperationTable) == 8);
static_assert(sizeof(OperationHandle) == 64 && alignof(OperationHandle) == 8);

static_assert(DescriptorProvider<SearchOnly>);
static_assert(Searchable<SearchOnly>);
static_assert(!BatchSearchable<SearchOnly>);
static_assert(!Mutable<SearchOnly>);
static_assert(!Saveable<SearchOnly>);

static_assert(Searchable<FullSegment>);
static_assert(BatchSearchable<FullSegment>);
static_assert(Mutable<FullSegment>);
static_assert(Saveable<FullSegment>);
static_assert(Checkpointable<FullSegment>);
static_assert(Freezable<FullSegment>);
static_assert(StatsProvider<FullSegment>);

static_assert(!Searchable<WrongSearchReturn>);
static_assert(!Mutable<MutableWithoutSearch>);
static_assert(BuildFactory<Builder>);

TEST(CoreV3Versioning, EveryBoundaryValueStartsAtCurrentVersionAndSize) {
  TypedTensorView tensor;
  SearchOptions options;
  SearchResponse response;
  SearchRequest request;
  SearchContext context;
  SegmentStats stats;

  EXPECT_EQ(tensor.header.struct_size, sizeof(tensor));
  EXPECT_EQ(options.header.struct_size, sizeof(options));
  EXPECT_EQ(response.header.struct_size, sizeof(response));
  EXPECT_EQ(request.header.struct_size, sizeof(request));
  EXPECT_EQ(context.header.struct_size, sizeof(context));
  EXPECT_EQ(stats.header.struct_size, sizeof(stats));
  EXPECT_EQ(tensor.header.abi_version, kContractAbiVersion);
  EXPECT_EQ(options.header.abi_version, kContractAbiVersion);
}

TEST(CoreV3LogicalId, OwnsUtf8AndLegacyIdentityCanonicalBytes) {
  auto string_id = LogicalId::from_utf8("alpha");
  auto same_string = LogicalId::from_utf8("alpha");
  auto later_string = LogicalId::from_utf8("beta");
  auto legacy = LogicalId::from_legacy_uint64(0x0102030405060708ULL);

  EXPECT_EQ(string_id, same_string);
  EXPECT_LT(string_id.compare(later_string), 0);
  EXPECT_NE(string_id, legacy);
  ASSERT_EQ(legacy.canonical_bytes().size(), 8U);
  EXPECT_EQ(std::to_integer<unsigned>(legacy.canonical_bytes().front()), 0x01U);
  EXPECT_EQ(std::to_integer<unsigned>(legacy.canonical_bytes().back()), 0x08U);
  EXPECT_EQ(legacy.view().kind, LogicalIdKind::legacy_uint64);
}

TEST(CoreV3TypedTensor, AcceptsAllFrozenScalarTypes) {
  std::array<float, 6> floats{};
  std::array<std::int8_t, 6> signed_bytes{};
  std::array<std::uint8_t, 6> unsigned_bytes{};

  const auto f32 = TypedTensorView::contiguous(floats.data(), 2, 3);
  const auto i8 = TypedTensorView::contiguous(signed_bytes.data(), 2, 3);
  const auto u8 = TypedTensorView::contiguous(unsigned_bytes.data(), 2, 3);

  EXPECT_TRUE(validate_tensor(f32, 3, OperationStage::validation).ok());
  EXPECT_TRUE(validate_tensor(i8, 3, OperationStage::validation).ok());
  EXPECT_TRUE(validate_tensor(u8, 3, OperationStage::validation).ok());
  EXPECT_EQ(f32.scalar_type, ScalarType::float32);
  EXPECT_EQ(i8.scalar_type, ScalarType::int8);
  EXPECT_EQ(u8.scalar_type, ScalarType::uint8);
  EXPECT_EQ(f32.row_stride, 3U * sizeof(float));
}

TEST(CoreV3TypedTensor, AppliesEmptyNullStrideAndOverflowRules) {
  const TypedTensorView empty(nullptr, ScalarType::float32, 0, 4, 16);
  EXPECT_TRUE(validate_tensor(empty, 4, OperationStage::validation).ok());

  const TypedTensorView missing(nullptr, ScalarType::float32, 1, 4, 16);
  EXPECT_EQ(validate_tensor(missing, 4, OperationStage::validation).code(),
            StatusCode::invalid_argument);

  std::array<float, 4> row{};
  const TypedTensorView short_stride(row.data(), ScalarType::float32, 1, 4, 12);
  EXPECT_EQ(validate_tensor(short_stride, 4, OperationStage::validation).detail(),
            StatusDetail::invalid_stride);

  const TypedTensorView overflow(row.data(),
                                 ScalarType::float32,
                                 2,
                                 1,
                                 std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(validate_tensor(overflow, 1, OperationStage::validation).detail(),
            StatusDetail::arithmetic_overflow);
}

TEST(CoreV3Resources, GrowingReservationHasAStableDenialAndGrowthPath) {
  MemoryReservation denied(0);
  auto status = denied.ensure(64, OperationStage::build, "test reservation denied");
  EXPECT_EQ(status.code(), StatusCode::resource_exhausted);
  EXPECT_EQ(status.detail(), StatusDetail::budget_denied);

  MemoryReservation growing(0);
  std::uint64_t observed{};
  growing.state = &observed;
  growing.grow = [](void *raw, std::uint64_t bytes, MemoryLease &lease) noexcept {
    *static_cast<std::uint64_t *>(raw) = bytes;
    lease.available_bytes = bytes;
    return Status::success();
  };
  EXPECT_TRUE(growing.ensure(96, OperationStage::build, "unused").ok());
  EXPECT_EQ(observed, 96U);
  EXPECT_TRUE(growing.permits(96));
}

TEST(CoreV3Response, EnforcesPerQueryOffsetsCountsAndFailureInvalidation) {
  std::array<SearchHit, 6> hits{};
  std::array<RowCount, 3> offsets{};
  std::array<RowCount, 2> counts{};
  std::array<Status, 2> statuses{};
  std::array<SearchCompleteness, 2> completeness{};
  SearchResponse response;
  response.hits = hits;
  response.offsets = offsets;
  response.valid_counts = counts;
  response.statuses = statuses;
  response.completeness = completeness;

  ASSERT_TRUE(validate_response(response, 2, 3, OperationStage::validation).ok());
  response.offsets[0] = 0;
  response.offsets[1] = 2;
  response.offsets[2] = 3;
  response.valid_counts[0] = 2;
  response.valid_counts[1] = 1;
  EXPECT_EQ(response.offsets[1] - response.offsets[0], response.valid_counts[0]);
  EXPECT_EQ(response.offsets[2] - response.offsets[1], response.valid_counts[1]);

  const auto failure = Status::error(StatusCode::corruption,
                                     OperationStage::search,
                                     StatusDetail::engine_exception,
                                     "fake corruption");
  response.invalidate(failure);
  EXPECT_EQ(offsets, (std::array<RowCount, 3>{0, 0, 0}));
  EXPECT_EQ(counts, (std::array<RowCount, 2>{0, 0}));
  EXPECT_EQ(response.statuses[0].code(), StatusCode::corruption);
  EXPECT_EQ(response.statuses[1].code(), StatusCode::corruption);
  EXPECT_EQ(response.completeness[0], SearchCompleteness::failed);
}

TEST(CoreV3Response, RejectsRowsTimesTopKOverflowBeforeTouchingSink) {
  SearchResponse response;
  std::array<RowCount, 1> offsets{};
  response.offsets = offsets;
  const auto status = validate_response(response,
                                        std::numeric_limits<RowCount>::max(),
                                        2,
                                        OperationStage::validation);
  EXPECT_EQ(status.code(), StatusCode::invalid_argument);
  EXPECT_EQ(status.detail(), StatusDetail::arithmetic_overflow);
}

}  // namespace
