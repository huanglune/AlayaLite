// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/detail/collection_target_builder.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-qg-no-laser-contract-" + std::to_string(platform::get_pid()));
    std::filesystem::remove_all(path_);
  }

  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

TEST(CollectionQgPlatformContract, ValidQgSealRejectsWithoutLaserAndNeverPublishesFlat) {
#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  GTEST_SKIP() << "this target must compile without the LASER implementation surface";
#else
  constexpr std::uint32_t kDim = 64;
  constexpr core::RowCount kRows = 40;
  TemporaryDirectory temporary;

  CollectionOptions options;
  options.root = temporary.path();
  options.dim = kDim;
  options.metric = core::Metric::l2;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::qg;
  options.quantization = CollectionQuantization::rabitq;
  auto created = Collection::create(options);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(collection->target_implementation_key(), "qg_laser_segment");

  std::vector<float> vectors(static_cast<std::size_t>(kRows) * kDim);
  std::vector<CollectionItem> items;
  items.reserve(kRows);
  for (core::RowCount row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kDim; ++column) {
      vectors[static_cast<std::size_t>(row) * kDim + column] =
          static_cast<float>((row * 17U + column * 11U) % 101U) / 101.0F;
    }
    CollectionItem item;
    item.logical_id = core::LogicalId::from_utf8("row-" + std::to_string(row));
    item.vector = core::TypedTensorView::contiguous(
        vectors.data() + static_cast<std::ptrdiff_t>(row * kDim), 1, kDim);
    items.push_back(std::move(item));
  }
  auto inserted = collection->add_batch(items, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(inserted.ok()) << inserted.status().diagnostic();

  auto sealed = collection->seal();
  ASSERT_FALSE(sealed.ok());
  EXPECT_EQ(sealed.status().code(), core::StatusCode::not_supported);
  EXPECT_NE(sealed.status().diagnostic().find("LASER"), std::string::npos);
  EXPECT_NE(sealed.status().diagnostic().find("not supported"), std::string::npos);
  EXPECT_NE(sealed.status().diagnostic().find("Flat fallback is disabled"), std::string::npos);

  const auto manifest_path =
      temporary.path() / internal::collection::kCollectionManifestFilename;
  if (std::filesystem::is_regular_file(manifest_path)) {
    const auto manifest = internal::collection::ArtifactManifestV2::load(manifest_path);
    EXPECT_TRUE(std::ranges::none_of(manifest.segments, [](const auto &entry) {
      return entry.lifecycle == internal::collection::SegmentLifecycleV2::sealed;
    }));
  }
  EXPECT_TRUE(collection->close().ok());
#endif
}

TEST(CollectionQgPlatformContract, PlatformGatePrecedesFlatFallbackEligibility) {
#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  GTEST_SKIP() << "this target must compile without the LASER implementation surface";
#else
  const internal::collection::CollectionSchema otherwise_ineligible_schema{
      16, core::Metric::l2, core::ScalarType::float32};
  internal::collection::detail::CollectionTargetBuildParams params;
  params.quantization = CollectionQuantization::rabitq;
  EXPECT_EQ(internal::collection::detail::qg_target_support(otherwise_ineligible_schema,
                                                            /*row_count=*/1,
                                                            params),
            internal::collection::detail::TargetSupport::supported)
      << "a no-LASER qg request must reach the rejecting builder instead of Flat fallback";
#endif
}

}  // namespace
}  // namespace alaya
