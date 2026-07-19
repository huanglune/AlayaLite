// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "index/collection/artifact_manifest_v2.hpp"
#include "index/collection/collection.hpp"
#include "index/collection/detail/collection_target_builder.hpp"

namespace {

constexpr std::uint32_t kRows = 48;
constexpr std::uint32_t kDim = 64;
constexpr std::uint64_t kSeed = 20260712;

auto vectors() -> std::vector<float> {
  std::vector<float> result(kRows * kDim);
  for (std::uint32_t row = 0; row < kRows; ++row) {
    for (std::uint32_t column = 0; column < kDim; ++column) {
      const auto value = (row * 37U + column * 19U + (row % 7U) * column) % 251U;
      result[row * kDim + column] = static_cast<float>(value) / 251.0F;
    }
  }
  return result;
}

}  // namespace

auto main(int argc, char **argv) -> int {
  if (argc != 2) {
    std::cerr << "usage: artifact_collection_qg_laser_generator OUTPUT_DIR\n";
    return 2;
  }
  try {
    const std::filesystem::path output(argv[1]);
    std::filesystem::remove_all(output);
    const auto data = vectors();
    const auto view = alaya::core::TypedTensorView::contiguous(data.data(), kRows, kDim);
    std::vector<alaya::internal::collection::RegisteredRow> rows;
    rows.reserve(kRows);
    for (std::uint32_t row = 0; row < kRows; ++row) {
      auto owned = alaya::internal::collection::OwnedVector::copy_row(view, row);
      if (!owned.ok()) {
        throw std::runtime_error(owned.status().diagnostic());
      }
      alaya::internal::collection::RegisteredRow registered;
      registered.logical_id = alaya::core::LogicalId::from_utf8("row-" + std::to_string(row));
      registered.row_id = alaya::core::SegmentRowId(row);
      registered.upsert_sequence = row + 1;
      registered.payload.vector = std::move(owned).value();
      rows.push_back(std::move(registered));
    }

    alaya::internal::collection::detail::CollectionTargetBuildParams params;
    params.quantization = alaya::CollectionQuantization::rabitq;
    params.max_neighbors = 32;
    params.ef_construction = 64;
    params.thread_count = 1;
    params.seed = kSeed;
    alaya::internal::collection::detail::CollectionTargetPublication publication;
    publication.collection_root = output;
    publication.segment_id = "seg_00000001";
    publication.metadata_epoch = 1;
    publication.metadata_checkpoint = "checkpoint_48.bin";
    publication.wal_cut = kRows;
    publication.row_versions = {1, kRows};
    publication.id_map_checkpoint = publication.metadata_checkpoint;
    publication.collection_features.manifest_v2_writer = true;
    alaya::core::BuildContext context;
    const alaya::internal::collection::CollectionSchema schema{
        kDim, alaya::core::Metric::l2, alaya::core::ScalarType::float32};
    auto built = alaya::internal::collection::detail::build_qg_laser_collection_target(
        schema, rows, params, publication, context);
    if (!built.ok()) {
      throw std::runtime_error(built.status().diagnostic());
    }
    const auto descriptor = built.value().segment.descriptor();
    if (descriptor.algorithm_id != alaya::core::algorithm::qg ||
        descriptor.engine_factory_id != alaya::core::algorithm::laser) {
      throw std::runtime_error("qg golden generator did not build the LASER implementation");
    }

    const auto manifest = alaya::internal::collection::ArtifactManifestV2::load(
        output / alaya::internal::collection::kCollectionManifestFilename);
    if (manifest.segments.size() != 1 ||
        manifest.segments.front().reader_compatibility.required_features !=
            std::vector<std::string>{"qg_laser_segment"}) {
      throw std::runtime_error("qg golden generator published the wrong implementation feature");
    }
    alaya::core::OpenContext open_context;
    auto reopened = alaya::internal::collection::detail::open_qg_collection_target(
        output, manifest.segments.front(), schema, open_context);
    if (!reopened.ok()) {
      throw std::runtime_error(reopened.status().diagnostic());
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
