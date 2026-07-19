// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "platform/detect.hpp"

namespace alaya {

namespace internal::collection {
class CollectionTestAccess {
 public:
  [[nodiscard]] static auto pin_epoch(const Collection &collection) -> RoutingSnapshotPtr {
    return collection.implementation_->pin_routing_snapshot();
  }
};
}  // namespace internal::collection

namespace {

constexpr std::uint32_t kDim = 128;
constexpr std::uint32_t kR = 32;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::atomic_uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-active-laser-maintenance-" + std::string(name) + "-" +
             std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
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

[[nodiscard]] auto options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions result;
  result.root = root;
  result.dim = kDim;
  result.metric = core::Metric::l2;
  result.scalar_type = core::ScalarType::float32;
  result.target_algorithm = core::algorithm::laser;
  result.active_engine = core::algorithm::laser;
  result.quantization = CollectionQuantization::rabitq;
  result.max_neighbors = kR;
  result.ef_construction = 128;
  result.build_threads = 2;
  return result;
}

[[nodiscard]] auto vector_with(float value) -> std::array<float, kDim> {
  std::array<float, kDim> result{};
  result.fill(value);
  return result;
}

[[nodiscard]] auto item(std::string id,
                        const std::array<float, kDim> &vector,
                        CollectionMetadata metadata = {}) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(std::move(id));
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, vector.size());
  result.metadata = std::move(metadata);
  return result;
}

[[nodiscard]] auto active_directory(const std::filesystem::path &root,
                                    std::uint64_t segment_id,
                                    std::uint64_t generation) -> std::filesystem::path {
  auto name = std::string("seg_") + std::string(8 - std::to_string(segment_id).size(), '0') +
              std::to_string(segment_id);
  return root / ".alaya_internal" / "active_laser" / (name + "_g" + std::to_string(generation));
}

[[nodiscard]] auto active_index(const std::filesystem::path &root,
                                std::uint64_t segment_id = 2,
                                std::uint64_t generation = 1) -> std::filesystem::path {
  return active_directory(root, segment_id, generation) /
         ("active_laser_R" + std::to_string(kR) + "_MD" + std::to_string(kDim) + ".index");
}

[[nodiscard]] auto read_header(const std::filesystem::path &index)
    -> std::array<char, laser::kSectorLen> {
  std::array<char, laser::kSectorLen> header{};
  std::ifstream input(index, std::ios::binary);
  if (!input.read(header.data(), static_cast<std::streamsize>(header.size()))) {
    throw std::runtime_error("failed to read active LASER QG header");
  }
  return header;
}

[[nodiscard]] auto superblock_copies(const std::filesystem::path &index)
    -> std::array<laser::QGSuperblockV2, 2> {
  const auto header = read_header(index);
  std::array<laser::QGSuperblockV2, 2> copies{};
  std::memcpy(&copies[0], header.data(), sizeof(copies[0]));
  std::memcpy(&copies[1], header.data() + laser::kQGSuperblockSize, sizeof(copies[1]));
  return copies;
}

[[nodiscard]] auto selected_superblock(const std::filesystem::path &index)
    -> laser::QGSuperblockV2 {
  const auto header = read_header(index);
  laser::QGSuperblockV2 selected;
  const int slot = laser::select_qg_superblock_checked(header.data(),
                                                       selected,
                                                       laser::kQgSupportedRequiredFeatures);
  if (slot < 0) {
    throw std::runtime_error("active LASER QG has no supported selected superblock");
  }
  return selected;
}

[[nodiscard]] auto top_ids(Collection &collection,
                           const std::array<float, kDim> &query,
                           std::uint64_t top_k = 1,
                           const CollectionFilter &filter = {})
    -> core::Result<CollectionSearchResponse> {
  return collection.search(core::TypedTensorView::contiguous(query.data(), 1, query.size()),
                           top_k,
                           filter);
}

[[nodiscard]] auto descriptor_format(const Collection &collection) -> std::uint32_t {
  const auto epoch = internal::collection::CollectionTestAccess::pin_epoch(collection);
  const auto active = epoch->find_active_mutable();
  return active == nullptr ? 0 : active->segment.descriptor().format_version;
}

struct LabelSlotEntry {
  std::uint32_t pid{};
  std::uint32_t generation{};
  std::uint64_t label{};
};
static_assert(sizeof(LabelSlotEntry) == 16);

[[nodiscard]] auto selected_label_bindings(const std::filesystem::path &index)
    -> std::vector<LabelSlotEntry> {
  const auto selected = selected_superblock(index);
  std::uint64_t slot{};
  std::uint64_t count{};
  std::memcpy(&slot, selected.reserved.data() + 8, sizeof(slot));
  std::memcpy(&count, selected.reserved.data() + 24, sizeof(count));
  std::vector<LabelSlotEntry> entries(static_cast<std::size_t>(count));
  if (entries.empty()) {
    return entries;
  }
  std::ifstream input(index.string() + ".labels.slot" + std::to_string(slot), std::ios::binary);
  if (!input.read(reinterpret_cast<char *>(entries.data()),
                  static_cast<std::streamsize>(entries.size() * sizeof(LabelSlotEntry)))) {
    throw std::runtime_error("failed to read selected active LASER label slot");
  }
  return entries;
}

TEST(CollectionActiveLaserMaintenance, ExplicitConsolidateReclaimsAndReusesAcrossDoubleReopen) {
  TemporaryDirectory root("reuse");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(descriptor_format(*collection), 2U);

  const auto old_vector = vector_with(10.0F);
  const auto competitor_vector = vector_with(1.0F);
  const auto new_vector = vector_with(0.0F);
  const auto old = item("old", old_vector);
  const auto competitor = item("competitor", competitor_vector);
  ASSERT_TRUE(collection->add(old).ok());
  ASSERT_TRUE(collection->add(competitor).ok());
  ASSERT_TRUE(collection->remove(old.logical_id).ok());

  const auto before = collection->stats();
  ASSERT_EQ(before.allocated_count, 2U);
  auto consolidated = collection->consolidate();
  ASSERT_TRUE(consolidated.ok()) << consolidated.status().diagnostic();
  EXPECT_EQ(consolidated.value().active_segment_id, 2U);
  EXPECT_EQ(consolidated.value().active_generation, 1U);
  const auto after = collection->stats();
  EXPECT_EQ(after.routing_generation, before.routing_generation);
  EXPECT_EQ(after.visibility_watermark, before.visibility_watermark);
  EXPECT_EQ(after.durable_watermark, before.durable_watermark);
  EXPECT_EQ(after.metadata_epoch, before.metadata_epoch);

  const auto replacement = item("new", new_vector);
  ASSERT_TRUE(collection->add(replacement).ok());
  EXPECT_EQ(collection->stats().allocated_count, before.allocated_count)
      << "the post-consolidate bundle must consume the reclaimed PID";
  auto nearest = top_ids(*collection, new_vector);
  ASSERT_TRUE(nearest.ok()) << nearest.status().diagnostic();
  ASSERT_EQ(nearest.value().ids.size(), 1U);
  EXPECT_EQ(nearest.value().ids.front(), replacement.logical_id)
      << "rank-only rerank must use the new Collection-retained vector";

  core::SegmentRowId old_row{};
  core::SegmentRowId new_row{};
  {
    const auto epoch = internal::collection::CollectionTestAccess::pin_epoch(*collection);
    old_row = epoch->versions.at(old.logical_id).address.row_id;
    new_row = epoch->versions.at(replacement.logical_id).address.row_id;
  }
  auto checkpoint = collection->checkpoint();
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
  const auto bindings = selected_label_bindings(active_index(root.path()));
  const auto new_binding = std::ranges::find_if(bindings, [&](const auto &binding) {
    return binding.label == static_cast<std::uint64_t>(new_row);
  });
  ASSERT_NE(new_binding, bindings.end());
  EXPECT_EQ(new_binding->generation, 1U);
  EXPECT_EQ(std::ranges::find_if(bindings,
                                 [&](const auto &binding) {
                                   return binding.label == static_cast<std::uint64_t>(old_row);
                                 }),
            bindings.end());

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  for (int reopen = 0; reopen < 2; ++reopen) {
    auto opened = Collection::open(root.path());
    ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
    collection = std::move(opened).value();
    EXPECT_EQ(collection->stats().allocated_count, 2U);
    nearest = top_ids(*collection, new_vector);
    ASSERT_TRUE(nearest.ok()) << nearest.status().diagnostic();
    ASSERT_EQ(nearest.value().ids.size(), 1U);
    EXPECT_EQ(nearest.value().ids.front(), replacement.logical_id);
    ASSERT_TRUE(collection->close().ok());
    collection.reset();
  }
}

TEST(CollectionActiveLaserMaintenance, ActivationProgressesMixedThenDualV3WithoutDescriptorDrift) {
  TemporaryDirectory root("activation");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const auto index = active_index(root.path());

  auto copies = superblock_copies(index);
  ASSERT_TRUE(laser::qg_superblock_valid(copies[0]));
  ASSERT_TRUE(laser::qg_superblock_valid(copies[1]));
  EXPECT_EQ(copies[0].format_version, laser::kQGFormatVersion);
  EXPECT_EQ(copies[1].format_version, laser::kQGFormatVersion);

  ASSERT_TRUE(collection->consolidate().ok());
  copies = superblock_copies(index);
  EXPECT_EQ(std::ranges::count_if(copies,
                                  [](const auto &copy) {
                                    return laser::qg_superblock_valid(copy) &&
                                           copy.format_version == laser::kQGFormatVersionV3;
                                  }),
            1U);
  EXPECT_EQ(std::ranges::count_if(copies,
                                  [](const auto &copy) {
                                    return laser::qg_superblock_valid(copy) &&
                                           copy.format_version == laser::kQGFormatVersion;
                                  }),
            1U);
  EXPECT_EQ(descriptor_format(*collection), 2U);

  ASSERT_TRUE(collection->checkpoint().ok());
  copies = superblock_copies(index);
  EXPECT_TRUE(std::ranges::all_of(copies, [](const auto &copy) {
    return laser::qg_superblock_valid(copy) && copy.format_version == laser::kQGFormatVersionV3;
  }));

  const auto row_vector = vector_with(0.25F);
  ASSERT_TRUE(collection->add(item("pid-activation", row_vector)).ok());
  const auto selected = selected_superblock(index);
  const auto required = laser::qg_read_required_feature_flags(selected);
  const auto full_features = laser::kQgFeatMaintenanceTxV1 | laser::kQgFeatPostRedoFreeListV1 |
                             laser::kQgFeatPidGenerationV1 | laser::kQgFeatCanonicalPrebindV1 |
                             laser::kQgFeatMutableLabelSlotV1;
  EXPECT_EQ(required & full_features, full_features);

  // A W1-only reader must reject the highest-generation PID-active slot rather
  // than silently fall back to a lower-feature copy.
  const auto header = read_header(index);
  laser::QGSuperblockV2 old_reader_selected;
  const auto old_mask = laser::kQgFeatMaintenanceTxV1 | laser::kQgFeatPostRedoFreeListV1;
  EXPECT_EQ(laser::select_qg_superblock_checked(header.data(), old_reader_selected, old_mask), -2);
  ASSERT_TRUE(collection->checkpoint().ok());
  copies = superblock_copies(index);
  EXPECT_TRUE(std::ranges::all_of(copies, [&](const auto &copy) {
    return laser::qg_superblock_valid(copy) && copy.format_version == laser::kQGFormatVersionV3 &&
           (laser::qg_read_required_feature_flags(copy) & full_features) == full_features;
  }));
  EXPECT_EQ(descriptor_format(*collection), 2U);
}

TEST(CollectionActiveLaserMaintenance, SealMakesActivatedSourceDualV3AndOpenSweepsItLater) {
  TemporaryDirectory root("seal-dual");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  for (std::uint64_t row = 0; row < 8; ++row) {
    const auto vector = vector_with(static_cast<float>(row));
    ASSERT_TRUE(collection->add(item("seal-" + std::to_string(row), vector)).ok());
  }
  ASSERT_TRUE(collection->remove(core::LogicalId::from_utf8("seal-0")).ok());
  ASSERT_TRUE(collection->consolidate().ok());

  const auto source_dir = active_directory(root.path(), 2, 1);
  const auto source_index = active_index(root.path(), 2, 1);
  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  ASSERT_TRUE(std::filesystem::is_directory(source_dir));
  const auto copies = superblock_copies(source_index);
  EXPECT_TRUE(std::ranges::all_of(copies, [](const auto &copy) {
    return laser::qg_superblock_valid(copy) && copy.format_version == laser::kQGFormatVersionV3;
  }));

  ASSERT_TRUE(collection->close().ok());
  EXPECT_TRUE(std::filesystem::is_directory(source_dir));
  collection.reset();
  auto reopened = Collection::open(root.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_FALSE(std::filesystem::exists(source_dir));
}

TEST(CollectionActiveLaserMaintenance, CloseKeepsSingleWriterLeaseUntilDestruction) {
  TemporaryDirectory root("flock");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  ASSERT_TRUE(collection->close().ok());

  auto still_leased = Collection::open(root.path());
  ASSERT_FALSE(still_leased.ok());
  EXPECT_NE(still_leased.status().diagnostic().find("single-writer lease"), std::string::npos);
  EXPECT_NE(still_leased.status().diagnostic().find("this process still holds"), std::string::npos);
  EXPECT_NE(still_leased.status().diagnostic().find("retry"), std::string::npos);
  collection.reset();
  auto reopened = Collection::open(root.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
}

TEST(CollectionActiveLaserMaintenance,
     PinnedRoutingSnapshotExplainsResidualWriterLeaseAndIsRetryable) {
  TemporaryDirectory root("pinned-writer");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  auto pinned = internal::collection::CollectionTestAccess::pin_epoch(*collection);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto residual = Collection::open(root.path());
  ASSERT_FALSE(residual.ok());
  EXPECT_NE(residual.status().diagnostic().find("this process still holds"), std::string::npos);
  EXPECT_NE(residual.status().diagnostic().find("retry"), std::string::npos);

  pinned.reset();
  auto reopened = Collection::open(root.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
}

TEST(CollectionActiveLaserMaintenance, CloseResetReopenLoopsUnderBackgroundSearchLoad) {
  TemporaryDirectory root("reopen-load");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  for (std::uint64_t row = 0; row < 16; ++row) {
    const auto vector = vector_with(static_cast<float>(row) * 0.1F);
    ASSERT_TRUE(collection->add(item("reopen-" + std::to_string(row), vector)).ok());
  }
  const auto query = vector_with(0.0F);

#if defined(__SANITIZE_THREAD__)
  constexpr int kCycles = 4;
#else
  constexpr int kCycles = 20;
#endif
  constexpr int kSearchThreads = 4;
  for (int cycle = 0; cycle < kCycles; ++cycle) {
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> completed{0};
    auto current = collection;
    std::vector<std::thread> workers;
    workers.reserve(kSearchThreads);
    for (int thread = 0; thread < kSearchThreads; ++thread) {
      workers.emplace_back([&, current] {
        while (!stop.load(std::memory_order_acquire)) {
          const auto result = top_ids(*current, query, 8);
          if (result.ok()) {
            completed.fetch_add(1, std::memory_order_relaxed);
          } else if (!stop.load(std::memory_order_acquire)) {
            // close() can race admission after it flips the lifecycle gate;
            // that fail-closed result is expected only once teardown starts.
            std::this_thread::yield();
          }
        }
      });
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (completed.load(std::memory_order_acquire) < 8 &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    EXPECT_GE(completed.load(std::memory_order_relaxed), 8U) << "cycle=" << cycle;
    ASSERT_TRUE(current->close().ok());
    stop.store(true, std::memory_order_release);
    for (auto &worker : workers) {
      worker.join();
    }
    current.reset();
    collection.reset();

    auto reopened = Collection::open(root.path());
    ASSERT_TRUE(reopened.ok()) << "cycle=" << cycle << ": " << reopened.status().diagnostic();
    collection = std::move(reopened).value();
  }
}

TEST(CollectionActiveLaserMaintenance, ActiveFilterRemainsPostfilterAcrossMaintenanceAndReopen) {
  TemporaryDirectory root("postfilter");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  std::set<core::LogicalId, internal::collection::LogicalIdLess> selected;
  for (std::uint64_t row = 0; row < 24; ++row) {
    const auto vector = vector_with(static_cast<float>(row) * 0.1F);
    const bool keep = row % 3 == 0;
    auto current = item("filter-" + std::to_string(row), vector, {{"selected", keep}});
    if (keep) {
      selected.insert(current.logical_id);
    }
    ASSERT_TRUE(collection->add(current).ok());
  }
  const CollectionFilter filter(
      [](const core::LogicalId &, const CollectionMetadata &metadata, std::string_view) {
        return std::get<bool>(metadata.at("selected"));
      },
      0.33);
  const auto query = vector_with(0.0F);
  const auto verify = [&] {
    auto result = top_ids(*collection, query, 12, filter);
    ASSERT_TRUE(result.ok()) << result.status().diagnostic();
    EXPECT_EQ(result.value().search_stats.filter_execution, core::FilterExecution::postfilter);
    EXPECT_GE(result.value().search_stats.filter_examined,
              result.value().search_stats.filter_passed);
    for (const auto &id : result.value().ids) {
      EXPECT_TRUE(selected.contains(id));
    }
  };

  verify();
  ASSERT_TRUE(collection->consolidate().ok());
  verify();
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  for (int reopen = 0; reopen < 2; ++reopen) {
    auto opened = Collection::open(root.path());
    ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
    collection = std::move(opened).value();
    verify();
    ASSERT_TRUE(collection->close().ok());
    collection.reset();
  }
}

TEST(CollectionActiveLaserMaintenance, DurabilityModesRejectSearchableWithoutPendingLeak) {
  enum class Mode { single, atomic_batch, per_row_batch };
  for (const auto mode : {Mode::single, Mode::atomic_batch, Mode::per_row_batch}) {
    SCOPED_TRACE(static_cast<int>(mode));
    TemporaryDirectory root("durability-" + std::to_string(static_cast<int>(mode)));
    auto created = Collection::create(options(root.path()));
    ASSERT_TRUE(created.ok()) << created.status().diagnostic();
    auto collection = std::move(created).value();
    const auto first_vector = vector_with(1.0F);
    const auto second_vector = vector_with(2.0F);
    const auto first = item("durable-a", first_vector);
    const auto second = item("durable-b", second_vector);

    std::uint64_t durable_rows = 0;
    if (mode == Mode::single) {
      auto receipt = collection->add(first);
      ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
      EXPECT_TRUE(receipt.value().searchable);
      EXPECT_EQ(receipt.value().durability, CollectionDurabilityState::wal_fsync);
      durable_rows = 1;
    } else {
      const std::array rows{first, second};
      const auto batch_mode = mode == Mode::atomic_batch
                                  ? CollectionBatchMutationMode::all_or_nothing
                                  : CollectionBatchMutationMode::per_row_independent;
      auto receipt = collection->add_batch(rows, batch_mode);
      ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
      EXPECT_TRUE(receipt.value().searchable);
      EXPECT_EQ(receipt.value().durability, CollectionDurabilityState::wal_fsync);
      ASSERT_EQ(receipt.value().rows.size(), rows.size());
      for (const auto &row : receipt.value().rows) {
        EXPECT_TRUE(row.searchable);
        EXPECT_EQ(row.durability, CollectionDurabilityState::wal_fsync);
      }
      durable_rows = rows.size();
    }

    CollectionWriteOptions weak;
    weak.durability = CollectionWriteDurability::searchable;
    const auto weak_first_vector = vector_with(3.0F);
    const auto weak_second_vector = vector_with(4.0F);
    const auto weak_first = item("weak-a", weak_first_vector);
    const auto weak_second = item("weak-b", weak_second_vector);
    if (mode == Mode::single) {
      EXPECT_FALSE(collection->add(weak_first, weak).ok());
    } else {
      const std::array rows{weak_first, weak_second};
      const auto batch_mode = mode == Mode::atomic_batch
                                  ? CollectionBatchMutationMode::all_or_nothing
                                  : CollectionBatchMutationMode::per_row_independent;
      auto rejected = collection->add_batch(rows, batch_mode, weak);
      if (mode == Mode::per_row_batch) {
        EXPECT_FALSE(rejected.ok());
      } else if (rejected.ok()) {
        EXPECT_FALSE(rejected.value().searchable);
        ASSERT_EQ(rejected.value().rows.size(), rows.size());
        EXPECT_TRUE(std::ranges::all_of(rejected.value().rows, [](const auto &row) {
          return row.row_status == CollectionRowMutationStatus::aborted;
        }));
      }
    }
    EXPECT_EQ(collection->stats().pending_count, 0U);
    EXPECT_EQ(collection->size(), durable_rows);
    ASSERT_TRUE(collection->checkpoint().ok());
    const auto following_vector = vector_with(5.0F);
    ASSERT_TRUE(collection->add(item("following", following_vector)).ok());
    ++durable_rows;

    ASSERT_TRUE(collection->close().ok());
    collection.reset();
    for (int reopen = 0; reopen < 2; ++reopen) {
      auto opened = Collection::open(root.path());
      ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
      collection = std::move(opened).value();
      EXPECT_EQ(collection->size(), durable_rows);
      EXPECT_EQ(collection->stats().pending_count, 0U);
      ASSERT_TRUE(collection->close().ok());
      collection.reset();
    }
  }
}

TEST(CollectionActiveLaserMaintenance, SearchExtensionAdmissionMatchesActiveAndSealed) {
  TemporaryDirectory root("search-extension-parity");
  auto created = Collection::create(options(root.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  for (std::uint64_t row = 0; row < 64; ++row) {
    const auto vector = vector_with(static_cast<float>(row) * 0.125F);
    ASSERT_TRUE(collection->add(item("extension-" + std::to_string(row), vector)).ok());
  }
  const auto query = vector_with(3.0F);
  const auto run = [&](std::uint32_t effort, std::uint32_t beam_width) {
    disk::LaserSegmentSearchExtension parameters;
    parameters.effort = effort;
    parameters.beam_width = beam_width;
    auto extension = disk::make_laser_segment_search_extension(parameters);
    core::SearchOptions search_options(5);
    search_options.extensions = std::span<const core::AlgorithmSearchExtension>(&extension, 1);
    core::SearchContext context;
    return collection->search(core::TypedTensorView::contiguous(query.data(), 1, query.size()),
                              search_options,
                              context);
  };

  ASSERT_TRUE(run(/*effort=*/5, /*beam_width=*/2).ok());
  ASSERT_TRUE(run(/*effort=*/96, /*beam_width=*/8).ok());
  auto invalid_active = run(/*effort=*/0, /*beam_width=*/2);
  ASSERT_FALSE(invalid_active.ok());
  EXPECT_EQ(invalid_active.status().detail(), core::StatusDetail::malformed_struct);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  ASSERT_TRUE(run(/*effort=*/5, /*beam_width=*/2).ok());
  ASSERT_TRUE(run(/*effort=*/96, /*beam_width=*/8).ok());
  auto invalid_sealed = run(/*effort=*/0, /*beam_width=*/2);
  ASSERT_FALSE(invalid_sealed.ok());
  EXPECT_EQ(invalid_sealed.status().code(), invalid_active.status().code());
  EXPECT_EQ(invalid_sealed.status().detail(), invalid_active.status().detail());
}

}  // namespace
}  // namespace alaya
