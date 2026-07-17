// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

// End-to-end coverage for the on-disk mutable LASER engine as the Collection's
// ACTIVE (writable) generation (2B): create(active_engine=laser) -> add batch ->
// search -> remove -> close -> reopen. Validates config persistence (B-08:
// active_algorithm()==laser survives reopen) and that committed active rows are
// restored by WAL replay through the corrected idempotency protocol (B-01).

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

constexpr std::uint32_t kDim = 128;  // LASER floor: power-of-two, >= 128.
constexpr core::RowCount kRows = 200;
constexpr std::uint64_t kTopK = 10;

class TemporaryDirectory {
 public:
  explicit TemporaryDirectory(std::string_view name) {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-active-laser-" + std::string(name) + "-" +
             std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
    std::filesystem::remove_all(path_);
  }
  ~TemporaryDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TemporaryDirectory(const TemporaryDirectory &) = delete;
  auto operator=(const TemporaryDirectory &) -> TemporaryDirectory & = delete;
  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }

 private:
  std::filesystem::path path_{};
};

struct Dataset {
  std::vector<float> vectors{};
  std::vector<core::LogicalId> ids{};
};

[[nodiscard]] auto make_dataset(core::RowCount rows, std::uint32_t seed) -> Dataset {
  Dataset result;
  result.vectors.resize(static_cast<std::size_t>(rows) * kDim);
  result.ids.reserve(static_cast<std::size_t>(rows));
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  for (core::RowCount row = 0; row < rows; ++row) {
    result.ids.push_back(core::LogicalId::from_utf8("row-" + std::to_string(row)));
    for (std::uint32_t column = 0; column < kDim; ++column) {
      result.vectors[static_cast<std::size_t>(row) * kDim + column] = dist(gen);
    }
  }
  return result;
}

[[nodiscard]] auto active_laser_options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = kDim;
  options.metric = core::Metric::l2;
  options.scalar_type = core::ScalarType::float32;
  options.target_algorithm = core::algorithm::laser;
  options.active_engine = core::algorithm::laser;
  options.quantization = CollectionQuantization::rabitq;
  options.max_neighbors = 32;
  options.build_threads = 4;
  options.ef_construction = 128;
  return options;
}

void insert_dataset(Collection &collection, const Dataset &dataset) {
  std::vector<CollectionItem> items;
  items.reserve(dataset.ids.size());
  for (core::RowCount row = 0; row < dataset.ids.size(); ++row) {
    CollectionItem item;
    item.logical_id = dataset.ids[static_cast<std::size_t>(row)];
    item.vector = core::TypedTensorView::contiguous(
        dataset.vectors.data() + static_cast<std::ptrdiff_t>(row) * kDim, 1, kDim);
    items.push_back(std::move(item));
  }
  auto added = collection.add_batch(items, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
}

[[nodiscard]] auto hits_near(Collection &collection, const float *query)
    -> std::vector<core::LogicalId> {
  auto response = collection.search(core::TypedTensorView::contiguous(query, 1, kDim), kTopK);
  std::vector<core::LogicalId> ids;
  if (!response.ok()) {
    return ids;
  }
  for (const auto &id : response.value().ids) {
    ids.push_back(id);
  }
  return ids;
}

[[nodiscard]] auto contains(const std::vector<core::LogicalId> &ids, const core::LogicalId &id)
    -> bool {
  return std::ranges::any_of(ids, [&](const auto &candidate) { return candidate == id; });
}

// Self-recall over a sample of rows: how many of the sampled rows are returned when
// querying with their own vector (LASER is approximate, so we assert a strong
// majority, not exactness).
[[nodiscard]] auto self_recall(Collection &collection, const Dataset &dataset,
                               const std::vector<core::RowCount> &sample) -> std::size_t {
  std::size_t found = 0;
  for (const auto index : sample) {
    const auto ids = hits_near(collection, dataset.vectors.data() +
                                               static_cast<std::ptrdiff_t>(index) * kDim);
    if (contains(ids, dataset.ids[static_cast<std::size_t>(index)])) {
      ++found;
    }
  }
  return found;
}

TEST(CollectionActiveLaser, CreateWriteSearchRemoveCloseReopen) {
  TemporaryDirectory dir("lifecycle");
  const auto dataset = make_dataset(kRows, /*seed=*/20260717U);
  std::vector<core::RowCount> sample;
  for (core::RowCount row = 0; row < kRows; row += 20) {
    sample.push_back(row);
  }

  auto created = Collection::create(active_laser_options(dir.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(collection->active_algorithm(), core::algorithm::laser);

  insert_dataset(*collection, dataset);
  EXPECT_GE(self_recall(*collection, dataset, sample), sample.size() * 7 / 10)
      << "most rows written to the active LASER segment must be self-recalled";

  // Remove one sampled row -> it must no longer be recalled as its own id.
  const auto removed_row = sample.at(2);
  ASSERT_TRUE(collection->remove(dataset.ids[static_cast<std::size_t>(removed_row)]).ok());
  {
    const auto ids = hits_near(*collection, dataset.vectors.data() +
                                                static_cast<std::ptrdiff_t>(removed_row) * kDim);
    EXPECT_FALSE(contains(ids, dataset.ids[static_cast<std::size_t>(removed_row)]))
        << "a removed row must not be returned under its own id";
  }

  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  // ---- reopen: engine identity + committed rows survive (B-08 + B-01) --------
  auto reopened = Collection::open(dir.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto restored = std::move(reopened).value();
  EXPECT_EQ(restored->active_algorithm(), core::algorithm::laser) << "B-08: active engine persists";

  std::vector<core::RowCount> survivors;
  for (const auto row : sample) {
    if (row != removed_row) {
      survivors.push_back(row);
    }
  }
  EXPECT_GE(self_recall(*restored, dataset, survivors), survivors.size() * 7 / 10)
      << "B-01: committed active rows must be restored by WAL replay on reopen";
  {
    const auto ids = hits_near(*restored, dataset.vectors.data() +
                                              static_cast<std::ptrdiff_t>(removed_row) * kDim);
    EXPECT_FALSE(contains(ids, dataset.ids[static_cast<std::size_t>(removed_row)]))
        << "a removed row stays removed across reopen";
  }
}

TEST(CollectionActiveLaser, SealActiveLaserThenSearchAndReopen) {
  TemporaryDirectory dir("seal");
  const auto dataset = make_dataset(kRows, /*seed=*/424242U);
  std::vector<core::RowCount> sample;
  for (core::RowCount row = 0; row < kRows; row += 20) {
    sample.push_back(row);
  }
  auto created = Collection::create(active_laser_options(dir.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  insert_dataset(*collection, dataset);

  auto sealed = collection->seal();
  ASSERT_TRUE(sealed.ok()) << sealed.status().diagnostic();
  EXPECT_EQ(collection->active_algorithm(), core::algorithm::laser);
  EXPECT_GE(self_recall(*collection, dataset, sample), sample.size() * 6 / 10)
      << "sealed rows must remain searchable through the sealed segment";

  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened = Collection::open(dir.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto restored = std::move(reopened).value();
  EXPECT_EQ(restored->active_algorithm(), core::algorithm::laser);
  EXPECT_GE(self_recall(*restored, dataset, sample), sample.size() * 6 / 10);

  // B-09: after seal + reopen the old active generation directory must be reclaimed;
  // exactly one active_laser directory (the current generation) may remain.
  const auto active_root = dir.path() / ".alaya_internal" / "active_laser";
  std::size_t active_dirs = 0;
  std::error_code error;
  if (std::filesystem::is_directory(active_root, error)) {
    for (const auto &entry : std::filesystem::directory_iterator(active_root, error)) {
      if (entry.is_directory(error)) {
        ++active_dirs;
      }
    }
  }
  EXPECT_LE(active_dirs, 1U) << "B-09: seal/rotate must not leak orphan active-laser directories";
}

// B-01/B-02: a per-row-independent batch is N separate logical transactions, each
// with its OWN op_id as the physical txid. The pre-fix design shared one batch_op_id
// so every row after the first satisfied "txid <= last_committed" and was dropped.
// The size after the batch must equal the row count -- no dropped writes.
TEST(CollectionActiveLaser, PerRowBatchAppliesEveryRow) {
  TemporaryDirectory dir("perrow");
  constexpr core::RowCount kBatch = 64;
  const auto dataset = make_dataset(kBatch, /*seed=*/13131U);
  auto created = Collection::create(active_laser_options(dir.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();

  std::vector<CollectionItem> items;
  items.reserve(dataset.ids.size());
  for (core::RowCount row = 0; row < kBatch; ++row) {
    CollectionItem item;
    item.logical_id = dataset.ids[static_cast<std::size_t>(row)];
    item.vector = core::TypedTensorView::contiguous(
        dataset.vectors.data() + static_cast<std::ptrdiff_t>(row) * kDim, 1, kDim);
    items.push_back(std::move(item));
  }
  auto added = collection->add_batch(items, CollectionBatchMutationMode::per_row_independent);
  ASSERT_TRUE(added.ok()) << added.status().diagnostic();
  EXPECT_EQ(collection->size(), kBatch)
      << "B-01: every row of a per-row-independent batch must apply (no shared-txid drop)";

  // And they survive reopen (each row replayed with its own physical txid).
  ASSERT_TRUE(collection->close().ok());
  collection.reset();
  auto reopened = Collection::open(dir.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->size(), kBatch);
}

[[nodiscard]] auto schema_field_count(const std::filesystem::path &root) -> std::size_t {
  std::ifstream input(root / ".alaya_internal" / "collection_facade_v1" / "schema.v1");
  std::size_t lines = 0;
  for (std::string line; std::getline(input, line);) {
    if (!line.empty()) {
      ++lines;
    }
  }
  return lines;
}

// B-08: a flat active collection keeps the pre-2B 14-field schema (byte-compatible
// with old readers); a laser active collection widens it to 15 fields, so an old
// binary's strict field-count check fails closed instead of silently reverting to
// flat.
TEST(CollectionActiveLaser, ActiveEngineSchemaWidthGatesOldReaders) {
  TemporaryDirectory flat_dir("schema_flat");
  CollectionOptions flat = active_laser_options(flat_dir.path());
  flat.active_engine = core::algorithm::flat;
  flat.target_algorithm = core::algorithm::flat;
  flat.quantization = CollectionQuantization::none;
  auto flat_created = Collection::create(flat);
  ASSERT_TRUE(flat_created.ok()) << flat_created.status().diagnostic();
  EXPECT_EQ(flat_created.value()->active_algorithm(), core::algorithm::flat);
  EXPECT_EQ(schema_field_count(flat_dir.path()), 14U);

  TemporaryDirectory laser_dir("schema_laser");
  auto laser_created = Collection::create(active_laser_options(laser_dir.path()));
  ASSERT_TRUE(laser_created.ok()) << laser_created.status().diagnostic();
  auto laser_collection = std::move(laser_created).value();
  EXPECT_EQ(laser_collection->active_algorithm(), core::algorithm::laser);
  EXPECT_EQ(schema_field_count(laser_dir.path()), 15U);

  // Reopen restores the laser active engine from the 15-field schema. Drop the
  // handle first: the on-disk active segment's single-writer flock is released on
  // destruction (not close()), so reopening the same path needs the old handle gone.
  ASSERT_TRUE(laser_collection->close().ok());
  laser_collection.reset();
  auto reopened = Collection::open(laser_dir.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_EQ(reopened.value()->active_algorithm(), core::algorithm::laser);
}

}  // namespace
}  // namespace alaya
