// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "utils/platform.hpp"

namespace alaya {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-canonical-collection-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
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
  result.dim = 2;
  result.metric = core::Metric::l2;
  result.scalar_type = core::ScalarType::float32;
  result.target_algorithm = core::algorithm::hnsw;
  result.build_threads = 3;
  return result;
}

[[nodiscard]] auto item(std::string id,
                        const std::array<float, 2> &vector,
                        std::string document = {},
                        CollectionMetadata metadata = {}) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(id);
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  result.metadata = std::move(metadata);
  result.document = std::move(document);
  return result;
}

[[nodiscard]] auto id_string(const core::LogicalId &id) -> std::string {
  const auto bytes = id.canonical_bytes();
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

TEST(CollectionFacade, CanonicalResultsReceiptsStatsCheckpointAndReopen) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  EXPECT_EQ(collection->target_algorithm(), core::algorithm::hnsw);
  EXPECT_EQ(collection->active_algorithm(), core::algorithm::flat);
  EXPECT_EQ(collection->target_implementation_key(), "hnsw_segment");
  EXPECT_EQ(collection->options().build_threads, 3U);

  const std::array<float, 2> first{0.0F, 0.0F};
  const std::array<float, 2> second{1.0F, 0.0F};
  auto first_receipt = collection->add(item("a", first, "A", {{"kind", std::string("first")}}));
  ASSERT_TRUE(first_receipt.ok()) << first_receipt.status().diagnostic();
  EXPECT_TRUE(first_receipt.value().searchable);
  EXPECT_EQ(first_receipt.value().durability, CollectionDurabilityState::wal_fsync);
  EXPECT_EQ(first_receipt.value().row_status, CollectionRowMutationStatus::inserted);
  auto second_receipt = collection->add(item("b", second, "B"));
  ASSERT_TRUE(second_receipt.ok()) << second_receipt.status().diagnostic();

  const std::array<float, 2> query{0.0F, 0.0F};
  auto single = collection->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(single.ok()) << single.status().diagnostic();
  ASSERT_EQ(single.value().ids.size(), 2U);
  EXPECT_EQ(single.value().distances.size(), single.value().ids.size());
  EXPECT_EQ(single.value().offsets, (std::vector<core::RowCount>{0, 2}));
  EXPECT_EQ(single.value().valid_counts, (std::vector<core::RowCount>{2}));
  EXPECT_EQ(id_string(single.value().ids[0]), "a");
  EXPECT_EQ(single.value().distances[0], 0.0F);
  EXPECT_EQ(single.value().distances[1], 1.0F);

  const std::array<float, 4> queries{0.0F, 0.0F, 1.0F, 0.0F};
  auto batch = collection->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 7);
  ASSERT_TRUE(batch.ok()) << batch.status().diagnostic();
  EXPECT_EQ(batch.value().offsets, (std::vector<core::RowCount>{0, 2, 4}));
  EXPECT_EQ(batch.value().valid_counts, (std::vector<core::RowCount>{2, 2}));
  EXPECT_EQ(batch.value().ids.size(), batch.value().distances.size());

  const auto stats = collection->stats();
  EXPECT_EQ(stats.size, 2U);
  EXPECT_EQ(stats.accepted_count, 2U);
  EXPECT_EQ(stats.pending_count, 0U);
  EXPECT_EQ(stats.pending_bytes, 0U);
  EXPECT_EQ(stats.searchable_vector_bytes, 16U);
  EXPECT_EQ(stats.accepted_vector_bytes, 16U);

  auto checkpoint = collection->checkpoint();
  ASSERT_TRUE(checkpoint.ok()) << checkpoint.status().diagnostic();
  EXPECT_EQ(checkpoint.value().wal_cut, stats.visibility_watermark);
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  EXPECT_FALSE(reopened.value()->imported_legacy_layout());
  auto record = reopened.value()->get_by_id(core::LogicalId::from_utf8("a"));
  ASSERT_TRUE(record.ok()) << record.status().diagnostic();
  ASSERT_TRUE(record.value().vector.has_value());
  EXPECT_EQ(record.value().document, "A");
  EXPECT_EQ(std::get<std::string>(record.value().metadata.at("kind")), "first");
  auto after = reopened.value()->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 10);
  ASSERT_TRUE(after.ok()) << after.status().diagnostic();
  EXPECT_EQ(after.value().offsets, single.value().offsets);
  EXPECT_EQ(after.value().distances, single.value().distances);
  ASSERT_TRUE(reopened.value()->close().ok());
}

TEST(CollectionFacade, BatchModesHaveStableStatusesAndNeverSilentlyDowngrade) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 2> zero{0.0F, 0.0F};
  ASSERT_TRUE(collection->add(item("existing", zero)).ok());

  std::vector<CollectionBatchRow> independent;
  independent.push_back({CollectionMutationAction::add,
                         core::LogicalId::from_utf8("existing"),
                         core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  independent.push_back({CollectionMutationAction::add,
                         core::LogicalId::from_utf8("new"),
                         core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  auto partial = collection->mutate_batch(independent);
  ASSERT_TRUE(partial.ok()) << partial.status().diagnostic();
  ASSERT_EQ(partial.value().rows.size(), 2U);
  EXPECT_EQ(partial.value().rows[0].row_status, CollectionRowMutationStatus::already_exists);
  EXPECT_FALSE(partial.value().rows[0].searchable);
  EXPECT_EQ(partial.value().rows[1].row_status, CollectionRowMutationStatus::inserted);
  EXPECT_TRUE(partial.value().rows[1].searchable);
  EXPECT_EQ(collection->size(), 2U);

  std::vector<CollectionBatchRow> duplicate;
  duplicate.push_back({CollectionMutationAction::upsert,
                       core::LogicalId::from_utf8("same"),
                       core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  duplicate.push_back({CollectionMutationAction::upsert,
                       core::LogicalId::from_utf8("same"),
                       core::TypedTensorView::contiguous(zero.data(), 1, 2)});
  auto atomic = collection->mutate_batch(duplicate, CollectionBatchMutationMode::all_or_nothing);
  ASSERT_TRUE(atomic.ok()) << atomic.status().diagnostic();
  ASSERT_EQ(atomic.value().rows.size(), 2U);
  EXPECT_EQ(atomic.value().rows[0].row_status, CollectionRowMutationStatus::aborted);
  EXPECT_EQ(atomic.value().rows[1].row_status, CollectionRowMutationStatus::conflict);
  EXPECT_EQ(collection->size(), 2U);
  ASSERT_TRUE(collection->close().ok());
}

TEST(CollectionFacade, RabitqRequiresExplicitQg) {
  TemporaryDirectory first;
  auto invalid = options(first.path());
  invalid.quantization = CollectionQuantization::rabitq;
  auto rejected = Collection::create(invalid);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), core::StatusCode::invalid_argument);
  EXPECT_NE(rejected.status().diagnostic().find("explicit index_type=qg"), std::string::npos);

  TemporaryDirectory second;
  auto valid = options(second.path());
  valid.quantization = CollectionQuantization::rabitq;
  valid.target_algorithm = core::algorithm::qg;
  auto created = Collection::create(valid);
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  EXPECT_EQ(created.value()->target_algorithm(), core::algorithm::qg);
  EXPECT_EQ(created.value()->target_engine_factory_key(), "qg");
  ASSERT_TRUE(created.value()->close().ok());
}

TEST(CollectionFacade, CxxDirectParitySequenceUsesTheCanonicalWireShape) {
  TemporaryDirectory temporary;
  auto created = Collection::create(options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();
  const std::array<float, 2> zero{0.0F, 0.0F};
  const std::array<float, 2> two{2.0F, 0.0F};
  const std::array<float, 2> one{1.0F, 0.0F};
  const std::array<float, 2> vertical{0.0F, 1.0F};
  ASSERT_TRUE(collection->add(item("a", zero, "A", {{"revision", std::int64_t{1}}})).ok());
  ASSERT_TRUE(collection->add(item("b", two, "B")).ok());
  ASSERT_TRUE(collection->upsert(item("a", one, "A2", {{"revision", std::int64_t{2}}})).ok());
  ASSERT_TRUE(collection->remove(core::LogicalId::from_utf8("b")).ok());
  ASSERT_TRUE(collection->add(item("c", vertical, "C")).ok());

  const std::array<float, 4> queries{1.0F, 0.0F, 0.0F, 1.0F};
  auto response =
      collection->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 9);
  ASSERT_TRUE(response.ok()) << response.status().diagnostic();
  EXPECT_EQ(response.value().offsets, (std::vector<core::RowCount>{0, 2, 4}));
  EXPECT_EQ(response.value().valid_counts, (std::vector<core::RowCount>{2, 2}));
  ASSERT_EQ(response.value().ids.size(), 4U);
  EXPECT_EQ(id_string(response.value().ids[0]), "a");
  EXPECT_EQ(id_string(response.value().ids[1]), "c");
  EXPECT_EQ(id_string(response.value().ids[2]), "c");
  EXPECT_EQ(id_string(response.value().ids[3]), "a");
  EXPECT_EQ(response.value().distances, (std::vector<float>{0.0F, 2.0F, 0.0F, 2.0F}));
  EXPECT_EQ(collection->size(), 2U);
  ASSERT_TRUE(collection->checkpoint().ok());
  ASSERT_TRUE(collection->close().ok());
  collection.reset();

  auto reopened = Collection::open(temporary.path());
  ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
  auto replayed =
      reopened.value()->batch_search(core::TypedTensorView::contiguous(queries.data(), 2, 2), 9);
  ASSERT_TRUE(replayed.ok()) << replayed.status().diagnostic();
  EXPECT_EQ(replayed.value().offsets, response.value().offsets);
  EXPECT_EQ(replayed.value().valid_counts, response.value().valid_counts);
  EXPECT_EQ(replayed.value().ids, response.value().ids);
  EXPECT_EQ(replayed.value().distances, response.value().distances);
  ASSERT_TRUE(reopened.value()->close().ok());
}

}  // namespace
}  // namespace alaya
