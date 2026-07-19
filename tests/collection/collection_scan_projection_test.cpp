// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = std::filesystem::temp_directory_path() /
            ("alaya-scan-projection-" + std::to_string(platform::get_pid()) + "-" +
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

[[nodiscard]] auto flat_options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = 2;
  options.target_algorithm = core::algorithm::flat;
  options.quantization = CollectionQuantization::none;
  return options;
}

[[nodiscard]] auto item(std::string id,
                        const std::array<float, 2> &vector,
                        std::string document,
                        std::string category) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(id);
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  result.document = std::move(document);
  result.metadata.emplace("category", std::move(category));
  return result;
}

[[nodiscard]] auto id_string(const core::LogicalId &id) -> std::string {
  const auto bytes = id.canonical_bytes();
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

TEST(CollectionScanProjection, FiltersAndLimitsBeforeMaterializingTheRequestedColumns) {
  TemporaryDirectory temporary;
  auto created = Collection::create(flat_options(temporary.path()));
  ASSERT_TRUE(created.ok()) << created.status().diagnostic();
  auto collection = std::move(created).value();

  const std::array<float, 2> first{0.0F, 0.0F};
  const std::array<float, 2> second{1.0F, 0.0F};
  const std::array<float, 2> third{2.0F, 0.0F};
  ASSERT_TRUE(collection->add(item("a", first, "A", "keep")).ok());
  ASSERT_TRUE(collection->add(item("b", second, "B", "drop")).ok());
  ASSERT_TRUE(collection->add(item("c", third, "C", "keep")).ok());

  const auto filter = CollectionFilter::metadata_equals("category", std::string("keep"));
  auto metadata_only = collection->scan(filter, 1, CollectionProjection::metadata);
  ASSERT_TRUE(metadata_only.ok()) << metadata_only.status().diagnostic();
  ASSERT_EQ(metadata_only.value().size(), 1U);
  EXPECT_EQ(id_string(metadata_only.value()[0].logical_id), "a");
  EXPECT_EQ(std::get<std::string>(metadata_only.value()[0].metadata.at("category")), "keep");
  EXPECT_TRUE(metadata_only.value()[0].document.empty());
  EXPECT_FALSE(metadata_only.value()[0].vector.has_value());

  auto documents_only = collection->scan(filter, 10, CollectionProjection::document);
  ASSERT_TRUE(documents_only.ok()) << documents_only.status().diagnostic();
  ASSERT_EQ(documents_only.value().size(), 2U);
  EXPECT_EQ(id_string(documents_only.value()[0].logical_id), "a");
  EXPECT_EQ(id_string(documents_only.value()[1].logical_id), "c");
  EXPECT_EQ(documents_only.value()[0].document, "A");
  EXPECT_EQ(documents_only.value()[1].document, "C");
  EXPECT_TRUE(documents_only.value()[0].metadata.empty());
  EXPECT_FALSE(documents_only.value()[0].vector.has_value());
}

}  // namespace
}  // namespace alaya
