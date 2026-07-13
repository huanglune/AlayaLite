// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/sha256.hpp"
#include "utils/platform.hpp"

namespace alaya {
namespace {

namespace fs = std::filesystem;

const fs::path kCorpusRoot =
    fs::path(ALAYA_SOURCE_DIR) / "python/tests/fixtures/legacy_recovery_corpus";

class ImportTemporaryDirectory {
 public:
  ImportTemporaryDirectory() {
    static std::uint64_t serial{};
    path_ = fs::temp_directory_path() /
            ("alaya-canonical-import-" + std::to_string(platform::get_pid()) + "-" +
             std::to_string(++serial));
    fs::remove_all(path_);
    fs::create_directories(path_);
  }
  ~ImportTemporaryDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }
  [[nodiscard]] auto path() const -> const fs::path & { return path_; }

 private:
  fs::path path_{};
};

struct FrozenFile {
  std::uint64_t bytes{};
  internal::collection::Sha256Digest digest{};

  friend auto operator==(const FrozenFile &, const FrozenFile &) -> bool = default;
};

[[nodiscard]] auto freeze_existing_files(const fs::path &root) -> std::map<fs::path, FrozenFile> {
  std::map<fs::path, FrozenFile> result;
  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) {
      result.emplace(entry.path().lexically_relative(root),
                     FrozenFile{entry.file_size(),
                                internal::collection::sha256_file(entry.path())});
    }
  }
  return result;
}

void expect_frozen_files(const fs::path &root, const std::map<fs::path, FrozenFile> &frozen) {
  for (const auto &[relative, expected] : frozen) {
    const auto path = root / relative;
    ASSERT_TRUE(fs::is_regular_file(path)) << relative;
    EXPECT_EQ(fs::file_size(path), expected.bytes) << relative;
    EXPECT_EQ(internal::collection::sha256_file(path), expected.digest) << relative;
  }
}

[[nodiscard]] auto id_string(const core::LogicalId &id) -> std::string {
  const auto bytes = id.canonical_bytes();
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

TEST(CollectionFacadeImporter, CanonicalOpenImportsTwoCorporaNonDestructivelyAndReopens) {
  struct TestCase {
    std::string_view name;
    std::string_view query_id;
  };
  constexpr std::array cases{TestCase{"i8_u32_collection_upsert_wal", "item-1"},
                             TestCase{"u8_u64_collection_delete_clean", "item-0"}};

  for (const auto &test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ImportTemporaryDirectory temporary;
    const auto root = temporary.path() / test_case.name;
    fs::copy(kCorpusRoot / test_case.name,
             root,
             fs::copy_options::recursive | fs::copy_options::copy_symlinks);
    const auto frozen = freeze_existing_files(root);

    auto opened = Collection::open(root);
    ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
    auto collection = std::move(opened).value();
    EXPECT_TRUE(collection->imported_legacy_layout());
    EXPECT_EQ(collection->size(), 4U);

    auto query_record = collection->get_by_id(core::LogicalId::from_utf8(test_case.query_id));
    ASSERT_TRUE(query_record.ok()) << query_record.status().diagnostic();
    ASSERT_TRUE(query_record.value().vector.has_value());
    auto response = collection->search(query_record.value().vector->view(), 1);
    ASSERT_TRUE(response.ok()) << response.status().diagnostic();
    ASSERT_EQ(response.value().valid_counts, (std::vector<core::RowCount>{1}));
    ASSERT_EQ(response.value().ids.size(), 1U);
    EXPECT_EQ(id_string(response.value().ids[0]), test_case.query_id);

    CollectionItem appended;
    appended.logical_id = core::LogicalId::from_utf8("canonical-new");
    appended.vector = query_record.value().vector->view();
    appended.document = "native owner";
    appended.metadata = {{"source", std::string("canonical")}};
    auto receipt = collection->add(appended);
    ASSERT_TRUE(receipt.ok()) << receipt.status().diagnostic();
    EXPECT_TRUE(receipt.value().searchable);
    ASSERT_TRUE(collection->checkpoint().ok());
    ASSERT_TRUE(collection->close().ok());
    collection.reset();

    auto reopened = Collection::open(root);
    ASSERT_TRUE(reopened.ok()) << reopened.status().diagnostic();
    EXPECT_TRUE(reopened.value()->imported_legacy_layout());
    EXPECT_EQ(reopened.value()->size(), 5U);
    EXPECT_TRUE(reopened.value()->get_by_id(core::LogicalId::from_utf8(test_case.query_id)).ok());
    auto appended_record = reopened.value()->get_by_id(core::LogicalId::from_utf8("canonical-new"));
    ASSERT_TRUE(appended_record.ok()) << appended_record.status().diagnostic();
    EXPECT_EQ(appended_record.value().document, "native owner");
    EXPECT_EQ(std::get<std::string>(appended_record.value().metadata.at("source")), "canonical");
    ASSERT_TRUE(reopened.value()->close().ok());

    expect_frozen_files(root, frozen);
  }
}

}  // namespace
}  // namespace alaya
