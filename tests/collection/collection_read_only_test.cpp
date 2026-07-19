// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <utility>
#include <vector>

#ifndef _WIN32
  #include <sys/wait.h>
  #include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "alaya/collection.hpp"
#include "index/collection/logical_wal.hpp"
#include "platform/detect.hpp"

namespace alaya {
namespace {

class TemporaryDirectory {
 public:
  TemporaryDirectory() {
    static std::uint64_t serial{};
    path_ =
        std::filesystem::temp_directory_path() /
        ("alaya-read-only-" + std::to_string(platform::get_pid()) + "-" + std::to_string(++serial));
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

using DirectoryBytes = std::map<std::string, std::vector<char>, std::less<>>;

[[nodiscard]] auto directory_bytes(const std::filesystem::path &root) -> DirectoryBytes {
  DirectoryBytes result;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::ifstream input(entry.path(), std::ios::binary);
    result.emplace(std::filesystem::relative(entry.path(), root).generic_string(),
                   std::vector<char>(std::istreambuf_iterator<char>(input),
                                     std::istreambuf_iterator<char>()));
  }
  return result;
}

[[nodiscard]] auto flat_options(const std::filesystem::path &root) -> CollectionOptions {
  CollectionOptions options;
  options.root = root;
  options.dim = 2;
  options.target_algorithm = core::algorithm::flat;
  options.quantization = CollectionQuantization::none;
  return options;
}

[[nodiscard]] auto item(std::string id, const std::array<float, 2> &vector) -> CollectionItem {
  CollectionItem result;
  result.logical_id = core::LogicalId::from_utf8(id);
  result.vector = core::TypedTensorView::contiguous(vector.data(), 1, 2);
  result.metadata.emplace("kind", std::string("kept"));
  result.document = "document-" + id;
  return result;
}

[[nodiscard]] auto create_populated(const std::filesystem::path &root)
    -> std::shared_ptr<Collection> {
  auto created = Collection::create(flat_options(root));
  if (!created.ok()) {
    return {};
  }
  auto collection = std::move(created).value();
  const std::array<float, 2> first{0.0F, 0.0F};
  const std::array<float, 2> second{1.0F, 0.0F};
  if (!collection->add(item("a", first)).ok() || !collection->add(item("b", second)).ok()) {
    return {};
  }
  return collection;
}

void expect_readonly(const core::Status &status) {
  EXPECT_EQ(status.code(), core::StatusCode::not_supported);
  EXPECT_EQ(status.detail(), core::StatusDetail::readonly_instance);
  EXPECT_FALSE(status.partial());
}

TEST(CollectionReadOnly, OpenReadsDurableWalWithoutChangingDirectoryBytes) {
  TemporaryDirectory temporary;
  auto writer = create_populated(temporary.path());
  ASSERT_NE(writer, nullptr);
  ASSERT_TRUE(writer->close().ok());
  writer.reset();

  const auto before = directory_bytes(temporary.path());
  ASSERT_TRUE(before.contains(".alaya_internal/collection.lock"));
  CollectionOpenOptions open_options;
  open_options.read_only = true;
  auto opened = Collection::open(temporary.path(), open_options);
  ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
  auto reader = std::move(opened).value();
  EXPECT_TRUE(reader->read_only());
  EXPECT_EQ(reader->size(), 2U);

  auto record = reader->get_by_id(core::LogicalId::from_utf8("a"));
  ASSERT_TRUE(record.ok()) << record.status().diagnostic();
  EXPECT_EQ(record.value().document, "document-a");
  const std::array<float, 2> query{0.0F, 0.0F};
  auto searched = reader->search(core::TypedTensorView::contiguous(query.data(), 1, 2), 2);
  ASSERT_TRUE(searched.ok()) << searched.status().diagnostic();
  ASSERT_EQ(searched.value().ids.size(), 2U);

  const std::array<float, 2> third{2.0F, 0.0F};
  auto added = reader->add(item("c", third));
  ASSERT_FALSE(added.ok());
  expect_readonly(added.status());
  auto removed = reader->remove(core::LogicalId::from_utf8("a"));
  ASSERT_FALSE(removed.ok());
  expect_readonly(removed.status());
  auto checkpoint = reader->checkpoint();
  ASSERT_FALSE(checkpoint.ok());
  expect_readonly(checkpoint.status());
  auto sealed = reader->seal();
  ASSERT_FALSE(sealed.ok());
  expect_readonly(sealed.status());
  auto compacted = reader->compact();
  ASSERT_FALSE(compacted.ok());
  expect_readonly(compacted.status());
  auto collected = reader->gc();
  ASSERT_FALSE(collected.ok());
  expect_readonly(collected.status());

  ASSERT_TRUE(reader->close().ok());
  reader.reset();
  EXPECT_EQ(directory_bytes(temporary.path()), before);
}

TEST(CollectionReadOnly, TornWalFailsWithoutRepairAndWritableOpenRepairsIt) {
  TemporaryDirectory temporary;
  auto writer = create_populated(temporary.path());
  ASSERT_NE(writer, nullptr);
  ASSERT_TRUE(writer->close().ok());
  writer.reset();

  const auto wal_path = temporary.path() / ".alaya_internal" /
                        std::string(internal::collection::kCollectionWalNamespace) /
                        std::string(internal::collection::kCollectionWalFilename);
  {
    std::ofstream output(wal_path, std::ios::binary | std::ios::app);
    const std::array<char, 3> torn{'b', 'a', 'd'};
    output.write(torn.data(), torn.size());
  }
  const auto torn_bytes = directory_bytes(temporary.path());

  CollectionOpenOptions read_only;
  read_only.read_only = true;
  auto rejected = Collection::open(temporary.path(), read_only);
  ASSERT_FALSE(rejected.ok());
  expect_readonly(rejected.status());
  EXPECT_EQ(directory_bytes(temporary.path()), torn_bytes);

  auto repaired = Collection::open(temporary.path());
  ASSERT_TRUE(repaired.ok()) << repaired.status().diagnostic();
  EXPECT_EQ(repaired.value()->size(), 2U);
  EXPECT_TRUE(repaired.value()->close().ok());
  EXPECT_NE(directory_bytes(temporary.path()), torn_bytes);
}

#ifndef _WIN32
[[nodiscard]] auto run_lock_probe(const std::filesystem::path &root,
                                  bool read_only,
                                  bool expect_success) -> int {
  const auto child = ::fork();
  if (child == 0) {
    (void)::setenv("ALAYA_READ_ONLY_PROBE_ROOT", root.c_str(), 1);
    (void)::setenv("ALAYA_READ_ONLY_PROBE_MODE", read_only ? "reader" : "writer", 1);
    (void)::setenv("ALAYA_READ_ONLY_PROBE_EXPECT", expect_success ? "success" : "conflict", 1);
    ::execl("/proc/self/exe",
            "collection_read_only_test",
            "--gtest_filter=CollectionReadOnlyChild.LockProbe",
            "--gtest_brief=1",
            static_cast<char *>(nullptr));
    ::_exit(127);
  }
  if (child < 0) {
    return -1;
  }
  int status{};
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)) {
    return -1;
  }
  return WEXITSTATUS(status);
}
#endif

TEST(CollectionReadOnlyChild, LockProbe) {
#ifdef _WIN32
  GTEST_SKIP() << "the local cross-process probe uses /proc/self/exe";
#else
  const auto *root = std::getenv("ALAYA_READ_ONLY_PROBE_ROOT");
  if (root == nullptr) {
    GTEST_SKIP() << "helper test is activated only by the parent lock-contract test";
  }
  CollectionOpenOptions options;
  options.read_only = std::string(std::getenv("ALAYA_READ_ONLY_PROBE_MODE")) == "reader";
  const auto expect_success = std::string(std::getenv("ALAYA_READ_ONLY_PROBE_EXPECT")) == "success";
  auto opened = Collection::open(root, options);
  if (expect_success) {
    ASSERT_TRUE(opened.ok()) << opened.status().diagnostic();
    EXPECT_EQ(opened.value()->read_only(), options.read_only);
    EXPECT_EQ(opened.value()->size(), 2U);
    EXPECT_TRUE(opened.value()->close().ok());
  } else {
    ASSERT_FALSE(opened.ok());
    EXPECT_EQ(opened.status().code(), core::StatusCode::conflict);
    EXPECT_EQ(opened.status().detail(), core::StatusDetail::already_exists);
  }
#endif
}

TEST(CollectionReadOnly, CrossProcessLockAllowsOneWriterOrManyReaders) {
#ifdef _WIN32
  GTEST_SKIP() << "the local cross-process probe uses /proc/self/exe";
#else
  TemporaryDirectory temporary;
  auto created = create_populated(temporary.path());
  ASSERT_NE(created, nullptr);
  ASSERT_TRUE(created->close().ok());
  created.reset();

  auto writer = Collection::open(temporary.path());
  ASSERT_TRUE(writer.ok()) << writer.status().diagnostic();
  EXPECT_EQ(run_lock_probe(temporary.path(), true, false), 0);
  ASSERT_TRUE(writer.value()->close().ok());
  writer.value().reset();

  CollectionOpenOptions read_only;
  read_only.read_only = true;
  auto reader = Collection::open(temporary.path(), read_only);
  ASSERT_TRUE(reader.ok()) << reader.status().diagnostic();
  EXPECT_EQ(run_lock_probe(temporary.path(), true, true), 0);
  EXPECT_EQ(run_lock_probe(temporary.path(), false, false), 0);
  ASSERT_TRUE(reader.value()->close().ok());
#endif
}

}  // namespace
}  // namespace alaya
