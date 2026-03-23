/*
 * Copyright 2025 AlayaDB.AI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "storage/buffer/buffer_pool.hpp"
#include "storage/diskann/diskann_storage.hpp"

namespace alaya {

class DiskANNStorageTest : public ::testing::Test {
 protected:
  using BufferPoolType = BufferPool<uint32_t>;
  using StorageType = DiskANNStorage<float, uint32_t>;

  static constexpr size_t kBufferPoolCapacity = 200;

  void SetUp() override {
    buffer_pool_ = std::make_unique<BufferPoolType>(kBufferPoolCapacity, kDataBlockSize, 1);
  }

  void TearDown() override {
    for (const auto &base : temp_bases_) {
      std::filesystem::remove(base + ".meta");
      std::filesystem::remove(base + ".meta.tmp");
      std::filesystem::remove(base + ".pq");
      std::filesystem::remove(base + ".data");
    }
  }

  auto make_base_path(const std::string &name) -> std::string {
    std::string base =
        (std::filesystem::temp_directory_path() / ("diskann_storage_test_" + name)).string();
    temp_bases_.push_back(base);
    return base;
  }

  std::unique_ptr<BufferPoolType> buffer_pool_;

 private:
  std::vector<std::string> temp_bases_;
};

TEST_F(DiskANNStorageTest, AllocateNodeIdKeepsMetaUnchangedWhenDataGrowFails) {
  const std::string kBasePath = make_base_path("allocate_node_id_atomicity");

  {
    StorageType storage(buffer_pool_.get());
    storage.create(kBasePath, 1, 8, 4);

    const uint32_t kInitialCapacity = storage.capacity();
    for (uint32_t i = 0; i < kInitialCapacity; ++i) {
      EXPECT_EQ(storage.allocate_node_id(), static_cast<int32_t>(i));
    }
    EXPECT_EQ(storage.num_active(), kInitialCapacity);
  }

  {
    StorageType storage(buffer_pool_.get());
    storage.open(kBasePath, false);

    const uint32_t kOriginalCapacity = storage.capacity();
    const uint64_t kOriginalActive = storage.num_active();

    EXPECT_THROW(static_cast<void>(storage.allocate_node_id()), std::runtime_error);
    EXPECT_EQ(storage.capacity(), kOriginalCapacity);
    EXPECT_EQ(storage.num_active(), kOriginalActive);
    EXPECT_FALSE(storage.meta().is_dirty());
  }

  {
    StorageType storage(buffer_pool_.get());
    storage.open(kBasePath, false);
    EXPECT_EQ(storage.capacity(), 64U);
    EXPECT_EQ(storage.num_active(), 64U);
  }
}

TEST_F(DiskANNStorageTest, CreateFailureRemovesPartialFiles) {
  const std::string kBasePath = make_base_path("create_cleanup");
  StorageType storage(buffer_pool_.get());

  EXPECT_THROW(storage.create(kBasePath, 16, 4096, 1024), std::invalid_argument);
  EXPECT_FALSE(storage.is_open());
  EXPECT_FALSE(std::filesystem::exists(kBasePath + ".meta"));
  EXPECT_FALSE(std::filesystem::exists(kBasePath + ".meta.tmp"));
  EXPECT_FALSE(std::filesystem::exists(kBasePath + ".pq"));
  EXPECT_FALSE(std::filesystem::exists(kBasePath + ".data"));
}

}  // namespace alaya
