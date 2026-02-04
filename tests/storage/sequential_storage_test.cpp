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

#include "storage/data/sequential_storage.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>  // for memset

namespace alaya {

class SequentialStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize the bits to 0
  }
};

TEST_F(SequentialStorageTest, InitTest) {
  alaya::SequentialStorage<int, uint32_t> storage;
  storage.init(sizeof(int), 10, 64);

  EXPECT_EQ(storage.item_size_, sizeof(int));
  EXPECT_EQ(storage.capacity_, 10);
  EXPECT_EQ(storage.alignment_, 64);
  EXPECT_NE(storage.data_, nullptr);
  EXPECT_NE(storage.bitmap_, nullptr);
}

TEST_F(SequentialStorageTest, InsertTest) {
  alaya::SequentialStorage<int, uint32_t> storage;
  storage.init(sizeof(int), 10, 64);

  int data1 = 42;
  int data2 = 100;

  uint32_t id1 = storage.insert(&data1);
  uint32_t id2 = storage.insert(&data2);

  EXPECT_EQ(id1, 0);
  EXPECT_EQ(id2, 1);
  EXPECT_EQ(*storage[id1], data1);
  EXPECT_EQ(*storage[id2], data2);
  EXPECT_TRUE(storage.is_valid(id1));
  EXPECT_TRUE(storage.is_valid(id2));
}

TEST_F(SequentialStorageTest, RemoveTest) {
  alaya::SequentialStorage<int, uint32_t> storage;
  storage.init(sizeof(int), 10, 64);

  int data1 = 42;
  uint32_t id1 = storage.insert(&data1);

  EXPECT_TRUE(storage.is_valid(id1));
  storage.remove(id1);
  EXPECT_FALSE(storage.is_valid(id1));
}

TEST_F(SequentialStorageTest, UpdateTest) {
  alaya::SequentialStorage<int, uint32_t> storage;
  storage.init(sizeof(int), 10, 64);

  int data1 = 42;
  int data2 = 100;

  uint32_t id1 = storage.insert(&data1);
  storage.update(id1, &data2);

  EXPECT_EQ(*storage[id1], data2);
}

TEST_F(SequentialStorageTest, OutOfCapacityTest) {
  alaya::SequentialStorage<int, uint32_t> storage;
  storage.init(sizeof(int), 1, 64);

  int data1 = 42;
  int data2 = 100;

  uint32_t id1 = storage.insert(&data1);
  uint32_t id2 = storage.insert(&data2);

  EXPECT_EQ(id1, 0);
  EXPECT_EQ(id2, static_cast<uint32_t>(-1));
}
}  // namespace alaya
