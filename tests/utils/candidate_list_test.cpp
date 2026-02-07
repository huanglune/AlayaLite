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

#include "utils/candidate_list.hpp"
#include <gtest/gtest.h>
#include "utils/thread_pool.hpp"

namespace alaya {

class CandidateListTest : public ::testing::Test {
 protected:
  void SetUp() override { pool_ = new CandidateList<float, int>(10, 5); }

  void TearDown() override { delete pool_; }

  CandidateList<float, int> *pool_;
};

TEST_F(CandidateListTest, InsertBoundaryTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 5.0);

  EXPECT_FALSE(pool_->insert(6, 6.0));
  EXPECT_EQ(pool_->size(), 5);
}

TEST_F(CandidateListTest, PopTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  EXPECT_EQ(pool_->top(), 2);

  EXPECT_EQ(pool_->pop(), 2);
  EXPECT_EQ(pool_->pop(), 1);
  EXPECT_EQ(pool_->pop(), 3);
}

// Test for multiple insertions and pops
TEST_F(CandidateListTest, MultipleInsertAndPopTest) {
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 0.5);
  pool_->insert(5, 4.0);

  EXPECT_EQ(pool_->size(), 5);  // Check the current size

  // Pop elements and check
  EXPECT_EQ(pool_->pop(), 4);  // ID with the smallest distance

  pool_->insert(6, 2.0);  // Insert a new element

  // Pop all elements and check the order
  EXPECT_EQ(pool_->pop(), 2);
  EXPECT_EQ(pool_->pop(), 6);
  EXPECT_EQ(pool_->pop(), 1);
  EXPECT_EQ(pool_->pop(), 3);
  EXPECT_EQ(pool_->pop(), 5);
  EXPECT_EQ(pool_->has_next(), false);  // Finally should be empty
}

TEST_F(CandidateListTest, BoundaryConditionsTest) {
  // Fill the pool
  pool_->insert(1, 2.5);
  pool_->insert(2, 1.5);
  pool_->insert(3, 3.0);
  pool_->insert(4, 0.5);
  pool_->insert(5, 4.0);

  // Try to insert an element exceeding capacity
  EXPECT_FALSE(pool_->insert(6, 5.0));  // Should return false
  EXPECT_EQ(pool_->size(), 5);          // Size should still be 5

  // Try to insert a negative value
  EXPECT_TRUE(pool_->insert(7, -1.0));  // Should successfully insert
  EXPECT_EQ(pool_->size(), 5);          // Size should increase
}

// Performance test
TEST_F(CandidateListTest, PerformanceTest) {
  const int kNumElements = 10000;
  for (int i = 0; i < kNumElements; ++i) {
    pool_->insert(i, static_cast<float>(kNumElements - i));  // Insert elements
  }
  EXPECT_EQ(pool_->size(), std::min(kNumElements, 5));  // Check size
}

// Concurrent test
TEST_F(CandidateListTest, ConcurrentInsertTest) {
  const int kNumThreads = 10;
  const int kInsertsPerThread = 100;

  // Lambda function for inserting elements
  auto insert_function = [this](int thread_id) -> void {
    for (int i = 0; i < kInsertsPerThread; ++i) {
      pool_->insert(thread_id * kInsertsPerThread + i, static_cast<float>(i));
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(insert_function, i);  // Create threads for insertion
  }

  for (auto &thread : threads) {
    thread.join();  // Wait for all threads to finish
  }

  // Check final size
  EXPECT_LE(pool_->size(), 5);  // Size should be less than or equal to capacity
}

// Test find_bsearch function
TEST_F(CandidateListTest, FindBsearchTest) {
  pool_->insert(1, 1.0);
  pool_->insert(2, 2.0);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 5.0);

  // find_bsearch returns the position where dist should be inserted
  EXPECT_EQ(pool_->find_bsearch(0.5), 0);  // Before all elements
  EXPECT_EQ(pool_->find_bsearch(1.5), 1);  // Between 1.0 and 2.0
  EXPECT_EQ(pool_->find_bsearch(2.5), 2);  // Between 2.0 and 3.0
  EXPECT_EQ(pool_->find_bsearch(5.5), 5);  // After all elements
  EXPECT_EQ(pool_->find_bsearch(3.0), 3);  // Equal to existing element
}

// Test capacity function
TEST_F(CandidateListTest, CapacityTest) {
  EXPECT_EQ(pool_->capacity(), 5);

  CandidateList<float, int> large_pool(100, 50);
  EXPECT_EQ(large_pool.capacity(), 50);
}

// Test is_full function
TEST_F(CandidateListTest, IsFullTest) {
  EXPECT_FALSE(pool_->is_full());

  pool_->insert(1, 1.0);
  EXPECT_FALSE(pool_->is_full());

  pool_->insert(2, 2.0);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  EXPECT_FALSE(pool_->is_full());

  pool_->insert(5, 5.0);
  EXPECT_TRUE(pool_->is_full());
}

// Test id and dist accessors
TEST_F(CandidateListTest, AccessorsTest) {
  pool_->insert(10, 1.0);
  pool_->insert(20, 2.0);
  pool_->insert(30, 3.0);

  // Elements should be sorted by distance
  EXPECT_EQ(pool_->id(0), 10);
  EXPECT_EQ(pool_->id(1), 20);
  EXPECT_EQ(pool_->id(2), 30);

  EXPECT_FLOAT_EQ(pool_->dist(0), 1.0);
  EXPECT_FLOAT_EQ(pool_->dist(1), 2.0);
  EXPECT_FLOAT_EQ(pool_->dist(2), 3.0);
}

// Test has_next and next_id functions
TEST_F(CandidateListTest, HasNextAndNextIdTest) {
  EXPECT_FALSE(pool_->has_next());  // Empty pool

  pool_->insert(1, 1.0);
  EXPECT_TRUE(pool_->has_next());
  EXPECT_EQ(pool_->next_id(), 1);

  pool_->insert(2, 0.5);
  EXPECT_TRUE(pool_->has_next());
  EXPECT_EQ(pool_->next_id(), 2);  // Smallest distance

  pool_->pop();
  EXPECT_TRUE(pool_->has_next());
  EXPECT_EQ(pool_->next_id(), 1);

  pool_->pop();
  EXPECT_FALSE(pool_->has_next());
}

// Test get_id, set_checked, is_checked functions
TEST_F(CandidateListTest, CheckedFlagTest) {
  int id = 42;

  // Initially not checked
  EXPECT_FALSE(pool_->is_checked(id));
  EXPECT_EQ(pool_->get_id(id), 42);

  // Set checked flag
  pool_->set_checked(id);
  EXPECT_TRUE(pool_->is_checked(id));
  EXPECT_EQ(pool_->get_id(id), 42);  // get_id should still return original ID

  // Test with larger ID
  int large_id = 1000000;
  EXPECT_FALSE(pool_->is_checked(large_id));
  pool_->set_checked(large_id);
  EXPECT_TRUE(pool_->is_checked(large_id));
  EXPECT_EQ(pool_->get_id(large_id), 1000000);
}

// Test empty pool behavior
TEST_F(CandidateListTest, EmptyPoolTest) {
  EXPECT_EQ(pool_->size(), 0);
  EXPECT_FALSE(pool_->is_full());
  EXPECT_FALSE(pool_->has_next());
  EXPECT_EQ(pool_->capacity(), 5);
}

// Test single element pool
TEST_F(CandidateListTest, SingleElementTest) {
  pool_->insert(42, 3.14);

  EXPECT_EQ(pool_->size(), 1);
  EXPECT_FALSE(pool_->is_full());
  EXPECT_TRUE(pool_->has_next());
  EXPECT_EQ(pool_->top(), 42);
  EXPECT_EQ(pool_->id(0), 42);
  EXPECT_FLOAT_EQ(pool_->dist(0), 3.14);

  EXPECT_EQ(pool_->pop(), 42);
  EXPECT_FALSE(pool_->has_next());
}

// Test insertion with same distance
TEST_F(CandidateListTest, SameDistanceInsertTest) {
  pool_->insert(1, 2.0);
  pool_->insert(2, 2.0);
  pool_->insert(3, 2.0);

  EXPECT_EQ(pool_->size(), 3);

  // All should be accessible
  int popped1 = pool_->pop();
  int popped2 = pool_->pop();
  int popped3 = pool_->pop();

  // All IDs should be popped
  EXPECT_TRUE((popped1 == 1 || popped1 == 2 || popped1 == 3));
  EXPECT_TRUE((popped2 == 1 || popped2 == 2 || popped2 == 3));
  EXPECT_TRUE((popped3 == 1 || popped3 == 2 || popped3 == 3));
  EXPECT_NE(popped1, popped2);
  EXPECT_NE(popped2, popped3);
  EXPECT_NE(popped1, popped3);
}

// Test insert replaces worst element when full
TEST_F(CandidateListTest, InsertReplacesWorstTest) {
  pool_->insert(1, 1.0);
  pool_->insert(2, 2.0);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 5.0);

  EXPECT_TRUE(pool_->is_full());
  EXPECT_EQ(pool_->size(), 5);

  // Insert element with smaller distance - should replace the worst (5.0)
  EXPECT_TRUE(pool_->insert(6, 0.5));
  EXPECT_EQ(pool_->size(), 5);

  // Check that 0.5 is now the smallest and 5.0 is gone
  EXPECT_EQ(pool_->top(), 6);
  EXPECT_FLOAT_EQ(pool_->dist(0), 0.5);
  EXPECT_FLOAT_EQ(pool_->dist(4), 4.0);  // 5.0 should be replaced
}

// Test with different data types
TEST(CandidateListTypesTest, DoubleDistanceTest) {
  CandidateList<double, uint32_t> pool(100, 10);

  pool.insert(1, 1.111111);
  pool.insert(2, 2.222222);
  pool.insert(3, 0.000001);

  EXPECT_EQ(pool.size(), 3);
  EXPECT_EQ(pool.top(), 3);
  EXPECT_DOUBLE_EQ(pool.dist(0), 0.000001);
}

// Test cursor behavior after multiple operations
TEST_F(CandidateListTest, CursorBehaviorTest) {
  pool_->insert(1, 3.0);
  pool_->insert(2, 1.0);
  pool_->insert(3, 2.0);

  // cur_ should point to smallest (id=2, dist=1.0)
  EXPECT_EQ(pool_->top(), 2);

  // Pop moves cursor
  pool_->pop();
  EXPECT_EQ(pool_->top(), 3);  // Next smallest is id=3, dist=2.0

  // Insert smaller element should update cursor
  pool_->insert(4, 0.5);
  EXPECT_EQ(pool_->next_id(), 4);
}

// Test kMask constant
TEST_F(CandidateListTest, MaskConstantTest) {
  EXPECT_EQ((CandidateList<float, int>::kMask), 2147483647);
  EXPECT_EQ((CandidateList<float, int>::kMask), 0x7FFFFFFF);
}

// Test large capacity pool
TEST(CandidateListLargeTest, LargeCapacityTest) {
  const int kCapacity = 1000;
  CandidateList<float, int> pool(10000, kCapacity);

  for (int i = 0; i < 2000; ++i) {
    pool.insert(i, static_cast<float>(i % 500));
  }

  EXPECT_EQ(pool.size(), kCapacity);
  EXPECT_TRUE(pool.is_full());

  // Verify sorted order
  for (size_t i = 1; i < pool.size(); ++i) {
    EXPECT_LE(pool.dist(i - 1), pool.dist(i));
  }
}

// Test reverse order insertion
TEST_F(CandidateListTest, ReverseOrderInsertTest) {
  pool_->insert(5, 5.0);
  pool_->insert(4, 4.0);
  pool_->insert(3, 3.0);
  pool_->insert(2, 2.0);
  pool_->insert(1, 1.0);

  // Should be sorted by distance
  EXPECT_EQ(pool_->id(0), 1);
  EXPECT_EQ(pool_->id(1), 2);
  EXPECT_EQ(pool_->id(2), 3);
  EXPECT_EQ(pool_->id(3), 4);
  EXPECT_EQ(pool_->id(4), 5);
}

// Test emplace_insert function
TEST_F(CandidateListTest, EmplaceInsertTest) {
  // First fill the pool with insert
  pool_->insert(1, 1.0);
  pool_->insert(2, 2.0);
  pool_->insert(3, 3.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 5.0);

  EXPECT_TRUE(pool_->is_full());

  // emplace_insert should replace the worst element if new dist is smaller
  pool_->emplace_insert(6, 0.5);

  // Check that 0.5 is now the smallest
  EXPECT_FLOAT_EQ(pool_->dist(0), 0.5);
  EXPECT_EQ(pool_->id(0), 6);

  // emplace_insert with larger distance should do nothing
  pool_->emplace_insert(7, 10.0);

  // Size should still be 5 and max distance should be 4.0
  EXPECT_EQ(pool_->size(), 5);
  EXPECT_FLOAT_EQ(pool_->dist(4), 4.0);
}

// Test emplace_insert maintains sorted order
TEST_F(CandidateListTest, EmplaceInsertSortedOrderTest) {
  pool_->insert(1, 1.0);
  pool_->insert(2, 3.0);
  pool_->insert(3, 5.0);
  pool_->insert(4, 7.0);
  pool_->insert(5, 9.0);

  // Insert in the middle
  pool_->emplace_insert(6, 4.0);

  // Verify sorted order
  EXPECT_FLOAT_EQ(pool_->dist(0), 1.0);
  EXPECT_FLOAT_EQ(pool_->dist(1), 3.0);
  EXPECT_FLOAT_EQ(pool_->dist(2), 4.0);
  EXPECT_FLOAT_EQ(pool_->dist(3), 5.0);
  EXPECT_FLOAT_EQ(pool_->dist(4), 7.0);

  EXPECT_EQ(pool_->id(2), 6);  // New element at position 2
}

// Test to_search_result basic functionality
TEST_F(CandidateListTest, ToSearchResultBasicTest) {
  pool_->insert(10, 1.0);
  pool_->insert(20, 3.0);
  pool_->insert(30, 2.0);

  auto result = pool_->to_search_result();

  EXPECT_EQ(result.ids_.size(), 3);
  EXPECT_EQ(result.distances_.size(), 3);

  // Elements should be in sorted order by distance
  EXPECT_EQ(result.ids_[0], 10);
  EXPECT_FLOAT_EQ(result.distances_[0], 1.0);
  EXPECT_EQ(result.ids_[1], 30);
  EXPECT_FLOAT_EQ(result.distances_[1], 2.0);
  EXPECT_EQ(result.ids_[2], 20);
  EXPECT_FLOAT_EQ(result.distances_[2], 3.0);
}

// Test to_search_result on empty pool
TEST_F(CandidateListTest, ToSearchResultEmptyTest) {
  auto result = pool_->to_search_result();

  EXPECT_EQ(result.ids_.size(), 0);
  EXPECT_EQ(result.distances_.size(), 0);
}

// Test to_search_result strips checked flag after pop
TEST_F(CandidateListTest, ToSearchResultAfterPopTest) {
  pool_->insert(1, 1.0);
  pool_->insert(2, 2.0);
  pool_->insert(3, 3.0);

  // Pop sets the checked flag on the internal id
  pool_->pop();

  auto result = pool_->to_search_result();

  EXPECT_EQ(result.ids_.size(), 3);
  // get_id should strip the checked flag, so all ids are clean
  EXPECT_EQ(result.ids_[0], 1);
  EXPECT_EQ(result.ids_[1], 2);
  EXPECT_EQ(result.ids_[2], 3);
}

// Test to_search_result on full pool
TEST_F(CandidateListTest, ToSearchResultFullPoolTest) {
  pool_->insert(1, 5.0);
  pool_->insert(2, 3.0);
  pool_->insert(3, 1.0);
  pool_->insert(4, 4.0);
  pool_->insert(5, 2.0);

  EXPECT_TRUE(pool_->is_full());

  auto result = pool_->to_search_result();

  EXPECT_EQ(result.ids_.size(), 5);
  EXPECT_EQ(result.distances_.size(), 5);

  // Verify sorted order: 1.0, 2.0, 3.0, 4.0, 5.0
  EXPECT_EQ(result.ids_[0], 3);
  EXPECT_FLOAT_EQ(result.distances_[0], 1.0);
  EXPECT_EQ(result.ids_[1], 5);
  EXPECT_FLOAT_EQ(result.distances_[1], 2.0);
  EXPECT_EQ(result.ids_[2], 2);
  EXPECT_FLOAT_EQ(result.distances_[2], 3.0);
  EXPECT_EQ(result.ids_[3], 4);
  EXPECT_FLOAT_EQ(result.distances_[3], 4.0);
  EXPECT_EQ(result.ids_[4], 1);
  EXPECT_FLOAT_EQ(result.distances_[4], 5.0);
}

}  // namespace alaya
