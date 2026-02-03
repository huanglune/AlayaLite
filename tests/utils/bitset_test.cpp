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

#include "utils/bitset.hpp"
#include <gtest/gtest.h>

namespace alaya {

// ============================================================================
// DynamicBitset Tests
// ============================================================================

class DynamicBitsetTest : public ::testing::Test {
 protected:
  void SetUp() override { bitset_ = new DynamicBitset(1000); }

  void TearDown() override { delete bitset_; }

  DynamicBitset *bitset_;
};

TEST_F(DynamicBitsetTest, InitialStateTest) {
  // All bits should be initially unset
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FALSE(bitset_->get(i));
  }
}

TEST_F(DynamicBitsetTest, SetAndGetTest) {
  bitset_->set(0);
  bitset_->set(63);
  bitset_->set(64);
  bitset_->set(127);
  bitset_->set(999);

  EXPECT_TRUE(bitset_->get(0));
  EXPECT_TRUE(bitset_->get(63));
  EXPECT_TRUE(bitset_->get(64));
  EXPECT_TRUE(bitset_->get(127));
  EXPECT_TRUE(bitset_->get(999));

  // Check that adjacent bits are not affected
  EXPECT_FALSE(bitset_->get(1));
  EXPECT_FALSE(bitset_->get(62));
  EXPECT_FALSE(bitset_->get(65));
  EXPECT_FALSE(bitset_->get(998));
}

TEST_F(DynamicBitsetTest, ResetTest) {
  bitset_->set(100);
  EXPECT_TRUE(bitset_->get(100));

  bitset_->reset(100);
  EXPECT_FALSE(bitset_->get(100));
}

TEST_F(DynamicBitsetTest, MultipleSetResetTest) {
  // Set multiple bits
  for (size_t i = 0; i < 500; i += 7) {
    bitset_->set(i);
  }

  // Verify they are set
  for (size_t i = 0; i < 500; i += 7) {
    EXPECT_TRUE(bitset_->get(i));
  }

  // Reset some of them
  for (size_t i = 0; i < 500; i += 14) {
    bitset_->reset(i);
  }

  // Verify the pattern
  for (size_t i = 0; i < 500; i += 7) {
    if ((i / 7) % 2 == 0) {
      EXPECT_FALSE(bitset_->get(i));
    } else {
      EXPECT_TRUE(bitset_->get(i));
    }
  }
}

TEST_F(DynamicBitsetTest, GetAddressTest) {
  void *addr = bitset_->get_address(0);
  EXPECT_NE(addr, nullptr);

  void *addr64 = bitset_->get_address(64);
  // Should be 8 bytes (sizeof(uint64_t)) apart
  EXPECT_EQ(static_cast<char *>(addr64) - static_cast<char *>(addr),
            static_cast<ptrdiff_t>(sizeof(uint64_t)));
}

// ============================================================================
// SparseBitset Tests
// ============================================================================

class SparseBitsetTest : public ::testing::Test {
 protected:
  SparseBitset bitset_;
};

TEST_F(SparseBitsetTest, InitialStateTest) {
  // All bits should be initially unset
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FALSE(bitset_.get(i));
  }
}

TEST_F(SparseBitsetTest, SetAndGetTest) {
  bitset_.set(0);
  bitset_.set(1000000);
  bitset_.set(999999999);

  EXPECT_TRUE(bitset_.get(0));
  EXPECT_TRUE(bitset_.get(1000000));
  EXPECT_TRUE(bitset_.get(999999999));

  EXPECT_FALSE(bitset_.get(1));
  EXPECT_FALSE(bitset_.get(999999));
}

TEST_F(SparseBitsetTest, ResetTest) {
  bitset_.set(42);
  EXPECT_TRUE(bitset_.get(42));

  bitset_.reset(42);
  EXPECT_FALSE(bitset_.get(42));
}

TEST_F(SparseBitsetTest, SparseDataTest) {
  // Test with very sparse data (large gaps between set bits)
  bitset_.set(1);
  bitset_.set(1000000);
  bitset_.set(2000000000);

  EXPECT_TRUE(bitset_.get(1));
  EXPECT_TRUE(bitset_.get(1000000));
  EXPECT_TRUE(bitset_.get(2000000000));

  // Memory efficient - only stores set positions
  EXPECT_FALSE(bitset_.get(500000));
  EXPECT_FALSE(bitset_.get(1000000000));
}

// ============================================================================
// HierarchicalBitset Tests
// ============================================================================

class HierarchicalBitsetTest : public ::testing::Test {
 protected:
  void SetUp() override { bitset_ = new HierarchicalBitset(10000); }

  void TearDown() override { delete bitset_; }

  HierarchicalBitset *bitset_;
};

TEST_F(HierarchicalBitsetTest, InitialStateTest) {
  // All bits should be initially unset
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FALSE(bitset_->get(i));
  }
}

TEST_F(HierarchicalBitsetTest, SetAndGetTest) {
  bitset_->set(0);
  bitset_->set(511);
  bitset_->set(512);
  bitset_->set(1000);

  EXPECT_TRUE(bitset_->get(0));
  EXPECT_TRUE(bitset_->get(511));
  EXPECT_TRUE(bitset_->get(512));
  EXPECT_TRUE(bitset_->get(1000));

  EXPECT_FALSE(bitset_->get(1));
  EXPECT_FALSE(bitset_->get(510));
  EXPECT_FALSE(bitset_->get(513));
}

TEST_F(HierarchicalBitsetTest, FindFirstSetEmptyTest) {
  // No bits set, should return -1
  EXPECT_EQ(bitset_->find_first_set(), -1);
}

TEST_F(HierarchicalBitsetTest, FindFirstSetSingleBitTest) {
  bitset_->set(500);
  EXPECT_EQ(bitset_->find_first_set(), 500);
}

TEST_F(HierarchicalBitsetTest, FindFirstSetMultipleBitsTest) {
  bitset_->set(1000);
  bitset_->set(500);
  bitset_->set(2000);

  // Should return the first set bit (smallest position)
  EXPECT_EQ(bitset_->find_first_set(), 500);
}

TEST_F(HierarchicalBitsetTest, FindFirstSetFirstBitTest) {
  bitset_->set(0);
  bitset_->set(5000);

  EXPECT_EQ(bitset_->find_first_set(), 0);
}

TEST_F(HierarchicalBitsetTest, FindFirstSetBlockBoundaryTest) {
  // Test at block boundaries (512 bits per block)
  bitset_->set(512);
  EXPECT_EQ(bitset_->find_first_set(), 512);

  HierarchicalBitset bitset2(10000);
  bitset2.set(1024);
  EXPECT_EQ(bitset2.find_first_set(), 1024);
}

}  // namespace alaya
