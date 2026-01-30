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

#include "utils/memory.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace alaya {

// Test AlignedBuffer allocation and alignment
TEST(AlignedBufferTest, BasicAllocation) {
  AlignedBuffer buf(kAlign4K);  // Allocate 4KB

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_GE(buf.size(), kAlign4K);
  // Check 4KB alignment
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.data()) % kAlign4K, 0);
  EXPECT_FALSE(buf.empty());
}

TEST(AlignedBufferTest, MoveSemantics) {
  AlignedBuffer buf1(kAlign4K);
  uint8_t *ptr = buf1.data();

  AlignedBuffer buf2(std::move(buf1));
  EXPECT_EQ(buf2.data(), ptr);
  EXPECT_TRUE(buf1.empty());   // NOLINT - moved-from state
  EXPECT_FALSE(buf2.empty());
}

TEST(AlignedBufferTest, Buffer64BAlignment) {
  AlignedBuffer64B buf(64);

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.data()) % kAlign64B, 0);
}

TEST(AlignedBufferTest, Buffer2MAlignment) {
  AlignedBuffer2M buf(kAlign2M);

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.data()) % kAlign2M, 0);
}

TEST(AlignedBufferTest, CustomAlignment) {
  constexpr size_t kCustomAlign = 8192;  // 8KB
  AlignedBufferT<kCustomAlign> buf(kCustomAlign);

  EXPECT_NE(buf.data(), nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buf.data()) % kCustomAlign, 0);
}

// Test AlignedAlloc with STL containers
TEST(AlignedAllocTest, VectorWithAlignedAlloc) {
  std::vector<float, AlignedAlloc<float, kAlign64B>> vec(1024);

  EXPECT_EQ(vec.size(), 1024);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(vec.data()) % kAlign64B, 0);
}

TEST(AlignedAllocTest, AutoAlignment) {
  // Small allocation should use 64B alignment
  std::vector<uint8_t, AlignedAlloc<uint8_t>> small_vec(100);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(small_vec.data()) % 64, 0);

  // Large allocation (>= 16KB) should use 2MB alignment
  std::vector<uint8_t, AlignedAlloc<uint8_t>> large_vec(32 * 1024);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(large_vec.data()) % (2 * 1024 * 1024), 0);
}

TEST(AlignedAllocTest, AllocatorEquality) {
  AlignedAlloc<int, kAlign64B> a1;
  AlignedAlloc<int, kAlign64B> a2;
  AlignedAlloc<int, kAlign4K> a3;

  EXPECT_TRUE(a1 == a2);
  EXPECT_FALSE(a1 == a3);
  EXPECT_FALSE(a1 != a2);
  EXPECT_TRUE(a1 != a3);
}

}  // namespace alaya
