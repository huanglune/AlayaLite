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

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "storage/diskann/meta_file.hpp"

namespace alaya {

// ============================================================================
// FreeList Tests
// ============================================================================

TEST(FreeListTest, PushPop) {
  FreeList fl;
  EXPECT_TRUE(fl.empty());
  EXPECT_EQ(fl.size(), 0U);
  EXPECT_EQ(fl.pop(), -1);

  fl.push(10);
  fl.push(20);
  fl.push(30);
  EXPECT_EQ(fl.size(), 3U);
  EXPECT_FALSE(fl.empty());

  // LIFO order
  EXPECT_EQ(fl.pop(), 30);
  EXPECT_EQ(fl.pop(), 20);
  EXPECT_EQ(fl.pop(), 10);
  EXPECT_EQ(fl.pop(), -1);
  EXPECT_TRUE(fl.empty());
}

TEST(FreeListTest, Clear) {
  FreeList fl;
  fl.push(1);
  fl.push(2);
  fl.clear();
  EXPECT_TRUE(fl.empty());
  EXPECT_EQ(fl.pop(), -1);
}

// ============================================================================
// MetaFileHeader Tests
// ============================================================================

TEST(MetaFileHeaderTest, SizeIs128Bytes) {
  EXPECT_EQ(sizeof(MetaFileHeader), kMetaFileHeaderSize);
  EXPECT_EQ(sizeof(MetaFileHeader), 128U);
}

TEST(MetaFileHeaderTest, InitSetsFields) {
  MetaFileHeader header;
  header.init(1024, 128, 64, 1, 0);

  EXPECT_EQ(header.magic_, kMetaFileMagic);
  EXPECT_EQ(header.version_, kMetaFileVersion);
  EXPECT_EQ(header.num_capacity_, 1024U);
  EXPECT_EQ(header.dim_, 128U);
  EXPECT_EQ(header.max_degree_, 64U);
  EXPECT_EQ(header.metric_type_, 1U);
  EXPECT_EQ(header.data_type_, 0U);
  EXPECT_EQ(header.num_active_points_, 0U);
  EXPECT_EQ(header.delete_watermark_, 0U);
  EXPECT_EQ(header.bitmap_offset_, kMetaFileHeaderSize);
  EXPECT_FLOAT_EQ(header.alpha_, 1.2F);
}

TEST(MetaFileHeaderTest, ChecksumRoundTrip) {
  MetaFileHeader header;
  header.init(1024, 128, 64);
  header.update_checksum();

  EXPECT_TRUE(header.verify_checksum());

  // Mutate a full 64-bit block to avoid XOR-rotate collision
  header.build_timestamp_ = 0xDEADBEEFCAFEBABE;
  EXPECT_FALSE(header.verify_checksum());
}

TEST(MetaFileHeaderTest, IsValidChecks) {
  MetaFileHeader header;
  header.init(1024, 128, 64);
  EXPECT_TRUE(header.is_valid());

  // Bad magic
  MetaFileHeader bad_magic = header;
  bad_magic.magic_ = 0;
  EXPECT_FALSE(bad_magic.is_valid());

  // Zero dim
  MetaFileHeader bad_dim = header;
  bad_dim.dim_ = 0;
  EXPECT_FALSE(bad_dim.is_valid());

  // Zero capacity
  MetaFileHeader bad_cap = header;
  bad_cap.num_capacity_ = 0;
  EXPECT_FALSE(bad_cap.is_valid());
}

TEST(MetaFileHeaderTest, CapacityAlignsToDim) {
  MetaFileHeader header;
  header.init(100, 10, 32);
  // dim_aligned_ should be rounded up: 10 * 4 = 40 -> round_up_pow2(40, 64) = 64 -> / 4 = 16
  EXPECT_EQ(header.dim_aligned_, 16U);
}

// ============================================================================
// MetaFile Tests — Fixture with temp file cleanup
// ============================================================================

class MetaFileTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (const auto &f : temp_files_) {
      std::filesystem::remove(f);
      std::filesystem::remove(f + ".tmp");
    }
  }

  auto make_path(const std::string &name) -> std::string {
    std::string path = "/tmp/meta_file_test_" + name + ".meta";
    temp_files_.push_back(path);
    return path;
  }

 private:
  std::vector<std::string> temp_files_;
};

// -------------------------------------------------------------------------
// Create / Open / Close basics
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, CreateAndBasicState) {
  auto path = make_path("create_basic");
  MetaFile meta;
  meta.create(path, 1024, 128, 64);

  EXPECT_TRUE(meta.is_open());
  EXPECT_FALSE(meta.is_dirty());  // saved during create
  EXPECT_EQ(meta.capacity(), math::round_up_pow2(uint32_t{1024}, 64));
  EXPECT_EQ(meta.dimension(), 128U);
  EXPECT_EQ(meta.max_degree(), 64U);
  EXPECT_EQ(meta.num_active_points(), 0U);
  EXPECT_EQ(meta.path(), path);

  meta.close();
  EXPECT_FALSE(meta.is_open());
}

TEST_F(MetaFileTest, CapacityRoundsUpTo64) {
  auto path = make_path("cap_align");
  MetaFile meta;
  meta.create(path, 100, 16, 32);
  // 100 -> round_up_pow2(100, 64) = 128
  EXPECT_EQ(meta.capacity(), 128U);
}

TEST_F(MetaFileTest, CreateAlreadyOpenThrows) {
  auto path = make_path("double_create");
  MetaFile meta;
  meta.create(path, 256, 16, 32);
  EXPECT_THROW(meta.create(path, 256, 16, 32), std::runtime_error);
}

TEST_F(MetaFileTest, OpenNonexistentThrows) {
  MetaFile meta;
  EXPECT_THROW(meta.open("/tmp/nonexistent_meta_file_xyz.meta"), std::runtime_error);
}

TEST_F(MetaFileTest, SaveNotOpenThrows) {
  MetaFile meta;
  EXPECT_THROW(meta.save(), std::runtime_error);
}

// -------------------------------------------------------------------------
// Persistence: create -> close -> open
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, PersistenceRoundTrip) {
  auto path = make_path("persist");

  // Create and populate
  {
    MetaFile meta;
    meta.create(path, 1024, 128, 64, 1, 0);
    meta.set_entry_point(42);
    meta.set_build_timestamp(123456789);
    meta.set_alpha(1.5F);

    // Allocate some slots
    for (int i = 0; i < 10; ++i) {
      (void)meta.allocate_slot();
    }
    EXPECT_EQ(meta.num_active_points(), 10U);

    // Delete slot 3 and 7
    meta.set_invalid(3);
    meta.set_invalid(7);
    EXPECT_EQ(meta.num_active_points(), 8U);

    meta.close();
  }

  // Reopen and verify
  {
    MetaFile meta;
    meta.open(path);

    EXPECT_TRUE(meta.is_open());
    EXPECT_EQ(meta.capacity(), 1024U);
    EXPECT_EQ(meta.dimension(), 128U);
    EXPECT_EQ(meta.max_degree(), 64U);
    EXPECT_EQ(meta.entry_point(), 42U);
    EXPECT_FLOAT_EQ(meta.header().alpha_, 1.5F);
    EXPECT_EQ(meta.header().build_timestamp_, 123456789U);
    EXPECT_EQ(meta.num_active_points(), 8U);

    // Bitmap state preserved: slots 0-9 valid except 3, 7
    for (uint32_t i = 0; i < 10; ++i) {
      if (i == 3 || i == 7) {
        EXPECT_FALSE(meta.is_valid(i)) << "Slot " << i << " should be invalid";
      } else {
        EXPECT_TRUE(meta.is_valid(i)) << "Slot " << i << " should be valid";
      }
    }
    for (uint32_t i = 10; i < 20; ++i) {
      EXPECT_FALSE(meta.is_valid(i));
    }

    meta.close();
  }
}

// -------------------------------------------------------------------------
// Node validity management
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, SetValidAndInvalid) {
  auto path = make_path("validity");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  EXPECT_FALSE(meta.is_valid(0));
  meta.set_valid(0);
  EXPECT_TRUE(meta.is_valid(0));
  EXPECT_EQ(meta.num_active_points(), 1U);

  // Idempotent set_valid
  meta.set_valid(0);
  EXPECT_EQ(meta.num_active_points(), 1U);

  meta.set_invalid(0);
  EXPECT_FALSE(meta.is_valid(0));
  EXPECT_EQ(meta.num_active_points(), 0U);

  // Idempotent set_invalid
  meta.set_invalid(0);
  EXPECT_EQ(meta.num_active_points(), 0U);
}

TEST_F(MetaFileTest, IsValidOutOfRange) {
  auto path = make_path("oor");
  MetaFile meta;
  meta.create(path, 64, 16, 32);

  EXPECT_FALSE(meta.is_valid(meta.capacity()));
  EXPECT_FALSE(meta.is_valid(meta.capacity() + 100));
}

TEST_F(MetaFileTest, SetValidOutOfRangeThrows) {
  auto path = make_path("set_oor");
  MetaFile meta;
  meta.create(path, 64, 16, 32);

  EXPECT_THROW(meta.set_valid(meta.capacity()), std::out_of_range);
}

TEST_F(MetaFileTest, SetInvalidOutOfRangeThrows) {
  auto path = make_path("inv_oor");
  MetaFile meta;
  meta.create(path, 64, 16, 32);

  EXPECT_THROW(meta.set_invalid(meta.capacity()), std::out_of_range);
}

TEST_F(MetaFileTest, DeleteWatermarkTracking) {
  auto path = make_path("watermark");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  meta.set_valid(5);
  meta.set_valid(50);
  meta.set_valid(100);

  EXPECT_EQ(meta.header().delete_watermark_, 0U);

  meta.set_invalid(5);
  EXPECT_EQ(meta.header().delete_watermark_, 6U);

  meta.set_invalid(100);
  EXPECT_EQ(meta.header().delete_watermark_, 101U);

  // Lower deletion doesn't reduce watermark
  meta.set_invalid(50);
  EXPECT_EQ(meta.header().delete_watermark_, 101U);
}

// -------------------------------------------------------------------------
// Slot allocation
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, AllocateSlotSequential) {
  auto path = make_path("alloc_seq");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  // First allocations should be sequential via bitmap scan
  std::vector<int32_t> slots;
  slots.reserve(10);
  for (int i = 0; i < 10; ++i) {
    slots.push_back(meta.allocate_slot());
  }
  EXPECT_EQ(meta.num_active_points(), 10U);

  // All slots should be valid and unique
  for (int32_t s : slots) {
    EXPECT_GE(s, 0);
    EXPECT_TRUE(meta.is_valid(static_cast<uint32_t>(s)));
  }
  // Check uniqueness
  std::ranges::sort(slots);
  auto [dup_begin, dup_end] = std::ranges::unique(slots);
  EXPECT_EQ(dup_begin, slots.end());
}

TEST_F(MetaFileTest, AllocateSlotReusesFreed) {
  auto path = make_path("alloc_reuse");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  // Allocate 5 slots
  std::vector<int32_t> first_batch;
  first_batch.reserve(5);
  for (int i = 0; i < 5; ++i) {
    first_batch.push_back(meta.allocate_slot());
  }

  // Free slot 2 and 4
  meta.set_invalid(static_cast<uint32_t>(first_batch[2]));
  meta.set_invalid(static_cast<uint32_t>(first_batch[4]));
  EXPECT_EQ(meta.num_active_points(), 3U);

  // Next allocations should reuse freed slots (LIFO from freelist)
  int32_t reused1 = meta.allocate_slot();
  int32_t reused2 = meta.allocate_slot();

  // Should get back the freed slots (LIFO: 4 then 2)
  EXPECT_EQ(reused1, first_batch[4]);
  EXPECT_EQ(reused2, first_batch[2]);
  EXPECT_EQ(meta.num_active_points(), 5U);
}

TEST_F(MetaFileTest, AllocateSlotAutoGrow) {
  auto path = make_path("alloc_grow");
  MetaFile meta;
  meta.create(path, 64, 16, 32);  // Small capacity

  uint32_t initial_cap = meta.capacity();

  // Fill all slots
  for (uint32_t i = 0; i < initial_cap; ++i) {
    int32_t slot = meta.allocate_slot();
    EXPECT_GE(slot, 0);
  }
  EXPECT_EQ(meta.num_active_points(), initial_cap);

  // Next allocation should trigger grow
  int32_t overflow_slot = meta.allocate_slot();
  EXPECT_GE(overflow_slot, 0);
  EXPECT_GT(meta.capacity(), initial_cap);
  EXPECT_EQ(meta.num_active_points(), initial_cap + 1);

  // The grown bitmap should be 64-aligned
  EXPECT_EQ(meta.capacity() % 64, 0U);
}

// -------------------------------------------------------------------------
// Freelist rebuild from bitmap on open
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, FreelistRebuildOnOpen) {
  auto path = make_path("rebuild");
  int32_t freed_slot = -1;

  // Create, allocate, delete, close
  {
    MetaFile meta;
    meta.create(path, 256, 16, 32);

    for (int i = 0; i < 10; ++i) {
      (void)meta.allocate_slot();
    }

    // Delete slot 5
    meta.set_invalid(5);
    freed_slot = 5;
    EXPECT_EQ(meta.num_active_points(), 9U);
    meta.close();
  }

  // Reopen — freelist should be rebuilt from bitmap
  {
    MetaFile meta;
    meta.open(path);
    EXPECT_EQ(meta.num_active_points(), 9U);

    // Allocate should reuse the freed slot
    int32_t reused = meta.allocate_slot();
    EXPECT_EQ(reused, freed_slot);
    EXPECT_EQ(meta.num_active_points(), 10U);
    meta.close();
  }
}

TEST_F(MetaFileTest, FreelistRebuildMultipleDeletes) {
  auto path = make_path("rebuild_multi");

  {
    MetaFile meta;
    meta.create(path, 256, 16, 32);

    for (int i = 0; i < 20; ++i) {
      (void)meta.allocate_slot();
    }

    // Delete slots 3, 7, 15
    meta.set_invalid(3);
    meta.set_invalid(7);
    meta.set_invalid(15);
    EXPECT_EQ(meta.num_active_points(), 17U);
    meta.close();
  }

  {
    MetaFile meta;
    meta.open(path);
    EXPECT_EQ(meta.num_active_points(), 17U);

    // Should be able to reuse all 3 freed slots
    std::vector<int32_t> reused;
    reused.reserve(3);
    for (int i = 0; i < 3; ++i) {
      reused.push_back(meta.allocate_slot());
    }
    EXPECT_EQ(meta.num_active_points(), 20U);

    // All reused slots should be among {3, 7, 15}
    std::ranges::sort(reused);
    EXPECT_EQ(reused, (std::vector<int32_t>{3, 7, 15}));
    meta.close();
  }
}

// -------------------------------------------------------------------------
// Dynamic capacity growth
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, GrowExplicit) {
  auto path = make_path("grow");
  MetaFile meta;
  meta.create(path, 64, 16, 32);

  uint32_t old_cap = meta.capacity();
  meta.grow(200);  // Will be rounded up to 256 (next multiple of 64)
  EXPECT_EQ(meta.capacity(), 256U);
  EXPECT_GT(meta.capacity(), old_cap);

  // Old data preserved, new slots available
  meta.set_valid(0);
  meta.set_valid(old_cap);  // This would have been out of range before grow
  EXPECT_TRUE(meta.is_valid(0));
  EXPECT_TRUE(meta.is_valid(old_cap));
  EXPECT_EQ(meta.num_active_points(), 2U);
}

TEST_F(MetaFileTest, GrowSmallerIsNoop) {
  auto path = make_path("grow_noop");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  uint32_t cap = meta.capacity();
  meta.grow(100);  // Smaller, should be ignored
  EXPECT_EQ(meta.capacity(), cap);
}

TEST_F(MetaFileTest, GrowPersistence) {
  auto path = make_path("grow_persist");

  {
    MetaFile meta;
    meta.create(path, 64, 16, 32);

    // Allocate some slots, grow, allocate more
    for (int i = 0; i < 10; ++i) {
      (void)meta.allocate_slot();
    }
    meta.grow(256);
    meta.set_valid(100);
    EXPECT_EQ(meta.num_active_points(), 11U);
    meta.close();
  }

  {
    MetaFile meta;
    meta.open(path);
    EXPECT_EQ(meta.capacity(), 256U);
    EXPECT_EQ(meta.num_active_points(), 11U);
    EXPECT_TRUE(meta.is_valid(100));
    meta.close();
  }
}

// -------------------------------------------------------------------------
// Move semantics
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, MoveConstruct) {
  auto path = make_path("move_ctor");
  MetaFile meta;
  meta.create(path, 256, 16, 32);
  (void)meta.allocate_slot();
  (void)meta.allocate_slot();

  MetaFile moved(std::move(meta));
  EXPECT_TRUE(moved.is_open());
  EXPECT_EQ(moved.num_active_points(), 2U);
  EXPECT_EQ(moved.capacity(), 256U);

  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(meta.is_open());
}

TEST_F(MetaFileTest, MoveAssign) {
  auto path1 = make_path("move_a1");
  auto path2 = make_path("move_a2");

  MetaFile meta1;
  meta1.create(path1, 128, 16, 32);
  (void)meta1.allocate_slot();

  MetaFile meta2;
  meta2.create(path2, 256, 32, 64);

  meta2 = std::move(meta1);
  EXPECT_TRUE(meta2.is_open());
  EXPECT_EQ(meta2.capacity(), 128U);
  EXPECT_EQ(meta2.num_active_points(), 1U);

  // NOLINTNEXTLINE(bugprone-use-after-move)
  EXPECT_FALSE(meta1.is_open());
}

// -------------------------------------------------------------------------
// Dirty tracking
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, DirtyTracking) {
  auto path = make_path("dirty");
  MetaFile meta;
  meta.create(path, 256, 16, 32);
  EXPECT_FALSE(meta.is_dirty());  // create() calls save()

  meta.set_valid(0);
  EXPECT_TRUE(meta.is_dirty());

  meta.save();
  EXPECT_FALSE(meta.is_dirty());

  meta.set_entry_point(10);
  EXPECT_TRUE(meta.is_dirty());

  meta.save();
  EXPECT_FALSE(meta.is_dirty());

  meta.set_invalid(0);
  EXPECT_TRUE(meta.is_dirty());

  // close() auto-saves if dirty
  meta.close();
  EXPECT_FALSE(meta.is_dirty());
}

// -------------------------------------------------------------------------
// Stress: allocate all, free all, reallocate all
// -------------------------------------------------------------------------

TEST_F(MetaFileTest, AllocFreeReallocCycle) {
  auto path = make_path("cycle");
  MetaFile meta;
  meta.create(path, 256, 16, 32);

  uint32_t cap = meta.capacity();

  // Allocate all
  std::vector<int32_t> slots;
  slots.reserve(cap);
  for (uint32_t i = 0; i < cap; ++i) {
    slots.push_back(meta.allocate_slot());
  }
  EXPECT_EQ(meta.num_active_points(), cap);

  // Free all
  for (int32_t s : slots) {
    meta.free_slot(static_cast<uint32_t>(s));
  }
  EXPECT_EQ(meta.num_active_points(), 0U);

  // Reallocate all — should reuse freed slots
  for (uint32_t i = 0; i < cap; ++i) {
    int32_t s = meta.allocate_slot();
    EXPECT_GE(s, 0);
    EXPECT_LT(static_cast<uint32_t>(s), cap);
  }
  EXPECT_EQ(meta.num_active_points(), cap);
}

}  // namespace alaya
