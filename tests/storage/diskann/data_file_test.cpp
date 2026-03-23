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
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "storage/buffer/buffer_pool.hpp"
#include "storage/diskann/data_file.hpp"

namespace alaya {

// ============================================================================
// NeighborList Tests
// ============================================================================

TEST(NeighborListTest, EmptyList) {
  constexpr size_t kMaxNbrs = 4;
  std::vector<uint8_t> buf(sizeof(NeighborList<uint32_t>) + kMaxNbrs * sizeof(uint32_t), 0);
  auto *list = reinterpret_cast<NeighborList<uint32_t> *>(buf.data());

  list->num_neighbors_ = 0;

  EXPECT_TRUE(list->empty());
  EXPECT_EQ(list->size(), 0U);
  EXPECT_EQ(list->begin(), list->end());
}

TEST(NeighborListTest, NonEmptyList) {
  constexpr size_t kMaxNbrs = 4;
  std::vector<uint8_t> buf(sizeof(NeighborList<uint32_t>) + kMaxNbrs * sizeof(uint32_t), 0);
  auto *list = reinterpret_cast<NeighborList<uint32_t> *>(buf.data());

  list->num_neighbors_ = 3;
  list->neighbor_ids()[0] = 10;
  list->neighbor_ids()[1] = 20;
  list->neighbor_ids()[2] = 30;

  EXPECT_FALSE(list->empty());
  EXPECT_EQ(list->size(), 3U);
  EXPECT_EQ(list->neighbor_ids()[0], 10U);
  EXPECT_EQ(list->neighbor_ids()[1], 20U);
  EXPECT_EQ(list->neighbor_ids()[2], 30U);

  std::vector<uint32_t> ids(list->begin(), list->end());
  EXPECT_EQ(ids, (std::vector<uint32_t>{10, 20, 30}));
}

TEST(NeighborListTest, ConstAccess) {
  constexpr size_t kMaxNbrs = 2;
  std::vector<uint8_t> buf(sizeof(NeighborList<uint32_t>) + kMaxNbrs * sizeof(uint32_t), 0);
  auto *list = reinterpret_cast<NeighborList<uint32_t> *>(buf.data());

  list->num_neighbors_ = 2;
  list->neighbor_ids()[0] = 5;
  list->neighbor_ids()[1] = 15;

  const auto *const_list = list;
  EXPECT_EQ(const_list->size(), 2U);
  EXPECT_EQ(const_list->neighbor_ids()[0], 5U);
  EXPECT_EQ(const_list->neighbor_ids()[1], 15U);

  std::vector<uint32_t> ids(const_list->begin(), const_list->end());
  EXPECT_EQ(ids, (std::vector<uint32_t>{5, 15}));
}

// ============================================================================
// DataFile Tests - Fixture with temp file cleanup
// ============================================================================

class DataFileTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kDim = 8;
  static constexpr uint32_t kMaxDeg = 4;
  static constexpr uint32_t kCapacity = 64;
  static constexpr size_t kBpCapacity = 200;

  using BPType = BufferPool<uint32_t>;

  std::unique_ptr<BPType> bp_;

  void SetUp() override { bp_ = std::make_unique<BPType>(kBpCapacity, kDataBlockSize, 1); }

  void TearDown() override {
    for (const auto &f : temp_files_) {
      std::filesystem::remove(f);
    }
  }

  auto make_path(const std::string &name) -> std::string {
    std::string path = "/tmp/data_file_test_" + name + ".dat";
    temp_files_.push_back(path);
    return path;
  }

  static auto make_vector(uint32_t dim, uint32_t seed) -> std::vector<float> {
    std::vector<float> vec(dim);
    for (uint32_t i = 0; i < dim; ++i) {
      vec[i] = static_cast<float>(seed * 100 + i);
    }
    return vec;
  }

  static auto make_neighbors(uint32_t count, uint32_t start = 0) -> std::vector<uint32_t> {
    std::vector<uint32_t> nbrs(count);
    std::iota(nbrs.begin(), nbrs.end(), start);
    return nbrs;
  }

 private:
  std::vector<std::string> temp_files_;
};

// -------------------------------------------------------------------------
// Layout calculation tests
// -------------------------------------------------------------------------

TEST_F(DataFileTest, LayoutParameters) {
  auto path = make_path("layout");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_EQ(df.capacity(), kCapacity);
  EXPECT_EQ(df.dimension(), kDim);
  EXPECT_EQ(df.max_degree(), kMaxDeg);

  size_t expected_row = sizeof(uint32_t) + kMaxDeg * sizeof(uint32_t) + kDim * sizeof(float);
  EXPECT_EQ(df.row_size(), expected_row);

  auto expected_npb = static_cast<uint32_t>(kDataBlockSize / expected_row);
  EXPECT_EQ(df.nodes_per_block(), expected_npb);
  EXPECT_GT(df.nodes_per_block(), 0U);

  EXPECT_TRUE(df.is_open());
  EXPECT_TRUE(df.is_writable());
  EXPECT_EQ(df.path(), path);
}

TEST_F(DataFileTest, FileSizeCalculation) {
  auto path = make_path("filesize");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  uint32_t npb = df.nodes_per_block();
  uint64_t num_blocks = (static_cast<uint64_t>(kCapacity) + npb - 1) / npb;
  EXPECT_EQ(df.total_file_size(), num_blocks * kDataBlockSize);
}

// -------------------------------------------------------------------------
// Create / Open / Close basics
// -------------------------------------------------------------------------

TEST_F(DataFileTest, CreateAndBasicState) {
  auto path = make_path("create_basic");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_TRUE(df.is_open());
  EXPECT_TRUE(df.is_writable());
  EXPECT_EQ(df.path(), path);
  EXPECT_TRUE(std::filesystem::exists(path));

  df.close();
  EXPECT_FALSE(df.is_open());
  EXPECT_FALSE(df.is_writable());
}

TEST_F(DataFileTest, CreateAlreadyOpenThrows) {
  auto path = make_path("double_create");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(df.create(path, kCapacity, kDim, kMaxDeg), std::runtime_error);
}

TEST_F(DataFileTest, NullBufferPoolThrows) {
  using DF = DataFile<float, uint32_t>;
  EXPECT_THROW(DF(nullptr), std::invalid_argument);
}

TEST_F(DataFileTest, OpenExistingFile) {
  auto path = make_path("open_existing");

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, true);

    EXPECT_TRUE(df.is_open());
    EXPECT_TRUE(df.is_writable());
    EXPECT_EQ(df.capacity(), kCapacity);
    EXPECT_EQ(df.dimension(), kDim);
    EXPECT_EQ(df.max_degree(), kMaxDeg);
    df.close();
  }
}

TEST_F(DataFileTest, OpenReadOnly) {
  auto path = make_path("open_readonly");

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    EXPECT_TRUE(df.is_open());
    EXPECT_FALSE(df.is_writable());
    df.close();
  }
}

TEST_F(DataFileTest, OpenAlreadyOpenThrows) {
  auto path = make_path("double_open");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(df.open(path, kCapacity, kDim, kMaxDeg), std::runtime_error);
}

TEST_F(DataFileTest, CloseIdempotent) {
  auto path = make_path("close_idem");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  df.close();
  EXPECT_FALSE(df.is_open());

  df.close();
  EXPECT_FALSE(df.is_open());
}

TEST_F(DataFileTest, GetNodeOutOfRangeThrows) {
  auto path = make_path("get_node_oor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(static_cast<void>(df.get_node(kCapacity)), std::out_of_range);
}

// -------------------------------------------------------------------------
// Row size too large throws
// -------------------------------------------------------------------------

TEST_F(DataFileTest, RowSizeExceedsBlockSizeThrows) {
  auto path = make_path("row_too_large");
  DataFile<float, uint32_t> df(bp_.get());

  EXPECT_THROW(df.create(path, 10, 1024, 1024), std::invalid_argument);
}

// -------------------------------------------------------------------------
// NodeRef read/write and mutation
// -------------------------------------------------------------------------

TEST_F(DataFileTest, GetNodeReadWrite) {
  auto path = make_path("get_node_rw");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 42);
  auto nbrs = make_neighbors(3, 10);

  {
    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(vec));
    ref.set_neighbors(std::span<const uint32_t>(nbrs));
  }

  {
    auto ref = df.get_node(0);
    auto read_vec = ref.vector();
    ASSERT_EQ(read_vec.size(), kDim);
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }

    auto read_nbrs = ref.neighbors();
    EXPECT_EQ(read_nbrs.size(), 3U);
    EXPECT_EQ(read_nbrs[0], 10U);
    EXPECT_EQ(read_nbrs[1], 11U);
    EXPECT_EQ(read_nbrs[2], 12U);
  }
}

TEST_F(DataFileTest, NodeRefMutableVectorAndNeighbors) {
  auto path = make_path("ref_mut");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 1);
  auto nbrs = make_neighbors(kMaxDeg, 0);

  {
    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(vec));
    ref.set_neighbors(std::span<const uint32_t>(nbrs));
  }

  {
    auto ref = df.get_node(0);
    auto mut_vec = ref.mutable_vector();
    for (auto &val : mut_vec) {
      val *= 2.0F;
    }
    auto &mut_nbrs = ref.mutable_neighbors();
    std::sort(mut_nbrs.begin(), mut_nbrs.end(), std::greater<uint32_t>()); // NOLINT
    ref.mark_dirty();
  }

  {
    auto ref = df.get_node(0);
    auto read_vec = ref.vector();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i] * 2.0F);
    }

    auto read_nbrs = ref.neighbors();
    EXPECT_EQ(read_nbrs.size(), kMaxDeg);
    for (uint32_t i = 0; i < kMaxDeg; ++i) {
      EXPECT_EQ(read_nbrs[i], kMaxDeg - 1 - i);
    }
  }
}

TEST_F(DataFileTest, ReadOnlyNodeRefRejectsMutation) {
  auto path = make_path("readonly_rejects_mutation");

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);
    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(make_vector(kDim, 7)));
    ref.set_neighbors(std::span<const uint32_t>(make_neighbors(2, 3)));
    df.close();
  }

  bp_->clear();

  DataFile<float, uint32_t> df(bp_.get());
  df.open(path, kCapacity, kDim, kMaxDeg, false);

  auto ref = df.get_node(0);
  auto vec = make_vector(kDim, 9);
  auto nbrs = make_neighbors(2, 5);
  EXPECT_THROW(ref.set_vector(std::span<const float>(vec)), std::runtime_error);
  EXPECT_THROW(ref.set_neighbors(std::span<const uint32_t>(nbrs)), std::runtime_error);
  EXPECT_THROW(static_cast<void>(ref.mutable_vector()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(ref.mutable_neighbors()), std::runtime_error);
  EXPECT_THROW(ref.mark_dirty(), std::runtime_error);
}

TEST_F(DataFileTest, SetNeighborsTooManyThrows) {
  auto path = make_path("too_many_nbrs");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  std::vector<uint32_t> too_many(kMaxDeg + 1);
  std::iota(too_many.begin(), too_many.end(), 0);

  auto ref = df.get_node(0);
  EXPECT_THROW(ref.set_neighbors(std::span<const uint32_t>(too_many)), std::length_error);
}

TEST_F(DataFileTest, SetVectorDimMismatchThrows) {
  auto path = make_path("vec_dim_err");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  std::vector<float> wrong_dim(kDim + 1, 1.0F);

  auto ref = df.get_node(0);
  EXPECT_THROW(ref.set_vector(std::span<const float>(wrong_dim)), std::invalid_argument);
}

// -------------------------------------------------------------------------
// Persistence via flush
// -------------------------------------------------------------------------

TEST_F(DataFileTest, PersistenceWithFlush) {
  auto path = make_path("persist");
  auto vec = make_vector(kDim, 99);
  auto nbrs = make_neighbors(3, 50);

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);

    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(vec));
    ref.set_neighbors(std::span<const uint32_t>(nbrs));

    df.flush();
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    auto ref = df.get_node(0);
    auto read_vec = ref.vector();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }

    auto read_nbrs = ref.neighbors();
    EXPECT_EQ(read_nbrs.size(), 3U);
    for (uint32_t i = 0; i < 3; ++i) {
      EXPECT_EQ(read_nbrs[i], 50 + i);
    }

    df.close();
  }
}

TEST_F(DataFileTest, ClosePersistsDirtyPages) {
  auto path = make_path("close_persist");
  auto vec = make_vector(kDim, 11);
  auto nbrs = make_neighbors(3, 20);

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);

    auto ref = df.get_node(2);
    ref.set_vector(std::span<const float>(vec));
    ref.set_neighbors(std::span<const uint32_t>(nbrs));
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    auto ref = df.get_node(2);
    auto read_vec = ref.vector();
    auto read_nbrs = ref.neighbors();
    ASSERT_EQ(read_vec.size(), kDim);
    ASSERT_EQ(read_nbrs.size(), nbrs.size());
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }
    for (size_t i = 0; i < nbrs.size(); ++i) {
      EXPECT_EQ(read_nbrs[i], nbrs[i]);
    }
  }
}

// -------------------------------------------------------------------------
// Prefetch / preload
// -------------------------------------------------------------------------

TEST_F(DataFileTest, PrefetchBlocksNoThrow) {
  auto path = make_path("prefetch");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  uint32_t blocks = df.num_blocks();
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < std::min<uint32_t>(blocks, 3); ++i) {
    ids.push_back(i);
  }

  EXPECT_NO_THROW(df.prefetch_blocks(std::span<const uint32_t>(ids)));
}

TEST_F(DataFileTest, PrefetchBlocksPopulateBufferPool) {
  auto path = make_path("prefetch_populates_cache");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  {
    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(make_vector(kDim, 7)));
    ref.set_neighbors(std::span<const uint32_t>(make_neighbors(2, 3)));
  }
  df.flush();
  bp_->clear();

  std::vector<uint32_t> ids{0};
  df.prefetch_blocks(std::span<const uint32_t>(ids));

  auto handle = bp_->get(0);
  EXPECT_FALSE(handle.empty());
}

TEST_F(DataFileTest, PreloadBlockOutOfRangeThrows) {
  auto path = make_path("preload_oor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(df.preload_block(df.num_blocks()), std::out_of_range);
}

// -------------------------------------------------------------------------
// Grow
// -------------------------------------------------------------------------

TEST_F(DataFileTest, GrowIncreasesCapacityAndFileSize) {
  auto path = make_path("grow");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  uint64_t old_total = df.total_file_size();
  uint32_t new_capacity = kCapacity * 2;

  df.grow(new_capacity);
  EXPECT_EQ(df.capacity(), new_capacity);
  EXPECT_GT(df.total_file_size(), old_total);
  EXPECT_GE(std::filesystem::file_size(path), df.total_file_size());
}

TEST_F(DataFileTest, GrowReadOnlyThrows) {
  auto path = make_path("grow_readonly");

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);
    EXPECT_THROW(df.grow(kCapacity * 2), std::runtime_error);
  }
}

// -------------------------------------------------------------------------
// Move semantics
// -------------------------------------------------------------------------

TEST_F(DataFileTest, MoveConstruct) {
  auto path = make_path("move_ctor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 42);
  {
    auto ref = df.get_node(0);
    ref.set_vector(std::span<const float>(vec));
  }

  DataFile<float, uint32_t> moved(std::move(df));
  EXPECT_TRUE(moved.is_open());
  EXPECT_EQ(moved.capacity(), kCapacity);
  EXPECT_EQ(moved.dimension(), kDim);

  auto ref = moved.get_node(0);
  auto read_vec = ref.vector();
  for (uint32_t i = 0; i < kDim; ++i) {
    EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
  }
}

TEST_F(DataFileTest, MoveConstructRebindsFlushCallback) {
  constexpr uint32_t kLargeCapacity = 256;
  auto path = make_path("move_ctor_flush_callback");
  BufferPool<uint32_t> small_bp(1, kDataBlockSize, 1);
  DataFile<float, uint32_t> df(&small_bp);
  df.create(path, kLargeCapacity, kDim, kMaxDeg);

  DataFile<float, uint32_t> moved(std::move(df));
  auto vec0 = make_vector(kDim, 21);
  auto vec1 = make_vector(kDim, 22);
  uint32_t second_block_node = moved.nodes_per_block();
  ASSERT_LT(second_block_node, kLargeCapacity);

  {
    auto ref = moved.get_node(0);
    ref.set_vector(std::span<const float>(vec0));
  }

  {
    auto ref = moved.get_node(second_block_node);
    ref.set_vector(std::span<const float>(vec1));
  }

  moved.close();
  small_bp.clear();

  DataFile<float, uint32_t> reopened(&small_bp);
  reopened.open(path, kLargeCapacity, kDim, kMaxDeg, false);

  auto ref = reopened.get_node(0);
  auto read_vec = ref.vector();
  ASSERT_EQ(read_vec.size(), kDim);
  for (uint32_t i = 0; i < kDim; ++i) {
    EXPECT_FLOAT_EQ(read_vec[i], vec0[i]);
  }
}

// -------------------------------------------------------------------------
// Different template parameter types
// -------------------------------------------------------------------------

TEST_F(DataFileTest, Int8DataType) {
  auto path = make_path("int8_dtype");
  DataFile<int8_t, uint32_t> df(bp_.get());
  df.create(path, kCapacity, 16, kMaxDeg);

  std::vector<int8_t> vec(16);
  for (int i = 0; i < 16; ++i) {
    vec[i] = static_cast<int8_t>(i - 8);
  }

  auto ref = df.get_node(0);
  ref.set_vector(std::span<const int8_t>(vec));

  auto read_vec = ref.vector();
  ASSERT_EQ(read_vec.size(), 16U);
  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(read_vec[i], vec[i]);
  }
}

}  // namespace alaya
