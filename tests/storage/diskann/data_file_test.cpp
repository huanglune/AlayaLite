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
#include <functional>
#include <numeric>
#include <span>
#include <string>
#include <vector>

#include "storage/diskann/data_file.hpp"
#include "storage/buffer/buffer_pool.hpp"

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

  // Range-based iteration
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

  void SetUp() override {
    bp_ = std::make_unique<BPType>(kBpCapacity, kDataBlockSize, 1);
  }

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

  // row_size = sizeof(uint32_t) + kMaxDeg * sizeof(uint32_t) + kDim * sizeof(float)
  //          = 4 + 16 + 32 = 52
  size_t expected_row = sizeof(uint32_t) + kMaxDeg * sizeof(uint32_t) + kDim * sizeof(float);
  EXPECT_EQ(df.row_size(), expected_row);

  // nodes_per_block = 4096 / 52 = 78
  uint32_t expected_npb = static_cast<uint32_t>(kDataBlockSize / expected_row);
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

  // Second close should be a no-op
  df.close();
  EXPECT_FALSE(df.is_open());
}

// -------------------------------------------------------------------------
// Row size too large throws
// -------------------------------------------------------------------------

TEST_F(DataFileTest, RowSizeExceedsBlockSizeThrows) {
  auto path = make_path("row_too_large");
  DataFile<float, uint32_t> df(bp_.get());

  // With dim=1024 and max_degree=1024:
  // row_size = 4 + 1024*4 + 1024*4 = 8196 > 4096
  EXPECT_THROW(df.create(path, 10, 1024, 1024), std::invalid_argument);
}

// -------------------------------------------------------------------------
// Write and read vector data
// -------------------------------------------------------------------------

TEST_F(DataFileTest, WriteAndReadVector) {
  auto path = make_path("write_read_vec");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 42);

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    ASSERT_EQ(read_vec.size(), kDim);
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }
  });
}

TEST_F(DataFileTest, WriteAndReadNeighbors) {
  auto path = make_path("write_read_nbrs");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto nbrs = make_neighbors(3, 10);  // [10, 11, 12]

  df.modify_node(0, [&](auto &editor) {
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), 3U);
    EXPECT_EQ(nlist.neighbor_ids()[0], 10U);
    EXPECT_EQ(nlist.neighbor_ids()[1], 11U);
    EXPECT_EQ(nlist.neighbor_ids()[2], 12U);
  });
}

TEST_F(DataFileTest, WriteVectorAndNeighborsTogether) {
  auto path = make_path("write_both");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 7);
  auto nbrs = make_neighbors(kMaxDeg, 100);

  df.modify_node(5, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  df.inspect_node(5, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }

    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), kMaxDeg);
    for (uint32_t i = 0; i < kMaxDeg; ++i) {
      EXPECT_EQ(nlist.neighbor_ids()[i], 100 + i);
    }
  });
}

TEST_F(DataFileTest, MultipleNodesInSameBlock) {
  auto path = make_path("multi_nodes");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  for (uint32_t id = 0; id < 10; ++id) {
    auto vec = make_vector(kDim, id);
    auto nbrs = make_neighbors(2, id * 10);

    df.modify_node(id, [&](auto &editor) {
      editor.set_vector(std::span<const float>(vec));
      editor.set_neighbors(std::span<const uint32_t>(nbrs));
    });
  }

  for (uint32_t id = 0; id < 10; ++id) {
    auto expected_vec = make_vector(kDim, id);

    df.inspect_node(id, [&](const auto &viewer) {
      auto read_vec = viewer.vector_view();
      for (uint32_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(read_vec[i], expected_vec[i]) << "Node " << id << " dim " << i;
      }

      const auto &nlist = viewer.neighbors_view();
      EXPECT_EQ(nlist.size(), 2U) << "Node " << id;
      EXPECT_EQ(nlist.neighbor_ids()[0], id * 10) << "Node " << id;
      EXPECT_EQ(nlist.neighbor_ids()[1], id * 10 + 1) << "Node " << id;
    });
  }
}

// -------------------------------------------------------------------------
// Cross-block access
// -------------------------------------------------------------------------

TEST_F(DataFileTest, CrossBlockAccess) {
  auto path = make_path("cross_block");

  // Use larger parameters so rows are bigger, fewer nodes per block
  constexpr uint32_t kLargeDim = 64;
  constexpr uint32_t kLargeDeg = 32;
  constexpr uint32_t kLargeCap = 64;

  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kLargeCap, kLargeDim, kLargeDeg);

  uint32_t npb = df.nodes_per_block();
  ASSERT_GT(npb, 0U);
  ASSERT_LT(npb, kLargeCap);  // Ensure we span at least 2 blocks

  auto vec0 = make_vector(kLargeDim, 0);
  auto vec1 = make_vector(kLargeDim, 1);

  // Write to first node of block 0 and first node of block 1
  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec0));
  });

  df.modify_node(npb, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec1));
  });

  // Read back both and verify
  df.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kLargeDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec0[i]);
    }
  });

  df.inspect_node(npb, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kLargeDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec1[i]);
    }
  });
}

// -------------------------------------------------------------------------
// Persistence: create -> write -> close -> open -> read
// -------------------------------------------------------------------------

TEST_F(DataFileTest, Persistence) {
  auto path = make_path("persist");
  auto vec = make_vector(kDim, 99);
  auto nbrs = make_neighbors(3, 50);

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);

    df.modify_node(0, [&](auto &editor) {
      editor.set_vector(std::span<const float>(vec));
      editor.set_neighbors(std::span<const uint32_t>(nbrs));
    });

    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    df.inspect_node(0, [&](const auto &viewer) {
      auto read_vec = viewer.vector_view();
      for (uint32_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
      }

      const auto &nlist = viewer.neighbors_view();
      EXPECT_EQ(nlist.size(), 3U);
      for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(nlist.neighbor_ids()[i], 50 + i);
      }
    });

    df.close();
  }
}

TEST_F(DataFileTest, PersistenceMultipleNodes) {
  auto path = make_path("persist_multi");
  constexpr uint32_t kNumNodes = 20;

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);

    for (uint32_t id = 0; id < kNumNodes; ++id) {
      auto vec = make_vector(kDim, id);
      auto nbrs = make_neighbors(kMaxDeg, id * 100);

      df.modify_node(id, [&](auto &editor) {
        editor.set_vector(std::span<const float>(vec));
        editor.set_neighbors(std::span<const uint32_t>(nbrs));
      });
    }

    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    for (uint32_t id = 0; id < kNumNodes; ++id) {
      auto expected_vec = make_vector(kDim, id);

      df.inspect_node(id, [&](const auto &viewer) {
        auto read_vec = viewer.vector_view();
        for (uint32_t i = 0; i < kDim; ++i) {
          EXPECT_FLOAT_EQ(read_vec[i], expected_vec[i]) << "Node " << id;
        }

        const auto &nlist = viewer.neighbors_view();
        EXPECT_EQ(nlist.size(), kMaxDeg) << "Node " << id;
        for (uint32_t i = 0; i < kMaxDeg; ++i) {
          EXPECT_EQ(nlist.neighbor_ids()[i], id * 100 + i) << "Node " << id;
        }
      });
    }

    df.close();
  }
}

// -------------------------------------------------------------------------
// Editor advanced interface
// -------------------------------------------------------------------------

TEST_F(DataFileTest, EditorMutableVector) {
  auto path = make_path("editor_mut_vec");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 1);

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
  });

  // Modify in-place via mutable_vector
  df.modify_node(0, [&](auto &editor) {
    auto mut_vec = editor.mutable_vector();
    for (auto &val : mut_vec) {
      val *= 2.0F;
    }
  });

  df.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i] * 2.0F);
    }
  });
}

TEST_F(DataFileTest, EditorMutableNeighbors) {
  auto path = make_path("editor_mut_nbrs");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto nbrs = make_neighbors(kMaxDeg, 0);

  df.modify_node(0, [&](auto &editor) {
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  // Modify in-place: sort in reverse
  df.modify_node(0, [&](auto &editor) {
    auto &mut_nbrs = editor.mutable_neighbors();
    std::sort(mut_nbrs.begin(), mut_nbrs.end(), std::greater<uint32_t>());
  });

  df.inspect_node(0, [&](const auto &viewer) {
    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), kMaxDeg);
    // Should be in reverse order: 3, 2, 1, 0
    for (uint32_t i = 0; i < kMaxDeg; ++i) {
      EXPECT_EQ(nlist.neighbor_ids()[i], kMaxDeg - 1 - i);
    }
  });
}

TEST_F(DataFileTest, EditorAsViewer) {
  auto path = make_path("editor_as_viewer");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 5);

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));

    // Use as_viewer to verify within the same modify call
    auto viewer = editor.as_viewer();
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }
  });
}

// -------------------------------------------------------------------------
// Viewer copy methods
// -------------------------------------------------------------------------

TEST_F(DataFileTest, ViewerCopyVectorTo) {
  auto path = make_path("copy_vec_to");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 3);
  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    std::vector<float> buf(kDim);
    viewer.copy_vector_to(std::span<float>(buf));
    EXPECT_EQ(buf, vec);
  });
}

TEST_F(DataFileTest, ViewerCopyVectorToDimMismatchThrows) {
  auto path = make_path("copy_vec_dim_err");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 0);
  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    std::vector<float> bad_buf(kDim + 1);
    EXPECT_THROW(viewer.copy_vector_to(std::span<float>(bad_buf)), std::invalid_argument);
  });
}

TEST_F(DataFileTest, ViewerCopyNeighborsTo) {
  auto path = make_path("copy_nbrs_to");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto nbrs = make_neighbors(3, 10);
  df.modify_node(0, [&](auto &editor) {
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    std::vector<uint32_t> buf(kMaxDeg, 0);
    viewer.copy_neighbors_to(std::span<uint32_t>(buf));
    EXPECT_EQ(buf[0], 10U);
    EXPECT_EQ(buf[1], 11U);
    EXPECT_EQ(buf[2], 12U);
  });
}

// -------------------------------------------------------------------------
// Convenience methods: write_vector / read_vector
// -------------------------------------------------------------------------

TEST_F(DataFileTest, WriteVectorConvenience) {
  auto path = make_path("write_vec_conv");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 77);
  df.write_vector(0, std::span<const float>(vec));

  std::vector<float> buf(kDim);
  df.read_vector(0, std::span<float>(buf));
  EXPECT_EQ(buf, vec);
}

TEST_F(DataFileTest, WriteNeighborsConvenience) {
  auto path = make_path("write_nbrs_conv");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto nbrs = make_neighbors(3, 20);
  df.write_neighbors(0, std::span<const uint32_t>(nbrs));

  // Verify via inspect_node (read_neighbors has a known bug)
  df.inspect_node(0, [&](const auto &viewer) {
    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), 3U);
    EXPECT_EQ(nlist.neighbor_ids()[0], 20U);
    EXPECT_EQ(nlist.neighbor_ids()[1], 21U);
    EXPECT_EQ(nlist.neighbor_ids()[2], 22U);
  });
}

// -------------------------------------------------------------------------
// Error cases
// -------------------------------------------------------------------------

TEST_F(DataFileTest, InspectNodeNotOpenThrows) {
  auto path = make_path("inspect_not_open");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);
  df.close();

  EXPECT_THROW(df.inspect_node(0, [](const auto & /*unused*/) {}), std::runtime_error);
}

TEST_F(DataFileTest, ModifyNodeNotOpenThrows) {
  auto path = make_path("modify_not_open");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);
  df.close();

  EXPECT_THROW(df.modify_node(0, [](auto & /*unused*/) {}), std::runtime_error);
}

TEST_F(DataFileTest, InspectNodeOutOfRangeThrows) {
  auto path = make_path("inspect_oor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(df.inspect_node(kCapacity, [](const auto & /*unused*/) {}), std::out_of_range);
  EXPECT_THROW(df.inspect_node(kCapacity + 100, [](const auto & /*unused*/) {}), std::out_of_range);
}

TEST_F(DataFileTest, ModifyNodeOutOfRangeThrows) {
  auto path = make_path("modify_oor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(df.modify_node(kCapacity, [](auto & /*unused*/) {}), std::out_of_range);
}

TEST_F(DataFileTest, ModifyNodeReadOnlyThrows) {
  auto path = make_path("modify_readonly");

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.create(path, kCapacity, kDim, kMaxDeg);
    df.close();
  }

  bp_->clear();

  {
    DataFile<float, uint32_t> df(bp_.get());
    df.open(path, kCapacity, kDim, kMaxDeg, false);

    EXPECT_THROW(df.modify_node(0, [](auto & /*unused*/) {}), std::runtime_error);
    df.close();
  }
}

TEST_F(DataFileTest, SetNeighborsTooManyThrows) {
  auto path = make_path("too_many_nbrs");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  std::vector<uint32_t> too_many(kMaxDeg + 1);
  std::iota(too_many.begin(), too_many.end(), 0);

  EXPECT_THROW(
      df.modify_node(0,
                     [&](auto &editor) {
                       editor.set_neighbors(std::span<const uint32_t>(too_many));
                     }),
      std::length_error);
}

TEST_F(DataFileTest, SetVectorDimMismatchThrows) {
  auto path = make_path("vec_dim_err");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  std::vector<float> wrong_dim(kDim + 1, 1.0F);

  EXPECT_THROW(
      df.modify_node(0,
                     [&](auto &editor) {
                       editor.set_vector(std::span<const float>(wrong_dim));
                     }),
      std::invalid_argument);
}

// -------------------------------------------------------------------------
// Overwrite node data
// -------------------------------------------------------------------------

TEST_F(DataFileTest, OverwriteNodeData) {
  auto path = make_path("overwrite");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec1 = make_vector(kDim, 1);
  auto vec2 = make_vector(kDim, 2);

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec1));
  });

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec2));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec2[i]);
    }
  });
}

// -------------------------------------------------------------------------
// Neighbors: empty list and fill remaining with invalid (-1)
// -------------------------------------------------------------------------

TEST_F(DataFileTest, EmptyNeighbors) {
  auto path = make_path("empty_nbrs");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  std::vector<uint32_t> empty_nbrs;

  df.modify_node(0, [&](auto &editor) {
    editor.set_neighbors(std::span<const uint32_t>(empty_nbrs));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), 0U);
    EXPECT_TRUE(nlist.empty());
  });
}

TEST_F(DataFileTest, NeighborsFillWithInvalid) {
  auto path = make_path("nbrs_fill_invalid");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto nbrs = make_neighbors(2, 10);  // 2 neighbors, max is 4

  df.modify_node(0, [&](auto &editor) {
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    const auto &nlist = viewer.neighbors_view();
    EXPECT_EQ(nlist.size(), 2U);

    auto *ids = nlist.neighbor_ids();
    EXPECT_EQ(ids[0], 10U);
    EXPECT_EQ(ids[1], 11U);
    // Slots beyond num_neighbors_ should be filled with -1 (0xFFFFFFFF)
    EXPECT_EQ(ids[2], static_cast<uint32_t>(-1));
    EXPECT_EQ(ids[3], static_cast<uint32_t>(-1));
  });
}

// -------------------------------------------------------------------------
// Move semantics
// -------------------------------------------------------------------------

TEST_F(DataFileTest, MoveConstruct) {
  auto path = make_path("move_ctor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  auto vec = make_vector(kDim, 42);
  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const float>(vec));
  });

  DataFile<float, uint32_t> moved(std::move(df));
  EXPECT_TRUE(moved.is_open());
  EXPECT_EQ(moved.capacity(), kCapacity);
  EXPECT_EQ(moved.dimension(), kDim);

  // Verify data accessible through moved object
  moved.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    for (uint32_t i = 0; i < kDim; ++i) {
      EXPECT_FLOAT_EQ(read_vec[i], vec[i]);
    }
  });
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

  df.modify_node(0, [&](auto &editor) {
    editor.set_vector(std::span<const int8_t>(vec));
  });

  df.inspect_node(0, [&](const auto &viewer) {
    auto read_vec = viewer.vector_view();
    ASSERT_EQ(read_vec.size(), 16U);
    for (int i = 0; i < 16; ++i) {
      EXPECT_EQ(read_vec[i], vec[i]);
    }
  });
}

// -------------------------------------------------------------------------
// Batch modify tests
// -------------------------------------------------------------------------

TEST_F(DataFileTest, BatchModifySingleBlock) {
  auto path = make_path("batch_single");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  // Batch write 10 nodes (all in one block since npb=78)
  df.batch_modify(0, 10, [&](uint32_t id, auto &editor) {
    auto vec = make_vector(kDim, id);
    auto nbrs = make_neighbors(2, id * 10);
    editor.set_vector(std::span<const float>(vec));
    editor.set_neighbors(std::span<const uint32_t>(nbrs));
  });

  // Verify each node
  for (uint32_t id = 0; id < 10; ++id) {
    auto expected_vec = make_vector(kDim, id);

    df.inspect_node(id, [&](const auto &viewer) {
      auto read_vec = viewer.vector_view();
      for (uint32_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(read_vec[i], expected_vec[i]) << "Node " << id;
      }

      const auto &nlist = viewer.neighbors_view();
      EXPECT_EQ(nlist.size(), 2U) << "Node " << id;
      EXPECT_EQ(nlist.neighbor_ids()[0], id * 10) << "Node " << id;
      EXPECT_EQ(nlist.neighbor_ids()[1], id * 10 + 1) << "Node " << id;
    });
  }
}

TEST_F(DataFileTest, BatchModifyCrossBlock) {
  auto path = make_path("batch_cross");

  // Use larger parameters to force multiple blocks
  constexpr uint32_t kLargeDim = 64;
  constexpr uint32_t kLargeDeg = 32;
  constexpr uint32_t kLargeCap = 64;

  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kLargeCap, kLargeDim, kLargeDeg);

  uint32_t npb = df.nodes_per_block();
  ASSERT_GT(npb, 0U);
  ASSERT_LT(npb, kLargeCap);

  // Batch write all nodes across multiple blocks
  df.batch_modify(0, kLargeCap, [&](uint32_t id, auto &editor) {
    auto vec = make_vector(kLargeDim, id);
    editor.set_vector(std::span<const float>(vec));
  });

  // Verify a sample of nodes from different blocks
  for (uint32_t id : {uint32_t{0}, npb - 1, npb, npb + 1, kLargeCap - 1}) {
    auto expected_vec = make_vector(kLargeDim, id);

    df.inspect_node(id, [&](const auto &viewer) {
      auto read_vec = viewer.vector_view();
      for (uint32_t i = 0; i < kLargeDim; ++i) {
        EXPECT_FLOAT_EQ(read_vec[i], expected_vec[i]) << "Node " << id;
      }
    });
  }
}

TEST_F(DataFileTest, BatchModifyPersistence) {
  auto path = make_path("batch_persist");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  df.batch_modify(0, 5, [&](uint32_t id, auto &editor) {
    auto vec = make_vector(kDim, id);
    editor.set_vector(std::span<const float>(vec));
  });

  df.close();
  bp_->clear();

  // Reopen and verify
  df.open(path, kCapacity, kDim, kMaxDeg, false);

  for (uint32_t id = 0; id < 5; ++id) {
    auto expected_vec = make_vector(kDim, id);

    df.inspect_node(id, [&](const auto &viewer) {
      auto read_vec = viewer.vector_view();
      for (uint32_t i = 0; i < kDim; ++i) {
        EXPECT_FLOAT_EQ(read_vec[i], expected_vec[i]) << "Node " << id;
      }
    });
  }

  df.close();
}

TEST_F(DataFileTest, BatchModifyReadOnlyThrows) {
  auto path = make_path("batch_readonly");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);
  df.close();
  bp_->clear();

  df.open(path, kCapacity, kDim, kMaxDeg, false);

  EXPECT_THROW(
      df.batch_modify(0, 1, [](uint32_t /*unused*/, auto & /*unused*/) {}), std::runtime_error);

  df.close();
}

TEST_F(DataFileTest, BatchModifyOutOfRangeThrows) {
  auto path = make_path("batch_oor");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  EXPECT_THROW(
      df.batch_modify(0, kCapacity + 1, [](uint32_t /*unused*/, auto & /*unused*/) {}),
      std::out_of_range);
}

TEST_F(DataFileTest, BatchModifyEmptyRange) {
  auto path = make_path("batch_empty");
  DataFile<float, uint32_t> df(bp_.get());
  df.create(path, kCapacity, kDim, kMaxDeg);

  // start == end -> no-op
  df.batch_modify(5, 5, [](uint32_t /*unused*/, auto & /*unused*/) {
    FAIL() << "Should not be called for empty range";
  });
}

}  // namespace alaya
