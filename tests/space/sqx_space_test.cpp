// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

#include "space/sq4_space.hpp"
#include "space/sq8_space.hpp"

namespace alaya {

template <typename SpaceType>
class SQxSpaceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    space_ = std::make_shared<SpaceType>(kCapacity, kDim, core::Metric::l2);
    file_name_ = std::string("test_sqx_space_") +
                 std::to_string(space_->get_data_size()) + ".bin";
    std::filesystem::remove(file_name_);
  }

  void TearDown() override { std::filesystem::remove(file_name_); }

  static constexpr uint32_t kDim = 4;
  static constexpr uint32_t kCapacity = 10;

  std::shared_ptr<SpaceType> space_;
  std::string file_name_;
};

using SQxTypes = ::testing::Types<SQ4Space<>, SQ8Space<>>;
TYPED_TEST_SUITE(SQxSpaceTest, SQxTypes);

TYPED_TEST(SQxSpaceTest, Initialization) {
  EXPECT_EQ(this->space_->get_dim(), this->kDim);
  EXPECT_EQ(this->space_->get_data_num(), 0);
  EXPECT_GT(this->space_->get_data_size(), 0);
}

TYPED_TEST(SQxSpaceTest, FitData) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  EXPECT_EQ(this->space_->get_data_num(), 2);
}

TYPED_TEST(SQxSpaceTest, InsertAndRemove) {
  float vec[] = {1, 2, 3, 4};
  auto id = this->space_->insert(vec);
  EXPECT_GE(id, 0);
  EXPECT_EQ(this->space_->get_data_num(), 1);
  this->space_->remove(id);
  EXPECT_EQ(this->space_->get_data_num(), 1);
}

TYPED_TEST(SQxSpaceTest, GetDistance) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  float dist = this->space_->get_distance(0, 1);
  EXPECT_FLOAT_EQ(dist, 64);
}

TYPED_TEST(SQxSpaceTest, SaveAndLoad) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  std::string_view path = this->file_name_;
  this->space_->save(path);

  TypeParam loaded;
  loaded.load(path);
  EXPECT_EQ(loaded.get_data_num(), 2);
}

TYPED_TEST(SQxSpaceTest, QueryComputerWithQuery) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  float query[] = {1, 2, 3, 4};
  auto qc = this->space_->get_query_computer(query);
  EXPECT_GE(qc(1), 64);
}

TYPED_TEST(SQxSpaceTest, QueryComputerWithId) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  auto qc = this->space_->get_query_computer(static_cast<uint32_t>(0));
  EXPECT_GE(qc(1), 64);
}

TYPED_TEST(SQxSpaceTest, Prefetch) {
  float data[] = {1, 2, 3, 4, 5, 6, 7, 8};
  this->space_->fit(data, 2);
  this->space_->prefetch_by_id(1);
  this->space_->prefetch_by_address(data);
}

TYPED_TEST(SQxSpaceTest, HandleInvalidInsert) {
  float vec[] = {1, 2, 3, 4};
  for (uint32_t i = 0; i < this->kCapacity; ++i) {
    this->space_->insert(vec);
  }
  auto id = this->space_->insert(vec);
  EXPECT_EQ(id, static_cast<uint32_t>(-1));
}

TYPED_TEST(SQxSpaceTest, HandleFileErrors) {
  TypeParam space;
  EXPECT_THROW(space.load(std::string_view("non_existent_file.bin")), std::runtime_error);
}

}  // namespace alaya
