// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "space/raw_space.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include "utils/metric_type.hpp"
namespace alaya {

using IDType = uint32_t;
using DataType = float;
using DistanceType = float;

class RawSpaceTest : public ::testing::Test {
 protected:
  RawSpaceTest() {
    space_ = std::make_unique<RawSpace<DataType, DistanceType, IDType>>(100, 3, MetricType::L2);
  }

  std::unique_ptr<RawSpace<DataType, DistanceType, IDType>> space_;
};

TEST_F(RawSpaceTest, TestFit) {
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  space_->fit(data.data(), 3);

  ASSERT_EQ(space_->get_data_num(), 3);
  ASSERT_EQ(space_->get_dim(), 3);
}

TEST_F(RawSpaceTest, TestInsertDelete) {
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  IDType id1 = space_->insert(data1.data());
  IDType id2 = space_->insert(data2.data());

  ASSERT_EQ(space_->get_avl_data_num(), 2);

  space_->remove(id1);

  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 1);

  space_->remove(id2);
  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 0);
}

TEST_F(RawSpaceTest, TestDistance) {
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  space_->insert(data1.data());
  space_->insert(data2.data());

  float distance = space_->get_distance(0, 1);

  float expected_distance =
      (1.0 - 4.0) * (1.0 - 4.0) + (2.0 - 5.0) * (2.0 - 5.0) + (3.0 - 6.0) * (3.0 - 6.0);

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceUInt8) {
  std::vector<uint8_t> data1 = {183, 0, 0};
  std::vector<uint8_t> data2 = {107, 2, 3};

  RawSpace<uint8_t> space(100, 3, MetricType::L2);

  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);

  float expected_distance = 5789;

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceInnerProductMetric) {
  std::vector<float> data1 = {1.0F, 2.0F, 3.0F};
  std::vector<float> data2 = {4.0F, 5.0F, 6.0F};

  RawSpace<DataType, DistanceType, IDType> space(100, 3, MetricType::IP);
  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);
  float expected_distance = -((1.0F * 4.0F) + (2.0F * 5.0F) + (3.0F * 6.0F));

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceCosineMetricUsesInnerProductPath) {
  std::vector<float> data1 = {1.0F, 0.0F, 2.0F};
  std::vector<float> data2 = {0.5F, 3.0F, 4.0F};

  RawSpace<DataType, DistanceType, IDType> space(100, 3, MetricType::COS);
  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);
  float expected_distance = -((1.0F * 0.5F) + (0.0F * 3.0F) + (2.0F * 4.0F));

  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestFitRejectsNullDataAndCapacityOverflow) {
  std::vector<float> data = {1.0F, 2.0F, 3.0F};

  EXPECT_THROW(space_->fit(static_cast<const float *>(nullptr), 1), std::invalid_argument);
  EXPECT_THROW(space_->fit(data.data(), 101), std::length_error);
}

}  // namespace alaya
