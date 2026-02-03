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

#include "space/raw_space.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <sys/types.h>

#include "utils/metric_type.hpp"
namespace alaya {

using IDType = uint32_t;
using DataType = float;
using DistanceType = float;

class RawSpaceTest : public ::testing::Test {
 protected:
  RawSpaceTest() {
    // First, we initialize the RawSpace object with a capacity of 100 and a dimensionality of 3.
    space_ = std::make_unique<RawSpace<DataType, DistanceType, IDType>>(100, 3, MetricType::L2);
  }

  std::unique_ptr<RawSpace<DataType, DistanceType, IDType>> space_;
};

// Let's test the fit method by inserting some data points.
TEST_F(RawSpaceTest, TestFit) {
  // Here's some test data: 3 points in 3D space.
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  // We call the fit method to load this data into our RawSpace.
  space_->fit(data.data(), 3);

  // Now, let's check if the number of data points has been correctly updated.
  // We expect it to be 3 since we inserted 3 points.
  ASSERT_EQ(space_->get_data_num(), 3);

  // The dimensionality of each data point should be 3 as well.
  ASSERT_EQ(space_->get_dim(), 3);
}

// Now let's test the insertion and deletion of data points.
TEST_F(RawSpaceTest, TestInsertDelete) {
  // Prepare some data points to insert.
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  // Insert the first data point and store the ID returned by the insert function.
  IDType id1 = space_->insert(data1.data());

  // Insert the second data point.
  IDType id2 = space_->insert(data2.data());

  // After inserting, we should have 2 data points in total.
  ASSERT_EQ(space_->get_avl_data_num(), 2);

  // Now, let's delete the first data point by its ID.
  space_->remove(id1);

  // After deletion, there should only be 1 data point left.
  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 1);

  space_->remove(id2);
  // After deletion, there should only be 0 data point left.
  ASSERT_EQ(space_->get_data_num(), 2);
  ASSERT_EQ(space_->get_avl_data_num(), 0);
}

// Let's test if the distance calculation is working as expected.
TEST_F(RawSpaceTest, TestDistance) {
  // Prepare two data points.
  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  // Insert both data points into the RawSpace.
  space_->insert(data1.data());
  space_->insert(data2.data());

  // Now we calculate the L2 distance between the two points.
  float distance = space_->get_distance(0, 1);

  // We know the L2 distance between these two points should be:
  float expected_distance =
      (1.0 - 4.0) * (1.0 - 4.0) + (2.0 - 5.0) * (2.0 - 5.0) + (3.0 - 6.0) * (3.0 - 6.0);

  // Check if the calculated distance matches the expected distance.
  ASSERT_FLOAT_EQ(distance, expected_distance);
}

TEST_F(RawSpaceTest, TestDistanceUInt8) {
  // Prepare two data points.
  std::vector<uint8_t> data1 = {183, 0, 0};
  std::vector<uint8_t> data2 = {107, 2, 3};

  RawSpace<uint8_t> space(100, 3, MetricType::L2);

  // Insert both data points into the RawSpace.
  space.insert(data1.data());
  space.insert(data2.data());

  // Now we calculate the L2 distance between the two points.
  float distance = space.get_distance(0, 1);

  // We know the L2 distance between these two points should be:
  float expected_distance = 5789;

  // Check if the calculated distance matches the expected distance.
  ASSERT_FLOAT_EQ(distance, expected_distance);
}

// Test IP (Inner Product) metric
TEST(RawSpaceMetricTest, TestIPMetric) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::IP);

  std::vector<float> data1 = {1.0, 2.0, 3.0};
  std::vector<float> data2 = {4.0, 5.0, 6.0};

  space.insert(data1.data());
  space.insert(data2.data());

  float distance = space.get_distance(0, 1);
  // IP distance is negative of dot product (for max-heap behavior)
  float expected = -(1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0);  // -32
  ASSERT_FLOAT_EQ(distance, expected);
}

// Test COS (Cosine) metric - vectors are normalized before storage
TEST(RawSpaceMetricTest, TestCOSMetric) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::COS);

  std::vector<float> data1 = {1.0, 0.0, 0.0};
  std::vector<float> data2 = {1.0, 1.0, 0.0};

  space.insert(data1.data());
  space.insert(data2.data());

  // Vectors are normalized, so distance should be negative cosine similarity
  float distance = space.get_distance(0, 1);
  // After normalization: data1 = [1, 0, 0], data2 = [1/sqrt(2), 1/sqrt(2), 0]
  // cos(theta) = 1/sqrt(2) ≈ 0.707
  ASSERT_NEAR(distance, -std::sqrt(0.5F), 0.01F);
}

// Test save and load functionality
TEST(RawSpaceIOTest, TestSaveLoad) {
  const std::string kTestFile = "/tmp/raw_space_test.bin";

  // Create and populate space
  RawSpace<float, float, uint32_t> space1(100, 3, MetricType::L2);
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  space1.fit(data.data(), 3);

  // Save to file
  space1.save(kTestFile);

  // Load into new space
  RawSpace<float, float, uint32_t> space2;
  space2.load(kTestFile);

  // Verify properties match
  ASSERT_EQ(space2.get_dim(), space1.get_dim());
  ASSERT_EQ(space2.get_data_num(), space1.get_data_num());
  ASSERT_EQ(space2.get_capacity(), space1.get_capacity());

  // Verify data matches
  for (uint32_t i = 0; i < 3; ++i) {
    float *data1 = space1.get_data_by_id(i);
    float *data2 = space2.get_data_by_id(i);
    for (uint32_t j = 0; j < 3; ++j) {
      ASSERT_FLOAT_EQ(data1[j], data2[j]);
    }
  }

  // Cleanup
  std::remove(kTestFile.c_str());
}

// Test QueryComputer with query pointer
TEST(RawSpaceQueryComputerTest, TestQueryComputerFromPointer) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::L2);
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  space.fit(data.data(), 2);

  std::vector<float> query = {1.0, 2.0, 3.0};
  auto computer = space.get_query_computer(query.data());

  // Distance to first vector (same as query) should be 0
  float dist0 = computer(0);
  ASSERT_FLOAT_EQ(dist0, 0.0F);

  // Distance to second vector
  float dist1 = computer(1);
  float expected = (1.0F - 4.0F) * (1.0F - 4.0F) + (2.0F - 5.0F) * (2.0F - 5.0F) +
                   (3.0F - 6.0F) * (3.0F - 6.0F);
  ASSERT_FLOAT_EQ(dist1, expected);
}

// Test QueryComputer with ID
TEST(RawSpaceQueryComputerTest, TestQueryComputerFromID) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::L2);
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  space.fit(data.data(), 2);

  // Use first vector as query
  auto computer = space.get_query_computer(static_cast<uint32_t>(0));

  // Distance to self should be 0
  float dist0 = computer(0);
  ASSERT_FLOAT_EQ(dist0, 0.0F);

  // Distance to second vector should be same as regular get_distance
  float dist1 = computer(1);
  float expected = space.get_distance(0, 1);
  ASSERT_FLOAT_EQ(dist1, expected);
}

// Test QueryComputer with deleted/invalid node
TEST(RawSpaceQueryComputerTest, TestQueryComputerInvalidNode) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::L2);
  std::vector<float> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  space.fit(data.data(), 2);

  // Delete first node
  space.remove(0);

  std::vector<float> query = {1.0, 2.0, 3.0};
  auto computer = space.get_query_computer(query.data());

  // Distance to deleted node should return max float
  float dist = computer(0);
  ASSERT_EQ(dist, std::numeric_limits<float>::max());
}

// Test prefetch functions (just verify they don't crash)
TEST(RawSpacePrefetchTest, TestPrefetch) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::L2);
  std::vector<float> data = {1.0, 2.0, 3.0};
  space.fit(data.data(), 1);

  // These should not crash
  EXPECT_NO_THROW(space.prefetch_by_id(0));
  EXPECT_NO_THROW(space.prefetch_by_address(space.get_data_by_id(0)));
}

// Test COS metric with fit
TEST(RawSpaceMetricTest, TestCOSMetricFit) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::COS);

  std::vector<float> data = {3.0, 4.0, 0.0, 0.0, 0.0, 5.0};  // Two vectors
  space.fit(data.data(), 2);

  // After normalization, first vector should be [0.6, 0.8, 0]
  // Second vector should be [0, 0, 1]
  float *vec0 = space.get_data_by_id(0);
  ASSERT_NEAR(vec0[0], 0.6F, 0.001F);
  ASSERT_NEAR(vec0[1], 0.8F, 0.001F);
  ASSERT_NEAR(vec0[2], 0.0F, 0.001F);

  float *vec1 = space.get_data_by_id(1);
  ASSERT_NEAR(vec1[0], 0.0F, 0.001F);
  ASSERT_NEAR(vec1[1], 0.0F, 0.001F);
  ASSERT_NEAR(vec1[2], 1.0F, 0.001F);
}

// Test get_dist_func returns valid function
TEST(RawSpaceGettersTest, TestGetDistFunc) {
  RawSpace<float, float, uint32_t> space(100, 3, MetricType::L2);

  auto dist_func = space.get_dist_func();
  ASSERT_NE(dist_func, nullptr);

  std::vector<float> a = {1.0, 2.0, 3.0};
  std::vector<float> b = {4.0, 5.0, 6.0};

  float dist = dist_func(a.data(), b.data(), 3);
  float expected = 27.0F;  // (1-4)^2 + (2-5)^2 + (3-6)^2
  ASSERT_FLOAT_EQ(dist, expected);
}

// Test various getter methods
TEST(RawSpaceGettersTest, TestGetters) {
  RawSpace<float, float, uint32_t> space(100, 4, MetricType::L2);

  ASSERT_EQ(space.get_capacity(), 100);
  ASSERT_EQ(space.get_dim(), 4);
  ASSERT_EQ(space.get_data_size(), 4 * sizeof(float));
  ASSERT_EQ(space.get_data_num(), 0);
  ASSERT_EQ(space.get_avl_data_num(), 0);
}

}  // namespace alaya
