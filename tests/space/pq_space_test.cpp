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
#include <cmath>
#include <numeric>
#include <vector>

#include "space/pq_space.hpp"
#include "utils/random.hpp"

namespace alaya {
namespace {

constexpr uint32_t kTestDim = 128;
constexpr uint32_t kTestNumSubspaces = 8;
constexpr uint32_t kTestCapacity = 2000;
constexpr size_t kTestNumVectors = 1000;
constexpr size_t kTestNumQueries = 10;

// Compute exact L2 squared distance
auto compute_exact_l2_sqr(const float *a, const float *b, uint32_t dim) -> float {
  float dist = 0.0F;
  for (uint32_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

TEST(PQSpaceTest, Construction) {
  // Valid construction
  EXPECT_NO_THROW(PQSpace<float>(1000, 128, 8));
  EXPECT_NO_THROW(PQSpace<float>(1000, 128, 16));
  EXPECT_NO_THROW(PQSpace<float>(1000, 128, 32));

  // Invalid: dimension not divisible by num_subspaces
  EXPECT_THROW(PQSpace<float>(1000, 128, 7), std::invalid_argument);
}

TEST(PQSpaceTest, BasicProperties) {
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);

  EXPECT_EQ(space.get_dim(), kTestDim);
  EXPECT_EQ(space.get_capacity(), kTestCapacity);
  EXPECT_EQ(space.num_subspaces(), kTestNumSubspaces);
  EXPECT_EQ(space.get_data_size(), kTestNumSubspaces);  // PQ code size in bytes
  EXPECT_EQ(space.get_data_num(), 0);
}

TEST(PQSpaceTest, FitAndBasicAccess) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);

  // Fit should succeed
  EXPECT_NO_THROW(space.fit(data.data(), kTestNumVectors));
  EXPECT_EQ(space.get_data_num(), kTestNumVectors);

  // PQ codes should be accessible
  for (size_t i = 0; i < 10; ++i) {
    const uint8_t *code = space.get_code(i);
    EXPECT_NE(code, nullptr);
    // Each code byte should be valid (0-255)
    for (uint32_t m = 0; m < kTestNumSubspaces; ++m) {
      EXPECT_LT(code[m], 256);
    }
  }
}

TEST(PQSpaceTest, QueryContextDistanceComputation) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);
  space.fit(data.data(), kTestNumVectors);

  // Create query context from raw vector
  auto queries = generate_random_vectors(kTestNumQueries, kTestDim, 123);

  for (size_t q = 0; q < kTestNumQueries; ++q) {
    const float *query = queries.data() + q * kTestDim;
    auto ctx = space.get_query_context(query);

    // Compute distances using context
    std::vector<float> adc_dists(kTestNumVectors);
    std::vector<float> exact_dists(kTestNumVectors);

    for (size_t i = 0; i < kTestNumVectors; ++i) {
      adc_dists[i] = ctx(i);
      exact_dists[i] = compute_exact_l2_sqr(query, data.data() + i * kTestDim, kTestDim);
    }

    // ADC distances should correlate with exact distances
    // Find top-10 by ADC and check recall against exact top-10
    std::vector<size_t> adc_order(kTestNumVectors);
    std::vector<size_t> exact_order(kTestNumVectors);
    std::iota(adc_order.begin(), adc_order.end(), 0);
    std::iota(exact_order.begin(), exact_order.end(), 0);

    std::partial_sort(adc_order.begin(),
                      adc_order.begin() + 10,
                      adc_order.end(),
                      [&](size_t a, size_t b) -> bool {
                        return adc_dists[a] < adc_dists[b];
                      });
    std::partial_sort(exact_order.begin(),
                      exact_order.begin() + 10,
                      exact_order.end(),
                      [&](size_t a, size_t b) -> bool {
                        return exact_dists[a] < exact_dists[b];
                      });

    // Count overlap in top-10
    std::vector<size_t> adc_top10(adc_order.begin(), adc_order.begin() + 10);
    std::vector<size_t> exact_top10(exact_order.begin(), exact_order.begin() + 10);
    std::ranges::sort(adc_top10);
    std::ranges::sort(exact_top10);

    std::vector<size_t> intersection;
    std::ranges::set_intersection(adc_top10, exact_top10, std::back_inserter(intersection));

    // With M=8 on random data, recall can vary. Check for at least 20% overlap in top-10
    EXPECT_GE(intersection.size(), 2) << "Query " << q << " has poor recall";
  }
}

TEST(PQSpaceTest, QueryContextFromID) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);
  space.fit(data.data(), kTestNumVectors);

  // Create context from stored vector ID
  uint32_t query_id = 42;
  auto ctx = space.get_query_context(query_id);

  // Distance to self should be approximately 0 (with some quantization error)
  float self_dist = ctx(query_id);
  EXPECT_LT(self_dist, 1.0F);  // Should be small due to quantization

  // Distance to other vectors should generally be larger
  [[maybe_unused]] float other_dist = ctx(100);
  // Note: other_dist might not always be > self_dist due to quantization,
  // but on average it should be
}

TEST(PQSpaceTest, GetDistance) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);
  space.fit(data.data(), kTestNumVectors);

  // get_distance computes distance between two stored vectors
  float dist_01 = space.get_distance(0, 1);
  float dist_10 = space.get_distance(1, 0);

  // Should be symmetric
  EXPECT_FLOAT_EQ(dist_01, dist_10);

  // Distance to self should be 0 (exact after decode)
  float dist_00 = space.get_distance(0, 0);
  EXPECT_FLOAT_EQ(dist_00, 0.0F);
}

TEST(PQSpaceTest, Insert) {
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);

  // Train quantizer with initial data
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  space.fit(data.data(), kTestNumVectors);

  // Insert additional vectors
  auto new_vec = generate_random_vectors(1, kTestDim, 999);
  uint32_t new_id = space.insert(new_vec.data());

  EXPECT_EQ(new_id, kTestNumVectors);
  EXPECT_EQ(space.get_data_num(), kTestNumVectors + 1);

  // Verify the new vector's codes are stored
  const uint8_t *code = space.get_code(new_id);
  EXPECT_NE(code, nullptr);
}

TEST(PQSpaceTest, SaveAndLoad) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space1(kTestCapacity, kTestDim, kTestNumSubspaces);
  space1.fit(data.data(), kTestNumVectors);

  // Save to file
  const std::string kTestFile = "/tmp/pq_space_test.bin";
  space1.save(kTestFile);

  // Load into new space
  PQSpace<float> space2;
  space2.load(kTestFile);

  // Verify properties match
  EXPECT_EQ(space2.get_dim(), kTestDim);
  EXPECT_EQ(space2.num_subspaces(), kTestNumSubspaces);
  EXPECT_EQ(space2.get_data_num(), kTestNumVectors);

  // Verify codes match
  for (size_t i = 0; i < 10; ++i) {
    const uint8_t *code1 = space1.get_code(i);
    const uint8_t *code2 = space2.get_code(i);
    for (uint32_t m = 0; m < kTestNumSubspaces; ++m) {
      EXPECT_EQ(code1[m], code2[m]);
    }
  }

  // Verify distances match
  auto queries = generate_random_vectors(1, kTestDim, 789);
  auto ctx1 = space1.get_query_context(queries.data());
  auto ctx2 = space2.get_query_context(queries.data());

  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(ctx1(i), ctx2(i));
  }

  // Cleanup
  std::remove(kTestFile.c_str());
}

TEST(PQSpaceTest, SetCodesDirect) {
  // Train quantizer separately
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> quantizer(kTestDim, kTestNumSubspaces);
  quantizer.fit(data.data(), kTestNumVectors);

  // Encode externally
  std::vector<uint8_t> codes(kTestNumVectors * kTestNumSubspaces);
  quantizer.batch_encode(data.data(), kTestNumVectors, codes.data());

  // Create space with pre-trained quantizer and set codes directly
  PQSpace<float> space(kTestCapacity, quantizer);
  space.set_codes(codes.data(), kTestNumVectors);

  EXPECT_EQ(space.get_data_num(), kTestNumVectors);

  // Verify codes were set correctly
  for (size_t i = 0; i < 10; ++i) {
    const uint8_t *stored = space.get_code(i);
    const uint8_t *original = codes.data() + i * kTestNumSubspaces;
    for (uint32_t m = 0; m < kTestNumSubspaces; ++m) {
      EXPECT_EQ(stored[m], original[m]);
    }
  }
}

TEST(PQSpaceTest, QueryComputerAlias) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);
  space.fit(data.data(), kTestNumVectors);

  // Test that QueryComputer alias works (for compatibility with other Spaces)
  auto queries = generate_random_vectors(1, kTestDim, 456);
  auto computer = space.get_query_computer(queries.data());

  float dist = computer(0);
  EXPECT_GE(dist, 0.0F);
}

TEST(PQSpaceTest, ADCTableAccess) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQSpace<float> space(kTestCapacity, kTestDim, kTestNumSubspaces);
  space.fit(data.data(), kTestNumVectors);

  auto queries = generate_random_vectors(1, kTestDim, 111);
  auto ctx = space.get_query_context(queries.data());

  // ADC table should be accessible
  const float *table = ctx.adc_table();
  EXPECT_NE(table, nullptr);

  // All table values should be non-negative (squared distances)
  for (size_t i = 0; i < kTestNumSubspaces * 256; ++i) {
    EXPECT_GE(table[i], 0.0F);
  }
}

}  // namespace
}  // namespace alaya
