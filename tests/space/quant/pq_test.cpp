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

#include <cmath>
#include <fstream>
#include <numeric>
#include <vector>

#include "space/quant/pq.hpp"
#include "utils/random.hpp"

namespace alaya {
namespace {

constexpr uint32_t kTestDim = 128;
constexpr uint32_t kTestNumSubspaces = 8;
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

TEST(PQQuantizerTest, ConstructorValidation) {
  // Valid construction
  EXPECT_NO_THROW(PQQuantizer<float>(128, 8));
  EXPECT_NO_THROW(PQQuantizer<float>(128, 16));
  EXPECT_NO_THROW(PQQuantizer<float>(128, 32));

  // Invalid: dimension not divisible by num_subspaces
  EXPECT_THROW(PQQuantizer<float>(128, 7), std::invalid_argument);
  EXPECT_THROW(PQQuantizer<float>(100, 8), std::invalid_argument);
}

TEST(PQQuantizerTest, BasicProperties) {
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);

  EXPECT_EQ(pq.dim(), kTestDim);
  EXPECT_EQ(pq.num_subspaces(), kTestNumSubspaces);
  EXPECT_EQ(pq.subspace_dim(), kTestDim / kTestNumSubspaces);
  EXPECT_EQ(pq.code_size(), kTestNumSubspaces);
}

TEST(PQQuantizerTest, FitAndEncode) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);

  // Training should succeed
  EXPECT_NO_THROW(pq.fit(data.data(), kTestNumVectors));

  // Encode a vector
  std::vector<uint8_t> codes(pq.code_size());
  pq.encode(data.data(), codes.data());

  // Each code should be valid (0-255)
  for (auto code : codes) {
    EXPECT_LT(code, 256);
  }
}

TEST(PQQuantizerTest, EncodeDecodeReconstruction) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);
  pq.fit(data.data(), kTestNumVectors);

  // Encode and decode
  std::vector<uint8_t> codes(pq.code_size());
  std::vector<float> reconstructed(kTestDim);

  pq.encode(data.data(), codes.data());
  pq.decode(codes.data(), reconstructed.data());

  // Reconstruction error should be bounded
  float error = compute_exact_l2_sqr(data.data(), reconstructed.data(), kTestDim);
  float norm =
      compute_exact_l2_sqr(data.data(), std::vector<float>(kTestDim, 0.0F).data(), kTestDim);

  // Relative error should be reasonable (< 50% for random data with M=8)
  EXPECT_LT(error / norm, 0.5F);
}

TEST(PQQuantizerTest, ADCTableComputation) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);
  pq.fit(data.data(), kTestNumVectors);

  // Create query and compute ADC table
  auto queries = generate_random_vectors(1, kTestDim, 123);
  std::vector<float> adc_table(pq.num_subspaces() * 256);
  pq.compute_adc_table(queries.data(), adc_table.data());

  // All distances should be non-negative
  for (auto dist : adc_table) {
    EXPECT_GE(dist, 0.0F);
  }
}

TEST(PQQuantizerTest, ADCDistanceAccuracy) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);
  pq.fit(data.data(), kTestNumVectors);

  // Encode all vectors
  std::vector<uint8_t> all_codes(kTestNumVectors * pq.code_size());
  pq.batch_encode(data.data(), kTestNumVectors, all_codes.data());

  // Test with several queries
  auto queries = generate_random_vectors(kTestNumQueries, kTestDim, 456);

  for (size_t q = 0; q < kTestNumQueries; ++q) {
    const float *query = queries.data() + q * kTestDim;

    // Compute ADC table
    std::vector<float> adc_table(pq.num_subspaces() * 256);
    pq.compute_adc_table(query, adc_table.data());

    // Compare ADC distance vs exact distance for first 100 vectors
    float total_adc_dist = 0.0F;
    float total_exact_dist = 0.0F;

    for (size_t i = 0; i < 100; ++i) {
      float adc_dist =
          pq.compute_distance_with_table(adc_table.data(), all_codes.data() + i * pq.code_size());
      float exact_dist = compute_exact_l2_sqr(query, data.data() + i * kTestDim, kTestDim);

      total_adc_dist += adc_dist;
      total_exact_dist += exact_dist;

      // ADC distance should approximate exact distance
      // (reconstruction introduces quantization error)
    }

    // Average ADC distance should be correlated with exact distance
    // This is a sanity check - the ratio should be roughly stable
    float ratio = total_adc_dist / total_exact_dist;
    EXPECT_GT(ratio, 0.5F);
    EXPECT_LT(ratio, 2.0F);
  }
}

TEST(PQQuantizerTest, SaveAndLoad) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq1(kTestDim, kTestNumSubspaces);
  pq1.fit(data.data(), kTestNumVectors);

  // Encode a vector with original quantizer
  std::vector<uint8_t> codes1(pq1.code_size());
  pq1.encode(data.data(), codes1.data());

  // Save to file
  const std::string kTestFile = "/tmp/pq_test.bin";
  {
    std::ofstream writer(kTestFile, std::ios::binary);
    pq1.save(writer);
  }

  // Load into new quantizer
  PQQuantizer<float> pq2;
  {
    std::ifstream reader(kTestFile, std::ios::binary);
    pq2.load(reader);
  }

  // Verify properties match
  EXPECT_EQ(pq2.dim(), kTestDim);
  EXPECT_EQ(pq2.num_subspaces(), kTestNumSubspaces);
  EXPECT_EQ(pq2.subspace_dim(), kTestDim / kTestNumSubspaces);

  // Encode same vector with loaded quantizer
  std::vector<uint8_t> codes2(pq2.code_size());
  pq2.encode(data.data(), codes2.data());

  // Codes should match exactly
  EXPECT_EQ(codes1, codes2);

  // ADC distances should match
  auto queries = generate_random_vectors(1, kTestDim, 789);
  std::vector<float> adc1(pq1.num_subspaces() * 256);
  std::vector<float> adc2(pq2.num_subspaces() * 256);
  pq1.compute_adc_table(queries.data(), adc1.data());
  pq2.compute_adc_table(queries.data(), adc2.data());

  for (size_t i = 0; i < adc1.size(); ++i) {
    EXPECT_FLOAT_EQ(adc1[i], adc2[i]);
  }

  // Cleanup
  std::remove(kTestFile.c_str());
}

TEST(PQQuantizerTest, BatchEncode) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq(kTestDim, kTestNumSubspaces);
  pq.fit(data.data(), kTestNumVectors);

  // Batch encode
  std::vector<uint8_t> batch_codes(kTestNumVectors * pq.code_size());
  pq.batch_encode(data.data(), kTestNumVectors, batch_codes.data());

  // Single encode for verification
  for (size_t i = 0; i < 10; ++i) {
    std::vector<uint8_t> single_codes(pq.code_size());
    pq.encode(data.data() + i * kTestDim, single_codes.data());

    // Should match batch result
    for (uint32_t m = 0; m < pq.code_size(); ++m) {
      EXPECT_EQ(batch_codes[i * pq.code_size() + m], single_codes[m]);
    }
  }
}

TEST(PQQuantizerTest, CopyAndMove) {
  auto data = generate_random_vectors(kTestNumVectors, kTestDim);
  PQQuantizer<float> pq1(kTestDim, kTestNumSubspaces);
  pq1.fit(data.data(), kTestNumVectors);

  // Copy constructor
  PQQuantizer<float> pq2(pq1);
  EXPECT_EQ(pq2.dim(), pq1.dim());
  EXPECT_EQ(pq2.num_subspaces(), pq1.num_subspaces());

  // Verify codebook was copied
  std::vector<uint8_t> codes1(pq1.code_size());
  std::vector<uint8_t> codes2(pq2.code_size());
  pq1.encode(data.data(), codes1.data());
  pq2.encode(data.data(), codes2.data());
  EXPECT_EQ(codes1, codes2);

  // Move constructor
  PQQuantizer<float> pq3(std::move(pq2));
  EXPECT_EQ(pq3.dim(), kTestDim);
  std::vector<uint8_t> codes3(pq3.code_size());
  pq3.encode(data.data(), codes3.data());
  EXPECT_EQ(codes1, codes3);
}

}  // namespace
}  // namespace alaya
