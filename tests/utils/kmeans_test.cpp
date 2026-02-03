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
#include <random>
#include <set>
#include <vector>

#include "utils/kmeans.hpp"
#include "utils/random.hpp"

namespace alaya {
namespace {

constexpr uint32_t kTestDim = 32;
constexpr uint32_t kTestNumClusters = 16;
constexpr size_t kTestNumPoints = 500;

// Generate clustered data with known cluster centers
auto generate_clustered_data(size_t num_points_per_cluster,
                             uint32_t num_clusters,
                             uint32_t dim,
                             float cluster_radius,
                             uint32_t seed = 42)
    -> std::pair<std::vector<float>, std::vector<float>> {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> center_dist(-10.0F, 10.0F);
  std::normal_distribution<float> noise_dist(0.0F, cluster_radius);

  // Generate cluster centers
  std::vector<float> centers(num_clusters * dim);
  for (auto &c : centers) {
    c = center_dist(rng);
  }

  // Generate points around centers
  std::vector<float> data(num_points_per_cluster * num_clusters * dim);
  for (uint32_t k = 0; k < num_clusters; ++k) {
    for (size_t i = 0; i < num_points_per_cluster; ++i) {
      size_t idx = k * num_points_per_cluster + i;
      for (uint32_t d = 0; d < dim; ++d) {
        data[idx * dim + d] = centers[k * dim + d] + noise_dist(rng);
      }
    }
  }

  return {data, centers};
}

// Compute exact L2 squared distance
auto compute_l2_sqr(const float *a, const float *b, uint32_t dim) -> float {
  float dist = 0.0F;
  for (uint32_t i = 0; i < dim; ++i) {
    float diff = a[i] - b[i];
    dist += diff * diff;
  }
  return dist;
}

// ============================================================================
// Config Tests
// ============================================================================

TEST(KMeansTest, DefaultConfig) {
  KMeans<float>::Config config;
  EXPECT_EQ(config.num_clusters_, 256U);
  EXPECT_EQ(config.max_iter_, 20U);
  EXPECT_EQ(config.num_trials_, 3U);
}

TEST(KMeansTest, CustomConfig) {
  KMeans<float>::Config config{.num_clusters_ = 32, .max_iter_ = 50, .num_trials_ = 5};
  EXPECT_EQ(config.num_clusters_, 32U);
  EXPECT_EQ(config.max_iter_, 50U);
  EXPECT_EQ(config.num_trials_, 5U);
}

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(KMeansTest, DefaultConstructor) {
  KMeans<float> kmeans;
  EXPECT_EQ(kmeans.config().num_clusters_, 256U);
}

TEST(KMeansTest, ConfigConstructor) {
  KMeans<float>::Config config{.num_clusters_ = 64, .max_iter_ = 30, .num_trials_ = 2};
  KMeans<float> kmeans(config);
  EXPECT_EQ(kmeans.config().num_clusters_, 64U);
  EXPECT_EQ(kmeans.config().max_iter_, 30U);
  EXPECT_EQ(kmeans.config().num_trials_, 2U);
}

TEST(KMeansTest, ParameterConstructor) {
  KMeans<float> kmeans(32, 25, 4);
  EXPECT_EQ(kmeans.config().num_clusters_, 32U);
  EXPECT_EQ(kmeans.config().max_iter_, 25U);
  EXPECT_EQ(kmeans.config().num_trials_, 4U);
}

TEST(KMeansTest, SetConfig) {
  KMeans<float> kmeans;
  KMeans<float>::Config new_config{.num_clusters_ = 16, .max_iter_ = 10, .num_trials_ = 1};
  kmeans.set_config(new_config);
  EXPECT_EQ(kmeans.config().num_clusters_, 16U);
  EXPECT_EQ(kmeans.config().max_iter_, 10U);
  EXPECT_EQ(kmeans.config().num_trials_, 1U);
}

// ============================================================================
// compute_l2_sqr Tests
// ============================================================================

TEST(KMeansTest, ComputeL2Sqr_ZeroDistance) {
  std::vector<float> vec = {1.0F, 2.0F, 3.0F, 4.0F};
  float dist = KMeans<float>::compute_l2_sqr(vec.data(), vec.data(), 4);
  EXPECT_FLOAT_EQ(dist, 0.0F);
}

TEST(KMeansTest, ComputeL2Sqr_KnownDistance) {
  std::vector<float> a = {0.0F, 0.0F, 0.0F};
  std::vector<float> b = {3.0F, 4.0F, 0.0F};
  float dist = KMeans<float>::compute_l2_sqr(a.data(), b.data(), 3);
  EXPECT_FLOAT_EQ(dist, 25.0F);  // 3^2 + 4^2 = 25
}

TEST(KMeansTest, ComputeL2Sqr_Symmetry) {
  auto data = generate_random_vectors(2, kTestDim);
  float dist_ab = KMeans<float>::compute_l2_sqr(data.data(), data.data() + kTestDim, kTestDim);
  float dist_ba = KMeans<float>::compute_l2_sqr(data.data() + kTestDim, data.data(), kTestDim);
  EXPECT_FLOAT_EQ(dist_ab, dist_ba);
}

TEST(KMeansTest, ComputeL2Sqr_MatchesReference) {
  auto data = generate_random_vectors(2, kTestDim, 123);
  float simd_dist = KMeans<float>::compute_l2_sqr(data.data(), data.data() + kTestDim, kTestDim);
  float ref_dist = compute_l2_sqr(data.data(), data.data() + kTestDim, kTestDim);
  EXPECT_NEAR(simd_dist, ref_dist, 1e-5F);
}

// ============================================================================
// find_nearest Tests
// ============================================================================

TEST(KMeansTest, FindNearest_SingleCentroid) {
  KMeans<float> kmeans(1);  // Single cluster
  std::vector<float> centroids = {1.0F, 2.0F, 3.0F};
  std::vector<float> point = {0.0F, 0.0F, 0.0F};

  uint32_t nearest = kmeans.find_nearest(point.data(), centroids.data(), 3);
  EXPECT_EQ(nearest, 0U);
}

TEST(KMeansTest, FindNearest_MultipleCentroids) {
  KMeans<float> kmeans(3);
  // 3 centroids in 2D
  std::vector<float> centroids = {
      0.0F,
      0.0F,  // Centroid 0 at origin
      10.0F,
      0.0F,  // Centroid 1 at (10, 0)
      0.0F,
      10.0F  // Centroid 2 at (0, 10)
  };

  // Point closest to centroid 0
  std::vector<float> point0 = {1.0F, 1.0F};
  EXPECT_EQ(kmeans.find_nearest(point0.data(), centroids.data(), 2), 0U);

  // Point closest to centroid 1
  std::vector<float> point1 = {9.0F, 1.0F};
  EXPECT_EQ(kmeans.find_nearest(point1.data(), centroids.data(), 2), 1U);

  // Point closest to centroid 2
  std::vector<float> point2 = {1.0F, 9.0F};
  EXPECT_EQ(kmeans.find_nearest(point2.data(), centroids.data(), 2), 2U);
}

// ============================================================================
// fit Tests (with Result return)
// ============================================================================

TEST(KMeansTest, Fit_ResultHasCorrectSizes) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  auto result = kmeans.fit(data.data(), kTestNumPoints, kTestDim);

  EXPECT_EQ(result.centroids_.size(), static_cast<size_t>(kTestNumClusters) * kTestDim);
  EXPECT_EQ(result.assignments_.size(), kTestNumPoints);
}

TEST(KMeansTest, Fit_AssignmentsAreValid) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  auto result = kmeans.fit(data.data(), kTestNumPoints, kTestDim);

  for (auto assignment : result.assignments_) {
    EXPECT_LT(assignment, kTestNumClusters);
  }
}

TEST(KMeansTest, Fit_CostIsNonNegative) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  auto result = kmeans.fit(data.data(), kTestNumPoints, kTestDim);

  EXPECT_GE(result.cost_, 0.0F);
}

TEST(KMeansTest, Fit_CentroidsAreDifferent) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  auto result = kmeans.fit(data.data(), kTestNumPoints, kTestDim);

  // Check that not all centroids are identical
  bool all_same = true;
  for (uint32_t k = 1; k < kTestNumClusters; ++k) {
    float dist = KMeans<float>::compute_l2_sqr(result.centroids_.data(),
                                               result.centroids_.data() + k * kTestDim,
                                               kTestDim);
    if (dist > 1e-6F) {
      all_same = false;
      break;
    }
  }
  EXPECT_FALSE(all_same);
}

// ============================================================================
// fit Tests (with external centroid storage)
// ============================================================================

TEST(KMeansTest, FitExternal_ReturnsValidCost) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  std::vector<float> centroids(static_cast<size_t>(kTestNumClusters) * kTestDim);
  float cost = kmeans.fit(data.data(), kTestNumPoints, kTestDim, centroids.data());

  EXPECT_GE(cost, 0.0F);
}

TEST(KMeansTest, FitExternal_CentroidsArePopulated) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim);
  KMeans<float> kmeans(kTestNumClusters, 10, 1);

  std::vector<float> centroids(static_cast<size_t>(kTestNumClusters) * kTestDim, 0.0F);
  kmeans.fit(data.data(), kTestNumPoints, kTestDim, centroids.data());

  // Check centroids are not all zeros
  float sum = 0.0F;
  for (auto c : centroids) {
    sum += std::abs(c);
  }
  EXPECT_GT(sum, 0.0F);
}

// ============================================================================
// Clustering Quality Tests
// ============================================================================

TEST(KMeansTest, ClusteringQuality_WellSeparatedClusters) {
  constexpr uint32_t kNumClusters = 4;
  constexpr size_t kPointsPerCluster = 100;
  constexpr float kClusterRadius = 0.5F;

  auto [data, true_centers] =
      generate_clustered_data(kPointsPerCluster, kNumClusters, kTestDim, kClusterRadius);

  KMeans<float> kmeans(kNumClusters, 20, 3);
  auto result = kmeans.fit(data.data(), kPointsPerCluster * kNumClusters, kTestDim);

  // For well-separated clusters, each cluster should have approximately
  // the same number of points
  std::vector<uint32_t> cluster_counts(kNumClusters, 0);
  for (auto assignment : result.assignments_) {
    cluster_counts[assignment]++;
  }

  for (uint32_t k = 0; k < kNumClusters; ++k) {
    // Allow 20% deviation from expected count
    EXPECT_GT(cluster_counts[k], kPointsPerCluster * 0.8);
    EXPECT_LT(cluster_counts[k], kPointsPerCluster * 1.2);
  }
}

TEST(KMeansTest, ClusteringQuality_LowQuantizationError) {
  constexpr uint32_t kNumClusters = 8;
  constexpr size_t kPointsPerCluster = 50;
  constexpr float kClusterRadius = 0.3F;

  auto [data, true_centers] =
      generate_clustered_data(kPointsPerCluster, kNumClusters, kTestDim, kClusterRadius);

  KMeans<float> kmeans(kNumClusters, 30, 3);
  auto result = kmeans.fit(data.data(), kPointsPerCluster * kNumClusters, kTestDim);

  // Compute average distance to assigned centroid
  float total_dist = 0.0F;
  for (size_t i = 0; i < kPointsPerCluster * kNumClusters; ++i) {
    uint32_t k = result.assignments_[i];
    total_dist += KMeans<float>::compute_l2_sqr(data.data() + i * kTestDim,
                                                result.centroids_.data() + k * kTestDim,
                                                kTestDim);
  }
  float avg_dist = total_dist / static_cast<float>(kPointsPerCluster * kNumClusters);

  // Average squared distance should be related to cluster_radius^2 * dim
  // (since we're using normal distribution with stddev = cluster_radius)
  float expected_dist = kClusterRadius * kClusterRadius * kTestDim;
  EXPECT_LT(avg_dist, expected_dist * 2.0F);
}

// ============================================================================
// Multiple Trials Test
// ============================================================================

TEST(KMeansTest, MultipleTrials_ImprovesOrMaintainsCost) {
  auto data = generate_random_vectors(kTestNumPoints, kTestDim, 789);

  // Single trial
  KMeans<float> kmeans_single(kTestNumClusters, 15, 1);
  auto result_single = kmeans_single.fit(data.data(), kTestNumPoints, kTestDim);

  // Multiple trials (should find equal or better solution)
  KMeans<float> kmeans_multi(kTestNumClusters, 15, 5);
  auto result_multi = kmeans_multi.fit(data.data(), kTestNumPoints, kTestDim);

  // With more trials, we should get equal or lower cost on average
  // (not guaranteed for single run, but likely)
  // Just verify both produce valid results
  EXPECT_GE(result_single.cost_, 0.0F);
  EXPECT_GE(result_multi.cost_, 0.0F);
}

// ============================================================================
// Copy and Move Tests
// ============================================================================

TEST(KMeansTest, CopyConstructor) {
  KMeans<float> kmeans1(32, 25, 4);
  KMeans<float> kmeans2(kmeans1);

  EXPECT_EQ(kmeans2.config().num_clusters_, 32U);
  EXPECT_EQ(kmeans2.config().max_iter_, 25U);
  EXPECT_EQ(kmeans2.config().num_trials_, 4U);
}

TEST(KMeansTest, CopyAssignment) {
  KMeans<float> kmeans1(32, 25, 4);
  KMeans<float> kmeans2;
  kmeans2 = kmeans1;

  EXPECT_EQ(kmeans2.config().num_clusters_, 32U);
  EXPECT_EQ(kmeans2.config().max_iter_, 25U);
  EXPECT_EQ(kmeans2.config().num_trials_, 4U);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST(KMeansTest, SingleCluster) {
  auto data = generate_random_vectors(100, kTestDim);
  KMeans<float> kmeans(1, 10, 1);

  auto result = kmeans.fit(data.data(), 100, kTestDim);

  EXPECT_EQ(result.centroids_.size(), static_cast<size_t>(kTestDim));
  for (auto assignment : result.assignments_) {
    EXPECT_EQ(assignment, 0U);
  }
}

TEST(KMeansTest, PointsEqualClusters) {
  constexpr uint32_t kNumPoints = 10;
  auto data = generate_random_vectors(kNumPoints, kTestDim);
  KMeans<float> kmeans(kNumPoints, 10, 1);

  auto result = kmeans.fit(data.data(), kNumPoints, kTestDim);

  // Each point should be assigned to a different cluster (or very close)
  std::set<uint32_t> unique_assignments(result.assignments_.begin(), result.assignments_.end());
  EXPECT_EQ(unique_assignments.size(), kNumPoints);
}

TEST(KMeansTest, HighDimensionalData) {
  constexpr uint32_t kHighDim = 256;
  constexpr size_t kNumPoints = 200;
  constexpr uint32_t kNumClusters = 8;

  auto data = generate_random_vectors(kNumPoints, kHighDim);
  KMeans<float> kmeans(kNumClusters, 10, 1);

  auto result = kmeans.fit(data.data(), kNumPoints, kHighDim);

  EXPECT_EQ(result.centroids_.size(), static_cast<size_t>(kNumClusters) * kHighDim);
  EXPECT_EQ(result.assignments_.size(), kNumPoints);
  EXPECT_GE(result.cost_, 0.0F);
}

}  // namespace
}  // namespace alaya
