// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "space/quant/rabitq_core.hpp"
#include "space/rabitq_space.hpp"

namespace alaya {

class RaBitQSpaceTest : public ::testing::Test {
 protected:
  using SpaceType = RaBitQSpace<float, float, uint32_t>;

  void SetUp() override {
    file_name_ = "test_rabitq_space.bin";
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  void TearDown() override {
    if (std::filesystem::exists(file_name_)) {
      std::filesystem::remove(file_name_);
    }
  }

  std::vector<float> make_test_data(uint32_t item_cnt) {
    std::vector<float> data(item_cnt * dim_);
    for (uint32_t i = 0; i < item_cnt; ++i) {
      for (size_t j = 0; j < dim_; ++j) {
        data[i * dim_ + j] = static_cast<float>(i * dim_ + j + 1);
      }
    }
    return data;
  }

  std::shared_ptr<SpaceType> space_;
  const size_t dim_ = 64;
  const uint32_t capacity_ = 10;
  std::string file_name_;
};

TEST_F(RaBitQSpaceTest, ConstructionAndFit) {
  const uint32_t item_cnt = 3;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);
  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  EXPECT_EQ(space_->get_data_num(), item_cnt);
  EXPECT_EQ(space_->get_dim(), dim_);
  EXPECT_EQ(space_->get_capacity(), capacity_);

  for (uint32_t i = 0; i < item_cnt; ++i) {
    const float *vec = space_->get_data_by_id(i);
    for (size_t j = 0; j < dim_; ++j) {
      EXPECT_FLOAT_EQ(vec[j], static_cast<float>(i * dim_ + j + 1));
    }
  }
}

TEST_F(RaBitQSpaceTest, DistanceComputation) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);

  std::vector<float> data(2 * dim_, 0.0f);
  std::fill(data.begin() + dim_, data.end(), 1.0f);

  space_->fit(data.data(), item_cnt);

  float dist = space_->get_distance(0, 1);
  EXPECT_FLOAT_EQ(dist, static_cast<float>(dim_));
}

TEST_F(RaBitQSpaceTest, SaveAndLoad) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);
  space_->set_ep(1);

  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  std::string_view filename = file_name_;
  space_->save(filename);

  auto loaded_space = std::make_shared<SpaceType>();
  loaded_space->load(filename);

  EXPECT_EQ(loaded_space->get_dim(), dim_);
  EXPECT_EQ(loaded_space->get_data_num(), item_cnt);
  EXPECT_EQ(loaded_space->get_capacity(), capacity_);
  EXPECT_EQ(loaded_space->get_ep(), 1u);

  for (uint32_t i = 0; i < item_cnt; ++i) {
    const float *orig = space_->get_data_by_id(i);
    const float *load = loaded_space->get_data_by_id(i);
    for (size_t j = 0; j < dim_; ++j) {
      EXPECT_FLOAT_EQ(orig[j], load[j]);
    }
  }

  EXPECT_FLOAT_EQ(space_->get_distance(0, 1), loaded_space->get_distance(0, 1));

  std::filesystem::remove(filename);
}

TEST_F(RaBitQSpaceTest, L2MetricIsAccepted) {
  EXPECT_NO_THROW(space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2));
}

TEST_F(RaBitQSpaceTest, ItemCntOverflow) {
  const uint32_t item_cnt = 11;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);

  std::vector<float> data(item_cnt * dim_, 0.0f);

  EXPECT_THROW(space_->fit(data.data(), item_cnt), std::length_error);
}

TEST_F(RaBitQSpaceTest, SaveNonExistentPath) {
  const uint32_t item_cnt = 2;
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);
  space_->set_ep(1);

  auto data = make_test_data(item_cnt);
  space_->fit(data.data(), item_cnt);

  std::string_view invalid_path = "/nonexistent_dir/invalid_file.bin";

  EXPECT_THROW(space_->save(invalid_path), std::runtime_error);
}

TEST_F(RaBitQSpaceTest, LoadNonExistentPath) {
  space_ = std::make_shared<SpaceType>(capacity_, dim_, core::Metric::l2);

  std::string_view invalid_path = "/nonexistent_dir/invalid_file.bin";

  EXPECT_THROW(space_->load(invalid_path), std::runtime_error);
}

// ---------------------------------------------------------------------------
// U4-preflight IP audit (amendment v2): the non-unit-norm "1 == ||o||=1 bug"
// hypothesis was falsified by codex's math review -- see the derivation
// comment on RaBitQCore::memory_factors' inner-product branch
// (space/quant/rabitq_core.hpp). These two tests are what that review asked
// for in its place: a formula-shape lock (exact, via a q=c construction with
// zero estimator approximation slack) and a characterization/regression lock
// (recall must not collapse for non-unit-norm data, but some decline vs. the
// unit-norm baseline is expected and not a bug).
// ---------------------------------------------------------------------------

TEST(RaBitQCoreTest, InnerProductBranchLocksToOneMinusDot) {
  // Querying with q_rot = centroid makes <half_signed, q_rot> collapse to
  // exactly centroid_dot_half_signed (computed identically on both sides --
  // no fastscan/LUT quantization is involved here, memory_factors is called
  // directly), which makes the K-estimator's own approximation of
  // <residual, q> algebraically *exact* rather than approximate. That in turn
  // makes est(q=c, o) collapse to *exactly* 1 - <c, o>, with zero
  // approximation slack, so this test pins the formula's shape: a "||o||=1"
  // bug (which would need the literal 1 replaced by dot(data,data)) and the
  // actual "candidate-independent constant" shape are exactly distinguishable
  // here whenever ||data|| and ||centroid|| are not 1 (asserted below).
  constexpr size_t kDim = 64;
  std::mt19937 rng(0xC0FFEE01U);
  std::uniform_real_distribution<float> val_dist(-4.0F, 4.0F);

  std::vector<float> data(kDim);
  std::vector<float> centroid(kDim);
  std::generate(data.begin(), data.end(), [&] {
    return val_dist(rng);
  });
  std::generate(centroid.begin(), centroid.end(), [&] {
    return val_dist(rng);
  });

  // Deliberately non-unit norm on both sides -- exactly the case a "||o||=1"
  // bug would get wrong instead of a candidate-independent constant.
  ASSERT_GT(std::abs(dot_product<float>(data.data(), data.data(), kDim) - 1.0F), 0.5F);
  ASSERT_GT(std::abs(dot_product<float>(centroid.data(), centroid.data(), kDim) - 1.0F), 0.5F);

  std::vector<int> sign_bits(kDim);
  const auto factors = RaBitQCore::memory_factors<float>(data.data(),
                                                         centroid.data(),
                                                         kDim,
                                                         sign_bits.data(),
                                                         core::Metric::inner_product);

  std::vector<float> half_signed(kDim);
  for (size_t i = 0; i < kDim; ++i) {
    half_signed[i] = static_cast<float>(sign_bits[i]) - 0.5F;
  }
  const float centroid_dot_half_signed =
      dot_product<float>(centroid.data(), half_signed.data(), kDim);
  const float centroid_dot_centroid = dot_product<float>(centroid.data(), centroid.data(), kDim);

  // est(q = centroid, o = data), assembled exactly like
  // RaBitQSpace::QueryComputer::batch_est_dist + load_centroid does:
  // f_rescale * <half_signed, q_rot> + f_add + g_add, with g_add = -<q,c>.
  const float est = factors.base + (factors.signed_query_scale * centroid_dot_half_signed) -
                    centroid_dot_centroid;
  const float expected = 1.0F - dot_product<float>(centroid.data(), data.data(), kDim);

  EXPECT_NEAR(est, expected, 1.0e-3F * std::max(1.0F, std::abs(expected)));
}

namespace {

auto random_unit_vector(std::mt19937 &rng, size_t dim) -> std::vector<float> {
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> v(dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  const float norm = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0F));
  for (auto &x : v) {
    x /= norm;
  }
  return v;
}

// Builds a RaBitQSpace(inner_product) with one centroid + kDegreeBound (32)
// neighbors -- scaled per-point by `point_scale` -- quantizes the neighbors
// against the centroid via the real production path (update_nei -> fastscan
// LUT), and measures top-`topk` recall of the fastscan estimate against an
// exact -dot oracle, averaged over several queries scaled by `query_scale`.
auto measure_ip_neighbor_recall(uint64_t seed,
                                size_t dim,
                                const std::function<float(size_t)> &point_scale,
                                const std::function<float(size_t)> &query_scale,
                                size_t topk) -> double {
  using SpaceType = RaBitQSpace<float, float, uint32_t>;
  constexpr uint32_t kDegreeBound = static_cast<uint32_t>(SpaceType::kDegreeBound);  // 32
  constexpr uint32_t kNumPoints = kDegreeBound + 1;  // centroid (id 0) + neighbors (id 1..32)
  constexpr uint32_t kCentroidId = 0;

  std::mt19937 rng(seed);
  std::vector<float> data(static_cast<size_t>(kNumPoints) * dim);
  for (uint32_t i = 0; i < kNumPoints; ++i) {
    const auto unit = random_unit_vector(rng, dim);
    const float scale = point_scale(i);
    for (size_t d = 0; d < dim; ++d) {
      data[(i * dim) + d] = unit[d] * scale;
    }
  }

  auto space = std::make_shared<SpaceType>(kNumPoints, dim, core::Metric::inner_product);
  space->fit(data.data(), kNumPoints);

  std::vector<Neighbor<uint32_t, float>> neighbors;
  neighbors.reserve(kDegreeBound);
  for (uint32_t i = 1; i < kNumPoints; ++i) {
    neighbors.emplace_back(i, 0.0F);
  }
  space->update_nei(kCentroidId, neighbors);

  constexpr int kNumQueries = 30;
  std::normal_distribution<float> noise(0.0F, 0.05F);
  double recall_sum = 0.0;
  for (int q = 0; q < kNumQueries; ++q) {
    // Queries resemble an existing neighbor direction with a small
    // perturbation (typical of real ANN workloads), then get their own scale
    // applied -- which need not match the neighbor's own scale.
    const uint32_t base_id = 1 + (static_cast<uint32_t>(q) % kDegreeBound);
    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) {
      query[d] = data[(base_id * dim) + d] + noise(rng);
    }
    const float norm =
        std::sqrt(std::inner_product(query.begin(), query.end(), query.begin(), 0.0F));
    const float scale = query_scale(static_cast<size_t>(q));
    for (auto &x : query) {
      x = (x / norm) * scale;
    }

    auto qc = space->get_query_computer(query.data());
    qc.load_centroid(kCentroidId);
    const float *est = qc.est_data();

    std::vector<std::pair<float, uint32_t>> exact;
    exact.reserve(kDegreeBound);
    for (uint32_t i = 0; i < kDegreeBound; ++i) {
      const uint32_t id = neighbors[i].id_;
      float dot = 0.0F;
      for (size_t d = 0; d < dim; ++d) {
        dot += query[d] * data[(id * dim) + d];
      }
      exact.emplace_back(-dot, i);
    }
    std::vector<std::pair<float, uint32_t>> estimated;
    estimated.reserve(kDegreeBound);
    for (uint32_t i = 0; i < kDegreeBound; ++i) {
      estimated.emplace_back(est[i], i);
    }
    std::sort(exact.begin(), exact.end());
    std::sort(estimated.begin(), estimated.end());

    std::set<uint32_t> exact_top;
    for (size_t i = 0; i < topk; ++i) {
      exact_top.insert(exact[i].second);
    }
    size_t hit = 0;
    for (size_t i = 0; i < topk; ++i) {
      if (exact_top.count(estimated[i].second) > 0) {
        ++hit;
      }
    }
    recall_sum += static_cast<double>(hit) / static_cast<double>(topk);
  }
  return recall_sum / kNumQueries;
}

}  // namespace

TEST(RaBitQSpaceIpNormTest, NonUnitNormRecallDoesNotCollapse) {
  // NOTE on what this measures: a single fastscan batch-of-32 estimate
  // against its exact oracle, with no graph-search-level re-ranking or
  // multi-hop aggregation -- i.e. the raw local accuracy of the primitive
  // RaBitQCore::memory_factors is trying to protect, not full QG end-to-end
  // recall (which is much higher because beam search explores and re-ranks
  // many such local batches). Averaged over several independent
  // centroid/neighbor-set trials and queries per trial for stability.
  constexpr size_t kDim = 64;
  constexpr size_t kTopK = 10;
  constexpr int kTrials = 8;

  const auto average_recall = [&](const std::function<float(size_t)> &point_scale,
                                  const std::function<float(size_t)> &query_scale) {
    double sum = 0.0;
    for (int trial = 0; trial < kTrials; ++trial) {
      sum += measure_ip_neighbor_recall(0x1234ABCDU + static_cast<uint64_t>(trial),
                                        kDim,
                                        point_scale,
                                        query_scale,
                                        kTopK);
    }
    return sum / kTrials;
  };

  const auto unit_recall = average_recall(
      [](size_t /*i*/) {
        return 1.0F;
      },
      [](size_t /*i*/) {
        return 1.0F;
      });

  // Per-point/per-query scale in [0.5, 3.0]; deliberately non-unit and
  // varies across candidates, unlike every pre-existing rabitq test which
  // only ever fed unit vectors (collection_qg_seal_test's
  // make_float_dataset:85-88 always normalizes).
  const auto scale_fn = [](size_t i) {
    return 0.5F + (2.5F * static_cast<float>(i % 5) / 4.0F);
  };
  const auto nonunit_recall = average_recall(scale_fn, scale_fn);

  std::cout << "measured_rabitq_space_ip_unit_recall_at_10=" << unit_recall << '\n';
  std::cout << "measured_rabitq_space_ip_nonunit_recall_at_10=" << nonunit_recall << '\n';

  // Sanity: the unit-norm baseline should be well above random-choice recall
  // (kTopK/kDegreeBound =~ 0.31 for 10-of-32) -- this is a single-hop batch
  // estimate, not full QG recall, so it is not expected to be anywhere near
  // the >=0.8 end-to-end thresholds collection_qg_seal_test uses.
  EXPECT_GE(unit_recall, 0.45);

  // Characterization + regression lock (U4-preflight IP audit, amendment v2):
  // non-unit-norm inner_product recall must NOT collapse relative to the
  // unit-norm baseline measured with the identical methodology above. RaBitQ's
  // quantization error grows with ||o-c||*||q-c||, so some decline under
  // non-unit scaling would not be surprising -- but a collapse toward
  // random-choice recall would mean a *different*, unknown bug, not the
  // "literal 1" formula (amendment v2's math review proved that is
  // order-preserving for any norm, see RaBitQCoreTest.
  // InnerProductBranchLocksToOneMinusDot above). If this ever fires: stop and
  // report, do not "fix" it by touching memory_factors' inner-product branch.
  EXPECT_GE(nonunit_recall, unit_recall - 0.20);
  EXPECT_GE(nonunit_recall, 0.35);
}

}  // namespace alaya
