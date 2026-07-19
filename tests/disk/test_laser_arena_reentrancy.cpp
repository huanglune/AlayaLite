// SPDX-FileCopyrightText: 2026 AlayaDB.AI
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "index/disk/laser_segment_importer.hpp"
#include "index/disk/segment_factory.hpp"
#include "index/disk/unified_laser_segment_searcher.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"
#include "platform/detect.hpp"

namespace alaya::disk {
namespace {

#if defined(__SANITIZE_THREAD__)
constexpr bool kRunningTsan = true;
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
constexpr bool kRunningTsan = true;
  #else
constexpr bool kRunningTsan = false;
  #endif
#else
constexpr bool kRunningTsan = false;
#endif

constexpr std::uint32_t kDegree = 32;
constexpr std::uint32_t kTopK = 10;
constexpr int kConcurrentThreads = 8;
constexpr std::size_t kPrimaryCount = kRunningTsan ? 384 : 2048;
constexpr std::size_t kSecondaryCount = kRunningTsan ? 256 : 768;

auto make_data(std::size_t count, std::uint32_t dim, std::uint32_t seed) -> std::vector<float> {
  std::mt19937 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);
  std::vector<float> data(count * dim);
  for (auto &value : data) {
    value = distribution(generator);
  }
  return data;
}

void write_fbin(const std::filesystem::path &path,
                const std::vector<float> &data,
                std::int32_t count,
                std::int32_t dim) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create LASER reentrancy fbin: " + path.string());
  }
  output.write(reinterpret_cast<const char *>(&count), sizeof(count));
  output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  output.write(reinterpret_cast<const char *>(data.data()),
               static_cast<std::streamsize>(data.size() * sizeof(float)));
  if (!output) {
    throw std::runtime_error("short LASER reentrancy fbin write: " + path.string());
  }
}

// The graph is built in the original coordinate system. An identity PCA
// sidecar exercises pca_query_scratch_ without changing those coordinates.
void write_identity_pca(const std::filesystem::path &path, std::uint64_t dim) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create LASER reentrancy PCA sidecar: " + path.string());
  }
  output.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
  const std::vector<float> mean(dim, 0.0F);
  output.write(reinterpret_cast<const char *>(mean.data()),
               static_cast<std::streamsize>(mean.size() * sizeof(float)));
  std::vector<float> row(dim, 0.0F);
  for (std::uint64_t index = 0; index < dim; ++index) {
    row[index] = 1.0F;
    output.write(reinterpret_cast<const char *>(row.data()),
                 static_cast<std::streamsize>(row.size() * sizeof(float)));
    row[index] = 0.0F;
  }
  if (!output) {
    throw std::runtime_error("short LASER reentrancy PCA write: " + path.string());
  }
}

struct ArenaSegmentFixture {
  std::filesystem::path root;
  std::filesystem::path segment_directory;
  std::vector<float> data;
  std::vector<std::uint64_t> labels;
  std::uint32_t dim{};

  ArenaSegmentFixture() = default;
  ArenaSegmentFixture(const ArenaSegmentFixture &) = delete;
  auto operator=(const ArenaSegmentFixture &) -> ArenaSegmentFixture & = delete;

  ~ArenaSegmentFixture() {
    std::error_code error;
    std::filesystem::remove_all(root, error);
  }

  static auto build(std::string tag,
                    std::size_t count,
                    std::uint32_t dimension,
                    std::uint32_t seed,
                    std::uint64_t label_base) -> std::unique_ptr<ArenaSegmentFixture> {
    auto fixture = std::make_unique<ArenaSegmentFixture>();
    fixture->root = std::filesystem::temp_directory_path() /
                    ("laser_arena_reentrancy_" + std::to_string(platform::get_pid()) + "_" + tag);
    std::error_code error;
    std::filesystem::remove_all(fixture->root, error);
    std::filesystem::create_directories(fixture->root / "raw");
    std::filesystem::create_directories(fixture->root / "segments");
    fixture->segment_directory = fixture->root / "segments/seg_00000001";
    fixture->dim = dimension;
    fixture->data = make_data(count, dimension, seed);
    fixture->labels.resize(count);
    for (std::size_t pid = 0; pid < count; ++pid) {
      fixture->labels[pid] = label_base + pid;
    }

    const auto raw_prefix = fixture->root / "raw/dsqg_seg_00000001";
    write_fbin(raw_prefix.string() + "_pca_base.fbin",
               fixture->data,
               static_cast<std::int32_t>(count),
               static_cast<std::int32_t>(dimension));

    vamana::VamanaBuildParams vamana_params;
    vamana_params.R = kDegree;
    vamana_params.L = 64;
    vamana_params.alpha = 1.2F;
    vamana_params.num_threads = kRunningTsan ? 1 : 4;
    vamana::VamanaBuilder vamana_builder(
        fixture->data.data(), count, dimension, vamana_params);
    vamana_builder.build();
    const auto vamana_path = raw_prefix.string() + "_vamana.index";
    vamana::save_graph(
        vamana_builder.graph(), vamana_path, kDegree, vamana_builder.medoid());

    laser::QuantizedGraph graph(count, kDegree, dimension, dimension, /*rotator_seed=*/7);
    laser::QGBuilder qg_builder(
        graph, /*ef_build=*/64, /*num_threads=*/kRunningTsan ? 1 : 4);
    qg_builder.build(vamana_path.c_str(), raw_prefix.c_str());
    write_identity_pca(raw_prefix.string() + "_pca.bin", dimension);

    LaserSegmentImportParams import_params;
    import_params.R = kDegree;
    LaserSegmentImporter importer(dimension, core::Metric::l2, import_params);
    (void)importer.import_from(fixture->root / "raw",
                               fixture->labels.data(),
                               fixture->labels.size(),
                               fixture->segment_directory);
    return fixture;
  }
};

auto make_options(std::uint32_t ef, std::uint32_t beam, bool return_distances)
    -> DiskSearchOptions {
  DiskSearchOptions options;
  options.top_k = kTopK;
  options.ef = ef;
  options.beam_width = beam;
  options.return_distances = return_distances;
  return options;
}

auto same_hits(const std::vector<DiskSearchHit> &expected,
               const std::vector<DiskSearchHit> &actual,
               bool return_distances) -> bool {
  if (expected.size() != actual.size()) {
    return false;
  }
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index].label != actual[index].label) {
      return false;
    }
    const auto actual_bits = std::bit_cast<std::uint32_t>(actual[index].distance);
    const bool exponent_is_ones = (actual_bits & 0x7F800000U) == 0x7F800000U;
    if (return_distances) {
      if (exponent_is_ones ||
          std::bit_cast<std::uint32_t>(expected[index].distance) !=
              actual_bits) {
        return false;
      }
    } else if (!exponent_is_ones || (actual_bits & 0x007FFFFFU) == 0) {
      return false;
    }
  }
  return true;
}

template <typename Function>
auto run_workers(int thread_count, Function function, std::chrono::nanoseconds *elapsed = nullptr)
    -> int {
  std::atomic<int> ready{};
  std::atomic<bool> go{};
  std::atomic<int> failures{};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<std::size_t>(thread_count));
  for (int thread = 0; thread < thread_count; ++thread) {
    workers.emplace_back([&, thread] {
      ready.fetch_add(1, std::memory_order_release);
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        if (!function(thread)) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  while (ready.load(std::memory_order_acquire) != thread_count) {
    std::this_thread::yield();
  }
  const auto start = std::chrono::steady_clock::now();
  go.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  if (elapsed != nullptr) {
    *elapsed = std::chrono::steady_clock::now() - start;
  }
  return failures.load(std::memory_order_relaxed);
}

auto scaling_thread_count() -> int {
  const char *value = std::getenv("ALAYA_ARENA_REENTRANCY_THREADS");
  if (value == nullptr) {
    return kConcurrentThreads;
  }
  const int parsed = std::atoi(value);
  return std::max(1, parsed);
}

class LaserArenaReentrancyTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!engine_supported_v1(DiskIndexType::Laser)) {
      return;
    }
    if (kRunningTsan) {
      omp_set_num_threads(1);
    }
    primary_ = ArenaSegmentFixture::build(
        "primary", kPrimaryCount, /*dimension=*/128, /*seed=*/701, /*label_base=*/100'000);
    secondary_ = ArenaSegmentFixture::build(
        "secondary", kSecondaryCount, /*dimension=*/64, /*seed=*/907, /*label_base=*/200'000);
  }

  static void TearDownTestSuite() {
    secondary_.reset();
    primary_.reset();
  }

  [[nodiscard]] static auto fixture_available() -> bool {
    return primary_ != nullptr && secondary_ != nullptr;
  }

  static std::unique_ptr<ArenaSegmentFixture> primary_;
  static std::unique_ptr<ArenaSegmentFixture> secondary_;
};

std::unique_ptr<ArenaSegmentFixture> LaserArenaReentrancyTest::primary_;
std::unique_ptr<ArenaSegmentFixture> LaserArenaReentrancyTest::secondary_;

TEST_F(LaserArenaReentrancyTest, ConcurrentSearchMatchesSerialWithDistanceContract) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER is unavailable";
  }
  UnifiedLaserSegmentSearcher searcher(primary_->segment_directory,
                                       laser::ResidencyMode::kResidentArena);
  ASSERT_EQ(searcher.set_params_call_count(), 0U);

  constexpr std::size_t kQueries = 16;
  const std::array options{
      make_options(/*ef=*/96, /*beam=*/4, /*return_distances=*/false),
      make_options(/*ef=*/96, /*beam=*/4, /*return_distances=*/true),
  };
  std::array<std::vector<std::vector<DiskSearchHit>>, 2> baselines;
  for (std::size_t mode = 0; mode < options.size(); ++mode) {
    baselines[mode].reserve(kQueries);
    for (std::size_t query = 0; query < kQueries; ++query) {
      baselines[mode].push_back(searcher.search(
          primary_->data.data() + query * primary_->dim, options[mode]));
      ASSERT_TRUE(same_hits(baselines[mode].back(),
                            baselines[mode].back(),
                            options[mode].return_distances));
    }
  }

  const int thread_count = scaling_thread_count();
  const int iterations = kRunningTsan ? 16 : 5000;
  std::chrono::nanoseconds elapsed{};
  const int failures = run_workers(
      thread_count,
      [&](int thread) {
        for (int iteration = 0; iteration < iterations; ++iteration) {
          const std::size_t mode = static_cast<std::size_t>(thread + iteration) % options.size();
          const std::size_t query =
              static_cast<std::size_t>(thread * 11 + iteration * 7) % kQueries;
          const auto actual = searcher.search(
              primary_->data.data() + query * primary_->dim, options[mode]);
          if (!same_hits(baselines[mode][query], actual, options[mode].return_distances)) {
            return false;
          }
        }
        return true;
      },
      &elapsed);
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(searcher.set_params_call_count(), 0U);

  const double seconds = std::chrono::duration<double>(elapsed).count();
  const double queries_per_second = static_cast<double>(thread_count * iterations) / seconds;
  std::cout << std::defaultfloat << std::setprecision(9)
            << "arena_reentrancy_smoke,threads=" << thread_count
            << ",queries=" << thread_count * iterations << ",seconds=" << seconds
            << ",qps=" << queries_per_second << '\n';
}

TEST_F(LaserArenaReentrancyTest, InterleavedPerCallEffortDoesNotBleed) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER is unavailable";
  }
  UnifiedLaserSegmentSearcher searcher(primary_->segment_directory,
                                       laser::ResidencyMode::kResidentArena);
  const std::array options{
      make_options(/*ef=*/10, /*beam=*/2, /*return_distances=*/true),
      make_options(/*ef=*/160, /*beam=*/16, /*return_distances=*/true),
  };
  constexpr std::size_t kQueries = 32;
  std::array<std::vector<std::vector<DiskSearchHit>>, 2> baselines;
  for (std::size_t mode = 0; mode < options.size(); ++mode) {
    baselines[mode].reserve(kQueries);
    for (std::size_t query = 0; query < kQueries; ++query) {
      baselines[mode].push_back(searcher.search(
          primary_->data.data() + query * primary_->dim, options[mode]));
    }
  }
  bool effort_is_observable = false;
  for (std::size_t query = 0; query < kQueries; ++query) {
    if (!same_hits(baselines[0][query], baselines[1][query], /*return_distances=*/true)) {
      effort_is_observable = true;
      break;
    }
  }
  ASSERT_TRUE(effort_is_observable)
      << "the deterministic fixture must distinguish the two per-call ef values";

  const int iterations = kRunningTsan ? 12 : 250;
  const int failures = run_workers(kConcurrentThreads, [&](int thread) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const std::size_t mode = static_cast<std::size_t>(thread + iteration) % options.size();
      const std::size_t query =
          static_cast<std::size_t>(thread * 13 + iteration * 5) % kQueries;
      const auto actual = searcher.search(
          primary_->data.data() + query * primary_->dim, options[mode]);
      if (!same_hits(baselines[mode][query], actual, /*return_distances=*/true)) {
        return false;
      }
    }
    return true;
  });
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(searcher.set_params_call_count(), 0U);
}

TEST_F(LaserArenaReentrancyTest, ThreadLocalScratchRebindsAcrossDifferentDimensions) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER is unavailable";
  }
  UnifiedLaserSegmentSearcher primary_searcher(primary_->segment_directory,
                                               laser::ResidencyMode::kResidentArena);
  UnifiedLaserSegmentSearcher secondary_searcher(secondary_->segment_directory,
                                                 laser::ResidencyMode::kResidentArena);
  const auto primary_options =
      make_options(/*ef=*/160, /*beam=*/16, /*return_distances=*/true);
  const auto secondary_options =
      make_options(/*ef=*/24, /*beam=*/2, /*return_distances=*/true);
  constexpr std::size_t kQueries = 16;
  std::vector<std::vector<DiskSearchHit>> primary_baselines;
  std::vector<std::vector<DiskSearchHit>> secondary_baselines;
  for (std::size_t query = 0; query < kQueries; ++query) {
    primary_baselines.push_back(primary_searcher.search(
        primary_->data.data() + query * primary_->dim, primary_options));
    secondary_baselines.push_back(secondary_searcher.search(
        secondary_->data.data() + query * secondary_->dim, secondary_options));
  }

  const int iterations = kRunningTsan ? 8 : 100;
  const int failures = run_workers(kConcurrentThreads, [&](int thread) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const std::size_t query =
          static_cast<std::size_t>(thread * 7 + iteration * 3) % kQueries;
      const auto run_primary = [&] {
        return same_hits(primary_baselines[query],
                         primary_searcher.search(
                             primary_->data.data() + query * primary_->dim, primary_options),
                         /*return_distances=*/true);
      };
      const auto run_secondary = [&] {
        return same_hits(secondary_baselines[query],
                         secondary_searcher.search(
                             secondary_->data.data() + query * secondary_->dim, secondary_options),
                         /*return_distances=*/true);
      };
      // Both orders force the same OS thread to reuse one TLS scratch across
      // 128d/ef160 and 64d/ef24 graphs, including a logical capacity shrink.
      const bool ok = (thread + iteration) % 2 == 0
                          ? (run_primary() && run_secondary())
                          : (run_secondary() && run_primary());
      if (!ok) {
        return false;
      }
    }
    return true;
  });
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(primary_searcher.set_params_call_count(), 0U);
  EXPECT_EQ(secondary_searcher.set_params_call_count(), 0U);
}

TEST_F(LaserArenaReentrancyTest, ConcurrentBatchLowersToReentrantQueries) {
  if (!fixture_available()) {
    GTEST_SKIP() << "LASER is unavailable";
  }
  UnifiedLaserSegmentSearcher searcher(primary_->segment_directory,
                                       laser::ResidencyMode::kResidentArena);
  constexpr std::uint32_t kQueries = 8;
  const std::array options{
      make_options(/*ef=*/64, /*beam=*/4, /*return_distances=*/false),
      make_options(/*ef=*/128, /*beam=*/8, /*return_distances=*/true),
  };
  std::array<std::vector<std::vector<DiskSearchHit>>, 2> baselines;
  for (std::size_t mode = 0; mode < options.size(); ++mode) {
    baselines[mode] = searcher.batch_search(primary_->data.data(), kQueries, options[mode]);
    ASSERT_EQ(baselines[mode].size(), kQueries);
    for (std::size_t query = 0; query < kQueries; ++query) {
      const auto single = searcher.search(
          primary_->data.data() + query * primary_->dim, options[mode]);
      ASSERT_TRUE(same_hits(single,
                            baselines[mode][query],
                            options[mode].return_distances));
    }
  }

  const int iterations = kRunningTsan ? 4 : 40;
  const int failures = run_workers(kConcurrentThreads, [&](int thread) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const std::size_t mode = static_cast<std::size_t>(thread + iteration) % options.size();
      const auto actual = searcher.batch_search(primary_->data.data(), kQueries, options[mode]);
      if (actual.size() != kQueries) {
        return false;
      }
      for (std::size_t query = 0; query < kQueries; ++query) {
        if (!same_hits(baselines[mode][query],
                       actual[query],
                       options[mode].return_distances)) {
          return false;
        }
      }
    }
    return true;
  });
  EXPECT_EQ(failures, 0);
  EXPECT_EQ(searcher.set_params_call_count(), 0U);
}

}  // namespace
}  // namespace alaya::disk
