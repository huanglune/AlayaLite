// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/any_segment.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using Nanoseconds = std::chrono::nanoseconds;
using alaya::core::AnySegment;
using alaya::core::Descriptor;
using alaya::core::Metric;
using alaya::core::OperationStage;
using alaya::core::RowCount;
using alaya::core::ScalarType;
using alaya::core::ScoreKind;
using alaya::core::SearchCompleteness;
using alaya::core::SearchContext;
using alaya::core::SearchHit;
using alaya::core::SearchRequest;
using alaya::core::SearchResponse;
using alaya::core::SegmentRowId;
using alaya::core::Status;
using alaya::core::StatusCode;
using alaya::core::StatusDetail;
using alaya::core::TypedTensorView;

constexpr std::uint32_t kDim = 8;
constexpr std::size_t kVectorCount = 32;

class SmallVectorSegment {
 public:
  SmallVectorSegment() {
    for (std::size_t row = 0; row < kVectorCount; ++row) {
      for (std::uint32_t column = 0; column < kDim; ++column) {
        vectors_[row * kDim + column] =
            std::sin(static_cast<float>((row + 1) * (column + 3)) * 0.071F);
      }
    }
  }

  [[nodiscard]] auto descriptor() const noexcept -> Descriptor {
    Descriptor value;
    value.algorithm_id = 0xA11A;
    value.dim = kDim;
    value.metric = Metric::l2;
    value.stored_scalar_type = ScalarType::float32;
    return value;
  }

  [[nodiscard]] auto search(const SearchRequest &request) const -> Status {
    if (request.queries.scalar_type != ScalarType::float32 || request.queries.rows != 1 ||
        request.queries.dim != kDim || request.options.top_k != 1) {
      return Status::error(StatusCode::invalid_argument,
                           OperationStage::search,
                           StatusDetail::malformed_struct,
                           "microbenchmark request is malformed");
    }
    const auto query = request.queries.row<float>(0);
    float best_distance = std::numeric_limits<float>::infinity();
    std::size_t best_row{};
    for (std::size_t row = 0; row < kVectorCount; ++row) {
      float distance{};
      for (std::uint32_t column = 0; column < kDim; ++column) {
        const auto delta = query[column] - vectors_[row * kDim + column];
        distance += delta * delta;
      }
      if (distance < best_distance) {
        best_distance = distance;
        best_row = row;
      }
    }

    auto &response = *request.response;
    response.query_count = 1;
    response.hits[0] =
        SearchHit(SegmentRowId(best_row), best_distance, ScoreKind::distance, Metric::l2);
    response.offsets[0] = 0;
    response.offsets[1] = 1;
    response.valid_counts[0] = 1;
    response.statuses[0] = Status::success();
    response.completeness[0] = SearchCompleteness::complete_k;
    return Status::success();
  }

 private:
  std::array<float, kVectorCount * kDim> vectors_{};
};

struct ResponseStorage {
  ResponseStorage() {
    response.hits = hits;
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
  }

  std::array<SearchHit, 1> hits{};
  std::array<RowCount, 2> offsets{};
  std::array<RowCount, 1> counts{};
  std::array<Status, 1> statuses{};
  std::array<SearchCompleteness, 1> completeness{};
  SearchResponse response{};
};

struct RunResult {
  double p50_microseconds{};
  double p99_microseconds{};
  double qps{};
  std::uint64_t checksum{};
};

[[nodiscard]] auto percentile(const std::vector<std::uint64_t> &sorted, double fraction) -> double {
  const auto index = static_cast<std::size_t>(
      std::min<double>(static_cast<double>(sorted.size() - 1),
                       std::ceil(fraction * static_cast<double>(sorted.size())) - 1.0));
  return static_cast<double>(sorted[index]) / 1000.0;
}

[[nodiscard]] auto run_once(const AnySegment &segment,
                            std::uint64_t iterations,
                            std::uint64_t warmup) -> RunResult {
  std::array<float, kDim> query{};
  SearchContext context;
  ResponseStorage storage;
  SearchRequest request;
  request.queries = TypedTensorView::contiguous(query.data(), 1, kDim);
  request.options.top_k = 1;
  request.context = &context;
  request.response = &storage.response;

  std::uint64_t checksum{};
  auto invoke = [&](std::uint64_t index) {
    query[0] = static_cast<float>(index % 17U) * 0.003F;
    const auto status = segment.search(request);
    if (!status.ok()) {
      throw std::runtime_error(status.diagnostic());
    }
    checksum += static_cast<std::uint64_t>(storage.hits[0].row_id);
  };
  for (std::uint64_t index = 0; index < warmup; ++index) {
    invoke(index);
  }

  std::vector<std::uint64_t> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  const auto run_begin = Clock::now();
  for (std::uint64_t index = 0; index < iterations; ++index) {
    const auto begin = Clock::now();
    invoke(index + warmup);
    const auto elapsed = std::chrono::duration_cast<Nanoseconds>(Clock::now() - begin).count();
    samples.push_back(static_cast<std::uint64_t>(elapsed));
  }
  const auto run_seconds = std::chrono::duration<double>(Clock::now() - run_begin).count();
  std::ranges::sort(samples);
  return RunResult{percentile(samples, 0.50),
                   percentile(samples, 0.99),
                   static_cast<double>(iterations) / run_seconds,
                   checksum};
}

[[nodiscard]] auto parse_count(const char *text, const char *name) -> std::uint64_t {
  char *end{};
  const auto value = std::strtoull(text, &end, 10);
  if (text == end || *end != '\0' || value == 0) {
    throw std::invalid_argument(std::string(name) + " must be a positive integer");
  }
  return value;
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    const auto iterations = argc > 1 ? parse_count(argv[1], "iterations") : 20'000U;
    const auto repeats = argc > 2 ? parse_count(argv[2], "repeats") : 5U;
    const auto warmup = argc > 3 ? parse_count(argv[3], "warmup") : 2'000U;
    if (argc > 4) {
      throw std::invalid_argument(
          "usage: any_segment_sync_benchmark [iterations] [repeats] [warmup]");
    }

    auto erased = AnySegment::from_sync(std::make_shared<SmallVectorSegment>());
    if (!erased.ok()) {
      throw std::runtime_error(erased.status().diagnostic());
    }
    const auto segment = std::move(erased).value();
    std::vector<RunResult> results;
    results.reserve(static_cast<std::size_t>(repeats));
    for (std::uint64_t repeat = 0; repeat < repeats; ++repeat) {
      results.push_back(run_once(segment, iterations, warmup));
      const auto &result = results.back();
      std::cout << "run=" << (repeat + 1) << " p50_us=" << std::fixed << std::setprecision(3)
                << result.p50_microseconds << " p99_us=" << result.p99_microseconds
                << " qps=" << std::setprecision(1) << result.qps << " checksum=" << result.checksum
                << '\n';
    }

    std::vector<double> p50;
    std::vector<double> p99;
    std::vector<double> qps;
    for (const auto &result : results) {
      p50.push_back(result.p50_microseconds);
      p99.push_back(result.p99_microseconds);
      qps.push_back(result.qps);
    }
    std::ranges::sort(p50);
    std::ranges::sort(p99);
    std::ranges::sort(qps);
    const auto median = [](const auto &values) {
      return values[values.size() / 2];
    };
    std::cout << "median p50_us=" << std::fixed << std::setprecision(3) << median(p50)
              << " p99_us=" << median(p99) << " qps=" << std::setprecision(1) << median(qps)
              << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "any_segment_sync_benchmark: " << error.what() << '\n';
    return 2;
  }
}
