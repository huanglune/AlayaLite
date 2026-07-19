// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <omp.h>

#include "index/collection/segmented_collection.hpp"
#include "index/disk/disk_flat_segment.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/laser_segment_importer.hpp"
#include "index/graph/frozen_graph_snapshot.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"

namespace {

using alaya::core::LogicalId;
using alaya::internal::collection::CollectionSearchRequest;
using alaya::internal::collection::CollectionSearchStats;
using alaya::internal::collection::OwnedVector;
using alaya::internal::collection::RecordPayload;
using alaya::internal::collection::RegisteredRow;
using alaya::internal::collection::SegmentRegistration;
using alaya::internal::collection::SegmentRole;
using alaya::internal::collection::SegmentedCollection;
using alaya::internal::collection::VersionState;

struct Args {
  std::filesystem::path root{};
  std::uint32_t dim{128};
  std::uint32_t rows{100'000};
  std::uint32_t flat_rows{4096};
  std::uint32_t queries{200};
  std::uint32_t warmup{16};
  std::uint32_t threads{16};
  bool rebuild{};
};

[[nodiscard]] auto parse_u32(std::string_view text, std::string_view name) -> std::uint32_t {
  std::size_t consumed{};
  const auto parsed = std::stoull(std::string(text), &consumed);
  if (consumed != text.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " is not a uint32");
  }
  return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] auto parse_args(int argc, char **argv) -> Args {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--rebuild") {
      args.rebuild = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + std::string(option));
    }
    const std::string_view value(argv[++index]);
    if (option == "--root") {
      args.root = value;
    } else if (option == "--dim") {
      args.dim = parse_u32(value, option);
    } else if (option == "--rows") {
      args.rows = parse_u32(value, option);
    } else if (option == "--flat-rows") {
      args.flat_rows = parse_u32(value, option);
    } else if (option == "--queries") {
      args.queries = parse_u32(value, option);
    } else if (option == "--warmup") {
      args.warmup = parse_u32(value, option);
    } else if (option == "--threads") {
      args.threads = parse_u32(value, option);
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  if (args.root.empty() || args.dim == 0 || args.rows < 256 || args.flat_rows == 0 ||
      args.queries == 0 || args.threads == 0 || args.threads > 16) {
    throw std::invalid_argument(
        "usage: result_contract_benchmark --root PATH --dim D [--rows N] "
        "[--flat-rows N] [--queries N] [--warmup N] [--threads 1..16] [--rebuild]");
  }
  return args;
}

[[nodiscard]] auto splitmix64(std::uint64_t &state) -> std::uint64_t {
  state += 0x9E3779B97F4A7C15ULL;
  auto value = state;
  value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31U);
}

void fill_vectors(std::vector<float> &values,
                  std::uint32_t rows,
                  std::uint32_t dim,
                  std::uint64_t seed) {
  values.resize(static_cast<std::size_t>(rows) * dim);
  for (std::uint32_t row = 0; row < rows; ++row) {
    double squared_norm{};
    for (std::uint32_t column = 0; column < dim; ++column) {
      const auto random = static_cast<std::uint32_t>(splitmix64(seed) >> 40U);
      const auto value = static_cast<float>(random) / static_cast<float>(1U << 23U) - 1.0F;
      values[static_cast<std::size_t>(row) * dim + column] = value;
      squared_norm += static_cast<double>(value) * value;
    }
    const auto inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    const auto scale = 0.5F + static_cast<float>((row * 17U) % 31U) / 31.0F;
    for (std::uint32_t column = 0; column < dim; ++column) {
      values[static_cast<std::size_t>(row) * dim + column] *= inverse_norm * scale;
    }
  }
}

void write_fbin(const std::filesystem::path &path,
                std::span<const float> vectors,
                std::uint32_t rows,
                std::uint32_t dim) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create " + path.string());
  }
  const auto n = static_cast<std::int32_t>(rows);
  const auto d = static_cast<std::int32_t>(dim);
  output.write(reinterpret_cast<const char *>(&n), sizeof(n));
  output.write(reinterpret_cast<const char *>(&d), sizeof(d));
  output.write(reinterpret_cast<const char *>(vectors.data()),
               static_cast<std::streamsize>(vectors.size_bytes()));
  if (!output) {
    throw std::runtime_error("failed writing " + path.string());
  }
}

[[nodiscard]] auto make_small_world_graph(std::uint32_t rows, std::uint32_t degree)
    -> alaya::FrozenGraphSnapshot {
  alaya::FrozenGraphSnapshot::Adjacency adjacency(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    auto &neighbors = adjacency[row];
    neighbors.reserve(degree);
    for (std::uint32_t bit = 0; neighbors.size() < degree && bit < 31; ++bit) {
      const auto delta = (std::uint64_t{1} << bit) % rows;
      const auto forward = static_cast<std::uint32_t>((row + delta) % rows);
      const auto backward = static_cast<std::uint32_t>((row + rows - delta) % rows);
      if (forward != row && std::ranges::find(neighbors, forward) == neighbors.end()) {
        neighbors.push_back(forward);
      }
      if (neighbors.size() < degree && backward != row &&
          std::ranges::find(neighbors, backward) == neighbors.end()) {
        neighbors.push_back(backward);
      }
    }
    for (std::uint32_t delta = 1; neighbors.size() < degree; ++delta) {
      const auto neighbor = (row + delta) % rows;
      if (neighbor != row && std::ranges::find(neighbors, neighbor) == neighbors.end()) {
        neighbors.push_back(neighbor);
      }
    }
  }
  return alaya::FrozenGraphSnapshot(std::move(adjacency), 0, degree);
}

[[nodiscard]] auto build_or_open_laser(const Args &args, std::span<const float> vectors)
    -> alaya::core::AnySegment {
  constexpr std::uint32_t kDegree = 32;
  const auto segment_directory = args.root / "laser/segments/seg_00000001";
  if (!std::filesystem::is_regular_file(segment_directory / "manifest.txt")) {
    const auto raw_directory = args.root / "laser-raw";
    std::filesystem::remove_all(raw_directory);
    std::filesystem::create_directories(raw_directory);
    std::filesystem::create_directories(segment_directory.parent_path());
    const auto prefix = raw_directory / "dsqg_seg_00000001";
    write_fbin(prefix.string() + "_pca_base.fbin", vectors, args.rows, args.dim);
    auto graph = make_small_world_graph(args.rows, kDegree);
    alaya::laser::QuantizedGraph qg(args.rows,
                                    kDegree,
                                    args.dim,
                                    args.dim,
                                    /*rotator_seed=*/0xA11A0718ULL);
    alaya::laser::QGBuilder builder(qg, /*ef_build=*/128, args.threads);
    builder.build_from_graph(graph, prefix.c_str());

    std::vector<std::uint64_t> labels(args.rows);
    std::iota(labels.begin(), labels.end(), std::uint64_t{0});
    alaya::disk::LaserSegmentImportParams import_options;
    import_options.R = kDegree;
    import_options.default_ef = 128;
    import_options.residency = "resident_arena";
    alaya::disk::LaserSegmentImporter importer(args.dim,
                                                alaya::core::Metric::l2,
                                                import_options);
    (void)importer.import_from(raw_directory,
                               labels.data(),
                               labels.size(),
                               segment_directory);
    std::filesystem::remove_all(raw_directory);
  }

  alaya::core::OpenContext context;
  auto opened = alaya::disk::LaserSegment::open_directory(segment_directory,
                                                           alaya::core::OpenOptions{},
                                                           context);
  if (!opened.ok()) {
    throw std::runtime_error(opened.status().diagnostic());
  }
  auto erased = alaya::disk::LaserSegment::into_any(std::move(opened).value());
  if (!erased.ok()) {
    throw std::runtime_error(erased.status().diagnostic());
  }
  return std::move(erased).value();
}

[[nodiscard]] auto build_or_open_flat(const Args &args, std::span<const float> vectors)
    -> alaya::core::AnySegment {
  const auto collection_root = args.root / "flat";
  const auto segment_directory = collection_root / "segments/seg_00000002";
  std::unique_ptr<alaya::disk::DiskFlatSegment> segment;
  if (std::filesystem::is_regular_file(segment_directory / "manifest.txt")) {
    alaya::core::OpenContext context;
    auto opened = alaya::disk::DiskFlatSegment::open_directory(segment_directory,
                                                                alaya::core::OpenOptions{},
                                                                context);
    if (!opened.ok()) {
      throw std::runtime_error(opened.status().diagnostic());
    }
    segment = std::move(opened).value();
  } else {
    std::vector<std::uint64_t> labels(args.flat_rows);
    std::iota(labels.begin(), labels.end(), static_cast<std::uint64_t>(args.rows));
    alaya::disk::DiskFlatBuildInput input(
        alaya::core::TypedTensorView::contiguous(vectors.data(), args.flat_rows, args.dim),
        labels);
    alaya::disk::DiskFlatPublicationOptions options;
    options.collection_root = collection_root;
    options.segment_id = "seg_00000002";
    alaya::core::BuildContext context;
    auto built = alaya::disk::DiskFlatSegmentFactory::build(std::move(input),
                                                            alaya::core::Metric::l2,
                                                            options,
                                                            context);
    if (!built.ok()) {
      throw std::runtime_error(built.status().diagnostic());
    }
    segment = std::move(built).value();
  }
  auto erased = alaya::disk::DiskFlatSegment::into_any(std::move(segment));
  if (!erased.ok()) {
    throw std::runtime_error(erased.status().diagnostic());
  }
  return std::move(erased).value();
}

[[nodiscard]] auto make_registration(std::uint64_t segment_id,
                                     alaya::core::AnySegment segment,
                                     std::span<const float> vectors,
                                     std::uint32_t rows,
                                     std::uint32_t dim,
                                     std::uint64_t id_base,
                                     bool retain_vectors) -> SegmentRegistration {
  SegmentRegistration registration;
  registration.segment_id = segment_id;
  registration.role = SegmentRole::sealed;
  registration.segment = std::move(segment);
  registration.rows.reserve(rows);
  const auto tensor = alaya::core::TypedTensorView::contiguous(vectors.data(), rows, dim);
  for (std::uint32_t row = 0; row < rows; ++row) {
    RecordPayload payload;
    if (retain_vectors) {
      auto copied = OwnedVector::copy_row(tensor, row);
      if (!copied.ok()) {
        throw std::runtime_error(copied.status().diagnostic());
      }
      payload.vector = std::move(copied).value();
    }
    const auto id = id_base + row;
    registration.rows.push_back(RegisteredRow{LogicalId::from_legacy_uint64(id),
                                               alaya::core::SegmentRowId(id),
                                               0,
                                               VersionState::live,
                                               std::move(payload)});
  }
  return registration;
}

struct Measurement {
  double qps{};
  double p50_us{};
  double p99_us{};
  double rerank_share_percent{};
  std::uint64_t rerank_count{};
  std::uint64_t checksum{};
};

struct PairMeasurement {
  Measurement rank_only{};
  Measurement distance{};
};

struct Samples {
  std::vector<std::uint64_t> latencies{};
  std::uint64_t rerank_nanoseconds{};
  std::uint64_t rerank_count{};
  std::uint64_t checksum{};
};

void run_one(const std::shared_ptr<SegmentedCollection> &collection,
             const float *query,
             std::uint32_t dim,
             std::uint32_t top_k,
             std::uint32_t effort,
             bool return_distances,
             Samples *samples) {
  alaya::disk::LaserSegmentSearchExtension extension_options;
  extension_options.effort = effort;
  extension_options.beam_width = 4;
  extension_options.return_distances = return_distances;
  auto extension = alaya::disk::LaserSegment::make_search_extension(extension_options);
  extension.unknown_policy = alaya::core::UnknownExtensionPolicy::ignore_safe;
  alaya::core::SearchStats engine_stats;
  alaya::core::SearchContext context;
  context.stats = &engine_stats;
  CollectionSearchStats collection_stats;
  CollectionSearchRequest request;
  request.queries = alaya::core::TypedTensorView::contiguous(query, 1, dim);
  request.options.top_k = top_k;
  request.options.extensions = std::span<const alaya::core::AlgorithmSearchExtension>(&extension, 1);
  request.context = &context;
  request.stats = &collection_stats;

  const auto started = std::chrono::steady_clock::now();
  auto result = collection->search(request);
  const auto elapsed = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started)
          .count());
  if (!result.ok()) {
    throw std::runtime_error(result.status().diagnostic());
  }
  if (result.value().queries.size() != 1 || !result.value().queries[0].status.ok()) {
    throw std::runtime_error("Collection returned a failed query");
  }
  if (samples == nullptr) {
    return;
  }
  samples->latencies.push_back(elapsed);
  samples->rerank_nanoseconds += collection_stats.rerank_nanoseconds;
  samples->rerank_count += engine_stats.rerank_count;
  for (const auto &hit : result.value().queries[0].hits) {
    samples->checksum ^= std::rotl(static_cast<std::uint64_t>(std::bit_cast<std::uint32_t>(hit.score)),
                                  static_cast<int>(samples->checksum & 31U));
    samples->checksum += hit.upsert_sequence + 0x9E3779B97F4A7C15ULL;
  }
}

[[nodiscard]] auto summarize(Samples samples) -> Measurement {
  if (samples.latencies.empty()) {
    throw std::runtime_error("no benchmark samples");
  }
  std::sort(samples.latencies.begin(), samples.latencies.end());
  const auto total = std::accumulate(samples.latencies.begin(),
                                     samples.latencies.end(),
                                     std::uint64_t{0});
  const auto percentile = [&](double value) {
    const auto index = std::min<std::size_t>(
        samples.latencies.size() - 1,
        static_cast<std::size_t>(std::ceil(value * samples.latencies.size())) - 1);
    return static_cast<double>(samples.latencies[index]) / 1000.0;
  };
  Measurement result;
  result.qps = static_cast<double>(samples.latencies.size()) * 1.0e9 / static_cast<double>(total);
  result.p50_us = percentile(0.50);
  result.p99_us = percentile(0.99);
  result.rerank_share_percent =
      100.0 * static_cast<double>(samples.rerank_nanoseconds) / static_cast<double>(total);
  result.rerank_count = samples.rerank_count;
  result.checksum = samples.checksum;
  return result;
}

[[nodiscard]] auto measure_pair(const std::shared_ptr<SegmentedCollection> &collection,
                                std::span<const float> queries,
                                std::uint32_t query_count,
                                std::uint32_t warmup,
                                std::uint32_t dim,
                                std::uint32_t top_k,
                                std::uint32_t effort) -> PairMeasurement {
  for (std::uint32_t index = 0; index < warmup; ++index) {
    const auto *query = queries.data() + static_cast<std::size_t>(index % query_count) * dim;
    run_one(collection, query, dim, top_k, effort, false, nullptr);
    run_one(collection, query, dim, top_k, effort, true, nullptr);
  }
  Samples rank_only;
  Samples distance;
  rank_only.latencies.reserve(query_count);
  distance.latencies.reserve(query_count);
  for (std::uint32_t index = 0; index < query_count; ++index) {
    const auto *query = queries.data() + static_cast<std::size_t>(index) * dim;
    if ((index & 1U) == 0U) {
      run_one(collection, query, dim, top_k, effort, false, &rank_only);
      run_one(collection, query, dim, top_k, effort, true, &distance);
    } else {
      run_one(collection, query, dim, top_k, effort, true, &distance);
      run_one(collection, query, dim, top_k, effort, false, &rank_only);
    }
  }
  return {summarize(std::move(rank_only)), summarize(std::move(distance))};
}

void print_measurement(const Args &args,
                       std::string_view topology,
                       std::string_view arm,
                       std::uint32_t top_k,
                       std::uint32_t effort,
                       const Measurement &value) {
  std::cout << "RESULT," << args.dim << ',' << args.rows << ',' << args.flat_rows << ','
            << topology << ',' << arm << ',' << top_k << ',' << effort << ',' << args.queries << ','
            << std::fixed << std::setprecision(3) << value.qps << ',' << value.p50_us << ','
            << value.p99_us << ',' << value.rerank_share_percent << ',' << value.rerank_count << ','
            << value.checksum << '\n';
}

void run_topology(const Args &args,
                  std::string_view topology,
                  const alaya::core::AnySegment &laser,
                  const alaya::core::AnySegment &flat,
                  std::span<const float> vectors,
                  std::span<const float> flat_vectors,
                  std::span<const float> queries,
                  bool mixed) {
  std::vector<SegmentRegistration> registrations;
  registrations.push_back(make_registration(1,
                                            laser,
                                            vectors,
                                            args.rows,
                                            args.dim,
                                            0,
                                            /*retain_vectors=*/true));
  if (mixed) {
    registrations.push_back(make_registration(2,
                                              flat,
                                              flat_vectors,
                                              args.flat_rows,
                                              args.dim,
                                              args.rows,
                                              /*retain_vectors=*/false));
  }
  auto opened = SegmentedCollection::open({args.dim,
                                           alaya::core::Metric::l2,
                                           alaya::core::ScalarType::float32},
                                          std::move(registrations));
  if (!opened.ok()) {
    throw std::runtime_error(opened.status().diagnostic());
  }
  auto collection = std::move(opened).value();
  for (const auto top_k : {10U, 100U}) {
    for (const auto effort : {128U, 512U}) {
      const auto pair = measure_pair(collection,
                                     queries,
                                     args.queries,
                                     args.warmup,
                                     args.dim,
                                     top_k,
                                     effort);
      print_measurement(args, topology, "A_rank_only", top_k, effort, pair.rank_only);
      print_measurement(args, topology, "B_distance", top_k, effort, pair.distance);
      const auto tax = pair.rank_only.p50_us == 0.0
                           ? 0.0
                           : 100.0 * (pair.rank_only.p50_us - pair.distance.p50_us) /
                                 pair.rank_only.p50_us;
      std::cout << "PAIRED_TAX," << args.dim << ',' << topology << ',' << top_k << ',' << effort
                << ',' << std::fixed << std::setprecision(3) << tax << '\n';
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const auto args = parse_args(argc, argv);
    omp_set_dynamic(0);
    omp_set_num_threads(static_cast<int>(args.threads));
    if (args.rebuild) {
      std::filesystem::remove_all(args.root);
    }
    std::filesystem::create_directories(args.root);

    std::vector<float> vectors;
    std::vector<float> flat_vectors;
    fill_vectors(vectors, args.rows, args.dim, 0xA11A0718D15C0001ULL + args.dim);
    fill_vectors(flat_vectors, args.flat_rows, args.dim, 0xA11A0718F1A70002ULL + args.dim);

    auto laser = build_or_open_laser(args, vectors);
    auto flat = build_or_open_flat(args, flat_vectors);
    std::vector<float> queries(static_cast<std::size_t>(args.queries) * args.dim);
    for (std::uint32_t query = 0; query < args.queries; ++query) {
      const bool from_flat = (query & 1U) != 0U;
      const auto source = from_flat ? (query * 97U) % args.flat_rows : (query * 9973U) % args.rows;
      const auto &source_vectors = from_flat ? flat_vectors : vectors;
      std::copy_n(source_vectors.data() + static_cast<std::size_t>(source) * args.dim,
                  args.dim,
                  queries.data() + static_cast<std::size_t>(query) * args.dim);
    }

    std::cout << "CONFIG,dim=" << args.dim << ",rows=" << args.rows
              << ",flat_rows=" << args.flat_rows << ",queries=" << args.queries
              << ",warmup=" << args.warmup << ",build_threads=" << args.threads
              << ",search_driver_threads=1,residency=resident_arena,metric=l2,R=32\n";
    std::cout << "HEADER,dim,rows,flat_rows,topology,arm,k,ef,queries,qps,p50_us,p99_us,"
                 "rerank_stage_percent,rerank_count,checksum\n";
    run_topology(args,
                 "laser_only",
                 laser,
                 flat,
                 vectors,
                 flat_vectors,
                 queries,
                 /*mixed=*/false);
    run_topology(args,
                 "laser_flat",
                 laser,
                 flat,
                 vectors,
                 flat_vectors,
                 queries,
                 /*mixed=*/true);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "result_contract_benchmark: " << error.what() << '\n';
    return 1;
  }
}
