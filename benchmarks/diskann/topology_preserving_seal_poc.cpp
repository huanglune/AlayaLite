// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Topology-preserving seal proof of concept.
//
// The materialized arm builds a Vamana graph once (standing in for an already
// active memory graph), freezes its public graph()/medoid() view, and writes a
// complete read-only DiskANN directory without invoking VamanaBuilder again.
// The rebuilt arm uses DiskANNIndex::build on the same rows and parameters.
// Both arms are then opened through DiskANNIndex and evaluated against exact GT.
//
// This is deliberately benchmark-local glue.  The production DiskANN writer
// already accepts an existing graph through write_disk_layout(), while the
// directory-level DiskANNIndex API currently exposes only build(raw vectors).
// The tiny meta.bin/ids.bin serializers below compose that public writer with
// the public PQTable and NodeCache writers without changing any engine or
// format implementation.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/node_cache.hpp"
#include "index/graph/diskann/pq_table.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "platform/fs.hpp"

namespace {

using Clock = std::chrono::steady_clock;
using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskANNLoadParams;
using alaya::diskann::DiskANNSearchParams;
using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::NodeCache;
using alaya::diskann::PQTable;

enum class Phase { all, build, search };

struct Args {
  std::filesystem::path base_path;
  std::filesystem::path query_path;
  std::filesystem::path gt_path;
  std::filesystem::path output_dir;
  Phase phase{Phase::all};
  std::uint64_t n{0};
  std::uint32_t nq{0};
  std::uint32_t r{64};
  std::uint32_t build_l{100};
  float alpha{1.2F};
  std::uint32_t build_threads{48};
  std::uint32_t search_threads{8};
  std::uint32_t beam_width{4};
  std::uint32_t pq_chunks{32};
  std::uint32_t pq_train_iters{15};
  double cache_ratio{0.01};
  std::uint64_t seed{1234};
  std::vector<std::uint32_t> search_ls{100, 200, 400};
  std::uint32_t repetitions{3};
  std::uint32_t warmup_queries{100};
  bool deterministic{false};
  bool rebuild{false};
};

struct FloatMatrix {
  std::vector<float> values;
  std::uint32_t rows{};
  std::uint32_t dim{};
  std::uint32_t source_rows{};

  [[nodiscard]] auto row(std::uint32_t index) const -> const float * {
    return values.data() + static_cast<std::size_t>(index) * dim;
  }
};

struct IntMatrix {
  std::vector<std::uint32_t> values;
  std::uint32_t rows{};
  std::uint32_t dim{};

  [[nodiscard]] auto row(std::uint32_t index) const -> const std::uint32_t * {
    return values.data() + static_cast<std::size_t>(index) * dim;
  }
};

struct TopologyStats {
  std::uint64_t digest{};
  std::uint64_t edges{};
  std::uint64_t zero_degree_nodes{};
  std::uint32_t max_degree{};
  double mean_degree{};
};

struct MaterializeMetrics {
  double total_s{};
  double layout_s{};
  double ids_s{};
  double pq_s{};
  double cache_s{};
  double meta_s{};
};

struct BuildMetrics {
  std::uint64_t n{};
  std::uint32_t dim{};
  double active_graph_build_s{};
  double freeze_snapshot_s{};
  MaterializeMetrics materialize{};
  double topology_verify_s{};
  double rebuilt_total_s{};
  TopologyStats frozen{};
  TopologyStats materialized{};
  TopologyStats rebuilt{};
  std::uint64_t materialized_bytes{};
  std::uint64_t rebuilt_bytes{};
  std::uint64_t materialized_pq_pivots_digest{};
  std::uint64_t rebuilt_pq_pivots_digest{};
  std::uint64_t materialized_pq_codes_digest{};
  std::uint64_t rebuilt_pq_codes_digest{};
};

struct SearchMetric {
  std::string arm;
  std::uint32_t search_l{};
  double recall_at_10{};
  double recall_at_100{};
  double qps{};
  std::vector<double> qps_samples;
};

[[noreturn]] void usage(const char *argv0, std::string_view error = {}) {
  if (!error.empty()) {
    std::cerr << "error: " << error << "\n\n";
  }
  std::cerr << "Usage: " << argv0 << " --output-dir DIR [--phase all|build|search] [options]\n"
            << "  build/all inputs:  --base BASE.fbin\n"
            << "  search/all inputs: --query QUERY.fbin --gt GT100.ibin\n\n"
            << "Build options:\n"
            << "  --n N                 prefix rows; 0 means all (default 0)\n"
            << "  --r R                 graph degree (default 64)\n"
            << "  --build-l L           Vamana build beam (default 100)\n"
            << "  --alpha A             Vamana alpha (default 1.2)\n"
            << "  --build-threads N     must be <=48 (default 48)\n"
            << "  --pq-chunks N         0 disables PQ (default 32)\n"
            << "  --pq-train-iters N    k-means iterations (default 15)\n"
            << "  --cache-ratio X       BFS node-cache ratio (default 0.01)\n"
            << "  --seed N              build/PQ seed (default 1234)\n"
            << "  --rebuild             remove both prior arm directories\n\n"
            << "Search options:\n"
            << "  --nq N                query prefix; 0 means all (default 0)\n"
            << "  --search-threads N    concurrent query workers (default 8)\n"
            << "  --beam-width N        DiskANN I/O beam (default 4)\n"
            << "  --search-l A,B,C      at least three values, all >=100\n"
            << "  --repetitions N       report median recall/QPS (default 3)\n"
            << "  --warmup-queries N    per-arm warmup count (default 100)\n"
            << "  --deterministic       enable deterministic beam barriers\n";
  std::exit(error.empty() ? 0 : 2);
}

auto take_value(int &index, int argc, char **argv, std::string_view flag) -> std::string {
  if (index + 1 >= argc) {
    usage(argv[0], std::string(flag) + " requires a value");
  }
  return argv[++index];
}

auto parse_search_ls(std::string_view value) -> std::vector<std::uint32_t> {
  std::vector<std::uint32_t> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto end = value.find(',', begin);
    const auto token =
        value.substr(begin, end == std::string_view::npos ? value.size() - begin : end - begin);
    if (token.empty()) {
      throw std::invalid_argument("--search-l contains an empty value");
    }
    result.push_back(static_cast<std::uint32_t>(std::stoul(std::string(token))));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return result;
}

auto parse_args(int argc, char **argv) -> Args {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (flag == "--help" || flag == "-h") {
      usage(argv[0]);
    } else if (flag == "--base") {
      args.base_path = take_value(i, argc, argv, flag);
    } else if (flag == "--query") {
      args.query_path = take_value(i, argc, argv, flag);
    } else if (flag == "--gt") {
      args.gt_path = take_value(i, argc, argv, flag);
    } else if (flag == "--output-dir") {
      args.output_dir = take_value(i, argc, argv, flag);
    } else if (flag == "--phase") {
      const auto value = take_value(i, argc, argv, flag);
      if (value == "all") {
        args.phase = Phase::all;
      } else if (value == "build") {
        args.phase = Phase::build;
      } else if (value == "search") {
        args.phase = Phase::search;
      } else {
        usage(argv[0], "--phase must be all, build, or search");
      }
    } else if (flag == "--n") {
      args.n = std::stoull(take_value(i, argc, argv, flag));
    } else if (flag == "--nq") {
      args.nq = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--r") {
      args.r = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--build-l") {
      args.build_l = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--alpha") {
      args.alpha = std::stof(take_value(i, argc, argv, flag));
    } else if (flag == "--build-threads") {
      args.build_threads = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--search-threads") {
      args.search_threads = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--beam-width") {
      args.beam_width = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--pq-chunks") {
      args.pq_chunks = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--pq-train-iters") {
      args.pq_train_iters = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--cache-ratio") {
      args.cache_ratio = std::stod(take_value(i, argc, argv, flag));
    } else if (flag == "--seed") {
      args.seed = std::stoull(take_value(i, argc, argv, flag));
    } else if (flag == "--search-l") {
      args.search_ls = parse_search_ls(take_value(i, argc, argv, flag));
    } else if (flag == "--repetitions") {
      args.repetitions = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--warmup-queries") {
      args.warmup_queries = static_cast<std::uint32_t>(std::stoul(take_value(i, argc, argv, flag)));
    } else if (flag == "--deterministic") {
      args.deterministic = true;
    } else if (flag == "--rebuild") {
      args.rebuild = true;
    } else {
      usage(argv[0], "unknown flag: " + flag);
    }
  }

  if (args.output_dir.empty()) {
    usage(argv[0], "--output-dir is required");
  }
  const bool do_build = args.phase != Phase::search;
  const bool do_search = args.phase != Phase::build;
  if (do_build && args.base_path.empty()) {
    usage(argv[0], "--base is required for the build phase");
  }
  if (do_search && (args.query_path.empty() || args.gt_path.empty())) {
    usage(argv[0], "--query and --gt are required for the search phase");
  }
  if (args.r == 0 || args.build_l < args.r || !std::isfinite(args.alpha) || args.alpha < 1.0F) {
    usage(argv[0], "build requires R>0, build-L>=R, and finite alpha>=1");
  }
  if (args.build_threads == 0 || args.build_threads > 48) {
    usage(argv[0], "--build-threads must be in [1,48]");
  }
  if (args.search_threads == 0 || args.beam_width == 0 || args.repetitions == 0) {
    usage(argv[0], "search threads, beam width, and repetitions must be non-zero");
  }
  if (args.search_ls.size() < 3) {
    usage(argv[0], "--search-l must contain at least three effort levels");
  }
  if (std::any_of(args.search_ls.begin(), args.search_ls.end(), [](std::uint32_t value) {
        return value < 100;
      })) {
    usage(argv[0], "all --search-l values must be >=100 for recall@100");
  }
  if (args.cache_ratio < 0.0 || !std::isfinite(args.cache_ratio)) {
    usage(argv[0], "--cache-ratio must be finite and non-negative");
  }
  return args;
}

auto seconds(Clock::time_point begin, Clock::time_point end) -> double {
  return std::chrono::duration<double>(end - begin).count();
}

auto read_fbin(const std::filesystem::path &path, std::uint64_t row_limit) -> FloatMatrix {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open fbin: " + path.string());
  }
  std::int32_t rows{};
  std::int32_t dim{};
  input.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!input || rows <= 0 || dim <= 0) {
    throw std::runtime_error("invalid fbin header: " + path.string());
  }
  const auto selected = row_limit == 0 ? static_cast<std::uint64_t>(rows)
                                       : std::min(row_limit, static_cast<std::uint64_t>(rows));
  if (selected > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error("fbin row count exceeds uint32 graph ids");
  }
  FloatMatrix result;
  result.rows = static_cast<std::uint32_t>(selected);
  result.source_rows = static_cast<std::uint32_t>(rows);
  result.dim = static_cast<std::uint32_t>(dim);
  result.values.resize(static_cast<std::size_t>(result.rows) * result.dim);
  input.read(reinterpret_cast<char *>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("truncated fbin payload: " + path.string());
  }
  return result;
}

auto read_ibin(const std::filesystem::path &path, std::uint32_t row_limit) -> IntMatrix {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open ibin: " + path.string());
  }
  std::int32_t rows{};
  std::int32_t dim{};
  input.read(reinterpret_cast<char *>(&rows), sizeof(rows));
  input.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!input || rows <= 0 || dim <= 0) {
    throw std::runtime_error("invalid ibin header: " + path.string());
  }
  const auto selected = row_limit == 0 ? static_cast<std::uint32_t>(rows)
                                       : std::min(row_limit, static_cast<std::uint32_t>(rows));
  IntMatrix result;
  result.rows = selected;
  result.dim = static_cast<std::uint32_t>(dim);
  result.values.resize(static_cast<std::size_t>(selected) * result.dim);
  input.read(reinterpret_cast<char *>(result.values.data()),
             static_cast<std::streamsize>(result.values.size() * sizeof(std::uint32_t)));
  if (!input) {
    throw std::runtime_error("truncated ibin payload: " + path.string());
  }
  return result;
}

inline void fnv_bytes(std::uint64_t &hash, const void *data, std::size_t size) {
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  const auto *bytes = static_cast<const std::uint8_t *>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= kPrime;
  }
}

template <typename T>
void fnv_value(std::uint64_t &hash, const T &value) {
  fnv_bytes(hash, std::addressof(value), sizeof(value));
}

auto topology_stats(const std::vector<std::vector<std::uint32_t>> &graph, std::uint32_t medoid)
    -> TopologyStats {
  TopologyStats result;
  result.digest = 14695981039346656037ULL;
  fnv_value(result.digest, medoid);
  const auto count = static_cast<std::uint64_t>(graph.size());
  fnv_value(result.digest, count);
  for (std::uint64_t node = 0; node < count; ++node) {
    const auto degree = static_cast<std::uint32_t>(graph[node].size());
    fnv_value(result.digest, node);
    fnv_value(result.digest, degree);
    if (!graph[node].empty()) {
      fnv_bytes(result.digest, graph[node].data(), graph[node].size() * sizeof(std::uint32_t));
    }
    result.edges += degree;
    result.zero_degree_nodes += static_cast<std::uint64_t>(degree == 0);
    result.max_degree = std::max(result.max_degree, degree);
  }
  result.mean_degree = graph.empty() ? 0.0 : static_cast<double>(result.edges) / graph.size();
  return result;
}

auto topology_stats(const std::filesystem::path &index_path) -> TopologyStats {
  const auto header = alaya::diskann::read_disk_layout_header(index_path.string());
  const auto geometry = DiskLayoutGeometry::compute(header.dim, header.max_degree);
  std::ifstream input(index_path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open disk layout for topology validation: " +
                             index_path.string());
  }
  input.seekg(static_cast<std::streamoff>(alaya::diskann::kSectorLen));
  std::vector<char> page(geometry.page_size);
  TopologyStats result;
  result.digest = 14695981039346656037ULL;
  fnv_value(result.digest, header.medoid);
  fnv_value(result.digest, header.num_points);
  std::uint64_t next_node = 0;
  for (std::uint64_t page_id = 0; page_id < geometry.num_pages(header.num_points); ++page_id) {
    input.read(page.data(), static_cast<std::streamsize>(page.size()));
    if (!input) {
      throw std::runtime_error("truncated disk layout while validating topology: " +
                               index_path.string());
    }
    for (std::uint64_t slot = 0; slot < geometry.nodes_per_sector && next_node < header.num_points;
         ++slot, ++next_node) {
      const char *record = page.data() + slot * geometry.node_len;
      std::uint32_t degree{};
      std::memcpy(&degree, record + header.dim * sizeof(float), sizeof(degree));
      if (degree > header.max_degree) {
        throw std::runtime_error("persisted node degree exceeds layout max_degree");
      }
      fnv_value(result.digest, next_node);
      fnv_value(result.digest, degree);
      const char *neighbors = record + header.dim * sizeof(float) + sizeof(degree);
      fnv_bytes(result.digest, neighbors, static_cast<std::size_t>(degree) * sizeof(std::uint32_t));
      for (std::uint32_t edge = 0; edge < degree; ++edge) {
        std::uint32_t neighbor{};
        std::memcpy(&neighbor,
                    neighbors + static_cast<std::size_t>(edge) * sizeof(neighbor),
                    sizeof(neighbor));
        if (neighbor >= header.num_points) {
          throw std::runtime_error("persisted neighbor id is out of range");
        }
      }
      result.edges += degree;
      result.zero_degree_nodes += static_cast<std::uint64_t>(degree == 0);
      result.max_degree = std::max(result.max_degree, degree);
    }
  }
  result.mean_degree =
      header.num_points == 0 ? 0.0 : static_cast<double>(result.edges) / header.num_points;
  return result;
}

auto file_digest(const std::filesystem::path &path) -> std::uint64_t {
  if (path.empty() || !std::filesystem::exists(path)) {
    return 0;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot hash file: " + path.string());
  }
  std::uint64_t result = 14695981039346656037ULL;
  std::array<char, 1U << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = input.gcount();
    if (count > 0) {
      fnv_bytes(result, buffer.data(), static_cast<std::size_t>(count));
    }
  }
  return result;
}

auto directory_bytes(const std::filesystem::path &directory) -> std::uint64_t {
  std::uint64_t result = 0;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      result += entry.file_size();
    }
  }
  return result;
}

template <typename T>
void write_value(std::ofstream &output, const T &value) {
  output.write(reinterpret_cast<const char *>(std::addressof(value)), sizeof(value));
}

void write_identity_ids(const std::filesystem::path &path, std::uint64_t n) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write ids: " + path.string());
  }
  write_value(output, n);
  constexpr std::uint64_t kBlock = 1U << 16;
  std::vector<std::uint64_t> labels(static_cast<std::size_t>(std::min(kBlock, n)));
  for (std::uint64_t begin = 0; begin < n; begin += labels.size()) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(labels.size(), n - begin));
    std::iota(labels.begin(), labels.begin() + static_cast<std::ptrdiff_t>(count), begin);
    output.write(reinterpret_cast<const char *>(labels.data()),
                 static_cast<std::streamsize>(count * sizeof(std::uint64_t)));
  }
  if (!output) {
    throw std::runtime_error("failed writing ids: " + path.string());
  }
}

void write_diskann_meta(const std::filesystem::path &path,
                        std::uint64_t n,
                        std::uint64_t dim,
                        std::uint32_t max_degree,
                        std::uint32_t medoid,
                        std::uint32_t pq_chunks) {
  const auto geometry = DiskLayoutGeometry::compute(dim, max_degree);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write meta: " + path.string());
  }
  const std::uint64_t magic = DiskANNIndex::kMetaMagic;
  const std::uint32_t version = DiskANNIndex::kMetaVersion;
  const std::uint8_t has_pq = pq_chunks == 0 ? 0 : 1;
  write_value(output, magic);
  write_value(output, version);
  write_value(output, n);
  write_value(output, dim);
  write_value(output, max_degree);
  write_value(output, medoid);
  write_value(output, has_pq);
  write_value(output, pq_chunks);
  write_value(output, geometry.node_len);
  write_value(output, geometry.nodes_per_sector);
  write_value(output, n);  // max_slot_id
  write_value(output, n);  // live_count
  if (!output) {
    throw std::runtime_error("failed writing meta: " + path.string());
  }
}

class StagingDirectoryGuard {
 public:
  explicit StagingDirectoryGuard(std::filesystem::path path) : path_(std::move(path)) {}
  ~StagingDirectoryGuard() {
    if (armed_) {
      std::error_code error;
      std::filesystem::remove_all(path_, error);
    }
  }
  void disarm() noexcept { armed_ = false; }

 private:
  std::filesystem::path path_;
  bool armed_{true};
};

auto materialize_diskann(const std::filesystem::path &target,
                         const FloatMatrix &base,
                         const std::vector<std::vector<std::uint32_t>> &graph,
                         std::uint32_t medoid,
                         const Args &args) -> MaterializeMetrics {
  const auto total_begin = Clock::now();
  if (std::filesystem::exists(target)) {
    throw std::runtime_error("materialized target already exists: " + target.string());
  }
  const auto staging = target.parent_path() / ("." + target.filename().string() + ".tmp." +
                                               std::to_string(alaya::platform::get_pid()));
  std::filesystem::remove_all(staging);
  std::filesystem::create_directories(staging);
  StagingDirectoryGuard guard(staging);

  MaterializeMetrics metrics;

  auto begin = Clock::now();
  alaya::diskann::write_disk_layout((staging / "diskann.index").string(),
                                    base.values.data(),
                                    graph,
                                    {base.rows, base.dim, args.r, medoid});
  auto end = Clock::now();
  metrics.layout_s = seconds(begin, end);

  begin = Clock::now();
  write_identity_ids(staging / "ids.bin", base.rows);
  end = Clock::now();
  metrics.ids_s = seconds(begin, end);

  begin = Clock::now();
  if (args.pq_chunks > 0) {
    PQTable pq;
    pq.train(base.values.data(),
             base.rows,
             base.dim,
             args.pq_chunks,
             args.pq_train_iters,
             args.seed,
             args.build_threads);
    pq.encode(base.values.data(), base.rows, args.build_threads);
    pq.save((staging / "pq_pivots.bin").string(), (staging / "pq_compressed.bin").string());
  }
  end = Clock::now();
  metrics.pq_s = seconds(begin, end);

  begin = Clock::now();
  NodeCache cache;
  cache.generate(graph, base.values.data(), medoid, base.rows, base.dim, args.r, args.cache_ratio);
  cache.save((staging / "cache_ids.bin").string(), (staging / "cache_nodes.bin").string());
  end = Clock::now();
  metrics.cache_s = seconds(begin, end);

  begin = Clock::now();
  write_diskann_meta(staging / "meta.bin", base.rows, base.dim, args.r, medoid, args.pq_chunks);
  end = Clock::now();
  metrics.meta_s = seconds(begin, end);

  std::filesystem::rename(staging, target);
  guard.disarm();
  metrics.total_s = seconds(total_begin, Clock::now());
  return metrics;
}

auto hex_digest(std::uint64_t value) -> std::string {
  std::ostringstream output;
  output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

void write_build_metrics(const std::filesystem::path &path,
                         const BuildMetrics &metrics,
                         const Args &args) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write build metrics: " + path.string());
  }
  output << std::setprecision(12);
  output << "metric,value\n";
  output << "n," << metrics.n << "\n";
  output << "dim," << metrics.dim << "\n";
  output << "R," << args.r << "\n";
  output << "build_L," << args.build_l << "\n";
  output << "alpha," << args.alpha << "\n";
  output << "build_threads," << args.build_threads << "\n";
  output << "pq_chunks," << args.pq_chunks << "\n";
  output << "pq_train_iters," << args.pq_train_iters << "\n";
  output << "cache_ratio," << args.cache_ratio << "\n";
  output << "seed," << args.seed << "\n";
  output << "active_graph_build_s," << metrics.active_graph_build_s << "\n";
  output << "freeze_snapshot_s," << metrics.freeze_snapshot_s << "\n";
  output << "materialize_total_s," << metrics.materialize.total_s << "\n";
  output << "materialize_layout_s," << metrics.materialize.layout_s << "\n";
  output << "materialize_ids_s," << metrics.materialize.ids_s << "\n";
  output << "materialize_pq_s," << metrics.materialize.pq_s << "\n";
  output << "materialize_cache_s," << metrics.materialize.cache_s << "\n";
  output << "materialize_meta_s," << metrics.materialize.meta_s << "\n";
  output << "freeze_plus_materialize_s," << metrics.freeze_snapshot_s + metrics.materialize.total_s
         << "\n";
  output << "topology_verify_s," << metrics.topology_verify_s << "\n";
  output << "rebuilt_total_s," << metrics.rebuilt_total_s << "\n";
  output << "seal_speedup,"
         << metrics.rebuilt_total_s / (metrics.freeze_snapshot_s + metrics.materialize.total_s)
         << "\n";
  output << "frozen_topology_digest," << hex_digest(metrics.frozen.digest) << "\n";
  output << "materialized_topology_digest," << hex_digest(metrics.materialized.digest) << "\n";
  output << "rebuilt_topology_digest," << hex_digest(metrics.rebuilt.digest) << "\n";
  output << "frozen_edges," << metrics.frozen.edges << "\n";
  output << "materialized_edges," << metrics.materialized.edges << "\n";
  output << "rebuilt_edges," << metrics.rebuilt.edges << "\n";
  output << "frozen_mean_degree," << metrics.frozen.mean_degree << "\n";
  output << "materialized_mean_degree," << metrics.materialized.mean_degree << "\n";
  output << "rebuilt_mean_degree," << metrics.rebuilt.mean_degree << "\n";
  output << "frozen_zero_degree_nodes," << metrics.frozen.zero_degree_nodes << "\n";
  output << "materialized_zero_degree_nodes," << metrics.materialized.zero_degree_nodes << "\n";
  output << "rebuilt_zero_degree_nodes," << metrics.rebuilt.zero_degree_nodes << "\n";
  output << "materialized_bytes," << metrics.materialized_bytes << "\n";
  output << "rebuilt_bytes," << metrics.rebuilt_bytes << "\n";
  output << "materialized_pq_pivots_digest," << hex_digest(metrics.materialized_pq_pivots_digest)
         << "\n";
  output << "rebuilt_pq_pivots_digest," << hex_digest(metrics.rebuilt_pq_pivots_digest) << "\n";
  output << "materialized_pq_codes_digest," << hex_digest(metrics.materialized_pq_codes_digest)
         << "\n";
  output << "rebuilt_pq_codes_digest," << hex_digest(metrics.rebuilt_pq_codes_digest) << "\n";
}

auto identity_labels(std::uint64_t n) -> std::vector<std::uint64_t> {
  std::vector<std::uint64_t> result(n);
  std::iota(result.begin(), result.end(), std::uint64_t{0});
  return result;
}

void run_build(const Args &args) {
  const auto materialized_dir = args.output_dir / "materialized";
  const auto rebuilt_dir = args.output_dir / "rebuilt";
  if (args.rebuild) {
    std::filesystem::remove_all(materialized_dir);
    std::filesystem::remove_all(rebuilt_dir);
    std::filesystem::remove(args.output_dir / "build_metrics.csv");
  }
  std::filesystem::create_directories(args.output_dir);
  if (std::filesystem::exists(materialized_dir) || std::filesystem::exists(rebuilt_dir)) {
    throw std::runtime_error("arm directory already exists; pass --rebuild or use --phase search");
  }

  std::cout << "[build] loading base " << args.base_path << "\n";
  FloatMatrix base = read_fbin(args.base_path, args.n);
  if (args.pq_chunks > 0 && base.dim % args.pq_chunks != 0) {
    throw std::runtime_error("base dimension must be divisible by --pq-chunks");
  }
  std::cout << "[build] base=" << base.rows << "x" << base.dim
            << " (source rows=" << base.source_rows << ") R=" << args.r << " L=" << args.build_l
            << " alpha=" << args.alpha << " threads=" << args.build_threads << "\n";

  BuildMetrics metrics;
  metrics.n = base.rows;
  metrics.dim = base.dim;

  alaya::vamana::VamanaBuildParams params;
  params.R = args.r;
  params.L = args.build_l;
  params.alpha = args.alpha;
  params.num_threads = args.build_threads;
  params.seed = args.seed;
  auto builder = std::make_unique<alaya::vamana::VamanaBuilder>(base.values.data(),
                                                                base.rows,
                                                                base.dim,
                                                                params);

  std::cout << "[build] constructing active-memory Vamana topology (excluded from seal time)\n";
  auto begin = Clock::now();
  builder->build();
  auto end = Clock::now();
  metrics.active_graph_build_s = seconds(begin, end);

  begin = Clock::now();
  const auto *frozen_graph_view = std::addressof(builder->graph());
  const auto frozen_medoid = builder->medoid();
  end = Clock::now();
  metrics.freeze_snapshot_s = seconds(begin, end);
  const auto &frozen_graph = *frozen_graph_view;
  metrics.frozen = topology_stats(frozen_graph, frozen_medoid);

  std::cout << "[build] materializing frozen topology -> " << materialized_dir << "\n";
  metrics.materialize =
      materialize_diskann(materialized_dir, base, frozen_graph, frozen_medoid, args);

  begin = Clock::now();
  metrics.materialized = topology_stats(materialized_dir / "diskann.index");
  end = Clock::now();
  metrics.topology_verify_s = seconds(begin, end);
  if (metrics.frozen.digest != metrics.materialized.digest ||
      metrics.frozen.edges != metrics.materialized.edges) {
    throw std::runtime_error("topology-preserving materialization verification failed");
  }
  metrics.materialized_bytes = directory_bytes(materialized_dir);
  metrics.materialized_pq_pivots_digest = file_digest(materialized_dir / "pq_pivots.bin");
  metrics.materialized_pq_codes_digest = file_digest(materialized_dir / "pq_compressed.bin");
  write_build_metrics(args.output_dir / "build_metrics.csv", metrics, args);

  // Drop the active graph before the independent baseline starts.  This keeps
  // peak memory bounded and makes it impossible for the rebuilt arm to reuse
  // the frozen adjacency accidentally.
  builder.reset();

  std::cout << "[build] independently rebuilding DiskANN -> " << rebuilt_dir << "\n";
  auto labels = identity_labels(base.rows);
  DiskANNBuildParams baseline_params;
  baseline_params.R = args.r;
  baseline_params.L = args.build_l;
  baseline_params.alpha = args.alpha;
  baseline_params.pq_n_chunks = args.pq_chunks;
  baseline_params.cache_ratio = args.cache_ratio;
  baseline_params.num_threads = args.build_threads;
  baseline_params.pq_train_iters = args.pq_train_iters;
  baseline_params.seed = args.seed;
  baseline_params.verbose = true;
  begin = Clock::now();
  DiskANNIndex::build(rebuilt_dir.string(),
                      base.values.data(),
                      labels.data(),
                      base.rows,
                      base.dim,
                      baseline_params);
  end = Clock::now();
  metrics.rebuilt_total_s = seconds(begin, end);
  metrics.rebuilt = topology_stats(rebuilt_dir / "diskann.index");
  metrics.rebuilt_bytes = directory_bytes(rebuilt_dir);
  metrics.rebuilt_pq_pivots_digest = file_digest(rebuilt_dir / "pq_pivots.bin");
  metrics.rebuilt_pq_codes_digest = file_digest(rebuilt_dir / "pq_compressed.bin");
  write_build_metrics(args.output_dir / "build_metrics.csv", metrics, args);

  std::cout << std::fixed << std::setprecision(3)
            << "[build] active graph=" << metrics.active_graph_build_s
            << " s, freeze+materialize=" << metrics.freeze_snapshot_s + metrics.materialize.total_s
            << " s, rebuild=" << metrics.rebuilt_total_s << " s, speedup="
            << metrics.rebuilt_total_s / (metrics.freeze_snapshot_s + metrics.materialize.total_s)
            << "x\n"
            << "[build] frozen/materialized digest=" << hex_digest(metrics.frozen.digest)
            << ", rebuilt digest=" << hex_digest(metrics.rebuilt.digest) << "\n";
}

auto median(std::vector<double> values) -> double {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto middle = values.size() / 2;
  return values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
}

auto recall_at(const std::vector<std::uint64_t> &labels,
               const IntMatrix &ground_truth,
               std::uint32_t k) -> double {
  std::uint64_t matches = 0;
  constexpr std::uint32_t kOutputWidth = 100;
  for (std::uint32_t query = 0; query < ground_truth.rows; ++query) {
    const auto *truth = ground_truth.row(query);
    const auto *found = labels.data() + static_cast<std::size_t>(query) * kOutputWidth;
    for (std::uint32_t rank = 0; rank < k; ++rank) {
      if (found[rank] == DiskANNIndex::kNoLabel ||
          found[rank] > std::numeric_limits<std::uint32_t>::max()) {
        continue;
      }
      const auto id = static_cast<std::uint32_t>(found[rank]);
      matches += static_cast<std::uint64_t>(std::find(truth, truth + k, id) != truth + k);
    }
  }
  return static_cast<double>(matches) /
         (static_cast<double>(ground_truth.rows) * static_cast<double>(k));
}

auto evaluate_arm(const std::string &arm,
                  const std::filesystem::path &directory,
                  const FloatMatrix &queries,
                  const IntMatrix &ground_truth,
                  const Args &args) -> std::vector<SearchMetric> {
  const auto header =
      alaya::diskann::read_disk_layout_header((directory / "diskann.index").string());
  if (header.dim != queries.dim) {
    throw std::runtime_error(arm + " index/query dimension mismatch");
  }
  DiskANNLoadParams load_params;
  load_params.num_threads = args.search_threads;
  load_params.beam_width = args.beam_width;
  load_params.scratch_search_list_size =
      *std::max_element(args.search_ls.begin(), args.search_ls.end());

  std::cout << "[search] opening " << arm << " arm " << directory << "\n";
  DiskANNIndex index;
  index.load(directory.string(), load_params);
  constexpr std::uint32_t kTopK = 100;
  std::vector<std::uint64_t> labels(static_cast<std::size_t>(queries.rows) * kTopK);
  std::vector<float> distances(static_cast<std::size_t>(queries.rows) * kTopK);

  DiskANNSearchParams search_params;
  search_params.search_list_size = *std::max_element(args.search_ls.begin(), args.search_ls.end());
  search_params.use_pq = args.pq_chunks > 0;
  search_params.rerank = args.pq_chunks > 0;
  search_params.rerank_count = search_params.search_list_size;
  search_params.deterministic = args.deterministic;
  const auto warmup = std::min(args.warmup_queries, queries.rows);
  if (warmup > 0) {
    index.batch_search(queries.values.data(),
                       warmup,
                       kTopK,
                       labels.data(),
                       distances.data(),
                       args.search_threads,
                       search_params);
  }

  std::vector<SearchMetric> result;
  for (const auto search_l : args.search_ls) {
    SearchMetric metric;
    metric.arm = arm;
    metric.search_l = search_l;
    std::vector<double> recall10_samples;
    std::vector<double> recall100_samples;
    search_params.search_list_size = search_l;
    search_params.rerank_count = args.pq_chunks > 0 ? search_l : 0;
    for (std::uint32_t repetition = 0; repetition < args.repetitions; ++repetition) {
      const auto begin = Clock::now();
      index.batch_search(queries.values.data(),
                         queries.rows,
                         kTopK,
                         labels.data(),
                         distances.data(),
                         args.search_threads,
                         search_params);
      const auto elapsed_s = seconds(begin, Clock::now());
      metric.qps_samples.push_back(static_cast<double>(queries.rows) / elapsed_s);
      recall10_samples.push_back(recall_at(labels, ground_truth, 10));
      recall100_samples.push_back(recall_at(labels, ground_truth, 100));
      std::cout << std::fixed << std::setprecision(3) << "[search] " << arm << " L=" << search_l
                << " rep=" << (repetition + 1) << "/" << args.repetitions
                << " recall@10=" << recall10_samples.back()
                << " recall@100=" << recall100_samples.back()
                << " QPS=" << metric.qps_samples.back() << "\n";
    }
    metric.recall_at_10 = median(std::move(recall10_samples));
    metric.recall_at_100 = median(std::move(recall100_samples));
    metric.qps = median(metric.qps_samples);
    result.push_back(std::move(metric));
  }
  return result;
}

void write_search_metrics(const std::filesystem::path &path,
                          std::span<const SearchMetric> metrics,
                          const Args &args,
                          std::uint32_t nq) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write search metrics: " + path.string());
  }
  output << std::setprecision(12);
  output << "arm,nq,search_threads,beam_width,deterministic,L,recall_at_10,recall_at_100,"
            "median_qps,qps_samples\n";
  for (const auto &metric : metrics) {
    output << metric.arm << ',' << nq << ',' << args.search_threads << ',' << args.beam_width << ','
           << static_cast<int>(args.deterministic) << ',' << metric.search_l << ','
           << metric.recall_at_10 << ',' << metric.recall_at_100 << ',' << metric.qps << ',';
    for (std::size_t i = 0; i < metric.qps_samples.size(); ++i) {
      output << (i == 0 ? "" : ";") << metric.qps_samples[i];
    }
    output << '\n';
  }
}

void run_search(const Args &args) {
  const auto materialized_dir = args.output_dir / "materialized";
  const auto rebuilt_dir = args.output_dir / "rebuilt";
  if (!std::filesystem::exists(materialized_dir) || !std::filesystem::exists(rebuilt_dir)) {
    throw std::runtime_error("both arm directories are required; run --phase build first");
  }
  std::cout << "[search] loading queries and exact ground truth\n";
  FloatMatrix queries = read_fbin(args.query_path, args.nq);
  IntMatrix ground_truth = read_ibin(args.gt_path, queries.rows);
  if (queries.rows != ground_truth.rows || ground_truth.dim < 100) {
    throw std::runtime_error("query/ground-truth shape mismatch or GT width <100");
  }

  std::vector<SearchMetric> metrics =
      evaluate_arm("materialized", materialized_dir, queries, ground_truth, args);
  // evaluate_arm owns and destroys its DiskANNIndex before returning, so only
  // one AIO-heavy reader exists when the rebuilt arm opens.
  auto rebuilt = evaluate_arm("rebuilt", rebuilt_dir, queries, ground_truth, args);
  metrics.insert(metrics.end(), rebuilt.begin(), rebuilt.end());
  write_search_metrics(args.output_dir / "search_metrics.csv", metrics, args, queries.rows);

  std::cout << "\n[comparison] median results\n"
            << "L,mat_r@10,rebuild_r@10,delta_pp,mat_r@100,rebuild_r@100,delta_pp,mat_qps,"
               "rebuild_qps,qps_ratio\n";
  for (std::size_t i = 0; i < args.search_ls.size(); ++i) {
    const auto &mat = metrics[i];
    const auto &reb = metrics[i + args.search_ls.size()];
    std::cout << mat.search_l << ',' << mat.recall_at_10 << ',' << reb.recall_at_10 << ','
              << (mat.recall_at_10 - reb.recall_at_10) * 100.0 << ',' << mat.recall_at_100 << ','
              << reb.recall_at_100 << ',' << (mat.recall_at_100 - reb.recall_at_100) * 100.0 << ','
              << mat.qps << ',' << reb.qps << ',' << mat.qps / reb.qps << '\n';
  }
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    const auto args = parse_args(argc, argv);
    if (args.phase != Phase::search) {
      run_build(args);
    }
    if (args.phase != Phase::build) {
      run_search(args);
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[topology-seal-poc] ERROR: " << error.what() << '\n';
    return 1;
  }
}
