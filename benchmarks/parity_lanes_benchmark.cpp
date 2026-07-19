// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
  #include <unistd.h>
#endif

#include "core/any_segment.hpp"
#include "index/disk/laser_segment.hpp"
#include "index/disk/laser_segment_importer.hpp"
#include "index/graph/qg/qg_segment.hpp"
#include "simd/cpu_features.hpp"
#include "simd/laser_dispatch.hpp"
#include "space/quant/rabitq/dispatch.hpp"
#include "space/rabitq_space.hpp"
#include "utils/memory.hpp"
#include "utils/openmp.hpp"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using MemorySpace = alaya::RaBitQSpace<>;
using MemorySegment = alaya::QgSegment<MemorySpace>;

constexpr std::size_t kLargeAlignment = 2U * 1024U * 1024U;

struct Args {
  fs::path query_path{};
  fs::path ground_truth_path{};
  fs::path laser_segment_directory{};
  fs::path memqg_index_path{};
  fs::path base_path{};
  fs::path laser_native_prefix{};
  fs::path output_prefix{};
  std::string dataset{"sift1m-128d"};
  std::vector<std::uint32_t> lanes{1, 4, 16};
  std::vector<std::uint32_t> laser_efs{40, 60, 100, 200};
  std::vector<std::uint32_t> memqg_efs{40, 60, 100, 200};
  std::vector<int> cpus{};
  std::uint32_t top_k{10};
  std::uint32_t repeats{3};
  std::uint32_t warmup_rounds{1};
  std::uint32_t warmup_queries{};
  std::uint32_t query_limit{};
  std::uint32_t beam_width{16};
  std::uint32_t build_threads{64};
  std::uint32_t ef_build{100};
  std::uint32_t laser_degree{32};
  double minimum_measure_seconds{};
  bool prepare_only{};
  bool force_rebuild_memqg{};
  bool laser_import_copy{};
};

[[nodiscard]] auto parse_u32(std::string_view value, std::string_view option) -> std::uint32_t {
  std::size_t consumed{};
  const auto parsed = std::stoull(std::string(value), &consumed);
  if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(option) + " is not a uint32: " + std::string(value));
  }
  return static_cast<std::uint32_t>(parsed);
}

[[nodiscard]] auto parse_int(std::string_view value, std::string_view option) -> int {
  std::size_t consumed{};
  const auto parsed = std::stoll(std::string(value), &consumed);
  if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(option) +
                                " is not a non-negative int: " + std::string(value));
  }
  return static_cast<int>(parsed);
}

[[nodiscard]] auto parse_double(std::string_view value, std::string_view option) -> double {
  std::size_t consumed{};
  const auto parsed = std::stod(std::string(value), &consumed);
  if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0) {
    throw std::invalid_argument(std::string(option) +
                                " is not a non-negative finite number: " + std::string(value));
  }
  return parsed;
}

template <typename Value, typename Parse>
[[nodiscard]] auto parse_list(std::string_view text, std::string_view option, Parse parse)
    -> std::vector<Value> {
  if (text.empty()) {
    throw std::invalid_argument(std::string(option) + " list is empty");
  }
  std::vector<Value> values;
  std::size_t begin{};
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    const auto token =
        text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin);
    if (token.empty()) {
      throw std::invalid_argument(std::string(option) + " contains an empty item");
    }
    values.push_back(parse(token, option));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return values;
}

[[nodiscard]] auto parse_cpu_list(std::string_view text) -> std::vector<int> {
  std::vector<int> cpus;
  std::size_t begin{};
  while (begin < text.size()) {
    const auto end = text.find(',', begin);
    const auto token =
        text.substr(begin, end == std::string_view::npos ? text.size() - begin : end - begin);
    const auto dash = token.find('-');
    if (dash == std::string_view::npos) {
      cpus.push_back(parse_int(token, "--cpu-list"));
    } else {
      const auto first = parse_int(token.substr(0, dash), "--cpu-list");
      const auto last = parse_int(token.substr(dash + 1), "--cpu-list");
      if (last < first) {
        throw std::invalid_argument("--cpu-list range is descending: " + std::string(token));
      }
      for (int cpu = first; cpu <= last; ++cpu) {
        cpus.push_back(cpu);
      }
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  if (cpus.empty()) {
    throw std::invalid_argument("--cpu-list is empty");
  }
  std::set<int> unique(cpus.begin(), cpus.end());
  if (unique.size() != cpus.size()) {
    throw std::invalid_argument("--cpu-list contains duplicate CPUs");
  }
  return cpus;
}

[[noreturn]] void print_usage_and_exit() {
  std::cout << "usage: parity_lanes_benchmark [options]\n\n"
               "Required measurement paths:\n"
               "  --query PATH                 fbin query matrix\n"
               "  --gt PATH                    ibin exact ground truth\n"
               "  --laser-segment PATH         importer-created resident_arena segment directory\n"
               "  --memqg-index PATH           current memory-QG artifact\n"
               "  --output-prefix PATH         writes PATH.csv and PATH.json\n\n"
               "Grid and protocol:\n"
               "  --lanes LIST                 default 1,4,16\n"
               "  --laser-efs LIST             default 40,60,100,200\n"
               "  --memqg-efs LIST             default 40,60,100,200\n"
               "  --topk N                     default 10\n"
               "  --repeats N                  per A/B order, default 3\n"
               "  --warmup-rounds N            untimed rounds before every point, default 1\n"
               "  --warmup-queries N           0 means all measured queries, default 0\n"
               "  --query-limit N              0 means all queries, default 0\n"
               "  --min-measure-seconds S      repeat full passes to this duration, default 0\n"
               "  --beam N                     LASER per-call beam, default 16\n"
               "  --cpu-list LIST              CPU ids/ranges; default is current allowed set\n"
               "  --dataset NAME               metadata label, default sift1m-128d\n\n"
               "Optional one-time preparation (excluded from measurement):\n"
               "  --base PATH                  fbin base used if MemQG artifact must be built\n"
               "  --build-threads N            default 64\n"
               "  --ef-build N                 default 100\n"
               "  --force-rebuild-memqg        replace the requested MemQG artifact\n"
               "  --laser-native-prefix PATH   native prefix, without _R*_MD*.index\n"
               "  --laser-degree N             default 32\n"
               "  --laser-import-copy          copy native files when hard links are unavailable\n"
               "  --prepare-only               prepare artifacts, then exit\n";
  std::exit(0);
}

[[nodiscard]] auto parse_args(int argc, char **argv) -> Args {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help" || option == "-h") {
      print_usage_and_exit();
    }
    if (option == "--prepare-only") {
      args.prepare_only = true;
      continue;
    }
    if (option == "--force-rebuild-memqg") {
      args.force_rebuild_memqg = true;
      continue;
    }
    if (option == "--laser-import-copy") {
      args.laser_import_copy = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + std::string(option));
    }
    const std::string_view value(argv[++index]);
    if (option == "--query") {
      args.query_path = value;
    } else if (option == "--gt") {
      args.ground_truth_path = value;
    } else if (option == "--laser-segment") {
      args.laser_segment_directory = value;
    } else if (option == "--memqg-index") {
      args.memqg_index_path = value;
    } else if (option == "--base") {
      args.base_path = value;
    } else if (option == "--laser-native-prefix") {
      args.laser_native_prefix = value;
    } else if (option == "--output-prefix") {
      args.output_prefix = value;
    } else if (option == "--dataset") {
      args.dataset = value;
    } else if (option == "--lanes") {
      args.lanes = parse_list<std::uint32_t>(value, option, parse_u32);
    } else if (option == "--laser-efs") {
      args.laser_efs = parse_list<std::uint32_t>(value, option, parse_u32);
    } else if (option == "--memqg-efs") {
      args.memqg_efs = parse_list<std::uint32_t>(value, option, parse_u32);
    } else if (option == "--cpu-list") {
      args.cpus = parse_cpu_list(value);
    } else if (option == "--topk") {
      args.top_k = parse_u32(value, option);
    } else if (option == "--repeats") {
      args.repeats = parse_u32(value, option);
    } else if (option == "--warmup-rounds") {
      args.warmup_rounds = parse_u32(value, option);
    } else if (option == "--warmup-queries") {
      args.warmup_queries = parse_u32(value, option);
    } else if (option == "--query-limit") {
      args.query_limit = parse_u32(value, option);
    } else if (option == "--min-measure-seconds") {
      args.minimum_measure_seconds = parse_double(value, option);
    } else if (option == "--beam") {
      args.beam_width = parse_u32(value, option);
    } else if (option == "--build-threads") {
      args.build_threads = parse_u32(value, option);
    } else if (option == "--ef-build") {
      args.ef_build = parse_u32(value, option);
    } else if (option == "--laser-degree") {
      args.laser_degree = parse_u32(value, option);
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }

  if (args.query_path.empty() || args.laser_segment_directory.empty() ||
      args.memqg_index_path.empty()) {
    throw std::invalid_argument("--query, --laser-segment, and --memqg-index are required");
  }
  if (!args.prepare_only && (args.ground_truth_path.empty() || args.output_prefix.empty())) {
    throw std::invalid_argument("measurement also requires --gt and --output-prefix");
  }
  if (args.top_k == 0 || args.repeats == 0 || args.beam_width == 0 || args.build_threads == 0 ||
      args.ef_build == 0 || args.laser_degree == 0 || args.lanes.empty() ||
      args.laser_efs.empty() || args.memqg_efs.empty()) {
    throw std::invalid_argument(
        "topk/repeats/beam/build values and all grid lists must be nonzero");
  }
  if (args.build_threads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("--build-threads exceeds int");
  }
  for (const auto lanes : args.lanes) {
    if (lanes == 0) {
      throw std::invalid_argument("--lanes entries must be nonzero");
    }
  }
  for (const auto ef : args.laser_efs) {
    if (ef < args.top_k) {
      throw std::invalid_argument("every LASER ef must be >= topk");
    }
  }
  for (const auto ef : args.memqg_efs) {
    if (ef < args.top_k) {
      throw std::invalid_argument("every MemQG ef must be >= topk");
    }
  }
  return args;
}

template <typename T>
struct Matrix {
  static_assert(std::is_trivially_copyable_v<T>);
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::vector<T, alaya::AlignedAlloc<T>> values{};

  [[nodiscard]] auto row(std::size_t index) const -> const T * {
    return values.data() + index * columns;
  }
};

template <typename T>
[[nodiscard]] auto read_matrix(const fs::path &path, std::uint32_t row_limit = 0) -> Matrix<T> {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open matrix " + path.string());
  }
  std::array<std::int32_t, 2> header{};
  input.read(reinterpret_cast<char *>(header.data()), sizeof(header));
  if (!input || header[0] <= 0 || header[1] <= 0) {
    throw std::runtime_error("invalid matrix header in " + path.string());
  }
  const auto file_rows = static_cast<std::uint64_t>(header[0]);
  const auto columns = static_cast<std::uint64_t>(header[1]);
  if (columns != 0 && file_rows > std::numeric_limits<std::uint64_t>::max() / columns) {
    throw std::runtime_error("matrix element count overflows in " + path.string());
  }
  const auto payload_elements = file_rows * columns;
  if (payload_elements > (std::numeric_limits<std::uint64_t>::max() - sizeof(header)) / sizeof(T)) {
    throw std::runtime_error("matrix byte size overflows in " + path.string());
  }
  const auto expected_bytes = sizeof(header) + payload_elements * sizeof(T);
  if (fs::file_size(path) != expected_bytes) {
    throw std::runtime_error("matrix file size disagrees with header in " + path.string());
  }
  const auto rows = row_limit == 0 ? file_rows : std::min<std::uint64_t>(file_rows, row_limit);
  if (rows > std::numeric_limits<std::uint32_t>::max() ||
      columns > std::numeric_limits<std::uint32_t>::max() ||
      rows * columns > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("matrix dimensions are not representable in " + path.string());
  }
  Matrix<T> matrix;
  matrix.rows = static_cast<std::uint32_t>(rows);
  matrix.columns = static_cast<std::uint32_t>(columns);
  matrix.values.resize(static_cast<std::size_t>(rows * columns));
  input.read(reinterpret_cast<char *>(matrix.values.data()),
             static_cast<std::streamsize>(matrix.values.size() * sizeof(T)));
  if (!input) {
    throw std::runtime_error("short matrix payload in " + path.string());
  }
  return matrix;
}

[[nodiscard]] auto read_first_line(const fs::path &path) -> std::string {
  std::ifstream input(path);
  std::string line;
  if (input) {
    std::getline(input, line);
  }
  return line;
}

[[nodiscard]] auto hostname() -> std::string {
#if defined(__linux__)
  std::array<char, 256> name{};
  if (::gethostname(name.data(), name.size()) == 0) {
    name.back() = '\0';
    return name.data();
  }
#endif
  return "unknown";
}

[[nodiscard]] auto allowed_cpus() -> std::vector<int> {
#if defined(__linux__)
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (::sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    throw std::runtime_error("sched_getaffinity failed: " + std::string(std::strerror(errno)));
  }
  std::vector<int> cpus;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      cpus.push_back(cpu);
    }
  }
  if (cpus.empty()) {
    throw std::runtime_error("process affinity contains no CPUs");
  }
  return cpus;
#else
  throw std::runtime_error("parity_lanes_benchmark requires Linux thread affinity");
#endif
}

[[nodiscard]] auto join_ints(std::span<const int> values) -> std::string {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

template <typename T>
[[nodiscard]] auto join_numbers(std::span<const T> values) -> std::string {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

void create_link_or_copy(const fs::path &source, const fs::path &target, bool copy) {
  if (!fs::is_regular_file(source)) {
    throw std::runtime_error("required native LASER artifact is missing: " + source.string());
  }
  if (fs::exists(target)) {
    std::error_code equivalent_error;
    if (fs::equivalent(source, target, equivalent_error) && !equivalent_error) {
      return;
    }
    throw std::runtime_error("staging target already exists and differs: " + target.string());
  }
  std::error_code error;
  if (copy) {
    fs::copy_file(source, target, fs::copy_options::none, error);
  } else {
    fs::create_hard_link(source, target, error);
  }
  if (error) {
    throw std::runtime_error(std::string(copy ? "copy" : "hard-link") + " failed: " +
                             source.string() + " -> " + target.string() + ": " + error.message());
  }
}

[[nodiscard]] auto native_laser_index_path(const Args &args, std::uint32_t dim) -> fs::path {
  return fs::path(args.laser_native_prefix.string() + "_R" + std::to_string(args.laser_degree) +
                  "_MD" + std::to_string(dim) + ".index");
}

[[nodiscard]] auto read_native_laser_count(const fs::path &index_path) -> std::uint64_t {
  std::ifstream input(index_path, std::ios::binary);
  std::uint64_t count{};
  input.read(reinterpret_cast<char *>(&count), sizeof(count));
  if (!input || count == 0) {
    throw std::runtime_error("cannot read LASER row count from " + index_path.string());
  }
  return count;
}

void prepare_laser_segment(const Args &args, std::uint32_t dim) {
  if (fs::is_regular_file(args.laser_segment_directory / "manifest.txt")) {
    return;
  }
  if (args.laser_native_prefix.empty()) {
    throw std::runtime_error("LASER segment is absent; provide --laser-native-prefix to import it");
  }
  const auto segment_parent = args.laser_segment_directory.parent_path();
  fs::create_directories(segment_parent);
  const auto segment_id = args.laser_segment_directory.filename().string();
  const auto staging =
      segment_parent.parent_path() /
      ("native-import-" + segment_id + "-" +
       std::to_string(static_cast<unsigned long long>(Clock::now().time_since_epoch().count())));
  fs::create_directories(staging);

  const auto native_index = native_laser_index_path(args, dim);
  const auto expected_prefix = "dsqg_" + segment_id;
  const auto expected_index = expected_prefix + "_R" + std::to_string(args.laser_degree) + "_MD" +
                              std::to_string(dim) + ".index";
  create_link_or_copy(native_index, staging / expected_index, args.laser_import_copy);
  for (const std::string_view suffix : {"_rotator", "_cache_ids", "_cache_nodes"}) {
    create_link_or_copy(fs::path(native_index.string() + std::string(suffix)),
                        fs::path((staging / expected_index).string() + std::string(suffix)),
                        args.laser_import_copy);
  }
  for (const std::string_view suffix : {"_medoids", "_medoids_indices", "_pca.bin"}) {
    const auto source = fs::path(args.laser_native_prefix.string() + std::string(suffix));
    if (fs::is_regular_file(source)) {
      create_link_or_copy(source,
                          staging / (expected_prefix + std::string(suffix)),
                          args.laser_import_copy);
    }
  }

  const auto count = read_native_laser_count(native_index);
  if (count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("LASER label count does not fit size_t");
  }
  std::vector<std::uint64_t, alaya::AlignedAlloc<std::uint64_t>> labels(
      static_cast<std::size_t>(count));
  std::iota(labels.begin(), labels.end(), std::uint64_t{0});
  alaya::disk::LaserSegmentImportParams params;
  params.R = args.laser_degree;
  params.main_dim = dim;
  params.default_ef = args.laser_efs.front();
  params.default_beam_width = args.beam_width;
  params.copy_files = false;
  params.residency = "resident_arena";
  alaya::disk::LaserSegmentImporter importer(dim, alaya::core::Metric::l2, params);
  (void)importer.import_from(staging,
                             labels.data(),
                             static_cast<std::uint64_t>(labels.size()),
                             args.laser_segment_directory);
  fs::remove_all(staging);
}

void build_memqg(const Args &args, std::uint32_t expected_dim) {
  if (fs::is_regular_file(args.memqg_index_path) && !args.force_rebuild_memqg) {
    return;
  }
  if (args.base_path.empty()) {
    throw std::runtime_error("MemQG artifact is absent; provide --base to build it");
  }
  if (args.force_rebuild_memqg) {
    std::error_code error;
    fs::remove(args.memqg_index_path, error);
    if (error) {
      throw std::runtime_error("cannot remove old MemQG artifact: " + error.message());
    }
  }
  fs::create_directories(args.memqg_index_path.parent_path());
  auto base = read_matrix<float>(args.base_path);
  if (base.columns != expected_dim) {
    throw std::runtime_error("base/query dimension mismatch while building MemQG");
  }
  alaya::platform::set_openmp_thread_count(static_cast<int>(args.build_threads));
  auto space = std::make_shared<MemorySpace>(base.rows, base.columns, alaya::core::Metric::l2);
  const auto started = Clock::now();
  space->fit(base.values.data(), base.rows);
  alaya::core::BuildContext build_context;
  alaya::QgBuildOptions build_options;
  build_options.ef_build = args.ef_build;
  build_options.thread_count = args.build_threads;
  auto segment = MemorySegment::build({alaya::core::TypedTensorView::contiguous(base.values.data(),
                                                                                base.rows,
                                                                                base.columns),
                                       space},
                                      build_options,
                                      build_context);
  const auto built = Clock::now();

  const auto temporary = fs::path(
      args.memqg_index_path.string() + ".tmp." +
      std::to_string(static_cast<unsigned long long>(Clock::now().time_since_epoch().count())));
  const auto temporary_string = temporary.string();
  const std::array locations{
      alaya::core::ArtifactLocation(MemorySegment::kArtifactName, temporary_string)};
  alaya::core::ArtifactWriter writer{std::span<const alaya::core::ArtifactLocation>(locations)};
  alaya::core::ArtifactManifest manifest;
  const auto status = segment->save(writer, alaya::core::SaveOptions{}, manifest);
  if (!status.ok()) {
    throw std::runtime_error("MemQG save failed: " + status.diagnostic());
  }
  fs::rename(temporary, args.memqg_index_path);
  const auto saved = Clock::now();
  std::cerr << "PREPARE,memqg,rows=" << base.rows << ",dim=" << base.columns
            << ",R=32,ef_build=" << args.ef_build
            << ",build_seconds=" << std::chrono::duration<double>(built - started).count()
            << ",save_seconds=" << std::chrono::duration<double>(saved - built).count() << '\n';
}

[[nodiscard]] auto open_memqg(const fs::path &path) -> alaya::core::AnySegment {
  const auto path_string = path.string();
  const std::array locations{
      alaya::core::ArtifactLocation(MemorySegment::kArtifactName, path_string)};
  alaya::core::OpenContext context;
  auto segment = MemorySegment::open(alaya::core::ArtifactView(
                                         std::span<const alaya::core::ArtifactLocation>(locations)),
                                     alaya::core::OpenOptions{},
                                     context);
  auto erased = MemorySegment::into_any(std::move(segment));
  if (!erased.ok()) {
    throw std::runtime_error("cannot erase MemQG segment: " + erased.status().diagnostic());
  }
  return std::move(erased).value();
}

[[nodiscard]] auto open_laser(const fs::path &directory) -> alaya::core::AnySegment {
  alaya::core::OpenContext context;
  auto segment =
      alaya::disk::LaserSegment::open_directory(directory, alaya::core::OpenOptions{}, context);
  if (!segment.ok()) {
    throw std::runtime_error("cannot open LASER segment: " + segment.status().diagnostic());
  }
  auto erased = alaya::disk::LaserSegment::into_any(std::move(segment).value());
  if (!erased.ok()) {
    throw std::runtime_error("cannot erase LASER segment: " + erased.status().diagnostic());
  }
  return std::move(erased).value();
}

struct SharedSearchParameters {
  alaya::QgSearchExtension qg{};
  alaya::disk::LaserSegmentSearchExtension laser{};
  std::array<alaya::core::AlgorithmSearchExtension, 2> extensions{};

  SharedSearchParameters(std::uint32_t ef, std::uint32_t beam) {
    qg.effort = ef;
    laser.effort = ef;
    laser.beam_width = beam;
    laser.return_distances = false;
    extensions[0] = alaya::make_qg_search_extension(qg);
    extensions[1] = alaya::disk::make_laser_segment_search_extension(laser);
    for (auto &extension : extensions) {
      extension.unknown_policy = alaya::core::UnknownExtensionPolicy::ignore_safe;
    }
  }
};

struct RunWork {
  const alaya::core::AnySegment *segment{};
  const Matrix<float> *queries{};
  std::uint32_t query_count{};
  std::uint32_t top_k{};
  SharedSearchParameters parameters;
  std::vector<alaya::core::SearchHit> hits;
  std::vector<alaya::core::RowCount> valid_counts;
  std::atomic<std::uint32_t> next_query{};
  std::atomic<bool> start{};
  std::atomic<bool> failed{};
  std::mutex error_mutex{};
  std::string error{};

  RunWork(const alaya::core::AnySegment &selected_segment,
          const Matrix<float> &query_matrix,
          std::uint32_t count,
          std::uint32_t requested_top_k,
          std::uint32_t ef,
          std::uint32_t beam)
      : segment(&selected_segment),
        queries(&query_matrix),
        query_count(count),
        top_k(requested_top_k),
        parameters(ef, beam),
        hits(static_cast<std::size_t>(count) * requested_top_k),
        valid_counts(count) {}

  void reset_for_run() {
    next_query.store(0, std::memory_order_relaxed);
    start.store(false, std::memory_order_relaxed);
    failed.store(false, std::memory_order_relaxed);
    std::lock_guard lock(error_mutex);
    error.clear();
  }

  void fail(std::string diagnostic) {
    if (!failed.exchange(true, std::memory_order_acq_rel)) {
      std::lock_guard lock(error_mutex);
      error = std::move(diagnostic);
      next_query.store(query_count, std::memory_order_release);
    }
  }
};

struct CallStorage {
  std::array<alaya::core::RowCount, 2> offsets{};
  std::array<alaya::core::RowCount, 1> counts{};
  std::array<alaya::core::Status, 1> statuses{};
  std::array<alaya::core::SearchCompleteness, 1> completeness{};
  alaya::core::SearchContext context{};
  alaya::core::SearchResponse response{};
  alaya::core::SearchRequest request{};

  CallStorage(std::uint32_t top_k,
              std::span<const alaya::core::AlgorithmSearchExtension> extensions) {
    response.offsets = offsets;
    response.valid_counts = counts;
    response.statuses = statuses;
    response.completeness = completeness;
    request.options.top_k = top_k;
    request.options.extensions = extensions;
    request.context = &context;
    request.response = &response;
  }
};

// This is the only measured dispatch function. Engine identity is absent: both arms differ solely
// in the AnySegment instance constructed above. Both algorithm extensions are present with
// ignore_safe so this request shape is also field-for-field identical across the arms.
void dispatch_query(const alaya::core::AnySegment &segment,
                    const float *query,
                    std::uint32_t dim,
                    std::span<alaya::core::SearchHit> output,
                    CallStorage &storage) {
  storage.response.hits = output;
  storage.request.queries = alaya::core::TypedTensorView::contiguous(query, 1, dim);
  const auto status = segment.search(storage.request);
  if (!status.ok()) {
    throw std::runtime_error(status.diagnostic());
  }
  if (!storage.statuses[0].ok()) {
    throw std::runtime_error(storage.statuses[0].diagnostic());
  }
}

class FixedWorkerPool {
 public:
  FixedWorkerPool(std::uint32_t lane_count, std::span<const int> cpus)
      : lane_count_(lane_count), cpus_(cpus.begin(), cpus.begin() + lane_count) {
    workers_.reserve(lane_count_);
    for (std::uint32_t lane = 0; lane < lane_count_; ++lane) {
      workers_.emplace_back([this, lane] {
        worker_loop(lane);
      });
    }
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [this] {
      return initialized_ == lane_count_;
    });
    if (!initialization_error_.empty()) {
      lock.unlock();
      shutdown();
      throw std::runtime_error(initialization_error_);
    }
  }

  FixedWorkerPool(const FixedWorkerPool &) = delete;
  auto operator=(const FixedWorkerPool &) -> FixedWorkerPool & = delete;

  ~FixedWorkerPool() { shutdown(); }

  [[nodiscard]] auto run(RunWork &work) -> double {
    work.reset_for_run();
    {
      std::lock_guard lock(mutex_);
      if (active_work_ != nullptr) {
        throw std::logic_error("fixed worker pool received overlapping work");
      }
      active_work_ = &work;
      arrived_ = 0;
      completed_ = 0;
      ++generation_;
    }
    wake_.notify_all();
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] {
        return arrived_ == lane_count_;
      });
    }
    const auto started = Clock::now();
    work.start.store(true, std::memory_order_release);
    {
      std::unique_lock lock(mutex_);
      done_.wait(lock, [this] {
        return completed_ == lane_count_;
      });
      active_work_ = nullptr;
    }
    const auto ended = Clock::now();
    if (work.failed.load(std::memory_order_acquire)) {
      std::lock_guard lock(work.error_mutex);
      throw std::runtime_error("worker search failed: " + work.error);
    }
    return std::chrono::duration<double>(ended - started).count();
  }

  [[nodiscard]] auto cpus() const -> std::span<const int> { return cpus_; }

 private:
  void shutdown() noexcept {
    {
      std::lock_guard lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
    }
    wake_.notify_all();
    for (auto &worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  [[nodiscard]] static auto pin_current_thread(int cpu) -> std::string {
#if defined(__linux__)
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
      return "CPU id is outside cpu_set_t: " + std::to_string(cpu);
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    const int status = ::pthread_setaffinity_np(::pthread_self(), sizeof(set), &set);
    if (status != 0) {
      return "pthread_setaffinity_np(cpu=" + std::to_string(cpu) +
             ") failed: " + std::string(std::strerror(status));
    }
    return {};
#else
    (void)cpu;
    return "pthread affinity is unsupported on this platform";
#endif
  }

  void worker_loop(std::uint32_t lane) noexcept {
    const auto affinity_error = pin_current_thread(cpus_[lane]);
    {
      std::lock_guard lock(mutex_);
      if (!affinity_error.empty() && initialization_error_.empty()) {
        initialization_error_ = affinity_error;
      }
      ++initialized_;
    }
    ready_.notify_one();

    std::uint64_t observed_generation{};
    for (;;) {
      RunWork *work{};
      {
        std::unique_lock lock(mutex_);
        wake_.wait(lock, [this, &observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) {
          return;
        }
        observed_generation = generation_;
        work = active_work_;
        ++arrived_;
      }
      ready_.notify_one();
      while (!work->start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      try {
        CallStorage storage(work->top_k, work->parameters.extensions);
        for (;;) {
          const auto query = work->next_query.fetch_add(1, std::memory_order_relaxed);
          if (query >= work->query_count || work->failed.load(std::memory_order_acquire)) {
            break;
          }
          auto output = std::span<alaya::core::SearchHit>(work->hits)
                            .subspan(static_cast<std::size_t>(query) * work->top_k, work->top_k);
          dispatch_query(*work->segment,
                         work->queries->row(query),
                         work->queries->columns,
                         output,
                         storage);
          work->valid_counts[query] = storage.counts[0];
        }
      } catch (const std::exception &error) {
        work->fail("lane=" + std::to_string(lane) + ",cpu=" + std::to_string(cpus_[lane]) + ": " +
                   error.what());
      } catch (...) {
        work->fail("lane=" + std::to_string(lane) + ",cpu=" + std::to_string(cpus_[lane]) +
                   ": unknown exception");
      }

      {
        std::lock_guard lock(mutex_);
        ++completed_;
      }
      done_.notify_one();
    }
  }

  std::uint32_t lane_count_{};
  std::vector<int> cpus_{};
  std::vector<std::thread> workers_{};
  std::mutex mutex_{};
  std::condition_variable wake_{};
  std::condition_variable ready_{};
  std::condition_variable done_{};
  RunWork *active_work_{};
  std::uint64_t generation_{};
  std::uint32_t initialized_{};
  std::uint32_t arrived_{};
  std::uint32_t completed_{};
  bool stopping_{};
  std::string initialization_error_{};
};

struct EngineArm {
  std::string name;
  alaya::core::AnySegment segment;
  std::vector<std::uint32_t> efs;
};

struct ResultSnapshot {
  std::vector<std::uint64_t> ids;
  std::vector<alaya::core::RowCount> counts;
};

[[nodiscard]] auto snapshot_results(const RunWork &work) -> ResultSnapshot {
  ResultSnapshot snapshot;
  snapshot.ids.assign(static_cast<std::size_t>(work.query_count) * work.top_k,
                      std::numeric_limits<std::uint64_t>::max());
  snapshot.counts = work.valid_counts;
  for (std::uint32_t query = 0; query < work.query_count; ++query) {
    const auto count = std::min<alaya::core::RowCount>(work.valid_counts[query], work.top_k);
    for (alaya::core::RowCount hit = 0; hit < count; ++hit) {
      snapshot.ids[static_cast<std::size_t>(query) * work.top_k + hit] = static_cast<std::uint64_t>(
          work.hits[static_cast<std::size_t>(query) * work.top_k + hit].row_id);
    }
  }
  return snapshot;
}

[[nodiscard]] auto result_mismatch(const ResultSnapshot &expected,
                                   const ResultSnapshot &actual,
                                   std::uint32_t top_k) -> std::optional<std::string> {
  if (expected.counts.size() != actual.counts.size() || expected.ids.size() != actual.ids.size()) {
    return "result shapes differ";
  }
  for (std::size_t query = 0; query < expected.counts.size(); ++query) {
    if (expected.counts[query] != actual.counts[query]) {
      return "query " + std::to_string(query) + " valid_count changed from " +
             std::to_string(expected.counts[query]) + " to " + std::to_string(actual.counts[query]);
    }
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto offset = query * top_k + hit;
      if (expected.ids[offset] != actual.ids[offset]) {
        return "query " + std::to_string(query) + ", hit " + std::to_string(hit) +
               " changed from " + std::to_string(expected.ids[offset]) + " to " +
               std::to_string(actual.ids[offset]);
      }
    }
  }
  return std::nullopt;
}

[[nodiscard]] auto recall_at_k(const ResultSnapshot &results,
                               const Matrix<std::uint32_t> &ground_truth,
                               std::uint32_t top_k) -> double {
  if (ground_truth.rows != results.counts.size() || ground_truth.columns < top_k) {
    throw std::runtime_error("ground truth shape is incompatible with measured results");
  }
  std::uint64_t matches{};
  for (std::uint32_t query = 0; query < ground_truth.rows; ++query) {
    if (results.counts[query] != top_k) {
      throw std::runtime_error("query " + std::to_string(query) + " returned " +
                               std::to_string(results.counts[query]) +
                               " hits, expected topk=" + std::to_string(top_k));
    }
    for (std::uint32_t hit = 0; hit < top_k; ++hit) {
      const auto id = results.ids[static_cast<std::size_t>(query) * top_k + hit];
      bool duplicate{};
      for (std::uint32_t prior = 0; prior < hit; ++prior) {
        duplicate = duplicate || results.ids[static_cast<std::size_t>(query) * top_k + prior] == id;
      }
      if (duplicate) {
        continue;
      }
      for (std::uint32_t truth = 0; truth < top_k; ++truth) {
        if (ground_truth.row(query)[truth] == id) {
          ++matches;
          break;
        }
      }
    }
  }
  return static_cast<double>(matches) /
         (static_cast<double>(ground_truth.rows) * static_cast<double>(top_k));
}

[[nodiscard]] auto result_checksum(const ResultSnapshot &snapshot) -> std::uint64_t {
  std::uint64_t checksum{0x6A09E667F3BCC909ULL};
  for (const auto id : snapshot.ids) {
    checksum ^= id + 0x9E3779B97F4A7C15ULL + (checksum << 6U) + (checksum >> 2U);
    checksum = std::rotl(checksum, 17);
  }
  return checksum;
}

struct Record {
  std::string dataset;
  std::string arm;
  std::uint32_t lanes{};
  std::uint32_t ef{};
  double qps{};
  double recall{};
  std::uint32_t repeat{};
  std::string order;
  std::uint32_t sequence_position{};
  std::uint32_t query_count{};
  std::uint32_t top_k{};
  std::uint32_t beam_width{};
  std::uint32_t warmup_rounds{};
  std::uint32_t warmup_queries{};
  std::uint32_t measurement_rounds{};
  std::uint64_t search_calls{};
  double elapsed_seconds{};
  std::uint64_t checksum{};
  std::string cpus;
};

[[nodiscard]] auto json_escape(std::string_view value) -> std::string {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(character) << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

class ResultWriter {
 public:
  explicit ResultWriter(const fs::path &prefix)
      : csv_path_(prefix.string() + ".csv"), json_path_(prefix.string() + ".json") {
    if (!prefix.parent_path().empty()) {
      fs::create_directories(prefix.parent_path());
    }
    csv_.open(csv_path_, std::ios::trunc);
    if (!csv_) {
      throw std::runtime_error("cannot create output " + csv_path_.string());
    }
    csv_ << "dataset,arm,lanes,ef,qps,recall,repeat,order,sequence_position,query_count,top_k,"
            "beam_width,warmup_rounds,warmup_queries,measurement_rounds,search_calls,"
            "elapsed_seconds,checksum,cpu_list\n";
  }

  void append(const Record &record) {
    records_.push_back(record);
    csv_ << record.dataset << ',' << record.arm << ',' << record.lanes << ',' << record.ef << ','
         << std::fixed << std::setprecision(3) << record.qps << ',' << std::setprecision(8)
         << record.recall << ',' << record.repeat << ',' << record.order << ','
         << record.sequence_position << ',' << record.query_count << ',' << record.top_k << ','
         << record.beam_width << ',' << record.warmup_rounds << ',' << record.warmup_queries << ','
         << record.measurement_rounds << ',' << record.search_calls << ',' << std::setprecision(9)
         << record.elapsed_seconds << ',' << std::hex << record.checksum << std::dec << ',' << '"'
         << record.cpus << '"' << '\n';
    csv_.flush();
  }

  void write_json(const Args &args,
                  std::span<const int> cpus,
                  const Matrix<float> &queries,
                  const Matrix<std::uint32_t> &ground_truth) const {
    std::ofstream output(json_path_, std::ios::trunc);
    if (!output) {
      throw std::runtime_error("cannot create output " + json_path_.string());
    }
    const auto &features = alaya::simd::get_cpu_features();
    const auto distance_level =
        alaya::simd::select_fp32_distance_level(features,
                                                alaya::simd::get_distance_dispatch_policy());
    output << "{\n"
              "  \"schema_version\": 2,\n"
              "  \"status\": \"complete\",\n"
              "  \"protocol\": \"external-fixed-pool/AnySegment-sync/single-row/fanout-1\",\n"
              "  \"dispatch\": \"shared request carrying ignore-safe QG+LASER extensions; only "
              "AnySegment construction differs\",\n"
              "  \"config\": {\n"
           << "    \"dataset\": \"" << json_escape(args.dataset) << "\",\n"
           << "    \"query_path\": \"" << json_escape(args.query_path.string()) << "\",\n"
           << "    \"ground_truth_path\": \"" << json_escape(args.ground_truth_path.string())
           << "\",\n"
           << "    \"laser_segment_directory\": \""
           << json_escape(args.laser_segment_directory.string()) << "\",\n"
           << "    \"memqg_index_path\": \"" << json_escape(args.memqg_index_path.string())
           << "\",\n"
           << "    \"query_rows\": " << queries.rows << ",\n"
           << "    \"dimension\": " << queries.columns << ",\n"
           << "    \"ground_truth_columns\": " << ground_truth.columns << ",\n"
           << "    \"top_k\": " << args.top_k << ",\n"
           << "    \"beam_width\": " << args.beam_width << ",\n"
           << "    \"lanes\": \"" << join_numbers<std::uint32_t>(args.lanes) << "\",\n"
           << "    \"laser_efs\": \"" << join_numbers<std::uint32_t>(args.laser_efs) << "\",\n"
           << "    \"memqg_efs\": \"" << join_numbers<std::uint32_t>(args.memqg_efs) << "\",\n"
           << "    \"repeats_per_order\": " << args.repeats << ",\n"
           << "    \"orders\": [\"forward_memqg_laser\", \"reverse_laser_memqg\"],\n"
           << "    \"warmup_rounds_per_point\": " << args.warmup_rounds << ",\n"
           << "    \"warmup_queries_per_round\": "
           << (args.warmup_queries == 0 ? queries.rows
                                        : std::min(args.warmup_queries, queries.rows))
           << ",\n"
           << "    \"minimum_measure_seconds\": " << args.minimum_measure_seconds << ",\n"
           << "    \"semantic_preconditions\": \"sealed/immutable; no deadline; no cancellation\"\n"
              "  },\n"
              "  \"machine\": {\n"
           << "    \"hostname\": \"" << json_escape(hostname()) << "\",\n"
           << "    \"selected_cpus\": \"" << join_ints(cpus) << "\",\n"
           << "    \"governor_cpu0\": \""
           << json_escape(read_first_line("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"))
           << "\",\n"
           << "    \"smt_active\": \""
           << json_escape(read_first_line("/sys/devices/system/cpu/smt/active")) << "\",\n"
           << "    \"transparent_hugepage_enabled\": \""
           << json_escape(read_first_line("/sys/kernel/mm/transparent_hugepage/enabled")) << "\",\n"
           << "    \"laser_simd\": \"" << alaya::laser::simd::get_laser_simd_name() << "\",\n"
           << "    \"memqg_rabitq_simd\": \"" << alaya::rabitq_simd::get_rabitq_simd_name()
           << "\",\n"
           << "    \"fp32_distance_simd\": \"" << alaya::simd::get_simd_level_name(distance_level)
           << "\",\n"
           << "    \"allocator\": \"alaya::AlignedAlloc (2MiB + MADV_HUGEPAGE for large "
              "engine/query allocations)\",\n"
           << "    \"query_buffer_2m_aligned\": "
           << (reinterpret_cast<std::uintptr_t>(queries.values.data()) % kLargeAlignment == 0
                   ? "true"
                   : "false")
           << "\n  },\n"
              "  \"measurements\": [\n";
    for (std::size_t index = 0; index < records_.size(); ++index) {
      const auto &record = records_[index];
      output << "    {\"dataset\": \"" << json_escape(record.dataset) << "\", \"arm\": \""
             << json_escape(record.arm) << "\", \"lanes\": " << record.lanes
             << ", \"ef\": " << record.ef << ", \"qps\": " << std::fixed << std::setprecision(3)
             << record.qps << ", \"recall\": " << std::setprecision(8) << record.recall
             << ", \"repeat\": " << record.repeat << ", \"order\": \"" << json_escape(record.order)
             << "\", \"sequence_position\": " << record.sequence_position
             << ", \"query_count\": " << record.query_count << ", \"top_k\": " << record.top_k
             << ", \"beam_width\": " << record.beam_width
             << ", \"warmup_rounds\": " << record.warmup_rounds
             << ", \"warmup_queries\": " << record.warmup_queries
             << ", \"measurement_rounds\": " << record.measurement_rounds
             << ", \"search_calls\": " << record.search_calls
             << ", \"elapsed_seconds\": " << std::setprecision(9) << record.elapsed_seconds
             << ", \"checksum\": \"" << std::hex << record.checksum << std::dec
             << "\", \"cpu_list\": \"" << json_escape(record.cpus) << "\"}"
             << (index + 1 == records_.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
  }

  [[nodiscard]] auto csv_path() const -> const fs::path & { return csv_path_; }
  [[nodiscard]] auto json_path() const -> const fs::path & { return json_path_; }

 private:
  fs::path csv_path_;
  fs::path json_path_;
  mutable std::ofstream csv_;
  std::vector<Record> records_;
};

[[nodiscard]] auto validate_segment(const EngineArm &arm, std::uint32_t expected_dim)
    -> alaya::core::RowCount {
  const auto descriptor = arm.segment.descriptor();
  if (descriptor.dim != expected_dim || descriptor.metric != alaya::core::Metric::l2) {
    throw std::runtime_error(arm.name + " descriptor disagrees with query dimension/L2 protocol");
  }
  alaya::core::SegmentStats stats;
  const auto status = arm.segment.stats(stats);
  if (!status.ok() || stats.live_rows == 0) {
    throw std::runtime_error(arm.name + " did not report a valid live row count");
  }
  if (!arm.segment.capabilities().concurrency.reentrant_search) {
    throw std::runtime_error(arm.name + " does not advertise reentrant search");
  }
  return stats.live_rows;
}

void warm_up(FixedWorkerPool &pool,
             const EngineArm &arm,
             const Matrix<float> &queries,
             const Args &args,
             std::uint32_t ef) {
  const auto warmup_queries =
      args.warmup_queries == 0 ? queries.rows : std::min(args.warmup_queries, queries.rows);
  for (std::uint32_t round = 0; round < args.warmup_rounds; ++round) {
    RunWork warmup(arm.segment, queries, warmup_queries, args.top_k, ef, args.beam_width);
    (void)pool.run(warmup);
  }
}

void run_grid(const Args &args,
              std::span<const int> selected_cpus,
              const Matrix<float> &queries,
              const Matrix<std::uint32_t> &ground_truth,
              EngineArm &memqg,
              EngineArm &laser,
              ResultWriter &writer) {
  std::map<std::string, ResultSnapshot> baselines;
  const auto warmup_query_count =
      args.warmup_queries == 0 ? queries.rows : std::min(args.warmup_queries, queries.rows);

  for (const auto lane_count : args.lanes) {
    FixedWorkerPool pool(lane_count, selected_cpus);
    for (std::uint32_t repeat = 1; repeat <= args.repeats; ++repeat) {
      const std::array<bool, 2> orders =
          repeat % 2U == 1U ? std::array<bool, 2>{true, false} : std::array<bool, 2>{false, true};
      for (const bool forward : orders) {
        std::array<EngineArm *, 2> arms = forward ? std::array<EngineArm *, 2>{&memqg, &laser}
                                                  : std::array<EngineArm *, 2>{&laser, &memqg};
        const std::string order = forward ? "forward_memqg_laser" : "reverse_laser_memqg";
        for (std::size_t position = 0; position < arms.size(); ++position) {
          auto &arm = *arms[position];
          for (const auto ef : arm.efs) {
            warm_up(pool, arm, queries, args, ef);
            RunWork work(arm.segment, queries, queries.rows, args.top_k, ef, args.beam_width);
            double elapsed{};
            std::uint32_t measurement_rounds{};
            do {
              elapsed += pool.run(work);
              ++measurement_rounds;
            } while (elapsed < args.minimum_measure_seconds);
            const auto search_calls = static_cast<std::uint64_t>(queries.rows) * measurement_rounds;
            auto snapshot = snapshot_results(work);
            const auto baseline_key = arm.name + ":" + std::to_string(ef);
            const auto baseline = baselines.find(baseline_key);
            if (baseline == baselines.end()) {
              baselines.emplace(baseline_key, snapshot);
            } else if (const auto mismatch =
                           result_mismatch(baseline->second, snapshot, args.top_k)) {
              throw std::runtime_error(
                  "RESULT_MISMATCH arm=" + arm.name + ",ef=" + std::to_string(ef) +
                  ",C=" + std::to_string(lane_count) + ",repeat=" + std::to_string(repeat) +
                  ",order=" + order + ": " + *mismatch);
            }
            const auto recall = recall_at_k(snapshot, ground_truth, args.top_k);
            const auto qps = static_cast<double>(search_calls) / elapsed;
            const Record record{args.dataset,
                                arm.name,
                                lane_count,
                                ef,
                                qps,
                                recall,
                                repeat,
                                order,
                                static_cast<std::uint32_t>(position + 1),
                                queries.rows,
                                args.top_k,
                                args.beam_width,
                                args.warmup_rounds,
                                warmup_query_count,
                                measurement_rounds,
                                search_calls,
                                elapsed,
                                result_checksum(snapshot),
                                join_ints(pool.cpus())};
            writer.append(record);
            std::cerr << "MEASURE,arm=" << arm.name << ",C=" << lane_count << ",ef=" << ef
                      << ",repeat=" << repeat << ",order=" << order << ",qps=" << std::fixed
                      << std::setprecision(1) << qps << ",recall=" << std::setprecision(5) << recall
                      << ",measurement_rounds=" << measurement_rounds << '\n';
          }
        }
      }
    }
  }
}

}  // namespace

auto main(int argc, char **argv) -> int {
  try {
    auto args = parse_args(argc, argv);
    auto query_probe = read_matrix<float>(args.query_path, 1);
    prepare_laser_segment(args, query_probe.columns);
    build_memqg(args, query_probe.columns);
    if (args.prepare_only) {
      std::cerr << "PREPARE,complete,laser_segment=" << args.laser_segment_directory
                << ",memqg_index=" << args.memqg_index_path << '\n';
      return 0;
    }

    auto queries = read_matrix<float>(args.query_path, args.query_limit);
    auto ground_truth = read_matrix<std::uint32_t>(args.ground_truth_path, args.query_limit);
    if (queries.rows != ground_truth.rows || ground_truth.columns < args.top_k) {
      throw std::runtime_error("query/ground-truth dimensions do not satisfy recall@topk");
    }
    auto cpus = args.cpus.empty() ? allowed_cpus() : args.cpus;
    const auto max_lanes = *std::max_element(args.lanes.begin(), args.lanes.end());
    if (cpus.size() < max_lanes) {
      throw std::runtime_error("CPU list is shorter than the maximum requested lane count");
    }
    const auto allowed = allowed_cpus();
    const std::set<int> allowed_set(allowed.begin(), allowed.end());
    for (const auto cpu : cpus) {
      if (!allowed_set.contains(cpu)) {
        throw std::runtime_error("selected CPU " + std::to_string(cpu) +
                                 " is outside the process affinity mask");
      }
    }
    cpus.resize(max_lanes);

    EngineArm memqg{"memqg", open_memqg(args.memqg_index_path), args.memqg_efs};
    EngineArm laser{"laser_arena", open_laser(args.laser_segment_directory), args.laser_efs};
    const auto memqg_rows = validate_segment(memqg, queries.columns);
    const auto laser_rows = validate_segment(laser, queries.columns);
    if (memqg_rows != laser_rows) {
      throw std::runtime_error("MemQG and LASER row counts differ");
    }

    ResultWriter writer(args.output_prefix);
    std::cerr << "CONFIG,dataset=" << args.dataset << ",queries=" << queries.rows
              << ",dim=" << queries.columns << ",topk=" << args.top_k
              << ",lanes=" << join_numbers<std::uint32_t>(args.lanes) << ",cpus=" << join_ints(cpus)
              << ",repeats_per_order=" << args.repeats << ",warmup_rounds=" << args.warmup_rounds
              << ",min_measure_seconds=" << args.minimum_measure_seconds
              << ",laser_simd=" << alaya::laser::simd::get_laser_simd_name()
              << ",memqg_simd=" << alaya::rabitq_simd::get_rabitq_simd_name() << '\n';
    run_grid(args, cpus, queries, ground_truth, memqg, laser, writer);
    writer.write_json(args, cpus, queries, ground_truth);
    std::cerr << "OUTPUT,csv=" << writer.csv_path() << ",json=" << writer.json_path() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "parity_lanes_benchmark: " << error.what() << '\n';
    return 1;
  }
}
