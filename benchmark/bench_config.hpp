/*
 * Shared benchmark configuration utilities.
 *
 * Provides TOML parsing helpers and common section structs ([dataset], [index],
 * [search]) so that individual benchmarks only need to define their own
 * [benchmark] section.
 *
 * Usage:
 *   auto parsed = bench::parse_common(argv[1]);
 *   // parsed.dataset_, parsed.index_, parsed.search_ are ready
 *   // parsed.root_ gives access to [benchmark] for custom fields
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <toml++/toml.hpp>

namespace bench {

// =============================================================================
// TOML helpers
// =============================================================================

/// Convert a toml::array to std::vector<T>.
template <typename T>
inline auto toml_array_to_vec(const toml::array &arr) -> std::vector<T> {
  std::vector<T> v;
  v.reserve(arr.size());
  for (auto &el : arr) {
    if constexpr (std::is_floating_point_v<T>) {
      v.push_back(static_cast<T>(el.value_or(double{0})));
    } else {
      v.push_back(static_cast<T>(el.value_or(T{0})));
    }
  }
  return v;
}

/// Read a scalar from a TOML table with a default.
template <typename T>
inline auto toml_get(const toml::table &tbl, std::string_view key, T default_val) -> T {
  auto *v = tbl.get(key);
  if (v == nullptr) {
    return default_val;
  }
  if constexpr (std::is_same_v<T, std::string>) {
    return v->value_or(default_val);
  } else if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(v->value_or(static_cast<double>(default_val)));
  } else {
    return static_cast<T>(v->value_or(default_val));
  }
}

/// Read an array field as std::vector<T>, returning default if absent.
template <typename T>
inline auto toml_get_vec(const toml::table &tbl, std::string_view key,
                         std::vector<T> default_val = {}) -> std::vector<T> {
  auto *a = tbl.get(key);
  if (a != nullptr && a->is_array()) {
    return toml_array_to_vec<T>(*a->as_array());
  }
  return default_val;
}

// =============================================================================
// Shared section structs
// =============================================================================

struct DatasetSection {
  std::string data_path_;
  std::string query_path_;
  std::string gt_path_;
  std::string metric_ = "L2";

  // Random dataset fallback (used by HNSW when data_path is empty)
  uint32_t random_data_num_ = 10000;
  uint32_t random_query_num_ = 100;
  uint32_t random_dim_ = 128;
  uint32_t random_gt_topk_ = 100;

  static auto from_toml(const toml::table &root) -> DatasetSection {
    DatasetSection s;
    auto *ds = root["dataset"].as_table();
    if (ds == nullptr) {
      return s;
    }
    s.data_path_ = toml_get<std::string>(*ds, "data_path", {});
    s.query_path_ = toml_get<std::string>(*ds, "query_path", {});
    s.gt_path_ = toml_get<std::string>(*ds, "gt_path", {});
    s.metric_ = toml_get<std::string>(*ds, "metric", "L2");
    s.random_data_num_ = toml_get<uint32_t>(*ds, "random_data_num", 10000);
    s.random_query_num_ = toml_get<uint32_t>(*ds, "random_query_num", 100);
    s.random_dim_ = toml_get<uint32_t>(*ds, "random_dim", 128);
    s.random_gt_topk_ = toml_get<uint32_t>(*ds, "random_gt_topk", 100);
    return s;
  }

  void print() const {
    printf("[dataset]\n");
    if (!data_path_.empty()) {
      printf("  data_path  = %s\n", data_path_.c_str());
      printf("  query_path = %s\n", query_path_.c_str());
      printf("  gt_path    = %s\n", gt_path_.c_str());
    } else {
      printf("  random: %u vectors, %u queries, dim=%u\n",
             random_data_num_, random_query_num_, random_dim_);
    }
    printf("  metric     = %s\n", metric_.c_str());
  }
};

struct IndexSection {
  std::vector<uint32_t> r_ = {16, 32};
  std::vector<uint32_t> ef_construction_ = {200};  // HNSW
  uint32_t beam_width_ = 4;                        // DiskANN
  uint32_t num_threads_ = 0;                       // 0 = hardware_concurrency

  static auto from_toml(const toml::table &root) -> IndexSection {
    IndexSection s;
    auto *idx = root["index"].as_table();
    if (idx == nullptr) {
      return s;
    }
    s.r_ = toml_get_vec<uint32_t>(*idx, "R", {16, 32});
    s.ef_construction_ = toml_get_vec<uint32_t>(*idx, "ef_construction", {200});
    s.beam_width_ = toml_get<uint32_t>(*idx, "beam_width", 4);
    s.num_threads_ = toml_get<uint32_t>(*idx, "num_threads", 0);
    return s;
  }

  void print() const {
    printf("[index]\n");
    printf("  R          = [");
    for (size_t i = 0; i < r_.size(); ++i) {
      printf("%s%u", i != 0U ? ", " : "", r_[i]);
    }
    printf("]\n");
    if (!ef_construction_.empty()) {
      printf("  ef_construction = [");
      for (size_t i = 0; i < ef_construction_.size(); ++i) {
        printf("%s%u", i != 0U ? ", " : "", ef_construction_[i]);
      }
      printf("]\n");
    }
    if (beam_width_ > 0U) {
      printf("  beam_width = %u\n", beam_width_);
    }
    printf("  num_threads= %u\n", num_threads_);
  }
};

struct SearchSection {
  std::vector<uint32_t> ef_search_ = {16, 32, 64, 128, 256};
  uint32_t topk_ = 10;
  uint32_t num_queries_ = 0;     // 0 = all queries
  uint32_t search_threads_ = 1;  // DiskANN

  static auto from_toml(const toml::table &root) -> SearchSection {
    SearchSection s;
    auto *sec = root["search"].as_table();
    if (sec == nullptr) {
      return s;
    }
    s.ef_search_ = toml_get_vec<uint32_t>(*sec, "ef_search", {16, 32, 64, 128, 256});
    s.topk_ = toml_get<uint32_t>(*sec, "topk", 10);
    s.num_queries_ = toml_get<uint32_t>(*sec, "num_queries", 0);
    s.search_threads_ = toml_get<uint32_t>(*sec, "search_threads", 1);
    return s;
  }

  void print() const {
    printf("[search]\n");
    printf("  ef_search  = [");
    for (size_t i = 0; i < ef_search_.size(); ++i) {
      printf("%s%u", i != 0U ? ", " : "", ef_search_[i]);
    }
    printf("]\n");
    printf("  topk       = %u\n", topk_);
    printf("  num_queries= %u\n", num_queries_);
    if (search_threads_ > 1U) {
      printf("  search_threads= %u\n", search_threads_);
    }
  }
};

// =============================================================================
// Top-level parse
// =============================================================================

/// Parsed common sections plus the raw TOML table for benchmark-specific fields.
struct ParsedConfig {
  DatasetSection dataset_;
  IndexSection index_;
  SearchSection search_;
  toml::table root_;
};

/// Parse a TOML config file and return shared sections + raw table.
inline auto parse_common(const char *path) -> ParsedConfig {
  auto tbl = toml::parse_file(path);
  ParsedConfig p;
  p.dataset_ = DatasetSection::from_toml(tbl);
  p.index_ = IndexSection::from_toml(tbl);
  p.search_ = SearchSection::from_toml(tbl);
  p.root_ = std::move(tbl);
  return p;
}

// =============================================================================
// Utility
// =============================================================================

/// Read current process RSS from /proc/self/status (Linux only).
inline auto get_rss_mb() -> double {
#if defined(__linux__)
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line)) {
    if (line.starts_with("VmRSS:")) {
      int64_t kb = std::stoll(line.substr(std::string("VmRSS:").size()));
      return static_cast<double>(kb) / 1024.0;
    }
  }
#endif
  return -1.0;
}

}  // namespace bench
