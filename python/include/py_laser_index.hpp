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

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base_py_index.hpp"
#include "index/laser/laser_build_params.hpp"
#include "index/laser/laser_builder.hpp"
#include "index/laser/laser_index.hpp"
#include "params.hpp"
#include "space/raw_space.hpp"

namespace py = pybind11;

namespace alaya {

class PyLaserIndex : public BasePyIndex {
 public:
  explicit PyLaserIndex(IndexParams params)
      : params_(std::move(params)), build_params_(params_.laser_build_params_) {}

  auto fit(py::array_t<float> &vectors, uint32_t ef_construction, uint32_t num_threads) -> void {
    if (params_.metric_ != MetricType::L2) {
      throw std::logic_error("LASER currently supports only L2 metric");
    }
    if (vectors.ndim() != 2) {
      throw std::runtime_error("Array must be 2D");
    }

    auto num_points = static_cast<uint32_t>(vectors.shape(0));
    auto dim = static_cast<uint32_t>(vectors.shape(1));
    auto *data = static_cast<float *>(vectors.request().ptr);

    build_params_.ef_construction_ = ef_construction;
    if (num_threads > 0) {
      build_params_.num_threads_ = num_threads;
    }

    data_space_ =
        std::make_shared<RawSpace<float, float, uint32_t>>(num_points, dim, MetricType::L2);
    data_space_->fit(data, num_points);
    data_dim_ = dim;

    auto build_dir = make_temp_dir();
    auto prefix = (build_dir / "laser").string();
    LaserBuilder<RawSpace<float, float, uint32_t>> builder(data_space_, build_params_);
    builder.build(prefix);

    num_points_ = num_points;
    full_dimension_ = dim;
    main_dim_ = build_params_.resolved_main_dim(dim);
    max_degree_ = build_params_.max_degree_;
    prefix_ = prefix;
    load_index(prefix_);
  }

  auto get_data_by_id(uint32_t id) -> py::array_t<float> {
    ensure_data_space();
    if (id >= data_space_->get_data_num()) {
      throw std::runtime_error("id out of range");
    }
    auto *data = data_space_->get_data_by_id(id);
    return py::array_t<float>({data_dim_}, {sizeof(float)}, data);
  }

  auto save(const std::string &index_path,
            const std::string & = std::string(),
            const std::string & = std::string()) -> void {
    ensure_loaded();

    auto target_dir = std::filesystem::path(index_path);
    std::filesystem::create_directories(target_dir);
    auto target_prefix = (target_dir / "laser").string();
    copy_index_artifacts(prefix_, target_prefix, max_degree_, main_dim_);

    if (data_space_ != nullptr) {
      data_space_->save((target_dir / "raw.data").string());
    }
    write_schema(target_dir);
  }

  auto load(const std::string &index_path,
            const std::string & = std::string(),
            const std::string & = std::string()) -> void {
    auto target_dir = std::filesystem::path(index_path);
    auto schema = read_schema(target_dir / "schema.json");

    num_points_ = schema.num_points_;
    main_dim_ = schema.main_dim_;
    max_degree_ = schema.max_degree_;
    full_dimension_ = schema.full_dimension_;
    data_dim_ = static_cast<uint32_t>(full_dimension_);
    build_params_ = schema.build_params_;
    prefix_ = (target_dir / "laser").string();

    auto raw_path = target_dir / "raw.data";
    if (std::filesystem::exists(raw_path)) {
      data_space_ = std::make_shared<RawSpace<float, float, uint32_t>>();
      data_space_->load(raw_path.string());
      data_space_->set_metric_function();
      data_dim_ = data_space_->get_dim();
      full_dimension_ = data_dim_;
    }

    load_index(prefix_);
  }

  auto insert(py::array_t<float> &, uint32_t) -> uint32_t {
    throw std::logic_error("LASER does not support incremental insert");
  }

  auto remove(uint32_t) -> void { throw std::logic_error("LASER does not support removal"); }

  auto search(py::array_t<float> &query, uint32_t topk, uint32_t ef) -> py::array_t<uint32_t> {
    ensure_loaded();
    ensure_data_space();
    validate_query(query);
    auto result_ids = reranked_search(query.data(), topk, ef);

    py::array_t<uint32_t> result(topk);
    std::copy(result_ids.begin(), result_ids.end(), result.mutable_data());
    return result;
  }

  auto batch_search(py::array_t<float> &queries, uint32_t topk, uint32_t ef, uint32_t num_threads)
      -> py::array_t<uint32_t> {
    ensure_loaded();
    ensure_data_space();
    validate_queries(queries);
    auto num_queries = static_cast<size_t>(queries.shape(0));
    py::array_t<uint32_t> result({num_queries, static_cast<size_t>(topk)});
    auto *query_ptr = queries.data();
    auto *result_ptr = result.mutable_data();

    for (size_t row = 0; row < num_queries; ++row) {
      auto ids = reranked_search(query_ptr + row * data_dim_,
                                 topk,
                                 ef,
                                 num_threads == 0 ? search_params_.num_threads
                                                  : static_cast<size_t>(num_threads));
      std::copy(ids.begin(), ids.end(), result_ptr + row * topk);
    }
    return result;
  }

  auto batch_search_with_distance(py::array_t<float> &queries,
                                  uint32_t topk,
                                  uint32_t ef,
                                  uint32_t num_threads) -> py::object {
    ensure_loaded();
    ensure_data_space();
    validate_queries(queries);

    auto ids = batch_search(queries, topk, ef, num_threads);
    py::array_t<float> distances(
        {static_cast<size_t>(queries.shape(0)), static_cast<size_t>(topk)});

    auto *id_ptr = ids.mutable_data();
    auto *dist_ptr = distances.mutable_data();
    auto dist_fn = data_space_->get_dist_func();
    auto *query_ptr = queries.data();
    auto dim = data_space_->get_dim();

    for (ssize_t row = 0; row < queries.shape(0); ++row) {
      const float *current_query = query_ptr + static_cast<size_t>(row) * dim;
      for (uint32_t col = 0; col < topk; ++col) {
        auto id = id_ptr[static_cast<size_t>(row) * topk + col];
        dist_ptr[static_cast<size_t>(row) * topk + col] =
            dist_fn(current_query, data_space_->get_data_by_id(id), dim);
      }
    }
    return py::make_tuple(ids, distances);
  }

 private:
  struct SavedSchema {
    uint32_t main_dim_{0};
    uint32_t max_degree_{0};
    uint32_t num_points_{0};
    uint32_t full_dimension_{0};
    LaserBuildParams build_params_;
  };

  IndexParams params_;
  LaserBuildParams build_params_;
  symqg::LaserSearchParams search_params_;
  std::unique_ptr<LaserIndex> index_;
  std::shared_ptr<RawSpace<float, float, uint32_t>> data_space_;
  std::string prefix_;
  uint32_t num_points_{0};
  uint32_t main_dim_{0};
  uint32_t max_degree_{0};
  uint32_t full_dimension_{0};

  static auto make_temp_dir() -> std::filesystem::path {
    auto stamp = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    auto dir = std::filesystem::temp_directory_path() / ("alayalite_laser_" + stamp);
    std::filesystem::create_directories(dir);
    return dir;
  }

  void ensure_loaded() const {
    if (index_ == nullptr || !index_->is_loaded()) {
      throw std::runtime_error("LASER index is not loaded");
    }
  }

  void ensure_data_space() const {
    if (data_space_ == nullptr) {
      throw std::runtime_error("LASER raw vectors are unavailable for this operation");
    }
  }

  void validate_query(const py::array_t<float> &query) const {
    if (query.ndim() != 1 || static_cast<uint32_t>(query.shape(0)) != data_dim_) {
      throw std::runtime_error("LASER query dimension must match fitted data dimension");
    }
  }

  void validate_queries(const py::array_t<float> &queries) const {
    if (queries.ndim() != 2 || static_cast<uint32_t>(queries.shape(1)) != data_dim_) {
      throw std::runtime_error("LASER query batch dimension must match fitted data dimension");
    }
  }

  void update_search_params(uint32_t ef_search, size_t num_threads) {
    search_params_.ef_search = ef_search;
    if (num_threads > 0) {
      search_params_.num_threads = num_threads;
    } else if (search_params_.num_threads == 0) {
      search_params_.num_threads = build_params_.resolved_num_threads();
    }
    if (search_params_.beam_width == 0) {
      search_params_.beam_width = 16;
    }
    if (search_params_.search_dram_budget_gb <= 0.0F) {
      search_params_.search_dram_budget_gb = 1.0F;
    }
    index_->set_search_params(search_params_);
  }

  auto reranked_search(const float *query,
                       uint32_t topk,
                       uint32_t ef_search,
                       size_t num_threads = 0) -> std::vector<uint32_t> {
    auto candidate_k =
        std::min<uint32_t>(num_points_,
                           std::max<uint32_t>(topk, std::max<uint32_t>(ef_search, 32)));
    update_search_params(std::max<uint32_t>(candidate_k, ef_search),
                         num_threads == 0 ? search_params_.num_threads : num_threads);

    std::vector<uint32_t> candidate_ids(candidate_k, 0);
    index_->search(query, candidate_k, candidate_ids.data());
    sanitize_ids(candidate_ids.data(), candidate_ids.size());

    auto dist_fn = data_space_->get_dist_func();
    std::unordered_set<uint32_t> seen;
    std::vector<std::tuple<float, uint32_t>> ranked;
    ranked.reserve(candidate_ids.size());
    for (auto id : candidate_ids) {
      if (id >= num_points_ || !seen.insert(id).second) {
        continue;
      }
      ranked.emplace_back(dist_fn(query, data_space_->get_data_by_id(id), data_dim_), id);
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
      if (std::get<0>(lhs) != std::get<0>(rhs)) {
        return std::get<0>(lhs) < std::get<0>(rhs);
      }
      return std::get<1>(lhs) < std::get<1>(rhs);
    });

    std::vector<uint32_t> result(topk, 0);
    size_t copied = 0;
    for (; copied < topk && copied < ranked.size(); ++copied) {
      result[copied] = std::get<1>(ranked[copied]);
    }
    for (; copied < topk; ++copied) {
      result[copied] = copied == 0 ? 0U : result[copied - 1];
    }
    return result;
  }

  void load_index(const std::string &prefix) {
    search_params_.ef_search =
        std::max<size_t>(search_params_.ef_search, std::max<size_t>(build_params_.ef_build_, 64));
    search_params_.num_threads = search_params_.num_threads == 0
                                     ? build_params_.resolved_num_threads()
                                     : search_params_.num_threads;
    search_params_.beam_width = search_params_.beam_width == 0 ? 16 : search_params_.beam_width;
    search_params_.search_dram_budget_gb =
        search_params_.search_dram_budget_gb <= 0.0F ? 1.0F : search_params_.search_dram_budget_gb;

    index_ = std::make_unique<LaserIndex>();
    index_->load(prefix, num_points_, max_degree_, main_dim_, full_dimension_, search_params_);
  }

  static void copy_file_if_exists(const std::filesystem::path &src,
                                  const std::filesystem::path &dst) {
    if (!std::filesystem::exists(src)) {
      throw std::runtime_error("Missing LASER artifact: " + src.string());
    }
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
  }

  static void copy_index_artifacts(const std::string &src_prefix,
                                   const std::string &dst_prefix,
                                   uint32_t max_degree,
                                   uint32_t main_dim) {
    auto src_base = src_prefix + "_R" + std::to_string(max_degree) + "_MD" +
                    std::to_string(main_dim) + ".index";
    auto dst_base = dst_prefix + "_R" + std::to_string(max_degree) + "_MD" +
                    std::to_string(main_dim) + ".index";

    copy_file_if_exists(src_prefix + "_pca.bin", dst_prefix + "_pca.bin");
    copy_file_if_exists(src_prefix + "_medoids_indices", dst_prefix + "_medoids_indices");
    copy_file_if_exists(src_prefix + "_medoids", dst_prefix + "_medoids");
    copy_file_if_exists(src_base, dst_base);
    copy_file_if_exists(src_base + "_rotator", dst_base + "_rotator");
    copy_file_if_exists(src_base + "_cache_ids", dst_base + "_cache_ids");
    copy_file_if_exists(src_base + "_cache_nodes", dst_base + "_cache_nodes");
  }

  void write_schema(const std::filesystem::path &dir) const {
    std::ofstream output(dir / "schema.json", std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Failed to write LASER schema.json");
    }

    output << "{\n";
    output << "  \"type\": \"index\",\n";
    output << "  \"index\": {\n";
    output << "    \"index_type\": \"laser\",\n";
    output << "    \"data_type\": \"float32\",\n";
    output << "    \"id_type\": \"uint32\",\n";
    output << "    \"quantization_type\": \"none\",\n";
    output << "    \"metric\": \"l2\",\n";
    output << "    \"capacity\": " << num_points_ << ",\n";
    output << "    \"max_nbrs\": " << max_degree_ << ",\n";
    output << "    \"main_dim\": " << main_dim_ << ",\n";
    output << "    \"max_degree\": " << max_degree_ << ",\n";
    output << "    \"num_points\": " << num_points_ << ",\n";
    output << "    \"dimension\": " << full_dimension_ << ",\n";
    output << "    \"full_dimension\": " << full_dimension_ << ",\n";
    output << "    \"laser_build_params\": {\n";
    output << "      \"main_dim\": " << build_params_.main_dim_ << ",\n";
    output << "      \"max_degree\": " << build_params_.max_degree_ << ",\n";
    output << "      \"ef_construction\": " << build_params_.ef_construction_ << ",\n";
    output << "      \"ef_build\": " << build_params_.ef_build_ << ",\n";
    output << "      \"alpha\": " << build_params_.alpha_ << ",\n";
    output << "      \"num_medoids\": " << build_params_.num_medoids_ << ",\n";
    output << "      \"pca_sample_ratio\": " << build_params_.pca_sample_ratio_ << ",\n";
    output << "      \"pca_sample_cap\": " << build_params_.pca_sample_cap_ << ",\n";
    output << "      \"medoid_sample_ratio\": " << build_params_.medoid_sample_ratio_ << ",\n";
    output << "      \"medoid_sample_cap\": " << build_params_.medoid_sample_cap_ << ",\n";
    output << "      \"num_threads\": " << build_params_.num_threads_ << ",\n";
    output << "      \"max_memory_mb\": " << build_params_.max_memory_mb_ << ",\n";
    output << "      \"keep_intermediates\": "
           << (build_params_.keep_intermediates_ ? "true" : "false") << "\n";
    output << "    }\n";
    output << "  }\n";
    output << "}\n";
  }

  static auto read_text(const std::filesystem::path &path) -> std::string {
    std::ifstream input(path);
    if (!input) {
      throw std::runtime_error("Failed to open schema.json: " + path.string());
    }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    return text;
  }

  static auto extract_string(const std::string &text, const std::string &key) -> std::string {
    std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
      return "";
    }
    return match[1].str();
  }

  static auto extract_uint(const std::string &text, const std::string &key) -> uint32_t {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(\\d+)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
      return 0;
    }
    return static_cast<uint32_t>(std::stoul(match[1].str()));
  }

  static auto extract_float(const std::string &text, const std::string &key) -> float {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+(?:\\.[0-9]+)?)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
      return 0.0F;
    }
    return std::stof(match[1].str());
  }

  static auto extract_bool(const std::string &text, const std::string &key) -> bool {
    std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
    std::smatch match;
    if (!std::regex_search(text, match, pattern)) {
      return false;
    }
    return match[1].str() == "true";
  }

  static auto read_schema(const std::filesystem::path &path) -> SavedSchema {
    auto text = read_text(path);
    if (extract_string(text, "index_type") != "laser") {
      throw std::runtime_error("schema.json does not describe a LASER index");
    }

    SavedSchema schema;
    schema.main_dim_ = extract_uint(text, "main_dim");
    schema.max_degree_ = extract_uint(text, "max_degree");
    schema.num_points_ = extract_uint(text, "num_points");
    schema.full_dimension_ =
        std::max(extract_uint(text, "full_dimension"), extract_uint(text, "dimension"));

    schema.build_params_.main_dim_ = extract_uint(text, "main_dim");
    schema.build_params_.max_degree_ = extract_uint(text, "max_degree");
    schema.build_params_.ef_construction_ = extract_uint(text, "ef_construction");
    schema.build_params_.ef_build_ = extract_uint(text, "ef_build");
    schema.build_params_.alpha_ = extract_float(text, "alpha");
    schema.build_params_.num_medoids_ = extract_uint(text, "num_medoids");
    schema.build_params_.pca_sample_ratio_ = extract_float(text, "pca_sample_ratio");
    schema.build_params_.pca_sample_cap_ = extract_uint(text, "pca_sample_cap");
    schema.build_params_.medoid_sample_ratio_ = extract_float(text, "medoid_sample_ratio");
    schema.build_params_.medoid_sample_cap_ = extract_uint(text, "medoid_sample_cap");
    schema.build_params_.num_threads_ = extract_uint(text, "num_threads");
    schema.build_params_.max_memory_mb_ = extract_uint(text, "max_memory_mb");
    schema.build_params_.keep_intermediates_ = extract_bool(text, "keep_intermediates");
    return schema;
  }

  void sanitize_ids(uint32_t *ids, size_t count) const {
    uint32_t fallback = 0;
    for (size_t i = 0; i < count; ++i) {
      if (ids[i] < num_points_) {
        fallback = ids[i];
        continue;
      }
      ids[i] = fallback;
    }
  }
};

}  // namespace alaya
