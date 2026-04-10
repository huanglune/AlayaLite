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

#include <algorithm>
#include <array>
#include <cstddef>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace alaya {

enum class LaserBuildPhase : uint8_t {
  kPca = 0,
  kPartition,
  kShardBuilds,
  kMerge,
  kMedoids,
  kQGBuild,
  kCount
};

enum class PhaseStatus : uint8_t { kPending = 0, kInProgress, kCompleted };

class BuildState {
 public:
  explicit BuildState(std::filesystem::path path) : path_(std::move(path)) { reset(""); }

  [[nodiscard]] static auto phase_name(LaserBuildPhase phase) -> std::string_view {
    static constexpr std::array<std::string_view, static_cast<size_t>(LaserBuildPhase::kCount)>
        kNames = {"pca", "partition", "shard_builds", "merge", "medoids", "qg_build"};
    return kNames[static_cast<size_t>(phase)];
  }

  [[nodiscard]] static auto phase_status_name(PhaseStatus status) -> std::string_view {
    switch (status) {
      case PhaseStatus::kPending:
        return "pending";
      case PhaseStatus::kInProgress:
        return "in_progress";
      case PhaseStatus::kCompleted:
        return "completed";
    }
    return "pending";
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path & { return path_; }
  [[nodiscard]] auto params_hash() const -> const std::string & { return params_hash_; }
  [[nodiscard]] auto exists() const -> bool { return std::filesystem::exists(path_); }
  [[nodiscard]] auto shard_count() const -> size_t { return shard_completed_.size(); }

  [[nodiscard]] auto is_completed(LaserBuildPhase phase) const -> bool {
    return status(phase) == PhaseStatus::kCompleted;
  }

  [[nodiscard]] auto status(LaserBuildPhase phase) const -> PhaseStatus {
    return phases_[static_cast<size_t>(phase)];
  }

  [[nodiscard]] auto is_shard_completed(size_t shard_id) const -> bool {
    return shard_id < shard_completed_.size() && shard_completed_[shard_id];
  }

  [[nodiscard]] auto completed_shards() const -> size_t {
    size_t count = 0;
    for (bool completed : shard_completed_) {
      count += completed ? 1U : 0U;
    }
    return count;
  }

  [[nodiscard]] auto all_completed() const -> bool {
    return std::ranges::all_of(phases_, [](PhaseStatus status_value) {
      return status_value == PhaseStatus::kCompleted;
    });
  }

  auto reset(std::string params_hash) -> void {
    params_hash_ = std::move(params_hash);
    phases_.fill(PhaseStatus::kPending);
    shard_completed_.clear();
  }

  auto load_or_reset(const std::string &params_hash) -> bool {
    if (!exists()) {
      reset(params_hash);
      save();
      return false;
    }

    load();
    if (params_hash_ != params_hash) {
      reset(params_hash);
      save();
      return false;
    }
    return true;
  }

  void load() {
    if (!exists()) {
      throw std::runtime_error("BuildState file does not exist: " + path_.string());
    }

    std::ifstream input(path_);
    if (!input) {
      throw std::runtime_error("Failed to open build state: " + path_.string());
    }

    std::ostringstream stream;
    stream << input.rdbuf();
    auto json = stream.str();

    params_hash_ = extract_string(json, "parameter_hash");
    for (size_t i = 0; i < static_cast<size_t>(LaserBuildPhase::kCount); ++i) {
      auto phase = static_cast<LaserBuildPhase>(i);
      auto parsed = parse_status(extract_string(json, std::string(phase_name(phase))));
      phases_[i] = parsed == PhaseStatus::kInProgress ? PhaseStatus::kPending : parsed;
    }

    auto shard_ids = extract_int_array(json, "completed_shards");
    size_t shard_count = static_cast<size_t>(extract_uint(json, "num_shards"));
    shard_completed_.assign(shard_count, false);
    for (size_t shard_id : shard_ids) {
      if (shard_id < shard_completed_.size()) {
        shard_completed_[shard_id] = true;
      }
    }
  }

  void save() const {
    auto dir = path_.parent_path();
    if (!dir.empty()) {
      std::filesystem::create_directories(dir);
    }

    std::ofstream output(path_, std::ios::trunc);
    if (!output) {
      throw std::runtime_error("Failed to write build state: " + path_.string());
    }

    output << "{\n";
    output << R"(  "parameter_hash": ")" << params_hash_ << "\",\n";
    output << "  \"phases\": {\n";
    for (size_t i = 0; i < static_cast<size_t>(LaserBuildPhase::kCount); ++i) {
      auto phase = static_cast<LaserBuildPhase>(i);
      output << "    \"" << phase_name(phase) << "\": \"" << phase_status_name(phases_[i]) << "\"";
      output << (i + 1 == static_cast<size_t>(LaserBuildPhase::kCount) ? "\n" : ",\n");
    }
    output << "  },\n";
    output << "  \"shards\": {\n";
    output << "    \"num_shards\": " << shard_completed_.size() << ",\n";
    output << "    \"completed_shards\": [";
    bool first = true;
    for (size_t shard_id = 0; shard_id < shard_completed_.size(); ++shard_id) {
      if (!shard_completed_[shard_id]) {
        continue;
      }
      output << (first ? "" : ", ") << shard_id;
      first = false;
    }
    output << "]\n";
    output << "  }\n";
    output << "}\n";
  }

  void ensure_shard_count(size_t shard_count) {
    if (shard_completed_.size() == shard_count) {
      return;
    }
    shard_completed_.assign(shard_count, false);
    save();
  }

  void mark_in_progress(LaserBuildPhase phase) {
    phases_[static_cast<size_t>(phase)] = PhaseStatus::kInProgress;
    save();
  }

  void mark_pending(LaserBuildPhase phase) {
    phases_[static_cast<size_t>(phase)] = PhaseStatus::kPending;
    save();
  }

  void mark_completed(LaserBuildPhase phase) {
    phases_[static_cast<size_t>(phase)] = PhaseStatus::kCompleted;
    save();
  }

  void mark_shard_completed(size_t shard_id) {
    if (shard_id >= shard_completed_.size()) {
      throw std::out_of_range("Shard completion index out of range");
    }
    shard_completed_[shard_id] = true;
    save();
  }

 private:
  std::filesystem::path path_;
  std::string params_hash_;
  std::array<PhaseStatus, static_cast<size_t>(LaserBuildPhase::kCount)> phases_{};
  std::vector<bool> shard_completed_;

  [[nodiscard]] static auto extract_string(const std::string &json, const std::string &key)
      -> std::string {
    auto key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
      return "";
    }
    auto colon_pos = json.find(':', key_pos);
    auto first_quote = json.find('"', colon_pos + 1);
    auto second_quote = json.find('"', first_quote + 1);
    if (colon_pos == std::string::npos || first_quote == std::string::npos ||
        second_quote == std::string::npos) {
      return "";
    }
    return json.substr(first_quote + 1, second_quote - first_quote - 1);
  }

  [[nodiscard]] static auto extract_uint(const std::string &json, const std::string &key)
      -> uint64_t {
    auto key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
      return 0;
    }
    auto colon_pos = json.find(':', key_pos);
    if (colon_pos == std::string::npos) {
      return 0;
    }
    auto digit_start = json.find_first_of("0123456789", colon_pos + 1);
    if (digit_start == std::string::npos) {
      return 0;
    }
    auto digit_end = json.find_first_not_of("0123456789", digit_start);
    return static_cast<uint64_t>(std::stoull(json.substr(digit_start, digit_end - digit_start)));
  }

  [[nodiscard]] static auto extract_int_array(const std::string &json, const std::string &key)
      -> std::vector<size_t> {
    auto key_pos = json.find("\"" + key + "\"");
    if (key_pos == std::string::npos) {
      return {};
    }
    auto open_bracket = json.find('[', key_pos);
    auto close_bracket = json.find(']', open_bracket + 1);
    if (open_bracket == std::string::npos || close_bracket == std::string::npos) {
      return {};
    }

    std::vector<size_t> values;
    std::stringstream parser(json.substr(open_bracket + 1, close_bracket - open_bracket - 1));
    std::string token;
    while (std::getline(parser, token, ',')) {
      std::stringstream trimmed(token);
      size_t value = 0;
      if (trimmed >> value) {
        values.push_back(value);
      }
    }
    return values;
  }

  [[nodiscard]] static auto parse_status(const std::string &status) -> PhaseStatus {
    if (status == "completed") {
      return PhaseStatus::kCompleted;
    }
    if (status == "in_progress") {
      return PhaseStatus::kInProgress;
    }
    return PhaseStatus::kPending;
  }
};

}  // namespace alaya
