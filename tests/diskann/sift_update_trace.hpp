// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace alaya::diskann::bench {

enum class UpdateTraceMode {
  Random,
  YiSequential,
  InsertOnly,
};

inline std::string trace_mode_name(UpdateTraceMode mode) {
  switch (mode) {
    case UpdateTraceMode::Random:
      return "random";
    case UpdateTraceMode::YiSequential:
      return "yi_sequential";
    case UpdateTraceMode::InsertOnly:
      return "insert_only";
  }
  throw std::invalid_argument("unknown update trace mode");
}

struct UpdateTraceConfig {
  std::filesystem::path output_dir;
  std::string file_prefix = "round_";
  uint32_t initial_count = 0;
  uint32_t total_count = 0;
  uint32_t rounds = 0;
  uint32_t update_size = 0;
  uint64_t seed = 1234;
  UpdateTraceMode mode = UpdateTraceMode::Random;
};

inline void validate_trace_config(const UpdateTraceConfig &cfg) {
  if (cfg.output_dir.empty()) {
    throw std::invalid_argument("UpdateTraceConfig: output_dir is empty");
  }
  if (cfg.file_prefix.empty()) {
    throw std::invalid_argument("UpdateTraceConfig: file_prefix is empty");
  }
  if (cfg.initial_count == 0 || cfg.total_count <= cfg.initial_count) {
    throw std::invalid_argument("UpdateTraceConfig: invalid initial/total count");
  }
  if (cfg.rounds == 0 || cfg.update_size == 0) {
    throw std::invalid_argument("UpdateTraceConfig: rounds/update_size must be > 0");
  }
  const uint64_t needed = static_cast<uint64_t>(cfg.rounds) * cfg.update_size;
  const uint64_t reserve = static_cast<uint64_t>(cfg.total_count) - cfg.initial_count;
  if (needed > reserve) {
    throw std::invalid_argument("UpdateTraceConfig: not enough reserve ids for inserts");
  }
  if (cfg.update_size > cfg.initial_count) {
    throw std::invalid_argument("UpdateTraceConfig: update_size exceeds initial live set");
  }
  if (cfg.mode == UpdateTraceMode::YiSequential && needed >= cfg.initial_count) {
    throw std::invalid_argument("UpdateTraceConfig: Yi sequential deletes exceed initial live ids");
  }
}

inline void write_round_file(const std::filesystem::path &path,
                             const std::vector<uint32_t> &deletes,
                             const std::vector<uint32_t> &inserts) {
  if (deletes.size() != inserts.size()) {
    throw std::invalid_argument("write_round_file: delete/insert size mismatch");
  }
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("write_round_file: cannot open " + path.string());
  }
  const uint32_t n = static_cast<uint32_t>(deletes.size());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(deletes.data()),
            static_cast<std::streamsize>(deletes.size() * sizeof(uint32_t)));
  out.write(reinterpret_cast<const char *>(inserts.data()),
            static_cast<std::streamsize>(inserts.size() * sizeof(uint32_t)));
  if (!out) {
    throw std::runtime_error("write_round_file: write failed " + path.string());
  }
}

inline void write_manifest(const UpdateTraceConfig &cfg) {
  const std::filesystem::path path = cfg.output_dir / "manifest.txt";
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    throw std::runtime_error("write_manifest: cannot open " + path.string());
  }
  out << "format=yi_update_trace_v1\n"
      << "file_prefix=" << cfg.file_prefix << "\n"
      << "initial_count=" << cfg.initial_count << "\n"
      << "total_count=" << cfg.total_count << "\n"
      << "rounds=" << cfg.rounds << "\n"
      << "update_size=" << cfg.update_size << "\n"
      << "seed=" << cfg.seed << "\n"
      << "mode=" << trace_mode_name(cfg.mode) << "\n";
}

inline void generate_random_trace(const UpdateTraceConfig &cfg) {
  std::vector<uint32_t> live;
  live.reserve(cfg.initial_count);
  for (uint32_t id = 0; id < cfg.initial_count; ++id) {
    live.push_back(id);
  }

  std::vector<uint32_t> reserve;
  reserve.reserve(static_cast<size_t>(cfg.total_count) - cfg.initial_count);
  for (uint32_t id = cfg.initial_count; id < cfg.total_count; ++id) {
    reserve.push_back(id);
  }

  std::mt19937_64 rng(cfg.seed);
  std::shuffle(reserve.begin(), reserve.end(), rng);
  size_t next_insert = 0;

  for (uint32_t round = 0; round < cfg.rounds; ++round) {
    std::vector<uint32_t> deletes;
    std::vector<uint32_t> inserts;
    deletes.reserve(cfg.update_size);
    inserts.reserve(cfg.update_size);

    for (uint32_t i = 0; i < cfg.update_size; ++i) {
      std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
      const size_t pos = pick(rng);
      deletes.push_back(live[pos]);
      live[pos] = live.back();
      live.pop_back();
    }

    for (uint32_t i = 0; i < cfg.update_size; ++i) {
      const uint32_t id = reserve[next_insert++];
      inserts.push_back(id);
      live.push_back(id);
    }

    write_round_file(cfg.output_dir / (cfg.file_prefix + std::to_string(round)), deletes, inserts);
  }

  write_manifest(cfg);
}

inline void generate_yi_sequential_trace(const UpdateTraceConfig &cfg) {
  uint32_t next_delete = 1;
  uint32_t next_insert = cfg.initial_count;
  for (uint32_t round = 0; round < cfg.rounds; ++round) {
    std::vector<uint32_t> deletes;
    std::vector<uint32_t> inserts;
    deletes.reserve(cfg.update_size);
    inserts.reserve(cfg.update_size);
    for (uint32_t i = 0; i < cfg.update_size; ++i) {
      deletes.push_back(next_delete++);
      inserts.push_back(next_insert++);
    }
    write_round_file(cfg.output_dir / (cfg.file_prefix + std::to_string(round)), deletes, inserts);
  }

  write_manifest(cfg);
}

inline void generate_insert_only_trace(const UpdateTraceConfig &cfg) {
  uint32_t next_insert = cfg.initial_count;
  for (uint32_t round = 0; round < cfg.rounds; ++round) {
    const uint32_t delete_count = 0;
    const uint32_t insert_count = cfg.update_size;
    const std::filesystem::path path =
        cfg.output_dir / (cfg.file_prefix + std::to_string(round));
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      throw std::runtime_error("generate_insert_only_trace: cannot open " + path.string());
    }
    out.write(reinterpret_cast<const char *>(&delete_count), sizeof(delete_count));
    out.write(reinterpret_cast<const char *>(&insert_count), sizeof(insert_count));
    for (uint32_t i = 0; i < insert_count; ++i) {
      out.write(reinterpret_cast<const char *>(&next_insert), sizeof(next_insert));
      ++next_insert;
    }
    if (!out) {
      throw std::runtime_error("generate_insert_only_trace: write failed " + path.string());
    }
  }
  write_manifest(cfg);
}

inline void generate_update_trace(const UpdateTraceConfig &cfg) {
  validate_trace_config(cfg);
  std::filesystem::create_directories(cfg.output_dir);
  switch (cfg.mode) {
    case UpdateTraceMode::Random:
      return generate_random_trace(cfg);
    case UpdateTraceMode::YiSequential:
      return generate_yi_sequential_trace(cfg);
    case UpdateTraceMode::InsertOnly:
      return generate_insert_only_trace(cfg);
  }
  throw std::invalid_argument("generate_update_trace: unknown trace mode");
}

}  // namespace alaya::diskann::bench
