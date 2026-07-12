// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "sift_update_trace.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct FbinHeader {
  uint32_t n = 0;
  uint32_t dim = 0;
};

FbinHeader read_fbin_header(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path.string());
  }
  int32_t n = 0;
  int32_t dim = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(n));
  in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
  if (!in || n <= 0 || dim <= 0) {
    throw std::runtime_error("bad fbin header: " + path.string());
  }
  return FbinHeader{static_cast<uint32_t>(n), static_cast<uint32_t>(dim)};
}

uint32_t parse_u32(const std::string &value, const char *name) {
  const unsigned long parsed = std::stoul(value);
  if (parsed > UINT32_MAX) {
    throw std::invalid_argument(std::string(name) + " overflows uint32");
  }
  return static_cast<uint32_t>(parsed);
}

uint64_t parse_u64(const std::string &value, const char *name) {
  try {
    return std::stoull(value);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
}

alaya::diskann::bench::UpdateTraceMode parse_mode(const std::string &value) {
  if (value == "random") {
    return alaya::diskann::bench::UpdateTraceMode::Random;
  }
  if (value == "yi_sequential") {
    return alaya::diskann::bench::UpdateTraceMode::YiSequential;
  }
  if (value == "insert_only") {
    return alaya::diskann::bench::UpdateTraceMode::InsertOnly;
  }
  throw std::invalid_argument("invalid --mode: " + value);
}

void print_usage(const char *argv0) {
  std::cerr << "Usage:\n"
            << "  " << argv0 << " <dataset_data_dir> <output_dir> [flags]\n\n"
            << "Flags:\n"
            << "  --initial_n N      initial live vectors (default: total - rounds*update_size)\n"
            << "  --rounds N         update rounds (default: 10)\n"
            << "  --update_size N    deletes and inserts per round (default: 10000)\n"
            << "  --seed N           deterministic RNG seed (default: 1234)\n"
            << "  --mode MODE        random, yi_sequential, or insert_only (default: random)\n"
            << "  --prefix STR       round filename prefix (default: round_)\n";
}

std::filesystem::path resolve_base_path(const std::filesystem::path &data_dir) {
  for (const char *name : {"sift_base.fbin", "gist_base.fbin"}) {
    const std::filesystem::path path = data_dir / name;
    if (std::filesystem::exists(path)) {
      return path;
    }
  }
  throw std::runtime_error("cannot find sift_base.fbin or gist_base.fbin under " +
                           data_dir.string());
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      print_usage(argv[0]);
      return 2;
    }

    const std::filesystem::path data_dir = argv[1];
    alaya::diskann::bench::UpdateTraceConfig cfg;
    cfg.output_dir = argv[2];

    bool initial_set = false;
    for (int i = 3; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--initial_n" && i + 1 < argc) {
        cfg.initial_count = parse_u32(argv[++i], "--initial_n");
        initial_set = true;
      } else if (arg == "--rounds" && i + 1 < argc) {
        cfg.rounds = parse_u32(argv[++i], "--rounds");
      } else if (arg == "--update_size" && i + 1 < argc) {
        cfg.update_size = parse_u32(argv[++i], "--update_size");
      } else if (arg == "--seed" && i + 1 < argc) {
        cfg.seed = parse_u64(argv[++i], "--seed");
      } else if (arg == "--mode" && i + 1 < argc) {
        cfg.mode = parse_mode(argv[++i]);
      } else if (arg == "--prefix" && i + 1 < argc) {
        cfg.file_prefix = argv[++i];
      } else {
        throw std::invalid_argument("unknown or incomplete argument: " + arg);
      }
    }

    if (cfg.rounds == 0) {
      cfg.rounds = 10;
    }
    if (cfg.update_size == 0) {
      cfg.update_size = 10000;
    }

    const FbinHeader base = read_fbin_header(resolve_base_path(data_dir));
    cfg.total_count = base.n;
    if (!initial_set) {
      const uint64_t reserve = static_cast<uint64_t>(cfg.rounds) * cfg.update_size;
      if (reserve >= cfg.total_count) {
        throw std::invalid_argument("rounds*update_size must be smaller than total SIFT size");
      }
      cfg.initial_count = static_cast<uint32_t>(cfg.total_count - reserve);
    }

    alaya::diskann::bench::generate_update_trace(cfg);
    std::cout << "[trace] wrote " << cfg.rounds << " rounds to " << cfg.output_dir << "\n"
              << "[trace] total=" << cfg.total_count << " initial=" << cfg.initial_count
              << " update_size=" << cfg.update_size << " seed=" << cfg.seed
              << " mode=" << alaya::diskann::bench::trace_mode_name(cfg.mode) << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "[trace] ERROR: " << e.what() << "\n";
    return 1;
  }
}
