// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace alaya::laser::bench {

struct OracleConfig {
  std::string prefix;
  std::string base;
  uint32_t degree = 32;
  uint32_t main_dim = 0;
  size_t samples = 1000;
  size_t ef_maintenance = 200;
  size_t prune_pool_cap = 300;
  size_t r_target = 0;
  float alpha = 1.2F;
  uint64_t seed = 42;
};

int run_fastscan_oracle(const OracleConfig &config);
int run_twohop_oracle(const OracleConfig &config);

}  // namespace alaya::laser::bench
