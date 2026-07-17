// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file gen_ip_official_fixture.cpp
 * @brief OFFLINE generator for the official RaBitQ-Library METRIC_IP fixture.
 *
 * This tool is deliberately NOT on the CI build graph (it is not referenced by
 * any CMakeLists). It links the official RaBitQ-Library at the pinned commit and
 * calls one_bit_code_with_factor(..., METRIC_IP) on inputs produced by the SAME
 * deterministic mapping (ip_fixture_common.hpp::make_row_inputs) the CI tests use,
 * then emits tests/laser/space/ip_official_fixture.tsv. The CI tests only ever
 * read that file (B-LIP-08).
 *
 * Regenerate (from the worktree root) with:
 *
 *   BL=/home/huangliang/workspace/alaya-dev/baselines/RaBitQ-Library
 *   g++ -std=c++17 -O2 -fopenmp \
 *       -I "$BL/include" -I tests/laser/space \
 *       tests/laser/space/tools/gen_ip_official_fixture.cpp -o /tmp/gen_ip_fixture
 *   /tmp/gen_ip_fixture > tests/laser/space/ip_official_fixture.tsv
 *
 * The pinned commit is b1f613d7412a041000d1e71aaa323d3e7554e733; the emitted
 * header records it and the CI loader hard-fails on any mismatch.
 */

#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ip_fixture_common.hpp"                       // alaya::ip_fixture (shared, std-only)
#include "rabitqlib/quantization/rabitq_impl.hpp"      // official METRIC_IP factor API

namespace ipf = alaya::ip_fixture;

int main() {
  std::cout << "# RaBitQ-Library official METRIC_IP factor fixture (IMMUTABLE; read-only in CI).\n";
  std::cout << "# commit: " << ipf::kPinnedRabitqCommit << "\n";
  std::cout << "# generator: tests/laser/space/tools/gen_ip_official_fixture.cpp\n";
  std::cout << "# api: rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor<float>"
               "(data, centroid, dim, bits, f_add, f_rescale, f_error, METRIC_IP)\n";
  std::cout << "# command: g++ -std=c++17 -O2 -fopenmp -I <RaBitQ-Library>/include "
               "-I tests/laser/space tests/laser/space/tools/gen_ip_official_fixture.cpp "
               "-o /tmp/gen_ip_fixture && /tmp/gen_ip_fixture > "
               "tests/laser/space/ip_official_fixture.tsv\n";
  std::cout << "# input_mapping: make_row_inputs(seed,dim) in ip_fixture_common.hpp -- integer LCG "
               "x=x*1664525+1013904223, value=((x>>24 & 0xFF)-128)/16 (exact float), sequences "
               "data[dim] then centroid[dim] then query[dim]; query unused for factors.\n";
  std::cout << "# input_fnv: FNV-1a64 over raw float bytes of data++centroid (locks input reproduction).\n";
  std::cout << "# columns: dim  trial  seed  input_fnv  popcount  f_add  f_rescale  f_error\n";

  std::cout << std::setprecision(9);
  for (const size_t dim : ipf::fixture_dims()) {
    for (int trial = 0; trial < ipf::kTrialsPerDim; ++trial) {
      // Deterministic, distinct per row. The value is stored, so the CI test
      // reproduces inputs from the stored seed without knowing this formula.
      const uint32_t seed =
          (0x9E3779B9U ^ (static_cast<uint32_t>(dim) * 2654435761U)) +
          (static_cast<uint32_t>(trial) * 40503U) + 0x1234567U;

      const ipf::RowInputs in = ipf::make_row_inputs(seed, dim);

      std::vector<int> bits(dim);
      float f_add = 0.0F;
      float f_rescale = 0.0F;
      float f_error = 0.0F;
      rabitqlib::quant::rabitq_impl::one_bit::one_bit_code_with_factor<float>(
          in.data.data(), in.centroid.data(), dim, bits.data(), f_add, f_rescale, f_error,
          rabitqlib::METRIC_IP);

      int popcount = 0;
      for (int b : bits) {
        popcount += b;
      }
      const uint64_t fnv = ipf::input_fingerprint(in.data, in.centroid);

      std::cout << dim << '\t' << trial << '\t' << seed << '\t' << fnv << '\t' << popcount << '\t'
                << f_add << '\t' << f_rescale << '\t' << f_error << '\n';
    }
  }
  return 0;
}
