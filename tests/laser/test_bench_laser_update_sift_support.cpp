// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "../../benchmarks/laser/bench_laser_update_sift_support.hpp"

namespace alaya::laser::bench {
namespace {

class FbinFile {
 public:
  FbinFile(int32_t n, int32_t dim, const std::vector<float> &data)
      : path_(std::filesystem::temp_directory_path() /
              ("mapped_float_matrix_" + std::to_string(::getpid()) + "_" +
               std::to_string(next_id_++) + ".fbin")) {
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create fbin fixture");
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char *>(data.data()),
              static_cast<std::streamsize>(data.size() * sizeof(float)));
  }

  ~FbinFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  [[nodiscard]] std::string path() const { return path_.string(); }

 private:
  std::filesystem::path path_;
  static inline uint64_t next_id_ = 0;
};

TEST(MappedFloatMatrixTest, ParsesFbinHeader) {
  const FbinFile file(2, 3, {0, 1, 2, 3, 4, 5});
  const MappedFloatMatrix matrix(file.path());
  EXPECT_EQ(matrix.n, 2U);
  EXPECT_EQ(matrix.dim, 3U);
}

TEST(MappedFloatMatrixTest, ReturnsRequestedRow) {
  const FbinFile file(2, 3, {0.25F, 1.25F, 2.25F, 3.25F, 4.25F, 5.25F});
  const MappedFloatMatrix matrix(file.path());
  EXPECT_FLOAT_EQ(matrix.row(0)[2], 2.25F);
  EXPECT_FLOAT_EQ(matrix.row(1)[0], 3.25F);
  EXPECT_FLOAT_EQ(matrix.row(1)[2], 5.25F);
}

TEST(MappedFloatMatrixTest, RejectsOutOfBoundsRow) {
  const FbinFile file(1, 2, {10.0F, 20.0F});
  const MappedFloatMatrix matrix(file.path());
  EXPECT_THROW(static_cast<void>(matrix.row(1)), std::out_of_range);
}

}  // namespace
}  // namespace alaya::laser::bench
