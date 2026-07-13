// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

// Unit tests for the LASER streaming-update research prototype
// (qg_updater.hpp). The load-bearing properties:
//   1. unpack_codes_block is the exact inverse of pack_codes (32-slot block).
//   2. QGUpdater::assemble_row reproduces builder-written rows byte-for-byte.
//   3. A slot patch leaves the row byte-identical to a from-scratch rebuild
//      with the same neighbor id sequence (patch == rebuild property).
//   4. Ghost-slot detection distinguishes zero-padded tail slots.
//   5. Inserted vectors are searchable after reload; tombstoned ids are
//      filtered from results while still being traversable.
//   6. V2 trailers/A-B superblocks migrate, round-trip, and recover correctly.
//   7. Consolidated PIDs are reused dark-until-publish and bound file growth.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../bench_laser_update_sift_support.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
// R32/d64 produces a byte-perfect 4*1024B page with no trailer slack and is
// intentionally rejected by format v2.  R64 keeps the same test dimension
// while leaving 512B of slack for two row trailers.
constexpr size_t kDeg = 64;
constexpr size_t kN = 2000;

#if defined(__SANITIZE_THREAD__)
constexpr bool kRunningTsan = true;
#elif defined(__has_feature)
  #if __has_feature(thread_sanitizer)
constexpr bool kRunningTsan = true;
  #else
constexpr bool kRunningTsan = false;
  #endif
#else
constexpr bool kRunningTsan = false;
#endif

std::vector<float> make_data(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0F, 1.0F);
  std::vector<float> data(n * dim);
  for (auto &v : data) {
    v = dist(gen);
  }
  return data;
}

void write_fbin(const std::string &path, const float *data, int32_t n, int32_t dim) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&n), 4);
  out.write(reinterpret_cast<const char *>(&dim), 4);
  out.write(reinterpret_cast<const char *>(data),
            static_cast<std::streamsize>(sizeof(float) * n * dim));
}

void write_ring_vamana(const std::string &path, size_t n, uint32_t degree) {
  const size_t header_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
  const size_t record_size = sizeof(uint32_t) * (static_cast<size_t>(degree) + 1);
  const size_t expected_file_size = header_size + n * record_size;
  const uint32_t start = 0;
  const size_t frozen_points = 0;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(out.is_open());
  out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(expected_file_size));
  out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
  out.write(reinterpret_cast<const char *>(&start), sizeof(start));
  out.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
  for (uint32_t i = 0; i < n; ++i) {
    out.write(reinterpret_cast<const char *>(&degree), sizeof(degree));
    for (uint32_t j = 0; j < degree; ++j) {
      const uint32_t neighbor = (i + j + 1) % static_cast<uint32_t>(n);
      out.write(reinterpret_cast<const char *>(&neighbor), sizeof(neighbor));
    }
  }
}

void write_uniform_vamana(const std::string &path,
                          size_t n,
                          uint32_t max_degree,
                          uint32_t row_degree) {
  if (row_degree == 0 || row_degree > max_degree || row_degree >= n) {
    throw std::invalid_argument("write_uniform_vamana: invalid degree geometry");
  }
  const size_t header_size = sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);
  const size_t record_size = sizeof(uint32_t) * (static_cast<size_t>(row_degree) + 1);
  const size_t expected_file_size = header_size + n * record_size;
  const uint32_t start = 0;
  const size_t frozen_points = 0;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) throw std::runtime_error("cannot create uniform Vamana fixture");
  out.write(reinterpret_cast<const char *>(&expected_file_size), sizeof(expected_file_size));
  out.write(reinterpret_cast<const char *>(&max_degree), sizeof(max_degree));
  out.write(reinterpret_cast<const char *>(&start), sizeof(start));
  out.write(reinterpret_cast<const char *>(&frozen_points), sizeof(frozen_points));
  for (uint32_t i = 0; i < n; ++i) {
    out.write(reinterpret_cast<const char *>(&row_degree), sizeof(row_degree));
    for (uint32_t j = 0; j < row_degree; ++j) {
      const uint32_t neighbor = (i + j + 1) % static_cast<uint32_t>(n);
      out.write(reinterpret_cast<const char *>(&neighbor), sizeof(neighbor));
    }
  }
}

struct TinyIndex {
  std::filesystem::path dir;
  std::string prefix;
  std::string v1_prefix;
  std::vector<float> data;

  static TinyIndex build(uint32_t seed) {
    TinyIndex t;
    t.dir = std::filesystem::temp_directory_path() /
            ("qg_updater_test_" + std::to_string(::getpid()) + "_" + std::to_string(seed));
    std::filesystem::create_directories(t.dir);
    t.prefix = (t.dir / "tiny").string();
    t.data = make_data(kN, kDim, seed);

    alaya::vamana::VamanaBuildParams vp;
    vp.R = kDeg;
    vp.L = 64;
    vp.alpha = 1.2F;
    // Vamana's medoid reduction is intentionally outside this updater test's
    // TSan scope and has a known racy accumulator; serialize fixture creation
    // so sanitizer findings belong to the concurrent updater operations below.
    vp.num_threads = kRunningTsan ? 1 : 4;
    alaya::vamana::VamanaBuilder vb(t.data.data(), kN, kDim, vp);
    vb.build();
    const std::string vamana_path = t.prefix + "_vamana.index";
    alaya::vamana::save_graph(vb.graph(), vamana_path, kDeg, vb.medoid());

    write_fbin(t.prefix + "_pca_base.fbin", t.data.data(), kN, kDim);

    QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
    QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/kRunningTsan ? 1 : 4);
    builder.build(vamana_path.c_str(), t.prefix.c_str());

    // Preserve an immutable v1 artifact even though most updater tests migrate
    // their working copy on construction.
    const std::string suffix =
        "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
    t.v1_prefix = (t.dir / "v1_snapshot").string();
    for (const std::string &ext :
         {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
      std::filesystem::copy_file(t.prefix + ext,
                                 t.v1_prefix + ext,
                                 std::filesystem::copy_options::overwrite_existing);
    }
    return t;
  }

  ~TinyIndex() {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
  }
};

std::vector<char> read_node_row(const QuantizedGraph &qg_meta,
                                const std::string &index_path,
                                PID id,
                                size_t node_len,
                                size_t page_size,
                                size_t npp) {
  (void)qg_meta;
  const int fd = ::open(index_path.c_str(), O_RDONLY);
  EXPECT_GE(fd, 0);
  std::vector<char> page(page_size);
  const uint64_t off = kSectorLen + page_size * (static_cast<uint64_t>(id) / npp);
  EXPECT_EQ(::pread(fd, page.data(), page_size, static_cast<off_t>(off)),
            static_cast<ssize_t>(page_size));
  ::close(fd);
  return {page.begin() + static_cast<int64_t>((id % npp) * node_len),
          page.begin() + static_cast<int64_t>((id % npp + 1) * node_len)};
}

uint64_t fnv1a(const void *data, size_t size, uint64_t hash = 14695981039346656037ULL) {
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::vector<uint64_t> logical_slot_hashes(const char *row,
                                          size_t padded_dim,
                                          size_t degree,
                                          size_t code_off,
                                          size_t factor_off,
                                          size_t neighbor_off) {
  const size_t words = padded_dim / 64;
  std::vector<uint64_t> hashes(degree, 14695981039346656037ULL);
  std::vector<uint64_t> bins(kBatchSize * words);
  const auto *codes = reinterpret_cast<const uint8_t *>(row + code_off);
  const auto *factors = reinterpret_cast<const float *>(row + factor_off);
  const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off);
  for (size_t block = 0; block < degree / kBatchSize; ++block) {
    unpack_codes_block(padded_dim, codes + block * padded_dim * 4, bins.data());
    for (size_t local = 0; local < kBatchSize; ++local) {
      const size_t slot = block * kBatchSize + local;
      uint64_t hash = fnv1a(bins.data() + local * words, words * sizeof(uint64_t));
      hash = fnv1a(&factors[slot], sizeof(float), hash);
      hash = fnv1a(&factors[degree + slot], sizeof(float), hash);
      hash = fnv1a(&factors[2 * degree + slot], sizeof(float), hash);
      hashes[slot] = fnv1a(&ids[slot], sizeof(PID), hash);
    }
  }
  return hashes;
}

std::vector<uint64_t> unpacked_row_codes(const char *row,
                                         size_t padded_dim,
                                         size_t degree,
                                         size_t code_off) {
  const size_t words = padded_dim / 64;
  std::vector<uint64_t> codes(degree * words);
  const auto *packed = reinterpret_cast<const uint8_t *>(row + code_off);
  for (size_t block = 0; block < degree / kBatchSize; ++block) {
    unpack_codes_block(padded_dim,
                       packed + block * padded_dim * 4,
                       codes.data() + block * kBatchSize * words);
  }
  return codes;
}

std::vector<uint64_t> index_page_hashes(const std::string &path, size_t page_size) {
  const uintmax_t size = std::filesystem::file_size(path);
  if (size < kSectorLen || (size - kSectorLen) % page_size != 0) {
    throw std::runtime_error("patch audit: malformed terminal index size");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) throw std::runtime_error("patch audit: cannot open terminal index");
  in.seekg(kSectorLen);
  std::vector<uint64_t> hashes;
  std::vector<char> page(page_size);
  for (uintmax_t off = kSectorLen; off < size; off += page_size) {
    in.read(page.data(), static_cast<std::streamsize>(page.size()));
    if (in.gcount() != static_cast<std::streamsize>(page.size())) {
      throw std::runtime_error("patch audit: short terminal page read");
    }
    hashes.push_back(fnv1a(page.data(), page.size()));
  }
  return hashes;
}

std::string index_suffix() {
  return "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
}

void copy_index_artifact(const std::string &from, const std::string &to) {
  const std::string suffix = index_suffix();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(from + ext,
                               to + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }
}

bool v1_ghost_slot(const char *row,
                   size_t slot,
                   size_t degree,
                   size_t padded_dim,
                   size_t code_off,
                   size_t factor_off,
                   size_t neighbor_off) {
  const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off);
  if (ids[slot] != 0) return false;
  const auto *fac = reinterpret_cast<const float *>(row + factor_off);
  if (fac[slot] != 0 || fac[degree + slot] != 0 || fac[2 * degree + slot] != 0) return false;
  const uint8_t *block =
      reinterpret_cast<const uint8_t *>(row + code_off) + (slot / kBatchSize) * padded_dim * 4;
  std::vector<uint64_t> bins(kBatchSize * padded_dim / 64);
  unpack_codes_block(padded_dim, block, bins.data());
  const uint64_t *words = bins.data() + (slot % kBatchSize) * (padded_dim / 64);
  for (size_t w = 0; w < padded_dim / 64; ++w) {
    if (words[w] != 0) return false;
  }
  return true;
}

TEST(QGUpdaterUnit, TrailerRoundTripThreeRowsPerPage) {
  constexpr size_t page_size = 4096;
  constexpr size_t npp = 3;
  constexpr size_t row_size = 1200;
  static_assert(npp * row_size + npp * sizeof(QGRowTrailer) <= page_size);
  std::vector<char> page(page_size, static_cast<char>(0x5a));
  const std::vector<char> row_bytes(page.begin(), page.begin() + npp * row_size);

  for (size_t slot = 0; slot < npp; ++slot) {
    qg_write_page_trailer(page.data(),
                          page_size,
                          npp,
                          slot,
                          {static_cast<uint16_t>(7 + slot),
                           static_cast<uint16_t>(slot == 1 ? kQGRowTombstone : 0)});
  }
  for (size_t slot = 0; slot < npp; ++slot) {
    const QGRowTrailer got = qg_read_page_trailer(page.data(), page_size, npp, slot);
    EXPECT_EQ(got.valid_degree, 7 + slot);
    EXPECT_EQ(got.flags, slot == 1 ? kQGRowTombstone : 0);
    EXPECT_EQ(qg_page_trailer_offset(page_size, npp, slot), page_size - 12 + slot * 4);
  }
  EXPECT_TRUE(std::equal(row_bytes.begin(), row_bytes.end(), page.begin()))
      << "trailer writes must not touch any row body";
}

TEST(QGUpdaterUnit, UnpackRoundTrip) {
  std::mt19937_64 gen(42);
  for (size_t pd : {64UL, 128UL, 256UL, 1024UL, 2048UL}) {
    std::vector<uint64_t> bins(kBatchSize * pd / 64);
    for (auto &w : bins) {
      w = gen();
    }
    std::vector<uint8_t> packed(pd * 4);
    pack_codes(pd, bins.data(), kBatchSize, packed.data());
    std::vector<uint64_t> unpacked(bins.size());
    unpack_codes_block(pd, packed.data(), unpacked.data());
    EXPECT_EQ(bins, unpacked) << "pd=" << pd;

    // patch one slot then repack: must equal packing the modified originals
    std::vector<uint64_t> mod = bins;
    for (size_t w = 0; w < pd / 64; ++w) {
      mod[17 * (pd / 64) + w] = gen();
    }
    std::vector<uint8_t> repacked = packed;
    std::vector<uint64_t> tmp(bins.size());
    unpack_codes_block(pd, repacked.data(), tmp.data());
    std::copy(mod.begin() + 17 * (pd / 64),
              mod.begin() + 18 * (pd / 64),
              tmp.begin() + 17 * (pd / 64));
    pack_codes(pd, tmp.data(), kBatchSize, repacked.data());
    std::vector<uint8_t> expect(pd * 4);
    pack_codes(pd, mod.data(), kBatchSize, expect.data());
    EXPECT_EQ(repacked, expect) << "pd=" << pd;
  }
}

TEST(QGUpdaterUnit, EdgePayloadMatchesRabitqCodes) {
  const size_t pd = 128;
  auto vecs = make_data(2, pd, 9);
  RowMatrix<float> x(1, pd);
  RowMatrix<float> c(1, pd);
  for (size_t j = 0; j < pd; ++j) {
    x(0, static_cast<int64_t>(j)) = vecs[pd + j];
    c(0, static_cast<int64_t>(j)) = vecs[j];
  }
  std::vector<uint8_t> packed(pd * 4);
  float triple = 0;
  float dq = 0;
  float vq = 0;
  RowMatrix<float> x_copy = x;  // rabitq_codes mutates rows into residuals
  rabitq_codes(x_copy, c, packed.data(), &triple, &dq, &vq);

  EdgePayload p = make_edge_payload(&c(0, 0), &x(0, 0), pd, 0.0F);
  ASSERT_FALSE(p.degenerate);
  EXPECT_EQ(p.triple_x, triple);
  EXPECT_EQ(p.factor_dq, dq);
  EXPECT_EQ(p.factor_vq, vq);

  std::vector<uint64_t> bins(kBatchSize * pd / 64, 0);
  unpack_codes_block(pd, packed.data(), bins.data());
  for (size_t w = 0; w < pd / 64; ++w) {
    EXPECT_EQ(bins[w], p.bin[w]) << "word " << w;
  }
}

TEST(QGUpdaterUnit, ExactTwoRowPageMigratesWithReservedTrailersAndLegacyStillRejects) {
  constexpr size_t n = 96;
  constexpr size_t main_dim = 128;
  constexpr size_t full_dim = 256;
  constexpr size_t degree = 32;
  constexpr size_t node_len = 2048;
  const auto geometry = qg_page_geometry(node_len);
  ASSERT_EQ(geometry.node_per_page, 1U);
  ASSERT_EQ(geometry.page_size, kSectorLen);

  const auto dir = std::filesystem::temp_directory_path() /
                   ("qg_updater_exact_page_" + std::to_string(::getpid()));
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string prefix = (dir / "exact").string();
  const std::string legacy_prefix = (dir / "legacy").string();
  const std::string vamana_path = prefix + "_vamana.index";
  const std::string suffix = "_R32_MD128.index";
  const std::string index_path = prefix + suffix;
  const std::string legacy_path = legacy_prefix + suffix;

  const auto data = make_data(n, full_dim, 20260712);
  write_fbin(prefix + "_pca_base.fbin", data.data(), n, full_dim);
  write_ring_vamana(vamana_path, n, degree);
  {
    QuantizedGraph build_qg(n, degree, main_dim, full_dim, /*rotator_seed=*/19);
    QGBuilder builder(build_qg, /*ef_build=*/64, /*num_threads=*/kRunningTsan ? 1 : 2);
    builder.build(vamana_path.c_str(), prefix.c_str());
  }

  std::array<uint64_t, kSectorLen / sizeof(uint64_t)> metas{};
  {
    std::ifstream in(index_path, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    in.read(reinterpret_cast<char *>(metas.data()), kSectorLen);
  }
  ASSERT_EQ(metas[3], node_len);
  ASSERT_EQ(metas[4], geometry.node_per_page);
  ASSERT_EQ(metas[8], kSectorLen + n * geometry.page_size);

  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(prefix + ext,
                               legacy_prefix + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }
  // Synthesize the old, byte-full npp2 v1 geometry. Its row bytes are never
  // touched: the compatibility contract here is read-only load plus updater rejection.
  metas[4] = 2;
  metas[8] = kSectorLen + ((n + 1) / 2) * kSectorLen;
  {
    std::fstream legacy(legacy_path, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(legacy.is_open());
    legacy.write(reinterpret_cast<const char *>(metas.data()), kSectorLen);
  }
  std::filesystem::resize_file(legacy_path, metas[8]);

  {
    QuantizedGraph qg(n, degree, main_dim, full_dim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    UpdateParams params;
    params.direct_io = false;
    params.write_cache = false;
    QGUpdater upd(qg, params);
    EXPECT_EQ(upd.generation(), 1U);
    EXPECT_EQ(upd.trailer(0).valid_degree, degree);
    upd.tombstone(1);
    upd.checkpoint();
    EXPECT_NE(upd.trailer(1).flags & kQGRowTombstone, 0U);
  }

  {
    QuantizedGraph legacy_qg(n, degree, main_dim, full_dim);
    EXPECT_NO_THROW(legacy_qg.load_disk_index(legacy_prefix.c_str(), 0.0F));
    try {
      QGUpdater rejected(legacy_qg, UpdateParams{});
      FAIL() << "legacy zero-slack v1 unexpectedly migrated";
    } catch (const std::invalid_argument &e) {
      EXPECT_NE(std::string(e.what()).find("insufficient slack"), std::string::npos);
    }
  }
  std::filesystem::remove_all(dir);
}

class QGUpdaterIndexTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { tiny_ = new TinyIndex(TinyIndex::build(1234)); }
  static void TearDownTestSuite() {
    delete tiny_;
    tiny_ = nullptr;
  }
  static TinyIndex *tiny_;
};
TinyIndex *QGUpdaterIndexTest::tiny_ = nullptr;

TEST_F(QGUpdaterIndexTest, MigratesV1DegreesAndPreservesSearchExactly) {
  const std::string prefix = (tiny_->dir / "migrate").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  const std::string path = prefix + index_suffix();
  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  const size_t npp = std::max<size_t>(1, kSectorLen / node_len);
  const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
  const size_t code_off = kDim * sizeof(float);
  const size_t factor_off = (kDim + (kDim / 64) * 2 * kDeg) * sizeof(float);
  const size_t neighbor_off = factor_off + 3 * kDeg * sizeof(float);

  std::array<char, kSectorLen> header{};
  {
    std::ifstream in(path, std::ios::binary);
    ASSERT_TRUE(in.is_open());
    in.read(header.data(), header.size());
  }
  QGSuperblockV2 absent;
  EXPECT_EQ(select_qg_superblock(header.data(), absent), -1);
  EXPECT_FALSE(qg_header_has_v2_magic(header.data()));

  std::vector<uint16_t> expected_degree(kN);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  // Beam 1 makes completion order deterministic, so this is a byte-for-byte
  // result comparison rather than an asynchronous beam scheduling check.
  qg.set_params(100, 1, 1);
  for (PID id = 0; id < kN; ++id) {
    const auto row = read_node_row(qg, path, id, node_len, page_size, npp);
    for (size_t slot = 0; slot < kDeg; ++slot) {
      expected_degree[id] += static_cast<uint16_t>(
          !v1_ghost_slot(row.data(), slot, kDeg, kDim, code_off, factor_off, neighbor_off));
    }
  }
  std::vector<uint32_t> before(32 * 10);
  for (size_t i = 0; i < 32; ++i) {
    qg.search(tiny_->data.data() + i * kDim, 10, before.data() + i * 10);
  }

  QGUpdater upd(qg, UpdateParams{});
  EXPECT_EQ(upd.generation(), 1U);
  EXPECT_EQ(upd.active_superblock_slot(), 0);
  for (PID id = 0; id < kN; ++id) {
    EXPECT_EQ(upd.trailer(id).valid_degree, expected_degree[id]) << "id=" << id;
  }
  std::vector<uint32_t> after(before.size());
  for (size_t i = 0; i < 32; ++i) {
    qg.search(tiny_->data.data() + i * kDim, 10, after.data() + i * 10);
  }
  EXPECT_EQ(after, before);
}

TEST_F(QGUpdaterIndexTest, AssembleRowMatchesBuilder) {
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(tiny_->prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  QGUpdater upd(qg, UpdateParams{});

  const std::string index_path =
      tiny_->prefix + "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  const size_t npp = std::max<size_t>(1, 4096 / node_len);
  const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;

  size_t tested = 0;
  for (PID u = 0; u < 50 && tested < 20; ++u) {
    auto row = read_node_row(qg, index_path, u, node_len, page_size, npp);
    const auto *ids = reinterpret_cast<const PID *>(row.data() + upd.neighbor_off_bytes());
    std::vector<PID> nb_ids;
    std::vector<const float *> nb_vecs;
    bool usable = true;
    for (size_t j = 0; j < kDeg; ++j) {
      if (upd.is_v1_ghost_slot(row.data(), j)) {
        for (size_t k = j; k < kDeg; ++k) {
          if (!upd.is_v1_ghost_slot(row.data(), k)) {
            usable = false;  // hole in the middle — builder wrote contiguous, skip
          }
        }
        break;
      }
      nb_ids.push_back(ids[j]);
      nb_vecs.push_back(tiny_->data.data() + static_cast<size_t>(ids[j]) * kDim);
    }
    if (!usable || nb_ids.empty()) {
      continue;
    }
    std::vector<char> rebuilt(node_len);
    upd.assemble_row(rebuilt.data(),
                     tiny_->data.data() + static_cast<size_t>(u) * kDim,
                     nb_vecs,
                     nb_ids);
    EXPECT_EQ(std::memcmp(rebuilt.data(), row.data(), node_len), 0) << "node " << u;
    ++tested;
  }
  EXPECT_GE(tested, 10U);
  qg.set_result_filter(nullptr);
}

TEST_F(QGUpdaterIndexTest, BenchCacheCapPagesFlagReachesUpdater) {
  const std::string prefix = (tiny_->dir / "cache_cap_flag").string();
  copy_index_artifact(tiny_->prefix, prefix);

  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  UpdateParams params;
  const size_t default_cap = params.cache_cap_pages;
  bench::apply_cache_cap_pages(0, params);
  EXPECT_EQ(params.cache_cap_pages, default_cap);
  bench::apply_cache_cap_pages(bench::parse_cache_cap_pages("17"), params);

  QGUpdater upd(qg, params);
  EXPECT_EQ(upd.cache_cap_pages(), 17U);
  EXPECT_EQ(upd.maintenance_evict_stride(), 4096U);
  EXPECT_EQ(upd.pool_pages(), 0U);
}

TEST_F(QGUpdaterIndexTest, MaintenanceInPassEvictionBoundsPoolAndPreservesRows) {
  constexpr size_t kCacheCap = 32;
  constexpr size_t kStride = 1;

  struct Result {
    std::vector<uint64_t> page_hashes;
    size_t pool_pages = 0;
    uint64_t peak_pool_pages = 0;
  };

  auto run_consolidate = [&](const std::string &name, size_t stride, size_t cache_cap) {
    const std::string prefix = (tiny_->dir / name).string();
    copy_index_artifact(tiny_->v1_prefix, prefix);
    Result result;
    {
      QuantizedGraph qg(kN, kDeg, kDim, kDim);
      qg.load_disk_index(prefix.c_str(), 0.0F);
      UpdateParams params;
      params.cache_cap_pages = cache_cap;
      params.maintenance_evict_stride = stride;
      QGUpdater upd(qg, params);
      for (PID id = 200; id < 232; ++id) upd.tombstone(id);
      upd.consolidate(1, kDeg - 4, false);
      result.pool_pages = upd.pool_pages();
      result.peak_pool_pages = upd.stats().maintenance_peak_pool_pages;
    }
    result.page_hashes = index_page_hashes(prefix + index_suffix(), kSectorLen);
    return result;
  };

  auto run_garden = [&](const std::string &name, size_t stride, size_t cache_cap) {
    const std::string prefix = (tiny_->dir / name).string();
    copy_index_artifact(tiny_->v1_prefix, prefix);
    Result result;
    {
      QuantizedGraph qg(kN, kDeg, kDim, kDim);
      qg.load_disk_index(prefix.c_str(), 0.0F);
      UpdateParams params;
      params.cache_cap_pages = cache_cap;
      params.maintenance_evict_stride = stride;
      params.maintain_indegree = true;
      QGUpdater upd(qg, params);
      upd.init_indegree(1);
      GardenParams garden;
      garden.frac = 0.05;
      garden.ef_maintenance = 64;
      garden.pump_budget = 0;
      garden.r_target = kDeg - 4;
      garden.policy = GardenParams::Policy::kRandom;
      upd.garden(1, garden);
      result.pool_pages = upd.pool_pages();
      result.peak_pool_pages = upd.stats().maintenance_peak_pool_pages;
    }
    result.page_hashes = index_page_hashes(prefix + index_suffix(), kSectorLen);
    return result;
  };

  const Result consolidate_old = run_consolidate("maintenance_consolidate_old", 0, kCacheCap);
  const Result consolidate_new = run_consolidate("maintenance_consolidate_new", kStride, kCacheCap);
  EXPECT_GT(consolidate_old.pool_pages, kCacheCap);
  EXPECT_LE(consolidate_new.peak_pool_pages, kCacheCap);
  EXPECT_LE(consolidate_new.pool_pages, kCacheCap);
  EXPECT_EQ(consolidate_new.page_hashes, consolidate_old.page_hashes);
  const Result consolidate_unreachable =
      run_consolidate("maintenance_consolidate_unreachable", kStride, kN);
  EXPECT_EQ(consolidate_unreachable.peak_pool_pages, consolidate_unreachable.pool_pages);
  EXPECT_EQ(consolidate_unreachable.page_hashes, consolidate_old.page_hashes);

  const Result garden_old = run_garden("maintenance_garden_old", 0, kCacheCap);
  const Result garden_new = run_garden("maintenance_garden_new", kStride, kCacheCap);
  EXPECT_GT(garden_old.pool_pages, kCacheCap);
  EXPECT_LE(garden_new.peak_pool_pages, kCacheCap);
  EXPECT_LE(garden_new.pool_pages, kCacheCap);
  EXPECT_EQ(garden_new.page_hashes, garden_old.page_hashes);
  const Result garden_unreachable = run_garden("maintenance_garden_unreachable", kStride, kN);
  EXPECT_EQ(garden_unreachable.peak_pool_pages, garden_unreachable.pool_pages);
  EXPECT_EQ(garden_unreachable.page_hashes, garden_old.page_hashes);
}

TEST_F(QGUpdaterIndexTest, InsertPatchTombstoneEndToEnd) {
  // Work on a copy so other tests see the pristine index.
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string copy_prefix = (tiny_->dir / "copy").string();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(tiny_->prefix + ext,
                               copy_prefix + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }

  const size_t n_insert = 64;
  auto new_data = make_data(n_insert, kDim, 777);

  UpdateStats stats;
  {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(copy_prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);
    UpdateParams params;
    params.ef_insert = 64;
    params.backlink_mode = UpdateParams::Backlink::kAlphaEvict;
    QGUpdater upd(qg, params);
    for (size_t i = 0; i < n_insert; ++i) {
      const PID id = upd.insert(new_data.data() + i * kDim);
      EXPECT_EQ(id, kN + i);
    }
    upd.finalize();
    stats = upd.stats();
    EXPECT_EQ(stats.inserts, n_insert);
    EXPECT_GT(stats.free_slot_fills + stats.evictions, 0U);
  }

  // ---- patch == rebuild property on rows that gained a backlink ----
  {
    QuantizedGraph qg(kN + n_insert, kDeg, kDim, kDim);
    qg.load_disk_index(copy_prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);
    QGUpdater upd(qg, UpdateParams{});
    const std::string index_path = copy_prefix + suffix;
    const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
    const size_t npp = std::max<size_t>(1, 4096 / node_len);
    const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;

    auto vec_of = [&](PID id) -> const float * {
      return id < kN ? tiny_->data.data() + static_cast<size_t>(id) * kDim
                     : new_data.data() + static_cast<size_t>(id - kN) * kDim;
    };

    size_t patched_rows_checked = 0;
    for (PID u = 0; u < kN && patched_rows_checked < 10; ++u) {
      auto row = read_node_row(qg, index_path, u, node_len, page_size, npp);
      const auto *ids = reinterpret_cast<const PID *>(row.data() + upd.neighbor_off_bytes());
      bool has_new = false;
      for (size_t j = 0; j < kDeg; ++j) {
        if (!upd.is_v1_ghost_slot(row.data(), j) && ids[j] >= kN) {
          has_new = true;
        }
      }
      if (!has_new) {
        continue;
      }
      std::vector<PID> nb_ids;
      std::vector<const float *> nb_vecs;
      bool usable = true;
      for (size_t j = 0; j < kDeg; ++j) {
        if (upd.is_v1_ghost_slot(row.data(), j)) {
          for (size_t k = j; k < kDeg; ++k) {
            usable = usable && upd.is_v1_ghost_slot(row.data(), k);
          }
          break;
        }
        nb_ids.push_back(ids[j]);
        nb_vecs.push_back(vec_of(ids[j]));
      }
      if (!usable) {
        continue;
      }
      std::vector<char> rebuilt(node_len);
      upd.assemble_row(rebuilt.data(), vec_of(u), nb_vecs, nb_ids);
      // Regions must be identical except factors, which may differ in low
      // bits: under -Ofast the same rabitq_codes reduction inlined at two call
      // sites (patch vs rebuild) can round differently.
      EXPECT_EQ(std::memcmp(rebuilt.data(), row.data(), upd.factor_off_bytes()), 0)
          << "raw/codes differ on node " << u;
      EXPECT_EQ(std::memcmp(rebuilt.data() + upd.neighbor_off_bytes(),
                            row.data() + upd.neighbor_off_bytes(),
                            node_len - upd.neighbor_off_bytes()),
                0)
          << "neighbor ids differ on node " << u;
      const auto *fa = reinterpret_cast<const float *>(rebuilt.data() + upd.factor_off_bytes());
      const auto *fb = reinterpret_cast<const float *>(row.data() + upd.factor_off_bytes());
      for (size_t j = 0; j < 3 * kDeg; ++j) {
        const float denom = std::max(1.0F, std::max(std::fabs(fa[j]), std::fabs(fb[j])));
        EXPECT_LE(std::fabs(fa[j] - fb[j]) / denom, 1e-5F)
            << "factor " << j << " node " << u << " rebuilt=" << fa[j] << " ondisk=" << fb[j];
      }
      ++patched_rows_checked;
    }
    EXPECT_GE(patched_rows_checked, 3U);
  }

  // ---- inserted vectors searchable after reload; tombstones filtered ----
  {
    QuantizedGraph qg(kN + n_insert, kDeg, kDim, kDim);
    qg.load_disk_index(copy_prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);

    size_t found = 0;
    std::vector<uint32_t> res(10);
    for (size_t i = 0; i < n_insert; ++i) {
      qg.search(new_data.data() + i * kDim, 10, res.data());
      if (res[0] == kN + i) {
        ++found;
      }
    }
    EXPECT_GE(found, n_insert * 9 / 10) << "inserted vectors must be discoverable";

    // original vectors still searchable
    size_t found_old = 0;
    for (size_t i = 0; i < 100; ++i) {
      qg.search(tiny_->data.data() + i * kDim, 10, res.data());
      if (res[0] == i) {
        ++found_old;
      }
    }
    EXPECT_GE(found_old, 90U);

    // tombstone the first 8 inserted ids -> excluded from results
    std::unordered_set<PID> dead;
    for (size_t i = 0; i < 8; ++i) {
      dead.insert(static_cast<PID>(kN + i));
    }
    qg.set_result_filter(&dead);
    for (size_t i = 0; i < 8; ++i) {
      qg.search(new_data.data() + i * kDim, 10, res.data());
      for (auto r : res) {
        EXPECT_EQ(dead.count(r), 0U) << "tombstoned id leaked into results";
      }
    }
    qg.set_result_filter(nullptr);
  }
}

TEST(QGUpdaterPd512PatchAudit, FillEvictRebuildResidualAndParallelPageHashes) {
  constexpr size_t raw_dim = 960;
  constexpr size_t main_dim = 512;
  constexpr size_t residual_dim = raw_dim - main_dim;
  constexpr size_t degree = 64;
  constexpr size_t initial_degree = 63;
  constexpr size_t base_n = 2048;
  constexpr size_t insert_n = 384;
  constexpr size_t batch_size = 64;
  constexpr size_t total_n = base_n + insert_n;
  constexpr size_t node_len = (32 * raw_dim + 128 * degree + degree * main_dim) / 8;
  const QGPageGeometry geometry = qg_page_geometry(node_len);
  ASSERT_EQ(geometry.node_per_page, 1U);
  ASSERT_EQ(geometry.page_size, 3 * kSectorLen);

  const auto dir = std::filesystem::temp_directory_path() /
                   ("qg_updater_pd512_patch_audit_" + std::to_string(::getpid()));
  struct Cleanup {
    std::filesystem::path dir;
    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove_all(dir, ec);
    }
  } cleanup{dir};
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);

  const std::string source_prefix = (dir / "source").string();
  const std::string vamana_path = source_prefix + "_vamana.index";
  const std::string suffix = "_R64_MD512.index";
  const auto base_data = make_data(base_n, raw_dim, 512448);
  const auto insert_data = make_data(insert_n, raw_dim, 96064);
  write_fbin(source_prefix + "_pca_base.fbin", base_data.data(), base_n, raw_dim);
  // R64 rows start at degree 63: the first successful backlink is a fill and
  // later backlinks to the same row must take the full-row eviction path.
  write_uniform_vamana(vamana_path, base_n, degree, initial_degree);
  {
    QuantizedGraph qg(base_n, degree, main_dim, raw_dim, /*rotator_seed=*/29);
    QGBuilder builder(qg, /*ef_build=*/96, /*num_threads=*/kRunningTsan ? 1 : 4);
    builder.build(vamana_path.c_str(), source_prefix.c_str());
  }

  auto copy_artifact = [&](const std::string &to) {
    for (const std::string &ext :
         {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
      std::filesystem::copy_file(source_prefix + ext,
                                 to + ext,
                                 std::filesystem::copy_options::overwrite_existing);
    }
  };
  const std::string serial_prefix = (dir / "serial").string();
  const std::string parallel_prefix = (dir / "parallel").string();
  copy_artifact(serial_prefix);
  copy_artifact(parallel_prefix);

  struct AuditResult {
    UpdateStats stats;
    std::vector<uint64_t> page_hashes;
    size_t patched_rows = 0;
    size_t untouched_slots = 0;
  };

  auto run = [&](const std::string &prefix, int thread_count) -> AuditResult {
    QuantizedGraph qg(base_n, degree, main_dim, raw_dim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    UpdateParams params;
    params.ef_insert = 96;
    params.prune_pool_cap = 96;
    params.backlink_mode = UpdateParams::Backlink::kEvict;
    params.stage_backlinks = true;
    params.write_cache = true;
    params.max_points = total_n + 64;
    QGUpdater upd(qg, params);

    const std::string index_path = prefix + suffix;
    std::vector<std::vector<char>> baseline(total_n);
    std::vector<uint16_t> baseline_degree(total_n);
    for (PID id = 0; id < base_n; ++id) {
      baseline[id] =
          read_node_row(qg, index_path, id, node_len, geometry.page_size, geometry.node_per_page);
      baseline_degree[id] = upd.trailer(id).valid_degree;
      EXPECT_EQ(baseline_degree[id], initial_degree)
          << "pd512 audit migration degree mismatch id=" << id << " threads=" << thread_count;
    }

    for (size_t start = 0; start < insert_n; start += batch_size) {
      const size_t end = std::min(insert_n, start + batch_size);
#pragma omp parallel for num_threads(thread_count) schedule(dynamic)
      for (int64_t raw_i = static_cast<int64_t>(start); raw_i < static_cast<int64_t>(end);
           ++raw_i) {
        const size_t i = static_cast<size_t>(raw_i);
        upd.insert_with_id(insert_data.data() + i * raw_dim, static_cast<PID>(base_n + i));
      }
      // writeback exercises the real staged fill/evict patch RMW, then makes
      // the page snapshot below independent of the resident update cache.
      upd.writeback(static_cast<size_t>(thread_count));
      upd.publish(base_n + end);
      for (size_t i = start; i < end; ++i) {
        const PID id = static_cast<PID>(base_n + i);
        baseline[id] =
            read_node_row(qg, index_path, id, node_len, geometry.page_size, geometry.node_per_page);
        baseline_degree[id] = upd.trailer(id).valid_degree;
      }
    }
    upd.finalize();

    AuditResult result;
    result.stats = upd.stats();
    EXPECT_EQ(result.stats.inserts, insert_n) << "threads=" << thread_count;
    EXPECT_GT(result.stats.free_slot_fills, 0U)
        << "fill path was not exercised threads=" << thread_count;
    EXPECT_GT(result.stats.evictions, 0U)
        << "evict path was not exercised threads=" << thread_count;

    auto vec_of = [&](PID id) -> const float * {
      if (id < base_n) return base_data.data() + static_cast<size_t>(id) * raw_dim;
      if (id < total_n) {
        return insert_data.data() + (static_cast<size_t>(id) - base_n) * raw_dim;
      }
      return nullptr;
    };

    for (PID id = 0; id < total_n; ++id) {
      const auto current =
          read_node_row(qg, index_path, id, node_len, geometry.page_size, geometry.node_per_page);
      if (baseline[id].size() != node_len) {
        ADD_FAILURE() << "missing baseline id=" << id << " threads=" << thread_count;
        continue;
      }
      if (std::memcmp(baseline[id].data(), current.data(), node_len) == 0) continue;
      ++result.patched_rows;

      const uint16_t current_degree = upd.trailer(id).valid_degree;
      const auto *baseline_ids =
          reinterpret_cast<const PID *>(baseline[id].data() + upd.neighbor_off_bytes());
      const auto *current_ids =
          reinterpret_cast<const PID *>(current.data() + upd.neighbor_off_bytes());
      const auto before_hash = logical_slot_hashes(baseline[id].data(),
                                                   main_dim,
                                                   degree,
                                                   upd.code_off_bytes(),
                                                   upd.factor_off_bytes(),
                                                   upd.neighbor_off_bytes());
      const auto after_hash = logical_slot_hashes(current.data(),
                                                  main_dim,
                                                  degree,
                                                  upd.code_off_bytes(),
                                                  upd.factor_off_bytes(),
                                                  upd.neighbor_off_bytes());
      for (size_t slot = 0; slot < degree; ++slot) {
        const bool before_valid = slot < baseline_degree[id];
        const bool after_valid = slot < current_degree;
        const bool untouched = before_valid == after_valid &&
                               (!before_valid || baseline_ids[slot] == current_ids[slot]);
        if (untouched) {
          ++result.untouched_slots;
          EXPECT_EQ(before_hash[slot], after_hash[slot])
              << "untouched-slot hash mismatch id=" << id << " slot=" << slot
              << " threads=" << thread_count;
        }
      }

      EXPECT_EQ(std::memcmp(baseline[id].data() + main_dim * sizeof(float),
                            current.data() + main_dim * sizeof(float),
                            residual_dim * sizeof(float)),
                0)
          << "residual [main,raw) changed id=" << id << " threads=" << thread_count;

      std::vector<PID> neighbor_ids;
      std::vector<const float *> neighbor_vecs;
      neighbor_ids.reserve(current_degree);
      neighbor_vecs.reserve(current_degree);
      bool valid_ids = true;
      for (size_t slot = 0; slot < current_degree; ++slot) {
        const float *neighbor_vec = vec_of(current_ids[slot]);
        if (neighbor_vec == nullptr) {
          ADD_FAILURE() << "out-of-range neighbor id=" << current_ids[slot] << " owner=" << id
                        << " slot=" << slot << " threads=" << thread_count;
          valid_ids = false;
          break;
        }
        neighbor_ids.push_back(current_ids[slot]);
        neighbor_vecs.push_back(neighbor_vec);
      }
      if (!valid_ids) continue;

      std::vector<char> rebuilt(node_len);
      upd.assemble_row(rebuilt.data(), vec_of(id), neighbor_vecs, neighbor_ids);
      EXPECT_EQ(std::memcmp(current.data(), rebuilt.data(), raw_dim * sizeof(float)), 0)
          << "raw vector mismatch vs rebuild id=" << id << " threads=" << thread_count;
      EXPECT_EQ(std::memcmp(current.data() + upd.code_off_bytes(),
                            rebuilt.data() + upd.code_off_bytes(),
                            upd.factor_off_bytes() - upd.code_off_bytes()),
                0)
          << "packed code mismatch vs rebuild id=" << id << " threads=" << thread_count;

      const auto current_codes =
          unpacked_row_codes(current.data(), main_dim, degree, upd.code_off_bytes());
      const auto rebuilt_codes =
          unpacked_row_codes(rebuilt.data(), main_dim, degree, upd.code_off_bytes());
      const size_t words = main_dim / 64;
      for (size_t slot = 0; slot < degree; ++slot) {
        for (size_t word = 0; word < words; ++word) {
          EXPECT_EQ(current_codes[slot * words + word], rebuilt_codes[slot * words + word])
              << "code mismatch id=" << id << " slot=" << slot << " word=" << word
              << " threads=" << thread_count;
        }
      }

      const auto *rebuilt_ids =
          reinterpret_cast<const PID *>(rebuilt.data() + upd.neighbor_off_bytes());
      for (size_t slot = 0; slot < degree; ++slot) {
        EXPECT_EQ(current_ids[slot], rebuilt_ids[slot])
            << "neighbor mismatch id=" << id << " slot=" << slot << " threads=" << thread_count;
      }
      const auto *current_factors =
          reinterpret_cast<const float *>(current.data() + upd.factor_off_bytes());
      const auto *rebuilt_factors =
          reinterpret_cast<const float *>(rebuilt.data() + upd.factor_off_bytes());
      for (size_t channel = 0; channel < 3; ++channel) {
        for (size_t slot = 0; slot < degree; ++slot) {
          const size_t index = channel * degree + slot;
          const float denom = std::max(1.0F,
                                       std::max(std::fabs(current_factors[index]),
                                                std::fabs(rebuilt_factors[index])));
          EXPECT_LE(std::fabs(current_factors[index] - rebuilt_factors[index]) / denom, 1e-5F)
              << "factor mismatch id=" << id << " slot=" << slot << " channel=" << channel
              << " threads=" << thread_count << " patched=" << current_factors[index]
              << " rebuilt=" << rebuilt_factors[index];
        }
      }
    }
    EXPECT_GT(result.patched_rows, 0U) << "threads=" << thread_count;
    EXPECT_GT(result.untouched_slots, 0U) << "threads=" << thread_count;
    result.page_hashes = index_page_hashes(index_path, geometry.page_size);
    return result;
  };

  const AuditResult serial = run(serial_prefix, 1);
  const AuditResult parallel = run(parallel_prefix, 32);
  EXPECT_EQ(serial.stats.free_slot_fills, parallel.stats.free_slot_fills);
  EXPECT_EQ(serial.stats.evictions, parallel.stats.evictions);
  EXPECT_EQ(serial.stats.forced_links, parallel.stats.forced_links);
  EXPECT_EQ(serial.page_hashes.size(), parallel.page_hashes.size());
  const size_t common_pages = std::min(serial.page_hashes.size(), parallel.page_hashes.size());
  for (size_t page = 0; page < common_pages; ++page) {
    EXPECT_EQ(serial.page_hashes[page], parallel.page_hashes[page])
        << "single-thread vs 32-thread terminal page mismatch page=" << page;
  }
  EXPECT_EQ(serial.page_hashes, parallel.page_hashes);
  std::cout << "pd512_patch_audit,serial_fills," << serial.stats.free_slot_fills
            << ",serial_evictions," << serial.stats.evictions << ",serial_patched_rows,"
            << serial.patched_rows << ",parallel_fills," << parallel.stats.free_slot_fills
            << ",parallel_evictions," << parallel.stats.evictions << ",parallel_patched_rows,"
            << parallel.patched_rows << ",untouched_slots," << parallel.untouched_slots
            << ",terminal_pages," << parallel.page_hashes.size() << ",page_hash_vector,"
            << fnv1a(parallel.page_hashes.data(), parallel.page_hashes.size() * sizeof(uint64_t))
            << "\n";
}

TEST_F(QGUpdaterIndexTest, ParallelBatchInsertAndConsolidate) {
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string copy_prefix = (tiny_->dir / "par").string();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(tiny_->prefix + ext,
                               copy_prefix + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }
  const size_t n_insert = 256;
  auto new_data = make_data(n_insert, kDim, 4242);

  {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(copy_prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);
    UpdateParams params;
    params.ef_insert = 64;
    params.backlink_mode = UpdateParams::Backlink::kEvict;
    params.max_points = kN + n_insert;
    params.maintain_indegree = true;
    QGUpdater upd(qg, params);
    upd.init_indegree(8);

    // batch three-phase publish, 8 writers per batch of 64
    for (size_t start = 0; start < n_insert; start += 64) {
#pragma omp parallel for num_threads(8) schedule(dynamic)
      for (int64_t i = static_cast<int64_t>(start); i < static_cast<int64_t>(start + 64); ++i) {
        upd.insert_with_id(new_data.data() + static_cast<size_t>(i) * kDim,
                           static_cast<PID>(kN + static_cast<size_t>(i)));
      }
      upd.flush(8);
      upd.publish(kN + start + 64);
    }
    EXPECT_EQ(upd.num_points(), kN + n_insert);

    // tombstone a slice of original nodes, then consolidate with headroom target
    for (PID id = 100; id < 300; ++id) {
      upd.tombstone(id);
    }
    upd.consolidate(8, /*r_target=*/kDeg - 4);
    GardenParams garden;
    garden.frac = 0.02;
    garden.ef_maintenance = 64;
    garden.pump_budget = 2;
    garden.r_target = kDeg - 4;
    upd.garden(8, garden);
    std::vector<int32_t> incremental(kN + n_insert);
    for (PID id = 0; id < kN + n_insert; ++id) incremental[id] = upd.indegree(id);
    upd.init_indegree(8);
    for (PID id = 0; id < kN + n_insert; ++id) {
      EXPECT_EQ(upd.indegree(id), incremental[id]) << "indegree mismatch at id " << id;
    }
    uint64_t disabled_turnover_sum = 0;
    for (PID id = 0; id < kN + n_insert; ++id) disabled_turnover_sum += upd.turnover(id);
    EXPECT_EQ(disabled_turnover_sum, 0U) << "default-disabled turnover path must stay inert";
    upd.finalize();
    const UpdateStats s = upd.stats();
    EXPECT_EQ(s.inserts, n_insert);
    EXPECT_GT(s.spliced_slots + s.ghosted_slots, 0U);
  }

  // reload: parallel-inserted vectors discoverable; no live row references a
  // tombstoned neighbor with a live payload
  {
    QuantizedGraph qg(kN + n_insert, kDeg, kDim, kDim);
    qg.load_disk_index(copy_prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);
    QGUpdater upd(qg, UpdateParams{});

    std::vector<uint32_t> res(10);
    size_t found = 0;
    for (size_t i = 0; i < n_insert; ++i) {
      qg.search(new_data.data() + i * kDim, 10, res.data());
      if (res[0] == kN + i) {
        ++found;
      }
    }
    EXPECT_GE(found, n_insert * 9 / 10) << "parallel-inserted vectors must be discoverable";

    const std::string index_path = copy_prefix + suffix;
    const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
    const size_t npp = std::max<size_t>(1, 4096 / node_len);
    const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
    size_t dead_refs = 0;
    for (PID u = 0; u < kN + n_insert; ++u) {
      if (u >= 100 && u < 300) {
        continue;  // tombstoned rows themselves may keep stale edges
      }
      auto row = read_node_row(qg, index_path, u, node_len, page_size, npp);
      const auto *ids = reinterpret_cast<const PID *>(row.data() + upd.neighbor_off_bytes());
      for (size_t j = 0; j < kDeg; ++j) {
        if (!upd.is_v1_ghost_slot(row.data(), j) && ids[j] >= 100 && ids[j] < 300) {
          ++dead_refs;
        }
      }
    }
    EXPECT_EQ(dead_refs, 0U) << "consolidate must purge all dead out-edges from live rows";
  }
}

TEST_F(QGUpdaterIndexTest, TurnoverEventsIncrementForBacklinksPurgeAndSplice) {
  const std::string prefix = (tiny_->dir / "turnover_events").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kEvict;
  params.max_points = kN + 1;
  params.maintain_turnover = true;
  QGUpdater upd(qg, params);
  upd.init_turnover();

  const auto sum_all = [&] {
    uint64_t sum = 0;
    for (PID id = 0; id < upd.num_points(); ++id) sum += upd.turnover(id);
    return sum;
  };

  const auto inserted = make_data(1, kDim, 61011);
  upd.insert(inserted.data());
  const UpdateStats after_insert = upd.stats();
  EXPECT_GT(after_insert.free_slot_fills, 0U);
  EXPECT_GT(after_insert.evictions, 0U);
  EXPECT_EQ(sum_all(), after_insert.free_slot_fills + after_insert.evictions);
  EXPECT_EQ(upd.turnover(kN), 0U) << "a newly assembled row starts at zero";

  for (PID id = 100; id < 300; ++id) upd.tombstone(id);
  const uint64_t before_consolidate_sum = sum_all();
  const UpdateStats before_consolidate = upd.stats();
  upd.consolidate(4, kDeg, false);
  const UpdateStats after_consolidate = upd.stats();
  const uint64_t spliced = after_consolidate.spliced_slots - before_consolidate.spliced_slots;
  const uint64_t purged_without_replacement =
      after_consolidate.ghosted_slots - before_consolidate.ghosted_slots;
  EXPECT_GT(spliced, 0U);
  EXPECT_EQ(sum_all(), before_consolidate_sum + 2 * spliced + purged_without_replacement);
}

TEST_F(QGUpdaterIndexTest, GardenRewriteClearsSelectedTurnover) {
  const std::string prefix = (tiny_->dir / "turnover_clear").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kEvict;
  params.max_points = kN + 8;
  params.maintain_indegree = true;
  params.maintain_turnover = true;
  QGUpdater upd(qg, params);
  upd.init_indegree(4);
  upd.init_turnover();
  const auto inserted = make_data(8, kDim, 61012);
  for (size_t i = 0; i < 8; ++i) upd.insert(inserted.data() + i * kDim);

  PID expected = 0;
  for (PID id = 1; id < upd.num_points(); ++id) {
    if (upd.turnover(id) > upd.turnover(expected)) expected = id;
  }
  const uint16_t expected_turnover = upd.turnover(expected);
  ASSERT_GT(expected_turnover, 0U);
  GardenParams garden;
  garden.frac = 0.0001;  // ceil(frac * 2008 live rows) == one row
  garden.ef_maintenance = 64;
  garden.pump_budget = 0;
  garden.r_target = kDeg - 4;
  garden.policy = GardenParams::Policy::kTurnover;
  const UpdateStats before = upd.stats();
  upd.garden(4, garden);
  const UpdateStats after = upd.stats();
  EXPECT_EQ(after.garden_selected_turnover_rows - before.garden_selected_turnover_rows, 1U);
  EXPECT_EQ(after.garden_selected_turnover_sum - before.garden_selected_turnover_sum,
            expected_turnover);
  EXPECT_EQ(upd.turnover(expected), 0U);
}

TEST_F(QGUpdaterIndexTest, TurnoverGardenSelectsExactTopFraction) {
  const std::string prefix = (tiny_->dir / "turnover_top_fraction").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  constexpr size_t kInsert = 32;
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kEvict;
  params.max_points = kN + kInsert;
  params.maintain_indegree = true;
  params.maintain_turnover = true;
  QGUpdater upd(qg, params);
  upd.init_indegree(4);
  upd.init_turnover();
  const auto inserted = make_data(kInsert, kDim, 61013);
  for (size_t i = 0; i < kInsert; ++i) upd.insert(inserted.data() + i * kDim);

  std::vector<std::pair<uint16_t, PID>> ranked;
  uint64_t all_sum = 0;
  for (PID id = 0; id < upd.num_points(); ++id) {
    ranked.emplace_back(upd.turnover(id), id);
    all_sum += upd.turnover(id);
  }
  std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
    return a.first != b.first ? a.first > b.first : a.second < b.second;
  });
  constexpr double kFrac = 0.01;
  const size_t selected =
      static_cast<size_t>(std::ceil(kFrac * static_cast<double>(ranked.size())));
  ASSERT_GT(ranked[selected - 1].first, 0U);
  uint64_t selected_sum = 0;
  std::vector<uint16_t> before(upd.num_points());
  std::vector<bool> expected(upd.num_points());
  for (PID id = 0; id < upd.num_points(); ++id) before[id] = upd.turnover(id);
  for (size_t i = 0; i < selected; ++i) {
    expected[ranked[i].second] = true;
    selected_sum += ranked[i].first;
  }

  GardenParams garden;
  garden.frac = kFrac;
  garden.ef_maintenance = 64;
  garden.pump_budget = 0;
  garden.r_target = kDeg - 4;
  garden.policy = GardenParams::Policy::kTurnover;
  const UpdateStats stats_before = upd.stats();
  upd.garden(4, garden);
  const UpdateStats stats_after = upd.stats();
  for (PID id = 0; id < upd.num_points(); ++id) {
    EXPECT_EQ(upd.turnover(id), expected[id] ? 0 : before[id]) << "row " << id;
  }
  EXPECT_EQ(stats_after.garden_selected_turnover_rows - stats_before.garden_selected_turnover_rows,
            selected);
  EXPECT_EQ(stats_after.garden_selected_turnover_sum - stats_before.garden_selected_turnover_sum,
            selected_sum);
  EXPECT_EQ(stats_after.garden_all_turnover_sum - stats_before.garden_all_turnover_sum, all_sum);
  EXPECT_EQ(upd.turnover_summary().sum, all_sum - selected_sum);
}

TEST_F(QGUpdaterIndexTest, ConsolidateWithoutSpliceLeavesDegreeHeadroom) {
  const std::string prefix = (tiny_->dir / "no_splice").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.max_points = kN + 64;
  params.splice_enabled = false;
  QGUpdater upd(qg, params);

  std::unordered_set<PID> dead;
  for (PID id = 100; id < 200; ++id) {
    upd.tombstone(id);
    dead.insert(id);
  }

  const std::string path = prefix + index_suffix();
  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  const size_t npp = std::max<size_t>(1, 4096 / node_len);
  const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
  std::vector<size_t> before_degree(kN);
  std::vector<size_t> dead_refs(kN);
  size_t total_dead_refs = 0;
  for (PID u = 0; u < kN; ++u) {
    if (dead.count(u) != 0) continue;
    const auto row = read_node_row(qg, path, u, node_len, page_size, npp);
    before_degree[u] = upd.trailer(u).valid_degree;
    const auto *ids = reinterpret_cast<const PID *>(row.data() + upd.neighbor_off_bytes());
    for (size_t j = 0; j < before_degree[u]; ++j) {
      dead_refs[u] += dead.count(ids[j]);
    }
    total_dead_refs += dead_refs[u];
  }
  ASSERT_GT(total_dead_refs, 0U);

  upd.consolidate(4, /*r_target=*/kDeg, /*reclaim_slots=*/false);
  for (PID u = 0; u < kN; ++u) {
    if (dead.count(u) != 0) continue;
    const auto row = read_node_row(qg, path, u, node_len, page_size, npp);
    const size_t after_degree = upd.trailer(u).valid_degree;
    EXPECT_EQ(after_degree, before_degree[u] - dead_refs[u]) << "row " << u;
    const auto *ids = reinterpret_cast<const PID *>(row.data() + upd.neighbor_off_bytes());
    for (size_t j = 0; j < after_degree; ++j) {
      EXPECT_EQ(dead.count(ids[j]), 0U) << "row " << u << " slot " << j;
    }
  }
  EXPECT_EQ(upd.stats().spliced_slots, 0U);
  EXPECT_EQ(upd.stats().ghosted_slots, total_dead_refs);
}

TEST_F(QGUpdaterIndexTest, WriteCacheCoalescesAndMatchesImmediateWrites) {
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string cached_prefix = (tiny_->dir / "cached").string();
  const std::string immediate_prefix = (tiny_->dir / "immediate").string();
  for (const auto &prefix : {cached_prefix, immediate_prefix}) {
    for (const std::string &ext :
         {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
      std::filesystem::copy_file(tiny_->prefix + ext,
                                 prefix + ext,
                                 std::filesystem::copy_options::overwrite_existing);
    }
  }

  constexpr size_t n_insert = 48;
  auto new_data = make_data(n_insert, kDim, 9917);
  auto run = [&](const std::string &prefix, bool write_cache) {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    qg.set_params(64, 1, 4);
    UpdateParams params;
    params.ef_insert = 64;
    params.backlink_mode = UpdateParams::Backlink::kEvict;
    params.max_points = kN + n_insert;
    params.write_cache = write_cache;
    QGUpdater upd(qg, params);
    for (size_t i = 0; i < n_insert; ++i) {
      upd.insert_with_id(new_data.data() + i * kDim, static_cast<PID>(kN + i));
    }
    upd.flush(1);
    upd.publish(kN + n_insert);
    upd.finalize();
    return upd.stats();
  };

  const UpdateStats cached = run(cached_prefix, true);
  const UpdateStats immediate = run(immediate_prefix, false);
  EXPECT_GT(cached.logical_row_writes, cached.flush_unique_pages);
  EXPECT_LT(cached.physical_writes, cached.logical_row_writes);
  EXPECT_GT(immediate.physical_writes, cached.physical_writes);

  std::ifstream a(cached_prefix + suffix, std::ios::binary);
  std::ifstream b(immediate_prefix + suffix, std::ios::binary);
  ASSERT_TRUE(a.is_open());
  ASSERT_TRUE(b.is_open());
  const std::vector<char> bytes_a((std::istreambuf_iterator<char>(a)), {});
  const std::vector<char> bytes_b((std::istreambuf_iterator<char>(b)), {});
  EXPECT_EQ(bytes_a, bytes_b)
      << "single-thread staged drain order is deterministic across cache modes";
}

TEST_F(QGUpdaterIndexTest, ExactEvictRemovesExactFarthestNeighbor) {
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string prefix = (tiny_->dir / "exact_evict").string();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(tiny_->prefix + ext,
                               prefix + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kExactEvict;
  params.max_points = kN + 1;
  params.write_cache = false;
  QGUpdater upd(qg, params);
  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  const size_t npp = std::max<size_t>(1, 4096 / node_len);
  const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
  std::vector<std::vector<char>> before(kN);
  std::vector<uint16_t> before_degree(kN);
  for (PID id = 0; id < kN; ++id) {
    before[id] = read_node_row(qg, prefix + suffix, id, node_len, page_size, npp);
    before_degree[id] = upd.trailer(id).valid_degree;
  }
  const auto inserted = make_data(1, kDim, 7781);
  upd.insert(inserted.data());

  size_t checked = 0;
  for (PID u = 0; u < kN; ++u) {
    if (before_degree[u] != kDeg) continue;  // headroom fill is not an eviction
    const auto after = read_node_row(qg, prefix + suffix, u, node_len, page_size, npp);
    const auto *old_ids =
        reinterpret_cast<const PID *>(before[u].data() + upd.neighbor_off_bytes());
    const auto *new_ids = reinterpret_cast<const PID *>(after.data() + upd.neighbor_off_bytes());
    std::unordered_set<PID> new_set(new_ids, new_ids + kDeg);
    std::vector<PID> removed;
    for (size_t j = 0; j < kDeg; ++j) {
      if (new_set.count(old_ids[j]) == 0) removed.push_back(old_ids[j]);
    }
    if (removed.size() != 1 || new_set.count(static_cast<PID>(kN)) == 0) continue;
    float worst = -1;
    PID farthest = 0;
    for (size_t j = 0; j < kDeg; ++j) {
      const float d = space::l2_sqr(tiny_->data.data() + static_cast<size_t>(u) * kDim,
                                    tiny_->data.data() + static_cast<size_t>(old_ids[j]) * kDim,
                                    kDim);
      if (d > worst) {
        worst = d;
        farthest = old_ids[j];
      }
    }
    EXPECT_EQ(removed[0], farthest) << "target row " << u;
    ++checked;
  }
  EXPECT_EQ(checked, upd.stats().evictions);
  EXPECT_GT(checked, 0U);
}

TEST_F(QGUpdaterIndexTest, HugeEvictMarginPreventsAllEvictions) {
  const std::string prefix = (tiny_->dir / "evict_margin").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kEvict;
  params.evict_margin = 1e30;
  params.max_points = kN + 8;
  QGUpdater upd(qg, params);
  const auto inserted = make_data(8, kDim, 89231);
  for (size_t i = 0; i < 8; ++i) upd.insert(inserted.data() + i * kDim);
  EXPECT_EQ(upd.stats().evictions, 0U);
  EXPECT_GT(upd.stats().est_skips, 0U);
}

TEST_F(QGUpdaterIndexTest, EvictTelemetryP1MatchesManualReplay) {
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string prefix = (tiny_->dir / "evict_tel").string();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(tiny_->prefix + ext,
                               prefix + ext,
                               std::filesystem::copy_options::overwrite_existing);
  }
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  UpdateParams params;
  params.ef_insert = 64;
  params.backlink_mode = UpdateParams::Backlink::kEvict;
  params.evict_telemetry = 1.0;
  params.max_points = kN + 4;
  params.write_cache = false;
  QGUpdater upd(qg, params);
  const auto inserted = make_data(4, kDim, 99181);
  for (size_t i = 0; i < 4; ++i) upd.insert(inserted.data() + i * kDim);
  const UpdateStats s = upd.stats();
  EXPECT_EQ(s.evict_tel_samples, s.evictions);
  EXPECT_EQ(std::accumulate(s.evict_tel_regret.begin(), s.evict_tel_regret.end(), uint64_t{0}),
            s.evict_tel_samples);
  EXPECT_EQ(s.evict_tel_agree, s.evict_tel_regret[0]);
  EXPECT_TRUE(std::isfinite(s.evict_tel_relerr_sum));
}

TEST(QGUpdaterUnit, GroupedRecallArithmetic) {
  const std::vector<uint64_t> hits{2, 1, 0, 3};
  const std::vector<uint64_t> totals{2, 2, 4, 4};
  const std::vector<int> groups{0, 1, 2, 1};
  const auto recall = grouped_recall(hits, totals, groups);
  EXPECT_DOUBLE_EQ(recall[0], 1.0);
  EXPECT_DOUBLE_EQ(recall[1], 4.0 / 6.0);
  EXPECT_DOUBLE_EQ(recall[2], 0.0);
  EXPECT_DOUBLE_EQ(grouped_recall({0}, {0}, {0})[1], -1.0);
}

TEST_F(QGUpdaterIndexTest, SuperblockSelectsGoodCopyAndAlternatesGeneration) {
  const std::string prefix = (tiny_->dir / "superblock").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  const std::string path = prefix + index_suffix();

  {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    QGUpdater upd(qg, UpdateParams{});  // migration writes A generation 1
    upd.checkpoint();                   // B generation 2
    EXPECT_EQ(upd.active_superblock_slot(), 1);
    EXPECT_EQ(upd.generation(), 2U);
  }

  // Deliberately destroy A while leaving the newer B copy intact.
  const int fd = ::open(path.c_str(), O_WRONLY);
  ASSERT_GE(fd, 0);
  std::array<char, kQGSuperblockSize> garbage{};
  std::fill(garbage.begin(), garbage.end(), static_cast<char>(0xa5));
  ASSERT_EQ(::pwrite(fd, garbage.data(), garbage.size(), 0), static_cast<ssize_t>(garbage.size()));
  ASSERT_EQ(::fsync(fd), 0);
  ::close(fd);

  {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    EXPECT_NO_THROW(qg.load_disk_index(prefix.c_str(), 0.0F));
    QGUpdater upd(qg, UpdateParams{});
    EXPECT_EQ(upd.active_superblock_slot(), 1);
    EXPECT_EQ(upd.generation(), 2U);
    upd.checkpoint();  // repairs A as generation 3
    EXPECT_EQ(upd.active_superblock_slot(), 0);
    EXPECT_EQ(upd.generation(), 3U);
  }
  {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    QGUpdater upd(qg, UpdateParams{});
    EXPECT_EQ(upd.active_superblock_slot(), 0);
    EXPECT_EQ(upd.generation(), 3U);
  }
}

TEST_F(QGUpdaterIndexTest, ReusedPidStaysDarkUntilPublishAndRowIsFullyRewritten) {
  const std::string prefix = (tiny_->dir / "reuse_dark").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  const std::string path = prefix + index_suffix();
  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  const size_t npp = std::max<size_t>(1, kSectorLen / node_len);
  const size_t page_size = (npp * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;

  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  const PID victim = qg.entry_point();
  qg.set_params(256, 1, 16);
  UpdateParams params;
  params.ef_insert = 256;
  params.prune_pool_cap = 512;
  params.write_cache = false;
  params.max_points = kN + 16;
  QGUpdater upd(qg, params);
  const auto old_row = read_node_row(qg, path, victim, node_len, page_size, npp);

  upd.tombstone(victim);
  EXPECT_NE(qg.entry_point(), victim) << "routing root must move before its PID is reusable";
  upd.consolidate(4);
  ASSERT_EQ(upd.free_list_head(), victim);
  ASSERT_EQ(upd.free_count(), 1U);
  EXPECT_EQ(upd.trailer(victim).flags & (kQGRowTombstone | kQGRowFree),
            kQGRowTombstone | kQGRowFree);

  auto y = make_data(1, kDim, 0x5eed);
  const PID reused = upd.allocate_and_insert(y.data());
  ASSERT_EQ(reused, victim);
  ASSERT_EQ(upd.free_count(), 0U);
  const QGRowTrailer dark_trailer = upd.trailer(reused);
  EXPECT_EQ(dark_trailer.flags & (kQGRowTombstone | kQGRowFree), kQGRowTombstone | kQGRowFree);

  const auto rewritten = read_node_row(qg, path, reused, node_len, page_size, npp);
  const auto *ids = reinterpret_cast<const PID *>(rewritten.data() + upd.neighbor_off_bytes());
  std::vector<PID> nb_ids(ids, ids + dark_trailer.valid_degree);
  std::vector<const float *> nb_vecs;
  for (PID id : nb_ids) {
    ASSERT_LT(id, kN);
    nb_vecs.push_back(tiny_->data.data() + static_cast<size_t>(id) * kDim);
  }
  std::vector<char> expected(node_len);
  upd.assemble_row(expected.data(), y.data(), nb_vecs, nb_ids);
  EXPECT_EQ(rewritten, expected) << "reused row must contain no free-link/old-row residue";
  EXPECT_NE(rewritten, old_row);

  std::vector<uint32_t> result(10);
  qg.search(y.data(), result.size(), result.data());
  EXPECT_EQ(std::count(result.begin(), result.end(), reused), 0)
      << "a reused id below committed watermark must remain result-filtered";

  upd.publish(upd.num_points());  // watermark is unchanged for a reused PID
  EXPECT_EQ(upd.trailer(reused).flags & (kQGRowTombstone | kQGRowFree), 0);
  qg.search(y.data(), result.size(), result.data());
  EXPECT_EQ(result[0], reused);
  EXPECT_EQ(upd.stats().freed_slots, 1U);
  EXPECT_EQ(upd.stats().reused_slots, 1U);
  upd.checkpoint();
  std::array<char, kSectorLen> header{};
  {
    std::ifstream in(path, std::ios::binary);
    in.read(header.data(), header.size());
  }
  QGSuperblockV2 persisted;
  ASSERT_GE(select_qg_superblock(header.data(), persisted), 0);
  EXPECT_EQ(persisted.entry_point, qg.entry_point());
  EXPECT_NE(persisted.entry_point, victim);
}

TEST_F(QGUpdaterIndexTest, ReusePlatformsFileWhileAppendControlGrows) {
  const std::string reuse_prefix = (tiny_->dir / "platform_reuse").string();
  const std::string append_prefix = (tiny_->dir / "platform_append").string();
  copy_index_artifact(tiny_->v1_prefix, reuse_prefix);
  copy_index_artifact(tiny_->v1_prefix, append_prefix);
  const auto replacement = make_data(5 * kN, kDim, 0x123456);

  auto run = [&](const std::string &prefix, bool reuse) {
    QuantizedGraph qg(kN, kDeg, kDim, kDim);
    qg.load_disk_index(prefix.c_str(), 0.0F);
    UpdateParams params;
    params.backlink_mode = UpdateParams::Backlink::kNone;
    params.ef_insert = 8;
    params.prune_pool_cap = 8;
    params.max_points = 6 * kN + 1024;
    QGUpdater upd(qg, params);
    std::vector<PID> live(kN);
    std::iota(live.begin(), live.end(), PID{0});
    std::vector<size_t> pages;
    for (size_t round = 0; round < 5; ++round) {
      for (PID id : live) upd.tombstone(id);
      upd.consolidate(4, 0, reuse);
      live.clear();
      live.reserve(kN);
      for (size_t i = 0; i < kN; ++i) {
        const float *vec = replacement.data() + (round * kN + i) * kDim;
        PID id;
        if (reuse) {
          id = upd.allocate_and_insert(vec);
        } else {
          id = static_cast<PID>(kN + round * kN + i);
          upd.insert_with_id(vec, id);
        }
        live.push_back(id);
      }
      upd.flush(1);
      upd.publish(upd.allocated_points());
      upd.checkpoint();
      pages.push_back(upd.file_pages());
      EXPECT_EQ(upd.live_count(), kN);
      EXPECT_EQ(std::filesystem::file_size(prefix + index_suffix()),
                kSectorLen + upd.file_pages() * 4096);
    }
    return std::make_pair(pages, upd.stats());
  };

  const auto [reuse_pages, reuse_stats] = run(reuse_prefix, true);
  const auto [append_pages, append_stats] = run(append_prefix, false);
  ASSERT_EQ(reuse_pages.size(), 5U);
  ASSERT_EQ(append_pages.size(), 5U);
  for (size_t pages : reuse_pages) EXPECT_EQ(pages, reuse_pages.front());
  for (size_t i = 1; i < append_pages.size(); ++i) {
    EXPECT_GT(append_pages[i], append_pages[i - 1]);
  }
  EXPECT_EQ(reuse_stats.reused_slots, 5 * kN);
  EXPECT_EQ(append_stats.reused_slots, 0U);
}

TEST_F(QGUpdaterIndexTest, ConcurrentSearchNeverReturnsUnpublishedTombstoneOrDarkPid) {
  const std::string prefix = (tiny_->dir / "concurrent_publish").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  constexpr size_t kRecycle = 8;
  const size_t batch_size = kRunningTsan ? 2 : 8;
  const size_t num_batches = kRunningTsan ? 3 : 8;
  const auto replacement = make_data(batch_size * num_batches, kDim, 0x713a);

  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  UpdateParams params;
  params.ef_insert = 64;
  params.prune_pool_cap = 128;
  params.max_points = kN + replacement.size() / kDim + 64;
  QGUpdater upd(qg, params);

  // Build reusable slots first, then add permanent tombstones that will not
  // be reclaimed by this test. Reused slots remain hidden through row rewrite
  // and only become visible at publish().
  for (PID id = 100; id < 100 + kRecycle; ++id) upd.tombstone(id);
  upd.consolidate(kRunningTsan ? 1 : 4, 0, true);
  std::unordered_set<PID> permanent_dead;
  for (PID id = 20; id < 28; ++id) {
    upd.tombstone(id);
    permanent_dead.insert(id);
  }

  std::atomic<bool> stop{false};
  std::atomic<bool> bad_result{false};
  std::atomic<uint64_t> query_count{0};
  const size_t query_threads = kRunningTsan ? 2 : 4;
  std::vector<std::thread> readers;
  for (size_t tid = 0; tid < query_threads; ++tid) {
    readers.emplace_back([&, tid] {
      try {
        size_t iteration = 0;
        while (!stop.load(std::memory_order_acquire)) {
          const size_t qi = (tid + iteration++ * query_threads) % 64;
          const auto result = upd.search(tiny_->data.data() + qi * kDim, 10, 64);
          const size_t published_after = upd.num_points();
          for (PID id : result) {
            if (id >= published_after || permanent_dead.count(id) != 0 ||
                (upd.trailer(id).flags & (kQGRowTombstone | kQGRowFree)) != 0) {
              bad_result.store(true, std::memory_order_release);
            }
          }
          query_count.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (...) {
        bad_result.store(true, std::memory_order_release);
      }
    });
  }

  for (size_t batch = 0; batch < num_batches; ++batch) {
    std::vector<PID> allocated(batch_size);
    std::atomic<size_t> next{0};
    std::vector<std::thread> writers;
    const size_t writer_count = kRunningTsan ? 2 : 4;
    for (size_t writer = 0; writer < writer_count; ++writer) {
      writers.emplace_back([&] {
        for (;;) {
          const size_t i = next.fetch_add(1, std::memory_order_relaxed);
          if (i >= batch_size) return;
          allocated[i] =
              upd.allocate_and_insert(replacement.data() + (batch * batch_size + i) * kDim);
        }
      });
    }
    for (auto &writer : writers) writer.join();
    upd.flush(kRunningTsan ? 2 : 4);

    // Deterministic visibility check while the batch is deliberately held
    // before publish: reused PIDs are dark and appended PIDs are above the
    // committed watermark, even though every row/backlink is already present.
    for (size_t i = 0; i < batch_size; ++i) {
      const auto result = upd.search(replacement.data() + (batch * batch_size + i) * kDim, 10, 128);
      for (PID id : allocated) EXPECT_EQ(std::count(result.begin(), result.end(), id), 0);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kRunningTsan ? 1 : 3));
    upd.publish(upd.allocated_points());
  }
  stop.store(true, std::memory_order_release);
  for (auto &reader : readers) reader.join();

  EXPECT_FALSE(bad_result.load());
  EXPECT_GT(query_count.load(), 0U);
  const UpdateStats stats = upd.stats();
  EXPECT_GT(stats.query_seqlock_read_calls, 0U);
  EXPECT_LE(stats.query_seqlock_read_retries, stats.seqlock_read_retries);
}

TEST_F(QGUpdaterIndexTest, SearchContinuesDuringConsolidate) {
  const std::string prefix = (tiny_->dir / "concurrent_consolidate").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  UpdateParams params;
  params.max_points = kN + 64;
  QGUpdater upd(qg, params);
  const size_t tombstones = kRunningTsan ? 16 : 96;
  std::unordered_set<PID> dead;
  for (size_t i = 0; i < tombstones; ++i) {
    const PID id = static_cast<PID>(200 + i);
    upd.tombstone(id);
    dead.insert(id);
  }

  std::atomic<bool> stop{false};
  std::atomic<bool> bad{false};
  std::atomic<uint64_t> queries{0};
  std::thread reader([&] {
    try {
      size_t i = 0;
      while (!stop.load(std::memory_order_acquire)) {
        const auto result = upd.search(tiny_->data.data() + (i++ % 64) * kDim, 10, 64);
        for (PID id : result) {
          if (id >= upd.num_points() || dead.count(id) != 0) bad.store(true);
        }
        ++queries;
      }
    } catch (...) {
      bad.store(true);
    }
  });
  while (queries.load(std::memory_order_acquire) == 0) std::this_thread::yield();
  upd.consolidate(kRunningTsan ? 1 : 8, kDeg - 4, false);
  stop.store(true, std::memory_order_release);
  reader.join();
  EXPECT_FALSE(bad.load());
  EXPECT_GT(queries.load(), 0U);
  EXPECT_GT(upd.stats().consolidated_rows, 0U);
}

TEST_F(QGUpdaterIndexTest, SearchContinuesDuringGarden) {
  const std::string prefix = (tiny_->dir / "concurrent_garden").string();
  copy_index_artifact(tiny_->v1_prefix, prefix);
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(prefix.c_str(), 0.0F);
  UpdateParams params;
  params.max_points = kN + 64;
  params.maintain_indegree = true;
  QGUpdater upd(qg, params);
  upd.init_indegree(kRunningTsan ? 1 : 8);

  std::atomic<bool> stop{false};
  std::atomic<bool> bad{false};
  std::atomic<uint64_t> queries{0};
  std::thread reader([&] {
    try {
      size_t i = 0;
      while (!stop.load(std::memory_order_acquire)) {
        const auto result = upd.search(tiny_->data.data() + (i++ % 64) * kDim, 10, 64);
        for (PID id : result) {
          if (id >= upd.num_points() ||
              (upd.trailer(id).flags & (kQGRowTombstone | kQGRowFree)) != 0) {
            bad.store(true);
          }
        }
        ++queries;
      }
    } catch (...) {
      bad.store(true);
    }
  });
  while (queries.load(std::memory_order_acquire) == 0) std::this_thread::yield();
  GardenParams garden;
  garden.frac = kRunningTsan ? 0.01 : 0.10;
  garden.ef_maintenance = kRunningTsan ? 32 : 64;
  garden.pump_budget = 2;
  garden.r_target = kDeg - 4;
  upd.garden(kRunningTsan ? 1 : 8, garden);
  stop.store(true, std::memory_order_release);
  reader.join();
  EXPECT_FALSE(bad.load());
  EXPECT_GT(queries.load(), 0U);
  EXPECT_GT(upd.stats().gardened_rows, 0U);
}

TEST_F(QGUpdaterIndexTest, GhostSlotDetection) {
  QuantizedGraph qg(kN, kDeg, kDim, kDim);
  qg.load_disk_index(tiny_->prefix.c_str(), 0.0F);
  qg.set_params(64, 1, 4);
  QGUpdater upd(qg, UpdateParams{});

  const size_t node_len = (32 * kDim + 128 * kDeg + kDeg * kDim) / 8;
  std::vector<char> row(node_len);
  const size_t d_use = 20;  // leave 12 ghost tail slots
  std::vector<PID> ids;
  std::vector<const float *> vecs;
  for (size_t j = 0; j < d_use; ++j) {
    ids.push_back(static_cast<PID>(j + 1));
    vecs.push_back(tiny_->data.data() + (j + 1) * kDim);
  }
  upd.assemble_row(row.data(), tiny_->data.data(), vecs, ids);
  for (size_t j = 0; j < d_use; ++j) {
    EXPECT_FALSE(upd.is_v1_ghost_slot(row.data(), j)) << "slot " << j;
  }
  for (size_t j = d_use; j < kDeg; ++j) {
    EXPECT_TRUE(upd.is_v1_ghost_slot(row.data(), j)) << "slot " << j;
  }
}

}  // namespace
}  // namespace alaya::laser
