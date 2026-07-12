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

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_builder.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/vamana/vamana_builder.hpp"
#include "index/graph/vamana/vamana_writer.hpp"

namespace alaya::laser {
namespace {

constexpr size_t kDim = 64;
constexpr size_t kDeg = 32;
constexpr size_t kN = 2000;

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

struct TinyIndex {
  std::filesystem::path dir;
  std::string prefix;
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
    vp.num_threads = 4;
    alaya::vamana::VamanaBuilder vb(t.data.data(), kN, kDim, vp);
    vb.build();
    const std::string vamana_path = t.prefix + "_vamana.index";
    alaya::vamana::save_graph(vb.graph(), vamana_path, kDeg, vb.medoid());

    write_fbin(t.prefix + "_pca_base.fbin", t.data.data(), kN, kDim);

    QuantizedGraph qg(kN, kDeg, kDim, kDim, /*rotator_seed=*/7);
    QGBuilder builder(qg, /*ef_build=*/64, /*num_threads=*/4);
    builder.build(vamana_path.c_str(), t.prefix.c_str());
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

TEST(QGUpdaterUnit, UnpackRoundTrip) {
  std::mt19937_64 gen(42);
  for (size_t pd : {64UL, 128UL, 256UL}) {
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
      if (upd.is_ghost_slot(row.data(), j)) {
        for (size_t k = j; k < kDeg; ++k) {
          if (!upd.is_ghost_slot(row.data(), k)) {
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
        if (!upd.is_ghost_slot(row.data(), j) && ids[j] >= kN) {
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
        if (upd.is_ghost_slot(row.data(), j)) {
          for (size_t k = j; k < kDeg; ++k) {
            usable = usable && upd.is_ghost_slot(row.data(), k);
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
        if (!upd.is_ghost_slot(row.data(), j) && ids[j] >= 100 && ids[j] < 300) {
          ++dead_refs;
        }
      }
    }
    EXPECT_EQ(dead_refs, 0U) << "consolidate must purge all dead out-edges from live rows";
  }
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
    std::filesystem::copy_file(tiny_->prefix + ext, prefix + ext,
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
  for (PID id = 0; id < kN; ++id) {
    before[id] = read_node_row(qg, prefix + suffix, id, node_len, page_size, npp);
  }
  const auto inserted = make_data(1, kDim, 7781);
  upd.insert(inserted.data());

  size_t checked = 0;
  for (PID u = 0; u < kN; ++u) {
    const auto after = read_node_row(qg, prefix + suffix, u, node_len, page_size, npp);
    const auto *old_ids = reinterpret_cast<const PID *>(before[u].data() + upd.neighbor_off_bytes());
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
      if (d > worst) { worst = d; farthest = old_ids[j]; }
    }
    EXPECT_EQ(removed[0], farthest) << "target row " << u;
    ++checked;
  }
  EXPECT_EQ(checked, upd.stats().evictions);
  EXPECT_GT(checked, 0U);
}

TEST_F(QGUpdaterIndexTest, EvictTelemetryP1MatchesManualReplay) {
  const std::string suffix = "_R" + std::to_string(kDeg) + "_MD" + std::to_string(kDim) + ".index";
  const std::string prefix = (tiny_->dir / "evict_tel").string();
  for (const std::string &ext :
       {suffix, suffix + "_rotator", suffix + "_cache_ids", suffix + "_cache_nodes"}) {
    std::filesystem::copy_file(tiny_->prefix + ext, prefix + ext,
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
    EXPECT_FALSE(upd.is_ghost_slot(row.data(), j)) << "slot " << j;
  }
  for (size_t j = d_use; j < kDeg; ++j) {
    EXPECT_TRUE(upd.is_ghost_slot(row.data(), j)) << "slot " << j;
  }
}

}  // namespace
}  // namespace alaya::laser
