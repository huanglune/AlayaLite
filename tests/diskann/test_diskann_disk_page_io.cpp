// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/graph/diskann/disk_page_io.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "coro/sync_wait.hpp"
#include "coro/task.hpp"
#include "coro/thread_pool.hpp"
#include "coro/when_all.hpp"
#include "index/graph/diskann/beam_search.hpp"
#include "index/graph/diskann/beam_search_async.hpp"
#include "index/graph/diskann/disk_layout.hpp"
#include "index/graph/diskann/diskann_index.hpp"
#include "index/graph/diskann/search_scratch.hpp"
#include "storage/io/page_reader_factory.hpp"
#include "storage/io/uring_reactor.hpp"

namespace {

#if defined(__linux__)

using alaya::diskann::DiskANNBuildParams;
using alaya::diskann::DiskANNIndex;
using alaya::diskann::DiskLayoutGeometry;
using alaya::diskann::DiskPageIO;
using alaya::diskann::NodeRecordView;

std::vector<float> make_vectors(uint64_t n, uint64_t dim, uint32_t seed = 123) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(n * dim);
  for (auto &x : v) {
    x = dist(rng);
  }
  return v;
}

std::vector<uint64_t> make_labels(uint64_t n) {
  std::vector<uint64_t> labels(n);
  for (uint64_t i = 0; i < n; ++i) {
    labels[i] = 1000 + i;
  }
  return labels;
}

class DiskPageIOTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("diskann_pageio_" + std::to_string(counter.fetch_add(1)));
  }
  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  void build(uint64_t n, uint64_t dim, uint32_t r, uint32_t pq_n_chunks = 0) {
    n_ = n;
    dim_ = dim;
    r_ = r;
    v_ = make_vectors(n, dim);
    labels_ = make_labels(n);
    DiskANNBuildParams bp;
    bp.R = r;
    bp.pq_n_chunks = pq_n_chunks;
    DiskANNIndex::build(dir_.string(), v_.data(), labels_.data(), n, dim, bp);
    geom_ = DiskLayoutGeometry::compute(dim, r);
  }

  std::string index_path() const { return (dir_ / "diskann.index").string(); }

  std::filesystem::path dir_;
  std::vector<float> v_;
  std::vector<uint64_t> labels_;
  uint64_t n_ = 0;
  uint64_t dim_ = 0;
  uint32_t r_ = 0;
  DiskLayoutGeometry geom_;
};

TEST_F(DiskPageIOTest, ReadNodeMatchesBuild) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  for (uint32_t id : {0u, 1u, 50u, 199u}) {
    const auto nd = io.read_node(id);
    ASSERT_EQ(nd.coords.size(), dim_);
    for (uint64_t d = 0; d < dim_; ++d) {
      EXPECT_FLOAT_EQ(nd.coords[d], v_[id * dim_ + d]) << "id=" << id << " d=" << d;
    }
    EXPECT_LE(nd.nbrs.size(), r_);
    for (const auto nb : nd.nbrs) {
      EXPECT_LT(nb, n_);  // build neighbors are in range
    }
  }
}

TEST_F(DiskPageIOTest, ReadNodesAsyncMatchesReadNode) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_);
  const std::vector<uint32_t> ids = {0, 1, 17, 63, 127, 191, 255};

  const std::vector<DiskPageIO::NodeData> batch = io.read_nodes_async(ids, /*threads=*/4);

  ASSERT_EQ(batch.size(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const DiskPageIO::NodeData single = io.read_node(ids[i]);
    EXPECT_EQ(batch[i].coords, single.coords) << "id=" << ids[i];
    EXPECT_EQ(batch[i].nbrs, single.nbrs) << "id=" << ids[i];
  }
}

TEST_F(DiskPageIOTest, ReadNodeAsyncTaskMatchesReadNode) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_);
  coro::thread_pool pool{{.thread_count = 4,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};

  const DiskPageIO::NodeData async_node = coro::sync_wait(io.read_node_async(127, pool));

  pool.shutdown();
  const DiskPageIO::NodeData single = io.read_node(127);
  EXPECT_EQ(async_node.coords, single.coords);
  EXPECT_EQ(async_node.nbrs, single.nbrs);
}

TEST_F(DiskPageIOTest, ReadNodesAsyncCanUseCallerThreadPool) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_);
  coro::thread_pool pool{{.thread_count = 4,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  const std::vector<uint32_t> ids = {2, 18, 64, 126, 190, 254};

  const std::vector<DiskPageIO::NodeData> batch = io.read_nodes_async(ids, pool);

  pool.shutdown();
  ASSERT_EQ(batch.size(), ids.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    const DiskPageIO::NodeData single = io.read_node(ids[i]);
    EXPECT_EQ(batch[i].coords, single.coords) << "id=" << ids[i];
    EXPECT_EQ(batch[i].nbrs, single.nbrs) << "id=" << ids[i];
  }
}

TEST_F(DiskPageIOTest, WriteNodeRoundTrip) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  std::vector<float> coords(dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    coords[d] = 0.25f + 0.01f * static_cast<float>(d);
  }
  const std::vector<uint32_t> nbrs = {3, 7, 11, 42, 100};
  io.write_node(5, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  const auto nd = io.read_node(5);
  ASSERT_EQ(nd.coords.size(), dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(nd.coords[d], coords[d]) << "d=" << d;
  }
  ASSERT_EQ(nd.nbrs.size(), nbrs.size());
  for (size_t i = 0; i < nbrs.size(); ++i) {
    EXPECT_EQ(nd.nbrs[i], nbrs[i]) << "i=" << i;
  }
}

TEST_F(DiskPageIOTest, WriteNodeReuseDoesNotDisturbCoResidentNodes) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  // Pick a node id that shares its page with neighbors; snapshot a co-resident.
  const uint32_t target = 16;
  const uint32_t sibling = target + 1;  // same sector page for nps>=2
  ASSERT_EQ(geom_.get_page_offset(target), geom_.get_page_offset(sibling));
  const auto sibling_before = io.read_node(sibling);

  std::vector<float> coords(dim_, 0.5f);
  const std::vector<uint32_t> nbrs = {1, 2};
  io.write_node(target, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  const auto sibling_after = io.read_node(sibling);
  EXPECT_EQ(sibling_before.coords, sibling_after.coords);
  EXPECT_EQ(sibling_before.nbrs, sibling_after.nbrs);
}

TEST_F(DiskPageIOTest, WriteNodeNeighborsPreservesCoords) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  const auto before = io.read_node(10);
  const std::vector<uint32_t> new_nbrs = {1, 2, 3};
  io.write_node_neighbors(10, static_cast<uint32_t>(new_nbrs.size()), new_nbrs.data());

  const auto after = io.read_node(10);
  ASSERT_EQ(after.coords.size(), before.coords.size());
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(after.coords[d], before.coords[d]) << "d=" << d;
  }
  ASSERT_EQ(after.nbrs.size(), new_nbrs.size());
  for (size_t i = 0; i < new_nbrs.size(); ++i) {
    EXPECT_EQ(after.nbrs[i], new_nbrs[i]) << "i=" << i;
  }
}

TEST_F(DiskPageIOTest, WriteNodeExtendsFileOnAppend) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_);
  const uint64_t before_size = io.file_size();
  // First id whose page does not yet exist (forces ftruncate extension).
  const uint32_t append_id = static_cast<uint32_t>(geom_.nodes_per_sector * geom_.num_pages(n_));
  ASSERT_GE(geom_.get_page_offset(append_id) + geom_.page_size, before_size);

  std::vector<float> coords(dim_, 1.0f);
  const std::vector<uint32_t> nbrs = {0, 1};
  io.write_node(append_id, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());
  EXPECT_GT(io.file_size(), before_size);  // file was extended

  const auto nd = io.read_node(append_id);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(nd.coords[d], coords[d]) << "d=" << d;
  }
  ASSERT_EQ(nd.nbrs.size(), nbrs.size());
}

TEST_F(DiskPageIOTest, CoordsCacheReturnsConsistentVectors) {
  build(100, 16, 16);
  DiskPageIO io(index_path(), geom_);
  const std::vector<float> first = io.read_coords_cached(7);
  const std::vector<float> cached = io.read_coords_cached(7);  // cache hit
  EXPECT_EQ(first, cached);
  ASSERT_EQ(cached.size(), dim_);
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(cached[d], v_[7 * dim_ + d]) << "d=" << d;
  }
}

TEST_F(DiskPageIOTest, DirtyPageIsWrittenOnlyAfterFlush) {
  build(200, 32, 32);
  DiskPageIO writer(index_path(), geom_, /*page_cache_capacity=*/2);
  DiskPageIO reader(index_path(), geom_, /*page_cache_capacity=*/0);

  const uint32_t id = 10;
  const auto before = reader.read_node(id);
  const std::vector<uint32_t> new_nbrs = {4, 5, 6, 7};

  writer.write_node_neighbors(id, static_cast<uint32_t>(new_nbrs.size()), new_nbrs.data());

  const auto visible_to_writer = writer.read_node(id);
  EXPECT_EQ(visible_to_writer.nbrs, new_nbrs);

  const auto still_on_disk = reader.read_node(id);
  EXPECT_EQ(still_on_disk.nbrs, before.nbrs);

  writer.flush_dirty_pages();
  const auto after_flush = reader.read_node(id);
  EXPECT_EQ(after_flush.nbrs, new_nbrs);
}

TEST_F(DiskPageIOTest, DirtyPageIsWrittenWhenEvicted) {
  build(200, 32, 32);
  DiskPageIO writer(index_path(), geom_, /*page_cache_capacity=*/1);
  DiskPageIO reader(index_path(), geom_, /*page_cache_capacity=*/0);

  const uint32_t dirty_id = 0;
  const uint32_t other_page_id = static_cast<uint32_t>(geom_.nodes_per_sector);
  ASSERT_NE(geom_.get_page_offset(dirty_id), geom_.get_page_offset(other_page_id));

  const std::vector<uint32_t> new_nbrs = {8, 9, 10};
  writer.write_node_neighbors(dirty_id, static_cast<uint32_t>(new_nbrs.size()), new_nbrs.data());

  const auto before_eviction = reader.read_node(dirty_id);
  EXPECT_NE(before_eviction.nbrs, new_nbrs);

  (void)writer.read_node(other_page_id);

  const auto after_eviction = reader.read_node(dirty_id);
  EXPECT_EQ(after_eviction.nbrs, new_nbrs);
}

class DiskPageIOReactorTest : public DiskPageIOTest {
 protected:
  void SetUp() override {
    DiskPageIOTest::SetUp();
    if (!alaya::UringReactor::is_available()) {
      GTEST_SKIP() << "io_uring not available on this kernel";
    }
  }
};

TEST_F(DiskPageIOReactorTest, ReadNodeAsyncMatchesBlockingRead) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/16);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  ASSERT_TRUE(io.reactor_enabled());
  coro::thread_pool pool{{.thread_count = 4,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  for (uint32_t id : {0u, 3u, 100u, 255u}) {
    const DiskPageIO::NodeData via_reactor = coro::sync_wait(io.read_node_async(id, pool));
    const DiskPageIO::NodeData blocking = io.read_node(id);
    EXPECT_EQ(via_reactor.coords, blocking.coords) << "id=" << id;
    EXPECT_EQ(via_reactor.nbrs, blocking.nbrs) << "id=" << id;
  }
}

TEST_F(DiskPageIOReactorTest, PrefetchCoordsInstallsExactCoords) {
  build(300, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/8);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  coro::thread_pool pool{{.thread_count = 4,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  const std::vector<uint32_t> ids = {0, 5, 5, 42, 128, 129, 130, 131, 299};  // dup on purpose
  auto task = [&]() -> coro::task<> {
    co_await pool.schedule();
    co_await io.prefetch_coords(ids.data(), ids.size(), pool);
  };
  coro::sync_wait(task());
  for (const uint32_t id : ids) {
    const std::vector<float> coords = io.read_coords_cached(id);
    ASSERT_EQ(coords.size(), dim_);
    for (uint64_t d = 0; d < dim_; ++d) {
      EXPECT_FLOAT_EQ(coords[d], v_[id * dim_ + d]) << "id=" << id << " d=" << d;
    }
  }
}

TEST_F(DiskPageIOReactorTest, PrefetchSeesDirtyCachedPage) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/64);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  coro::thread_pool pool{{.thread_count = 2,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  // Dirty a page in the cache (not flushed), then wave-read through the reactor:
  // phase 1 must serve the dirty cached copy, not the stale disk bytes.
  const uint32_t dirty_id = 17;
  const std::vector<uint32_t> new_nbrs = {1, 2, 3};
  io.write_node_neighbors(dirty_id, static_cast<uint32_t>(new_nbrs.size()), new_nbrs.data());
  const std::vector<uint32_t> ids = {dirty_id};
  const auto out =
      coro::sync_wait(io.read_neighbors_batch_async(ids.data(), 1, pool));  // count==1 fallback
  const DiskPageIO::NodeData via_reactor = coro::sync_wait(io.read_node_async(dirty_id, pool));
  EXPECT_EQ(via_reactor.nbrs, new_nbrs);
  ASSERT_EQ(out.size(), 1U);
  EXPECT_EQ(out[0], new_nbrs);
}

TEST_F(DiskPageIOReactorTest, NeighborsBatchAsyncMatchesSequential) {
  build(512, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/8);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  coro::thread_pool pool{{.thread_count = 4,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < 512; i += 3) {
    ids.push_back(i);
  }
  ids.push_back(4);  // duplicate id and shared pages
  const auto wave = coro::sync_wait(
      io.read_neighbors_batch_async(ids.data(), static_cast<uint32_t>(ids.size()), pool));
  const auto seq = io.read_neighbors_batch(ids.data(), static_cast<uint32_t>(ids.size()));
  ASSERT_EQ(wave.size(), seq.size());
  for (size_t i = 0; i < ids.size(); ++i) {
    EXPECT_EQ(wave[i], seq[i]) << "pos " << i << " id=" << ids[i];
  }
}

TEST_F(DiskPageIOReactorTest, PrefetchPagesThenWriteRoundTrips) {
  build(256, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/32);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  coro::thread_pool pool{{.thread_count = 2,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  const std::vector<uint32_t> targets = {8, 9, 200};
  auto task = [&]() -> coro::task<> {
    co_await pool.schedule();
    co_await io.prefetch_pages(targets.data(), targets.size(), pool);
  };
  coro::sync_wait(task());
  const std::vector<uint32_t> nbrs = {7, 6, 5};
  for (const uint32_t id : targets) {
    io.write_node_neighbors(id, static_cast<uint32_t>(nbrs.size()), nbrs.data());
  }
  for (const uint32_t id : targets) {
    EXPECT_EQ(io.read_node(id).nbrs, nbrs) << "id=" << id;
    // Co-resident node on the same page survives the RMW.
    const uint32_t sibling = (id % 2 == 0) ? id + 1 : id - 1;
    const DiskPageIO::NodeData sib = io.read_node(sibling);
    for (uint64_t d = 0; d < dim_; ++d) {
      ASSERT_FLOAT_EQ(sib.coords[d], v_[sibling * dim_ + d]) << "sibling=" << sibling;
    }
  }
}

TEST_F(DiskPageIOReactorTest, ConcurrentWavesAndWritesStayCoherent) {
  build(600, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/16);
  alaya::UringReactor reactor;
  io.set_reactor(&reactor);
  coro::thread_pool pool{{.thread_count = 8,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  std::atomic<bool> stop{false};
  std::thread writer([&]() {
    const std::vector<uint32_t> nbrs = {11, 22, 33};
    for (uint32_t round = 0; !stop.load(); ++round) {
      io.write_node_neighbors(300 + (round % 100), 3, nbrs.data());
    }
  });
  // Coords never change in this test (only neighbor rewrites), so any torn or
  // stale page surfaces as a mismatch. ASSERT_* can't run inside coroutines
  // (it expands to `return;`), hence the counter.
  std::atomic<uint64_t> mismatches{0};
  auto one_wave = [&](uint32_t seed) -> coro::task<> {
    co_await pool.schedule();
    std::vector<uint32_t> ids;
    for (uint32_t i = 0; i < 64; ++i) {
      ids.push_back((seed * 97 + i * 7) % 600);
    }
    co_await io.prefetch_coords(ids.data(), ids.size(), pool);
    for (const uint32_t id : ids) {
      const std::vector<float> coords = io.read_coords_cached(id);
      if (std::memcmp(coords.data(), &v_[id * dim_], dim_ * sizeof(float)) != 0) {
        mismatches.fetch_add(1);
      }
    }
  };
  auto run = [&]() -> coro::task<> {
    std::vector<coro::task<>> tasks;
    for (uint32_t t = 0; t < 32; ++t) {
      tasks.emplace_back(one_wave(t));
    }
    co_await coro::when_all(std::move(tasks));
  };
  coro::sync_wait(run());
  stop.store(true);
  writer.join();
  EXPECT_EQ(mismatches.load(), 0U);
}

// ---- search peek/fill protocol (the unified-pool view for searches) ----

namespace {
// Raw page bytes straight from the file, bypassing every cache.
std::vector<char> raw_page(const std::string &path, uint64_t off, uint64_t page_size) {
  std::vector<char> bytes(page_size);
  std::ifstream in(path, std::ios::binary);
  in.seekg(static_cast<std::streamoff>(off));
  in.read(bytes.data(), static_cast<std::streamsize>(page_size));
  EXPECT_TRUE(in.good());
  return bytes;
}
}  // namespace

TEST_F(DiskPageIOTest, SearchFillInstallsPageForLaterPeeks) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/64);
  const uint64_t off = geom_.get_page_offset(0);

  std::vector<char> peek_buf(geom_.page_size);
  uint64_t version = 1234;
  ASSERT_FALSE(io.search_peek_page(off, peek_buf.data(), &version));
  EXPECT_EQ(version, 0U);  // never written -> version 0

  auto bytes = raw_page(index_path(), off, geom_.page_size);
  io.search_fill_page(off, bytes.data(), version);

  std::vector<char> again(geom_.page_size, 0);
  uint64_t v2 = 0;
  ASSERT_TRUE(io.search_peek_page(off, again.data(), &v2));
  EXPECT_EQ(0, std::memcmp(again.data(), bytes.data(), geom_.page_size));
}

TEST_F(DiskPageIOTest, SearchFillRefusesStalePageAfterWriteAndEviction) {
  build(200, 32, 32);
  // capacity 1 -> a single one-page shard: the second write below evicts the
  // first page, exposing the version-check branch of the fill protocol.
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/1);
  const uint32_t target = 0;
  const uint64_t off = geom_.get_page_offset(target);

  std::vector<char> stale(geom_.page_size);
  uint64_t version = 0;
  ASSERT_FALSE(io.search_peek_page(off, stale.data(), &version));
  stale = raw_page(index_path(), off, geom_.page_size);  // pre-write content

  // Concurrent-writer stand-in: rewrite the node (bumps the page version),
  // then touch a different page so the written page is evicted (flushed).
  std::vector<float> coords(dim_, 0.75f);
  const std::vector<uint32_t> nbrs = {1, 2, 3};
  io.write_node(target, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());
  uint32_t other = target;
  while (geom_.get_page_offset(other) == off) {
    ++other;
  }
  io.write_node(other, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  // The stale offer must be rejected AND the caller's buffer refreshed to the
  // written content.
  io.search_fill_page(off, stale.data(), version);
  const NodeRecordView view{stale.data() + geom_.offset_to_node(target), dim_};
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(view.coords()[d], coords[d]) << "d=" << d;
  }
}

TEST_F(DiskPageIOTest, SearchFillCacheWinsWhenPagePresent) {
  build(200, 32, 32);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/64);
  const uint32_t target = 0;
  const uint64_t off = geom_.get_page_offset(target);

  std::vector<char> stale(geom_.page_size);
  uint64_t version = 0;
  ASSERT_FALSE(io.search_peek_page(off, stale.data(), &version));
  stale = raw_page(index_path(), off, geom_.page_size);

  std::vector<float> coords(dim_, -0.5f);
  const std::vector<uint32_t> nbrs = {7};
  io.write_node(target, coords.data(), static_cast<uint32_t>(nbrs.size()), nbrs.data());

  io.search_fill_page(off, stale.data(), version);  // cache holds the write -> cache wins
  const NodeRecordView view{stale.data() + geom_.offset_to_node(target), dim_};
  for (uint64_t d = 0; d < dim_; ++d) {
    EXPECT_FLOAT_EQ(view.coords()[d], coords[d]) << "d=" << d;
  }
}

#if defined(ALAYA_LASER_USE_LIBAIO)
// The async update search must return exactly what the sync deterministic
// scheduler returns when every read is a miss (empty NodeCache): both absorb
// in neighbor-list order, so the frontier evolution — and the result — is
// byte-identical. Cache hits only relax tie-ordering (the documented sync
// async-vs-deterministic contract), which an empty cache sidesteps.
TEST_F(DiskPageIOTest, AsyncGreedySearchMatchesDeterministicSync) {
  if (!alaya::UringReactor::is_available()) {
    GTEST_SKIP() << "io_uring not available on this kernel";
  }
  build(300, 16, 24);

  auto reader = alaya::storage::io::open_page_reader(
      index_path(), {}, alaya::storage::io::PageReaderBackend::libaio);

  // Medoid lives at byte 16 of the header sector.
  uint32_t medoid = 0;
  {
    std::ifstream in(index_path(), std::ios::binary);
    in.seekg(16);
    in.read(reinterpret_cast<char *>(&medoid), sizeof(medoid));
    ASSERT_TRUE(in.good());
  }

  const alaya::diskann::NodeCache cache;  // empty: every read is a miss
  alaya::diskann::SearchContext ctx;
  ctx.reader = reader.get();
  ctx.geom = &geom_;
  ctx.cache = &cache;
  ctx.pq = nullptr;
  ctx.medoid = medoid;
  ctx.num_points = n_;

  alaya::diskann::SearchParams sp;
  sp.search_list_size = 30;
  sp.use_pq = false;
  sp.rerank = false;
  sp.deterministic = true;

  alaya::diskann::ThreadDataScratchConfig cfg;
  cfg.n_page_slots = 8;
  cfg.page_size = geom_.page_size;
  cfg.pq_table_entries = 0;
  cfg.max_slot_id = n_;
  cfg.max_degree = r_;
  cfg.search_list_size = sp.search_list_size;
  cfg.query_dim = dim_;
  cfg.buffer_alignment = reader->constraints().buffer_alignment;
  alaya::diskann::ThreadData td_sync;
  td_sync.alloc_scratch(cfg);
  alaya::diskann::ThreadData td_async;
  td_async.alloc_scratch(cfg);

  alaya::UringReactor reactor;
  coro::thread_pool pool{{.thread_count = 2,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  constexpr int fd = -1;

  const std::vector<float> queries = make_vectors(24, dim_, /*seed=*/777);
  uint32_t top_k = 10;
  for (uint32_t q = 0; q < 24; ++q) {
    const float *query = queries.data() + static_cast<uint64_t>(q) * dim_;
    const auto sync_out = alaya::diskann::disk_greedy_search(ctx, query, top_k, sp, td_sync,
                                                             /*stats=*/nullptr);
    auto async_task = [&]() -> coro::task<std::vector<std::pair<uint32_t, float>>> {
      co_await pool.schedule();
      co_return co_await alaya::diskann::disk_greedy_search_async(
          ctx, query, top_k, sp, td_async, /*stats=*/nullptr, reactor, pool, fd);
    };
    const auto async_out = coro::sync_wait(async_task());
    ASSERT_EQ(async_out.size(), sync_out.size()) << "query " << q;
    for (size_t i = 0; i < sync_out.size(); ++i) {
      EXPECT_EQ(async_out[i].first, sync_out[i].first) << "query " << q << " rank " << i;
      EXPECT_EQ(async_out[i].second, sync_out[i].second) << "query " << q << " rank " << i;
    }
  }

  td_sync.free_scratch();
  td_async.free_scratch();
  reader->shutdown();
}

// Same contract for the PQ beam variant: with an empty NodeCache the async
// per-beam waves process misses in popped order — byte-identical to the sync
// deterministic scheduler.
TEST_F(DiskPageIOTest, AsyncPqBeamSearchMatchesDeterministicSync) {
  if (!alaya::UringReactor::is_available()) {
    GTEST_SKIP() << "io_uring not available on this kernel";
  }
  const uint32_t pq_chunks = 4;
  build(300, 16, 24, pq_chunks);

  auto reader = alaya::storage::io::open_page_reader(
      index_path(), {}, alaya::storage::io::PageReaderBackend::libaio);

  uint32_t medoid = 0;
  {
    std::ifstream in(index_path(), std::ios::binary);
    in.seekg(16);
    in.read(reinterpret_cast<char *>(&medoid), sizeof(medoid));
    ASSERT_TRUE(in.good());
  }

  alaya::diskann::PQTable pq;
  pq.load((dir_ / "pq_pivots.bin").string(),
          (dir_ / "pq_compressed.bin").string(),
          n_,
          dim_,
          pq_chunks);

  const alaya::diskann::NodeCache cache;  // empty: every read is a miss
  alaya::diskann::SearchContext ctx;
  ctx.reader = reader.get();
  ctx.geom = &geom_;
  ctx.cache = &cache;
  ctx.pq = &pq;
  ctx.medoid = medoid;
  ctx.num_points = n_;

  alaya::diskann::SearchParams sp;
  sp.search_list_size = 30;
  sp.use_pq = true;
  sp.rerank = false;
  sp.deterministic = true;

  alaya::diskann::ThreadDataScratchConfig cfg;
  cfg.n_page_slots = 8;
  cfg.page_size = geom_.page_size;
  cfg.pq_table_entries = pq_chunks * alaya::diskann::kPQNumCentroids;
  cfg.max_slot_id = n_;
  cfg.max_degree = r_;
  cfg.search_list_size = sp.search_list_size;
  cfg.query_dim = dim_;
  cfg.buffer_alignment = reader->constraints().buffer_alignment;
  alaya::diskann::ThreadData td_sync;
  td_sync.alloc_scratch(cfg);
  alaya::diskann::ThreadData td_async;
  td_async.alloc_scratch(cfg);

  alaya::UringReactor reactor;
  coro::thread_pool pool{{.thread_count = 2,
                          .on_thread_start_functor = nullptr,
                          .on_thread_stop_functor = nullptr}};
  constexpr int fd = -1;

  const std::vector<float> queries = make_vectors(24, dim_, /*seed=*/888);
  uint32_t top_k = 10;
  for (uint32_t q = 0; q < 24; ++q) {
    const float *query = queries.data() + static_cast<uint64_t>(q) * dim_;
    const auto sync_out = alaya::diskann::cached_beam_search(ctx, query, top_k, sp, td_sync,
                                                             /*stats=*/nullptr);
    auto async_task = [&]() -> coro::task<std::vector<std::pair<uint32_t, float>>> {
      co_await pool.schedule();
      co_return co_await alaya::diskann::pq_beam_search_async(
          ctx, query, top_k, sp, td_async, /*stats=*/nullptr, reactor, pool, fd);
    };
    const auto async_out = coro::sync_wait(async_task());
    ASSERT_EQ(async_out.size(), sync_out.size()) << "query " << q;
    for (size_t i = 0; i < sync_out.size(); ++i) {
      EXPECT_EQ(async_out[i].first, sync_out[i].first) << "query " << q << " rank " << i;
      EXPECT_EQ(async_out[i].second, sync_out[i].second) << "query " << q << " rank " << i;
    }
  }

  td_sync.free_scratch();
  td_async.free_scratch();
  reader->shutdown();
}
// With ctx.page_io set, a search FILLS the shard cache with every page it
// reads; repeating the identical deterministic query is then served entirely
// from the pool — zero device reads — with an identical result. This is the
// unified-buffer-pool behavior (Yi) the peek/fill pair exists for.
TEST_F(DiskPageIOTest, SearchFillMakesRepeatSearchHitThePool) {
  build(300, 16, 24);
  DiskPageIO io(index_path(), geom_, /*page_cache_capacity=*/4096);

  auto reader = alaya::storage::io::open_page_reader(
      index_path(), {}, alaya::storage::io::PageReaderBackend::libaio);

  uint32_t medoid = 0;
  {
    std::ifstream in(index_path(), std::ios::binary);
    in.seekg(16);
    in.read(reinterpret_cast<char *>(&medoid), sizeof(medoid));
    ASSERT_TRUE(in.good());
  }

  const alaya::diskann::NodeCache cache;  // empty BFS cache: only the pool can hit
  alaya::diskann::SearchContext ctx;
  ctx.reader = reader.get();
  ctx.geom = &geom_;
  ctx.cache = &cache;
  ctx.pq = nullptr;
  ctx.medoid = medoid;
  ctx.num_points = n_;
  ctx.page_io = &io;

  alaya::diskann::SearchParams sp;
  sp.search_list_size = 30;
  sp.use_pq = false;
  sp.rerank = false;
  sp.deterministic = true;

  alaya::diskann::ThreadDataScratchConfig cfg;
  cfg.n_page_slots = 8;
  cfg.page_size = geom_.page_size;
  cfg.pq_table_entries = 0;
  cfg.max_slot_id = n_;
  cfg.max_degree = r_;
  cfg.search_list_size = sp.search_list_size;
  cfg.query_dim = dim_;
  cfg.buffer_alignment = reader->constraints().buffer_alignment;
  alaya::diskann::ThreadData td;
  td.alloc_scratch(cfg);

  const std::vector<float> query = make_vectors(1, dim_, /*seed=*/4242);
  alaya::diskann::SearchStats cold;
  const auto out_cold = alaya::diskann::disk_greedy_search(ctx, query.data(), 10, sp, td, &cold);
  alaya::diskann::SearchStats warm;
  const auto out_warm = alaya::diskann::disk_greedy_search(ctx, query.data(), 10, sp, td, &warm);

  EXPECT_GT(cold.n_ios, 0U);
  EXPECT_EQ(warm.n_ios, 0U) << "repeat search should be served from the pool";
  // Identical deterministic traversal: every cold-run node (device read OR
  // co-resident page hit — small indices pack ~24 nodes per page, so even the
  // cold run hits pages its own earlier reads filled) is a pool hit when warm.
  EXPECT_EQ(warm.n_page_cache_hits, cold.n_page_cache_hits + cold.n_ios);
  ASSERT_EQ(out_warm.size(), out_cold.size());
  for (size_t i = 0; i < out_cold.size(); ++i) {
    EXPECT_EQ(out_warm[i].first, out_cold[i].first) << "rank " << i;
    EXPECT_EQ(out_warm[i].second, out_cold[i].second) << "rank " << i;
  }

  td.free_scratch();
  reader->shutdown();
}

#endif  // ALAYA_LASER_USE_LIBAIO

#else  // !__linux__

TEST(DiskPageIOTest, SkippedOnNonLinux) {
  GTEST_SKIP() << "DiskPageIO in-place updates require Linux O_DIRECT";
}

#endif  // __linux__

}  // namespace
