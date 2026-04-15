/**
 * @file quantized_graph.hpp
 * @brief Disk-resident quantized graph with zero-alloc beam search.
 *
 * Ported from Laser's qg.hpp with major refactoring:
 * - All per-query allocations replaced with LaserSearchContext members
 * - HashBasedBooleanSet replaced with TaggedVisitedSet (O(1) reset)
 * - std::deque/unordered_map replaced with FixedRingBuffer/OngoingTable
 * - batch_search uses thread-affine context (one acquire per thread)
 * - OMP affinity check added in set_params()
 */
// NOLINTBEGIN(performance-no-int-to-ptr,google-runtime-references,modernize-use-trailing-return-type)

#pragma once

#include <libaio.h>
#include <omp.h>
#include <sys/mman.h>

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "utils/hash_map.hpp"

// NUMA support: requires explicit CMake option to link libnuma
#ifdef LASER_USE_NUMA
  #include <numaif.h>
#endif

#include "index/laser/io/aligned_file_reader.hpp"
#include "index/laser/laser_common.hpp"
#include "index/laser/laser_search_context.hpp"
#include "index/laser/laser_types.hpp"
#include "index/laser/qg_query.hpp"
#include "index/laser/qg_scanner.hpp"
#include "index/laser/quantization/rabitq.hpp"
#include "index/laser/thread_data.hpp"
#include "index/laser/transform/fht_rotator.hpp"
#include "index/laser/transform/pca_transform.hpp"
#include "simd/distance_l2.hpp"
#include "utils/aligned_array.hpp"
#include "utils/candidate_list.hpp"
#include "utils/concurrent_queue.hpp"
#include "utils/memory.hpp"
#include "utils/prefetch.hpp"

class MemLimitQGBuilder;  // Forward declaration for benchmark friend access

namespace symqg {

struct Factor {
  float triple_x_;
  float factor_dq_;
  float factor_vq_;
};

/**
 * @brief Check if OMP_PROC_BIND is set; warn if not.
 */
inline void check_omp_affinity() {
  static bool warned = false;
  if (warned) {
    return;
  }
  const char *proc_bind = std::getenv("OMP_PROC_BIND");
  if (proc_bind == nullptr || std::string(proc_bind) == "false") {
    std::cerr << "[WARN] OMP_PROC_BIND is not set. OpenMP threads may migrate "
                 "across NUMA nodes, degrading search latency. Set: "
                 "export OMP_PROC_BIND=spread OMP_PLACES=cores\n";
    warned = true;
  }
}

/**
 * @brief Pre-flight check for AIO kernel capacity.
 */
inline void check_aio_capacity(size_t num_threads, size_t events_per_thread) {
  size_t required = num_threads * events_per_thread;
  std::ifstream nr_file("/proc/sys/fs/aio-nr");
  std::ifstream max_file("/proc/sys/fs/aio-max-nr");
  if (!nr_file.is_open() || !max_file.is_open()) {
    return;  // non-Linux or restricted access
  }
  size_t aio_nr = 0;
  size_t aio_max_nr = 0;
  nr_file >> aio_nr;
  max_file >> aio_max_nr;
  size_t available = aio_max_nr - aio_nr;
  if (required > available) {
    std::cerr << "[WARN] AIO quota may be insufficient. Required: " << required << " slots ("
              << num_threads << " threads * " << events_per_thread
              << " events), Available: " << available << " (aio-nr=" << aio_nr
              << ", aio-max-nr=" << aio_max_nr << "). "
              << "Fix: sudo sysctl -w fs.aio-max-nr=1048576\n";
  }
}

class QuantizedGraph {
  friend class QGBuilder;
  friend class ::MemLimitQGBuilder;

 private:
  size_t num_points_ = 0;
  size_t degree_bound_ = 0;
  size_t dimension_ = 0;
  size_t residual_dimension_ = 0;
  size_t padded_dim_ = 0;
  PID entry_point_ = 0;

  data::Array<float, std::vector<size_t>, alaya::AlignedAlloc<float, 1 << 22, true>> data_;
  QGScanner scanner_;
  FHTRotator rotator_;
  PCATransform pca_transform_;
  LinuxAlignedFileReader aligned_file_reader_;
  alaya::ConcurrentQueue<ThreadData> thread_data_;
  size_t ef_search_ = 200;

  size_t node_len_ = 0;
  size_t page_size_ = 0;
  size_t node_per_page_ = 0;

  size_t max_beam_width_ = 16;
  std::string index_file_name_;

  std::vector<PID> medoids_;
  std::vector<float> medoids_vector_;
  std::vector<PID> cache_ids_;
  std::vector<char> cache_nodes_;
  alaya::fast::map<PID, char *> caches_;

  // NUMA mmap tracking: non-null when cache was allocated via mmap+mbind.
  void *numa_mmap_ptr_ = nullptr;
  size_t numa_mmap_size_ = 0;

  size_t nthreads_ = 1;

  struct ThreadDataGuard {
    alaya::ConcurrentQueue<ThreadData> &pool_;
    ThreadData data_;

    ThreadDataGuard(alaya::ConcurrentQueue<ThreadData> &pool, ThreadData &&data)
        : pool_(pool), data_(std::move(data)) {}

    ~ThreadDataGuard() {
      pool_.push(std::move(data_));
      pool_.push_notify_one();
    }

    ThreadDataGuard(const ThreadDataGuard &) = delete;
    auto operator=(const ThreadDataGuard &) -> ThreadDataGuard & = delete;
    ThreadDataGuard(ThreadDataGuard &&) = delete;
    auto operator=(ThreadDataGuard &&) -> ThreadDataGuard & = delete;
  };

  size_t res_dim_offset_ = 0;
  size_t code_offset_ = 0;
  size_t factor_offset_ = 0;
  size_t neighbor_offset_ = 0;
  size_t row_offset_ = 0;

  static constexpr size_t kMetaNumPointsIdx = 0;
  static constexpr size_t kMetaMainDimIdx = 1;
  static constexpr size_t kMetaEntryPointIdx = 2;
  static constexpr size_t kMetaNodeLenIdx = 3;
  static constexpr size_t kMetaNodePerPageIdx = 4;
  static constexpr size_t kMetaFileSizeIdx = 8;

  void initialize();

  void disk_search_qg(const float *__restrict__ query,
                      uint32_t knn,
                      uint32_t *__restrict__ results,
                      ThreadData &data);

  [[nodiscard]] auto get_page_offset(uint64_t node_id) const -> uint64_t {
    return kSectorLen + page_size_ * (node_id / node_per_page_);
  }

  [[nodiscard]] auto offset_to_node(uint64_t node_id) const -> uint64_t {
    return (node_id % node_per_page_) * node_len_;
  }

  [[nodiscard]] auto full_dimension() const -> size_t { return dimension_ + residual_dimension_; }

  [[nodiscard]] auto gen_index_path(const char *prefix) const -> std::string {
    return std::string(prefix) + "_R" + std::to_string(degree_bound_) + "_MD" +
           std::to_string(dimension_) + ".index";
  }

  auto scan_neighbors(const QGQuery &q_obj,
                      const float *cur_data,
                      float *appro_dist,
                      alaya::CandidateList<float, uint32_t> &search_pool,
                      uint32_t cur_degree,
                      TaggedVisitedSet &visited,
                      LaserSearchContext &ctx) const -> float;

  void init_thread_pool();
  [[nodiscard]] auto read_index_metadata() const -> std::vector<uint64_t>;
  void validate_index_metadata_or_throw(const std::vector<uint64_t> &metadata) const;
  void load_rotator_from_disk();
  void load_optional_pca_from_disk(const char *prefix);
  [[nodiscard]] auto compute_online_cache_num(float search_dram_budget) const -> size_t;
  void load_optional_cache_from_disk(size_t online_cache_num);
  [[nodiscard]] static auto read_binary_exact(std::ifstream &input, void *dst, size_t bytes)
      -> bool;
  [[nodiscard]] auto load_medoid_ids_from_disk(const std::string &medoids_indices_file) -> bool;
  [[nodiscard]] auto load_medoid_vectors_from_disk(const std::string &medoids_file) -> bool;
  [[nodiscard]] auto load_cache_ids_from_disk(std::ifstream &cache_ids_input,
                                              size_t online_cache_num) -> size_t;
  [[nodiscard]] auto read_cache_nodes_header(std::ifstream &cache_vectors_input,
                                             size_t &cache_nodes_num,
                                             size_t &node_len) const -> bool;
  [[nodiscard]] auto load_cache_nodes_standard(std::ifstream &cache_vectors_input,
                                               size_t cache_bytes) -> bool;
  auto acquire_thread_data() -> ThreadData;
  void rebuild_cache_lookup(char *cache_base, size_t cache_count);
#ifdef LASER_USE_NUMA
  auto try_load_cache_with_numa(std::ifstream &cache_vectors_input, size_t cache_bytes) -> bool;
#endif
  void reset_loaded_state_before_reload();

  template <typename ProcessFn, typename WaitFn>
  void process_prepared_nodes(size_t &remaining,
                              ProcessFn &&process_node,
                              WaitFn &&wait_for_nodes,
                              LaserSearchContext &ctx,
                              size_t prefetch_lines);

 public:
  explicit QuantizedGraph(size_t num, size_t max_deg, size_t main_dim, size_t dim);
  ~QuantizedGraph();

  [[nodiscard]] auto num_vertices() const { return num_points_; }
  [[nodiscard]] auto dimension() const { return dimension_; }
  [[nodiscard]] auto residual_dimension() const { return residual_dimension_; }
  [[nodiscard]] auto degree_bound() const { return degree_bound_; }
  [[nodiscard]] auto entry_point() const { return entry_point_; }
  void set_ep(PID entry) { entry_point_ = entry; }

  void load_disk_index(const char *prefix, float search_dram_budget);
  void set_params(size_t ef_search, size_t num_threads, int beam_width);
  void load_medoids(const char *prefix);
  void load_cache(const std::string &cache_ids_file,
                  const std::string &cache_nodes_file,
                  size_t online_cache_num);

  void search(const float *__restrict__ query, uint32_t knn, uint32_t *__restrict__ results);

  void batch_search(const float *__restrict__ query,
                    uint32_t knn,
                    uint32_t *__restrict__ results,
                    size_t num_queries);

  [[nodiscard]] auto cached_node_count() const -> size_t { return cache_ids_.size(); }
  [[nodiscard]] auto cache_size_bytes() const -> size_t { return cache_nodes_.size(); }

  void destroy_thread_data();
};

// ============================================================================
// Implementation
// ============================================================================

// QuantizedGraph node layout (all offsets in float-count units):
//
//   [0 .. dimension_)                    main PCA-rotated vector    (32 bits/dim)
//   [dimension_ .. code_offset_)         residual vector            (32 bits/dim)
//   [code_offset_ .. factor_offset_)     RaBitQ packed codes        (padded_dim/64 * 2 * degree uint8s)
//   [factor_offset_ .. neighbor_offset_) correction factors         (3 floats per neighbor: triple_x, factor_dq, factor_vq)
//   [neighbor_offset_ .. row_offset_)    neighbor PIDs              (1 uint32 per neighbor)
//
// node_len_ in bytes = (32*main_dim + 32*residual_dim + 128*degree + degree*padded_dim) / 8
//                    = main_vec_bytes + residual_bytes + factor_bytes(12*deg) + code_bytes(padded*deg/8) + neighbor_bytes(4*deg)
//
// Nodes are packed into pages for Direct I/O. node_per_page_ = floor(kSectorLen / node_len_).
// page_size_ is rounded up to kSectorLen (4096) alignment for O_DIRECT.
inline QuantizedGraph::QuantizedGraph(size_t num, size_t max_deg, size_t main_dim, size_t dim)
    : num_points_(num),
      degree_bound_(max_deg),
      dimension_(main_dim),
      residual_dimension_(dim - main_dim),
      padded_dim_(1 << alaya::math::ceil_log2(main_dim)),
      scanner_(padded_dim_, degree_bound_),
      rotator_(main_dim),
      node_len_((32 * main_dim + 32 * (dim - main_dim) + 128 * max_deg + max_deg * padded_dim_) /
                8) {
  if (num == 0) {
    throw std::invalid_argument("QuantizedGraph: num_points must be > 0");
  }
  if (max_deg == 0) {
    throw std::invalid_argument("QuantizedGraph: max_degree must be > 0");
  }
  if (main_dim == 0 || main_dim > dim) {
    throw std::invalid_argument("QuantizedGraph: main_dim must be in (0, dim]");
  }
  node_per_page_ = std::max(static_cast<size_t>(1), kSectorLen / node_len_);
  page_size_ = (node_per_page_ * node_len_ + kSectorLen - 1) / kSectorLen * kSectorLen;

  if (main_dim != padded_dim_) {
    throw std::runtime_error("Laser: dimension must be a power of 2");
  }
  initialize();
}

inline QuantizedGraph::~QuantizedGraph() {
  destroy_thread_data();
#ifdef LASER_USE_NUMA
  if (numa_mmap_ptr_ != nullptr) {
    munmap(numa_mmap_ptr_, numa_mmap_size_);
    numa_mmap_ptr_ = nullptr;
    numa_mmap_size_ = 0;
  }
#endif
}

inline void QuantizedGraph::initialize() {
  assert(padded_dim_ % 64 == 0);
  assert(padded_dim_ >= dimension_);

  res_dim_offset_ = dimension_;                                           // residual vector start (float units)
  code_offset_ = dimension_ + residual_dimension_;                        // RaBitQ codes start
  factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // correction factors start
  neighbor_offset_ = factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);  // neighbor PIDs start
  row_offset_ = neighbor_offset_ + degree_bound_;                         // end of node (float units)
}

inline void QuantizedGraph::init_thread_pool() {
  check_omp_affinity();

  size_t aio_events = 2 * max_beam_width_;
  check_aio_capacity(nthreads_, aio_events);

  aligned_file_reader_.open(index_file_name_);

  size_t full_dim = full_dimension();

#pragma omp parallel for num_threads(static_cast<int>(nthreads_))
  for (size_t thread = 0; thread < nthreads_; thread++) {
#pragma omp critical
    {
      aligned_file_reader_.register_thread(aio_events);
      ThreadData data;
      data.ctx_ = aligned_file_reader_.get_ctx();
      data.allocate(padded_dim_,
                    degree_bound_,
                    max_beam_width_,
                    page_size_,
                    ef_search_,
                    full_dim,
                    num_points_);
      thread_data_.push(std::move(data));
    }
  }
}

inline void QuantizedGraph::set_params(size_t ef_search, size_t num_threads, int beam_width) {
  if (ef_search == 0) {
    throw std::invalid_argument("set_params: ef_search must be > 0");
  }
  if (num_threads == 0) {
    throw std::invalid_argument("set_params: num_threads must be > 0");
  }
  if (beam_width <= 0) {
    throw std::invalid_argument("set_params: beam_width must be > 0");
  }

  nthreads_ = num_threads;
  max_beam_width_ = static_cast<size_t>(beam_width);
  ef_search_ = ef_search;

  destroy_thread_data();

  if (index_file_name_.empty()) {
    throw std::runtime_error("Laser: load index before calling set_params()");
  }

  init_thread_pool();
}

inline void QuantizedGraph::destroy_thread_data() {
  while (thread_data_.size() > 0) {
    ThreadData data = thread_data_.pop();
    while (data.sector_scratch_ == nullptr) {
      thread_data_.wait_for_push_notify();
      data = thread_data_.pop();
    }
    data.deallocate();
  }
  aligned_file_reader_.deregister_all_threads();
  aligned_file_reader_.close();
}

inline void QuantizedGraph::search(const float *__restrict__ query,
                                   uint32_t knn,
                                   uint32_t *__restrict__ results) {
  ThreadDataGuard guard(thread_data_, acquire_thread_data());
  disk_search_qg(query, knn, results, guard.data_);
}

inline void QuantizedGraph::batch_search(const float *__restrict__ query,
                                         uint32_t knn,
                                         uint32_t *__restrict__ results,
                                         size_t num_queries) {
  // Thread-affine context: each thread acquires once, reuses across queries
#pragma omp parallel num_threads(static_cast<int>(nthreads_))
  {
    ThreadDataGuard guard(thread_data_, acquire_thread_data());

    size_t full_dim = full_dimension();
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < num_queries; ++i) {
      disk_search_qg(query + i * full_dim, knn, results + i * knn, guard.data_);
    }
  }
}

inline auto QuantizedGraph::acquire_thread_data() -> ThreadData {
  ThreadData data = thread_data_.pop();
  while (data.sector_scratch_ == nullptr) {
    thread_data_.wait_for_push_notify();
    data = thread_data_.pop();
  }
  return data;
}

inline void QuantizedGraph::disk_search_qg(const float *__restrict__ query,
                                           uint32_t knn,
                                           uint32_t *__restrict__ results,
                                           ThreadData &data) {
  auto &ctx = data.search_ctx_;
  ctx.reset();
  data.search_pool_.clear();

  // PCA Transform
  const float *transformed_query = query;
  if (pca_transform_.is_loaded()) {
    pca_transform_.transform(query, data.pca_query_scratch_);
    transformed_query = data.pca_query_scratch_;
  }

  // Query preparation (uses ctx buffers — zero alloc)
  QGQuery q_obj(transformed_query, padded_dim_);
  q_obj.query_prepare(rotator_, scanner_, ctx);

  const float *residual_query = transformed_query + dimension_;
  float sqr_qr = (residual_dimension_ > 0)
                     ? alaya::simd::l2_sqr_norm(residual_query, residual_dimension_)
                     : 0.0F;
  q_obj.set_sqr_qr(sqr_qr);

  // Initialize search pool with entry points
  if (!medoids_.empty()) {
    PID best_medoid = 0;
    float best_dist = FLT_MAX;
    size_t full_dim = full_dimension();
    for (size_t cur_m = 0; cur_m < medoids_.size(); cur_m++) {
      auto cur_dist = alaya::simd::l2_sqr<float, float>(transformed_query,
                                                        medoids_vector_.data() + full_dim * cur_m,
                                                        dimension_);
      if (cur_dist < best_dist) {
        best_medoid = medoids_[cur_m];
        best_dist = cur_dist;
      }
    }
    data.search_pool_.insert(best_medoid, FLT_MAX);
  }
  data.search_pool_.insert(entry_point_, FLT_MAX);

  auto &res_pool = ctx.result_buffer();
  res_pool.reset(knn);

  auto &ongoing = ctx.ongoing_table();
  auto &prepared = ctx.prepared_ring();
  auto &visited = ctx.visited_set();
  auto &cache_nhoods = ctx.cache_nhoods();
  auto &free_slots = ctx.free_slot_stack();

  // Initialize free slot stack with sector scratch buffers
  for (size_t i = 0; i < 2 * max_beam_width_; i++) {
    free_slots.push(data.sector_scratch_ + i * page_size_);
  }

  size_t frontier_req_count = 0;
  size_t cur_beam_size = 1;

  // Node processing lambda
  auto process_node = [&](PID cur_node, float *cur_data) {
    float sqr_y = scan_neighbors(q_obj,
                                 cur_data,
                                 ctx.appro_dist(),
                                 data.search_pool_,
                                 degree_bound_,
                                 visited,
                                 ctx);
    if (residual_dimension_ > 0) {
      float *residual_data = cur_data + dimension_;
      sqr_y += alaya::simd::l2_sqr<float, float>(reinterpret_cast<const float *>(residual_data),
                                                 residual_query,
                                                 residual_dimension_);
    }
    res_pool.insert(cur_node, sqr_y);
  };

  // I/O completion handler: non-blocking probe first, then blocking fallback.
  // Non-blocking catches already-completed events without syscall overhead.
  // Blocking fallback avoids infinite spin when events are still in-flight.
  // Prefetch node data into L2 when I/O completes, so it's warm by process time
  size_t prefetch_lines = std::min(node_len_ / 64, static_cast<size_t>(20));

  auto collect_events = [&](io_event *evts, int ret) {
    for (int i = 0; i < ret; i++) {
      auto id = static_cast<PID>(reinterpret_cast<uintptr_t>(evts[i].data));
      // Check AIO completion status: res < 0 means I/O error,
      // res != page_size_ means short read (partial page)
      if (evts[i].res < 0 ||
          static_cast<size_t>(evts[i].res) < page_size_) {
        // I/O failed or short read: reclaim slot, skip this node
        char *buf = ongoing.find(id);
        if (buf != nullptr) {
          ongoing.erase(id);
          free_slots.push(buf);
        }
        continue;
      }
      char *buf = ongoing.find(id);
      if (buf != nullptr) {
        const char *node_ptr = buf + offset_to_node(id);
        alaya::mem_prefetch_l2(node_ptr, prefetch_lines);
        prepared.push_back({id, buf});
        ongoing.erase(id);
      }
    }
  };

  auto wait_for_nodes = [&]() {
    io_event *evts = ctx.io_events();
    auto max_nr = static_cast<int64_t>(cur_beam_size);
    // Fast path: non-blocking poll
    int ret = io_getevents(data.ctx_, 0, max_nr, evts, nullptr);
    if (ret > 0) {
      collect_events(evts, ret);
      return;
    }
    // Slow path: block until at least 1 event
    ret = io_getevents(data.ctx_, 1, max_nr, evts, nullptr);
    if (ret > 0) {
      collect_events(evts, ret);
    }
  };

  size_t previous_remain_num = 0;
  AlignedRead *frontier_reqs = ctx.frontier_reqs();

  // Main search loop
  while (data.search_pool_.has_next()) {
    frontier_req_count = 0;
    cache_nhoods.clear();
    size_t n_ops = 0;

    cur_beam_size =
        std::min(max_beam_width_,
                 static_cast<size_t>(std::ceil(2.0F * static_cast<float>(cur_beam_size))));

    // Build I/O request batch
    while (data.search_pool_.has_next() && frontier_req_count < cur_beam_size) {
      PID cur_node = data.search_pool_.pop();
      if (visited.get(cur_node)) {
        continue;
      }
      visited.set(cur_node);

      auto cache_it = caches_.find(cur_node);
      if (cache_it != caches_.end()) {
        cache_nhoods.emplace_back(cur_node, cache_it->second);
      } else {
        if (free_slots.empty()) {
          break;
        }
        char *slot = free_slots.pop();
        ongoing.insert(cur_node, slot);
        frontier_reqs[frontier_req_count] =
            AlignedRead(get_page_offset(cur_node), page_size_, cur_node, slot);
        ++frontier_req_count;
      }
    }

    // Submit async I/O (zero-copy: pre-allocated iocb buffers)
    if (frontier_req_count > 0) {
      n_ops = aligned_file_reader_.submit_reqs(frontier_reqs,
                                               frontier_req_count,
                                               data.ctx_,
                                               ctx.iocb_buf(),
                                               ctx.iocb_ptrs_buf());
    }

    // Process cached nodes with look-ahead prefetch (no double lookup)
    for (size_t ci = 0; ci < cache_nhoods.size(); ++ci) {
      if (ci + 1 < cache_nhoods.size()) {
        alaya::mem_prefetch_l1(reinterpret_cast<const char *>(cache_nhoods[ci + 1].second),
                               prefetch_lines);
      }
      process_node(cache_nhoods[ci].first, reinterpret_cast<float *>(cache_nhoods[ci].second));
    }

    // Pipelined processing
    auto remain_num = static_cast<size_t>(0.5 * n_ops);
    size_t need_process_num = n_ops + previous_remain_num - remain_num;
    previous_remain_num = remain_num;

    process_prepared_nodes(need_process_num, process_node, wait_for_nodes, ctx, prefetch_lines);
  }

  // Drain remaining
  process_prepared_nodes(previous_remain_num, process_node, wait_for_nodes, ctx, prefetch_lines);

  res_pool.copy_results(results);
}

inline auto QuantizedGraph::scan_neighbors(const QGQuery &q_obj,
                                           const float *cur_data,
                                           float *appro_dist,
                                           alaya::CandidateList<float, uint32_t> &search_pool,
                                           uint32_t cur_degree,
                                           TaggedVisitedSet &visited,
                                           LaserSearchContext &ctx) const -> float {
  auto sqr_y = alaya::simd::l2_sqr<float, float>(q_obj.query_data(), cur_data, dimension_);

  const auto *packed_code = reinterpret_cast<const uint8_t *>(&cur_data[code_offset_]);
  const auto *factor = &cur_data[factor_offset_];
  scanner_.scan_neighbors(appro_dist,
                          ctx.lut(),
                          sqr_y,
                          q_obj.lower_val(),
                          q_obj.width(),
                          q_obj.sqr_qr(),
                          q_obj.sumq(),
                          packed_code,
                          factor,
                          ctx);

  const PID *ptr_nb = reinterpret_cast<const PID *>(&cur_data[neighbor_offset_]);
  for (uint32_t i = 0; i < cur_degree; ++i) {
    PID cur_neighbor = ptr_nb[i];
    float tmp_dist = appro_dist[i];
    if (visited.get(cur_neighbor)) {
      continue;
    }
    search_pool.insert(cur_neighbor, tmp_dist);
  }

  return sqr_y;
}

template <typename ProcessFn, typename WaitFn>
inline void QuantizedGraph::process_prepared_nodes(size_t &remaining,
                                                   ProcessFn &&process_node,
                                                   WaitFn &&wait_for_nodes,
                                                   LaserSearchContext &ctx,
                                                   size_t prefetch_lines) {
  auto &prepared = ctx.prepared_ring();
  auto &free_slots = ctx.free_slot_stack();

  while (remaining > 0) {
    if (!prepared.empty()) {
      auto node = prepared.pop_front();
      if (!prepared.empty()) {
        auto &next = prepared.front();
        alaya::mem_prefetch_l1(next.second + offset_to_node(next.first), prefetch_lines);
      }
      process_node(node.first,
                   reinterpret_cast<float *>(node.second + offset_to_node(node.first)));
      --remaining;
      free_slots.push(node.second);
    } else {
      wait_for_nodes();
    }
  }
}

inline auto QuantizedGraph::read_binary_exact(std::ifstream &input, void *dst, size_t bytes)
    -> bool {
  if (bytes == 0) {
    return true;
  }
  input.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(bytes));
  return static_cast<size_t>(input.gcount()) == bytes;
}

inline auto QuantizedGraph::read_index_metadata() const -> std::vector<uint64_t> {
  std::ifstream input(index_file_name_, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open index file: " + index_file_name_);
  }

  std::vector<uint64_t> metadata(kSectorLen / sizeof(uint64_t), 0);
  input.read(reinterpret_cast<char *>(metadata.data()), kSectorLen);
  if (!input) {
    throw std::runtime_error("Failed to read index metadata header: " + index_file_name_);
  }
  return metadata;
}

inline void QuantizedGraph::validate_index_metadata_or_throw(
    const std::vector<uint64_t> &metadata) const {
  if (metadata[kMetaNumPointsIdx] != num_points_ || metadata[kMetaMainDimIdx] != dimension_ ||
      metadata[kMetaNodeLenIdx] != node_len_ || metadata[kMetaNodePerPageIdx] != node_per_page_) {
    throw std::runtime_error(
        "Index metadata mismatch. Expected: num_points=" + std::to_string(num_points_) +
        " dim=" + std::to_string(dimension_) + " node_len=" + std::to_string(node_len_) +
        " node_per_page=" + std::to_string(node_per_page_) +
        ". Got: " + std::to_string(metadata[kMetaNumPointsIdx]) + "/" +
        std::to_string(metadata[kMetaMainDimIdx]) + "/" +
        std::to_string(metadata[kMetaNodeLenIdx]) + "/" +
        std::to_string(metadata[kMetaNodePerPageIdx]));
  }

  auto file_size = std::filesystem::file_size(index_file_name_);
  if (metadata[kMetaFileSizeIdx] != file_size) {
    throw std::runtime_error("Index file size mismatch: header says " +
                             std::to_string(metadata[kMetaFileSizeIdx]) + " but file is " +
                             std::to_string(file_size) + " bytes");
  }
}

inline void QuantizedGraph::load_rotator_from_disk() {
  std::string rotator_path = index_file_name_ + "_rotator";
  std::ifstream rotator_input(rotator_path, std::ios::binary);
  if (!rotator_input.is_open()) {
    throw std::runtime_error("Missing rotator file: " + rotator_path);
  }
  rotator_.load(rotator_input);
}

inline void QuantizedGraph::load_optional_pca_from_disk(const char *prefix) {
  std::string pca_path = std::string(prefix) + "_pca.bin";
  if (std::filesystem::exists(pca_path)) {
    pca_transform_.load(pca_path);
  } else {
    std::cerr << "[WARN] PCA file not found: " << pca_path << '\n';
  }
}

inline auto QuantizedGraph::compute_online_cache_num(float search_dram_budget) const -> size_t {
  auto cache_space = static_cast<size_t>(search_dram_budget * 1000 * 1000 * 1000 * 0.8);
  return std::min(cache_space / node_len_, static_cast<size_t>(kCacheRatio * num_points_));
}

inline void QuantizedGraph::load_optional_cache_from_disk(size_t online_cache_num) {
  std::string cache_ids_file = index_file_name_ + "_cache_ids";
  std::string cache_nodes_file = index_file_name_ + "_cache_nodes";
  if (std::filesystem::exists(cache_ids_file) && std::filesystem::exists(cache_nodes_file)) {
    load_cache(cache_ids_file, cache_nodes_file, online_cache_num);
  }
}

inline void QuantizedGraph::reset_loaded_state_before_reload() {
  if (!index_file_name_.empty()) {
    destroy_thread_data();
  }

  entry_point_ = 0;
  medoids_.clear();
  medoids_vector_.clear();
  cache_ids_.clear();
  cache_nodes_.clear();
  caches_.clear();

#ifdef LASER_USE_NUMA
  if (numa_mmap_ptr_ != nullptr) {
    munmap(numa_mmap_ptr_, numa_mmap_size_);
    numa_mmap_ptr_ = nullptr;
    numa_mmap_size_ = 0;
  }
#endif
}

inline void QuantizedGraph::load_disk_index(const char *prefix, float search_dram_budget) {
  reset_loaded_state_before_reload();
  index_file_name_ = gen_index_path(prefix);
  if (!std::filesystem::exists(index_file_name_)) {
    throw std::runtime_error("Index file not found: " + index_file_name_);
  }

  auto metadata = read_index_metadata();
  validate_index_metadata_or_throw(metadata);
  entry_point_ = static_cast<PID>(metadata[kMetaEntryPointIdx]);

  load_rotator_from_disk();
  init_thread_pool();

  load_medoids(prefix);
  load_optional_pca_from_disk(prefix);
  load_optional_cache_from_disk(compute_online_cache_num(search_dram_budget));
}

inline auto QuantizedGraph::load_medoid_ids_from_disk(const std::string &medoids_indices_file)
    -> bool {
  std::ifstream medoid_input(medoids_indices_file, std::ios::binary);
  if (!medoid_input.is_open()) {
    return false;
  }

  int32_t medoid_num = 0;
  int32_t columns = 0;
  if (!read_binary_exact(medoid_input, &medoid_num, sizeof(medoid_num)) ||
      !read_binary_exact(medoid_input, &columns, sizeof(columns))) {
    return false;
  }
  if (medoid_num <= 0 || columns != 1) {
    return false;
  }

  medoids_.resize(static_cast<size_t>(medoid_num));
  if (!read_binary_exact(medoid_input, medoids_.data(), medoids_.size() * sizeof(PID))) {
    medoids_.clear();
    return false;
  }
  return true;
}

inline auto QuantizedGraph::load_medoid_vectors_from_disk(const std::string &medoids_file) -> bool {
  std::ifstream medoid_vec_input(medoids_file, std::ios::binary);
  if (!medoid_vec_input.is_open()) {
    return false;
  }

  int32_t medoid_num = 0;
  int32_t dim = 0;
  if (!read_binary_exact(medoid_vec_input, &medoid_num, sizeof(medoid_num)) ||
      !read_binary_exact(medoid_vec_input, &dim, sizeof(dim))) {
    return false;
  }

  size_t full_dim = full_dimension();
  if (medoid_num <= 0 || dim <= 0 || static_cast<size_t>(dim) != full_dim) {
    return false;
  }

  medoids_vector_.resize(static_cast<size_t>(medoid_num) * full_dim);
  if (!read_binary_exact(medoid_vec_input,
                         medoids_vector_.data(),
                         medoids_vector_.size() * sizeof(float))) {
    medoids_vector_.clear();
    return false;
  }
  return true;
}

inline auto QuantizedGraph::load_cache_ids_from_disk(std::ifstream &cache_ids_input,
                                                     size_t online_cache_num) -> size_t {
  size_t cache_ids_num = 0;
  if (!read_binary_exact(cache_ids_input, &cache_ids_num, sizeof(size_t))) {
    return 0;
  }

  auto load_count = std::min(online_cache_num, cache_ids_num);
  cache_ids_.resize(load_count);
  if (!read_binary_exact(cache_ids_input, cache_ids_.data(), load_count * sizeof(PID))) {
    cache_ids_.clear();
    return 0;
  }

  return load_count;
}

inline auto QuantizedGraph::read_cache_nodes_header(std::ifstream &cache_vectors_input,
                                                    size_t &cache_nodes_num,
                                                    size_t &node_len) const -> bool {
  return read_binary_exact(cache_vectors_input, &cache_nodes_num, sizeof(size_t)) &&
         read_binary_exact(cache_vectors_input, &node_len, sizeof(size_t));
}

inline auto QuantizedGraph::load_cache_nodes_standard(std::ifstream &cache_vectors_input,
                                                      size_t cache_bytes) -> bool {
  cache_nodes_.resize(cache_bytes);
  if (!read_binary_exact(cache_vectors_input, cache_nodes_.data(), cache_bytes)) {
    cache_nodes_.clear();
    return false;
  }
  return true;
}

inline void QuantizedGraph::load_medoids(const char *prefix) {
  std::string medoids_indices_file = std::string(prefix) + "_medoids_indices";
  std::string medoids_file = std::string(prefix) + "_medoids";

  if (!std::filesystem::exists(medoids_file) || !std::filesystem::exists(medoids_indices_file)) {
    return;
  }

  medoids_.clear();
  medoids_vector_.clear();

  if (!load_medoid_ids_from_disk(medoids_indices_file) ||
      !load_medoid_vectors_from_disk(medoids_file)) {
    std::cerr << "[WARN] Failed to load medoid data from: " << medoids_file
              << "; falling back to entry point only\n";
    medoids_.clear();
    medoids_vector_.clear();
    return;
  }
  size_t full_dim = full_dimension();
  if (medoids_.size() * full_dim != medoids_vector_.size()) {
    std::cerr << "[WARN] Medoid data size mismatch; falling back to entry point only\n";
    medoids_.clear();
    medoids_vector_.clear();
  }
}

inline void QuantizedGraph::rebuild_cache_lookup(char *cache_base, size_t cache_count) {
  for (size_t i = 0; i < cache_count; ++i) {
    caches_[cache_ids_[i]] = cache_base + i * node_len_;
  }
}

#ifdef LASER_USE_NUMA
inline auto QuantizedGraph::try_load_cache_with_numa(std::ifstream &cache_vectors_input,
                                                     size_t cache_bytes) -> bool {
  if (cache_bytes == 0) {
    return false;
  }

  void *numa_ptr =
      mmap(nullptr, cache_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (numa_ptr == MAP_FAILED) {
    return false;
  }

  unsigned long nodemask = ~0UL;  // all nodes
  long ret = mbind(numa_ptr, cache_bytes, MPOL_INTERLEAVE, &nodemask, sizeof(nodemask) * 8, 0);
  if (ret != 0) {
    munmap(numa_ptr, cache_bytes);
    return false;
  }

  cache_vectors_input.read(reinterpret_cast<char *>(numa_ptr),
                           static_cast<std::streamsize>(cache_bytes));
  if (!cache_vectors_input) {
    munmap(numa_ptr, cache_bytes);
    return false;
  }

  // Track mmap allocation for cleanup in destructor.
  numa_mmap_ptr_ = numa_ptr;
  numa_mmap_size_ = cache_bytes;
  cache_nodes_.clear();
  rebuild_cache_lookup(reinterpret_cast<char *>(numa_ptr), cache_ids_.size());
  return true;
}
#endif

inline void QuantizedGraph::load_cache(const std::string &cache_ids_file,
                                       const std::string &cache_nodes_file,
                                       size_t online_cache_num) {
#ifdef LASER_USE_NUMA
  if (numa_mmap_ptr_ != nullptr) {
    munmap(numa_mmap_ptr_, numa_mmap_size_);
    numa_mmap_ptr_ = nullptr;
    numa_mmap_size_ = 0;
  }
#endif
  cache_ids_.clear();
  cache_nodes_.clear();
  caches_.clear();

  std::ifstream cache_ids_input(cache_ids_file, std::ios::binary);
  std::ifstream cache_vectors_input(cache_nodes_file, std::ios::binary);
  if (!cache_ids_input.is_open() || !cache_vectors_input.is_open()) {
    return;
  }

  size_t cache_count = load_cache_ids_from_disk(cache_ids_input, online_cache_num);
  if (cache_count == 0) {
    return;
  }

  size_t cache_nodes_num = 0;
  size_t stored_node_len = 0;
  if (!read_cache_nodes_header(cache_vectors_input, cache_nodes_num, stored_node_len)) {
    cache_ids_.clear();
    return;
  }
  if (stored_node_len != node_len_) {
    std::cerr << "[WARN] Cache node_len mismatch (stored=" << stored_node_len
              << " expected=" << node_len_ << "); skipping cache\n";
    cache_ids_.clear();
    return;
  }

  cache_count = std::min(cache_count, cache_nodes_num);
  cache_ids_.resize(cache_count);
  size_t cache_bytes = cache_count * node_len_;
  if (cache_bytes == 0) {
    std::cerr << "[WARN] Cache has 0 loadable nodes; skipping cache\n";
    cache_ids_.clear();
    return;
  }

#ifdef LASER_USE_NUMA
  if (try_load_cache_with_numa(cache_vectors_input, cache_bytes)) {
    return;
  }
#endif

  if (!load_cache_nodes_standard(cache_vectors_input, cache_bytes)) {
    std::cerr << "[WARN] Failed to read cache nodes from: " << cache_nodes_file << "\n";
    cache_ids_.clear();
    return;
  }
  rebuild_cache_lookup(cache_nodes_.data(), cache_count);
}

}  // namespace symqg
// NOLINTEND(performance-no-int-to-ptr,google-runtime-references,modernize-use-trailing-return-type)
