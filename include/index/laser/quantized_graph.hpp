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
#include "index/laser/space/l2.hpp"
#include "index/laser/thread_date.hpp"
#include "index/laser/transform/fht_rotator.hpp"
#include "index/laser/transform/pca_transform.hpp"
#include "index/laser/utils/concurrent_queue.hpp"
#include "index/laser/utils/io.hpp"
#include "index/laser/utils/memory.hpp"
#include "index/laser/utils/search_buffer.hpp"
#include "utils/aligned_array.hpp"

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
    const char* proc_bind = std::getenv("OMP_PROC_BIND");
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
        std::cerr << "[WARN] AIO quota may be insufficient. Required: " << required
                  << " slots (" << num_threads << " threads * " << events_per_thread
                  << " events), Available: " << available
                  << " (aio-nr=" << aio_nr << ", aio-max-nr=" << aio_max_nr << "). "
                  << "Fix: sudo sysctl -w fs.aio-max-nr=1048576\n";
    }
}

class QuantizedGraph {
    friend class QGBuilder;

   private:
    size_t num_points_ = 0;
    size_t degree_bound_ = 0;
    size_t dimension_ = 0;
    size_t residual_dimension_ = 0;
    size_t padded_dim_ = 0;
    PID entry_point_ = 0;

    data::Array<float, std::vector<size_t>,
                memory::AlignedAllocator<float, 1 << 22, true>> data_;
    QGScanner scanner_;
    FHTRotator rotator_;
    PCATransform pca_transform_;
    LinuxAlignedFileReader aligned_file_reader_;
    ConcurrentQueue<ThreadDate> thread_data_;
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
    alaya::fast::map<PID, char*> caches_;

    size_t nthreads_ = 1;

    size_t res_dim_offset_ = 0;
    size_t code_offset_ = 0;
    size_t factor_offset_ = 0;
    size_t neighbor_offset_ = 0;
    size_t row_offset_ = 0;

    void initialize();

    void disk_search_qg(
        const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results,
        ThreadDate& data
    );

    [[nodiscard]] auto get_page_offset(uint64_t node_id) const -> uint64_t {
        return kSectorLen + page_size_ * (node_id / node_per_page_);
    }

    [[nodiscard]] auto offset_to_node(uint64_t node_id) const -> uint64_t {
        return (node_id % node_per_page_) * node_len_;
    }

    [[nodiscard]] auto gen_index_path(const char* prefix) const -> std::string {
        return std::string(prefix) + "_R" + std::to_string(degree_bound_) +
               "_MD" + std::to_string(dimension_) + ".index";
    }

    auto scan_neighbors(
        const QGQuery& q_obj,
        const float* cur_data,
        float* appro_dist,
        buffer::SearchBuffer& search_pool,
        uint32_t cur_degree,
        TaggedVisitedSet& visited,
        LaserSearchContext& ctx
    ) const -> float;

   public:
    explicit QuantizedGraph(size_t num, size_t max_deg, size_t main_dim, size_t dim);
    ~QuantizedGraph();

    [[nodiscard]] auto num_vertices() const { return num_points_; }
    [[nodiscard]] auto dimension() const { return dimension_; }
    [[nodiscard]] auto residual_dimension() const { return residual_dimension_; }
    [[nodiscard]] auto degree_bound() const { return degree_bound_; }
    [[nodiscard]] auto entry_point() const { return entry_point_; }
    void set_ep(PID entry) { entry_point_ = entry; }

    void load_disk_index(const char* prefix, float search_dram_budget);
    void set_params(size_t ef_search, size_t num_threads, int beam_width);
    void load_medoids(const char* prefix);
    void load_cache(std::string& cache_ids_file, std::string& cache_nodes_file,
                    size_t online_cache_num);

    void search(const float* __restrict__ query, uint32_t knn,
                uint32_t* __restrict__ results);

    void batch_search(const float* __restrict__ query, uint32_t knn,
                      uint32_t* __restrict__ results, size_t num_queries);

    [[nodiscard]] auto cached_node_count() const -> size_t { return cache_ids_.size(); }
    [[nodiscard]] auto cache_size_bytes() const -> size_t { return cache_nodes_.size(); }

    void destroy_thread_data();
};

// ============================================================================
// Implementation
// ============================================================================

inline QuantizedGraph::QuantizedGraph(
    size_t num, size_t max_deg, size_t main_dim, size_t dim
)
    : num_points_(num)
    , degree_bound_(max_deg)
    , dimension_(main_dim)
    , residual_dimension_(dim - main_dim)
    , padded_dim_(1 << ceil_log2(main_dim))
    , scanner_(padded_dim_, degree_bound_)
    , rotator_(main_dim)
    , node_len_((32 * main_dim + 32 * (dim - main_dim) +
                 128 * max_deg + max_deg * padded_dim_) / 8) {
    node_per_page_ = std::max(static_cast<size_t>(1), kSectorLen / node_len_);
    page_size_ = (node_per_page_ * node_len_ + kSectorLen - 1) / kSectorLen * kSectorLen;

    if (main_dim != padded_dim_) {
        throw std::runtime_error("Laser: dimension must be a power of 2");
    }
    initialize();
}

inline QuantizedGraph::~QuantizedGraph() {
    destroy_thread_data();
}

inline void QuantizedGraph::initialize() {
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dimension_);

    res_dim_offset_ = dimension_;
    code_offset_ = dimension_ + residual_dimension_;
    factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;
    neighbor_offset_ = factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
    row_offset_ = neighbor_offset_ + degree_bound_;
}

inline void QuantizedGraph::set_params(
    size_t ef_search, size_t num_threads, int beam_width
) {
    nthreads_ = num_threads;
    max_beam_width_ = static_cast<size_t>(beam_width);
    ef_search_ = ef_search;

    destroy_thread_data();

    if (index_file_name_.empty()) {
        throw std::runtime_error("Laser: load index before calling set_params()");
    }

    check_omp_affinity();

    size_t aio_events = 2 * max_beam_width_;
    check_aio_capacity(nthreads_, aio_events);

    aligned_file_reader_.open(index_file_name_);

    size_t full_dim = dimension_ + residual_dimension_;

#pragma omp parallel for num_threads(static_cast<int>(nthreads_))
    for (size_t thread = 0; thread < nthreads_; thread++) {
#pragma omp critical
        {
            aligned_file_reader_.register_thread(aio_events);
            ThreadDate data;
            data.ctx_ = aligned_file_reader_.get_ctx();
            data.allocate(
                padded_dim_, degree_bound_, max_beam_width_,
                page_size_, ef_search_, full_dim, num_points_
            );
            thread_data_.push(data);
        }
    }
}

inline void QuantizedGraph::destroy_thread_data() {
    while (thread_data_.size() > 0) {
        ThreadDate data = thread_data_.pop();
        while (data.sector_scratch_ == nullptr) {
            thread_data_.wait_for_push_notify();
            data = thread_data_.pop();
        }
        data.deallocate();
    }
    aligned_file_reader_.deregister_all_threads();
    aligned_file_reader_.close();
}

inline void QuantizedGraph::search(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results
) {
    ThreadDate data = thread_data_.pop();
    while (data.sector_scratch_ == nullptr) {
        thread_data_.wait_for_push_notify();
        data = thread_data_.pop();
    }
    disk_search_qg(query, knn, results, data);
    thread_data_.push(data);
    thread_data_.push_notify_all();
}

inline void QuantizedGraph::batch_search(
    const float* __restrict__ query, uint32_t knn,
    uint32_t* __restrict__ results, size_t num_queries
) {
    // Thread-affine context: each thread acquires once, reuses across queries
#pragma omp parallel num_threads(static_cast<int>(nthreads_))
    {
        ThreadDate data = thread_data_.pop();
        while (data.sector_scratch_ == nullptr) {
            thread_data_.wait_for_push_notify();
            data = thread_data_.pop();
        }

        size_t full_dim = dimension_ + residual_dimension_;
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < num_queries; ++i) {
            disk_search_qg(
                query + i * full_dim, knn, results + i * knn, data
            );
        }

        thread_data_.push(data);
        thread_data_.push_notify_all();
    }
}

inline void QuantizedGraph::disk_search_qg(
    const float* __restrict__ query, uint32_t knn, uint32_t* __restrict__ results,
    ThreadDate& data
) {
    auto& ctx = data.search_ctx_;
    ctx.reset();
    data.search_pool_.clear();

    // PCA Transform
    const float* transformed_query = query;
    if (pca_transform_.is_loaded()) {
        pca_transform_.transform(query, data.pca_query_scratch_);
        transformed_query = data.pca_query_scratch_;
    }

    // Query preparation (uses ctx buffers — zero alloc)
    QGQuery q_obj(transformed_query, padded_dim_);
    q_obj.query_prepare(rotator_, scanner_, ctx);

    const float* residual_query = transformed_query + dimension_;
    float sqr_qr = (residual_dimension_ > 0)
        ? space::l2_sqr_single(residual_query, residual_dimension_)
        : 0.0F;
    q_obj.set_sqr_qr(sqr_qr);

    // Initialize search pool with entry points
    if (!medoids_.empty()) {
        PID best_medoid = 0;
        float best_dist = FLT_MAX;
        size_t full_dim = dimension_ + residual_dimension_;
        for (size_t cur_m = 0; cur_m < medoids_.size(); cur_m++) {
            float cur_dist = space::l2_sqr(
                transformed_query,
                medoids_vector_.data() + full_dim * cur_m,
                dimension_
            );
            if (cur_dist < best_dist) {
                best_medoid = medoids_[cur_m];
                best_dist = cur_dist;
            }
        }
        data.search_pool_.insert(best_medoid, FLT_MAX);
    }
    data.search_pool_.insert(entry_point_, FLT_MAX);

    auto& res_pool = ctx.result_buffer();
    res_pool.reset(knn);

    auto& ongoing = ctx.ongoing_table();
    auto& prepared = ctx.prepared_ring();
    auto& visited = ctx.visited_set();
    auto& cache_nhoods = ctx.cache_nhoods();
    auto& free_slots = ctx.free_slot_stack();

    // Initialize free slot stack with sector scratch buffers
    for (size_t i = 0; i < 2 * max_beam_width_; i++) {
        free_slots.push(data.sector_scratch_ + i * page_size_);
    }

    size_t frontier_req_count = 0;
    size_t cur_beam_size = 1;

    // Node processing lambda
    auto process_node = [&](PID cur_node, float* cur_data) {
        float sqr_y = scan_neighbors(
            q_obj, cur_data, ctx.appro_dist(),
            data.search_pool_, degree_bound_, visited, ctx
        );
        if (residual_dimension_ > 0) {
            float* residual_data = cur_data + dimension_;
            sqr_y += space::l2_sqr(
                reinterpret_cast<const float*>(residual_data),
                residual_query, residual_dimension_
            );
        }
        res_pool.insert(cur_node, sqr_y);
    };

    // I/O completion handler: non-blocking probe first, then blocking fallback.
    // Non-blocking catches already-completed events without syscall overhead.
    // Blocking fallback avoids infinite spin when events are still in-flight.
    // Prefetch node data into L2 when I/O completes, so it's warm by process time
    size_t prefetch_lines = std::min(node_len_ / 64, static_cast<size_t>(20));

    auto collect_events = [&](io_event* evts, int ret) {
        for (int i = 0; i < ret; i++) {
            auto id = static_cast<PID>(
                reinterpret_cast<uintptr_t>(evts[i].data)
            );
            char* buf = ongoing.find(id);
            if (buf != nullptr) {
                // Prefetch the node data into L2 cache while it's being queued
                const char* node_ptr = buf + offset_to_node(id);
                memory::mem_prefetch_l2(node_ptr, prefetch_lines);
                prepared.push_back({id, buf});
                ongoing.erase(id);
            }
        }
    };

    auto wait_for_nodes = [&]() {
        io_event* evts = ctx.io_events();
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
    AlignedRead* frontier_reqs = ctx.frontier_reqs();

    // Main search loop
    while (data.search_pool_.has_next()) {
        frontier_req_count = 0;
        cache_nhoods.clear();
        size_t n_ops = 0;

        cur_beam_size = std::min(
            max_beam_width_,
            static_cast<size_t>(std::ceil(2.0F * static_cast<float>(cur_beam_size)))
        );

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
                char* slot = free_slots.pop();
                ongoing.insert(cur_node, slot);
                frontier_reqs[frontier_req_count] = AlignedRead(
                    get_page_offset(cur_node), page_size_, cur_node, slot
                );
                ++frontier_req_count;
            }
        }

        // Submit async I/O (zero-copy: pre-allocated iocb buffers)
        if (frontier_req_count > 0) {
            n_ops = aligned_file_reader_.submit_reqs(
                frontier_reqs, frontier_req_count, data.ctx_,
                ctx.iocb_buf(), ctx.iocb_ptrs_buf()
            );
        }

        // Process cached nodes with look-ahead prefetch (no double lookup)
        for (size_t ci = 0; ci < cache_nhoods.size(); ++ci) {
            if (ci + 1 < cache_nhoods.size()) {
                memory::mem_prefetch_l1(
                    reinterpret_cast<const char*>(cache_nhoods[ci + 1].second),
                    prefetch_lines
                );
            }
            process_node(
                cache_nhoods[ci].first,
                reinterpret_cast<float*>(cache_nhoods[ci].second)
            );
        }

        // Pipelined processing
        auto remain_num = static_cast<size_t>(0.5 * n_ops);
        size_t need_process_num = n_ops + previous_remain_num - remain_num;
        previous_remain_num = remain_num;

        while (need_process_num > 0) {
            if (!prepared.empty()) {
                auto node = prepared.pop_front();
                // L1 prefetch the next node while processing current
                if (!prepared.empty()) {
                    auto& next = prepared.front();
                    memory::mem_prefetch_l1(
                        next.second + offset_to_node(next.first), prefetch_lines
                    );
                }
                process_node(
                    node.first,
                    reinterpret_cast<float*>(node.second + offset_to_node(node.first))
                );
                --need_process_num;
                free_slots.push(node.second);
            } else {
                wait_for_nodes();
            }
        }
    }

    // Drain remaining
    while (previous_remain_num > 0) {
        if (!prepared.empty()) {
            auto node = prepared.pop_front();
            if (!prepared.empty()) {
                auto& next = prepared.front();
                memory::mem_prefetch_l1(
                    next.second + offset_to_node(next.first), prefetch_lines
                );
            }
            process_node(
                node.first,
                reinterpret_cast<float*>(node.second + offset_to_node(node.first))
            );
            --previous_remain_num;
            free_slots.push(node.second);
        } else {
            wait_for_nodes();
        }
    }

    res_pool.copy_results(results);
}

inline auto QuantizedGraph::scan_neighbors(
    const QGQuery& q_obj,
    const float* cur_data,
    float* appro_dist,
    buffer::SearchBuffer& search_pool,
    uint32_t cur_degree,
    TaggedVisitedSet& visited,
    LaserSearchContext& ctx
) const -> float {
    float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);

    const auto* packed_code = reinterpret_cast<const uint8_t*>(&cur_data[code_offset_]);
    const auto* factor = &cur_data[factor_offset_];
    scanner_.scan_neighbors(
        appro_dist,
        ctx.lut(),
        sqr_y,
        q_obj.lower_val(),
        q_obj.width(),
        q_obj.sqr_qr(),
        q_obj.sumq(),
        packed_code,
        factor,
        ctx
    );

    const PID* ptr_nb = reinterpret_cast<const PID*>(&cur_data[neighbor_offset_]);
    for (uint32_t i = 0; i < cur_degree; ++i) {
        PID cur_neighbor = ptr_nb[i];
        float tmp_dist = appro_dist[i];
        if (search_pool.is_full(tmp_dist) || visited.get(cur_neighbor)) {
            continue;
        }
        search_pool.insert(cur_neighbor, tmp_dist);
    }

    return sqr_y;
}

inline void QuantizedGraph::load_disk_index(
    const char* prefix, float search_dram_budget
) {
    index_file_name_ = gen_index_path(prefix);
    if (!std::filesystem::exists(index_file_name_)) {
        throw std::runtime_error(
            "Index file not found: " + index_file_name_
        );
    }
    std::ifstream input(index_file_name_, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Failed to open index file: " + index_file_name_
        );
    }

    std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
    input.read(reinterpret_cast<char*>(metas.data()), kSectorLen);

    // Metadata validation
    if (metas[0] != num_points_ || metas[1] != dimension_ ||
        metas[3] != node_len_ || metas[4] != node_per_page_) {
        throw std::runtime_error(
            "Index metadata mismatch. Expected: num_points=" +
            std::to_string(num_points_) + " dim=" + std::to_string(dimension_) +
            " node_len=" + std::to_string(node_len_) +
            " node_per_page=" + std::to_string(node_per_page_) +
            ". Got: " + std::to_string(metas[0]) + "/" +
            std::to_string(metas[1]) + "/" + std::to_string(metas[3]) +
            "/" + std::to_string(metas[4])
        );
    }
    auto expected_size = std::filesystem::file_size(index_file_name_);
    if (metas[8] != expected_size) {
        throw std::runtime_error(
            "Index file size mismatch: header says " +
            std::to_string(metas[8]) + " but file is " +
            std::to_string(expected_size) + " bytes"
        );
    }

    entry_point_ = static_cast<PID>(metas[2]);

    // Load rotator
    std::string rotator_path = index_file_name_ + "_rotator";
    std::ifstream rotator_input(rotator_path, std::ios::binary);
    if (!rotator_input.is_open()) {
        throw std::runtime_error("Missing rotator file: " + rotator_path);
    }
    rotator_.load(rotator_input);

    // Initialize workspace
    aligned_file_reader_.open(index_file_name_);
    size_t aio_events = 2 * max_beam_width_;
    size_t full_dim = dimension_ + residual_dimension_;

#pragma omp parallel for num_threads(static_cast<int>(nthreads_))
    for (size_t thread = 0; thread < nthreads_; thread++) {
#pragma omp critical
        {
            aligned_file_reader_.register_thread(aio_events);
            ThreadDate data;
            data.ctx_ = aligned_file_reader_.get_ctx();
            data.allocate(
                padded_dim_, degree_bound_, max_beam_width_,
                page_size_, ef_search_, full_dim, num_points_
            );
            thread_data_.push(data);
        }
    }

    // Load medoids
    load_medoids(prefix);

    // Load PCA
    std::string pca_path = std::string(prefix) + "_pca.bin";
    if (std::filesystem::exists(pca_path)) {
        pca_transform_.load(pca_path);
    } else {
        std::cerr << "[WARN] PCA file not found: " << pca_path << '\n';
    }

    // Load cache
    auto cache_space = static_cast<size_t>(
        search_dram_budget * 1000 * 1000 * 1000 * 0.8
    );
    size_t online_cache_num = std::min(
        cache_space / node_len_,
        static_cast<size_t>(kCacheRatio * num_points_)
    );
    std::string cache_ids_file = index_file_name_ + "_cache_ids";
    std::string cache_nodes_file = index_file_name_ + "_cache_nodes";
    if (std::filesystem::exists(cache_ids_file) &&
        std::filesystem::exists(cache_nodes_file)) {
        load_cache(cache_ids_file, cache_nodes_file, online_cache_num);
    }
}

inline void QuantizedGraph::load_medoids(const char* prefix) {
    std::string medoids_indices_file = std::string(prefix) + "_medoids_indices";
    std::string medoids_file = std::string(prefix) + "_medoids";

    if (!std::filesystem::exists(medoids_file) ||
        !std::filesystem::exists(medoids_indices_file)) {
        return;
    }

    std::ifstream medoid_input(medoids_indices_file, std::ios::binary);
    if (!medoid_input.is_open()) {
        return;
    }
    int medoid_num = 0;
    int tmp = 0;
    medoid_input.read(reinterpret_cast<char*>(&medoid_num), sizeof(int));
    medoid_input.read(reinterpret_cast<char*>(&tmp), sizeof(int));
    medoids_.resize(
        static_cast<size_t>(medoid_num) * static_cast<size_t>(tmp)
    );
    medoid_input.read(
        reinterpret_cast<char*>(medoids_.data()),
        static_cast<std::streamsize>(sizeof(int) * medoid_num * tmp)
    );

    std::ifstream medoid_vec_input(medoids_file, std::ios::binary);
    if (!medoid_vec_input.is_open()) {
        return;
    }
    int dim = 0;
    medoid_vec_input.read(reinterpret_cast<char*>(&medoid_num), sizeof(int));
    medoid_vec_input.read(reinterpret_cast<char*>(&dim), sizeof(int));

    size_t full_dim = dimension_ + residual_dimension_;
    medoids_vector_.resize(
        static_cast<size_t>(medoid_num) * full_dim
    );
    medoid_vec_input.read(
        reinterpret_cast<char*>(medoids_vector_.data()),
        static_cast<std::streamsize>(sizeof(float) * medoid_num * full_dim)
    );
}

inline void QuantizedGraph::load_cache(
    std::string& cache_ids_file, std::string& cache_nodes_file,
    size_t online_cache_num
) {
    std::ifstream cache_ids_input(cache_ids_file, std::ios::binary);
    std::ifstream cache_vectors_input(cache_nodes_file, std::ios::binary);
    if (!cache_ids_input.is_open() || !cache_vectors_input.is_open()) {
        return;
    }

    size_t cache_ids_num = 0;
    size_t cache_nodes_num = 0;
    size_t tmp_node_len = 0;
    cache_ids_input.read(reinterpret_cast<char*>(&cache_ids_num), sizeof(size_t));
    online_cache_num = std::min(online_cache_num, cache_ids_num);

    cache_ids_.resize(online_cache_num);
    cache_ids_input.read(
        reinterpret_cast<char*>(cache_ids_.data()),
        static_cast<std::streamsize>(sizeof(PID) * online_cache_num)
    );

    cache_vectors_input.read(
        reinterpret_cast<char*>(&cache_nodes_num), sizeof(size_t)
    );
    cache_vectors_input.read(
        reinterpret_cast<char*>(&tmp_node_len), sizeof(size_t)
    );

    size_t cache_bytes = online_cache_num * node_len_;

#ifdef LASER_USE_NUMA
    // NUMA-interleaved allocation: distribute cache pages across NUMA nodes
    void* numa_ptr = mmap(
        nullptr, cache_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0
    );
    if (numa_ptr != MAP_FAILED) {
        unsigned long nodemask = ~0UL;  // all nodes
        long ret = mbind(
            numa_ptr, cache_bytes, MPOL_INTERLEAVE,
            &nodemask, sizeof(nodemask) * 8, 0
        );
        if (ret == 0) {
            cache_vectors_input.read(
                reinterpret_cast<char*>(numa_ptr),
                static_cast<std::streamsize>(cache_bytes)
            );
            // Use a sentinel-sized vector and point caches_ into mmap'd memory
            cache_nodes_.resize(0);
            for (size_t i = 0; i < cache_ids_.size(); i++) {
                caches_[cache_ids_[i]] =
                    reinterpret_cast<char*>(numa_ptr) + i * node_len_;
            }
            return;
        }
        // mbind failed, fall back to standard allocation
        munmap(numa_ptr, cache_bytes);
    }
#endif
    // Standard fallback
    cache_nodes_.resize(cache_bytes);
    cache_vectors_input.read(
        reinterpret_cast<char*>(cache_nodes_.data()),
        static_cast<std::streamsize>(cache_bytes)
    );

    for (size_t i = 0; i < cache_ids_.size(); i++) {
        PID cur_id = cache_ids_[i];
        caches_[cur_id] = cache_nodes_.data() + i * node_len_;
    }
}

}  // namespace symqg
