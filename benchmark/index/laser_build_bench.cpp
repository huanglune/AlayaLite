/**
 * @file laser_build_bench.cpp
 * @brief End-to-end LASER build + search benchmark with memory budget control.
 *
 * Runs PCA → Medoid → QG using io_uring disk-based access, then benchmarks
 * search QPS/recall/latency. Peak RSS stays within the max_memory budget.
 *
 * Usage:
 *   ./laser_build_bench <base_fvecs> <query_fvecs> <gt_ivecs>
 *       <vamana_index> <output_dir>
 *       [max_memory_mib=1024] [max_degree=64] [main_dim=256]
 *       [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]
 */

// NOLINTBEGIN

#include <fcntl.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Core>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "index/laser/laser_common.hpp"
#include "index/laser/laser_index.hpp"
#include "index/laser/quantized_graph.hpp"
#include "index/laser/space/l2.hpp"
#include "index/laser/transform/pca_transform.hpp"
#include "index/laser/utils/tools.hpp"
#include "storage/io/io_uring_engine.hpp"
#include "utils/kmeans.hpp"
#include "utils/timer.hpp"

// ============================================================================
// RSS measurement
// ============================================================================

static auto get_rss_mib() -> double {
    std::ifstream ifs("/proc/self/status");
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.starts_with("VmRSS:")) {
            size_t pos = line.find_first_of("0123456789");
            if (pos != std::string::npos) return std::stod(line.substr(pos)) / 1024.0;
        }
    }
    return 0.0;
}

static void print_rss(const char* label) {
    std::cout << "  [RSS] " << std::left << std::setw(35) << label << std::fixed
              << std::setprecision(1) << get_rss_mib() << " MiB\n" << std::flush;
}

// ============================================================================
// Disk-based file readers (io_uring)
// ============================================================================

class FvecsFileReader {
 public:
    ~FvecsFileReader() { if (fd_ >= 0) ::close(fd_); }

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open: " + path);
        int32_t dim = 0;
        if (::pread(fd_, &dim, sizeof(dim), 0) != sizeof(dim) || dim <= 0)
            throw std::runtime_error("Invalid fvecs: " + path);
        dim_ = static_cast<uint32_t>(dim);
        vec_stride_ = sizeof(int32_t) + static_cast<size_t>(dim_) * sizeof(float);
        vec_bytes_ = static_cast<size_t>(dim_) * sizeof(float);
        struct stat st {};
        fstat(fd_, &st);
        num_vectors_ = static_cast<uint32_t>(static_cast<size_t>(st.st_size) / vec_stride_);
    }

    void read_by_ids(const uint32_t* ids, uint32_t count, float* out) {
        batch_read(count, [&](uint32_t i) {
            return static_cast<uint64_t>(ids[i]) * vec_stride_ + sizeof(int32_t);
        }, out);
    }

    void read_sequential(uint32_t start, uint32_t count, float* out) {
        batch_read(count, [&](uint32_t i) {
            return static_cast<uint64_t>(start + i) * vec_stride_ + sizeof(int32_t);
        }, out);
    }

    [[nodiscard]] auto dim() const -> uint32_t { return dim_; }
    [[nodiscard]] auto num_vectors() const -> uint32_t { return num_vectors_; }

 private:
    int fd_{-1};
    uint32_t dim_{0}, num_vectors_{0};
    size_t vec_stride_{0}, vec_bytes_{0};
    alaya::IOUringEngine engine_{256};

    template <typename OffsetFn>
    void batch_read(uint32_t count, OffsetFn offset_fn, float* out) {
        constexpr uint32_t kBatch = 256;
        std::vector<alaya::IORequest> reqs(kBatch);
        for (uint32_t s = 0; s < count; s += kBatch) {
            uint32_t n = std::min(kBatch, count - s);
            for (uint32_t i = 0; i < n; ++i)
                reqs[i] = alaya::IORequest(out + static_cast<size_t>(s + i) * dim_,
                                           vec_bytes_, offset_fn(s + i));
            auto span = std::span<alaya::IORequest>(reqs.data(), n);
            engine_.wait(engine_.submit_reads(fd_, span), -1);
        }
    }
};

class FbinFileReader {
 public:
    ~FbinFileReader() { if (fd_ >= 0) ::close(fd_); }

    void open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Cannot open: " + path);
        int32_t hdr[2]{};
        if (::pread(fd_, hdr, sizeof(hdr), 0) != sizeof(hdr))
            throw std::runtime_error("Invalid fbin: " + path);
        num_vectors_ = static_cast<uint32_t>(hdr[0]);
        dim_ = static_cast<uint32_t>(hdr[1]);
        vec_bytes_ = static_cast<size_t>(dim_) * sizeof(float);
    }

    void read_by_ids(const uint32_t* ids, uint32_t count, float* out) {
        batch_read(count, [&](uint32_t i) {
            return 8ULL + static_cast<uint64_t>(ids[i]) * vec_bytes_;
        }, out);
    }

    void read_sequential(uint32_t start, uint32_t count, float* out) {
        batch_read(count, [&](uint32_t i) {
            return 8ULL + static_cast<uint64_t>(start + i) * vec_bytes_;
        }, out);
    }

    [[nodiscard]] auto dim() const -> uint32_t { return dim_; }
    [[nodiscard]] auto num_vectors() const -> uint32_t { return num_vectors_; }

 private:
    int fd_{-1};
    uint32_t dim_{0}, num_vectors_{0};
    size_t vec_bytes_{0};
    alaya::IOUringEngine engine_{256};

    template <typename OffsetFn>
    void batch_read(uint32_t count, OffsetFn offset_fn, float* out) {
        constexpr uint32_t kBatch = 256;
        std::vector<alaya::IORequest> reqs(kBatch);
        for (uint32_t s = 0; s < count; s += kBatch) {
            uint32_t n = std::min(kBatch, count - s);
            for (uint32_t i = 0; i < n; ++i)
                reqs[i] = alaya::IORequest(out + static_cast<size_t>(s + i) * dim_,
                                           vec_bytes_, offset_fn(s + i));
            auto span = std::span<alaya::IORequest>(reqs.data(), n);
            engine_.wait(engine_.submit_reads(fd_, span), -1);
        }
    }
};

// ============================================================================
// VamanaGraphReader — offset-indexed chunked access
// ============================================================================

class VamanaGraphReader {
 public:
    void open(const std::string& path) {
        path_ = path;
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Cannot open vamana: " + path);

        in.read(reinterpret_cast<char*>(&file_size_), sizeof(size_t));
        in.read(reinterpret_cast<char*>(&max_degree_), sizeof(uint32_t));
        in.read(reinterpret_cast<char*>(&entry_point_), sizeof(uint32_t));
        size_t frozen = 0;
        in.read(reinterpret_cast<char*>(&frozen), sizeof(size_t));

        constexpr size_t kHdrSz = sizeof(size_t) + 2 * sizeof(uint32_t) + sizeof(size_t);
        size_t pos = kHdrSz;
        while (pos < file_size_) {
            offsets_.push_back(pos);
            uint32_t deg = 0;
            in.read(reinterpret_cast<char*>(&deg), sizeof(uint32_t));
            degrees_.push_back(deg);
            in.seekg(static_cast<std::streamoff>(deg * sizeof(uint32_t)), std::ios::cur);
            pos += sizeof(uint32_t) + static_cast<size_t>(deg) * sizeof(uint32_t);
        }
    }

    void read_chunk(uint32_t start, uint32_t count,
                    std::vector<std::vector<uint32_t>>& out) const {
        out.resize(count);
        std::ifstream in(path_, std::ios::binary);
        for (uint32_t i = 0; i < count; ++i) {
            in.seekg(static_cast<std::streamoff>(offsets_[start + i] + sizeof(uint32_t)));
            out[i].resize(degrees_[start + i]);
            in.read(reinterpret_cast<char*>(out[i].data()),
                    static_cast<std::streamsize>(degrees_[start + i] * sizeof(uint32_t)));
        }
    }

    [[nodiscard]] auto compute_in_degrees() const -> std::vector<uint32_t> {
        auto n = static_cast<uint32_t>(offsets_.size());
        std::vector<uint32_t> in_deg(n, 0);
        std::ifstream in(path_, std::ios::binary);
        for (uint32_t nid = 0; nid < n; ++nid) {
            in.seekg(static_cast<std::streamoff>(offsets_[nid] + sizeof(uint32_t)));
            std::vector<uint32_t> nbrs(degrees_[nid]);
            in.read(reinterpret_cast<char*>(nbrs.data()),
                    static_cast<std::streamsize>(degrees_[nid] * sizeof(uint32_t)));
            for (auto id : nbrs) in_deg[id]++;
        }
        return in_deg;
    }

    [[nodiscard]] auto num_nodes() const -> uint32_t {
        return static_cast<uint32_t>(offsets_.size());
    }
    [[nodiscard]] auto max_degree() const -> uint32_t { return max_degree_; }
    [[nodiscard]] auto entry_point() const -> uint32_t { return entry_point_; }

 private:
    std::string path_;
    size_t file_size_{0};
    uint32_t max_degree_{0}, entry_point_{0};
    std::vector<size_t> offsets_;
    std::vector<uint32_t> degrees_;
};

// ============================================================================
// Build Phase 1: Incremental PCA (io_uring, two-pass over disk)
//
// Pass 1: incremental mean (batch-by-batch from disk)
// Pass 2: incremental covariance (batch-by-batch, centralized with mean)
// Then eigen decomposition on the covariance matrix.
// Peak memory = max(batch_buffer, covariance_matrix) — no full sample load.
// ============================================================================

static void build_pca(FvecsFileReader& reader,
                      const std::string& pca_bin_path,
                      const std::string& pca_base_path,
                      uint32_t max_memory_mib,
                      uint32_t num_threads) {
    uint32_t num_base = reader.num_vectors();
    uint32_t dim = reader.dim();
    size_t vec_bytes = static_cast<size_t>(dim) * sizeof(float);

    // Sample size fixed by ratio/cap — not limited by max_memory
    uint32_t sample_size = std::min({
        static_cast<uint32_t>(static_cast<float>(num_base) * 0.25F),
        500000U, num_base});
    sample_size = std::max(1U, sample_size);

    // Batch size limited by max_memory (90% budget for batch buffer)
    size_t budget = static_cast<size_t>(max_memory_mib) * 1024UL * 1024UL * 9UL / 10UL;
    uint32_t batch_size = std::max(1000U,
        std::min(sample_size, static_cast<uint32_t>(budget / vec_bytes)));

    std::cout << "  sample: " << sample_size << ", batch: " << batch_size
              << " (" << ((sample_size + batch_size - 1) / batch_size)
              << " passes)" << std::endl;

    // Generate sorted sample IDs (only IDs in memory, not vectors)
    std::vector<uint32_t> ids(num_base);
    std::iota(ids.begin(), ids.end(), 0U);
    std::mt19937 rng(42);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sample_size);
    std::sort(ids.begin(), ids.end());

    int prev_eigen = Eigen::nbThreads();
    Eigen::setNbThreads(static_cast<int>(num_threads));

    symqg::PCATransform pca(dim, dim);
    auto edim = static_cast<Eigen::Index>(dim);
    std::vector<float> batch(static_cast<size_t>(batch_size) * dim);

    // --- Pass 1: incremental mean ---
    alaya::Timer t;
    Eigen::Map<Eigen::VectorXf> mean(pca.mean_data(), edim);
    mean.setZero();

    for (uint32_t s = 0; s < sample_size; s += batch_size) {
        uint32_t cnt = std::min(batch_size, sample_size - s);
        reader.read_by_ids(ids.data() + s, cnt, batch.data());
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            block(batch.data(), static_cast<Eigen::Index>(cnt), edim);
        mean += block.colwise().sum();
    }
    mean /= static_cast<float>(sample_size);
    std::cout << "  mean: " << std::fixed << std::setprecision(1) << t.elapsed_s()
              << " s" << std::endl;
    t.reset();

    // --- Pass 2: incremental covariance ---
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> cov(edim, edim);
    cov.setZero();

    for (uint32_t s = 0; s < sample_size; s += batch_size) {
        uint32_t cnt = std::min(batch_size, sample_size - s);
        reader.read_by_ids(ids.data() + s, cnt, batch.data());
        auto bcnt = static_cast<Eigen::Index>(cnt);

        // Centralize in-place
        Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            block(batch.data(), bcnt, edim);
        block.rowwise() -= mean.transpose();

        // Accumulate cov += block^T * block
        cov.noalias() += block.transpose() * block;
    }
    cov /= static_cast<float>(sample_size - 1);
    std::cout << "  covariance: " << t.elapsed_s() << " s" << std::endl;
    t.reset();

    ids.clear();
    ids.shrink_to_fit();
    batch.clear();
    batch.shrink_to_fit();

    // --- Eigen decomposition ---
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(cov);
    cov.resize(0, 0);  // free covariance memory

    const auto& evecs = solver.eigenvectors();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        pca_mat(pca.pca_matrix_data(), edim, edim);
    // Eigenvalues are ascending — reverse to descending order
    for (Eigen::Index i = 0; i < edim; ++i)
        for (Eigen::Index j = 0; j < edim; ++j)
            pca_mat(i, j) = evecs(j, edim - 1 - i);

    pca.mark_loaded();
    std::cout << "  eigen: " << t.elapsed_s() << " s" << std::endl;
    pca.save(pca_bin_path);
    print_rss("after PCA train");
    t.reset();

    // --- Batch transform → fbin (streaming, reuses batch buffer) ---
    std::ofstream out(pca_base_path, std::ios::binary | std::ios::trunc);
    auto np = static_cast<int32_t>(num_base);
    auto dm = static_cast<int32_t>(dim);
    out.write(reinterpret_cast<const char*>(&np), sizeof(np));
    out.write(reinterpret_cast<const char*>(&dm), sizeof(dm));

    constexpr uint32_t kXformBatch = 10000;
    std::vector<float> buf_in(static_cast<size_t>(kXformBatch) * dim);
    std::vector<float> buf_out(static_cast<size_t>(kXformBatch) * dim);
    for (uint32_t s = 0; s < num_base; s += kXformBatch) {
        uint32_t cnt = std::min(kXformBatch, num_base - s);
        reader.read_sequential(s, cnt, buf_in.data());
        pca.transform_batch(buf_in.data(), buf_out.data(), cnt);
        out.write(reinterpret_cast<const char*>(buf_out.data()),
                  static_cast<std::streamsize>(static_cast<size_t>(cnt) * dim * sizeof(float)));
    }
    std::cout << "  transform: " << t.elapsed_s() << " s" << std::endl;
    Eigen::setNbThreads(prev_eigen);
    print_rss("after PCA");
}

// ============================================================================
// Build Phase 2: Medoid (io_uring + blocked NN)
// ============================================================================

static void build_medoids(FbinFileReader& reader,
                          const std::filesystem::path& prefix,
                          uint32_t num_medoids,
                          uint32_t max_memory_mib,
                          uint32_t num_threads) {
    uint32_t num_pts = reader.num_vectors();
    uint32_t dim = reader.dim();
    size_t vec_bytes = static_cast<size_t>(dim) * sizeof(float);
    size_t budget = static_cast<size_t>(max_memory_mib) * 1024UL * 1024UL * 9UL / 10UL;
    uint32_t max_vecs = static_cast<uint32_t>(budget / vec_bytes);

    // Sample size is fixed by ratio/cap — NOT limited by max_memory.
    // Sample buffer is freed before NN scan starts, so they don't overlap.
    uint32_t sample_size = std::min({
        static_cast<uint32_t>(static_cast<float>(num_pts) * 0.1F),
        500000U, num_pts});
    sample_size = std::max(sample_size, num_medoids);
    // Only scan_block is bounded by max_memory
    uint32_t scan_block = std::max(1000U, std::min(max_vecs, num_pts));
    std::cout << "  sample: " << sample_size << ", scan_block: " << scan_block << std::endl;

    // Sample
    alaya::Timer t;
    std::vector<uint32_t> ids(num_pts);
    std::iota(ids.begin(), ids.end(), 0U);
    std::mt19937 rng(42);
    std::shuffle(ids.begin(), ids.end(), rng);
    ids.resize(sample_size);
    std::sort(ids.begin(), ids.end());
    std::vector<float> sample(static_cast<size_t>(sample_size) * dim);
    reader.read_by_ids(ids.data(), sample_size, sample.data());
    ids.clear();
    ids.shrink_to_fit();
    std::cout << "  sample read: " << std::fixed << std::setprecision(1)
              << t.elapsed_s() << " s" << std::endl;
    t.reset();

    // KMeans
    alaya::KMeans<float> km({.num_clusters_ = num_medoids, .max_iter_ = 20,
                              .num_trials_ = 3, .num_threads_ = num_threads});
    auto cl = km.fit(sample.data(), sample_size, dim);
    std::cout << "  KMeans: " << t.elapsed_s() << " s" << std::endl;
    sample.clear();
    sample.shrink_to_fit();
    t.reset();

    // Blocked NN scan
    std::vector<uint32_t> best_ids(num_medoids, 0);
    std::vector<float> best_dists(num_medoids, std::numeric_limits<float>::max());
    std::vector<float> block(static_cast<size_t>(scan_block) * dim);
    int nt = static_cast<int>(num_threads);

    for (uint32_t s = 0; s < num_pts; s += scan_block) {
        uint32_t cnt = std::min(scan_block, num_pts - s);
        reader.read_sequential(s, cnt, block.data());
#pragma omp parallel num_threads(nt)
        {
            std::vector<float> ld(num_medoids, std::numeric_limits<float>::max());
            std::vector<uint32_t> li(num_medoids, 0);
#pragma omp for schedule(static)
            for (uint32_t i = 0; i < cnt; ++i) {
                const float* pt = block.data() + static_cast<size_t>(i) * dim;
                for (uint32_t c = 0; c < num_medoids; ++c) {
                    float d = alaya::KMeans<float>::compute_l2_sqr(
                        cl.centroids_.data() + static_cast<size_t>(c) * dim, pt, dim);
                    if (d < ld[c]) { ld[c] = d; li[c] = s + i; }
                }
            }
#pragma omp critical
            for (uint32_t c = 0; c < num_medoids; ++c)
                if (ld[c] < best_dists[c]) { best_dists[c] = ld[c]; best_ids[c] = li[c]; }
        }
    }
    std::cout << "  NN scan: " << t.elapsed_s() << " s" << std::endl;

    // Fetch medoid vectors and write outputs
    std::vector<std::pair<uint32_t, uint32_t>> id_map(num_medoids);
    for (uint32_t c = 0; c < num_medoids; ++c) id_map[c] = {best_ids[c], c};
    std::sort(id_map.begin(), id_map.end());
    std::vector<uint32_t> sorted_ids(num_medoids);
    for (uint32_t i = 0; i < num_medoids; ++i) sorted_ids[i] = id_map[i].first;
    std::vector<float> mvecs(static_cast<size_t>(num_medoids) * dim);
    reader.read_by_ids(sorted_ids.data(), num_medoids, mvecs.data());

    // Reorder to centroid order
    std::vector<float> mvecs_ordered(static_cast<size_t>(num_medoids) * dim);
    for (uint32_t i = 0; i < num_medoids; ++i)
        std::copy_n(mvecs.data() + static_cast<size_t>(i) * dim, dim,
                    mvecs_ordered.data() + static_cast<size_t>(id_map[i].second) * dim);

    auto write_bin = [](const std::string& path, const void* data, size_t bytes,
                        int32_t rows, int32_t cols) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    };
    write_bin(prefix.string() + "_medoids_indices", best_ids.data(),
              num_medoids * sizeof(int32_t), static_cast<int32_t>(num_medoids), 1);
    write_bin(prefix.string() + "_medoids", mvecs_ordered.data(),
              static_cast<size_t>(num_medoids) * dim * sizeof(float),
              static_cast<int32_t>(num_medoids), static_cast<int32_t>(dim));
    print_rss("after Medoid");
}

// ============================================================================
// Build Phase 3: QG (chunked graph, friend of QuantizedGraph)
// ============================================================================

class MemLimitQGBuilder {
 public:
    static void build(const std::string& pca_base_path,
                      VamanaGraphReader& vamana,
                      const std::filesystem::path& prefix,
                      uint32_t num_points, uint32_t full_dim,
                      uint32_t max_degree, uint32_t main_dim,
                      uint32_t max_memory_mib, uint32_t num_threads) {
        symqg::QuantizedGraph g(num_points, max_degree, main_dim, full_dim);
        auto idx_path = prefix.string() + "_R" + std::to_string(max_degree) +
                        "_MD" + std::to_string(main_dim) + ".index";
        auto data_path = prefix.string() + "_pca_base.fbin";
        if (!std::filesystem::exists(data_path))
            std::filesystem::create_symlink(std::filesystem::absolute(pca_base_path), data_path);

        // In-degree + cache order
        alaya::Timer t;
        g.set_ep(vamana.entry_point());
        auto in_deg = vamana.compute_in_degrees();
        g.cache_ids_.resize(num_points);
        std::iota(g.cache_ids_.begin(), g.cache_ids_.end(), 0);
        std::sort(g.cache_ids_.begin(), g.cache_ids_.end(),
                  [&](symqg::PID a, symqg::PID b) { return in_deg[a] > in_deg[b]; });
        in_deg.clear();
        in_deg.shrink_to_fit();
        std::cout << "  in-degree: " << t.elapsed_s() << " s" << std::endl;
        t.reset();

        // Phase 1: sector-aligned temp file
        std::string tmp = idx_path + "_tmp.fbin";
        {
            std::ifstream vi(data_path, std::ios::binary);
            std::ofstream vo(tmp, std::ios::binary);
            int n = 0, d = 0;
            vi.read(reinterpret_cast<char*>(&n), sizeof(int));
            vi.read(reinterpret_cast<char*>(&d), sizeof(int));
            size_t psz = (d * sizeof(float) + symqg::kSectorLen - 1) /
                         symqg::kSectorLen * symqg::kSectorLen;
            std::vector<char> buf(psz);
            for (int i = 0; i < n; ++i) {
                std::memset(buf.data(), 0, buf.size());
                vi.read(buf.data(), d * sizeof(float));
                vo.write(buf.data(), static_cast<std::streamsize>(psz));
            }
        }
        std::cout << "  copy vectors: " << t.elapsed_s() << " s" << std::endl;
        t.reset();

        // Write index header
        size_t page_num = (num_points + g.node_per_page_ - 1) / g.node_per_page_;
        size_t total_sz = g.page_size_ * page_num + symqg::kSectorLen;
        {
            std::vector<uint64_t> m(symqg::kSectorLen / sizeof(uint64_t), 0);
            m[0] = num_points; m[1] = main_dim; m[2] = g.entry_point_;
            m[3] = g.node_len_; m[4] = g.node_per_page_; m[8] = total_sz;
            std::ofstream mo(idx_path, std::ios::binary);
            mo.write(reinterpret_cast<const char*>(m.data()), symqg::kSectorLen);
            mo.seekp(static_cast<std::streamoff>(total_sz - 1), std::ios::beg);
            char z = 0; mo.write(&z, 1);
        }

        // Compute chunk size
        constexpr size_t kEdgeBytes = 64 * sizeof(uint32_t) + 40;
        size_t budget = static_cast<size_t>(max_memory_mib) * 1024UL * 1024UL;
        uint32_t chunk = std::max(1000U,
            std::min(num_points, static_cast<uint32_t>(budget / kEdgeBytes)));
        std::cout << "  chunk: " << chunk << " ("
                  << ((num_points + chunk - 1) / chunk) << " passes)" << std::endl;

        // Phase 2: chunked parallel build
        int ofd = ::open(idx_path.c_str(), O_WRONLY);
        assert(ofd >= 0);
        LinuxAlignedFileReader vr;
        vr.open(tmp, false);

        size_t fpsz = (full_dim * sizeof(float) + symqg::kSectorLen - 1) /
                      symqg::kSectorLen * symqg::kSectorLen;
        size_t nbsz = max_degree * fpsz;

        std::vector<IOContext> ctx(num_threads);
#pragma omp parallel num_threads(static_cast<int>(num_threads))
        {
#pragma omp critical
            { vr.register_thread(); ctx[omp_get_thread_num()] = vr.get_ctx(); }
        }

        for (uint32_t cs = 0; cs < num_points; cs += chunk) {
            uint32_t cc = std::min(chunk, num_points - cs);
            std::vector<std::vector<uint32_t>> nbrs;
            vamana.read_chunk(cs, cc, nbrs);

#pragma omp parallel num_threads(static_cast<int>(num_threads))
            {
                auto tctx = ctx[omp_get_thread_num()];
                char* pg = reinterpret_cast<char*>(
                    symqg::memory::align_allocate<symqg::kSectorLen>(g.page_size_));
                char* nb = reinterpret_cast<char*>(
                    symqg::memory::align_allocate<symqg::kSectorLen>(nbsz));
                symqg::RowMatrix<float> cp(1, g.padded_dim_), cr(1, g.padded_dim_);
                std::vector<AlignedRead> rq;
                rq.reserve(max_degree + 1);

#pragma omp for schedule(dynamic)
                for (uint32_t li = 0; li < cc; ++li) {
                    uint32_t gi = cs + li;
                    auto& nbs = nbrs[li];
                    std::memset(pg, 0, g.page_size_);
                    size_t deg = nbs.size();
                    if (deg == 0) continue;

                    auto* np = reinterpret_cast<symqg::PID*>(pg + g.neighbor_offset_ * 4);
                    for (size_t j = 0; j < deg; ++j) np[j] = nbs[j];

                    symqg::RowMatrix<float> xp(deg, g.padded_dim_);
                    xp.setZero(); cp.setZero();
                    rq.clear();
                    for (size_t j = 0; j < deg; ++j)
                        rq.emplace_back(nbs[j] * fpsz, fpsz, nbs[j], nb + j * fpsz);
                    rq.emplace_back(gi * fpsz, fpsz, gi, pg);
                    vr.read(rq, tctx, false);

                    for (size_t j = 0; j < deg; ++j)
                        std::copy(reinterpret_cast<const float*>(nb + j * fpsz),
                                  reinterpret_cast<const float*>(nb + j * fpsz) + g.dimension_,
                                  &xp(static_cast<long>(j), 0));
                    std::copy(reinterpret_cast<const float*>(pg),
                              reinterpret_cast<const float*>(pg) + g.dimension_, &cp(0, 0));

                    symqg::RowMatrix<float> xr(deg, g.padded_dim_);
                    cr.setZero();
                    for (long j = 0; j < static_cast<long>(deg); ++j)
                        g.rotator_.rotate(&xp(j, 0), &xr(j, 0));
                    g.rotator_.rotate(&cp(0, 0), &cr(0, 0));

                    float* fx = reinterpret_cast<float*>(pg + 4 * g.factor_offset_);
                    symqg::rabitq_codes(xr, cr,
                        reinterpret_cast<uint8_t*>(pg + 4 * g.code_offset_),
                        fx, fx + g.degree_bound_, fx + 2 * g.degree_bound_);

                    for (size_t j = 0; j < deg; ++j) {
                        const float* rd = reinterpret_cast<const float*>(nb + j * fpsz)
                                          + g.dimension_;
                        float sq = 0;
                        for (size_t k = 0; k < g.residual_dimension_; ++k) sq += rd[k] * rd[k];
                        fx[j] += sq;
                    }

                    size_t pid = gi / g.node_per_page_;
                    size_t noff = (gi % g.node_per_page_) * g.node_len_;
                    ::pwrite(ofd, pg, g.node_len_,
                             static_cast<off_t>(symqg::kSectorLen + pid * g.page_size_ + noff));
                }
                std::free(pg);
                std::free(nb);
            }
        }
        ::close(ofd);
        std::cout << "  parallel build: " << t.elapsed_s() << " s" << std::endl;
        t.reset();

        // Phase 3: cache
        auto cn = static_cast<size_t>(static_cast<double>(num_points) * symqg::kCacheRatio);
        g.cache_ids_.resize(cn);
        {
            std::ofstream ci(idx_path + "_cache_ids", std::ios::binary);
            ci.write(reinterpret_cast<const char*>(&cn), sizeof(size_t));
            ci.write(reinterpret_cast<const char*>(g.cache_ids_.data()),
                     static_cast<std::streamsize>(sizeof(symqg::PID) * cn));
        }
        {
            std::ofstream co(idx_path + "_cache_nodes", std::ios::binary);
            co.write(reinterpret_cast<const char*>(&cn), sizeof(size_t));
            co.write(reinterpret_cast<const char*>(&g.node_len_), sizeof(size_t));
            constexpr size_t kCB = 1024;
            char* cb = reinterpret_cast<char*>(
                symqg::memory::align_allocate<symqg::kSectorLen>(kCB * g.page_size_));
            LinuxAlignedFileReader cr2;
            cr2.open(idx_path);
            cr2.register_thread();
            std::vector<AlignedRead> crq;
            crq.reserve(kCB + 1);
            for (size_t i = 0; i < cn; i += kCB) {
                size_t b = std::min(cn - i, kCB);
                for (size_t j = 0; j < b; ++j) {
                    auto cid = g.cache_ids_[i + j];
                    crq.emplace_back(symqg::kSectorLen + (cid / g.node_per_page_) * g.page_size_,
                                     g.page_size_, cid, cb + j * g.page_size_);
                }
                cr2.read(crq, cr2.get_ctx(), false);
                crq.clear();
                for (size_t j = 0; j < b; ++j) {
                    auto cid = g.cache_ids_[i + j];
                    co.write(cb + j * g.page_size_ + (cid % g.node_per_page_) * g.node_len_,
                             static_cast<std::streamsize>(g.node_len_));
                }
            }
            cr2.deregister_all_threads();
            cr2.close();
            std::free(cb);
        }
        // Rotator
        std::ofstream ro(idx_path + "_rotator", std::ios::binary);
        g.rotator_.save(ro);

        vr.deregister_all_threads();
        vr.close();
        std::remove(tmp.c_str());
        std::cout << "  cache + rotator: " << t.elapsed_s() << " s" << std::endl;
        print_rss("after QG");
    }
};

// ============================================================================
// Search helpers (small data — full-load is fine)
// ============================================================================

static auto read_fvecs(const std::string& path)
    -> std::pair<std::vector<float>, std::pair<uint32_t, uint32_t>> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open: " + path);
    int32_t dim = 0;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    fin.seekg(0, std::ios::end);
    size_t vec_sz = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(float);
    auto num = static_cast<uint32_t>(static_cast<size_t>(fin.tellg()) / vec_sz);
    std::vector<float> data(static_cast<size_t>(num) * dim);
    fin.seekg(0);
    for (uint32_t i = 0; i < num; ++i) {
        int32_t d = 0;
        fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(data.data() + static_cast<size_t>(i) * dim),
                 static_cast<std::streamsize>(dim * sizeof(float)));
    }
    return {data, {num, static_cast<uint32_t>(dim)}};
}

static auto read_ivecs(const std::string& path)
    -> std::pair<std::vector<int32_t>, std::pair<uint32_t, uint32_t>> {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open: " + path);
    int32_t dim = 0;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    fin.seekg(0, std::ios::end);
    size_t vec_sz = sizeof(int32_t) + static_cast<size_t>(dim) * sizeof(int32_t);
    auto num = static_cast<uint32_t>(static_cast<size_t>(fin.tellg()) / vec_sz);
    std::vector<int32_t> data(static_cast<size_t>(num) * dim);
    fin.seekg(0);
    for (uint32_t i = 0; i < num; ++i) {
        int32_t d = 0;
        fin.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(data.data() + static_cast<size_t>(i) * dim),
                 static_cast<std::streamsize>(dim * sizeof(int32_t)));
    }
    return {data, {num, static_cast<uint32_t>(dim)}};
}

static auto compute_recall(const uint32_t* results, const int32_t* gt,
                           size_t nq, size_t k, size_t gt_k) -> double {
    size_t correct = 0;
    for (size_t i = 0; i < nq; ++i)
        for (size_t j = 0; j < k; ++j) {
            auto rid = static_cast<int32_t>(results[i * k + j]);
            for (size_t g = 0; g < std::min(k, gt_k); ++g)
                if (rid == gt[i * gt_k + g]) { ++correct; break; }
        }
    return static_cast<double>(correct) / static_cast<double>(nq * k) * 100.0;
}

static auto percentile(std::vector<double>& v, double p) -> double {
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * static_cast<double>(v.size() - 1);
    auto lo = static_cast<size_t>(idx);
    auto hi = lo + 1;
    if (hi >= v.size()) return v.back();
    double f = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - f) + v[hi] * f;
}

// ============================================================================
// Main
// ============================================================================

auto main(int argc, char* argv[]) -> int {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <base_fvecs> <query_fvecs> <gt_ivecs>"
                  << " <vamana_index> <output_dir>"
                  << " [max_memory_mib=1024] [max_degree=64] [main_dim=256]"
                  << " [num_medoids=300] [num_threads=0] [dram_budget_gb=1.0]\n";
        return 1;
    }

    std::string base_path   = argv[1];
    std::string query_path  = argv[2];
    std::string gt_path     = argv[3];
    std::string vamana_path = argv[4];
    std::string output_dir  = argv[5];
    uint32_t max_memory_mib = (argc > 6)  ? static_cast<uint32_t>(std::stoul(argv[6]))  : 1024;
    uint32_t max_degree     = (argc > 7)  ? static_cast<uint32_t>(std::stoul(argv[7]))  : 64;
    uint32_t main_dim       = (argc > 8)  ? static_cast<uint32_t>(std::stoul(argv[8]))  : 256;
    uint32_t num_medoids    = (argc > 9)  ? static_cast<uint32_t>(std::stoul(argv[9]))  : 300;
    uint32_t num_threads    = (argc > 10) ? static_cast<uint32_t>(std::stoul(argv[10])) : 0;
    float dram_budget       = (argc > 11) ? std::stof(argv[11]) : 1.0F;
    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();

    constexpr uint32_t kTopK = 10;
    constexpr size_t kWarmup = 2;
    constexpr size_t kRuns = 5;
    std::vector<size_t> ef_values = {80, 100, 150, 200, 300, 500};

    std::filesystem::create_directories(output_dir);
    auto prefix = std::filesystem::path(output_dir) / "dsqg_gist";

    print_rss("initial");
    std::cout << "\n=== Configuration ===" << '\n';
    std::cout << "  max_memory: " << max_memory_mib << " MiB, threads: " << num_threads << '\n';
    std::cout << "  max_degree: " << max_degree << ", main_dim: " << main_dim
              << ", medoids: " << num_medoids << '\n';

    // Open base vectors (header only)
    FvecsFileReader fvecs;
    fvecs.open(base_path);
    uint32_t num_base = fvecs.num_vectors();
    uint32_t full_dim = fvecs.dim();
    std::cout << "  vectors: " << num_base << " x " << full_dim << '\n';

    auto pca_bin  = prefix.string() + "_pca.bin";
    auto pca_base = prefix.string() + "_pca_base.fbin";

    // ================================================================
    // Build Phase 1: PCA
    // ================================================================
    std::cout << "\n=== Phase 1: PCA ===" << '\n' << std::flush;
    alaya::Timer t_pca;
    build_pca(fvecs, pca_bin, pca_base, max_memory_mib, num_threads);
    double pca_s = t_pca.elapsed_s();
    std::cout << "  total: " << std::fixed << std::setprecision(1) << pca_s << " s\n";

    // ================================================================
    // Build Phase 2: Medoid
    // ================================================================
    std::cout << "\n=== Phase 2: Medoid ===" << '\n' << std::flush;
    alaya::Timer t_med;
    FbinFileReader fbin;
    fbin.open(pca_base);
    build_medoids(fbin, prefix, num_medoids, max_memory_mib, num_threads);
    double med_s = t_med.elapsed_s();
    std::cout << "  total: " << std::setprecision(1) << med_s << " s\n";

    // ================================================================
    // Build Phase 3: QG
    // ================================================================
    std::cout << "\n=== Phase 3: QG Build ===" << '\n' << std::flush;
    alaya::Timer t_qg;
    VamanaGraphReader vamana;
    vamana.open(vamana_path);
    MemLimitQGBuilder::build(pca_base, vamana, prefix,
                             num_base, full_dim, max_degree, main_dim,
                             max_memory_mib, num_threads);
    double qg_s = t_qg.elapsed_s();
    std::cout << "  total: " << std::setprecision(1) << qg_s << " s\n";

    double build_s = pca_s + med_s + qg_s;
    std::cout << "\n=== Build Summary ===" << '\n';
    std::cout << "  PCA: " << pca_s << " s | Medoid: " << med_s
              << " s | QG: " << qg_s << " s | Total: " << build_s << " s\n";
    std::cout << "  Budget: " << max_memory_mib << " MiB\n";
    print_rss("after build");

    // ================================================================
    // Search Phase
    // ================================================================
    std::cout << "\n=== Loading Queries & Ground Truth ===" << '\n' << std::flush;
    auto [qdata, qshape] = read_fvecs(query_path);
    auto [nq, qdim] = qshape;
    auto [gtdata, gtshape] = read_ivecs(gt_path);
    auto [gt_n, gt_k] = gtshape;
    std::cout << "  Queries: " << nq << " x " << qdim
              << ", GT: " << gt_n << " x " << gt_k << '\n';

    std::cout << "\n=== Loading Index ===" << '\n' << std::flush;
    alaya::LaserIndex index;
    symqg::LaserSearchParams sp;
    sp.ef_search = 200;
    sp.num_threads = 1;
    sp.beam_width = 16;
    sp.search_dram_budget_gb = dram_budget;
    alaya::Timer t_load;
    index.load(prefix.string(), num_base, max_degree, main_dim, full_dim, sp);
    std::cout << "  Load: " << std::setprecision(1) << t_load.elapsed_ms() << " ms, "
              << "cache: " << index.cached_node_count() << " nodes ("
              << (index.cache_size_bytes() / (1024 * 1024)) << " MB)\n";
    print_rss("after index load");

    std::cout << "\n" << std::string(90, '=') << '\n';
    std::cout << std::left << std::setw(10) << "EF" << std::setw(15) << "QPS"
              << std::setw(15) << "Recall(%)" << std::setw(18) << "Mean(us)"
              << std::setw(18) << "P99.9(us)" << '\n';
    std::cout << std::string(90, '-') << '\n' << std::flush;

    for (size_t ef : ef_values) {
        sp.ef_search = ef;
        sp.num_threads = 1;
        sp.beam_width = 16;
        index.set_search_params(sp);

        std::vector<uint32_t> results(static_cast<size_t>(nq) * kTopK);

        // Warmup
        for (size_t w = 0; w < std::min(kWarmup, static_cast<size_t>(nq)); ++w) {
            uint32_t tmp[kTopK];
            index.search(qdata.data() + w * qdim, kTopK, tmp);
        }

        double sum_qps = 0, sum_lat = 0, sum_p99 = 0;
        double last_recall = 0;
        for (size_t run = 0; run < kRuns; ++run) {
            std::vector<double> lats;
            lats.reserve(nq);
            double total = 0;
            for (uint32_t i = 0; i < nq; ++i) {
                alaya::Timer qt;
                index.search(qdata.data() + static_cast<size_t>(i) * qdim,
                             kTopK, results.data() + static_cast<size_t>(i) * kTopK);
                double us = qt.elapsed_us();
                lats.push_back(us);
                total += us;
            }
            total /= 1e6;
            sum_qps += nq / total;
            sum_lat += std::accumulate(lats.begin(), lats.end(), 0.0) / nq;
            sum_p99 += percentile(lats, 99.9);
            last_recall = compute_recall(results.data(), gtdata.data(), nq, kTopK, gt_k);
        }

        std::cout << std::left << std::setw(10) << ef
                  << std::setw(15) << std::fixed << std::setprecision(1) << (sum_qps / kRuns)
                  << std::setw(15) << std::setprecision(2) << last_recall
                  << std::setw(18) << std::setprecision(1) << (sum_lat / kRuns)
                  << std::setw(18) << (sum_p99 / kRuns) << '\n' << std::flush;
    }
    std::cout << std::string(90, '=') << '\n';
    print_rss("final");
    return 0;
}

// NOLINTEND
