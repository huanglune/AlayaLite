/**
 * @file qg_builder.hpp
 * @brief Out-of-core index builder for Laser's quantized graph.
 *
 * Builds disk-resident index from a pre-built Vamana graph and vector data.
 * Build path is not zero-alloc critical — no SearchContext changes needed.
 */
// NOLINTBEGIN

#pragma once

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/progress_bar.hpp"
#include "utils/timer.hpp"

#include "index/laser/laser_common.hpp"
#include "index/laser/quantized_graph.hpp"
#include "index/laser/utils/partitioner.hpp"
#include "index/laser/utils/vamana_graph_reader.hpp"
#include "simd/distance_l2.hpp"
#include "utils/math.hpp"

namespace symqg {
constexpr size_t kMaxBsIter = 5;

class QGBuilder {
 private:
  QuantizedGraph &qg_;
  size_t ef_build_;
  size_t num_threads_;
  size_t num_nodes_;
  size_t dim_;
  size_t degree_bound_;
  size_t max_memory_mb_;
  DistFunc<float> dist_func_;

 public:
  explicit QGBuilder(QuantizedGraph &index,
                     uint32_t ef_build,
                     size_t num_threads,
                     size_t max_memory_mb = 4096)
      : qg_{index},
        ef_build_{ef_build},
        num_threads_{
            std::min(num_threads, static_cast<size_t>(std::thread::hardware_concurrency()) * 4)},
        num_nodes_{qg_.num_vertices()},
        dim_{qg_.dimension()},
        degree_bound_(qg_.degree_bound()),
        max_memory_mb_(max_memory_mb),
        dist_func_{alaya::simd::l2_sqr<float, float>} {}

  void init_from_vamana(alaya::VamanaGraphReader &vamana);

  void build(alaya::VamanaGraphReader &vamana, const char *filename) {
    alaya::Timer timer;
    init_from_vamana(vamana);
    std::cout << "  [QG] load vamana: " << std::fixed << std::setprecision(1) << timer.elapsed_s()
              << " s" << std::endl;
    timer.reset();

    std::string index_path = qg_.gen_index_path(filename);
    std::string data_path = std::string(filename) + "_pca_base.fbin";
    std::string tmp_path = std::string(filename) + "_tmp.fbin";

    write_metadata_and_preallocate(index_path);
    int output_fd = ::open(index_path.c_str(), O_WRONLY);
    if (output_fd < 0) {
      throw std::runtime_error("Failed to open index file for writing: " + index_path);
    }

    copy_vectors_to_temp(data_path, tmp_path);
    std::cout << "\n  [QG] phase1 copy vectors: " << timer.elapsed_s() << " s" << std::endl;
    timer.reset();

    parallel_build_index(vamana, tmp_path, output_fd);
    ::close(output_fd);
    std::cout << "\n  [QG] phase2 parallel build: " << timer.elapsed_s() << " s" << std::endl;
    timer.reset();

    save_rotator(index_path);
    std::remove(tmp_path.c_str());

    write_cache(index_path);
    std::cout << "  [QG] phase3 cache: " << timer.elapsed_s() << " s" << std::endl;
    std::cout << "Done.\n";
  }

 private:
  void write_metadata_and_preallocate(const std::string &index_path) {
    std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
    metas[0] = qg_.num_points_;
    metas[1] = qg_.dimension_;
    metas[2] = qg_.entry_point_;
    metas[3] = qg_.node_len_;
    metas[4] = qg_.node_per_page_;
    size_t page_num = (qg_.num_points_ + qg_.node_per_page_ - 1) / qg_.node_per_page_;
    metas[8] = qg_.page_size_ * page_num + kSectorLen;

    std::ofstream meta_out(index_path, std::ios::binary);
    if (!meta_out.is_open()) {
      throw std::runtime_error("Failed to create index file: " + index_path);
    }
    meta_out.write(reinterpret_cast<const char *>(metas.data()), kSectorLen);
    meta_out.seekp(static_cast<std::streamoff>(metas[8] - 1), std::ios::beg);
    char zero = 0;
    meta_out.write(&zero, 1);
  }

  void copy_vectors_to_temp(const std::string &data_path, const std::string &tmp_path) {
    std::ofstream tmp_output(tmp_path, std::ios::binary);
    if (!tmp_output.is_open()) {
      throw std::runtime_error("Failed to create temp file: " + tmp_path);
    }

    std::ifstream vector_input(data_path, std::ios::binary);
    int n = 0;
    int d = 0;
    vector_input.read(reinterpret_cast<char *>(&n), sizeof(int));
    vector_input.read(reinterpret_cast<char *>(&d), sizeof(int));
    if (d != static_cast<int>(qg_.dimension_ + qg_.residual_dimension_)) {
      throw std::runtime_error("Dimension mismatch in PCA base file");
    }

    size_t vector_tmp_page_size = (d * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;
    std::vector<char> buffer(vector_tmp_page_size);

    alaya::ProgressBar copy_bar("Copy vectors", qg_.num_points_);
    for (size_t i = 0; i < qg_.num_points_; i++) {
      std::memset(buffer.data(), 0, buffer.size());
      vector_input.read(reinterpret_cast<char *>(buffer.data()), d * sizeof(float));
      tmp_output.write(buffer.data(), static_cast<std::streamsize>(vector_tmp_page_size));
      copy_bar.tick();
    }
  }

  void parallel_build_index(alaya::VamanaGraphReader &vamana,
                            const std::string &tmp_path,
                            int output_fd) {
    LinuxAlignedFileReader vector_reader;
    vector_reader.open(tmp_path, /*direct_io=*/false);

    size_t full_dim = qg_.dimension_ + qg_.residual_dimension_;
    size_t full_page_size = (full_dim * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;
    size_t neighbor_buf_size = qg_.degree_bound_ * full_page_size;
    constexpr size_t kEdgeBytes = 64 * sizeof(uint32_t) + 40;
    if (num_nodes_ == 0) {
      vector_reader.close();
      return;
    }
    size_t memory_budget_bytes = max_memory_mb_ * 1024 * 1024;
    size_t chunk_size = memory_budget_bytes / kEdgeBytes;
    chunk_size = std::max<size_t>(chunk_size, 1000);
    chunk_size = std::min(chunk_size, num_nodes_);

    alaya::ProgressBar build_bar("Build QG index", qg_.num_points_);
    std::vector<std::vector<uint32_t>> nbrs;
    size_t cc = 0;

#pragma omp parallel num_threads(static_cast <int>(num_threads_))
    {
      IOContext ctx{};
#pragma omp critical
      {
        vector_reader.register_thread();
        ctx = vector_reader.get_ctx();
      }
      char *cur_page = reinterpret_cast<char *>(
          std::aligned_alloc(kSectorLen,
                             alaya::math::round_up_general(qg_.page_size_, kSectorLen)));
      char *neighbor_buf = reinterpret_cast<char *>(
          std::aligned_alloc(kSectorLen,
                             alaya::math::round_up_general(neighbor_buf_size, kSectorLen)));
      RowMatrix<float> c_pad(1, qg_.padded_dim_);
      RowMatrix<float> c_rotated(1, qg_.padded_dim_);
      std::vector<AlignedRead> reqs;
      reqs.reserve(qg_.degree_bound_ + 1);

      for (size_t cs = 0; cs < num_nodes_; cs += chunk_size) {
#pragma omp single
        {
          cc = std::min(chunk_size, num_nodes_ - cs);
          vamana.read_chunk(static_cast<uint32_t>(cs), static_cast<uint32_t>(cc), nbrs);
        }

#pragma omp barrier
#pragma omp for schedule(dynamic)
        for (size_t i = 0; i < cc; ++i) {
          build_bar.tick();
          const size_t node_id = cs + i;
          const auto &neighbors = nbrs[i];
          if (neighbors.empty()) {
            continue;
          }

          std::memset(cur_page, 0, qg_.page_size_);
          build_node(node_id,
                     neighbors,
                     cur_page,
                     neighbor_buf,
                     c_pad,
                     c_rotated,
                     reqs,
                     full_dim,
                     full_page_size,
                     vector_reader,
                     ctx);

          size_t page_id = node_id / qg_.node_per_page_;
          size_t node_offset = (node_id % qg_.node_per_page_) * qg_.node_len_;
          auto write_off = static_cast<off_t>(kSectorLen + page_id * qg_.page_size_ + node_offset);
          ::pwrite(output_fd, cur_page, qg_.node_len_, write_off);
        }
#pragma omp barrier
      }

      std::free(cur_page);
      std::free(neighbor_buf);
    }

    vector_reader.deregister_all_threads();
    vector_reader.close();
  }

  void build_node(size_t node_id,
                  const std::vector<uint32_t> &neighbors,
                  char *cur_page,
                  char *neighbor_buf,
                  RowMatrix<float> &c_pad,
                  RowMatrix<float> &c_rotated,
                  std::vector<AlignedRead> &reqs,
                  size_t full_dim,
                  size_t full_page_size,
                  LinuxAlignedFileReader &vector_reader,
                  IOContext &ctx) {
    size_t cur_degree = neighbors.size();
    PID *neighbor_ptr = reinterpret_cast<PID *>(cur_page + qg_.neighbor_offset_ * 4);
    for (size_t j = 0; j < cur_degree; ++j) {
      neighbor_ptr[j] = neighbors[j];
    }

    RowMatrix<float> x_pad(cur_degree, qg_.padded_dim_);
    x_pad.setZero();
    c_pad.setZero();

    reqs.clear();
    for (size_t j = 0; j < cur_degree; ++j) {
      auto nid = neighbors[j];
      reqs.emplace_back(nid * full_page_size,
                        full_page_size,
                        nid,
                        reinterpret_cast<void *>(neighbor_buf + j * full_page_size));
    }
    reqs.emplace_back(node_id * full_page_size,
                      full_page_size,
                      node_id,
                      reinterpret_cast<void *>(cur_page));

    vector_reader.read(reqs, ctx, false);

    for (size_t j = 0; j < cur_degree; ++j) {
      const auto *vec = reinterpret_cast<const float *>(neighbor_buf + j * full_page_size);
      std::copy(vec, vec + qg_.dimension_, &x_pad(static_cast<long>(j), 0));
    }
    const auto *cent = reinterpret_cast<const float *>(cur_page);
    std::copy(cent, cent + qg_.dimension_, &c_pad(0, 0));

    RowMatrix<float> x_rotated(cur_degree, qg_.padded_dim_);
    c_rotated.setZero();
    for (long j = 0; j < static_cast<long>(cur_degree); ++j) {
      qg_.rotator_.rotate(&x_pad(j, 0), &x_rotated(j, 0));
    }
    qg_.rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

    auto *fac_ptr = reinterpret_cast<float *>(cur_page + 4 * qg_.factor_offset_);
    auto *packed_code_ptr = reinterpret_cast<uint8_t *>(cur_page + 4 * qg_.code_offset_);
    float *triple_x = fac_ptr;
    float *factor_dq = triple_x + qg_.degree_bound_;
    float *factor_vq = factor_dq + qg_.degree_bound_;
    rabitq_codes(x_rotated, c_rotated, packed_code_ptr, triple_x, factor_dq, factor_vq);

    for (size_t j = 0; j < cur_degree; ++j) {
      const auto *nd = reinterpret_cast<const float *>(neighbor_buf + j * full_page_size);
      const float *res_data = nd + qg_.dimension_;
      float sqr_xr = 0;
      for (size_t k = 0; k < qg_.residual_dimension_; ++k) {
        sqr_xr += res_data[k] * res_data[k];
      }
      triple_x[j] += sqr_xr;
    }
  }

  void save_rotator(const std::string &index_path) {
    std::string rotator_path = index_path + "_rotator";
    std::ofstream rotator_output(rotator_path, std::ios::binary);
    if (!rotator_output.is_open()) {
      throw std::runtime_error("Failed to create rotator file: " + rotator_path);
    }
    qg_.rotator_.save(rotator_output);
  }

  void write_cache(const std::string &index_path) {
    std::cout << "write cache...\n";
    auto cache_num = static_cast<size_t>(static_cast<double>(qg_.num_vertices()) * kCacheRatio);
    qg_.cache_ids_.resize(cache_num);

    std::string cache_ids_file = index_path + "_cache_ids";
    std::string cache_nodes_file = index_path + "_cache_nodes";

    std::ofstream cache_ids_out(cache_ids_file, std::ios::binary);
    cache_ids_out.write(reinterpret_cast<const char *>(&cache_num), sizeof(size_t));
    cache_ids_out.write(reinterpret_cast<const char *>(qg_.cache_ids_.data()),
                        static_cast<std::streamsize>(sizeof(PID) * cache_num));
    cache_ids_out.close();

    std::ofstream cache_nodes_out(cache_nodes_file, std::ios::binary);
    if (!cache_nodes_out.is_open()) {
      throw std::runtime_error("Failed to create cache nodes file: " + cache_nodes_file);
    }
    cache_nodes_out.write(reinterpret_cast<const char *>(&cache_num), sizeof(size_t));
    cache_nodes_out.write(reinterpret_cast<const char *>(&qg_.node_len_), sizeof(size_t));

    constexpr size_t kCacheBatchSize = 1024;
    char *cache_buffer = reinterpret_cast<char *>(
        std::aligned_alloc(kSectorLen,
                           alaya::math::round_up_general(kCacheBatchSize * qg_.page_size_,
                                                         kSectorLen)));
    LinuxAlignedFileReader cache_reader;
    cache_reader.open(index_path);
    cache_reader.register_thread();
    std::vector<AlignedRead> cache_reqs;
    cache_reqs.reserve(kCacheBatchSize + 1);

    for (size_t i = 0; i < cache_num; i += kCacheBatchSize) {
      size_t cur_batch = std::min(cache_num - i, kCacheBatchSize);
      for (size_t j = 0; j < cur_batch; ++j) {
        auto cache_id = qg_.cache_ids_[i + j];
        size_t page_id = cache_id / qg_.node_per_page_;
        cache_reqs.emplace_back(kSectorLen + (page_id * qg_.page_size_),
                                qg_.page_size_,
                                cache_id,
                                cache_buffer + (j * qg_.page_size_));
      }
      cache_reader.read(cache_reqs, cache_reader.get_ctx(), false);
      cache_reqs.clear();

      for (size_t j = 0; j < cur_batch; ++j) {
        auto cache_id = qg_.cache_ids_[i + j];
        size_t node_offset = (cache_id % qg_.node_per_page_) * qg_.node_len_;
        cache_nodes_out.write(reinterpret_cast<const char *>(cache_buffer + (j * qg_.page_size_) +
                                                             node_offset),
                              static_cast<std::streamsize>(qg_.node_len_));
      }
    }
    cache_nodes_out.close();
    cache_reader.deregister_all_threads();
    cache_reader.close();
    std::free(cache_buffer);
  }
};

inline void QGBuilder::init_from_vamana(alaya::VamanaGraphReader &vamana) {
  if (vamana.max_degree() != qg_.degree_bound()) {
    throw std::runtime_error("Vamana degree mismatch: expected " +
                             std::to_string(qg_.degree_bound()) + ", got " +
                             std::to_string(vamana.max_degree()));
  }
  qg_.set_ep(vamana.entry_point());

  std::vector<uint32_t> in_degrees = vamana.compute_in_degrees();
  if (in_degrees.size() != num_nodes_) {
    throw std::runtime_error("Vamana node count mismatch: expected " + std::to_string(num_nodes_) +
                             ", got " + std::to_string(in_degrees.size()));
  }

  qg_.cache_ids_.resize(num_nodes_);
  std::iota(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), 0);
  std::sort(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), [&](PID a, PID b) {
    return in_degrees[a] > in_degrees[b];
  });
}

}  // namespace symqg
// NOLINTEND
