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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/timer.hpp"

#include "index/laser/laser_common.hpp"
#include "index/laser/quantized_graph.hpp"
#include "index/laser/space/l2.hpp"
#include "index/laser/utils/partitioner.hpp"
#include "index/laser/utils/tools.hpp"

namespace symqg {
constexpr size_t kMaxBsIter = 5;
using CandidateList = std::vector<Candidate<float>>;

class QGBuilder {
 private:
  QuantizedGraph &qg_;
  size_t ef_build_;
  size_t num_threads_;
  size_t num_nodes_;
  size_t dim_;
  size_t degree_bound_;
  size_t max_candidate_pool_size_ = 750;
  size_t max_pruned_size_ = 300;
  DistFunc<float> dist_func_;
  std::vector<CandidateList> new_neighbors_;
  std::vector<CandidateList> pruned_neighbors_;
  std::vector<uint32_t> degrees_;

 public:
  explicit QGBuilder(QuantizedGraph &index, uint32_t ef_build, size_t num_threads)
      : qg_{index},
        ef_build_{ef_build},
        num_threads_{std::min(num_threads, total_threads() * 4)},
        num_nodes_{qg_.num_vertices()},
        dim_{qg_.dimension()},
        degree_bound_(qg_.degree_bound()),
        dist_func_{space::l2_sqr},
        new_neighbors_(qg_.num_vertices()),
        pruned_neighbors_(qg_.num_vertices()),
        degrees_(qg_.num_vertices(), static_cast<uint32_t>(degree_bound_)) {}

  void init_from_vamana(const std::string &filename);

  void build(const char *vamana_file, const char *filename) {
    alaya::Timer timer;
    init_from_vamana(std::string(vamana_file));
    std::cout << "  [QG] load vamana: " << std::fixed << std::setprecision(1)
              << timer.elapsed_s() << " s" << std::endl;
    timer.reset();

    std::string index_path = qg_.gen_index_path(filename);
    std::string data_path = std::string(filename) + "_pca_base.fbin";
    std::string tmp_path = std::string(filename) + "_tmp.fbin";

    // Write metadata sector and pre-allocate file
    std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
    metas[0] = qg_.num_points_;
    metas[1] = qg_.dimension_;
    metas[2] = qg_.entry_point_;
    metas[3] = qg_.node_len_;
    metas[4] = qg_.node_per_page_;
    size_t page_num = (qg_.num_points_ + qg_.node_per_page_ - 1) / qg_.node_per_page_;
    metas[8] = qg_.page_size_ * page_num + kSectorLen;
    {
      std::ofstream meta_out(index_path, std::ios::binary);
      assert(meta_out.is_open());
      meta_out.write(reinterpret_cast<const char *>(metas.data()), kSectorLen);
      meta_out.seekp(static_cast<std::streamoff>(metas[8] - 1), std::ios::beg);
      char zero = 0;
      meta_out.write(&zero, 1);
    }
    int output_fd = ::open(index_path.c_str(), O_WRONLY);
    assert(output_fd >= 0);

    // Phase 1: Create sector-aligned temp vector file
    std::ofstream tmp_output(tmp_path, std::ios::binary);
    assert(tmp_output.is_open());

    std::ifstream vector_input(data_path, std::ios::binary);
    int n = 0;
    int d = 0;
    vector_input.read(reinterpret_cast<char *>(&n), sizeof(int));
    vector_input.read(reinterpret_cast<char *>(&d), sizeof(int));
    assert(d == static_cast<int>(qg_.dimension_ + qg_.residual_dimension_));

    size_t vector_tmp_page_size = (d * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;
    std::vector<char> buffer(vector_tmp_page_size);

    std::cout << "copy vectors..." << std::endl;
    for (size_t i = 0; i < qg_.num_points_; i++) {
      if (i % 10000 == 0) {
        float progress = static_cast<float>(i) * 100 / qg_.num_points_;
        int bar_width = 50;
        int pos = static_cast<int>(bar_width * progress / 100);
        std::cout << "\r[";
        for (int j = 0; j < bar_width; ++j) {
          if (j < pos)
            std::cout << "=";
          else if (j == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%" << std::flush;
      }
      std::memset(buffer.data(), 0, buffer.size());
      vector_input.read(reinterpret_cast<char *>(buffer.data()), d * sizeof(float));
      tmp_output.write(buffer.data(), static_cast<std::streamsize>(vector_tmp_page_size));
    }
    vector_input.close();
    tmp_output.close();
    std::cout << "\n  [QG] phase1 copy vectors: "
              << timer.elapsed_s() << " s" << std::endl;
    timer.reset();

    // Phase 2: Parallel out-of-core index construction
    // Per-thread scratch: avoids ConcurrentQueue mutex per node (1M nodes).
    // Build-time: read from page cache (temp file just written in Phase 1)
    LinuxAlignedFileReader vector_reader;
    vector_reader.open(tmp_path, /*direct_io=*/false);

    size_t full_dim = qg_.dimension_ + qg_.residual_dimension_;
    size_t full_page_size = (full_dim * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;
    size_t neighbor_buf_size = qg_.degree_bound_ * full_page_size;

    std::cout << "\nupdate qg..." << std::endl;

#pragma omp parallel num_threads(static_cast<int>(num_threads_))
    {
      // One-time per-thread allocation
      IOContext ctx{};
#pragma omp critical
      {
        vector_reader.register_thread();
        ctx = vector_reader.get_ctx();
      }
      char *cur_page =
          reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(qg_.page_size_));
      char *neighbor_buf =
          reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(neighbor_buf_size));
      RowMatrix<float> c_pad(1, qg_.padded_dim_);
      RowMatrix<float> c_rotated(1, qg_.padded_dim_);
      std::vector<AlignedRead> reqs;
      reqs.reserve(qg_.degree_bound_ + 1);

#pragma omp for schedule(dynamic)
      for (size_t i = 0; i < qg_.num_points_; ++i) {
        if (i % 10000 == 0) {
          float progress = static_cast<float>(i) * 100 / qg_.num_points_;
          int bar_width = 50;
          int pos = static_cast<int>(bar_width * progress / 100);
          std::cout << "\r[";
          for (int j = 0; j < bar_width; ++j) {
            if (j < pos) std::cout << "=";
            else if (j == pos) std::cout << ">";
            else std::cout << " ";
          }
          std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%"
                    << std::flush;
        }

        std::memset(cur_page, 0, qg_.page_size_);
        size_t cur_degree = new_neighbors_[i].size();
        if (cur_degree == 0) {
          continue;
        }

        PID *neighbor_ptr = reinterpret_cast<PID *>(cur_page + qg_.neighbor_offset_ * 4);
        for (size_t j = 0; j < cur_degree; ++j) {
          neighbor_ptr[j] = new_neighbors_[i][j].id;
        }

        RowMatrix<float> x_pad(cur_degree, qg_.padded_dim_);
        x_pad.setZero();
        c_pad.setZero();

        reqs.clear();
        for (size_t j = 0; j < cur_degree; ++j) {
          auto nid = new_neighbors_[i][j].id;
          reqs.emplace_back(nid * full_page_size,
                            full_page_size,
                            nid,
                            reinterpret_cast<void *>(neighbor_buf + j * full_page_size));
        }
        reqs.emplace_back(
            i * full_page_size, full_page_size, i, reinterpret_cast<void *>(cur_page));

        vector_reader.read(reqs, ctx, false);

        for (size_t j = 0; j < cur_degree; ++j) {
          const auto *vec =
              reinterpret_cast<const float *>(neighbor_buf + j * full_page_size);
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
        auto *packed_code_ptr =
            reinterpret_cast<uint8_t *>(cur_page + 4 * qg_.code_offset_);
        float *triple_x = fac_ptr;
        float *factor_dq = triple_x + qg_.degree_bound_;
        float *factor_vq = factor_dq + qg_.degree_bound_;
        rabitq_codes(x_rotated, c_rotated, packed_code_ptr, triple_x, factor_dq, factor_vq);

        for (size_t j = 0; j < cur_degree; ++j) {
          const auto *nd =
              reinterpret_cast<const float *>(neighbor_buf + j * full_page_size);
          const float *res_data = nd + qg_.dimension_;
          float sqr_xr = 0;
          for (size_t k = 0; k < qg_.residual_dimension_; ++k) {
            sqr_xr += res_data[k] * res_data[k];
          }
          triple_x[j] += sqr_xr;
        }

        size_t page_id = i / qg_.node_per_page_;
        size_t node_offset = (i % qg_.node_per_page_) * qg_.node_len_;
        auto write_off =
            static_cast<off_t>(kSectorLen + page_id * qg_.page_size_ + node_offset);
        ::pwrite(output_fd, cur_page, qg_.node_len_, write_off);
      }

      std::free(cur_page);
      std::free(neighbor_buf);
    }
    ::close(output_fd);
    std::cout << "\n  [QG] phase2 parallel build: "
              << timer.elapsed_s() << " s" << std::endl;
    timer.reset();

    // Save rotation matrix
    std::string rotator_path = index_path + "_rotator";
    std::ofstream rotator_output(rotator_path, std::ios::binary);
    assert(rotator_output.is_open());
    qg_.rotator_.save(rotator_output);
    rotator_output.close();

    // Cleanup temp file
    std::remove(tmp_path.c_str());

    // Phase 3: Build cache file
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
    assert(cache_nodes_out.is_open());
    cache_nodes_out.write(reinterpret_cast<const char *>(&cache_num), sizeof(size_t));
    cache_nodes_out.write(reinterpret_cast<const char *>(&qg_.node_len_), sizeof(size_t));

    size_t batch_size = 1024;
    char *cache_buffer =
        reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(batch_size * qg_.page_size_));
    LinuxAlignedFileReader cache_reader;
    cache_reader.open(index_path);
    cache_reader.register_thread();
    std::vector<AlignedRead> cache_reqs;
    cache_reqs.reserve(batch_size + 1);

    for (size_t i = 0; i < cache_num; i += batch_size) {
      size_t cur_batch = std::min(cache_num - i, batch_size);
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

    // Per-thread scratch already freed inside omp parallel block
    vector_reader.deregister_all_threads();
    vector_reader.close();
    std::cout << "  [QG] phase3 cache: "
              << timer.elapsed_s() << " s" << std::endl;
    std::cout << "Done.\n";
  }
};

inline void QGBuilder::init_from_vamana(const std::string &filename) {
  size_t expected_file_size = 0;
  size_t file_frozen_pts = 0;
  uint32_t start = 0;
  uint32_t max_observed_degree = 0;
  uint32_t max_range_of_graph = 0;

  std::ifstream in;
  in.exceptions(std::ios::badbit | std::ios::failbit);
  in.open(filename, std::ios::binary);
  in.read(reinterpret_cast<char *>(&expected_file_size), sizeof(size_t));
  in.read(reinterpret_cast<char *>(&max_observed_degree), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&start), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&file_frozen_pts), sizeof(size_t));

  size_t vamana_metadata_size =
      sizeof(size_t) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(size_t);

  assert(max_observed_degree == qg_.degree_bound());
  qg_.set_ep(start);

  std::vector<uint32_t> in_degrees(num_nodes_, 0);
  size_t bytes_read = vamana_metadata_size;
  size_t cc = 0;
  uint32_t nodes_read = 0;

  while (bytes_read != expected_file_size) {
    uint32_t k = 0;
    in.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));

    ++nodes_read;
    std::vector<uint32_t> tmp(k);
    in.read(reinterpret_cast<char *>(tmp.data()),
            static_cast<std::streamsize>(k * sizeof(uint32_t)));

    for (PID cur_neigh : tmp) {
      in_degrees[cur_neigh]++;
      new_neighbors_[nodes_read - 1].emplace_back(cur_neigh, 0.0F);
    }

    cc += k;
    bytes_read += sizeof(uint32_t) * (static_cast<uint32_t>(k) + 1);
    if (k > max_range_of_graph) {
      max_range_of_graph = k;
    }
  }

  qg_.cache_ids_.resize(num_nodes_);
  std::iota(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), 0);
  std::sort(qg_.cache_ids_.begin(), qg_.cache_ids_.end(), [&](PID a, PID b) {
    return in_degrees[a] > in_degrees[b];
  });

  assert(nodes_read == num_nodes_);
}

}  // namespace symqg
// NOLINTEND
