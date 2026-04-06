/**
 * @file laser_index.hpp
 * @brief LaserIndex facade — the public API for Laser indices in AlayaLite.
 *
 * Follows DiskANN's standalone index pattern: own load/search/batch_search
 * methods, not going through the generic PyIndex<Builder, Space> dispatch.
 *
 * Zero-copy: search() passes const float* directly to QuantizedGraph
 * without allocating std::vector<float> or copying query data.
 */

#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include "index/laser/laser_types.hpp"
#include "index/laser/quantized_graph.hpp"

namespace alaya {

class LaserIndex {
   public:
    LaserIndex() = default;

    /**
     * @brief Load a pre-built Laser index from disk.
     *
     * @param prefix     File path prefix (files: {prefix}_R{R}_MD{D}.index, etc.)
     * @param num_points Number of vectors in the index
     * @param degree     Graph degree bound (R)
     * @param main_dim   Main vector dimension (MD, must be power of 2)
     * @param full_dim   Full dimension including residuals (>= main_dim)
     * @param params     Search parameters (ef_search, num_threads, beam_width, DRAM budget)
     */
    void load(
        const std::string& prefix,
        size_t num_points,
        size_t degree,
        size_t main_dim,
        size_t full_dim,
        const symqg::LaserSearchParams& params
    ) {
        graph_ = std::make_unique<symqg::QuantizedGraph>(
            num_points, degree, main_dim, full_dim
        );
        graph_->load_disk_index(prefix.c_str(), params.search_dram_budget_gb);

        num_points_ = num_points;
        dimension_ = main_dim;
        full_dimension_ = full_dim;
        loaded_ = true;

        set_search_params(params);
    }

    /**
     * @brief Reconfigure search parameters without full reload.
     */
    void set_search_params(const symqg::LaserSearchParams& params) {
        if (!loaded_) {
            throw std::runtime_error("LaserIndex: load() must be called first");
        }
        graph_->set_params(
            params.ef_search,
            params.num_threads,
            static_cast<int>(params.beam_width)
        );
    }

    /**
     * @brief Single-query search. Zero-copy: query pointer passes through directly.
     *
     * @param query   Raw float query vector (full_dimension_ floats)
     * @param k       Number of nearest neighbors to return
     * @param results Output array for k neighbor IDs (must have space for k uint32_t)
     */
    void search(const float* query, uint32_t k, uint32_t* results) {
        if (!loaded_) {
            throw std::runtime_error("LaserIndex: not loaded");
        }
        graph_->search(query, k, results);
    }

    /**
     * @brief Batch search. Thread-affine: each thread acquires context once.
     *
     * @param queries     Contiguous array of query vectors (num_queries * full_dimension_)
     * @param num_queries Number of queries
     * @param k           Number of nearest neighbors per query
     * @param results     Output array (num_queries * k uint32_t)
     */
    void batch_search(
        const float* queries, size_t num_queries, uint32_t k, uint32_t* results
    ) {
        if (!loaded_) {
            throw std::runtime_error("LaserIndex: not loaded");
        }
        graph_->batch_search(queries, k, results, num_queries);
    }

    [[nodiscard]] auto is_loaded() const -> bool { return loaded_; }
    [[nodiscard]] auto num_points() const -> size_t { return num_points_; }
    [[nodiscard]] auto dimension() const -> size_t { return dimension_; }
    [[nodiscard]] auto full_dimension() const -> size_t { return full_dimension_; }
    [[nodiscard]] auto cached_node_count() const -> size_t {
        return graph_ ? graph_->cached_node_count() : 0;
    }
    [[nodiscard]] auto cache_size_bytes() const -> size_t {
        return graph_ ? graph_->cache_size_bytes() : 0;
    }

   private:
    std::unique_ptr<symqg::QuantizedGraph> graph_;
    size_t num_points_ = 0;
    size_t dimension_ = 0;
    size_t full_dimension_ = 0;
    bool loaded_ = false;
};

}  // namespace alaya
