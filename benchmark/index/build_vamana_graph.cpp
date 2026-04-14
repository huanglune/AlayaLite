/**
 * @file build_vamana_graph.cpp
 * @brief Standalone tool to build a Vamana graph using AlayaLite's ShardVamanaBuilder
 *        and export in the official .vamana format for use with laser_build_bench.
 *
 * Usage:
 *   ./build_vamana_graph <base_fvecs> <output.vamana>
 *       [max_degree=64] [ef_construction=128] [alpha=1.2]
 *       [num_threads=0] [max_memory_mb=8192]
 */

// NOLINTBEGIN

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "index/diskann/cross_shard_merger.hpp"
#include "index/diskann/shard_vamana_builder.hpp"
#include "index/laser/laser_builder.hpp"
#include "simd/distance_l2.hpp"
#include "utils/io_utils.hpp"
#include "utils/timer.hpp"

auto main(int argc, char* argv[]) -> int {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <base_fvecs> <output.vamana>"
                  << " [max_degree=64] [ef_construction=128] [alpha=1.2]"
                  << " [num_threads=0] [max_memory_mb=8192]\n";
        return 1;
    }

    std::string base_path = argv[1];
    std::string output_path = argv[2];
    uint32_t max_degree = (argc > 3) ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
    uint32_t ef_construction = (argc > 4) ? static_cast<uint32_t>(std::stoul(argv[4])) : 128;
    float alpha = (argc > 5) ? std::stof(argv[5]) : 1.2F;
    uint32_t num_threads = (argc > 6) ? static_cast<uint32_t>(std::stoul(argv[6])) : 0;
    size_t max_memory_mb = (argc > 7) ? static_cast<size_t>(std::stoull(argv[7])) : 8192;

    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();

    std::cout << "=== AlayaLite Vamana Graph Builder ===" << std::endl;
    std::cout << "  R=" << max_degree << " L=" << ef_construction
              << " alpha=" << alpha << " threads=" << num_threads
              << " mem=" << max_memory_mb << "MB" << std::endl;

    // Load vectors
    alaya::Timer t_total;
    alaya::Timer t_load;
    std::cout << "\n--- Loading vectors ---" << std::endl;
    auto vecs = alaya::load_float_vectors(base_path);
    auto num_vecs = vecs.num_;
    auto dim = vecs.dim_;
    std::cout << "  " << num_vecs << " x " << dim << " loaded in "
              << std::fixed << std::setprecision(1) << t_load.elapsed_s() << "s" << std::endl;

    // Single-shard Vamana build
    std::cout << "\n--- Vamana build ---" << std::endl;
    alaya::Timer t_build;

    std::vector<uint32_t> all_ids(num_vecs);
    std::iota(all_ids.begin(), all_ids.end(), 0U);

    typename alaya::ShardVamanaBuilder<float, uint32_t>::Config config;
    config.max_degree_ = max_degree;
    config.ef_construction_ = ef_construction;
    config.alpha_ = alpha;
    config.max_memory_mb_ = max_memory_mb;
    config.num_threads_ = num_threads;

    alaya::ShardVamanaBuilder<float, uint32_t> builder(
        std::move(vecs.data_), dim, std::move(all_ids),
        alaya::simd::l2_sqr<float, float>, config);

    builder.build(nullptr);
    double build_s = t_build.elapsed_s();
    std::cout << "  Build time: " << build_s << "s ("
              << static_cast<double>(num_vecs) / build_s << " vec/s)" << std::endl;

    // Export to Vamana format
    std::cout << "\n--- Exporting to Vamana format ---" << std::endl;
    alaya::Timer t_export;

    alaya::VamanaFormatWriter writer(output_path, max_degree, num_vecs);
    writer.open();
    auto exported = builder.export_nodes();

    // Compute degree stats while exporting
    uint64_t total_degree = 0;
    uint32_t min_degree = max_degree;
    uint32_t max_observed = 0;
    for (const auto& node : exported) {
        auto deg = static_cast<uint32_t>(node.neighbors_.size());
        total_degree += deg;
        min_degree = std::min(min_degree, deg);
        max_observed = std::max(max_observed, deg);

        alaya::CrossShardMerger::MergedNode merged;
        merged.global_id_ = node.global_id_;
        merged.neighbor_ids_.reserve(node.neighbors_.size());
        for (const auto& nbr : node.neighbors_) {
            merged.neighbor_ids_.push_back(nbr.id_);
        }
        writer.write_node(merged);
    }
    writer.finalize();

    double export_s = t_export.elapsed_s();
    double avg_degree = static_cast<double>(total_degree) / num_vecs;
    std::cout << "  Export time: " << export_s << "s" << std::endl;
    std::cout << "\n=== Graph Stats ===" << std::endl;
    std::cout << "  Avg degree: " << std::setprecision(2) << avg_degree << std::endl;
    std::cout << "  Min degree: " << min_degree << ", Max degree: " << max_observed << std::endl;
    std::cout << "  Total time: " << t_total.elapsed_s() << "s" << std::endl;
    std::cout << "  Build only: " << build_s << "s" << std::endl;
    std::cout << "  Output: " << output_path << " ("
              << std::filesystem::file_size(output_path) / (1024 * 1024) << " MB)" << std::endl;
    return 0;
}

// NOLINTEND
