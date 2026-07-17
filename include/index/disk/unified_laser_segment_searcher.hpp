// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>  // NOLINT(build/c++11)
#include <vector>

#include "index/disk/laser_segment_searcher.hpp"
#include "index/disk/types.hpp"

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0
  #include "index/graph/laser/qg/residency.hpp"
  #include "index/graph/laser/qg/row_admission.hpp"
#endif

namespace alaya::disk {

#if defined(ALAYA_ENABLE_LASER) && ALAYA_ENABLE_LASER != 0

// UnifiedLaserSegmentSearcher: ONE Laser segment type whose residency is a
// load-time policy instead of an index-family choice. It reuses the full
// LaserSegmentSearcher manifest/artifact validation and load path (the row
// store is identical either way), then routes searches through a
// laser::ResidencyProvider:
//   kPagedPool     -> the legacy beam/AIO path (byte-identical to
//                     LaserSegmentSearcher::search)
//   kResidentArena -> the resident-arena kernel, materialized from the same
//                     on-disk segment at prepare() time
class UnifiedLaserSegmentSearcher : public SegmentSearcher {
 public:
  explicit UnifiedLaserSegmentSearcher(const std::filesystem::path &seg_dir,
                                       laser::ResidencyMode residency,
                                       laser::NumaPolicy numa = {})
      : legacy_(seg_dir), provider_(laser::make_residency_provider(residency, numa)) {
    provider_->prepare(legacy_.graph());
  }

  UnifiedLaserSegmentSearcher(const UnifiedLaserSegmentSearcher &) = delete;
  auto operator=(const UnifiedLaserSegmentSearcher &) -> UnifiedLaserSegmentSearcher & = delete;
  UnifiedLaserSegmentSearcher(UnifiedLaserSegmentSearcher &&) = delete;
  auto operator=(UnifiedLaserSegmentSearcher &&) -> UnifiedLaserSegmentSearcher & = delete;
  ~UnifiedLaserSegmentSearcher() override = default;

  auto search(const float *query, const DiskSearchOptions &opts) const
      -> std::vector<DiskSearchHit> override {
    if (provider_->mode() == laser::ResidencyMode::kPagedPool &&
        opts.filter.kind == core::SegmentFilterKind::none) {
      // Legacy path owns its own lock and set_params cache. Only the
      // kind=none path still reaches it -- byte-identical to before the
      // admission contract landed. A non-none filter falls through to the
      // shared path below, which drives the kernel through legacy_.graph()
      // (the unified seam) instead of legacy_.search() itself: the legacy
      // searcher's own search() semantics are not touched.
      return legacy_.search(query, opts);
    }
    if (opts.top_k == 0) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: top_k must be > 0");
    }
    if (query == nullptr) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: query must not be null");
    }
    if (opts.beam_width > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: beam_width exceeds int max");
    }

    // Same discipline as the legacy searcher: serialize so set_params'
    // thread-data destroy + rebuild cannot race in-flight searches.
    const std::lock_guard<std::mutex> lock(search_mutex_);

    auto &graph = legacy_.graph();
    const auto effective_top_k =
        static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(opts.top_k), size()));
    const LastSetParams requested{
        static_cast<size_t>(std::max(opts.ef, effective_top_k)),
        1,
        static_cast<int>(opts.beam_width),
    };
    if (requested != last_set_params_) {
      graph.set_params(requested.ef_search, requested.num_threads, requested.beam_width);
      set_params_call_count_.fetch_add(1, std::memory_order_relaxed);
      last_set_params_ = requested;
    }

    std::vector<uint64_t> admission_storage;
    laser::RowAdmission admission_value{};
    const laser::RowAdmission *admission =
        compile_admission(opts.filter, size(), admission_storage, admission_value);

    // provider_->search() is a pure pass-through to the matching kernel
    // entry (paged: QuantizedGraph::search, arena: arena_search_qg) --
    // using it here regardless of residency mode keeps this one call site
    // in sync with whichever kernel the provider was constructed for.
    std::vector<uint32_t> pid_buf(effective_top_k);
    provider_->search(graph, query, effective_top_k, pid_buf.data(), admission);

    const uint64_t *labels = legacy_.labels();
    std::vector<DiskSearchHit> out;
    out.reserve(effective_top_k);
    for (uint32_t pid : pid_buf) {
      if (pid >= size()) {
        throw std::runtime_error("UnifiedLaserSegmentSearcher: QuantizedGraph returned PID " +
                                 std::to_string(pid) + " outside segment count " +
                                 std::to_string(size()));
      }
      out.push_back(DiskSearchHit{labels[pid], std::numeric_limits<float>::quiet_NaN()});
    }
    return out;
  }

  // Arena-mode batches route to the native batch kernel (per-thread slot
  // borrowing inside QuantizedGraph::arena_batch_search). Paged mode keeps
  // the base one-query-at-a-time fan-out, byte-identical to before.
  auto batch_search(const float *queries, uint32_t num_queries, const DiskSearchOptions &opts) const
      -> std::vector<std::vector<DiskSearchHit>> override {
    if (provider_->mode() == laser::ResidencyMode::kPagedPool || num_queries == 0) {
      return SegmentSearcher::batch_search(queries, num_queries, opts);
    }
    if (opts.top_k == 0) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: top_k must be > 0");
    }
    if (queries == nullptr) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: queries must not be null");
    }
    if (opts.beam_width > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("UnifiedLaserSegmentSearcher: beam_width exceeds int max");
    }

    const std::lock_guard<std::mutex> lock(search_mutex_);

    auto &graph = legacy_.graph();
    const auto effective_top_k =
        static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(opts.top_k), size()));
    const size_t batch_threads =
        std::max<size_t>(1, std::min<size_t>(num_queries, std::thread::hardware_concurrency()));
    const LastSetParams requested{
        static_cast<size_t>(std::max(opts.ef, effective_top_k)),
        batch_threads,
        static_cast<int>(opts.beam_width),
    };
    if (requested != last_set_params_) {
      graph.set_params(requested.ef_search, requested.num_threads, requested.beam_width);
      set_params_call_count_.fetch_add(1, std::memory_order_relaxed);
      last_set_params_ = requested;
    }

    std::vector<uint64_t> admission_storage;
    laser::RowAdmission admission_value{};
    const laser::RowAdmission *admission =
        compile_admission(opts.filter, size(), admission_storage, admission_value);

    std::vector<uint32_t> pid_buf(static_cast<size_t>(num_queries) * effective_top_k);
    provider_
        ->batch_search(graph, queries, effective_top_k, pid_buf.data(), num_queries, admission);

    const uint64_t *labels = legacy_.labels();
    std::vector<std::vector<DiskSearchHit>> out;
    out.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
      std::vector<DiskSearchHit> hits;
      hits.reserve(effective_top_k);
      for (uint32_t k = 0; k < effective_top_k; ++k) {
        const uint32_t pid = pid_buf[static_cast<size_t>(q) * effective_top_k + k];
        if (pid >= size()) {
          throw std::runtime_error("UnifiedLaserSegmentSearcher: QuantizedGraph returned PID " +
                                   std::to_string(pid) + " outside segment count " +
                                   std::to_string(size()));
        }
        hits.push_back(DiskSearchHit{labels[pid], std::numeric_limits<float>::quiet_NaN()});
      }
      out.push_back(std::move(hits));
    }
    return out;
  }

  auto size() const -> uint64_t override { return legacy_.size(); }
  auto dim() const -> uint32_t override { return legacy_.dim(); }
  auto type() const -> DiskIndexType override { return DiskIndexType::Laser; }

  [[nodiscard]] auto residency() const noexcept -> laser::ResidencyMode {
    return provider_->mode();
  }

  // Unified-segment seam: same pid->label view LaserSegmentSearcher exposes
  // (see its own labels() doc comment), just forwarded from legacy_ -- the
  // row store (and therefore this map) is identical across both residencies
  // this class supports. Needed by LaserSegment (the AnySegment face) to
  // translate a Collection-supplied, label-indexed bitmap into PID space
  // regardless of which residency it opened with (see decision 6/7, U2-c
  // manifest).
  [[nodiscard]] auto labels() const noexcept -> const uint64_t * { return legacy_.labels(); }

  // Test-only observer, mirrors LaserSegmentSearcher::set_params_call_count
  // for the arena path (the paged path counts inside legacy_).
  auto set_params_call_count() const noexcept -> uint64_t {
    return set_params_call_count_.load(std::memory_order_relaxed);
  }

 private:
  struct LastSetParams {
    size_t ef_search;
    size_t num_threads;
    int beam_width;

    friend auto operator==(const LastSetParams &, const LastSetParams &) -> bool = default;
  };

  // Compile a core::SegmentFilterView into the RowAdmission the kernel
  // wants. `capacity` is this segment's PID space (size()). `storage` and
  // `value` are caller-owned scratch that must outlive the returned
  // pointer -- exactly one query's worth of kernel calls.
  //
  // kind=none -> nullptr (today's behavior, zero cost).
  // kind=bitmap -> zero-copy wrap of the caller's payload (validated:
  //   word-aligned, and payload_size*8 >= capacity).
  // kind=sorted_rows -> materialized into `storage`.
  // kind=predicate/composite -> not representable at this layer; the
  //   Collection is responsible for pre-compiling those into a bitmap
  //   against its logical registry before it reaches a disk segment.
  [[nodiscard]] static auto compile_admission(const core::SegmentFilterView &filter,
                                              uint64_t capacity,
                                              std::vector<uint64_t> &storage,
                                              laser::RowAdmission &value)
      -> const laser::RowAdmission * {
    switch (filter.kind) {
      case core::SegmentFilterKind::none:
        return nullptr;
      case core::SegmentFilterKind::bitmap: {
        if (filter.payload == nullptr) {
          throw std::invalid_argument("UnifiedLaserSegmentSearcher: bitmap filter payload is null");
        }
        if (reinterpret_cast<uintptr_t>(filter.payload) % alignof(uint64_t) != 0) {
          throw std::invalid_argument(
              "UnifiedLaserSegmentSearcher: bitmap filter payload is not word-aligned");
        }
        if (filter.payload_size % sizeof(uint64_t) != 0 || filter.payload_size * 8 < capacity) {
          throw std::invalid_argument(
              "UnifiedLaserSegmentSearcher: bitmap filter payload is too small for this segment");
        }
        value = laser::admission_from_bitmap_payload(filter.payload, filter.payload_size, capacity);
        return &value;
      }
      case core::SegmentFilterKind::sorted_rows: {
        if (filter.payload == nullptr && filter.payload_size != 0) {
          throw std::invalid_argument(
              "UnifiedLaserSegmentSearcher: sorted_rows filter payload is null");
        }
        if (filter.payload_size % sizeof(uint64_t) != 0) {
          throw std::invalid_argument(
              "UnifiedLaserSegmentSearcher: sorted_rows filter payload size is not a multiple of "
              "8");
        }
        const auto *rows = static_cast<const uint64_t *>(filter.payload);
        const uint64_t n = filter.payload_size / sizeof(uint64_t);
        value = laser::admission_from_sorted_rows(rows, n, capacity, storage);
        return &value;
      }
      case core::SegmentFilterKind::predicate:
      case core::SegmentFilterKind::composite:
      default:
        throw std::runtime_error("Collection must pre-compile predicate views");
    }
  }

  mutable LaserSegmentSearcher legacy_;
  std::unique_ptr<laser::ResidencyProvider> provider_;
  mutable LastSetParams last_set_params_{0, 0, 0};
  mutable std::atomic<uint64_t> set_params_call_count_{0};
  mutable std::mutex search_mutex_;
};

#else

class UnifiedLaserSegmentSearcher : public SegmentSearcher {
 public:
  [[noreturn]] explicit UnifiedLaserSegmentSearcher(const std::filesystem::path & /*seg_dir*/) {
    throw std::runtime_error(
        "UnifiedLaserSegmentSearcher: engine 'disk_laser' not implemented in v1");
  }

  [[noreturn]] auto search(const float * /*query*/, const DiskSearchOptions & /*opts*/) const
      -> std::vector<DiskSearchHit> override {
    throw std::runtime_error(
        "UnifiedLaserSegmentSearcher: engine 'disk_laser' not implemented in v1");
  }
  auto size() const -> uint64_t override { return 0; }
  auto dim() const -> uint32_t override { return 0; }
  auto type() const -> DiskIndexType override { return DiskIndexType::Laser; }
};

#endif

}  // namespace alaya::disk
