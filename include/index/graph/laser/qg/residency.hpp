// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "index/graph/laser/qg/qg.hpp"

namespace alaya::laser {

// Residency is a per-load policy over one LASER row store, not an index-family
// choice: both providers drive the same rows and the same traversal kernel and
// differ only in where rows live (U-line plan C: RowFormat x TraversalKernel x
// ResidencyProvider).
enum class ResidencyMode : uint8_t {
  kPagedPool,      // beam search over the buffer-pool/AIO path (disk_search_qg)
  kResidentArena,  // direct pid*node_len addressing over a resident arena
};

// NUMA placement hook for resident allocations. v1 implements first-touch only:
// the materializing thread's zero-fill IS the placement, so kFirstTouch needs
// no further action. kInterleave is an interface-level hook and is rejected at
// prepare() time until an implementation lands.
struct NumaPolicy {
  enum class Kind : uint8_t {
    kFirstTouch,
    kInterleave,
  };
  Kind kind = Kind::kFirstTouch;
};

class ResidencyProvider {
 public:
  ResidencyProvider() = default;
  ResidencyProvider(const ResidencyProvider &) = delete;
  auto operator=(const ResidencyProvider &) -> ResidencyProvider & = delete;
  ResidencyProvider(ResidencyProvider &&) = delete;
  auto operator=(ResidencyProvider &&) -> ResidencyProvider & = delete;
  virtual ~ResidencyProvider() = default;

  [[nodiscard]] virtual auto mode() const noexcept -> ResidencyMode = 0;

  // Bring `qg` into this provider's residency state. Idempotent. Must run
  // after load_disk_index() and before the first search; not thread-safe
  // against concurrent searches.
  virtual void prepare(QuantizedGraph &qg) = 0;

  virtual void search(QuantizedGraph &qg,
                      const float *query,
                      uint32_t knn,
                      uint32_t *results) = 0;

  virtual void batch_search(QuantizedGraph &qg,
                            const float *queries,
                            uint32_t knn,
                            uint32_t *results,
                            size_t num_queries) = 0;
};

class PagedPoolProvider final : public ResidencyProvider {
 public:
  [[nodiscard]] auto mode() const noexcept -> ResidencyMode override {
    return ResidencyMode::kPagedPool;
  }

  // load_disk_index() already staged the partial cache and the AIO pool.
  void prepare(QuantizedGraph & /*qg*/) override {}

  void search(QuantizedGraph &qg,
              const float *query,
              uint32_t knn,
              uint32_t *results) override {
    qg.search(query, knn, results);
  }

  void batch_search(QuantizedGraph &qg,
                    const float *queries,
                    uint32_t knn,
                    uint32_t *results,
                    size_t num_queries) override {
    qg.batch_search(queries, knn, results, num_queries);
  }
};

class ResidentArenaProvider final : public ResidencyProvider {
 public:
  explicit ResidentArenaProvider(NumaPolicy numa = {}) : numa_(numa) {}

  [[nodiscard]] auto mode() const noexcept -> ResidencyMode override {
    return ResidencyMode::kResidentArena;
  }

  [[nodiscard]] auto numa_policy() const noexcept -> NumaPolicy { return numa_; }

  void prepare(QuantizedGraph &qg) override {
    if (numa_.kind == NumaPolicy::Kind::kInterleave) {
      throw std::logic_error("ResidentArenaProvider: NUMA interleave not implemented in v1");
    }
    qg.ensure_resident_arena();
  }

  void search(QuantizedGraph &qg,
              const float *query,
              uint32_t knn,
              uint32_t *results) override {
    qg.arena_search_qg(query, knn, results);
  }

  void batch_search(QuantizedGraph &qg,
                    const float *queries,
                    uint32_t knn,
                    uint32_t *results,
                    size_t num_queries) override {
    qg.arena_batch_search(queries, knn, results, num_queries);
  }

 private:
  NumaPolicy numa_;
};

inline auto make_residency_provider(ResidencyMode mode, NumaPolicy numa = {})
    -> std::unique_ptr<ResidencyProvider> {
  switch (mode) {
    case ResidencyMode::kPagedPool:
      return std::make_unique<PagedPoolProvider>();
    case ResidencyMode::kResidentArena:
      return std::make_unique<ResidentArenaProvider>(numa);
  }
  throw std::invalid_argument("make_residency_provider: unknown ResidencyMode");
}

inline constexpr std::string_view kResidencyPagedPoolName = "paged_pool";
inline constexpr std::string_view kResidencyResidentArenaName = "resident_arena";

inline auto residency_mode_to_string(ResidencyMode mode) -> std::string_view {
  switch (mode) {
    case ResidencyMode::kPagedPool:
      return kResidencyPagedPoolName;
    case ResidencyMode::kResidentArena:
      return kResidencyResidentArenaName;
  }
  return {};
}

inline auto residency_mode_from_string(std::string_view s) -> ResidencyMode {
  if (s == kResidencyPagedPoolName) {
    return ResidencyMode::kPagedPool;
  }
  if (s == kResidencyResidentArenaName) {
    return ResidencyMode::kResidentArena;
  }
  throw std::invalid_argument("unknown residency mode string: '" + std::string(s) +
                              "' (expected 'paged_pool' or 'resident_arena')");
}

}  // namespace alaya::laser
