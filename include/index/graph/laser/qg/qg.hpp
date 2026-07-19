// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "concurrentqueue.h"  // NOLINT
#include "core/value_types.hpp"
#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/qg/qg_scanner.hpp"
#include "index/graph/laser/qg/row_admission.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/ip.hpp"
#include "index/graph/laser/space/l2.hpp"
#include "index/graph/laser/utils/aligned_file_reader.hpp"
#if defined(_WIN32)
  #include "index/graph/laser/utils/aligned_file_reader_factory.hpp"
#else
  #include "index/graph/laser/utils/page_reader_adapter.hpp"
#endif
#include "index/graph/laser/utils/array.hpp"
#include "index/graph/laser/utils/buffer.hpp"
#include "index/graph/laser/utils/concurrent_queue.hpp"
#include "index/graph/laser/utils/io.hpp"
#include "index/graph/laser/utils/memory.hpp"
#include "index/graph/laser/utils/pca_transform.hpp"
#include "index/graph/laser/utils/rotator.hpp"
#include "index/graph/laser/utils/tools.hpp"
#include "platform/detect.hpp"
#include "third_party/ngt/hashset.hpp"
#include "utils/kernel_section_profile.hpp"
#include "utils/memory.hpp"
#include "utils/prefetch.hpp"

namespace alaya::laser {
/**
 * @brief this Factor only for illustration, the true storage is continuous
 * degree_bound_*triple_x + degree_bound_*factor_dq + degree_bound_*factor_vq
 *
 * Note: ||x_r||^2 (residual dimension norm) is pre-added to triple_x during build
 */
struct Factor {
  float triple_x;   // Sqr of distance to centroid + 2 * x * x1 / x0 + ||x_r||^2 (residual dim)
  float factor_dq;  // Factor of delta * ||q_r|| * (FastScanRes - sum_q)
  float factor_vq;  // Factor of v_l * ||q_r||
};

// Graph-independent search workspace. It deliberately owns only ordinary
// containers: no graph pointer, file descriptor, IOContext, page-reader
// completion state, or buffer whose layout is tied to a particular graph.
// That makes one instance safe to retain in thread_local storage and reuse
// across collections; ensure() re-establishes every capacity needed by the
// current (graph, ef, dimension) tuple before a query touches it.
struct ArenaScratch {
  HashBasedBooleanSet visited_;
  buffer::SearchBuffer search_pool_;
  std::vector<float> pca_query_scratch_;

  void ensure(size_t num_points, size_t ef_search, size_t query_dimension) {
    search_pool_.resize(ef_search);

    const size_t max_size = std::numeric_limits<size_t>::max();
    const size_t ef_squared =
        ef_search != 0 && ef_search > max_size / ef_search ? max_size : ef_search * ef_search;
    // HashBasedBooleanSet(0) still creates a small table, but an untouched
    // default instance has no table at all. Normalize the requested capacity
    // to one so even tiny graphs initialize it before get()/set().
    const size_t visited_capacity = std::max<size_t>(1, std::min(num_points / 10, ef_squared));
    if (visited_capacity_ < visited_capacity) {
      visited_ = HashBasedBooleanSet(visited_capacity);
      visited_capacity_ = visited_capacity;
    } else {
      visited_.clear();
    }

    if (pca_query_scratch_.size() < query_dimension) {
      pca_query_scratch_.resize(query_dimension);
    }
  }

 private:
  size_t visited_capacity_ = 0;
};

// Per-graph paged-search lease. Unlike ArenaScratch, these members are bound
// to the graph's page reader / registered file and must never live in a
// process-wide thread_local. The embedded graph-free scratch remains here for
// the legacy paged queue; resident-arena search uses ArenaScratch directly.
struct ThreadData {
  ArenaScratch search_scratch_;
  IOContext ctx_;
#if !defined(_WIN32)
  std::shared_ptr<PageReadCompletions> completions_ = std::make_shared<PageReadCompletions>();
#endif
  char *sector_scratch_ = nullptr;
  size_t sector_scratch_slots_ = 0;
  char *neighbor_vector_scratch_ = nullptr;
  char *cur_page_scratch_ = nullptr;
};

struct ClusterStats {
  float avg_dist;
  float std_dev;
  float z_min;
  float z_max;
};

constexpr size_t kSectorLen = 4096;
constexpr size_t kQGRowTrailerSize = 4;

struct QGPageGeometry {
  size_t node_per_page;
  size_t page_size;
};

/** Derive page geometry for newly built indexes, reserving one v2 trailer per row. */
// Loop-form row prefetch: the Duff-device helpers cap at 20 cache lines,
// but an arena row is up to 120 lines (7,680 B at 768d).
inline void prefetch_row_l1(const char *ptr, size_t num_lines) {
  for (size_t i = 0; i < num_lines; ++i) {
    ::alaya::prefetch_l1(ptr + i * 64);
  }
}

inline void prefetch_row_l2(const char *ptr, size_t num_lines) {
  for (size_t i = 0; i < num_lines; ++i) {
    ::alaya::prefetch_l2(ptr + i * 64);
  }
}

// Candidate-row prefetch depth for the resident-arena kernel, in 64B lines.
// The memory QG kernel issues mem_prefetch_l2(next candidate, 10 lines) after
// every pool insert; the arena kernel historically issued none, which showed
// up as +19.5% LLC misses at the same ef. E1 (fullcache-adjudication appendix 4):
// the win saturates at a fixed in-flight budget of ~20 lines (~1.3KB); short
// rows want the whole row. Default = min(row_lines, 20); ALAYA_ARENA_PF_LINES
// overrides (0 disables — A/B control).
inline size_t arena_prefetch_default(size_t row_lines, int64_t env_lines) {
  if (env_lines >= 0) {
    return static_cast<size_t>(env_lines);
  }
  constexpr size_t kBudgetLines = 20;
  return std::min(row_lines, kBudgetLines);
}

inline size_t arena_prefetch_lines(size_t row_lines) {
  static const int64_t kEnvLines = [] {
    const char *v = std::getenv("ALAYA_ARENA_PF_LINES");
    return v == nullptr ? int64_t{-1} : static_cast<int64_t>(std::strtoul(v, nullptr, 10));
  }();
  return arena_prefetch_default(row_lines, kEnvLines);
}

inline QGPageGeometry qg_page_geometry(size_t node_len) {
  if (node_len == 0) {
    throw std::invalid_argument("qg_page_geometry: node_len must be > 0");
  }
  size_t node_per_page = std::max<size_t>(1, kSectorLen / node_len);
  size_t page_size = (node_per_page * node_len + kSectorLen - 1) / kSectorLen * kSectorLen;
  while (node_per_page > 1 &&
         page_size - node_per_page * node_len < node_per_page * kQGRowTrailerSize) {
    --node_per_page;
  }
  if (node_per_page == 1 && page_size - node_len < kQGRowTrailerSize) {
    page_size += kSectorLen;
  }
  return {node_per_page, page_size};
}

// LASER QG format-v2 metadata.  The two copies occupy [0, 512) and
// [512, 1024) of the existing 4 KiB metadata sector.  The compatibility
// fields let the ordinary read-only QG loader validate/open a
// v2 file without changing any graph-search layout or addressing code.
constexpr uint64_t kQGSuperblockMagic = 0x324751524553414cULL;  // "LASERQG2"
constexpr uint32_t kQGFormatVersion = 2;
constexpr size_t kQGSuperblockSize = 512;
constexpr size_t kQGSuperblockCopies = 2;

// v3 outer format: the 2C maintenance/reuse features are activated. The 512-byte
// struct, page layout, and kind=6 image size are UNCHANGED from v2; only
// format_version flips and the reserved[56..408) 2C extension carries magic +
// feature bits. create_empty / the immutable builder keep producing v2; only an
// activation checkpoint writes v3 (design section 7.2, codex B-2C-06).
constexpr uint32_t kQGFormatVersionV3 = 3;

// 2C superblock extension sub-layout inside QGSuperblockV2::reserved[408]
// (host-endian). See qg_updater.hpp kWal2c* for the absolute struct offsets.
//   reserved[56..64) wal2c_magic (u64)
//   reserved[64..68) wal2c_layout_version (u32)
//   reserved[68..72) required_feature_flags (u32)
constexpr size_t kWal2cReservedOffset = 56;
constexpr uint64_t kWal2cMagic = 0x57414C32435F5631ULL;  // "WAL2C_V1"
constexpr uint32_t kWal2cLayoutVersion = 1;

// required_feature_flags bits (design section 7.1). "required" == a reader that
// does not understand the bit MUST fail closed rather than open the segment.
constexpr uint32_t kQgFeatMaintenanceTxV1 = 1U << 0U;
constexpr uint32_t kQgFeatPostRedoFreeListV1 = 1U << 1U;
constexpr uint32_t kQgFeatPidGenerationV1 = 1U << 2U;
constexpr uint32_t kQgFeatCanonicalPrebindV1 = 1U << 3U;
constexpr uint32_t kQgFeatMutableLabelSlotV1 = 1U << 4U;

// The required-feature bitmask THIS build understands. It grows per 2C phase:
// W1 understands the maintenance-transaction features (consolidate under WAL); W2
// adds the pid-reuse triple (pid_generation + canonical_prebind + mutable_label).
// This mask MUST grow in the SAME commit that lands the canonical writer (step 5)
// AND the canonical replay lane (step 3) -- extending it before the recovery path
// exists would let a new binary open a v3 pid base it cannot actually recover
// (design B.1 / codex risk 1). A v3 base requiring a bit outside this mask (a future
// phase) still fails closed.
constexpr uint32_t kQgSupportedRequiredFeatures =
    kQgFeatMaintenanceTxV1 | kQgFeatPostRedoFreeListV1 | kQgFeatPidGenerationV1 |
    kQgFeatCanonicalPrebindV1 | kQgFeatMutableLabelSlotV1;

struct QGSuperblockV2 {
  uint64_t magic = 0;
  uint32_t format_version = 0;
  uint32_t checksum = 0;
  uint64_t generation = 0;
  uint64_t num_points = 0;
  uint64_t live_count = 0;
  PID free_list_head = kPidMax;
  uint32_t reserved0 = 0;
  uint64_t free_count = 0;

  // v1 loader metadata mirrored into the reserved portion of the 512-byte
  // superblock. The required v2 fields above remain authoritative.
  uint64_t entry_point = 0;
  uint64_t dimension = 0;
  uint64_t node_len = 0;
  uint64_t node_per_page = 0;
  uint64_t page_size = 0;
  uint64_t file_size = 0;
  // reserved[] carries the op-WAL lineage + 2A label/tx state as a host-endian
  // sub-layout (see QGUpdater kUidReservedOffset / kLabelStateReservedOffset /
  // kTxStateReservedOffset): [0..8) segment_uid, [8..40) label slot/gen/count/
  // checksum, [40..56) last_committed_txid/applied_collection_op_id, [56..408) 2C.
  std::array<uint8_t, 408> reserved{};
};
static_assert(sizeof(QGSuperblockV2) == kQGSuperblockSize);

inline uint32_t qg_crc32(const void *data, size_t len) {
  uint32_t crc = 0xffffffffU;
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (size_t i = 0; i < len; ++i) {
    crc ^= bytes[i];
    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1U) ^ (0xedb88320U & (0U - (crc & 1U)));
    }
  }
  return ~crc;
}

inline uint32_t qg_superblock_checksum(const QGSuperblockV2 &sb) {
  QGSuperblockV2 copy = sb;
  copy.checksum = 0;
  return qg_crc32(&copy, sizeof(copy));
}

// Immutable QG v1 files predate metric persistence: metadata words [5..7]
// were reserved and canonical-zero, while word 8 stores file_size. Keep that
// zero encoding as the backward-compatible proof for L2 so existing L2
// artifacts remain byte-identical. Non-L2 files must carry this checksummed
// proof; a manifest alone is not allowed to reinterpret a physically-L2 QG as
// IP/cosine.
constexpr std::size_t kQGNativeSemanticsOffset = 5 * sizeof(std::uint64_t);
constexpr std::uint64_t kQGNativeSemanticsMagic = 0x3154454d47514cULL;  // "LQGMET1"
constexpr std::uint32_t kQGNativeSemanticsVersion = 1;

struct QGNativeSemanticsProofV1 {
  std::uint64_t magic{};
  std::uint64_t payload{};
  std::uint64_t checksum{};
};
static_assert(sizeof(QGNativeSemanticsProofV1) == 3 * sizeof(std::uint64_t));

struct QGNativeSemantics {
  core::Metric metric{core::Metric::l2};
  core::MetricPreprocessing preprocessing{core::MetricPreprocessing::none};
  bool explicit_proof{};
};

[[nodiscard]] inline auto qg_expected_preprocessing(core::Metric metric)
    -> core::MetricPreprocessing {
  switch (metric) {
    case core::Metric::l2:
    case core::Metric::inner_product:
      return core::MetricPreprocessing::none;
    case core::Metric::cosine:
      return core::MetricPreprocessing::l2_normalized;
  }
  throw std::invalid_argument("QuantizedGraph: unsupported metric");
}

inline void qg_validate_metric_configuration(core::Metric metric,
                                             core::MetricPreprocessing preprocessing) {
  if (preprocessing != qg_expected_preprocessing(metric)) {
    throw std::invalid_argument("QuantizedGraph: metric and native preprocessing proof disagree");
  }
}

[[nodiscard]] inline auto qg_native_semantics_payload(core::Metric metric,
                                                      core::MetricPreprocessing preprocessing)
    -> std::uint64_t {
  qg_validate_metric_configuration(metric, preprocessing);
  return static_cast<std::uint64_t>(kQGNativeSemanticsVersion) |
         (static_cast<std::uint64_t>(metric) << 32U) |
         (static_cast<std::uint64_t>(preprocessing) << 40U);
}

[[nodiscard]] inline auto qg_native_semantics_checksum(std::uint64_t magic, std::uint64_t payload)
    -> std::uint64_t {
  const std::array<std::uint64_t, 2> words{magic, payload};
  const auto crc = qg_crc32(words.data(), sizeof(words));
  return static_cast<std::uint64_t>(crc) | (static_cast<std::uint64_t>(~crc) << 32U);
}

inline void qg_write_native_semantics(void *header,
                                      std::size_t header_size,
                                      core::Metric metric,
                                      core::MetricPreprocessing preprocessing) {
  qg_validate_metric_configuration(metric, preprocessing);
  if (metric == core::Metric::l2) {
    // Canonical legacy encoding. QGBuilder zero-initializes the metadata
    // sector, so deliberately writing nothing preserves every L2 byte.
    return;
  }
  if (header == nullptr ||
      header_size < kQGNativeSemanticsOffset + sizeof(QGNativeSemanticsProofV1)) {
    throw std::invalid_argument("QG native semantics metadata is truncated");
  }
  QGNativeSemanticsProofV1 proof;
  proof.magic = kQGNativeSemanticsMagic;
  proof.payload = qg_native_semantics_payload(metric, preprocessing);
  proof.checksum = qg_native_semantics_checksum(proof.magic, proof.payload);
  std::memcpy(static_cast<char *>(header) + kQGNativeSemanticsOffset, &proof, sizeof(proof));
}

[[nodiscard]] inline auto qg_read_native_semantics(const void *header, std::size_t header_size)
    -> QGNativeSemantics {
  if (header == nullptr || header_size < sizeof(std::uint64_t)) {
    throw std::invalid_argument("QG native semantics metadata is truncated");
  }

  // Mutable v2/v3 superblocks have no non-L2 codec today. Their first metric
  // implementation remains the canonical implicit L2 encoding.
  std::uint64_t magic_a{};
  std::memcpy(&magic_a, header, sizeof(magic_a));
  std::uint64_t magic_b{};
  if (header_size >= kQGSuperblockSize + sizeof(magic_b)) {
    std::memcpy(&magic_b, static_cast<const char *>(header) + kQGSuperblockSize, sizeof(magic_b));
  }
  if (magic_a == kQGSuperblockMagic || magic_b == kQGSuperblockMagic) {
    return {};
  }

  if (header_size < kQGNativeSemanticsOffset + sizeof(QGNativeSemanticsProofV1)) {
    return {};
  }
  QGNativeSemanticsProofV1 proof;
  std::memcpy(&proof, static_cast<const char *>(header) + kQGNativeSemanticsOffset, sizeof(proof));
  if (proof.magic == 0 && proof.payload == 0 && proof.checksum == 0) {
    return {};
  }
  if (proof.magic != kQGNativeSemanticsMagic ||
      proof.checksum != qg_native_semantics_checksum(proof.magic, proof.payload) ||
      static_cast<std::uint32_t>(proof.payload) != kQGNativeSemanticsVersion ||
      (proof.payload >> 48U) != 0) {
    throw std::invalid_argument("QG native metric/preprocessing proof is malformed");
  }
  const auto metric = static_cast<core::Metric>((proof.payload >> 32U) & 0xffU);
  const auto preprocessing = static_cast<core::MetricPreprocessing>((proof.payload >> 40U) & 0xffU);
  qg_validate_metric_configuration(metric, preprocessing);
  return {metric, preprocessing, true};
}

inline void qg_validate_native_semantics(const void *header,
                                         std::size_t header_size,
                                         core::Metric expected_metric,
                                         core::MetricPreprocessing expected_preprocessing) {
  qg_validate_metric_configuration(expected_metric, expected_preprocessing);
  const auto actual = qg_read_native_semantics(header, header_size);
  if ((!actual.explicit_proof && expected_metric != core::Metric::l2) ||
      actual.metric != expected_metric || actual.preprocessing != expected_preprocessing) {
    throw std::invalid_argument(
        "QG native metric/preprocessing proof disagrees with the requested semantics");
  }
}

inline void qg_validate_native_semantics_file(const std::filesystem::path &path,
                                              core::Metric expected_metric,
                                              core::MetricPreprocessing expected_preprocessing) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open QG native artifact for semantics validation: " +
                             path.string());
  }
  std::array<char, kSectorLen> header{};
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  const auto bytes = static_cast<std::size_t>(input.gcount());
  qg_validate_native_semantics(header.data(), bytes, expected_metric, expected_preprocessing);
}

// Structural (not-corrupt) integrity: correct magic and a matching checksum ONLY.
// BLOCKER-3: the outer format version is deliberately NOT part of structural validity.
// A checksum-legal block of a FUTURE outer version (v4+) must count as structurally
// valid so the fail-closed selector picks it as the highest-generation copy and then
// REJECTS the whole file (version/feature unsupported) -- rather than treating it as
// "corrupt" and silently downgrading to an older, supported slot. The version/support
// decision lives entirely in qg_superblock_supported below.
inline bool qg_superblock_valid(const QGSuperblockV2 &sb) {
  return sb.magic == kQGSuperblockMagic && sb.checksum == qg_superblock_checksum(sb);
}

// Read the 2C extension fields out of the reserved area (host-endian).
inline uint64_t qg_read_wal2c_magic(const QGSuperblockV2 &sb) {
  uint64_t magic = 0;
  std::memcpy(&magic, sb.reserved.data() + kWal2cReservedOffset, sizeof(magic));
  return magic;
}
inline uint32_t qg_read_wal2c_layout_version(const QGSuperblockV2 &sb) {
  uint32_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 8, sizeof(v));
  return v;
}
inline uint32_t qg_read_required_feature_flags(const QGSuperblockV2 &sb) {
  uint32_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 12, sizeof(v));
  return v;
}
inline uint64_t qg_read_maintenance_activation_gen(const QGSuperblockV2 &sb) {
  uint64_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 24, sizeof(v));
  return v;
}
inline uint64_t qg_read_pid_reuse_activation_gen(const QGSuperblockV2 &sb) {
  uint64_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 32, sizeof(v));
  return v;
}
inline uint32_t qg_read_max_pid_generation(const QGSuperblockV2 &sb) {
  uint32_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 40, sizeof(v));
  return v;
}
inline uint32_t qg_read_nonzero_pid_generation_count(const QGSuperblockV2 &sb) {
  uint32_t v = 0;
  std::memcpy(&v, sb.reserved.data() + kWal2cReservedOffset + 44, sizeof(v));
  return v;
}
// The 2C reserved sub-layout defines exactly [kWal2cReservedOffset, +kWal2cStateBytes);
// every byte from there to the end of reserved[] must stay zero in EVERY version.
constexpr size_t kWal2cStateBytes = 48;
inline bool qg_reserved_range_is_zero(const QGSuperblockV2 &sb, size_t from, size_t to) {
  for (size_t i = from; i < to && i < sb.reserved.size(); ++i) {
    if (sb.reserved[i] != 0) {
      return false;
    }
  }
  return true;
}

// Feature-dependency invariant (design section 7.1, codex BLOCKER-5): the maintenance
// pair (maintenance_tx + post_redo_free_list) is all-or-none; the pid-reuse triple
// (pid_generation + canonical_prebind + mutable_label_slot) is all-or-none AND depends
// on the maintenance pair. A partial set (e.g. pid without maintenance) is a forged /
// corrupt v3 base -- accepting it would let a reader treat the base as pid-active but
// maintenance-inactive, whose next checkpoint would revert to a v2 image carrying stale
// non-zero 2C reserved bytes and self-lock the file on the following open.
inline bool qg_required_features_self_consistent(uint32_t required) {
  const bool maint = (required & kQgFeatMaintenanceTxV1) != 0;
  const bool postredo = (required & kQgFeatPostRedoFreeListV1) != 0;
  const bool pid = (required & kQgFeatPidGenerationV1) != 0;
  const bool canonical = (required & kQgFeatCanonicalPrebindV1) != 0;
  const bool mutable_label = (required & kQgFeatMutableLabelSlotV1) != 0;
  if (maint != postredo) return false;                         // maintenance pair all-or-none
  if (pid != canonical || pid != mutable_label) return false;  // pid triple all-or-none
  if (pid && !maint) return false;  // pid depends on the maintenance pair
  return true;
}

// Whether THIS build supports a structurally-valid block. v2 requires the 2C
// extension to be canonical-zero; v3 requires the 2C magic, a supported layout
// version, self-consistent deps, and (required & ~supported)==0.
inline bool qg_superblock_supported(const QGSuperblockV2 &sb, uint32_t supported_mask) {
  if (sb.format_version == kQGFormatVersion) {
    // BLOCKER-3: the WHOLE 2C reserved region must be canonical-zero for a v2 base (not
    // just the first three fields). A forged v2 carrying a stray activation generation or
    // pid-generation summary in [72..408) would otherwise be accepted and then misroute
    // its own next reopen into the canonical lane ("self-locking" the file).
    return qg_reserved_range_is_zero(sb, kWal2cReservedOffset, sb.reserved.size());
  }
  if (sb.format_version == kQGFormatVersionV3) {
    if (qg_read_wal2c_magic(sb) != kWal2cMagic) return false;
    if (qg_read_wal2c_layout_version(sb) == 0 ||
        qg_read_wal2c_layout_version(sb) > kWal2cLayoutVersion) {
      return false;
    }
    const uint32_t required = qg_read_required_feature_flags(sb);
    if (!qg_required_features_self_consistent(required)) return false;
    // BLOCKER-5: a v3 base is only ever produced by an activation checkpoint, which always
    // carries the maintenance pair. A v3 image lacking it is forged; rejecting it here keeps
    // maintenance_activated_ true for every accepted v3 base (so a checkpoint never reverts
    // to a v2 image carrying stale 2C reserved bytes).
    if ((required & kQgFeatMaintenanceTxV1) == 0) return false;
    if ((required & ~supported_mask) != 0) return false;
    // BLOCKER-3: the 2C reserved TAIL beyond the defined state fields must be zero in v3 too.
    if (!qg_reserved_range_is_zero(sb,
                                   kWal2cReservedOffset + kWal2cStateBytes,
                                   sb.reserved.size())) {
      return false;
    }
    const uint64_t maint_gen = qg_read_maintenance_activation_gen(sb);
    const uint64_t pid_gen = qg_read_pid_reuse_activation_gen(sb);
    const bool pid = (required & kQgFeatPidGenerationV1) != 0;
    // BLOCKER-3 (leg-7): maintenance is REQUIRED for every v3 (checked above), so its
    // activation generation must be in (0, sb.generation] in the PURE validation phase.
    // The struct carries sb.generation, so the <= own-generation upper bound is enforced
    // HERE too -- a read-only QuantizedGraph open never calls the loader's adopt_label_state(),
    // so this selector is the only gate that stops a maintenance-gen-zero / activation-gen-future
    // v3 from being adopted as the highest-generation base and re-selected on every later open.
    if (maint_gen == 0 || maint_gen > sb.generation) return false;
    if (!pid) {
      // BLOCKER-3 inactive-state: a maintenance-only v3 carries NO pid-reuse state, so its
      // activation generation + reuse summary must be canonical-zero.
      if (pid_gen != 0 || qg_read_max_pid_generation(sb) != 0 ||
          qg_read_nonzero_pid_generation_count(sb) != 0) {
        return false;
      }
    } else {
      // BLOCKER-3 activation ordering: pid reuse activates at or after maintenance, both
      // stamp a non-zero generation, and neither is newer than the block's own generation.
      if (pid_gen == 0 || pid_gen < maint_gen || pid_gen > sb.generation) return false;
    }
    return true;
  }
  return false;
}

// Returns 0 (A), 1 (B), or -1 when neither copy is valid.
inline int select_qg_superblock(const char *header, QGSuperblockV2 &selected) {
  QGSuperblockV2 copies[kQGSuperblockCopies];
  std::memcpy(&copies[0], header, sizeof(QGSuperblockV2));
  std::memcpy(&copies[1], header + kQGSuperblockSize, sizeof(QGSuperblockV2));
  const bool valid_a = qg_superblock_valid(copies[0]);
  const bool valid_b = qg_superblock_valid(copies[1]);
  if (!valid_a && !valid_b) return -1;
  const int slot = valid_b && (!valid_a || copies[1].generation > copies[0].generation) ? 1 : 0;
  selected = copies[slot];
  return slot;
}

// Fail-closed A/B selector (codex B-2C-06). Returns the selected slot (0/1), -1
// when NEITHER copy is structurally valid, or -2 when the highest-generation
// structurally-valid copy is a newer format this build does NOT support (unknown
// required feature bits / unsupported layout version / broken feature deps) -- the
// whole file must then be rejected, never silently downgraded to an older
// supported slot. A merely checksum-corrupt newer slot is skippable (fall back to
// the older valid slot); a checksum-VALID but unsupported newer slot is not.
inline int select_qg_superblock_checked(const char *header,
                                        QGSuperblockV2 &selected,
                                        uint32_t supported_mask) {
  QGSuperblockV2 copies[kQGSuperblockCopies];
  std::memcpy(&copies[0], header, sizeof(QGSuperblockV2));
  std::memcpy(&copies[1], header + kQGSuperblockSize, sizeof(QGSuperblockV2));
  const bool valid_a = qg_superblock_valid(copies[0]);
  const bool valid_b = qg_superblock_valid(copies[1]);
  if (!valid_a && !valid_b) {
    return -1;
  }
  // Highest-generation structurally-valid copy (A wins ties, matching the legacy
  // selector so a torn same-generation B re-adopt stays on A).
  const int best = valid_b && (!valid_a || copies[1].generation > copies[0].generation) ? 1 : 0;
  if (!qg_superblock_supported(copies[best], supported_mask)) {
    return -2;
  }
  selected = copies[best];
  return best;
}

inline bool qg_header_has_v2_magic(const char *header) {
  uint64_t magic_a = 0;
  uint64_t magic_b = 0;
  std::memcpy(&magic_a, header, sizeof(magic_a));
  std::memcpy(&magic_b, header + kQGSuperblockSize, sizeof(magic_b));
  return magic_a == kQGSuperblockMagic || magic_b == kQGSuperblockMagic;
}

class QuantizedGraph {
  friend class QGBuilder;
  friend class QGUpdater;  // research prototype: streaming updates (qg_updater.hpp)

 private:
  size_t num_points_ = 0;    // num points
  size_t degree_bound_ = 0;  // degree bound
  size_t dimension_ = 0;     // dimension
  size_t residual_dimension_ = 0;
  size_t padded_dim_ = 0;  // padded dimension
  core::Metric metric_ = core::Metric::l2;
  core::MetricPreprocessing preprocessing_ = core::MetricPreprocessing::none;
  PID entry_point_ = 0;  // Entry point of graph

  // Dead field: grep confirms no other reference to `data_` in this class
  // (see REPORT-allocator-merge.md W1 audit). AlignedAlloc's 2MB tier is
  // THP-sufficient -- Linux's transparent-hugepage granule is 2MB, so the
  // previous 4MB alignment bought nothing over it -- replaced here anyway
  // for allocator-merge parity with every other call site in this file.
  data::Array<float, std::vector<size_t>, ::alaya::AlignedAlloc<float>> data_;
  QGScanner scanner_;
  FHTRotator rotator_;
  PCATransform pca_transform_;  // PCA transform for online query transformation
#if defined(_WIN32)
  std::unique_ptr<AlignedFileReader> aligned_file_reader_;
#else
  std::unique_ptr<storage::io::PageReader> page_reader_;
#endif

  // Paged ThreadData is graph-bound, so it cannot use the process-wide TLS
  // strategy used by the resident arena. The freelist is a lock-free MPMC
  // queue on the hot borrow/return path. A mutex is taken only when an empty
  // freelist has to grow to a new concurrent-query high-water mark.
  enum class PagedLeaseState : uint8_t { kClosed, kAccepting, kDraining };
  static constexpr uint64_t kPagedLeaseDrainBit = uint64_t{1} << 63U;
  static constexpr uint64_t kPagedLeaseCountMask = ~kPagedLeaseDrainBit;
  // make_laser_page_reader() configures both Linux backends at depth 128.
  // Bounding graph-bound leases by the same limit also bounds Windows
  // register_thread() calls if that unsupported port is enabled later.
  static constexpr size_t kPagedLeaseLimit = 128;

  class PagedThreadDataLease {
   public:
    PagedThreadDataLease() = default;
    PagedThreadDataLease(QuantizedGraph *owner, ThreadData *data) noexcept
        : owner_(owner), data_(data) {}
    PagedThreadDataLease(const PagedThreadDataLease &) = delete;
    auto operator=(const PagedThreadDataLease &) -> PagedThreadDataLease & = delete;
    PagedThreadDataLease(PagedThreadDataLease &&other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)),
          data_(std::exchange(other.data_, nullptr)),
          reusable_(std::exchange(other.reusable_, false)) {}
    auto operator=(PagedThreadDataLease &&) -> PagedThreadDataLease & = delete;
    ~PagedThreadDataLease();

    [[nodiscard]] auto data() const noexcept -> ThreadData & {
      assert(data_ != nullptr);
      return *data_;
    }
    void mark_reusable() noexcept { reusable_ = true; }

   private:
    QuantizedGraph *owner_ = nullptr;
    ThreadData *data_ = nullptr;
    bool reusable_ = false;
  };

  class PagedThreadDataFreelist {
   public:
    [[nodiscard]] auto enqueue(ThreadData *data) -> bool { return queue_.enqueue(data); }
    [[nodiscard]] auto try_dequeue(ThreadData *&data) -> bool { return queue_.try_dequeue(data); }

   private:
    moodycamel::ConcurrentQueue<ThreadData *> queue_{32};
  };

  PagedThreadDataFreelist thread_data_;
  std::vector<std::unique_ptr<ThreadData>> owned_thread_data_;
  std::mutex thread_data_create_mutex_;
  std::mutex thread_data_lifecycle_mutex_;
  std::mutex thread_data_drain_mutex_;
  std::condition_variable thread_data_drain_cv_;
  std::atomic<PagedLeaseState> paged_lease_state_{PagedLeaseState::kClosed};
  // One atomic word closes the admission gate and counts borrowers. This
  // avoids the check-then-increment race of separate state/count atomics:
  // drain's fetch_or either observes an earlier borrower or wins before its
  // CAS, so teardown can never miss a lease that later touches old resources.
  std::atomic<uint64_t> paged_lease_gate_{kPagedLeaseDrainBit};
  std::atomic<size_t> paged_lease_high_water_{0};
  std::atomic<size_t> paged_lease_quarantined_{0};
  int dc_count_;
  // Defaults read by the source-compatible single-query overloads below.
  // Production entries carry ef/beam per call and never read this state.
  std::atomic<size_t> compat_ef_search_{200};

  size_t node_len_;
  size_t page_size_;
  size_t node_per_page_;

  // workspace for disk-based quantized graph
  size_t min_beam_width_ = 2;
  std::atomic<size_t> compat_max_beam_width_{16};
  std::string index_file_name_;

  std::vector<PID> medoids_;
  std::vector<float> medoids_vector_;

  std::vector<ClusterStats> cluster_stats_;

  std::vector<PID> cache_ids_;
  // AlignedAlloc = 2MB-aligned + MADV_HUGEPAGE for large blocks. At 768d a row
  // spans 2-3 4K pages; without hugepages every pop pays TLB walks and the HW
  // streamer stops at each page boundary (memqg's StaticStorage already gets
  // this via the same allocator — parity is required for the arena kernel).
  std::vector<char, ::alaya::AlignedAlloc<char>> cache_nodes_;
  std::unordered_map<PID, char *> caches_;
  // Full-cache probe: true when the loaded cache covers every node in identity
  // order, so cache_nodes_ can be addressed as a resident arena (pid * node_len_).
  bool arena_identity_ = false;

  int query_time_ = 0;
  float total_io_time_ = 0;
  float total_io_time1_ = 0;
  int64_t total_cpu_time_ = 0;
  std::atomic<uint64_t> total_read_num_{0};
  float total_iter_num_ = 0;
  float total_cache_num_ = 0;
  int64_t total_ks_time_ = 0;
  float total_query_latency_ = 0;
  float total_n_hops_ = 0;

  std::vector<PID> mem_graph_enter_points_;

  // Optional tombstone filter (research prototype): ids in this set are
  // traversed for routing but excluded from search results. Owned by the
  // caller (see QGUpdater); must outlive searches. nullptr = no filtering.
  // Contract: this seam evolves into the single admission predicate
  // (user filter AND live bitmap) — docs/design/segment-admission-contract.md.
  // Do not add a second per-candidate filter mechanism beside it.
  const std::unordered_set<PID> *result_filter_ = nullptr;

  /*
   * Position of different data in each row
   *      RawData + QuantizationCodes + Factors + neighborIDs
   * Since we guarantee the degree for each vertex equals degree_bound (multiple of 32),
   * we do not need to store the degree for each vertex
   */
  size_t res_dim_offset_ = 0;
  size_t code_offset_ = 0;      // pos of packed code
  size_t factor_offset_ = 0;    // pos of Factor
  size_t neighbor_offset_ = 0;  // pos of Neighbors
  size_t row_offset_ = 0;       // length of entire row

  void initialize();
  void allocate_data();
  void init_workspace();

  [[nodiscard]] auto acquire_thread_data(size_t beam_width) -> PagedThreadDataLease;
  [[nodiscard]] auto make_thread_data(size_t beam_capacity) -> std::unique_ptr<ThreadData>;
  void release_thread_data(ThreadData *data, bool reusable) noexcept;
  void finish_thread_data_borrow() noexcept;
  void begin_thread_data_drain_locked();
  void shutdown_and_clear_thread_data_locked() noexcept;
  void rebuild_thread_data_locked(size_t seed_count, size_t beam_capacity);
  static void free_thread_data_storage(ThreadData &data) noexcept;

  // search on disk-based quantized graph
  void disk_search_qg(const float *ALAYA_RESTRICT query,
                      uint32_t knn,
                      uint32_t *ALAYA_RESTRICT results,
                      size_t ef_search,
                      size_t beam_width,
                      const RowAdmission *admission,
                      float *ALAYA_RESTRICT distances);

  void copy_vectors(const float *);

  // for beam search
  [[nodiscard]] uint64_t get_page_offset(uint64_t node_id) const {
    return kSectorLen + page_size_ * (node_id / node_per_page_);
  }

  [[nodiscard]] uint64_t offset_to_node(uint64_t node_id) const {
    return (node_id % node_per_page_) * node_len_;
  }

  [[nodiscard]] std::string gen_index_path(const char *prefix) const {
    return std::string(prefix) + "_R" + std::to_string(degree_bound_) + "_MD" +
           std::to_string(dimension_) + ".index";
  }

  void update_qg_out_of_memory(PID,
                               const std::vector<Candidate<float>> &,
                               AlignedFileReader &,
                               ThreadData);

  // pf_base/pf_lines: resident-arena candidate prefetch — after each pool
  // insert, prefetch the head of the current-best candidate's row (memqg
  // kernel parity). nullptr/0 (the disk path) keeps behavior unchanged.
  float scan_neighbors(const QGQuery &q_obj,
                       const float *cur_data,
                       float *appro_dist,
                       buffer::SearchBuffer &search_pool,
                       uint32_t cur_degree,
                       const HashBasedBooleanSet &visited,
                       const char *pf_base = nullptr,
                       size_t pf_lines = 0) const;

  [[nodiscard]] auto exact_distance(const float *lhs, const float *rhs, size_t dim) const -> float {
    return metric_ == core::Metric::l2 ? space::l2_sqr(lhs, rhs, dim) : space::ip(lhs, rhs, dim);
  }

 public:
  explicit QuantizedGraph(
      size_t num,
      size_t max_deg,
      size_t main_dim,
      size_t dim,
      uint64_t rotator_seed = 0,
      std::string rotator_dump_path = "",
      core::Metric metric = core::Metric::l2,
      core::MetricPreprocessing preprocessing = core::MetricPreprocessing::none);

  ~QuantizedGraph();

  [[nodiscard]] auto num_vertices() const { return this->num_points_; }

  [[nodiscard]] auto dimension() const { return this->dimension_; }

  [[nodiscard]] auto residual_dimension() const { return this->residual_dimension_; }

  [[nodiscard]] auto metric() const noexcept { return metric_; }

  [[nodiscard]] auto preprocessing() const noexcept { return preprocessing_; }

  [[nodiscard]] auto degree_bound() const { return this->degree_bound_; }

  [[nodiscard]] auto entry_point() const { return this->entry_point_; }

  void set_ep(PID entry) { this->entry_point_ = entry; }

  void set_result_filter(const std::unordered_set<PID> *filter) { result_filter_ = filter; }

  // recovery_mode skips the exact file_size superblock check: a crash between a
  // checkpoint's ftruncate and its superblock write can leave the file longer
  // than the selected superblock claims, and op-WAL replay reconciles the length
  // afterward (unified-wal-vocabulary.md clause C/E). Ordinary readers keep the
  // strict check.
  void load_disk_index(const char *, float, bool recovery_mode = false);

  void set_params(size_t ef_search, size_t num_threads, int beam_width);

  // Lifecycle observers used by concurrency/teardown tests. They expose only
  // atomic counters, never a graph-bound ThreadData pointer.
  [[nodiscard]] auto paged_leases_in_flight() const noexcept -> size_t {
    return static_cast<size_t>(paged_lease_gate_.load(std::memory_order_acquire) &
                               kPagedLeaseCountMask);
  }
  [[nodiscard]] auto paged_lease_high_water() const noexcept -> size_t {
    return paged_lease_high_water_.load(std::memory_order_acquire);
  }
  [[nodiscard]] auto paged_lease_quarantined() const noexcept -> size_t {
    return paged_lease_quarantined_.load(std::memory_order_acquire);
  }
  [[nodiscard]] auto paged_leases_accepting() const noexcept -> bool {
    return (paged_lease_gate_.load(std::memory_order_acquire) & kPagedLeaseDrainBit) == 0;
  }

  void load_medoids(const char *);

  void load_cache(std::string &cache_ids_file,
                  std::string &cache_nodes_file,
                  size_t online_cache_num);

  void load_cluster_stats(const char *filename);

  /* search and copy results to KNN */
  // Source-compatible overload: reads the ef/beam defaults last installed by
  // set_params(). New production callers pass both values explicitly.
  void search(const float *ALAYA_RESTRICT query,
              uint32_t knn,
              uint32_t *ALAYA_RESTRICT results,
              const RowAdmission *admission = nullptr,
              float *ALAYA_RESTRICT distances = nullptr);
  void search(const float *ALAYA_RESTRICT query,
              uint32_t knn,
              uint32_t *ALAYA_RESTRICT results,
              size_t ef_search,
              size_t beam_width,
              const RowAdmission *admission = nullptr,
              float *ALAYA_RESTRICT distances = nullptr);

  // Full-cache probe: same scan_neighbors kernel on a resident arena — direct
  // pid*node_len addressing over cache_nodes_, no beam/AIO orchestration.
  // The source-compatible overload reads the set_params() ef/beam defaults.
  void arena_search_qg(const float *ALAYA_RESTRICT query,
                       uint32_t knn,
                       uint32_t *ALAYA_RESTRICT results,
                       const RowAdmission *admission = nullptr,
                       float *ALAYA_RESTRICT distances = nullptr);
  void arena_search_qg(const float *ALAYA_RESTRICT query,
                       uint32_t knn,
                       uint32_t *ALAYA_RESTRICT results,
                       size_t ef_search,
                       size_t beam_width,
                       const RowAdmission *admission = nullptr,
                       float *ALAYA_RESTRICT distances = nullptr);
  void arena_search_with(ArenaScratch &scratch,
                         const float *ALAYA_RESTRICT query,
                         uint32_t knn,
                         uint32_t *ALAYA_RESTRICT results,
                         size_t ef_search,
                         const RowAdmission *admission = nullptr,
                         float *ALAYA_RESTRICT distances = nullptr);
  void arena_batch_search(const float *ALAYA_RESTRICT query,
                          uint32_t knn,
                          uint32_t *ALAYA_RESTRICT results,
                          size_t num_queries,
                          size_t ef_search,
                          size_t beam_width,
                          const RowAdmission *admission = nullptr,
                          float *ALAYA_RESTRICT distances = nullptr);

  // Residency seams (driven by the residency.hpp providers):
  //  - arena_resident(): true when cache_nodes_ is a full identity-ordered arena.
  //  - ensure_resident_arena(): materialize the arena straight from the index
  //    file when the cache sidecar didn't already provide one — residency is a
  //    load-time policy, not a build-time family choice. Not thread-safe
  //    against concurrent searches.
  //  - arena_reserve_rows()/arena_mirror_write(): QGUpdater seam — reserve
  //    capacity for appendable PIDs up front, then reflect committed page
  //    writes into the arena so resident searches observe updates
  //    (pass-through mirror; no seqlock, research-grade like the updater).
  [[nodiscard]] bool arena_resident() const noexcept { return arena_identity_; }
  void ensure_resident_arena();
  void arena_reserve_rows(size_t rows);
  void arena_mirror_write(uint64_t file_off, const char *buf, size_t len);

  // Batch is semantic sugar over reentrant per-query calls. These entries do
  // not create threads; callers that want concurrency supply external lanes.
  void batch_search(const float *ALAYA_RESTRICT query,
                    uint32_t knn,
                    uint32_t *ALAYA_RESTRICT results,
                    size_t num_queries,
                    size_t ef_search,
                    size_t beam_width,
                    const RowAdmission *admission = nullptr,
                    float *ALAYA_RESTRICT distances = nullptr);

  void destroy_thread_data();
};

inline QuantizedGraph::QuantizedGraph(size_t num,
                                      size_t max_deg,
                                      size_t main_dim,
                                      size_t dim,
                                      uint64_t rotator_seed,
                                      std::string rotator_dump_path,
                                      core::Metric metric,
                                      core::MetricPreprocessing preprocessing)
    : num_points_(num),
      degree_bound_(max_deg),
      dimension_(main_dim),
      residual_dimension_(dim - dimension_)  // Residual dimension for extended vector
                                             // representation (e.g., for GIST dataset)
      ,
      padded_dim_(1 << ceil_log2(dimension_)),
      metric_(metric),
      preprocessing_(preprocessing),
      scanner_(padded_dim_, degree_bound_),
      rotator_(dimension_, rotator_seed),
#if defined(_WIN32)
      aligned_file_reader_(make_aligned_file_reader()),
#endif
      node_len_((32 * dimension_ + 32 * residual_dimension_ + 128 * degree_bound_ +
                 degree_bound_ * padded_dim_) /
                8) {
  qg_validate_metric_configuration(metric_, preprocessing_);
  if (metric_ != core::Metric::l2 && residual_dimension_ != 0) {
    throw std::invalid_argument(
        "QuantizedGraph: IP/cosine require main_dim == dim (no L2 residual tail)");
  }
  // Dump the rotator's sign-scaled mat_ vector BEFORE any consumer
  // (RaBitQ training, search path) reads from it. When `rotator_dump_path`
  // is empty, no file is written and the on-disk dsqg.index is unchanged.
  if (!rotator_dump_path.empty()) {
    rotator_.dump_signs(rotator_dump_path);
  }

  if (node_len_ == 0) {
    throw std::invalid_argument("QuantizedGraph: node_len_ must be > 0");
  }
  const QGPageGeometry geometry = qg_page_geometry(node_len_);
  node_per_page_ = geometry.node_per_page;
  page_size_ = geometry.page_size;

  // FHT/RaBitQ/FastScan operate on padded_dim_; raw vectors and exact-distance
  // terms keep using dimension_. The rotator zero-fills [dimension_, padded_dim_),
  // so a non-power-of-two main dimension is a supported padded layout.
  if (padded_dim_ < dimension_ || padded_dim_ % 64 != 0) {
    throw std::invalid_argument(
        "QuantizedGraph: padded dimension must cover the main dimension and be a multiple of 64");
  }
  // The disk write path packs node_per_page_ nodes per page in id order, matching
  // get_page_offset() and offset_to_node().
  if (page_size_ % kSectorLen != 0) {
    throw std::invalid_argument("QuantizedGraph: page_size_ must be sector aligned");
  }
  if (node_per_page_ * node_len_ > page_size_) {
    throw std::invalid_argument("QuantizedGraph: node_per_page_ * node_len_ must be <= page_size_");
  }
  if (node_per_page_ != geometry.node_per_page || page_size_ != geometry.page_size) {
    throw std::invalid_argument("QuantizedGraph: page geometry mismatch");
  }
  assert(node_len_ > 0);
  assert(page_size_ % kSectorLen == 0);
  assert(node_per_page_ * node_len_ <= page_size_);
  assert(node_per_page_ == geometry.node_per_page);
  assert(page_size_ == geometry.page_size);
  std::cout << "main_dim: " << main_dim << ", dim: " << dim << ", dimension_: " << dimension_
            << ", residual_dimension_: " << residual_dimension_ << std::endl;
  initialize();
}

inline QuantizedGraph::PagedThreadDataLease::~PagedThreadDataLease() {
  if (owner_ != nullptr) {
    owner_->release_thread_data(data_, reusable_);
  }
}

inline void QuantizedGraph::free_thread_data_storage(ThreadData &data) noexcept {
  if (data.sector_scratch_ != nullptr) {
    memory::align_free(data.sector_scratch_);
    data.sector_scratch_ = nullptr;
  }
  data.sector_scratch_slots_ = 0;
}

inline auto QuantizedGraph::make_thread_data(size_t beam_capacity) -> std::unique_ptr<ThreadData> {
  if (beam_capacity == 0 || page_size_ == 0 ||
      beam_capacity > std::numeric_limits<size_t>::max() / 2 / page_size_) {
    throw std::invalid_argument("QuantizedGraph: paged lease beam capacity is invalid");
  }
  auto data = std::make_unique<ThreadData>();
#if defined(_WIN32)
  aligned_file_reader_->register_thread();
  data->ctx_ = aligned_file_reader_->get_ctx();
  data->sector_scratch_ =
      reinterpret_cast<char *>(memory::align_allocate<kSectorLen>(2 * beam_capacity * page_size_));
#else
  data->sector_scratch_ =
      static_cast<char *>(allocate_page_read_buffer(*page_reader_, 2 * beam_capacity * page_size_));
#endif
  data->sector_scratch_slots_ = 2 * beam_capacity;
  return data;
}

inline void QuantizedGraph::finish_thread_data_borrow() noexcept {
  const auto previous = paged_lease_gate_.fetch_sub(1, std::memory_order_acq_rel);
  assert((previous & kPagedLeaseCountMask) != 0);
  if ((previous & kPagedLeaseDrainBit) != 0 && (previous & kPagedLeaseCountMask) == 1) {
    // The drain waiter's predicate reads the gate word outside the mutex, so a
    // bare notify could fire between its predicate evaluating non-zero and the
    // wait blocking — a lost wakeup that hangs set_params()/teardown. The empty
    // critical section defers this notify past that window.
    {
      std::lock_guard<std::mutex> drain_guard(thread_data_drain_mutex_);
    }
    thread_data_drain_cv_.notify_all();
  }
}

inline void QuantizedGraph::release_thread_data(ThreadData *data, bool reusable) noexcept {
  assert(data != nullptr);
  if (reusable && thread_data_.enqueue(data)) {
    finish_thread_data_borrow();
    return;
  }
  // An exception can leave PageReader callbacks in flight. Keep that lease
  // owned but out of circulation until shutdown() has joined/drained the
  // backend; only then may its completion object and sector buffer be freed.
  paged_lease_quarantined_.fetch_add(1, std::memory_order_relaxed);
  finish_thread_data_borrow();
}

inline auto QuantizedGraph::acquire_thread_data(size_t beam_width) -> PagedThreadDataLease {
  if (beam_width == 0) {
    throw std::invalid_argument("QuantizedGraph::search: beam_width must be > 0");
  }
  auto gate = paged_lease_gate_.load(std::memory_order_acquire);
  for (;;) {
    if ((gate & kPagedLeaseDrainBit) != 0) {
      const auto state = paged_lease_state_.load(std::memory_order_acquire);
      if (state == PagedLeaseState::kClosed) {
        throw std::logic_error(
            "QuantizedGraph::search: paged lease pool is closed; call load_disk_index() first");
      }
      throw std::runtime_error(
          "QuantizedGraph::search: paged lease pool is draining for set_params; retry");
    }
    if ((gate & kPagedLeaseCountMask) == kPagedLeaseCountMask) {
      throw std::runtime_error("QuantizedGraph::search: paged lease counter overflow");
    }
    if (paged_lease_gate_.compare_exchange_weak(gate,
                                                gate + 1,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
      break;
    }
  }

  ThreadData *data = nullptr;
  if (!thread_data_.try_dequeue(data)) {
    try {
      std::lock_guard create_lock(thread_data_create_mutex_);
      // A lease may have returned while this borrower waited for another
      // high-water allocation. Retry before constructing graph-bound state.
      if (!thread_data_.try_dequeue(data)) {
        if (owned_thread_data_.size() >= kPagedLeaseLimit) {
          throw std::runtime_error(
              "QuantizedGraph::search: paged lease pool is saturated at the page-reader "
              "limit; retry");
        }
        auto created = make_thread_data(compat_max_beam_width_.load(std::memory_order_acquire));
        data = created.get();
        owned_thread_data_.push_back(std::move(created));
        paged_lease_high_water_.store(owned_thread_data_.size(), std::memory_order_release);
      }
    } catch (...) {
      finish_thread_data_borrow();
      throw;
    }
  }

  assert(data != nullptr);
  if (beam_width > data->sector_scratch_slots_ / 2) {
    release_thread_data(data, true);
    throw std::logic_error(
        "QuantizedGraph::search: per-call beam exceeds the paged lease capacity; "
        "call set_params() before concurrent search");
  }
  return PagedThreadDataLease(this, data);
}

inline void QuantizedGraph::begin_thread_data_drain_locked() {
  paged_lease_state_.store(PagedLeaseState::kDraining, std::memory_order_release);
  paged_lease_gate_.fetch_or(kPagedLeaseDrainBit, std::memory_order_acq_rel);
  std::unique_lock drain_lock(thread_data_drain_mutex_);
  thread_data_drain_cv_.wait(drain_lock, [this] {
    return (paged_lease_gate_.load(std::memory_order_acquire) & kPagedLeaseCountMask) == 0;
  });
}

inline void QuantizedGraph::shutdown_and_clear_thread_data_locked() noexcept {
  // Reader shutdown is deliberately first. The thread-pool backend joins its
  // workers here, and libaio drains its outstanding requests, so neither can
  // write a sector buffer or completion queue after the leases are freed.
#if defined(_WIN32)
  if (aligned_file_reader_) {
    aligned_file_reader_->close();
    aligned_file_reader_->deregister_all_threads();
  }
#else
  if (page_reader_) {
    page_reader_->shutdown();
    page_reader_.reset();
  }
#endif

  ThreadData *ignored = nullptr;
  while (thread_data_.try_dequeue(ignored)) {
  }
  for (auto &data : owned_thread_data_) {
    free_thread_data_storage(*data);
  }
  owned_thread_data_.clear();
  paged_lease_high_water_.store(0, std::memory_order_release);
  paged_lease_quarantined_.store(0, std::memory_order_release);
}

inline void QuantizedGraph::rebuild_thread_data_locked(size_t seed_count, size_t beam_capacity) {
  assert((paged_lease_gate_.load(std::memory_order_acquire) & kPagedLeaseCountMask) == 0);
  try {
#if defined(_WIN32)
    aligned_file_reader_->open(index_file_name_);
#else
    page_reader_ = make_laser_page_reader(index_file_name_);
#endif
    for (size_t index = 0; index < seed_count; ++index) {
      auto data = make_thread_data(beam_capacity);
      auto *raw = data.get();
      owned_thread_data_.push_back(std::move(data));
      if (!thread_data_.enqueue(raw)) {
        throw std::bad_alloc();
      }
    }
    paged_lease_high_water_.store(owned_thread_data_.size(), std::memory_order_release);
    paged_lease_state_.store(PagedLeaseState::kAccepting, std::memory_order_release);
    paged_lease_gate_.store(0, std::memory_order_release);
  } catch (...) {
    shutdown_and_clear_thread_data_locked();
    paged_lease_state_.store(PagedLeaseState::kClosed, std::memory_order_release);
    throw;
  }
}

inline void QuantizedGraph::destroy_thread_data() {
  std::lock_guard lifecycle_lock(thread_data_lifecycle_mutex_);
  begin_thread_data_drain_locked();
  shutdown_and_clear_thread_data_locked();
  paged_lease_state_.store(PagedLeaseState::kClosed, std::memory_order_release);
}

inline QuantizedGraph::~QuantizedGraph() { destroy_thread_data(); }

inline void QuantizedGraph::set_params(size_t ef_search, size_t num_threads, int beam_width) {
  // Compatibility shell and paged-pool provisioning hook. Production search
  // carries ef/beam per call; num_threads seeds the elastic paged pool and
  // beam_width fixes each seeded lease's capacity.
  if (index_file_name_.empty()) {
    throw std::logic_error("QuantizedGraph::set_params: call load_disk_index() first");
  }
  if (ef_search == 0 || num_threads == 0 || num_threads > kPagedLeaseLimit || beam_width <= 0) {
    throw std::invalid_argument(
        "QuantizedGraph::set_params: effort/beam must be > 0 and threads must not exceed the "
        "page-reader lease limit");
  }

  std::lock_guard lifecycle_lock(thread_data_lifecycle_mutex_);
  begin_thread_data_drain_locked();
  shutdown_and_clear_thread_data_locked();
  compat_ef_search_.store(ef_search, std::memory_order_release);
  compat_max_beam_width_.store(static_cast<size_t>(beam_width), std::memory_order_release);
  rebuild_thread_data_locked(num_threads, static_cast<size_t>(beam_width));
}

/*
 * search single query
 */
inline void QuantizedGraph::search(const float *ALAYA_RESTRICT query,
                                   uint32_t knn,
                                   uint32_t *ALAYA_RESTRICT results,
                                   const RowAdmission *admission,
                                   float *ALAYA_RESTRICT distances) {
  search(query,
         knn,
         results,
         compat_ef_search_.load(std::memory_order_acquire),
         compat_max_beam_width_.load(std::memory_order_acquire),
         admission,
         distances);
}

inline void QuantizedGraph::search(const float *ALAYA_RESTRICT query,
                                   uint32_t knn,
                                   uint32_t *ALAYA_RESTRICT results,
                                   size_t ef_search,
                                   size_t beam_width,
                                   const RowAdmission *admission,
                                   float *ALAYA_RESTRICT distances) {
  disk_search_qg(query, knn, results, ef_search, beam_width, admission, distances);
}

inline void QuantizedGraph::batch_search(const float *ALAYA_RESTRICT query,
                                         uint32_t knn,
                                         uint32_t *ALAYA_RESTRICT results,
                                         size_t num_queries,
                                         size_t ef_search,
                                         size_t beam_width,
                                         const RowAdmission *admission,
                                         float *ALAYA_RESTRICT distances) {
  for (size_t i = 0; i < num_queries; ++i) {
    disk_search_qg(query + i * (dimension_ + residual_dimension_),
                   knn,
                   results + i * knn,
                   ef_search,
                   beam_width,
                   admission,
                   distances == nullptr ? nullptr : distances + i * knn);
  }
}

inline void QuantizedGraph::arena_search_qg(const float *ALAYA_RESTRICT query,
                                            uint32_t knn,
                                            uint32_t *ALAYA_RESTRICT results,
                                            const RowAdmission *admission,
                                            float *ALAYA_RESTRICT distances) {
  arena_search_qg(query,
                  knn,
                  results,
                  compat_ef_search_,
                  compat_max_beam_width_,
                  admission,
                  distances);
}

inline void QuantizedGraph::arena_search_qg(const float *ALAYA_RESTRICT query,
                                            uint32_t knn,
                                            uint32_t *ALAYA_RESTRICT results,
                                            size_t ef_search,
                                            size_t /*beam_width*/,
                                            const RowAdmission *admission,
                                            float *ALAYA_RESTRICT distances) {
  static thread_local ArenaScratch scratch;
  arena_search_with(scratch, query, knn, results, ef_search, admission, distances);
}

inline void QuantizedGraph::arena_search_with(ArenaScratch &scratch,
                                              const float *ALAYA_RESTRICT query,
                                              uint32_t knn,
                                              uint32_t *ALAYA_RESTRICT results,
                                              size_t ef_search,
                                              const RowAdmission *admission,
                                              float *ALAYA_RESTRICT distances) {
  if (!arena_identity_) {
    throw std::runtime_error(
        "arena_search_qg: requires a 100% identity-ordered node cache sidecar");
  }
  scratch.ensure(num_points_, ef_search, dimension_ + residual_dimension_);

  ALAYA_KSP_COUNT(queries);
  ALAYA_KSP_BEGIN(prep);
  const float *transformed_query = query;
  if (pca_transform_.is_loaded()) {
    pca_transform_.transform(query, scratch.pca_query_scratch_.data());
    transformed_query = scratch.pca_query_scratch_.data();
  }

  QGQuery q_obj(transformed_query, padded_dim_);
  q_obj.query_prepare(rotator_, scanner_);
  const float *residual_query = transformed_query + dimension_;
  float sqr_qr = 0;
  for (size_t i = 0; i < residual_dimension_; ++i) {
    sqr_qr += residual_query[i] * residual_query[i];
  }
  q_obj.set_sqr_qr(sqr_qr);

  if (!medoids_.empty()) {
    PID best_medoid = 0;
    float best_dist = FLT_MAX;
    for (size_t cur_m = 0; cur_m < medoids_.size(); cur_m++) {
      float cur_expanded_dist =
          exact_distance(transformed_query,
                         medoids_vector_.data() + (dimension_ + residual_dimension_) * cur_m,
                         dimension_);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids_[cur_m];
        best_dist = cur_expanded_dist;
      }
    }
    scratch.search_pool_.insert(best_medoid, FLT_MAX);
  }
  scratch.search_pool_.insert(entry_point_, FLT_MAX);
  ALAYA_KSP_END(prep);

  buffer::ResultBuffer res_pool(knn);
  std::vector<float> appro_dist(degree_bound_);
  const char *arena = cache_nodes_.data();
  const size_t pf_lines = arena_prefetch_lines((node_len_ + 63) / 64);
  const char *pf_base = pf_lines > 0 ? arena : nullptr;
  if (pf_base != nullptr) {
    prefetch_row_l1(arena + static_cast<size_t>(entry_point_) * node_len_, pf_lines);
  }

  while (scratch.search_pool_.has_next()) {
    const PID cur_node = scratch.search_pool_.pop();
    if (scratch.visited_.get(cur_node)) {
      continue;
    }
    scratch.visited_.set(cur_node);
    const auto *cur_data =
        reinterpret_cast<const float *>(arena + static_cast<size_t>(cur_node) * node_len_);
    float sqr_y = scan_neighbors(q_obj,
                                 cur_data,
                                 appro_dist.data(),
                                 scratch.search_pool_,
                                 this->degree_bound_,
                                 scratch.visited_,
                                 pf_base,
                                 pf_lines);
    if (residual_dimension_ > 0) {
      sqr_y += exact_distance(cur_data + dimension_, residual_query, residual_dimension_);
    }
    const bool admit = admission != nullptr
                           ? admission->test(cur_node)
                           : (result_filter_ == nullptr ||
                              result_filter_->find(cur_node) == result_filter_->end());
    if (admit) {
      res_pool.insert(cur_node, sqr_y);
    }
  }

  res_pool.copy_results(results, distances);
}

inline void QuantizedGraph::arena_batch_search(const float *ALAYA_RESTRICT query,
                                               uint32_t knn,
                                               uint32_t *ALAYA_RESTRICT results,
                                               size_t num_queries,
                                               size_t ef_search,
                                               size_t beam_width,
                                               const RowAdmission *admission,
                                               float *ALAYA_RESTRICT distances) {
  for (size_t i = 0; i < num_queries; ++i) {
    arena_search_qg(query + i * (dimension_ + residual_dimension_),
                    knn,
                    results + i * knn,
                    ef_search,
                    beam_width,
                    admission,
                    distances == nullptr ? nullptr : distances + i * knn);
  }
}

/**
 * @brief Performs k-nearest neighbor search on a disk-based quantized graph using asynchronous I/O.
 *
 * This function implements a beam search algorithm optimized for disk-resident graph indices.
 * It uses Linux AIO (Asynchronous I/O) to overlap disk reads with computation, achieving
 * high throughput by processing multiple nodes in parallel.
 *
 * Algorithm Overview:
 * 1. Initialize search with entry point(s) - either medoids (cluster centers) or global entry point
 * 2. Iteratively expand the search frontier using beam search with adaptive beam width
 * 3. For each candidate node:
 *    - If cached in memory: process immediately without disk I/O
 *    - If on disk: submit async read request and process when data arrives
 * 4. Use RaBitQ (Randomized Bit Quantization) for fast approximate distance computation
 * 5. Maintain a result pool to track the k nearest neighbors found so far
 *
 * Key Optimizations:
 * - Asynchronous I/O: Overlaps disk reads with CPU computation
 * - Adaptive beam width: Starts small and grows exponentially up to the
 *   per-call beam_width
 * - In-memory caching: Frequently accessed nodes are cached to avoid repeated disk reads
 * - Pipelined processing: Processes nodes from previous iteration while waiting for new I/O
 *
 * @param query     Pointer to the query vector (unrotated). The vector should have
 *                  (dimension_ + residual_dimension_) float elements. The first dimension_
 *                  elements are the main vector, and the remaining are residual components.
 * @param knn       Number of nearest neighbors to retrieve.
 * @param results   Output array to store the IDs of k nearest neighbors. Must have
 *                  space for at least knn uint32_t elements.
 *
 * @note This function is thread-safe. Each call leases graph-bound ThreadData from an
 *       elastic per-graph freelist, including scratch buffers and completion state.
 * @note The query vector is internally rotated using Fast Hadamard Transform for
 *       compatibility with the RaBitQ quantization scheme.
 */
inline void QuantizedGraph::disk_search_qg(const float *ALAYA_RESTRICT query,
                                           uint32_t knn,
                                           uint32_t *ALAYA_RESTRICT results,
                                           size_t ef_search,
                                           size_t beam_width,
                                           const RowAdmission *admission,
                                           float *ALAYA_RESTRICT distances) {
  auto lease = acquire_thread_data(beam_width);
  ThreadData &data = lease.data();
  data.search_scratch_.ensure(num_points_, ef_search, dimension_ + residual_dimension_);

  // ==================== PCA Transform ====================
  // Transform the original query using PCA for dimension reordering.
  // After transformation, high-variance dimensions are placed first.
  const float *transformed_query = query;
  if (pca_transform_.is_loaded()) {
    pca_transform_.transform(query, data.search_scratch_.pca_query_scratch_.data());
    transformed_query = data.search_scratch_.pca_query_scratch_.data();
  }

  // Performance timing variables (for profiling purposes)
  auto query_start = std::chrono::high_resolution_clock::now();
  int64_t submit_time = 0;
  int64_t wait_time = 0;
  int64_t process_time = 0;
  float n_hops = 0;  // Number of I/O rounds (disk access iterations)

  // ==================== Query Preparation ====================
  // Create query object and apply Fast Hadamard Transform rotation.
  // This rotation aligns the query with the quantized representation used in RaBitQ.
  QGQuery q_obj(transformed_query, padded_dim_);
  q_obj.query_prepare(rotator_, scanner_);

  // Pointer to residual query components (used for datasets like GIST with extended dimensions)
  const float *residual_query = transformed_query + dimension_;

  // Compute ||q_r||^2 for residual dimensions to improve approximate distance precision
  float sqr_qr = 0;
  for (size_t i = 0; i < residual_dimension_; ++i) {
    sqr_qr += residual_query[i] * residual_query[i];
  }
  q_obj.set_sqr_qr(sqr_qr);

  // ==================== Search Pool Initialization ====================
  // Initialize the search frontier with starting points.
  // If medoids (cluster centers) are available, find the closest one to the query
  // and use it as an additional entry point for better search quality.
  if (!medoids_.empty()) {
    PID best_medoid = 0;
    float best_dist = FLT_MAX;
    size_t best_medoid_idx = 0;
    // Linear scan through medoids to find the closest one
    for (size_t cur_m = 0; cur_m < medoids_.size(); cur_m++) {
      float cur_expanded_dist =
          exact_distance(transformed_query,
                         medoids_vector_.data() + (dimension_ + residual_dimension_) * cur_m,
                         dimension_);
      if (cur_expanded_dist < best_dist) {
        best_medoid = medoids_[cur_m];
        best_dist = cur_expanded_dist;
        best_medoid_idx = cur_m;
      }
    }
    // Insert best medoid with max distance (distance will be computed when visited)
    data.search_scratch_.search_pool_.insert(best_medoid, FLT_MAX);
  }
  // Always include the global entry point as a starting position
  data.search_scratch_.search_pool_.insert(entry_point_, FLT_MAX);

  // ==================== Result and Distance Buffers ====================
  // Result pool maintains the top-k nearest neighbors found during search
  buffer::ResultBuffer res_pool(knn);

  // Buffer for storing approximate distances computed via RaBitQ fast scan.
  // RaBitQ computes distances to all neighbors of a node in a single SIMD-optimized pass.
  std::vector<float> appro_dist(degree_bound_);

  // ==================== Asynchronous I/O Data Structures ====================
  // frontier_read_reqs: Batch of aligned read requests to submit to AIO
#if defined(_WIN32)
  std::vector<AlignedRead> frontier_read_reqs;
#else
  std::vector<storage::io::ReadRequest> frontier_read_reqs;
#endif
  frontier_read_reqs.reserve(2 * beam_width);

  // issued_nodes preserves the logical frontier order across asynchronous I/O.
  // Completions only flip read_ready; processing always consumes this FIFO.
  std::deque<std::pair<PID, char *>> issued_nodes;
  std::unordered_map<PID, bool> read_ready;

  // free_slots: Pool of available memory buffers for disk reads (double-buffering scheme)
  std::deque<char *> free_slots;
  for (size_t i = 0; i < 2 * beam_width; i++) {
    free_slots.push_back(data.sector_scratch_ + i * page_size_);
  }

  // Backend-independent event buffer for collecting completed I/O operations.
#if defined(_WIN32)
  std::vector<AlignedReadEvent> evts;
#else
  std::vector<storage::io::ReadResult> evts;
#endif

  // cache_nhoods: Nodes found in memory cache (no disk I/O needed)
  std::vector<PID> cache_nhoods;

  // Adaptive beam width: starts at 1 and grows exponentially
  size_t cur_beam_size = 1;

  // ==================== Node Processing Lambda ====================
  // This lambda processes a single node: computes exact distance to query,
  // scans neighbors using RaBitQ approximation, and updates search/result pools.
  auto process_node = [&](PID cur_node, float *cur_data) {
    // Scan neighbors and compute approximate distances using RaBitQ.
    // Also computes exact L2 distance from query to current node.
    float sqr_y = scan_neighbors(q_obj,
                                 cur_data,
                                 appro_dist.data(),
                                 data.search_scratch_.search_pool_,
                                 this->degree_bound_,
                                 data.search_scratch_.visited_);
    // Add residual dimension distance if applicable (e.g., for GIST dataset)
    if (residual_dimension_ > 0) {
      float *residual_data = cur_data + dimension_;
      sqr_y += exact_distance(reinterpret_cast<const float *>(residual_data),
                              residual_query,
                              residual_dimension_);
    }
    // Insert current node with exact distance into result pool (unless the
    // node is filtered out: tombstoned, or excluded by a per-call admission
    // predicate. Routing still passes through it either way.)
    const bool admit = admission != nullptr
                           ? admission->test(cur_node)
                           : (result_filter_ == nullptr ||
                              result_filter_->find(cur_node) == result_filter_->end());
    if (admit) {
      res_pool.insert(cur_node, sqr_y);
    }
  };

  // ==================== I/O Completion Handler Lambda ====================
  // Collects completed I/O events and marks their issued nodes ready.
  // Uses non-blocking reader polling to check for completed I/O without waiting.
  auto wait_for_nodes = [&]() {
#if defined(_WIN32)
    const auto ret = static_cast<std::size_t>(
        aligned_file_reader_->poll_events(data.ctx_, static_cast<int>(cur_beam_size), evts));
#else
    const auto ret = poll_page_reads(*data.completions_, cur_beam_size, evts);
#endif

    // Process each completed I/O event
    for (std::size_t i = 0; i < ret; i++) {
      const auto id = static_cast<PID>(evts[i].id);
      const auto state = read_ready.find(id);
      if (state == read_ready.end() || state->second) {
        throw std::runtime_error(
            "disk_search_qg: I/O completion id is unknown or duplicated "
            "(page reader returned an unexpected id)");
      }
      state->second = true;
    }
  };

  // Completion timing is deliberately not a search input. A later request may
  // finish first, but its row stays buffered until every earlier issued row has
  // been processed. This retains batched asynchronous reads and the existing
  // half-batch pipeline while making frontier mutation deterministic.
  auto process_next_issued = [&]() -> bool {
    if (issued_nodes.empty()) {
      throw std::runtime_error("disk_search_qg: issued-node accounting underflow");
    }
    const auto state = read_ready.find(issued_nodes.front().first);
    if (state == read_ready.end()) {
      throw std::runtime_error("disk_search_qg: issued node has no read state");
    }
    if (!state->second) return false;

    const auto node = issued_nodes.front();
    issued_nodes.pop_front();
    read_ready.erase(state);
    process_node(node.first, reinterpret_cast<float *>(node.second + offset_to_node(node.first)));
    free_slots.push_back(node.second);
    return true;
  };

  // Track remaining nodes from previous iteration for pipelined processing
  size_t previous_remain_num = 0;

  // Total I/O operations counter (for profiling)
  int64_t io_num = 0;

  // ==================== Main Search Loop ====================
  // Iteratively expand search frontier until no more candidates remain.
  // Uses beam search with adaptive width and pipelined I/O processing.
  while (data.search_scratch_.search_pool_.has_next()) {
    frontier_read_reqs.clear();
    cache_nhoods.clear();
    size_t n_ops = 0;
    size_t need_process_num = 0;
    size_t remain_num = 0;

    // Adaptive beam width: double the beam size each iteration (up to max)
    // This helps balance between exploration breadth and I/O efficiency.
    cur_beam_size =
        std::min(beam_width, static_cast<size_t>(std::ceil(2 * static_cast<float>(cur_beam_size))));

    auto wait_start = std::chrono::high_resolution_clock::now();

    // -------------------- Build I/O Request Batch --------------------
    // Pop candidates from search pool and prepare I/O requests for non-cached nodes.
    while (data.search_scratch_.search_pool_.has_next() &&
           frontier_read_reqs.size() < cur_beam_size) {
      PID cur_node = data.search_scratch_.search_pool_.pop();

      // Skip already visited nodes to avoid redundant processing
      if (data.search_scratch_.visited_.get(cur_node)) {
        continue;
      }
      data.search_scratch_.visited_.set(cur_node);

      // Check if node is in memory cache
      if (caches_.find(cur_node) != caches_.end()) {
        // Cache hit: add to cache_nhoods for immediate processing
        cache_nhoods.push_back(cur_node);
      } else {
        // Cache miss: need to read from disk
        if (free_slots.empty()) {
          throw std::runtime_error("QuantizedGraph::search: free_buffer pool exhausted");
        }
        // Allocate a buffer slot for this read
        char *slot = free_slots.front();
        assert(slot != nullptr);
        free_slots.pop_front();
        const bool inserted = read_ready.emplace(cur_node, false).second;
        if (!inserted) {
          throw std::runtime_error("disk_search_qg: duplicate issued node");
        }
        issued_nodes.emplace_back(cur_node, slot);
        // Create aligned read request (page-aligned for direct I/O)
#if defined(_WIN32)
        frontier_read_reqs.emplace_back(get_page_offset(cur_node), page_size_, cur_node, slot);
#else
        frontier_read_reqs.push_back(
            {.id = cur_node,
             .offset = get_page_offset(cur_node),
             .buffer = std::span(reinterpret_cast<std::byte *>(slot), page_size_)});
#endif
      }
      total_read_num_.fetch_add(1, std::memory_order_relaxed);
    }

    // -------------------- Submit Async I/O Requests --------------------
    // Submit batch of read requests to the AIO subsystem
    if (!frontier_read_reqs.empty()) {
      n_hops++;  // Count as one I/O round
      auto submit_start = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
      n_ops = aligned_file_reader_->submit_reqs(frontier_read_reqs, data.ctx_);
#else
      n_ops = submit_page_reads(*page_reader_, frontier_read_reqs, *data.completions_);
#endif
      io_num += n_ops;
      auto submit_end = std::chrono::high_resolution_clock::now();
      submit_time +=
          std::chrono::duration_cast<std::chrono::microseconds>(submit_end - submit_start).count();
    }

    // -------------------- Process Cached Nodes --------------------
    // Process cached nodes (these are stored in memory and don't require disk I/O)
    for (auto &cache_id : cache_nhoods) {
      auto *cur_data = caches_.at(cache_id);
      process_node(cache_id, reinterpret_cast<float *>(cur_data));
    }

    // -------------------- Pipelined Processing --------------------
    // Calculate how many nodes to process in this iteration.
    // We process half of new I/O ops plus leftovers from previous iteration,
    // allowing I/O and computation to overlap (pipelining).
    remain_num = 0.5 * n_ops;
    need_process_num = n_ops + previous_remain_num - remain_num;
    previous_remain_num = remain_num;

    // Process issued nodes (from previous or current iteration) in frontier order.
    while (need_process_num > 0) {
      if (process_next_issued()) {
        need_process_num--;
      } else {
        // The oldest issued node is not ready; collect more completions.
        wait_for_nodes();
      }
    }
  }

  // ==================== Process Remaining Nodes ====================
  // After the main loop exits, there may still be nodes in the pipeline
  // that haven't been processed yet. Drain the remaining nodes.
  while (previous_remain_num > 0) {
    if (process_next_issued()) {
      previous_remain_num--;
    } else {
      // Wait for the oldest remaining issued node to complete.
      wait_for_nodes();
    }
  }
  assert(issued_nodes.empty());
  assert(read_ready.empty());

  // Record query end time for latency measurement
  auto query_end = std::chrono::high_resolution_clock::now();

  auto latency =
      std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start).count();

  // ==================== Copy Results and Cleanup ====================
  // Copy the k nearest neighbor IDs from result pool to output array
  res_pool.copy_results(results, distances);

  lease.mark_reusable();
}

// scan a data row (including data vec and quantization codes for its neighbors)
// return exact distance for current vertex
inline float QuantizedGraph::scan_neighbors(const QGQuery &q_obj,
                                            const float *cur_data,
                                            float *appro_dist,
                                            buffer::SearchBuffer &search_pool,
                                            uint32_t cur_degree,
                                            const HashBasedBooleanSet &visited,
                                            const char *pf_base,
                                            size_t pf_lines) const {
  ALAYA_KSP_COUNT(pops);
  ALAYA_KSP_BEGIN(exact);
  float sqr_y = exact_distance(q_obj.query_data(), cur_data, dimension_);
  ALAYA_KSP_END(exact);

  /* Compute approximate distance by Fast Scan */
  const auto *packed_code = reinterpret_cast<const uint8_t *>(&cur_data[code_offset_]);
  const auto *factor = &cur_data[factor_offset_];
  ALAYA_KSP_BEGIN(scan);
  this->scanner_.scan_neighbors(appro_dist,
                                q_obj.lut().data(),
                                sqr_y,
                                q_obj.lower_val(),
                                q_obj.width(),
                                q_obj.sqr_qr(),
                                q_obj.sumq(),
                                packed_code,
                                factor);
  ALAYA_KSP_END(scan);

  ALAYA_KSP_BEGIN(pool);
  const PID *ptr_nb = reinterpret_cast<const PID *>(&cur_data[neighbor_offset_]);
  for (uint32_t i = 0; i < cur_degree; ++i) {
    PID cur_neighbor = ptr_nb[i];
    float tmp_dist = appro_dist[i];
    if (search_pool.is_full(tmp_dist) || visited.get(cur_neighbor)) {
      continue;
    }
    search_pool.insert(cur_neighbor, tmp_dist);
    if (pf_base != nullptr) {
      prefetch_row_l2(pf_base + static_cast<size_t>(search_pool.next_id()) * node_len_, pf_lines);
    }
  }
  ALAYA_KSP_END(pool);

  return sqr_y;
}

inline void QuantizedGraph::initialize() {
  /* check size */
  assert(padded_dim_ % 64 == 0);
  assert(padded_dim_ >= dimension_);

  this->res_dim_offset_ = dimension_;
  this->code_offset_ = dimension_ + residual_dimension_;  // Pos of packed code (aligned)
  this->factor_offset_ = code_offset_ + padded_dim_ / 64 * 2 * degree_bound_;  // Pos of Factor
  this->neighbor_offset_ = factor_offset_ + sizeof(Factor) * degree_bound_ / sizeof(float);
  this->row_offset_ = neighbor_offset_ + degree_bound_;
}

inline void QuantizedGraph::init_workspace() {
  std::lock_guard lifecycle_lock(thread_data_lifecycle_mutex_);
  begin_thread_data_drain_locked();
  shutdown_and_clear_thread_data_locked();
  rebuild_thread_data_locked(/*seed_count=*/1,
                             compat_max_beam_width_.load(std::memory_order_acquire));
}

inline void QuantizedGraph::update_qg_out_of_memory(
    PID cur_id,
    const std::vector<Candidate<float>> &new_neighbors,
    AlignedFileReader &vector_reader,
    ThreadData thread_data) {
  size_t cur_degree = new_neighbors.size();
  std::memset(thread_data.cur_page_scratch_, 0, page_size_);
  if (cur_degree == 0) {
    return;
  }
  char *vector_buf = thread_data.neighbor_vector_scratch_;
  char *page_buf = thread_data.cur_page_scratch_;

  size_t full_page_size = ((dimension_ + residual_dimension_) * sizeof(float) + kSectorLen - 1) /
                          kSectorLen * kSectorLen;
  size_t main_page_size = (dimension_ * sizeof(float) + kSectorLen - 1) / kSectorLen * kSectorLen;

  kernels::linalg::RowMajorMatrix<float> x_pad(cur_degree, padded_dim_);  // padded neighbors mat
  kernels::linalg::RowMajorMatrix<float> c_pad(1, padded_dim_);  // padded duplicate centroid mat
  x_pad.setZero();
  c_pad.setZero();

  std::vector<AlignedRead> frontier_read_reqs;
  frontier_read_reqs.reserve(cur_degree + 1);

  // Create read requests for each neighbor node's vector data from disk
  for (size_t i = 0; i < cur_degree; ++i) {
    auto neighbor_id = new_neighbors[i].id;
    uint64_t offset = neighbor_id * full_page_size;
    uint64_t len = full_page_size;
    void *buf = reinterpret_cast<void *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    frontier_read_reqs.emplace_back(offset, len, neighbor_id, buf);
  }

  // Create read request for the current node (centroid) vector data
  uint64_t cur_offset = cur_id * full_page_size;
  uint64_t cur_len = full_page_size;
  void *cur_buf = reinterpret_cast<void *>(reinterpret_cast<char *>(page_buf));
  frontier_read_reqs.emplace_back(cur_offset, cur_len, cur_id, cur_buf);

  // Execute batch read operation to fetch all neighbor vectors and centroid in one I/O call
  vector_reader.read(frontier_read_reqs, thread_data.ctx_);

  // Write neighbor IDs into the node payload AFTER the read, since the read
  // request above lands `full_page_size` bytes at `page_buf[0..full_page_size]`
  // and would clobber any write made before it. Neighbor IDs live at
  // `neighbor_offset_ * sizeof(float)` within the per-node payload.
  PID *neighbor_ptr = reinterpret_cast<PID *>(page_buf + neighbor_offset_ * 4);
  for (size_t i = 0; i < cur_degree; ++i) {
    neighbor_ptr[i] = new_neighbors[i].id;
  }

  /* Copy data */
  for (size_t i = 0; i < cur_degree; ++i) {
    auto neighbor_id = new_neighbors[i].id;
    const auto *cur_data =
        reinterpret_cast<const float *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    std::copy(cur_data, cur_data + dimension_, &x_pad(static_cast<int64_t>(i), 0));
  }
  const auto *cur_cent = reinterpret_cast<const float *>(reinterpret_cast<char *>(page_buf));
  std::copy(cur_cent, cur_cent + dimension_, &c_pad(0, 0));

  /* rotate Matrix */
  kernels::linalg::RowMajorMatrix<float> x_rotated(cur_degree, padded_dim_);
  kernels::linalg::RowMajorMatrix<float> c_rotated(1, padded_dim_);
  for (int64_t i = 0; i < static_cast<int64_t>(cur_degree); ++i) {
    this->rotator_.rotate(&x_pad(i, 0), &x_rotated(i, 0));
  }
  this->rotator_.rotate(&c_pad(0, 0), &c_rotated(0, 0));

  // Get codes and factors for rabitq
  auto *fac_ptr = reinterpret_cast<float *>(page_buf + 4 * factor_offset_);
  auto *packed_code_ptr = reinterpret_cast<uint8_t *>(page_buf + 4 * code_offset_);
  float *triple_x = fac_ptr;
  float *factor_dq = triple_x + this->degree_bound_;
  float *factor_vq = factor_dq + this->degree_bound_;
  rabitq_codes(x_rotated, c_rotated, packed_code_ptr, triple_x, factor_dq, factor_vq, metric_);

  // Add ||x_r||^2 (residual dimensions) directly to triple_x for improved precision
  // This avoids storing a separate sqr_xr array and saves computation during search
  for (size_t i = 0; i < cur_degree; ++i) {
    const auto *neighbor_data =
        reinterpret_cast<const float *>(reinterpret_cast<char *>(vector_buf) + i * full_page_size);
    const float *residual_data = neighbor_data + dimension_;
    float sqr_xr_val = 0;
    for (size_t j = 0; j < residual_dimension_; ++j) {
      sqr_xr_val += residual_data[j] * residual_data[j];
    }
    triple_x[i] += sqr_xr_val;
  }
}

inline void QuantizedGraph::load_disk_index(const char *filename,
                                            float search_DRAM_budget,
                                            bool recovery_mode) {
  index_file_name_ = gen_index_path(filename);
  if (!std::filesystem::exists(index_file_name_)) {
    throw std::runtime_error("QuantizedGraph::load_disk_index: file not found: " +
                             index_file_name_);
  }
  std::ifstream input(index_file_name_, std::ios::binary);
  assert(input.is_open());

  std::array<char, kSectorLen> header{};
  input.read(header.data(), header.size());
  if (!input) {
    throw std::runtime_error("QuantizedGraph::load_disk_index: short metadata sector");
  }
  qg_validate_native_semantics(header.data(), header.size(), metric_, preprocessing_);

  QGSuperblockV2 sb;
  const int sb_slot = select_qg_superblock_checked(header.data(), sb, kQgSupportedRequiredFeatures);
  if (sb_slot == -2) {
    throw std::runtime_error(
        "QuantizedGraph::load_disk_index: superblock is a newer format this build does not "
        "support (fail closed)");
  }
  if (sb_slot >= 0) {
    if (sb.dimension != dimension_ || sb.node_len != node_len_ ||
        sb.node_per_page != node_per_page_ || sb.page_size != page_size_) {
      throw std::runtime_error("QuantizedGraph::load_disk_index: v2 geometry mismatch");
    }
    if (!recovery_mode && sb.file_size != std::filesystem::file_size(index_file_name_)) {
      throw std::runtime_error("QuantizedGraph::load_disk_index: v2 file-size mismatch");
    }
    num_points_ = static_cast<size_t>(sb.num_points);
    entry_point_ = static_cast<PID>(sb.entry_point);
  } else {
    if (qg_header_has_v2_magic(header.data())) {
      throw std::runtime_error(
          "QuantizedGraph::load_disk_index: both v2 superblocks have invalid checksums");
    }
    std::array<uint64_t, kSectorLen / sizeof(uint64_t)> metas{};
    std::memcpy(metas.data(), header.data(), header.size());
    const uint64_t file_size = std::filesystem::file_size(index_file_name_);
    if (metas[0] != num_points_ || metas[1] != dimension_ || metas[3] != node_len_ ||
        metas[4] == 0 || metas[8] != file_size) {
      throw std::runtime_error("QuantizedGraph::load_disk_index: invalid v1 metadata");
    }

    const size_t loaded_npp = static_cast<size_t>(metas[4]);
    const size_t page_count = (num_points_ + loaded_npp - 1) / loaded_npp;
    if (page_count == 0 || file_size < kSectorLen || (file_size - kSectorLen) % page_count != 0) {
      throw std::runtime_error("QuantizedGraph::load_disk_index: invalid v1 file geometry");
    }
    const size_t loaded_page_size = static_cast<size_t>((file_size - kSectorLen) / page_count);
    const QGPageGeometry new_geometry = qg_page_geometry(node_len_);
    const size_t legacy_npp = std::max<size_t>(1, kSectorLen / node_len_);
    const size_t legacy_page_size =
        (legacy_npp * node_len_ + kSectorLen - 1) / kSectorLen * kSectorLen;
    const bool is_new_geometry =
        loaded_npp == new_geometry.node_per_page && loaded_page_size == new_geometry.page_size;
    const bool is_legacy_geometry =
        loaded_npp == legacy_npp && loaded_page_size == legacy_page_size;
    if (loaded_page_size % kSectorLen != 0 || loaded_npp * node_len_ > loaded_page_size ||
        (!is_new_geometry && !is_legacy_geometry)) {
      throw std::runtime_error("QuantizedGraph::load_disk_index: unsupported v1 page geometry");
    }
    node_per_page_ = loaded_npp;
    page_size_ = loaded_page_size;
    entry_point_ = static_cast<PID>(metas[2]);
  }

  std::string rotator_path = std::string(index_file_name_) + "_rotator";
  std::ifstream rotator_input(rotator_path, std::ios::binary);
  assert(rotator_input.is_open());
  rotator_.load(rotator_input);

  init_workspace();

  load_medoids(filename);

  // Load PCA parameters for online query transformation
  std::string pca_path = std::string(filename) + "_pca.bin";
  if (std::filesystem::exists(pca_path)) {
    pca_transform_.load(pca_path);
  } else {
    std::cerr << "Warning: PCA file not found: " << pca_path << std::endl;
  }

  // Calculate available cache space: use only 80% of the DRAM budget for caching.
  // The remaining 20% is reserved for other runtime memory allocations (e.g., search buffers,
  // AIO scratch space, and temporary data structures during query processing).
  auto cache_space = static_cast<size_t>(search_DRAM_budget * 1000 * 1000 * 1000 * 0.8);

  // Determine the number of nodes to cache: take the minimum of:
  // 1. Maximum nodes that fit in the available cache space (cache_space / node_len_)
  // 2. Maximum allowed cache ratio (kCacheRatio * num_points_) to limit memory usage
  size_t online_cache_num =
      std::min(cache_space / node_len_, static_cast<size_t>(kCacheRatio * num_points_));

  std::string cache_ids_file = std::string(index_file_name_) + "_cache_ids";
  std::string cache_nodes_file = std::string(index_file_name_) + "_cache_nodes";
  load_cache(cache_ids_file, cache_nodes_file, online_cache_num);
}

inline void QuantizedGraph::load_medoids(const char *filename) {
  // std::cout << "loading medoids..." << std::endl;
  std::string medoids_indices_file = std::string(filename) + "_medoids_indices";
  std::string medoids_file = std::string(filename) + "_medoids";

  if (!(std::filesystem::exists(medoids_file) && std::filesystem::exists(medoids_indices_file))) {
    return;
  }

  std::ifstream medoid_input(medoids_indices_file, std::ios::binary);
  assert(medoid_input.is_open());
  int medoid_num;
  int tmp;
  medoid_input.read(reinterpret_cast<char *>(&medoid_num), sizeof(int));
  medoid_input.read(reinterpret_cast<char *>(&tmp), sizeof(int));
  medoids_.resize(static_cast<uint64_t>(medoid_num) * static_cast<uint64_t>(tmp));
  medoid_input.read(reinterpret_cast<char *>(medoids_.data()),
                    static_cast<std::streamsize>(sizeof(int) * medoid_num * tmp));
  medoid_input.close();

  std::ifstream mediod_vector_input(medoids_file, std::ios::binary);
  assert(mediod_vector_input.is_open());
  int dim = 0;
  mediod_vector_input.read(reinterpret_cast<char *>(&medoid_num), sizeof(int));
  mediod_vector_input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  if (medoid_num != static_cast<int>(medoids_.size())) {
    throw std::runtime_error(
        "QuantizedGraph::load_medoids: medoid count mismatch between indices and vectors file");
  }
  if (dim != static_cast<int>(dimension_ + residual_dimension_)) {
    throw std::runtime_error(
        "QuantizedGraph::load_medoids: medoid dimension mismatch vs. index dimension");
  }
  medoids_vector_.resize(static_cast<uint64_t>(medoid_num * (dimension_ + residual_dimension_)));
  mediod_vector_input.read(reinterpret_cast<char *>(medoids_vector_.data()),
                           static_cast<std::streamsize>(sizeof(float) * medoid_num *
                                                        (dimension_ + residual_dimension_)));
  mediod_vector_input.close();
}

inline void QuantizedGraph::load_cache(std::string &cache_ids_file,
                                       std::string &cache_nodes_file,
                                       size_t online_cache_num) {
  std::ifstream cache_ids_input(cache_ids_file, std::ios::binary);
  std::ifstream cache_vectors_input(cache_nodes_file, std::ios::binary);
  assert(cache_ids_input.is_open());
  assert(cache_vectors_input.is_open());
  size_t cache_ids_num;
  size_t cache_nodes_num;
  size_t tmp_node_len;
  cache_ids_input.read(reinterpret_cast<char *>(&cache_ids_num), sizeof(size_t));
  cache_vectors_input.read(reinterpret_cast<char *>(&cache_nodes_num), sizeof(size_t));
  cache_vectors_input.read(reinterpret_cast<char *>(&tmp_node_len), sizeof(size_t));
  online_cache_num = std::min(online_cache_num, std::min(cache_ids_num, cache_nodes_num));
  std::cout << "online_cache_num: " << online_cache_num << std::endl;
  cache_ids_.resize(online_cache_num);
  cache_ids_input.read(reinterpret_cast<char *>(cache_ids_.data()),
                       static_cast<std::streamsize>(sizeof(PID) * online_cache_num));
  assert(tmp_node_len == node_len_);
  cache_nodes_.resize(online_cache_num * node_len_);
  cache_vectors_input.read(reinterpret_cast<char *>(cache_nodes_.data()),
                           static_cast<std::streamsize>(sizeof(char) * online_cache_num *
                                                        node_len_));
  for (unsigned i = 0; i < cache_ids_.size(); i++) {
    PID cur_id = cache_ids_[i];
    caches_[cur_id] = cache_nodes_.data() + i * node_len_;
  }
  arena_identity_ = cache_ids_.size() == num_points_;
  if (arena_identity_) {
    for (size_t i = 0; i < cache_ids_.size(); ++i) {
      if (cache_ids_[i] != static_cast<PID>(i)) {
        arena_identity_ = false;
        break;
      }
    }
  }
}

inline void QuantizedGraph::ensure_resident_arena() {
  if (arena_identity_) {
    return;
  }
  if (index_file_name_.empty()) {
    throw std::logic_error("QuantizedGraph::ensure_resident_arena: call load_disk_index() first");
  }
  std::ifstream in(index_file_name_, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("QuantizedGraph::ensure_resident_arena: cannot open " +
                             index_file_name_);
  }
  // Zero-fill on the calling thread = NUMA first-touch placement (the
  // NumaPolicy::kFirstTouch default needs no further action here).
  std::vector<char, ::alaya::AlignedAlloc<char>> arena(num_points_ * node_len_, 0);
  std::vector<char> page(page_size_);
  const size_t num_pages = (num_points_ + node_per_page_ - 1) / node_per_page_;
  for (size_t p = 0; p < num_pages; ++p) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(kSectorLen + p * page_size_));
    in.read(page.data(), static_cast<std::streamsize>(page_size_));
    const auto got = static_cast<size_t>(std::max<std::streamsize>(in.gcount(), 0));
    const size_t base = p * node_per_page_;
    const size_t rows_in_page = std::min(node_per_page_, num_points_ - base);
    if (got < rows_in_page * node_len_) {
      throw std::runtime_error("QuantizedGraph::ensure_resident_arena: short read at page " +
                               std::to_string(p) + " of " + index_file_name_);
    }
    for (size_t s = 0; s < rows_in_page; ++s) {
      std::memcpy(arena.data() + (base + s) * node_len_, page.data() + s * node_len_, node_len_);
    }
  }
  cache_nodes_ = std::move(arena);
  cache_ids_.resize(num_points_);
  std::iota(cache_ids_.begin(), cache_ids_.end(), PID{0});
  caches_.clear();
  for (size_t i = 0; i < cache_ids_.size(); ++i) {
    caches_[cache_ids_[i]] = cache_nodes_.data() + i * node_len_;
  }
  arena_identity_ = true;
}

inline void QuantizedGraph::arena_reserve_rows(size_t rows) {
  if (!arena_identity_) {
    throw std::logic_error(
        "QuantizedGraph::arena_reserve_rows: resident arena not materialized "
        "(load a full cache sidecar or call ensure_resident_arena() first)");
  }
  const size_t want_bytes = rows * node_len_;
  if (cache_nodes_.size() >= want_bytes) {
    return;
  }
  // Reallocation moves the arena, so the paged-path pointer map must be
  // rebuilt. Callers run this before serving searches (updater ctor time);
  // growth never runs concurrently with readers.
  cache_nodes_.resize(want_bytes, 0);
  for (size_t i = 0; i < cache_ids_.size(); ++i) {
    caches_[cache_ids_[i]] = cache_nodes_.data() + i * node_len_;
  }
}

inline void QuantizedGraph::arena_mirror_write(uint64_t file_off, const char *buf, size_t len) {
  if (!arena_identity_ || len == 0 || file_off < kSectorLen) {
    return;  // metadata-sector writes (superblock A/B copies) carry no row bytes
  }
  const size_t arena_rows = cache_nodes_.size() / node_len_;
  const uint64_t rel = file_off - kSectorLen;
  const uint64_t end = rel + len;
  for (uint64_t page = rel / page_size_; page * page_size_ < end; ++page) {
    const uint64_t page_start = page * page_size_;
    for (size_t slot = 0; slot < node_per_page_; ++slot) {
      const uint64_t row_start = page_start + slot * node_len_;
      if (row_start < rel || row_start + node_len_ > end) {
        continue;  // the updater writes whole pages; partially covered rows stay paged
      }
      const size_t pid = static_cast<size_t>(page) * node_per_page_ + slot;
      if (pid >= arena_rows) {
        continue;
      }
      std::memcpy(cache_nodes_.data() + pid * node_len_, buf + (row_start - rel), node_len_);
    }
  }
}

}  // namespace alaya::laser
