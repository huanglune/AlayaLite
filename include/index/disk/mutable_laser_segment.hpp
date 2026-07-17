// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// A single-writer, durable, mutable handle over a sealed LASER v2 segment.
//
// This is the segment-level lifecycle surface for the G1 op-WAL: it opens a
// segment directory recovery-aware (unified-wal-vocabulary.md clause C — never
// the strict read-only LaserSegmentSearcher, whose immutable count check
// conflicts with a grown v2 superblock, clause H), attaches a QGUpdater with the
// WAL forced on, and serves add/tombstone/search over the same QuantizedGraph.
//
// Exclusivity (clause H/I): the handle takes an advisory flock on `<index>.opwal`
// for its whole lifetime, so a second writer fails fast rather than corrupting
// the shared graph. Any WAL/critical-index error poisons the underlying updater
// (fail closed); the handle propagates that as an exception.
//
// Labels (v1 limit): a row's label equals its PID for every appended row; rows
// below the sealed base count map through the immutable ids sidecar. The
// Collection layer owns any richer logical<->row map; wiring MutableLaserSegment
// into the collection lazy-laser slot is deliberately out of scope here.

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "index/disk/segment_manifest.hpp"
#include "index/disk/types.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_updater.hpp"
#include "index/graph/laser/qg/residency.hpp"
#include "platform/fs.hpp"
#include "storage/mmap_file.hpp"

namespace alaya::disk {

class MutableLaserSegment {
 public:
  MutableLaserSegment(const std::filesystem::path &seg_dir,
                      laser::UpdateParams params,
                      laser::ResidencyMode residency,
                      bool allow_empty = false)
      : residency_(residency) {
    // The active mutable LASER segment (allow_empty) carries count=0 in its
    // manifest -- all of its rows live in the op-WAL -- so it opens through the
    // allow-empty manifest policy and permits base_count==0. Sealed / importer /
    // read-only opens keep the strict count>0 contract (allow_empty=false, the
    // default that leaves every existing caller byte-identical).
    const auto manifest = allow_empty ? SegmentManifest::load_allow_empty(seg_dir / "manifest.txt")
                                      : SegmentManifest::load(seg_dir / "manifest.txt");
    if (manifest.index_type != DiskIndexType::Laser) {
      throw std::runtime_error("MutableLaserSegment: segment is not disk_laser: " +
                               seg_dir.string());
    }
    if (manifest.dim == 0 || (manifest.count == 0 && !allow_empty)) {
      throw std::runtime_error("MutableLaserSegment: manifest dim/count is zero: " +
                               seg_dir.string());
    }
    dim_ = static_cast<size_t>(manifest.dim);
    base_count_ = manifest.count;
    const auto prefix = require_extra(manifest, "x_laser_filename_prefix", seg_dir);
    const uint32_t r = parse_u32(manifest, "x_R", seg_dir);
    const uint32_t main_dim = parse_u32(manifest, "x_main_dim", seg_dir);
    const std::string index_file =
        prefix + "_R" + std::to_string(r) + "_MD" + std::to_string(main_dim) + ".index";
    const std::string index_prefix = (seg_dir / prefix).string();
    // A dedicated lock file: the .opwal itself is atomically renamed on
    // checkpoint (reset), which would silently break an flock held on it.
    const auto lock_path = seg_dir / (index_file + ".writer.lock");

    // Claim the single-writer lease before opening anything mutable.
    acquire_writer_lock(lock_path);

    try {
      qg_ = std::make_unique<laser::QuantizedGraph>(static_cast<size_t>(base_count_),
                                                    static_cast<size_t>(r),
                                                    static_cast<size_t>(main_dim),
                                                    static_cast<size_t>(manifest.dim));
      // Recovery-aware base load (clause C): skip the strict file_size check; the
      // op-WAL replay reconciles the physical length.
      qg_->load_disk_index(index_prefix.c_str(), 0.0F, /*recovery_mode=*/true);
      if (residency_ == laser::ResidencyMode::kResidentArena) {
        qg_->ensure_resident_arena();  // materialize + enable the write_at arena mirror
      }
      params.enable_wal = true;  // the handle is always durable
      if (params.max_points == 0) {
        params.max_points = 2 * static_cast<size_t>(base_count_) + 4096;
      }
      updater_ = std::make_unique<laser::QGUpdater>(*qg_, params);
    } catch (...) {
      release_writer_lock();
      throw;
    }

    // base_count_==0 (empty active segment) has no sealed ids sidecar to map, and
    // MMapFile rejects an empty file (POSIX mmap of length 0), so skip the mmap and
    // leave ids_view_ null: effective_label() never dereferences it because no live
    // PID is < base_count_ (every row is an explicitly-bound append).
    if (base_count_ != 0) {
      ids_mmap_ = alaya::storage::MMapFile(seg_dir / manifest.ids_file);
      if (ids_mmap_.size() != base_count_ * sizeof(uint64_t)) {
        release_writer_lock();
        throw std::runtime_error("MutableLaserSegment: ids sidecar size mismatch in " +
                                 seg_dir.string());
      }
      ids_view_ = ids_mmap_.as<uint64_t>();
    }

    // B-06 global bijection check (segment layer, after both the updater recovery
    // and the ids sidecar mmap, so the base labels are visible): fail closed.
    try {
      rebuild_reverse_index();
    } catch (...) {
      release_writer_lock();
      throw;
    }
  }

  ~MutableLaserSegment() { release_writer_lock(); }

  MutableLaserSegment(const MutableLaserSegment &) = delete;
  auto operator=(const MutableLaserSegment &) -> MutableLaserSegment & = delete;

  // Append one row and publish it durably. Returns the new PID (== its label).
  auto add(const float *vec) -> laser::PID {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    const auto id = updater_->allocate_and_insert(vec);
    updater_->publish(updater_->allocated_points());
    return id;
  }

  // Append a dense batch under one publish (three-phase batch contract). Returns
  // the base PID; labels are base .. base+n-1.
  auto add_batch(const float *vecs, size_t n) -> laser::PID {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    const auto base = static_cast<laser::PID>(updater_->allocated_points());
    for (size_t i = 0; i < n; ++i) {
      (void)updater_->allocate_and_insert(vecs + i * dim_);
    }
    updater_->publish(updater_->allocated_points());
    return base;
  }

  // Commit a physical label bundle (2A): n rows carrying n explicit labels under a
  // single durable transaction (kind=7 binds + kind=8 tx_publish). Returns the
  // appended PID range [base, base+n). Preconditions throw (caller error); any I/O
  // failure poisons the handle. 2A is the physical base: label-uniqueness at the
  // logical layer is the caller's (2B's) responsibility; the construction-time
  // bijection check is defense-in-depth.
  auto commit_physical_bundle(uint64_t txid,
                              uint64_t applied_collection_op_id,
                              const float *vecs,
                              const uint64_t *labels,
                              size_t n) -> std::pair<laser::PID, laser::PID> {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    if (n == 0) {
      throw std::invalid_argument("MutableLaserSegment::commit_physical_bundle: empty bundle");
    }
    const auto range =
        updater_->commit_physical_bundle(txid, applied_collection_op_id, vecs, labels, n);
    // Ruling 10: keep the segment-layer label->PID reverse index in step with the
    // freshly-bound rows so the adapter resolves erase/previous targets in O(1).
    for (size_t i = 0; i < n; ++i) {
      label_to_pid_[labels[i]] = static_cast<laser::PID>(range.first + i);
    }
    return range;
  }

  // Ruling 10: resolve a logical label to its live PID in O(1) for the Collection
  // adapter's erase / previous-tombstone path. Returns nullopt when the label has
  // no live PID (never bound, or already tombstoned) -- the adapter reads that as
  // an idempotent hit on replay and a high-severity diagnostic at runtime.
  [[nodiscard]] auto pid_for_label(uint64_t label) -> std::optional<laser::PID> {
    const std::lock_guard<std::mutex> guard(mutex_);
    const auto it = label_to_pid_.find(label);
    return it == label_to_pid_.end() ? std::nullopt : std::optional<laser::PID>(it->second);
  }

  // Mark a row deleted and force the tombstone durable (its own next fsync).
  void tombstone(laser::PID id) {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    // Ruling 10: drop this PID's label from the reverse index before hiding the row
    // so a later erase/previous of the same label correctly resolves to nullopt.
    const uint64_t lbl = effective_label(id, updater_->label_snapshot());
    const auto it = label_to_pid_.find(lbl);
    if (it != label_to_pid_.end() && it->second == id) {
      label_to_pid_.erase(it);
    }
    updater_->tombstone(id);
    updater_->publish(updater_->num_points());  // group-commit the tombstone
  }

  void flush() {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    updater_->writeback(1);                           // persist dirty pages + mirror the arena
  }
  void checkpoint() {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
    updater_->checkpoint();
  }

  [[nodiscard]] auto search(const float *query, const DiskSearchOptions &opts)
      -> std::vector<DiskSearchHit> {
    if (opts.top_k == 0) {
      throw std::invalid_argument("MutableLaserSegment: top_k must be > 0");
    }
    updater_->ensure_readable();  // entry poison gate (B-02); lock-free
    const size_t ef = std::max<size_t>(opts.ef, opts.top_k);
    const auto pids = updater_->search(query, opts.top_k, ef);
    // Acquire the label snapshot AFTER search took its committed watermark: the
    // snapshot is published before committed, so it covers every committed PID's
    // binding, and identity fallback never fires spuriously (B-02).
    const auto snap = updater_->label_snapshot();
    std::vector<DiskSearchHit> out;
    out.reserve(pids.size());
    for (const auto pid : pids) {
      out.push_back(
          DiskSearchHit{effective_label(pid, snap), std::numeric_limits<float>::quiet_NaN()});
    }
    updater_->ensure_readable();  // exit poison gate (B-02)
    return out;
  }

  [[nodiscard]] auto batch_search(const float *queries,
                                  uint32_t num_queries,
                                  const DiskSearchOptions &opts)
      -> std::vector<std::vector<DiskSearchHit>> {
    updater_->ensure_readable();  // entry poison gate (B-02)
    std::vector<std::vector<DiskSearchHit>> out;
    out.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
      out.push_back(search(queries + static_cast<size_t>(q) * dim_, opts));
    }
    updater_->ensure_readable();  // exit poison gate (B-02)
    return out;
  }

  [[nodiscard]] auto size() const -> size_t { return updater_->num_points(); }
  [[nodiscard]] auto dim() const -> size_t { return dim_; }
  [[nodiscard]] auto base_count() const -> uint64_t { return base_count_; }
  // 2B accessors for the Collection adapter: the idempotency basis (applied op /
  // last committed txid) and live/allocated counts for stats.
  [[nodiscard]] auto live_count() const -> uint64_t { return updater_->live_count(); }
  [[nodiscard]] auto allocated_count() const -> size_t { return updater_->allocated_points(); }
  [[nodiscard]] auto applied_collection_op_id() const -> uint64_t {
    return updater_->applied_collection_op_id();
  }
  [[nodiscard]] auto last_committed_txid() const -> uint64_t {
    return updater_->last_committed_txid();
  }

  // Create a brand-new EMPTY (count=0) active LASER segment directory: a
  // checksum-valid v2 superblock (num_points=0), a matching FHT rotator, two
  // cache sidecars with legal empty headers, an empty ids sidecar, and a count=0
  // manifest. The op-WAL and label slots are auto-created on the first open (a
  // fresh lineage, uid=0), whose auto-checkpoint stamps the durable lineage id and
  // writes the second superblock slot. Open it via the allow_empty constructor.
  // Ruling 5 / codex B-06. main_dim==dim (no residual/PCA) for the active engine.
  static void create_empty(const std::filesystem::path &seg_dir,
                           const std::string &segment_id,
                           uint32_t dim,
                           uint32_t main_dim,
                           uint32_t r,
                           core::Metric metric,
                           const std::string &prefix = "active_laser") {
    std::filesystem::create_directories(seg_dir);
    // Geometry: mirror QuantizedGraph's ctor (qg.hpp:491-516). No public accessor
    // exposes node_len_/page geometry; load_disk_index re-derives and hard-checks
    // it, so an incorrect value fails the very first open loudly (covered by test).
    const size_t dimension = main_dim;
    const size_t residual = static_cast<size_t>(dim) - dimension;
    const size_t padded_dim = size_t{1} << laser::ceil_log2(dimension);
    const size_t node_len = (32 * dimension + 32 * residual + 128 * static_cast<size_t>(r) +
                             static_cast<size_t>(r) * padded_dim) /
                            8;
    const laser::QGPageGeometry geometry = laser::qg_page_geometry(node_len);

    const std::string index_file =
        prefix + "_R" + std::to_string(r) + "_MD" + std::to_string(main_dim) + ".index";
    const auto index_path = seg_dir / index_file;

    // (1) v2 superblock in slot A. num_points=0, entry_point=0; the file is just
    // the 4 KiB metadata sector (slot B zeroed) -- the first-open checkpoint
    // ftruncates/writes the rest. reserved{} all-zero => uid=0 => fresh lineage.
    laser::QGSuperblockV2 superblock;
    superblock.magic = laser::kQGSuperblockMagic;
    superblock.format_version = laser::kQGFormatVersion;
    superblock.generation = 1;
    superblock.num_points = 0;
    superblock.live_count = 0;
    superblock.free_list_head = laser::kPidMax;
    superblock.free_count = 0;
    superblock.entry_point = 0;
    superblock.dimension = dimension;
    superblock.node_len = node_len;
    superblock.node_per_page = geometry.node_per_page;
    superblock.page_size = geometry.page_size;
    superblock.file_size = laser::kSectorLen;
    superblock.checksum = laser::qg_superblock_checksum(superblock);

    std::vector<char> header(laser::kSectorLen, 0);
    std::memcpy(header.data(), &superblock, sizeof(superblock));
    ::alaya::platform::write_all_fsync(index_path, header.data(), header.size());

    // (2) FHT rotator sidecar: raw mat_ bytes, symmetric with rotator_.load(). A
    // freshly-constructed FHTRotator(main_dim, seed) round-trips regardless of the
    // internal layout, so this needs no byte-level format knowledge.
    {
      laser::FHTRotator rotator(dimension, /*seed=*/42);
      std::ofstream out(index_path.string() + "_rotator", std::ios::binary | std::ios::trunc);
      if (!out) {
        throw std::runtime_error("MutableLaserSegment::create_empty: cannot open rotator sidecar");
      }
      rotator.save(out);
      out.flush();
      if (!out) {
        throw std::runtime_error("MutableLaserSegment::create_empty: rotator sidecar write failed");
      }
    }
    // (3) cache sidecars with legal empty headers (load_cache reads them
    // unconditionally): _cache_ids={size_t 0}; _cache_nodes={size_t 0, node_len}.
    {
      const size_t zero = 0;
      ::alaya::platform::write_all_fsync(index_path.string() + "_cache_ids", &zero, sizeof(zero));
      size_t nodes_header[2] = {0, node_len};
      ::alaya::platform::write_all_fsync(index_path.string() + "_cache_nodes",
                                         nodes_header,
                                         sizeof(nodes_header));
    }
    // (4) empty ids sidecar (0 bytes): base_count==0 skips mmapping it, but keep the
    // file present so the directory is self-describing.
    ::alaya::platform::write_all_fsync(seg_dir / "ids.u64.bin", nullptr, 0);
    // (5) manifest with count=0 (read back via load_allow_empty on the active path).
    SegmentManifest manifest;
    manifest.segment_id = segment_id;
    manifest.index_type = DiskIndexType::Laser;
    manifest.metric = metric;
    manifest.dim = dim;
    manifest.count = 0;
    manifest.ids_file = "ids.u64.bin";
    manifest.vectors_file = "";
    manifest.x_extras["x_laser_filename_prefix"] = prefix;
    manifest.x_extras["x_R"] = std::to_string(r);
    manifest.x_extras["x_main_dim"] = std::to_string(main_dim);
    manifest.save(seg_dir / "manifest.txt");
  }

 private:
  static auto require_extra(const SegmentManifest &manifest,
                            const char *key,
                            const std::filesystem::path &seg_dir) -> const std::string & {
    const auto it = manifest.x_extras.find(key);
    if (it == manifest.x_extras.end() || it->second.empty()) {
      throw std::runtime_error("MutableLaserSegment: manifest is missing " + std::string(key) +
                               " for " + seg_dir.string());
    }
    return it->second;
  }
  static auto parse_u32(const SegmentManifest &manifest,
                        const char *key,
                        const std::filesystem::path &seg_dir) -> uint32_t {
    const auto &value = require_extra(manifest, key, seg_dir);
    const auto parsed = std::stoull(value);
    if (parsed == 0 || parsed > std::numeric_limits<uint32_t>::max()) {
      throw std::runtime_error("MutableLaserSegment: " + std::string(key) + " out of range in " +
                               seg_dir.string());
    }
    return static_cast<uint32_t>(parsed);
  }

  // Effective label of a committed PID: base rows map through the immutable ids
  // sidecar; appended rows use an explicit binding from the snapshot, else fall
  // back to identity (U2-a legacy appends carry no binding).
  [[nodiscard]] auto effective_label(laser::PID pid,
                                     const std::shared_ptr<const laser::LabelBindings> &snap) const
      -> uint64_t {
    // W2 (design section 3.2): an explicit binding is the override for ALL PIDs, not
    // just appended ones -- a reused base PID carries a fresh binding that supersedes
    // its immutable-sidecar label. Order: explicit binding, then base sidecar, then
    // identity. (Before pid-reuse activation no base PID has a binding, so this is a
    // no-op vs the old base-first order.)
    const uint64_t *bound = snap ? snap->find(pid) : nullptr;
    if (bound != nullptr) {
      return *bound;
    }
    if (static_cast<uint64_t>(pid) < base_count_) {
      return ids_view_[pid];
    }
    return static_cast<uint64_t>(pid);
  }

  // B-06: over ALL committed live PIDs, the effective label (base sidecar U legacy
  // identity U explicit appended) must be injective. A label owned by two live PIDs
  // is an ambiguous mapping, so construction fails (fail-closed). Tombstoned/free
  // rows keep their forward pid->label binding but do not occupy the live domain.
  // B-06 + ruling 10: over ALL committed live PIDs the effective label (base
  // sidecar U legacy identity U explicit appended binding) must be injective, so a
  // label owned by two live PIDs fails construction (fail-closed). The same pass
  // populates the retained label->PID reverse index (rebuilt on every open, then
  // maintained incrementally by commit_physical_bundle/tombstone). Tombstoned/free
  // rows keep their forward pid->label binding but do not occupy the live domain.
  void rebuild_reverse_index() {
    const auto snap = updater_->label_snapshot();
    const size_t committed = updater_->num_points();
    label_to_pid_.clear();
    label_to_pid_.reserve(committed);
    for (size_t p = 0; p < committed; ++p) {
      const auto pid = static_cast<laser::PID>(p);
      if (!updater_->row_is_live(pid)) {
        continue;
      }
      const uint64_t lbl = effective_label(pid, snap);
      const auto [it, inserted] = label_to_pid_.emplace(lbl, pid);
      if (!inserted) {
        throw std::runtime_error("MutableLaserSegment: label " + std::to_string(lbl) +
                                 " maps to two live PIDs (" + std::to_string(it->second) + " and " +
                                 std::to_string(pid) + ")");
      }
    }
  }

  void acquire_writer_lock(const std::filesystem::path &lock_path) {
    wal_lock_fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (wal_lock_fd_ < 0) {
      throw std::runtime_error("MutableLaserSegment: cannot open writer-lock file " +
                               lock_path.string());
    }
    if (::flock(wal_lock_fd_, LOCK_EX | LOCK_NB) != 0) {
      ::close(wal_lock_fd_);
      wal_lock_fd_ = -1;
      throw std::runtime_error("MutableLaserSegment: another writer holds " + lock_path.string() +
                               " (single-writer lease)");
    }
  }
  void release_writer_lock() noexcept {
    if (wal_lock_fd_ >= 0) {
      ::flock(wal_lock_fd_, LOCK_UN);
      ::close(wal_lock_fd_);
      wal_lock_fd_ = -1;
    }
  }

  laser::ResidencyMode residency_;
  size_t dim_ = 0;
  uint64_t base_count_ = 0;
  std::unique_ptr<laser::QuantizedGraph> qg_;
  std::unique_ptr<laser::QGUpdater> updater_;
  alaya::storage::MMapFile ids_mmap_;
  const uint64_t *ids_view_ = nullptr;
  int wal_lock_fd_ = -1;
  // Ruling 10: segment-layer label->live-PID reverse index. In-memory only (never
  // persisted): rebuilt from the label snapshot on open (rebuild_reverse_index)
  // and maintained on commit_physical_bundle (add) / tombstone (remove). Guarded
  // by mutex_ like every other mutating-path member.
  std::unordered_map<uint64_t, laser::PID> label_to_pid_{};
  // Single-writer handle mutex (W0): every public mutating method holds it, so
  // add/add_batch/tombstone/flush/checkpoint/commit_physical_bundle never race
  // each other. search/batch_search stay lock-free and use the poison read gate.
  std::mutex mutex_;
};

}  // namespace alaya::disk
