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
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
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
#include "storage/mmap_file.hpp"

namespace alaya::disk {

class MutableLaserSegment {
 public:
  MutableLaserSegment(const std::filesystem::path &seg_dir,
                      laser::UpdateParams params,
                      laser::ResidencyMode residency)
      : residency_(residency) {
    const auto manifest = SegmentManifest::load(seg_dir / "manifest.txt");
    if (manifest.index_type != DiskIndexType::Laser) {
      throw std::runtime_error("MutableLaserSegment: segment is not disk_laser: " +
                               seg_dir.string());
    }
    if (manifest.dim == 0 || manifest.count == 0) {
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

    ids_mmap_ = alaya::storage::MMapFile(seg_dir / manifest.ids_file);
    if (ids_mmap_.size() != base_count_ * sizeof(uint64_t)) {
      release_writer_lock();
      throw std::runtime_error("MutableLaserSegment: ids sidecar size mismatch in " +
                               seg_dir.string());
    }
    ids_view_ = ids_mmap_.as<uint64_t>();

    // B-06 global bijection check (segment layer, after both the updater recovery
    // and the ids sidecar mmap, so the base labels are visible): fail closed.
    try {
      verify_label_bijection();
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
    return updater_->commit_physical_bundle(txid, applied_collection_op_id, vecs, labels, n);
  }

  // Mark a row deleted and force the tombstone durable (its own next fsync).
  void tombstone(laser::PID id) {
    const std::lock_guard<std::mutex> guard(mutex_);  // single-writer handle mutex (W0)
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
    if (static_cast<uint64_t>(pid) < base_count_) {
      return ids_view_[pid];
    }
    const uint64_t *bound = snap ? snap->find(pid) : nullptr;
    return bound != nullptr ? *bound : static_cast<uint64_t>(pid);
  }

  // B-06: over ALL committed live PIDs, the effective label (base sidecar U legacy
  // identity U explicit appended) must be injective. A label owned by two live PIDs
  // is an ambiguous mapping, so construction fails (fail-closed). Tombstoned/free
  // rows keep their forward pid->label binding but do not occupy the live domain.
  void verify_label_bijection() const {
    const auto snap = updater_->label_snapshot();
    const size_t committed = updater_->num_points();
    std::unordered_map<uint64_t, laser::PID> owner;
    owner.reserve(committed);
    for (size_t p = 0; p < committed; ++p) {
      const auto pid = static_cast<laser::PID>(p);
      if (!updater_->row_is_live(pid)) {
        continue;
      }
      const uint64_t lbl = effective_label(pid, snap);
      const auto [it, inserted] = owner.emplace(lbl, pid);
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
  // Single-writer handle mutex (W0): every public mutating method holds it, so
  // add/add_batch/tombstone/flush/checkpoint/commit_physical_bundle never race
  // each other. search/batch_search stay lock-free and use the poison read gate.
  std::mutex mutex_;
};

}  // namespace alaya::disk
