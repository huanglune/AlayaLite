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
#include <stdexcept>
#include <string>
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
  }

  ~MutableLaserSegment() { release_writer_lock(); }

  MutableLaserSegment(const MutableLaserSegment &) = delete;
  auto operator=(const MutableLaserSegment &) -> MutableLaserSegment & = delete;

  // Append one row and publish it durably. Returns the new PID (== its label).
  auto add(const float *vec) -> laser::PID {
    const auto id = updater_->allocate_and_insert(vec);
    updater_->publish(updater_->allocated_points());
    return id;
  }

  // Append a dense batch under one publish (three-phase batch contract). Returns
  // the base PID; labels are base .. base+n-1.
  auto add_batch(const float *vecs, size_t n) -> laser::PID {
    const auto base = static_cast<laser::PID>(updater_->allocated_points());
    for (size_t i = 0; i < n; ++i) {
      (void)updater_->allocate_and_insert(vecs + i * dim_);
    }
    updater_->publish(updater_->allocated_points());
    return base;
  }

  // Mark a row deleted and force the tombstone durable (its own next fsync).
  void tombstone(laser::PID id) {
    updater_->tombstone(id);
    updater_->publish(updater_->num_points());  // group-commit the tombstone
  }

  void flush() { updater_->writeback(1); }  // persist dirty pages + mirror the arena
  void checkpoint() { updater_->checkpoint(); }

  [[nodiscard]] auto search(const float *query, const DiskSearchOptions &opts)
      -> std::vector<DiskSearchHit> {
    if (opts.top_k == 0) {
      throw std::invalid_argument("MutableLaserSegment: top_k must be > 0");
    }
    const size_t ef = std::max<size_t>(opts.ef, opts.top_k);
    const auto pids = updater_->search(query, opts.top_k, ef);
    std::vector<DiskSearchHit> out;
    out.reserve(pids.size());
    for (const auto pid : pids) {
      out.push_back(DiskSearchHit{label_for(pid), std::numeric_limits<float>::quiet_NaN()});
    }
    return out;
  }

  [[nodiscard]] auto batch_search(const float *queries,
                                  uint32_t num_queries,
                                  const DiskSearchOptions &opts)
      -> std::vector<std::vector<DiskSearchHit>> {
    std::vector<std::vector<DiskSearchHit>> out;
    out.reserve(num_queries);
    for (uint32_t q = 0; q < num_queries; ++q) {
      out.push_back(search(queries + static_cast<size_t>(q) * dim_, opts));
    }
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

  [[nodiscard]] auto label_for(laser::PID pid) const -> uint64_t {
    return static_cast<uint64_t>(pid) < base_count_ ? ids_view_[pid] : static_cast<uint64_t>(pid);
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
};

}  // namespace alaya::disk
