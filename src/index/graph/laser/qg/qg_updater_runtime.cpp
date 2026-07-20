// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#include "index/disk/mutable_laser_segment.hpp"

#include "index/graph/laser/qg/detail/qg_updater_core.hpp"

namespace alaya::disk {

MutableLaserSegment::MutableLaserSegment(const std::filesystem::path &seg_dir,
                                         laser::UpdateParams params,
                                         laser::ResidencyMode residency,
                                         bool allow_empty)
    : residency_(residency) {
  // The active mutable LASER segment (allow_empty) carries count=0 in its
  // manifest -- all of its rows live in the op-WAL -- so it opens through the
  // allow-empty manifest policy and permits base_count==0. Sealed / importer /
  // read-only opens keep the strict count>0 contract (allow_empty=false, the
  // default that leaves every existing caller byte-identical).
  const auto manifest = allow_empty ? SegmentManifest::load_allow_empty(seg_dir / "manifest.txt")
                                    : SegmentManifest::load(seg_dir / "manifest.txt");
  if (manifest.index_type != DiskIndexType::Laser) {
    throw std::runtime_error("MutableLaserSegment: segment is not disk_laser: " + seg_dir.string());
  }
  if (manifest.metric != core::Metric::l2) {
    throw std::runtime_error(
        "MutableLaserSegment: active mutation remains L2-only; non-L2 is sealed/read-only");
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
    teardown_writer_resources();
    throw;
  }

  // base_count_==0 (empty active segment) has no sealed ids sidecar to map, and
  // MMapFile rejects an empty file (POSIX mmap of length 0), so skip the mmap and
  // leave ids_view_ null: effective_label() never dereferences it because no live
  // PID is < base_count_ (every row is an explicitly-bound append).
  if (base_count_ != 0) {
    try {
      ids_mmap_ = alaya::storage::MMapFile(seg_dir / manifest.ids_file);
      if (ids_mmap_.size() != base_count_ * sizeof(uint64_t)) {
        throw std::runtime_error("MutableLaserSegment: ids sidecar size mismatch in " +
                                 seg_dir.string());
      }
      ids_view_ = ids_mmap_.as<uint64_t>();
    } catch (...) {
      teardown_writer_resources();
      throw;
    }
  }

  // B-06 global bijection check (segment layer, after both the updater recovery
  // and the ids sidecar mmap, so the base labels are visible): fail closed.
  try {
    rebuild_reverse_index();
  } catch (...) {
    teardown_writer_resources();
    throw;
  }
}

MutableLaserSegment::~MutableLaserSegment() {
  // Keep the process/external writer lease through dependent teardown, not
  // merely until the destructor body starts. QGUpdater references qg_, and
  // QuantizedGraph shutdown drains its PageReader workers/completions; only
  // after both are gone may a close->reset->reopen caller acquire the file.
  teardown_writer_resources();
}

void MutableLaserSegment::teardown_writer_resources() noexcept {
  updater_.reset();
  qg_.reset();
  release_writer_lock();
}

}  // namespace alaya::disk

namespace alaya::laser::detail {

PID qg_updater_allocate_and_insert(QGUpdater &updater, const float *vec) {
  return updater.allocate_and_insert(vec);
}

size_t qg_updater_allocated_points(const QGUpdater &updater) { return updater.allocated_points(); }

void qg_updater_publish(QGUpdater &updater, size_t new_committed) {
  updater.publish(new_committed);
}

PhysicalBundleResult qg_updater_commit_physical_bundle_tokens(QGUpdater &updater,
                                                              uint64_t txid,
                                                              uint64_t applied_op_id,
                                                              const float *vecs,
                                                              const uint64_t *labels,
                                                              size_t n) {
  return updater.commit_physical_bundle_tokens(txid, applied_op_id, vecs, labels, n);
}

std::shared_ptr<const LabelBindings> qg_updater_label_snapshot(const QGUpdater &updater) {
  return updater.label_snapshot();
}

void qg_updater_tombstone(QGUpdater &updater, PID id) { updater.tombstone(id); }

uint32_t qg_updater_durable_generation(const QGUpdater &updater, PID id) {
  return updater.durable_generation(id);
}

bool qg_updater_row_is_live(const QGUpdater &updater, PID id) { return updater.row_is_live(id); }

size_t qg_updater_num_points(const QGUpdater &updater) { return updater.num_points(); }

void qg_updater_writeback(QGUpdater &updater, size_t num_threads) {
  updater.writeback(num_threads);
}

void qg_updater_consolidate(QGUpdater &updater,
                            size_t num_threads,
                            size_t r_target,
                            bool reclaim_slots,
                            bool bloom_consolidate) {
  updater.consolidate(num_threads, r_target, reclaim_slots, bloom_consolidate);
}

uint64_t qg_updater_free_count(const QGUpdater &updater) { return updater.free_count(); }

bool qg_updater_pid_generation_activated(const QGUpdater &updater) {
  return updater.pid_generation_activated();
}

bool qg_updater_is_poisoned(const QGUpdater &updater) noexcept { return updater.is_poisoned(); }

void qg_updater_checkpoint(QGUpdater &updater) { updater.checkpoint(); }

void qg_updater_require_dual_v3_if_activated(QGUpdater &updater) {
  updater.require_dual_v3_if_activated();
}

void qg_updater_ensure_readable(const QGUpdater &updater) { updater.ensure_readable(); }

std::vector<PID> qg_updater_search(QGUpdater &updater,
                                   const float *query,
                                   size_t k,
                                   size_t ef,
                                   size_t max_beam_width,
                                   float *distances) {
  return updater.search(query, k, ef, max_beam_width, distances);
}

uint64_t qg_updater_live_count(const QGUpdater &updater) { return updater.live_count(); }

uint64_t qg_updater_applied_collection_op_id(const QGUpdater &updater) {
  return updater.applied_collection_op_id();
}

uint64_t qg_updater_last_committed_txid(const QGUpdater &updater) {
  return updater.last_committed_txid();
}

UpdateStats qg_updater_stats(const QGUpdater &updater) { return updater.stats(); }

}  // namespace alaya::laser::detail
