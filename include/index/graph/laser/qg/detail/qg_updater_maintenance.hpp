// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Class-body fragment included exactly once by qg_updater.hpp.
// Maintenance transaction entry points.

  void garden(size_t num_threads, const GardenParams &gp) {
    if (enable_wal_) {
      throw std::logic_error(
          "QGUpdater::garden is out of the G1 op-WAL scope (clause A): its page rewrites need a "
          "consolidate-style WAL transaction (next wave)");
    }
    if (!params_.maintain_indegree) {
      throw std::logic_error("QGUpdater::garden requires maintain_indegree");
    }
    if (params_.garden_churn_threshold > 0) {
      const size_t n = committed_.load(std::memory_order_acquire);
      const auto threshold =
          static_cast<uint64_t>(params_.garden_churn_threshold * static_cast<double>(n));
      if (churn_since_garden_.load(std::memory_order_relaxed) < threshold) {
        stats_.garden_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      churn_since_garden_.store(0, std::memory_order_relaxed);
    }
    const auto t0 = std::chrono::steady_clock::now();
    const size_t n = committed_.load(std::memory_order_acquire);
    std::vector<PID> live;
    live.reserve(std::min<uint64_t>(n, live_count_.load(std::memory_order_acquire)));
    for (size_t i = 0; i < n; ++i) {
      if (!is_hidden(static_cast<PID>(i))) live.push_back(static_cast<PID>(i));
    }
    if (gp.policy == GardenParams::Policy::kTurnover && !params_.maintain_turnover) {
      throw std::logic_error("QGUpdater::garden kTurnover requires maintain_turnover");
    }
    uint64_t all_turnover = 0;
    if (params_.maintain_turnover) {
      for (PID id : live) all_turnover += turnover(id);
    }
    const uint64_t all_turnover_rows = live.size();
    const double frac = std::clamp(gp.frac, 0.0, 1.0);
    size_t k = static_cast<size_t>(std::ceil(frac * static_cast<double>(live.size())));
    k = std::min(k, live.size());
    if (gp.policy == GardenParams::Policy::kLowIndegree) {
      std::sort(live.begin(), live.end(), [&](PID a, PID b) {
        const int32_t ia = indegree(a), ib = indegree(b);
        return ia != ib ? ia < ib : a < b;
      });
    } else if (gp.policy == GardenParams::Policy::kRandom) {
      std::mt19937_64 rng(0x4c41534552ULL);
      std::shuffle(live.begin(), live.end(), rng);
    } else {
      std::sort(live.begin(), live.end(), [&](PID a, PID b) {
        const uint16_t ta = turnover(a), tb = turnover(b);
        return ta != tb ? ta > tb : a < b;
      });
    }
    live.resize(k);
    uint64_t selected_turnover = 0;
    if (params_.maintain_turnover) {
      for (PID id : live) selected_turnover += turnover(id);
      stats_.garden_all_turnover_sum.fetch_add(all_turnover, std::memory_order_relaxed);
      stats_.garden_all_turnover_rows.fetch_add(all_turnover_rows, std::memory_order_relaxed);
      stats_.garden_selected_turnover_sum.fetch_add(selected_turnover, std::memory_order_relaxed);
      stats_.garden_selected_turnover_rows.fetch_add(live.size(), std::memory_order_relaxed);
    }
    const size_t target = gp.r_target == 0 ? deg_ : std::min(gp.r_target, deg_);
    const int nt = static_cast<int>(std::max<size_t>(1, num_threads));
    const size_t stride = params_.maintenance_evict_stride;
    const bool in_pass_evict =
        stride != 0 && params_.write_cache && params_.cache_cap_pages < file_pages();
    const size_t rows_per_batch = in_pass_evict ? stride : std::max<size_t>(1, live.size());
    if (!in_pass_evict) {
      parallel_for_catch(0, static_cast<int64_t>(live.size()), nt, 1, [&](int64_t i) {
        garden_row(live[static_cast<size_t>(i)], gp, target);
      });
    } else {
      size_t batch_begin = 0;
      while (batch_begin < live.size()) {
        const size_t batch_end = std::min(live.size(), batch_begin + rows_per_batch);
        parallel_for_catch(static_cast<int64_t>(batch_begin),
                           static_cast<int64_t>(batch_end),
                           nt,
                           1,
                           [&](int64_t i) {
                             garden_row(live[static_cast<size_t>(i)], gp, target);
                           });
        batch_begin = batch_end;
        const bool need_evict = note_maintenance_pool_and_test_high();
        if (need_evict) enforce_maintenance_watermark(num_threads);
      }
    }
    if (stride != 0) note_maintenance_pool_and_test_high();
    flush_dirty(num_threads);
    stats_.garden_us.fetch_add(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0)
            .count());
  }

  /** Persist dirty pages and atomically advance the alternate A/B superblock. */
  void checkpoint() {
    const std::lock_guard<std::mutex> checkpoint_guard(checkpoint_mutex_);
    checkpoint_locked();
  }

  // JC-23: the checkpoint body with the checkpoint_mutex_ ALREADY HELD. The canonical
  // reuse writer (and consolidate's activation) run under checkpoint_mutex_ and must be
  // able to drive a checkpoint/activation without re-locking. Public checkpoint() takes
  // the lock; every in-lane caller uses checkpoint_locked() directly.
  void checkpoint_locked() {
    if (enable_wal_ && !replaying_) {
      ensure_writable();
      // Admission (clause D): no in-flight allocation gap and no staged edges, so
      // the superblock image is built from committed state only (never publishes
      // dark rows). W3 enforces mutation x checkpoint exclusion at the handle.
      if (allocated_points_.load(std::memory_order_acquire) !=
          committed_.load(std::memory_order_acquire)) {
        throw std::logic_error(
            "QGUpdater::checkpoint requires allocated == committed (publish the batch first)");
      }
      if (has_staged_edges()) {
        throw std::logic_error("QGUpdater::checkpoint requires no staged backlinks");
      }
      if (maintenance_active_) {
        throw std::logic_error(
            "QGUpdater::checkpoint requires no active maintenance epoch (design section 1.4)");
      }
      // Reuse admission (design 1.4 / codex B.6): an active/reserved bundle -- including
      // an all-reuse reservation whose HWM never moves -- and an unfinished free-chain
      // rebuild must block a checkpoint so a half-built bundle is never absorbed as a
      // kind=6 base. (allocated==committed above cannot see an all-reuse reservation.)
      if (bundle_state_ != BundleState::kIdle) {
        throw std::logic_error("QGUpdater::checkpoint requires no active bundle reservation");
      }
      if (reservation_count_ != 0) {
        throw std::logic_error("QGUpdater::checkpoint requires no reserved reuse PIDs");
      }
      if (!free_chain_rebuild_complete_) {
        throw std::logic_error("QGUpdater::checkpoint requires a rebuilt free chain");
      }
    }
    drain_staged_edges(1);
    flush_dirty(1);  // forces the WAL prefix first under enable_wal
    const size_t n = allocated_points_.load(std::memory_order_acquire);
    const size_t page_num = (n + npp_ - 1) / npp_;
    const uint64_t file_size = kSectorLen + page_num * page_size_;
    QGSuperblockV2 next = superblock_;
    next.magic = kQGSuperblockMagic;
    next.format_version = kQGFormatVersion;
    next.generation = superblock_.generation + 1;
    next.num_points = n;
    next.live_count = live_count_.load(std::memory_order_acquire);
    next.free_list_head = free_list_head_.load(std::memory_order_acquire);
    next.free_count = free_count_.load(std::memory_order_acquire);
    next.entry_point = qg_.entry_point_;
    next.file_size = file_size;
    if (enable_wal_) {
      write_superblock_uid(next, segment_uid_);  // carry the lineage forward
      // Label slot integration: the explicit binding set is append-only, so a
      // strict count increase means new bindings to persist. Write them to the
      // INACTIVE slot (double-buffered), then carry the new label tuple; otherwise
      // copy the current tuple. The tx watermarks always ride along (B-03: even a
      // no-increment checkpoint must carry the txid history forward).
      const auto snap = load_label_snapshot();
      const uint64_t current_count = snap->bindings.size();
      LabelSlotState next_label;
      // Dirty when the key count grew (append) OR the content revision moved without a
      // count change (design 3.1: a same-count reuse rebind). In append-only mode the
      // two coincide, so the OR is byte-identical to the historical count gate.
      if (current_count > label_count_ ||
          label_content_revision_ != persisted_label_content_revision_) {
        const int inactive_label_slot = active_label_slot_ == 0 ? 1 : 0;
        const auto body = serialize_label_slot(*snap);
        write_label_slot_file(inactive_label_slot, body);  // pwrite + fsync (slot durable)
        wal_failpoint(SegmentOpFailPoint::label_slot_written_before_flip);
        next_label.slot = static_cast<uint64_t>(inactive_label_slot);
        next_label.generation = label_generation_ + 1;
        next_label.count = current_count;
        next_label.checksum = static_cast<uint64_t>(alaya::wal::crc32(body));
      } else {
        next_label.slot = static_cast<uint64_t>(active_label_slot_);
        next_label.generation = label_generation_;
        next_label.count = label_count_;
        next_label.checksum = label_checksum_;
      }
      write_superblock_label_state(next, next_label);
      write_superblock_tx_state(next,
                                TxWatermarkState{last_committed_txid_, applied_collection_op_id_});
      // 2C activation (design section 7.2): once maintenance is activated (or an
      // activation checkpoint is in flight), the base becomes v3 and carries the
      // maintenance feature bits + last_completed_consolidate_epoch. create_empty /
      // the immutable producer never take this path, so they stay v2.
      if (maintenance_activated_ || maintenance_activating_) {
        next.format_version = kQGFormatVersionV3;
        // Carry an existing v3 state forward; when activating from a v2 base start
        // from a zeroed state (never reinterpret v2's reserved bytes as v3 state).
        Wal2cState w2c = superblock_.format_version == kQGFormatVersionV3
                             ? read_superblock_wal2c_state(superblock_)
                             : Wal2cState{};
        w2c.magic = kWal2cMagic;
        w2c.layout_version = kWal2cLayoutVersion;
        w2c.required_feature_flags |= kQgFeatMaintenanceTxV1 | kQgFeatPostRedoFreeListV1;
        w2c.last_completed_consolidate_epoch = last_completed_consolidate_epoch_;
        if (w2c.maintenance_activation_sb_generation == 0) {
          w2c.maintenance_activation_sb_generation = next.generation;
        }
        // PID-reuse activation (design 7.2): OR the pid_generation + canonical_prebind +
        // mutable_label bits (self-consistency triple) and stamp the reuse activation
        // generation. Only fires when a reuse bundle is activating -- a maintenance-only
        // consolidate never sets these, so its v3 base stays maintenance-only.
        if (pid_reuse_activating_ || pid_generation_activated_) {
          w2c.required_feature_flags |=
              kQgFeatPidGenerationV1 | kQgFeatCanonicalPrebindV1 | kQgFeatMutableLabelSlotV1;
          if (w2c.pid_reuse_activation_sb_generation == 0) {
            w2c.pid_reuse_activation_sb_generation = next.generation;
          }
          // Activation summary (design 7.1 / JC-16): the max non-zero generation and the
          // count of reused (generation>0) bindings in the slot being persisted. Recovery
          // recomputes both from the loaded slot and fails closed on a mismatch, so a
          // forged slot cannot silently under/over-report its reuse population.
          uint32_t max_gen = 0;
          uint32_t nz_count = 0;
          for (const auto &[pid, binding] : snap->bindings) {
            (void)pid;
            if (binding.pid_generation != 0) {
              ++nz_count;
              max_gen = (std::max)(max_gen, binding.pid_generation);
            }
          }
          w2c.max_pid_generation = max_gen;
          w2c.nonzero_pid_generation_count = nz_count;
        }
        write_superblock_wal2c_state(next, w2c);
      }
    }
    next.checksum = qg_superblock_checksum(next);
    const int next_slot = active_superblock_slot_ == 0 ? 1 : 0;
    // In-memory adoption of the just-written base, shared by both branches. Runs AFTER
    // the durable superblock/flip on the enable_wal path (inside the poison guard below).
    const auto adopt_in_memory = [&] {
      superblock_ = next;
      active_superblock_slot_ = next_slot;
      qg_.num_points_ = committed_.load(std::memory_order_acquire);
      if (enable_wal_) {
        const auto ls = read_superblock_label_state(next);
        active_label_slot_ = static_cast<int>(ls.slot);
        label_generation_ = ls.generation;
        label_count_ = ls.count;
        label_checksum_ = ls.checksum;
        persisted_label_content_revision_ = label_content_revision_;  // slot now matches revision
        const auto w2c = read_superblock_wal2c_state(next);
        last_completed_consolidate_epoch_ = w2c.last_completed_consolidate_epoch;
        maintenance_activation_gen_ = w2c.maintenance_activation_sb_generation;
        pid_reuse_activation_gen_ = w2c.pid_reuse_activation_sb_generation;
        if (next.format_version == kQGFormatVersionV3) {
          maintenance_activated_ = true;
          if ((w2c.required_feature_flags & kQgFeatPidGenerationV1) != 0) {
            pid_generation_activated_ = true;
          }
        }
      }
    };
    if (enable_wal_ && !replaying_) {
      // BLOCKER-3 (leg-8, r2 section 1) + B3 completion (leg-9): the constructed next image must
      // pass the SAME pre-write transition validation replay applies, BEFORE it becomes a durable
      // WAL flip. Rejects a generation overflow / geometry drift / file-length / capacity / count /
      // label or activation-state regression the checkpoint should never mint, so replay never has
      // to reconcile a base this build produced but cannot re-derive. The expected HWM is
      // committed_ (admission guaranteed allocated == committed above, and next.num_points ==
      // allocated), so this cross-checks the checkpoint filled num_points from the committed
      // watermark.
      validate_flip_transition(next,
                               next.generation,
                               static_cast<uint8_t>(next_slot),
                               committed_.load(std::memory_order_acquire));
      // Sequence (clause D): flip frame append+fsync (WAL) -> ftruncate + superblock
      // pwrite+fsync (index) -> reset WAL. Each boundary is a crash-matrix cut.
      const auto image =
          std::span<const std::byte>(reinterpret_cast<const std::byte *>(&next), sizeof(next));
      const auto flip = encode_superblock_flip(segment_uid_,
                                               next.generation,
                                               static_cast<uint8_t>(next_slot),
                                               image);
      // BLOCKER-4 (leg-7): the flip append + fsync + on_wal_fsync observer live INSIDE the same
      // catch-all as the superblock write / WAL reset / in-memory adoption. wal_append(fsync)
      // fsyncs the flip (a durable G+1 flip now exists in the WAL) and THEN calls
      // notify_wal_fsync(); a one-shot observer that throws there -- or ANY exception once the
      // flip is durable (superblock pwrite/fsync, WAL reset, a post-flip observer/failpoint, or
      // an allocation inside the adoption) -- must poison this handle. A surviving handle that
      // threw WITHOUT poisoning could retry checkpoint() and write a SECOND, different G+1 flip
      // that replay cannot reconcile ("same generation image differs"). A process crash still
      // rolls forward (replay applies the single durable flip); only a live handle must stop
      // here. Poisoning on an append failure BEFORE durability is conservative and safe (a crash
      // there just rolls back to the old base).
      try {
        wal_append(flip, alaya::wal::WalFile::Sync::fsync);
        wal_failpoint(SegmentOpFailPoint::after_flip_append_before_superblock_write);
        if (::ftruncate(fd_, static_cast<off_t>(file_size)) != 0) {
          poison("checkpoint ftruncate failed");
        }
        write_superblock(next_slot, next);  // pwrite + fsync (index durable)
        wal_failpoint(SegmentOpFailPoint::after_superblock_write_before_wal_reset);
        // Retain exactly one flip marker as the new base; a crash after the durable
        // superblock but before this reset is a safe roll-forward (old WAL suffices).
        op_wal_->reset_to_single_frame(kSegmentOpRecordType, 0, ++wal_op_id_, 0, flip);
        notify_wal_fsync();
        wal_failpoint(SegmentOpFailPoint::after_wal_reset);
        adopt_in_memory();
      } catch (...) {
        poison_current_exception(
            "checkpoint failed after the flip became durable (roll-forward on reopen)");
      }
    } else {
      if (::ftruncate(fd_, static_cast<off_t>(file_size)) != 0) {
        throw std::runtime_error("QGUpdater: ftruncate failed");
      }
      write_superblock(next_slot, next);
      adopt_in_memory();
    }
  }

  void finalize() { checkpoint(); }
