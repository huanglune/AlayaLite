// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Class-body fragment included exactly once by qg_updater.hpp.
// WAL state codec, recovery, and replay implementation.

  struct LabelSlotState {
    uint64_t slot = 0;
    uint64_t generation = 0;
    uint64_t count = 0;
    uint64_t checksum = 0;  // low 32 bits = crc32 of the sorted slot body; high 32 must be 0
  };
  struct TxWatermarkState {
    uint64_t last_committed_txid = 0;
    uint64_t applied_collection_op_id = 0;
  };

  [[nodiscard]] static uint64_t read_superblock_uid(const QGSuperblockV2 &sb) {
    uint64_t uid = 0;
    std::memcpy(&uid, sb.reserved.data() + kUidReservedOffset, sizeof(uid));
    return uid;
  }
  static void write_superblock_uid(QGSuperblockV2 &sb, uint64_t uid) {
    std::memcpy(sb.reserved.data() + kUidReservedOffset, &uid, sizeof(uid));
  }
  [[nodiscard]] static LabelSlotState read_superblock_label_state(const QGSuperblockV2 &sb) {
    LabelSlotState s;
    const auto *base = sb.reserved.data() + kLabelStateReservedOffset;
    std::memcpy(&s.slot, base + 0, 8);
    std::memcpy(&s.generation, base + 8, 8);
    std::memcpy(&s.count, base + 16, 8);
    std::memcpy(&s.checksum, base + 24, 8);
    return s;
  }
  static void write_superblock_label_state(QGSuperblockV2 &sb, const LabelSlotState &s) {
    auto *base = sb.reserved.data() + kLabelStateReservedOffset;
    std::memcpy(base + 0, &s.slot, 8);
    std::memcpy(base + 8, &s.generation, 8);
    std::memcpy(base + 16, &s.count, 8);
    std::memcpy(base + 24, &s.checksum, 8);
  }
  [[nodiscard]] static TxWatermarkState read_superblock_tx_state(const QGSuperblockV2 &sb) {
    TxWatermarkState s;
    const auto *base = sb.reserved.data() + kTxStateReservedOffset;
    std::memcpy(&s.last_committed_txid, base + 0, 8);
    std::memcpy(&s.applied_collection_op_id, base + 8, 8);
    return s;
  }
  static void write_superblock_tx_state(QGSuperblockV2 &sb, const TxWatermarkState &s) {
    auto *base = sb.reserved.data() + kTxStateReservedOffset;
    std::memcpy(base + 0, &s.last_committed_txid, 8);
    std::memcpy(base + 8, &s.applied_collection_op_id, 8);
  }

  // --- 2C maintenance/reuse state (superblock reserved[56..104), design 7.1) ---
  // reserved-relative: magic@56(u64) layout@64(u32) required@68(u32)
  //   last_completed_consolidate_epoch@72(u64) maintenance_activation_gen@80(u64)
  //   pid_reuse_activation_gen@88(u64) max_pid_generation@96(u32) nz_gen_count@100(u32)
  static constexpr size_t kWal2cStateReservedOffset = 56;
  static_assert(kWal2cStateReservedOffset == kWal2cReservedOffset,
                "qg.hpp and qg_updater.hpp disagree on the 2C reserved offset");
  struct Wal2cState {
    uint64_t magic = 0;
    uint32_t layout_version = 0;
    uint32_t required_feature_flags = 0;
    uint64_t last_completed_consolidate_epoch = 0;
    uint64_t maintenance_activation_sb_generation = 0;
    uint64_t pid_reuse_activation_sb_generation = 0;
    uint32_t max_pid_generation = 0;
    uint32_t nonzero_pid_generation_count = 0;
  };
  [[nodiscard]] static Wal2cState read_superblock_wal2c_state(const QGSuperblockV2 &sb) {
    Wal2cState s;
    const auto *b = sb.reserved.data() + kWal2cStateReservedOffset;
    std::memcpy(&s.magic, b + 0, 8);
    std::memcpy(&s.layout_version, b + 8, 4);
    std::memcpy(&s.required_feature_flags, b + 12, 4);
    std::memcpy(&s.last_completed_consolidate_epoch, b + 16, 8);
    std::memcpy(&s.maintenance_activation_sb_generation, b + 24, 8);
    std::memcpy(&s.pid_reuse_activation_sb_generation, b + 32, 8);
    std::memcpy(&s.max_pid_generation, b + 40, 4);
    std::memcpy(&s.nonzero_pid_generation_count, b + 44, 4);
    return s;
  }
  static void write_superblock_wal2c_state(QGSuperblockV2 &sb, const Wal2cState &s) {
    auto *b = sb.reserved.data() + kWal2cStateReservedOffset;
    std::memcpy(b + 0, &s.magic, 8);
    std::memcpy(b + 8, &s.layout_version, 4);
    std::memcpy(b + 12, &s.required_feature_flags, 4);
    std::memcpy(b + 16, &s.last_completed_consolidate_epoch, 8);
    std::memcpy(b + 24, &s.maintenance_activation_sb_generation, 8);
    std::memcpy(b + 32, &s.pid_reuse_activation_sb_generation, 8);
    std::memcpy(b + 40, &s.max_pid_generation, 4);
    std::memcpy(b + 44, &s.nonzero_pid_generation_count, 4);
  }

  // --- 2A label slot serialization + durable double-buffered slot files ---
  // Slot body = explicit little-endian {pid u32, pid_generation u32, label u64}
  // per binding, ascending pid (std::map order). Checksum = crc32 of that body.
  [[nodiscard]] static std::vector<std::byte> serialize_label_slot(const LabelBindings &lb) {
    std::vector<std::byte> out;
    out.reserve(lb.bindings.size() * 16);
    for (const auto &[pid, binding] : lb.bindings) {
      alaya::wal::put_u32(out, static_cast<std::uint32_t>(pid));
      alaya::wal::put_u32(out, binding.pid_generation);  // W2: real per-PID incarnation
      alaya::wal::put_u64(out, binding.label);
    }
    return out;
  }

  // Load + fully validate a slot file against the superblock tuple (all failures
  // poison / fail-closed). Caller guarantees a non-canonical-empty tuple.
  LabelBindings load_label_slot_bindings(const std::string &path,
                                         uint64_t count,
                                         uint64_t checksum,
                                         uint64_t num_points,
                                         bool pid_active) {  // wal-2c MAJOR-9
    if ((checksum >> 32) != 0) {
      poison("label slot checksum high 32 bits must be zero");
    }
    if (count > static_cast<uint64_t>((std::numeric_limits<std::size_t>::max)()) / 16) {
      poison("label slot count overflows the slot size");
    }
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      poison("label slot file is missing for a non-empty label tuple");
    }
    const auto file_size = std::filesystem::file_size(path, ec);
    if (ec) {
      poison("cannot stat the label slot file");
    }
    const uint64_t expected = count * 16;
    if (file_size != expected) {
      poison("label slot size does not equal count*16");
    }
    std::vector<std::byte> bytes(static_cast<std::size_t>(expected));
    if (expected > 0) {
      std::ifstream in(path, std::ios::binary);
      if (!in) {
        poison("cannot open the label slot file");
      }
      in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
      if (!in) {
        poison("short read on the label slot file");
      }
    }
    if (static_cast<uint64_t>(alaya::wal::crc32(bytes)) != (checksum & 0xffffffffULL)) {
      poison("label slot checksum mismatch");
    }
    LabelBindings lb;
    uint64_t prev_pid = 0;
    bool first = true;
    for (uint64_t i = 0; i < count; ++i) {
      const std::size_t off = static_cast<std::size_t>(i) * 16;
      const auto pid = alaya::wal::get_u32(bytes, off);
      const auto gen = alaya::wal::get_u32(bytes, off + 4);
      const auto label = alaya::wal::get_u64(bytes, off + 8);
      // wal-2c MAJOR-9: every binding must reference a committed row -- a forged slot with a
      // pid >= the base HWM would map a label onto a non-existent (or future) PID.
      if (static_cast<uint64_t>(pid) >= num_points) {
        poison("label slot entry references a pid at or beyond the committed high-water mark");
      }
      // W2c relaxes this once pid_generation_v1 is activated; until then (append-only)
      // every slot entry must carry generation 0 (fail-closed on a forged non-zero). The
      // caller passes the pid-reuse state OF THE IMAGE being validated (leg-7 BLOCKER-3: a
      // flip image is validated against its own pid state before it becomes the base).
      if (gen != 0 && !pid_active) {
        poison("label slot entry has a non-zero generation before pid-reuse activation");
      }
      if (!first && pid <= prev_pid) {
        poison("label slot entries are not strictly ascending by pid");
      }
      first = false;
      prev_pid = pid;
      lb.bindings.emplace(static_cast<PID>(pid), PidBinding{gen, label});
    }
    return lb;
  }

  // Overwrite the inactive slot in place (its directory entry is already durable
  // from precreate_label_slots, so no rename — B-05). Never touches the active slot.
  void write_label_slot_file(int slot, const std::vector<std::byte> &body) {
    const auto &path = label_slot_path_[slot];
    {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      if (!out) {
        poison("cannot open the inactive label slot for write");
      }
      if (!body.empty()) {
        out.write(reinterpret_cast<const char *>(body.data()),
                  static_cast<std::streamsize>(body.size()));
      }
      out.flush();
      if (!out) {
        poison("cannot write the inactive label slot");
      }
    }
    try {
      alaya::platform::sync_file_or_throw(path);
    } catch (const std::exception &error) {
      poison(std::string("label slot fsync failed: ") + error.what());
    }
    if (io_observer_ != nullptr && !replaying_ && io_observer_->on_label_slot_fsync) {
      io_observer_->on_label_slot_fsync();
    }
  }

  // B-05 root-cause fix: pre-create BOTH slot files (empty) + fsync each + fsync
  // the parent directory once, so every later flip only overwrites an inactive
  // slot's contents (its directory entry is already durable). Existing (non-empty)
  // slot files are left untouched.
  void precreate_label_slots() {
    bool created = false;
    std::error_code ec;
    for (int s = 0; s < 2; ++s) {
      if (!std::filesystem::exists(label_slot_path_[s], ec)) {
        {
          std::ofstream f(label_slot_path_[s], std::ios::binary | std::ios::trunc);
          if (!f) {
            poison("cannot pre-create a label slot file");
          }
        }
        try {
          alaya::platform::sync_file_or_throw(label_slot_path_[s]);
        } catch (const std::exception &error) {
          poison(std::string("label slot pre-create fsync failed: ") + error.what());
        }
        created = true;
      }
    }
    if (created) {
      try {
        alaya::platform::sync_directory_or_throw(
            std::filesystem::path(label_slot_path_[0]).parent_path());
      } catch (const std::exception &error) {
        poison(std::string("label slot directory fsync failed: ") + error.what());
      }
    }
  }

  // Snapshot pointer accessors: a tiny dedicated mutex (never the handle mutex)
  // makes the shared_ptr swap/copy race-free and portable (no atomic<shared_ptr>).
  [[nodiscard]] std::shared_ptr<const LabelBindings> load_label_snapshot() const {
    const std::lock_guard<std::mutex> guard(label_snapshot_mutex_);
    return label_snapshot_;
  }
  void store_label_snapshot(std::shared_ptr<const LabelBindings> snap) {
    const std::lock_guard<std::mutex> guard(label_snapshot_mutex_);
    label_snapshot_ = std::move(snap);
  }

  // Publish the immutable snapshot from the recovery scratch map (release via mutex).
  void publish_label_snapshot() {
    auto snap = std::make_shared<LabelBindings>();
    snap->bindings = label_working_;
    store_label_snapshot(std::shared_ptr<const LabelBindings>(std::move(snap)));
  }

  // Adopt the label + tx state of a base into the running/base watermarks and the
  // recovery scratch map (used at recovery start and on a flip adoption in replay).
  void adopt_label_state(const QGSuperblockV2 &sb) {
    const auto ls = read_superblock_label_state(sb);
    const auto tx = read_superblock_tx_state(sb);
    // 2C maintenance state travels with every base (recovery start + flip re-adopt).
    const auto w2c = read_superblock_wal2c_state(sb);
    last_completed_consolidate_epoch_ = w2c.last_completed_consolidate_epoch;
    maintenance_activated_ = sb.format_version == kQGFormatVersionV3 &&
                             (w2c.required_feature_flags & kQgFeatMaintenanceTxV1) != 0;
    pid_generation_activated_ = sb.format_version == kQGFormatVersionV3 &&
                                (w2c.required_feature_flags & kQgFeatPidGenerationV1) != 0;
    maintenance_activation_gen_ = w2c.maintenance_activation_sb_generation;
    pid_reuse_activation_gen_ = w2c.pid_reuse_activation_sb_generation;
    // Fail-closed activation-generation bounds (design 7.1 / JC-16): an activated base
    // must stamp a non-zero activation generation no newer than its own superblock
    // generation. A zero or future generation is a forged / corrupt v3 base.
    if (maintenance_activated_ &&
        (maintenance_activation_gen_ == 0 || maintenance_activation_gen_ > sb.generation)) {
      poison("v3 base has an out-of-range maintenance activation generation");
    }
    if (pid_generation_activated_ &&
        (pid_reuse_activation_gen_ == 0 || pid_reuse_activation_gen_ > sb.generation)) {
      poison("v3 base has an out-of-range pid-reuse activation generation");
    }
    // BLOCKER-5: pid reuse depends on the maintenance pair, so a pid-active base MUST be
    // maintenance-active. (qg_superblock_supported already enforces this on the accepted
    // base; assert it here too so a checkpoint never reverts a pid base to a v2 image.)
    if (pid_generation_activated_ && !maintenance_activated_) {
      poison("v3 base is pid-reuse-active but not maintenance-active (feature dependency)");
    }
    // BLOCKER-3: pid reuse activates at or after maintenance -- a pid activation generation
    // BELOW the maintenance activation generation is an impossible / forged ordering. (The
    // selector's qg_superblock_supported enforces this for the accepted base; assert it here
    // too so a flip image adopted mid-replay is held to the same ordering.)
    if (pid_generation_activated_ && maintenance_activated_ &&
        pid_reuse_activation_gen_ < maintenance_activation_gen_) {
      poison("v3 base pid-reuse activation generation precedes maintenance activation");
    }
    last_committed_txid_ = tx.last_committed_txid;
    applied_collection_op_id_ = tx.applied_collection_op_id;
    base_committed_txid_ = tx.last_committed_txid;
    base_applied_op_id_ = tx.applied_collection_op_id;
    base_num_points_ = sb.num_points;
    label_generation_ = ls.generation;
    label_count_ = ls.count;
    label_checksum_ = ls.checksum;
    persisted_label_content_revision_ = label_content_revision_;  // the just-loaded slot is clean
    staged_binds_.clear();
    if (ls.generation == 0 && ls.count == 0 && ls.checksum == 0) {
      // MAJOR-4: an empty slot holds zero bindings, so its actual (max_generation,
      // reuse_count) summary is (0,0). A superblock declaring a non-zero summary over an
      // empty slot is a forged/impossible state -> fail closed (the summary cross-check
      // below is skipped by this early return, so it must be enforced here).
      if (w2c.max_pid_generation != 0 || w2c.nonzero_pid_generation_count != 0) {
        poison("empty label slot but the superblock declares a non-zero pid-generation summary");
      }
      // BLOCKER-3: an out-of-range slot index must fail closed even for the empty tuple --
      // silently coercing slot>1 to 0 would accept a forged superblock.
      if (ls.slot > 1) {
        poison("label slot index out of range in the superblock (empty tuple)");
      }
      active_label_slot_ = static_cast<int>(ls.slot);
      label_working_.clear();
      return;  // canonical legacy empty: the slot file may be absent
    }
    if (ls.slot > 1) {
      poison("label slot index out of range in the superblock");
    }
    active_label_slot_ = static_cast<int>(ls.slot);
    auto lb = load_label_slot_bindings(label_slot_path_[ls.slot],
                                       ls.count,
                                       ls.checksum,
                                       sb.num_points,
                                       pid_generation_activated_);
    // Activation summary cross-check (design 7.1 / JC-16, sixth fail-closed condition):
    // the slot's actual max non-zero generation and reuse-binding count must equal the
    // summary the writing checkpoint stamped. A mismatch means a slot swapped out from
    // under its superblock summary -> fail closed. Only meaningful once pid reuse is
    // activated (before activation every generation is 0, so both sides are 0).
    if (pid_generation_activated_) {
      uint32_t max_gen = 0;
      uint32_t nz_count = 0;
      for (const auto &[pid, binding] : lb.bindings) {
        (void)pid;
        if (binding.pid_generation != 0) {
          ++nz_count;
          max_gen = (std::max)(max_gen, binding.pid_generation);
        }
      }
      if (max_gen != w2c.max_pid_generation || nz_count != w2c.nonzero_pid_generation_count) {
        poison("label slot pid-generation summary does not match the superblock");
      }
    }
    label_working_ = std::move(lb.bindings);
  }

  // Set the lock-free poison latch with NO allocation (BLOCKER-2/3): the atomic is the
  // authoritative fail-closed signal, so it must be settable even while the process is
  // out of memory (e.g. inside a catch handling a std::bad_alloc). The reason string is
  // a best-effort diagnostic that may fail to allocate; ensure_writable() reads the
  // atomic first so a reason-less poison still fails every subsequent writable check.
  void poison_latch() noexcept { poisoned_.store(true, std::memory_order_release); }
  [[noreturn]] void poison(const std::string &reason) {
    // Latch first (noexcept) so a concurrent search (which never takes the handle write
    // mutex) observes the poison via the atomic without racing the std::string; and so a
    // bad_alloc while building the reason/throw message below still leaves the handle
    // permanently fail-closed.
    poison_latch();
    try {
      if (poison_reason_.empty()) {
        poison_reason_ = reason;
      }
    } catch (...) {
      // The latch is set; a missing diagnostic never un-poisons the handle.
    }
    throw std::runtime_error(poison_reason_.empty()
                                 ? std::string("QGUpdater op-WAL writer poisoned")
                                 : "QGUpdater op-WAL writer poisoned: " + poison_reason_);
  }
  void ensure_writable() const {
    // BLOCKER-2: check the atomic latch FIRST. A poison() that could not allocate its
    // reason string still set the atomic; reading only poison_reason_ would then let a
    // writer keep going (the bad_alloc-in-catch bypass). The reason is appended only as
    // a diagnostic when it is present.
    if (poisoned_.load(std::memory_order_acquire)) {
      throw std::runtime_error(poison_reason_.empty()
                                   ? std::string("QGUpdater op-WAL writer is poisoned")
                                   : "QGUpdater op-WAL writer is poisoned: " + poison_reason_);
    }
  }
  // BLOCKER-2: catch-all poison used by every mutating region past a durable/reservation
  // boundary. Latches (noexcept) BEFORE any allocation so a non-std::exception failpoint
  // throw (e.g. `throw 7`) or a bad_alloc cannot escape a still-non-idle handle without
  // fail-closing it, then rethrows the ORIGINAL exception (preserving its type). Must be
  // called only from inside a catch clause (uses `throw;`).
  [[noreturn]] void poison_current_exception(const char *fallback_reason) {
    poison_latch();
    try {
      if (poison_reason_.empty()) {
        poison_reason_ = fallback_reason;
      }
    } catch (...) {
      // The latch is set; a missing diagnostic never un-poisons the handle.
    }
    throw;  // rethrow the exception currently being handled
  }
  void wal_failpoint(SegmentOpFailPoint fp) {
    if (params_.failpoint_hook) {
      params_.failpoint_hook(fp);
    }
  }
  void notify_wal_fsync() {
    if (io_observer_ != nullptr && !replaying_ && io_observer_->on_wal_fsync) {
      io_observer_->on_wal_fsync();
    }
  }
  void notify_index_fsync() {
    if (io_observer_ != nullptr && !replaying_ && io_observer_->on_index_fsync) {
      io_observer_->on_index_fsync();
    }
  }

  [[nodiscard]] bool has_staged_edges() {
    for (auto &stripe : staged_) {
      const std::lock_guard<std::mutex> guard(stripe.mutex);
      if (!stripe.by_target.empty()) {
        return true;
      }
    }
    return false;
  }

  // Append one SEGMENT_OP frame; poison the writer on any WAL error. batch_id is 0
  // for kind=1..6 (byte-identical to before); kind=7/8 carry tx_id in batch_id, and
  // replay validates frame.batch_id == payload.tx_id (B-04).
  void wal_append(const std::vector<std::byte> &payload,
                  alaya::wal::WalFile::Sync sync,
                  std::uint64_t batch_id = 0) {
    try {
      op_wal_->append(kSegmentOpRecordType, 0, ++wal_op_id_, batch_id, payload, sync);
    } catch (const std::exception &error) {
      poison(std::string("WAL append failed: ") + error.what());
    }
    if (sync == alaya::wal::WalFile::Sync::fsync) {
      notify_wal_fsync();
    }
  }

  // Force the buffered WAL prefix durable (group commit / force-before-writeback).
  void force_wal() {
    if (!enable_wal_ || replaying_ || op_wal_ == nullptr) {
      return;
    }
    ensure_writable();
    try {
      op_wal_->fsync();
    } catch (const std::exception &error) {
      poison(std::string("WAL force failed: ") + error.what());
    }
    notify_wal_fsync();
  }

  // Append a whole-page after-image (row_patch). Under the armed reuse contract, the existing pid
  // field carries the logical PID actually touched by this RMW (I-2 line evidence); legacy double-
  // false writers retain the historical page-first value byte-for-byte. The wire layout is
  // unchanged.
  void log_page_after_image(PID pid_in_page, const char *page_bytes) {
    const uint64_t offset = page_offset(pid_in_page);
    const auto first_pid = static_cast<uint64_t>(page_index(pid_in_page) * npp_);
    const uint64_t evidence_pid = (pid_generation_activated_ || enable_pid_reuse_)
                                      ? static_cast<uint64_t>(pid_in_page)
                                      : first_pid;
    auto payload =
        encode_row_patch(segment_uid_,
                         superblock_.generation,
                         evidence_pid,
                         offset,
                         std::span<const std::byte>(reinterpret_cast<const std::byte *>(page_bytes),
                                                    page_size_));
    wal_append(payload, alaya::wal::WalFile::Sync::buffered);
  }

  // Establish lineage and open the WAL at ctor time; run recovery if non-empty.
  void open_op_wal_and_recover() {
    segment_uid_ = read_superblock_uid(superblock_);
    const bool fresh_lineage = segment_uid_ == 0;
    if (fresh_lineage) {
      std::random_device rd;
      segment_uid_ = (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
      if (segment_uid_ == 0) {
        segment_uid_ = 0x9E3779B97F4A7C15ULL;
      }
    }
    label_slot_path_[0] = qg_.index_file_name_ + ".labels.slot0";
    label_slot_path_[1] = qg_.index_file_name_ + ".labels.slot1";
    store_label_snapshot(std::make_shared<const LabelBindings>());
    // wal-2c BLOCKER-3: stream recovery -- the constructor finds the torn-tail boundary
    // without loading the whole .opwal, and every replay pass below drives off visit_frames.
    op_wal_ = std::make_unique<alaya::wal::WalFile>(qg_.index_file_name_ + ".opwal",
                                                    /*stream_recovery=*/true);
    if (fresh_lineage) {
      // A non-empty .opwal on an UNSTAMPED (uid==0) superblock is the legal
      // fresh-enable crash window: a fresh checkpoint fsynced its flip frame but
      // crashed before the superblock write, so the base is still unstamped yet the
      // WAL holds an orphan flip. No committed op-WAL data can exist on an unstamped
      // base, so the orphan is discardable -- checkpoint() below overwrites the WAL
      // via reset_to_single_frame. Guard: only flip frames are legal here; anything
      // else is unexpected (data on an unstamped base) and poisons (fail-closed).
      alaya::wal::WalFile::
          visit_frames(op_wal_->path(), [&](const alaya::wal::ScannedFrame &frame) -> bool {
            SegmentOp probe;
            try {
              probe = decode_segment_op(frame.payload);
            } catch (const std::exception &) {
              poison(
                  "enable_wal on an unstamped superblock but the .opwal has an undecodable frame");
            }
            if (probe.kind != SegmentOpKind::superblock_flip) {
              poison(
                  "enable_wal on an unstamped superblock but the .opwal has committed op frames");
            }
            return true;
          });
      label_working_.clear();
      checkpoint();              // stamps uid + canonical label/tx tuple; overwrites the WAL
      publish_label_snapshot();  // empty snapshot
      precreate_label_slots();   // B-05: both slots + parent dir durable before any content flip
      return;
    }
    adopt_label_state(superblock_);  // load persisted bindings + tx watermarks from the base
    // Always converge through the full recovery pass (design section 2.1): even an
    // empty op-WAL must reach rebuild_state_after_replay so routing/free/hidden are
    // derived once from the on-disk trailers under the replaying_ guard.
    replay_and_rebuild();      // promotes staged bundles into label_working_ (W3)
    publish_label_snapshot();  // seal the recovered label state (tail convergence)
    precreate_label_slots();
  }

  // Canonical PID-reuse bundle replay staging (design 3.5 / codex B.5). Unlike the 2A
  // LabelBindStage, a canonical bind carries the reused-incarnation generation (0 for a
  // dense append). Preimage/final page refs are FrameLocations (re-read + re-CRC'd via
  // WalFile::read_frame at kind=8) so recovery memory stays O(touched pages).
  struct CanonicalBind {
    uint64_t row_op_id = 0;
    PID pid = 0;
    uint32_t pid_generation = 0;
    uint64_t label = 0;
  };
  struct CanonicalBundleStage {
    uint64_t txid = 0;
    uint64_t segment_generation = 0;
    uint64_t begin_offset = 0;
    bool saw_row_patch = false;  // once a kind=1 is seen, any further kind=7 poisons
    std::vector<CanonicalBind> binds;
    std::map<size_t, alaya::wal::FrameLocation>
        preimage_refs;                                        // page -> FIRST kind=1 (pre-tx FREE)
    std::map<size_t, alaya::wal::FrameLocation> latest_refs;  // page -> LAST kind=1 (final image)
  };

  // Recovery-only commit unit for armed standalone / legacy effects. Kind=1 has no txid, so pages
  // begin unowned and may be attached to exactly one legacy tx by the surrounding kind=7 grammar.
  // A second candidate tx poisons before any page write. Full pages stay in the WAL; memory holds
  // only page -> FrameLocation plus bounded row/tombstone evidence. Every commit re-reads and
  // re-validates the selected frames (including CRC in WalFile::read_frame).
  struct EffectCommitUnit {
    uint64_t generation = 0;
    uint64_t begin_offset = 0;
    uint64_t legacy_txid = 0;  // 0 => standalone/unowned; legacy producer txids are strictly > 0
    bool saw_standalone_tombstone = false;
    std::map<size_t, alaya::wal::FrameLocation> latest_refs;
    std::map<size_t, std::set<PID>> allocation_evidence;  // page -> un-published touched PIDs
    std::set<PID> staged_legacy_tombstones;

    [[nodiscard]] bool has_staged_effects() const {
      return !latest_refs.empty() || !allocation_evidence.empty() ||
             !staged_legacy_tombstones.empty();
    }
    [[nodiscard]] bool active() const { return has_staged_effects() || saw_standalone_tombstone; }
    void clear() { *this = EffectCommitUnit{}; }
  };
  // True once PID reuse is activated (design 7.2) AND this frame's generation is at or
  // after the reuse activation generation. Dormant pre-activation (activation gen 0 =>
  // always false => every kind=7 takes the legacy 2A staging path unchanged).
  [[nodiscard]] bool is_canonical_generation(uint64_t seg_generation) const {
    // BLOCKER-3: gate on pid_generation_activated_ too. A v2 base whose reserved bytes hold
    // a stray non-zero pid_reuse_activation_gen_ (now rejected up front, but defense in
    // depth) must never route a legacy kind=7/8 into the canonical lane.
    return pid_generation_activated_ && pid_reuse_activation_gen_ != 0 &&
           seg_generation >= pid_reuse_activation_gen_;
  }

  // NEW-BLOCKER-1/2 (leg-8, r2 sections 3): disposition of a STANDALONE top-level effect op
  // (row_patch / tombstone / publish -- NOT inside a maintenance epoch or a canonical bundle) once
  // pid reuse is armed. Classified by the frame's generation against the replay cursor
  // (superblock_.generation, advanced by replay_flip):
  //   - gen == cursor: a legit new write at the current base (e.g. a tombstone's trailer
  //     after-image) -> APPLY (byte-identical to the pre-leg-8 path).
  //   - gen  < cursor: an already-absorbed historical-prefix frame -- a checkpoint flip advanced
  //   the
  //     cursor past it, so the base at the cursor generation already reflects it -> VALIDATE-ONLY
  //     (verify structural legality but never re-apply). This is the NEW-BLOCKER-1 fix: a stale /
  //     forged old-generation after-image can no longer clobber the higher-generation base (a
  //     one-byte page edit stays out; a generation-unaware kind=2 ABA can no longer tombstone a
  //     reused incarnation), while the legit crash-after-flip-before-reset prefix -- whose
  //     effect ops legitimately precede the absorbing flip -- is not falsely convicted.
  //   - gen  > cursor: impossible in a valid log (a frame is written at the then-current base
  //     generation, which only reaches G+1 AFTER its flip is replayed) -> POISON (fail closed).
  // Gated on pid_generation_activated_ || enable_pid_reuse_ so the legacy 2A/2B replay path is
  // byte-for-byte unchanged (this returns kApply there, exactly the pre-leg-8 behavior).
  enum class StandaloneEffect { kApply, kValidateOnly, kPoison };
  [[nodiscard]] StandaloneEffect classify_standalone_effect(uint64_t seg_generation) const {
    if (!pid_generation_activated_ && !enable_pid_reuse_) {
      return StandaloneEffect::kApply;  // legacy 2A/2B: unchanged
    }
    const uint64_t cursor = superblock_.generation;
    if (seg_generation == cursor) {
      return StandaloneEffect::kApply;
    }
    if (seg_generation < cursor) {
      return StandaloneEffect::kValidateOnly;
    }
    return StandaloneEffect::kPoison;
  }

  // BLOCKER-5: narrow a wire (uint64) PID to PID only after a range check. A wire PID at or
  // above kPidMax would otherwise wrap to a small in-range value and slip past the later
  // [old_hwm,new_hwm) / < old_hwm bound checks (reuse of a still-referenced row).
  [[nodiscard]] PID canonical_checked_pid(uint64_t wire_pid) {
    if (wire_pid >= static_cast<uint64_t>(kPidMax)) {
      poison("op-WAL canonical bundle pid is out of range (would wrap the PID width)");
    }
    return static_cast<PID>(wire_pid);
  }

  // Extract the trailer for `pid`'s slot from a decoded whole-page row_patch after-image.
  [[nodiscard]] QGRowTrailer canonical_page_trailer(const SegmentOp &op, PID pid) {
    if (op.bytes.size() != page_size_) {
      poison("op-WAL canonical page after-image is not a whole page");
    }
    return qg_read_page_trailer(reinterpret_cast<const char *>(op.bytes.data()),
                                page_size_,
                                npp_,
                                static_cast<size_t>(pid) % npp_);
  }

  // Open a canonical bundle lane on the first kind=7 of a reuse-generation transaction.
  void canonical_open_lane(CanonicalBundleStage &st,
                           const SegmentOp &op,
                           const alaya::wal::ScannedFrame &frame,
                           bool &in_canonical_bundle) {
    if (frame.batch_id != op.tx_id) {
      poison("op-WAL canonical label_bind frame batch_id != payload tx_id");
    }
    st = CanonicalBundleStage{};
    st.txid = op.tx_id;
    st.segment_generation = op.segment_generation;
    st.begin_offset = frame.offset;
    st.binds.push_back({op.row_op_id, canonical_checked_pid(op.pid), op.pid_generation, op.label});
    in_canonical_bundle = true;
  }

  // One frame inside an open canonical lane. Legal frames: consecutive same-txid kind=7
  // (only before the first kind=1), reuse-preimage + final kind=1, the matching kind=8;
  // anything else poisons (design 3.5 / codex B.5).
  void canonical_lane_step(CanonicalBundleStage &st,
                           const SegmentOp &op,
                           const alaya::wal::ScannedFrame &frame,
                           bool &in_canonical_bundle,
                           uint64_t &committed_watermark) {
    switch (op.kind) {
      case SegmentOpKind::label_bind: {
        if (st.saw_row_patch) {
          poison("op-WAL canonical bundle has a kind=7 after a kind=1");
        }
        if (op.tx_id != st.txid) {
          poison("op-WAL canonical bundle mixes tx_ids across its kind=7 frames");
        }
        if (op.segment_generation != st.segment_generation) {
          poison("op-WAL canonical bundle kind=7 generation != the bundle generation");
        }
        if (frame.batch_id != op.tx_id) {
          poison("op-WAL canonical label_bind frame batch_id != payload tx_id");
        }
        st.binds.push_back(
            {op.row_op_id, canonical_checked_pid(op.pid), op.pid_generation, op.label});
        break;
      }
      case SegmentOpKind::row_patch: {
        if (op.segment_generation != st.segment_generation) {
          poison("op-WAL canonical bundle row_patch generation != the bundle generation");
        }
        const size_t page = replay_validate_row_patch_geometry(op);
        st.saw_row_patch = true;
        st.preimage_refs.try_emplace(page,
                                     alaya::wal::FrameLocation{frame.offset, frame.size});  // first
        st.latest_refs[page] = alaya::wal::FrameLocation{frame.offset, frame.size};         // last
        break;
      }
      case SegmentOpKind::tx_publish: {
        if (op.tx_id != st.txid) {
          poison("op-WAL canonical tx_publish tx_id != the bundle tx_id");
        }
        // MAJOR-1: the kind=8 must carry the SAME segment generation as the kind=7/kind=1
        // frames it commits (they are all validated == st.segment_generation). Otherwise a
        // CRC-legal kind7(G) -> kind1(G) -> kind8(G+1) would commit at the wrong generation.
        if (op.segment_generation != st.segment_generation) {
          poison("op-WAL canonical tx_publish generation != the bundle generation");
        }
        if (frame.batch_id != op.tx_id) {
          poison("op-WAL canonical tx_publish frame batch_id != payload tx_id");
        }
        canonical_finalize_bundle(st, op, committed_watermark);
        in_canonical_bundle = false;
        break;
      }
      default:
        poison("op-WAL canonical bundle contains an unexpected op kind (only 7/1/8 are legal)");
    }
  }

  // Validate a canonical bundle at its kind=8 and, for a NEW transaction, apply its final
  // pages + promote bindings (design 3.4/3.5, B-2C-02). Any inconsistency poisons. An
  // already-absorbed prefix (txid <= base) is validate-only and never re-applied.
  void canonical_finalize_bundle(CanonicalBundleStage &st,
                                 const SegmentOp &op,
                                 uint64_t &committed_watermark) {
    if (op.binding_count == 0) {
      poison("op-WAL canonical tx_publish binding_count must be >= 1");
    }
    if (st.binds.size() != op.binding_count) {
      poison("op-WAL canonical staged bind count != binding_count");
    }
    const bool absorbed = op.tx_id <= base_committed_txid_;
    // Structural set validation (both paths): row_op_id is exactly {0..count-1} and
    // unique; pids are globally unique. Split the binds into append (gen 0) and reuse.
    std::vector<uint8_t> row_seen(static_cast<size_t>(op.binding_count), 0);
    std::unordered_set<uint64_t> pid_seen;
    pid_seen.reserve(st.binds.size() * 2);
    uint64_t append_count = 0;
    for (const auto &b : st.binds) {
      if (b.row_op_id >= op.binding_count || row_seen[static_cast<size_t>(b.row_op_id)] != 0) {
        poison("op-WAL canonical bundle row_op_id set is not exactly {0..count-1}");
      }
      row_seen[static_cast<size_t>(b.row_op_id)] = 1;
      if (!pid_seen.insert(static_cast<uint64_t>(b.pid)).second) {
        poison("op-WAL canonical bundle has a duplicate pid");
      }
      if (b.pid_generation == 0) {
        ++append_count;
      }
    }
    // HWM algebra + generation chain (design 3.4). For an absorbed prefix the running
    // watermark is already the POST-absorbed base, so the pre-transaction algebra cannot
    // be reconstructed here without the retained-kind=6 model (JC-22): skip the algebra +
    // page apply, keeping the base's durable effect. A NEW bundle validates + applies.
    if (!absorbed) {
      // BLOCKER-5: a NEW (non-absorbed) canonical bundle must have been written at the
      // current replay-cursor generation (superblock_.generation, advanced by replay_flip).
      // Otherwise a CRC-legal but cross-generation bundle -- kind7(G+1)/kind1(G+1)/kind8(G+1)
      // spliced onto a base at generation G -- would be applied at the wrong generation.
      // (The maintenance epoch binds its BEGIN to the same cursor; canonical had no such
      // check.) An absorbed prefix legitimately carries an older generation, so it is
      // exempt -- its historical generation would need the retained-kind=6 model (JC-22).
      if (st.segment_generation != superblock_.generation) {
        poison("op-WAL canonical bundle generation != the replay cursor generation");
      }
      const uint64_t old_hwm = committed_watermark;
      const uint64_t new_hwm = op.new_pid_watermark;
      if (op.tx_id <= last_committed_txid_) {
        poison("op-WAL canonical tx_publish tx_id is not strictly increasing");
      }
      if (op.applied_collection_op_id < applied_collection_op_id_) {
        poison("op-WAL canonical tx_publish applied_collection_op_id regressed");
      }
      if (new_hwm != old_hwm + append_count) {
        poison("op-WAL canonical tx_publish new_pid_watermark != old_hwm + append_count");
      }
      if (new_hwm > static_cast<uint64_t>(kPidMax) || new_hwm > row_generations_.size()) {
        poison("op-WAL canonical tx_publish new_pid_watermark exceeds the PID capacity");
      }
      std::vector<uint8_t> append_seen(static_cast<size_t>(append_count), 0);
      for (const auto &b : st.binds) {
        const uint64_t pid = static_cast<uint64_t>(b.pid);
        if (b.pid_generation == 0) {
          if (pid < old_hwm || pid >= new_hwm) {
            poison("op-WAL canonical append pid outside [old_hwm, new_hwm)");
          }
          append_seen[static_cast<size_t>(pid - old_hwm)] = 1;
        } else {
          if (pid >= old_hwm) {
            poison("op-WAL canonical reused pid is not below old_hwm");
          }
          uint32_t old_gen = 0;
          const auto it = label_working_.find(static_cast<PID>(pid));
          if (it != label_working_.end()) {
            old_gen = it->second.pid_generation;
          }
          if (old_gen == (std::numeric_limits<uint32_t>::max)()) {
            poison("op-WAL canonical reuse of a saturated (UINT32_MAX) generation pid");
          }
          if (b.pid_generation != old_gen + 1) {
            poison("op-WAL canonical reused pid generation is not old_gen + 1");
          }
          const size_t page = page_index(static_cast<PID>(pid));
          const auto pit = st.preimage_refs.find(page);
          if (pit == st.preimage_refs.end()) {
            poison("op-WAL canonical reused pid has no pre-transaction FREE preimage page");
          }
          const auto preimg = alaya::wal::WalFile::read_frame(op_wal_->path(), pit->second);
          const SegmentOp pop = decode_segment_op(preimg.payload);
          const QGRowTrailer ptr = canonical_page_trailer(pop, static_cast<PID>(pid));
          if ((ptr.flags & (kQGRowTombstone | kQGRowFree)) != (kQGRowTombstone | kQGRowFree)) {
            poison("op-WAL canonical reused pid preimage trailer is not TOMBSTONE|FREE");
          }
        }
      }
      for (uint8_t seen : append_seen) {
        if (seen == 0) {
          poison("op-WAL canonical append pids do not densely cover [old_hwm, new_hwm)");
        }
      }
    }
    // B-2C-02 (both paths): every bound pid has a final page whose trailer is live.
    for (const auto &b : st.binds) {
      const size_t page = page_index(b.pid);
      const auto lit = st.latest_refs.find(page);
      if (lit == st.latest_refs.end()) {
        poison("op-WAL canonical bound pid has no final page after-image (B-2C-02)");
      }
      const auto finalf = alaya::wal::WalFile::read_frame(op_wal_->path(), lit->second);
      const SegmentOp fop = decode_segment_op(finalf.payload);
      const QGRowTrailer ftr = canonical_page_trailer(fop, b.pid);
      if (ftr.valid_degree > deg_) {
        poison("op-WAL canonical final trailer valid_degree exceeds the graph degree");
      }
      if ((ftr.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
        poison("op-WAL canonical bound pid final trailer is not live (B-2C-02)");
      }
    }
    if (absorbed) {
      return;  // JC-22: validate-only; the selected base already holds this bundle's effect.
    }
    // Apply the final pages in page order, then promote {generation,label} and advance
    // the watermarks (design 3.5 apply order: pages -> bindings -> HWM/txid/applied).
    for (const auto &[page, ref] : st.latest_refs) {
      (void)page;
      const auto finalf = alaya::wal::WalFile::read_frame(op_wal_->path(), ref);
      const SegmentOp fop = decode_segment_op(finalf.payload);
      write_at(fop.offset, reinterpret_cast<const char *>(fop.bytes.data()), fop.bytes.size());
    }
    for (const auto &b : st.binds) {
      label_working_.insert_or_assign(b.pid, PidBinding{b.pid_generation, b.label});
    }
    ++label_content_revision_;
    committed_watermark = op.new_pid_watermark;
    last_committed_txid_ = op.tx_id;
    applied_collection_op_id_ = op.applied_collection_op_id;
  }

  void replay_and_rebuild() {
    replaying_ = true;
    free_chain_rebuild_complete_ = false;  // set true again only after the canonical rebuild below
    uint64_t committed_watermark = committed_.load(std::memory_order_acquire);
    // Maintenance epoch state machine (design section 1.3): row_patch frames between
    // a durable BEGIN and END stage a latest-per-page after-image but do NOT touch
    // the index; END redoes them all; an unmatched BEGIN (EOF) is discarded and the
    // WAL is semantically truncated back to the BEGIN boundary.
    bool in_epoch = false;
    bool epoch_apply = false;  // false => absorbed-by-base prefix: validate order, do not re-apply
    uint64_t epoch_id = 0;
    uint64_t epoch_generation = 0;  // wal-2c BLOCKER-5: the generation the epoch was written at
    uint64_t epoch_begin_offset = 0;
    std::unordered_map<size_t, std::vector<std::byte>> epoch_pages;  // page_index -> latest bytes
    // Canonical PID-reuse bundle lane (design 3.5): mutually exclusive with the
    // maintenance epoch above; dormant until reuse is activated (is_canonical_generation).
    bool in_canonical_bundle = false;
    CanonicalBundleStage cstage;
    // Armed standalone/legacy commit-unit state. Unlike the former global page->bytes map, this
    // records an explicit owner and keeps only WAL frame references. It therefore cannot be
    // consumed by the first unrelated kind=5/kind=8 and recovery memory is O(unique pages *
    // FrameLocation), not O(unique pages * page_size). The double-false legacy path never touches
    // this state.
    EffectCommitUnit effect_unit;
    // wal-2c BLOCKER-3: stream the op-WAL one frame at a time (O(max frame), not O(WAL bytes)).
    alaya::wal::WalFile::
        visit_frames(op_wal_->path(), [&](const alaya::wal::ScannedFrame &frame) -> bool {
          if (frame.type != kSegmentOpRecordType) {
            replaying_ = false;
            poison("op-WAL contains a non-SEGMENT_OP record type");
          }
          SegmentOp op;
          try {
            op = decode_segment_op(frame.payload);
          } catch (const std::exception &error) {
            replaying_ = false;
            poison(std::string("op-WAL frame decode failed: ") + error.what());
          }
          if (op.segment_id != segment_uid_) {
            replaying_ = false;
            poison("op-WAL lineage mismatch (stale or foreign .opwal)");
          }
          // kind=1..6 frames always carry batch_id 0 (clause 11 / REPORT-wal-2a); only
          // the label ops (7/8) carry a tx_id. Reject a producer that violates this.
          if (op.kind != SegmentOpKind::label_bind && op.kind != SegmentOpKind::tx_publish &&
              frame.batch_id != 0) {
            replaying_ = false;
            poison("op-WAL kind=1..6 frame has a non-zero batch_id");
          }
          if (in_epoch) {
            switch (op.kind) {
              case SegmentOpKind::row_patch: {
                if (op.segment_generation != epoch_generation) {  // wal-2c BLOCKER-5
                  replaying_ = false;
                  poison("op-WAL maintenance row_patch generation != the epoch BEGIN generation");
                }
                const size_t page = replay_validate_row_patch_geometry(op);
                epoch_pages[page].assign(op.bytes.begin(), op.bytes.end());  // latest wins
                break;
              }
              case SegmentOpKind::consolidate_end: {
                if (op.epoch != epoch_id) {
                  replaying_ = false;
                  poison("op-WAL consolidate_end epoch does not match its begin");
                }
                if (op.segment_generation != epoch_generation) {  // wal-2c BLOCKER-5
                  replaying_ = false;
                  poison("op-WAL consolidate_end generation != the epoch BEGIN generation");
                }
                // Only a NEW epoch (> the base's last_completed) is applied; an
                // absorbed-by-base prefix (E <= base epoch) is validated for order but
                // NOT re-applied, so a stale image can never clobber a higher base
                // generation (codex B-2C-05 pitfall 5).
                if (epoch_apply) {
                  for (const auto &[page, bytes] : epoch_pages) {
                    write_at(kSectorLen + page * page_size_,
                             reinterpret_cast<const char *>(bytes.data()),
                             page_size_);
                  }
                  last_completed_consolidate_epoch_ = epoch_id;
                }
                in_epoch = false;
                epoch_pages.clear();
                break;
              }
              default:
                replaying_ = false;
                poison("op-WAL maintenance epoch contains an unexpected op kind");
            }
            return true;
          }
          if (in_canonical_bundle) {
            canonical_lane_step(cstage, op, frame, in_canonical_bundle, committed_watermark);
            return true;
          }
          switch (op.kind) {
            case SegmentOpKind::row_patch:
              // NEW-BLOCKER-1 (leg-8): classify by generation against the replay cursor. A
              // pre-leg-8 build applied EVERY standalone row_patch by absolute offset with no
              // generation check, so a CRC-legal old-generation whole-page after-image spliced onto
              // a pid-active base clobbered a committed page permanently (the final ftruncate only
              // trims beyond the HWM).
              switch (classify_standalone_effect(op.segment_generation)) {
                case StandaloneEffect::kApply:
                  if (pid_generation_activated_ || enable_pid_reuse_) {
                    stage_effect_page(effect_unit, op, frame, committed_watermark);
                  } else {
                    replay_row_patch(op);  // legacy 2A/2B: immediate apply, byte-for-byte unchanged
                  }
                  break;
                case StandaloneEffect::kValidateOnly:
                  // Absorbed/old-generation after-image: validate geometry (structural legality)
                  // but do NOT write -- the base at the cursor generation already holds the
                  // authoritative page.
                  (void)replay_validate_row_patch_geometry(op, /*touched_pid_evidence=*/true);
                  break;
                case StandaloneEffect::kPoison:
                  replaying_ = false;
                  poison("op-WAL standalone row_patch generation is newer than the replay cursor");
              }
              break;
            case SegmentOpKind::tombstone:
              switch (classify_standalone_effect(op.segment_generation)) {
                case StandaloneEffect::kApply:
                  replay_apply_or_stage_tombstone(op, effect_unit);
                  break;
                case StandaloneEffect::kValidateOnly:
                  // NEW-BLOCKER-1 (leg-8): an old-generation tombstone is already reflected in the
                  // base (or is a generation-unaware ABA forge against a reused incarnation) -- do
                  // NOT apply it, so a stale kind=2 can never re-hide a live reused PID.
                  break;
                case StandaloneEffect::kPoison:
                  replaying_ = false;
                  poison("op-WAL standalone tombstone generation is newer than the replay cursor");
              }
              break;
            case SegmentOpKind::publish:
              switch (classify_standalone_effect(op.segment_generation)) {
                case StandaloneEffect::kApply:
                  if (pid_generation_activated_ || enable_pid_reuse_) {
                    commit_effect_unit_at_publish(op.watermark, committed_watermark, effect_unit);
                  } else {
                    committed_watermark = std::max(committed_watermark, op.watermark);  // legacy
                  }
                  break;
                case StandaloneEffect::kValidateOnly:
                  // Do not advance the watermark from an absorbed/forged old-generation publish.
                  break;
                case StandaloneEffect::kPoison:
                  replaying_ = false;
                  poison("op-WAL standalone publish generation is newer than the replay cursor");
              }
              break;
            case SegmentOpKind::superblock_flip:
              replay_flip_with_effect_unit(op, committed_watermark, effect_unit);
              break;
            case SegmentOpKind::label_bind:
              if (is_canonical_generation(op.segment_generation)) {
                resolve_effect_unit_before_lane(effect_unit, "canonical");
                canonical_open_lane(cstage, op, frame, in_canonical_bundle);
              } else {
                replay_label_bind(op, frame.batch_id);
                attach_legacy_owner(effect_unit, op.tx_id);
              }
              break;
            case SegmentOpKind::tx_publish:
              if (is_canonical_generation(op.segment_generation)) {
                replaying_ = false;
                poison("op-WAL canonical tx_publish with no open bundle (missing kind=7 prefix)");
              }
              replay_tx_publish(op, frame.batch_id, committed_watermark, effect_unit);
              break;
            case SegmentOpKind::consolidate_begin: {
              if (op.epoch == 0 || op.epoch > last_completed_consolidate_epoch_ + 1) {
                replaying_ = false;
                poison("op-WAL consolidate epoch is not the next legal step");
              }
              // wal-2c BLOCKER-5: every consolidate frame is written at the then-current base
              // generation (encode_consolidate_marker / encode_row_patch use
              // superblock_.generation), which cannot change inside an epoch -- a mid-epoch
              // flip/other op poisons below. Bind the epoch to the BEGIN generation and require
              // kind=1 / END to all match it, so a spliced cross-generation frame (BEGIN(E,G)
              // row_patch(G+9) END(E,G+4)) is rejected.
              //
              // I-1 (leg-9, r3 independent audit): classify the BEGIN generation against the replay
              // cursor (superblock_.generation, advanced by replay_flip) with the SAME three-way
              // split classify_standalone_effect uses, instead of demanding an exact cursor match:
              //   gen  > cursor: impossible in a valid log (a frame is written at the then-current
              //   base,
              //                  which only reaches G+1 after its flip is replayed) -> poison.
              //   gen == cursor: a new epoch (op.epoch == last_completed+1) or a same-generation
              //                  absorbed prefix -- epoch_apply below picks new vs absorbed.
              //   gen  < cursor: an epoch written at an OLDER generation whose result a checkpoint
              //   flip
              //                  has since carried into the higher-generation base (crash AFTER the
              //                  checkpoint's superblock write, BEFORE the WAL reset). Legal ONLY
              //                  as an absorbed prefix (op.epoch <= last_completed, so the base
              //                  already reflects it) -> stage + validate-only. A new epoch
              //                  (op.epoch > last_completed) at an old generation is impossible /
              //                  forged -> poison.
              // Without this split a legal consolidate->checkpoint->SB-fsync->crash-before-reset
              // state (BEGIN(E,G) still in the WAL under a selected G+1 base) fail-closed forever.
              const uint64_t cursor = superblock_.generation;
              if (op.segment_generation > cursor) {
                replaying_ = false;
                poison("op-WAL consolidate_begin generation is newer than the replay cursor");
              }
              if (op.segment_generation < cursor && op.epoch > last_completed_consolidate_epoch_) {
                replaying_ = false;
                poison("op-WAL consolidate_begin at an older generation is not an absorbed prefix");
              }
              resolve_effect_unit_before_lane(effect_unit, "maintenance");
              // op.epoch <= last_completed => absorbed-by-base prefix (a checkpoint flip
              // already carried it); the redo below is byte-identical and idempotent.
              in_epoch = true;
              epoch_apply = op.epoch > last_completed_consolidate_epoch_;
              epoch_id = op.epoch;
              epoch_generation = op.segment_generation;
              epoch_begin_offset = frame.offset;
              epoch_pages.clear();
              break;
            }
            case SegmentOpKind::consolidate_end:
              replaying_ = false;
              poison("op-WAL consolidate_end without a matching begin");
          }
          return true;
        });
    // Unmatched BEGIN at EOF: discard the incomplete epoch (nothing was installed --
    // END was never durable, so the index is still S_old) and truncate the WAL back
    // to the BEGIN boundary so the next append starts clean (design section 1.3).
    if (in_epoch) {
      epoch_pages.clear();
      try {
        op_wal_->truncate_to(epoch_begin_offset);
      } catch (const std::exception &error) {
        replaying_ = false;
        poison(std::string("op-WAL semantic truncation of an unmatched BEGIN failed: ") +
               error.what());
      }
    }
    if (in_canonical_bundle) {
      // Torn canonical bundle: no durable kind=8, and kind=1 was only staged (never
      // applied), so the index is still S_old. Discard the staged refs/binds and truncate
      // the WAL back to the first kind=7 so the next append starts clean (design 3.5 / R1).
      try {
        op_wal_->truncate_to(cstage.begin_offset);
      } catch (const std::exception &error) {
        replaying_ = false;
        poison(std::string("op-WAL semantic truncation of a torn canonical bundle failed: ") +
               error.what());
      }
    }
    // EOF explicitly resolves a torn standalone/legacy unit. Frame references and staged legacy
    // tombstones are recovery-local and have not been applied, so discarding them leaves S_old; a
    // standalone kind=2 was already applied idempotently and remains authoritative.
    effect_unit.clear();
    // I-4 (leg-9, r3): staged_binds_ is a REPLAY-LOCAL scratch (recovery-only, never read on the
    // writer path). Any orphan kind=7 binds that never matched a kind=8 (a torn bundle whose commit
    // was lost) are dropped here, explicitly and unconditionally -- adopt_label_state() only clears
    // them on a G+1 install flip, so a SAME-generation activation flip (already the selected base,
    // early-returns in replay_flip without re-adopting) previously left them dangling until the
    // next recovery. Clearing at the single convergence point makes the "orphan dropped at EOF"
    // comment in replay_label_bind literally true for every flip disposition.
    staged_binds_.clear();
    committed_.store(committed_watermark, std::memory_order_release);
    // replaying_ stays true through the entire rebuild so no derived-state repair
    // (routing/free/indegree) re-enters the WAL write path (design section 2.1 /
    // execution note 3). Cleared only after the single convergence point returns.
    rebuild_state_after_replay(committed_watermark);
    replaying_ = false;
  }

  // Validate a row_patch's whole-page geometry (clause F) and return its page
  // index. Shared by the immediate-apply path and the maintenance-epoch staging.
  size_t replay_validate_row_patch_geometry(const SegmentOp &op,
                                            bool touched_pid_evidence = false) {
    if (op.offset < kSectorLen) {
      poison("row_patch offset overlaps the A/B metadata sector");
    }
    if (op.bytes.size() != page_size_) {
      poison("row_patch is not a whole-page after-image");
    }
    const uint64_t relative = op.offset - kSectorLen;
    if (relative % page_size_ != 0) {
      poison("row_patch offset is not page-aligned");
    }
    const size_t page = static_cast<size_t>(relative / page_size_);
    if (page >= page_versions_.size()) {
      poison("row_patch page exceeds the configured capacity");
    }
    if (touched_pid_evidence) {
      if (op.pid >= static_cast<uint64_t>(kPidMax) || op.pid >= row_generations_.size() ||
          op.pid / npp_ != page) {
        poison("row_patch touched-pid evidence does not belong to the patched page");
      }
    } else if (op.pid != static_cast<uint64_t>(page) * npp_) {
      poison("row_patch pid/offset geometry mismatch");
    }
    return page;
  }

  void replay_row_patch(const SegmentOp &op) {
    (void)replay_validate_row_patch_geometry(op);
    write_at(op.offset, reinterpret_cast<const char *>(op.bytes.data()), op.bytes.size());
  }

  [[nodiscard]] uint64_t single_staged_legacy_txid() {
    uint64_t candidate = 0;
    for (const auto &[txid, binds] : staged_binds_) {
      if (binds.empty()) {
        continue;
      }
      if (candidate != 0 && candidate != txid) {
        poison("op-WAL kind=1/kind=2 has ambiguous ownership across legacy transactions");
      }
      candidate = txid;
    }
    return candidate;
  }

  void begin_or_check_effect_unit(EffectCommitUnit &unit,
                                  uint64_t generation,
                                  uint64_t frame_offset) {
    if (!unit.active()) {
      unit.generation = generation;
      unit.begin_offset = frame_offset;
      return;
    }
    if (unit.generation != generation) {
      poison("op-WAL effect commit unit mixes segment generations");
    }
  }

  void attach_legacy_owner(EffectCommitUnit &unit, uint64_t txid) {
    if (!unit.active()) {
      return;
    }
    // A preceding standalone tombstone is a distinct idempotent unit. Its kind=1 pages are
    // redundant with the immediately persisted trailer/rebuild state and must not become owned by a
    // later legacy bind merely because kind=1 has no txid.
    if (unit.saw_standalone_tombstone) {
      if (!unit.allocation_evidence.empty()) {
        poison("op-WAL unpublished allocation evidence crosses into a legacy label unit");
      }
      validate_effect_unit_pages(unit);
      unit.clear();
      return;
    }
    if (!unit.has_staged_effects()) {
      return;
    }
    if (unit.legacy_txid == 0) {
      unit.legacy_txid = txid;
      return;
    }
    if (unit.legacy_txid != txid) {
      poison("op-WAL effect commit unit is interleaved across two legacy txids");
    }
  }

  void stage_effect_page(EffectCommitUnit &unit,
                         const SegmentOp &op,
                         const alaya::wal::ScannedFrame &frame,
                         uint64_t committed_watermark) {
    const size_t page = replay_validate_row_patch_geometry(op, /*touched_pid_evidence=*/true);
    begin_or_check_effect_unit(unit, op.segment_generation, frame.offset);
    if (unit.legacy_txid == 0 && !unit.saw_standalone_tombstone) {
      unit.legacy_txid = single_staged_legacy_txid();
    }
    // Latest-wins is safe only if it does not silently roll back a row that supplied allocation
    // evidence. Re-read the prior final image and retain each PID's evidence only when that PID's
    // row+trailer is byte-identical in the replacement image, or when this frame explicitly says it
    // touched that same PID (in which case the evidence is refreshed below).
    const auto previous = unit.latest_refs.find(page);
    auto evidence = unit.allocation_evidence.find(page);
    if (previous != unit.latest_refs.end() && evidence != unit.allocation_evidence.end()) {
      const SegmentOp old = reread_effect_page(previous->second, unit.generation, page);
      for (auto it = evidence->second.begin(); it != evidence->second.end();) {
        if (static_cast<uint64_t>(*it) != op.pid && !effect_row_image_equal(old, op, *it)) {
          it = evidence->second.erase(it);
        } else {
          ++it;
        }
      }
      if (evidence->second.empty()) {
        unit.allocation_evidence.erase(evidence);
      }
    }
    unit.latest_refs[page] = alaya::wal::FrameLocation{frame.offset, frame.size};
    if (op.pid >= committed_watermark) {
      unit.allocation_evidence[page].insert(static_cast<PID>(op.pid));
    }
  }

  [[nodiscard]] SegmentOp reread_effect_page(const alaya::wal::FrameLocation &ref,
                                             uint64_t expected_generation,
                                             size_t expected_page) {
    alaya::wal::ScannedFrame frame;
    try {
      frame = alaya::wal::WalFile::read_frame(op_wal_->path(), ref);
    } catch (const std::exception &error) {
      poison(std::string("op-WAL effect page re-read failed: ") + error.what());
    }
    if (frame.type != kSegmentOpRecordType || frame.batch_id != 0) {
      poison("op-WAL effect page reference does not name a standalone kind=1 frame");
    }
    SegmentOp op;
    try {
      op = decode_segment_op(frame.payload);
    } catch (const std::exception &error) {
      poison(std::string("op-WAL effect page re-decode failed: ") + error.what());
    }
    if (op.kind != SegmentOpKind::row_patch || op.segment_id != segment_uid_ ||
        op.segment_generation != expected_generation) {
      poison("op-WAL effect page reference changed identity or generation");
    }
    if (replay_validate_row_patch_geometry(op, /*touched_pid_evidence=*/true) != expected_page) {
      poison("op-WAL effect page reference changed page identity");
    }
    return op;
  }

  [[nodiscard]] bool effect_row_image_equal(const SegmentOp &lhs,
                                            const SegmentOp &rhs,
                                            PID pid) const {
    const size_t slot = static_cast<size_t>(pid) % npp_;
    const size_t node_offset = slot * node_len_;
    if (std::memcmp(lhs.bytes.data() + node_offset, rhs.bytes.data() + node_offset, node_len_) !=
        0) {
      return false;
    }
    const size_t trailer_offset = qg_page_trailer_offset(page_size_, npp_, slot);
    return std::memcmp(lhs.bytes.data() + trailer_offset,
                       rhs.bytes.data() + trailer_offset,
                       sizeof(QGRowTrailer)) == 0;
  }

  [[nodiscard]] bool has_allocation_evidence(const EffectCommitUnit &unit, PID pid) const {
    const auto page_it = unit.allocation_evidence.find(page_index(pid));
    return page_it != unit.allocation_evidence.end() && page_it->second.count(pid) != 0;
  }

  void validate_allocation_evidence(const EffectCommitUnit &unit,
                                    uint64_t old_hwm,
                                    uint64_t new_hwm) {
    for (uint64_t raw = old_hwm; raw < new_hwm; ++raw) {
      const PID pid = static_cast<PID>(raw);
      if (!has_allocation_evidence(unit, pid)) {
        poison("op-WAL publish advances the HWM over a PID with no row allocation evidence");
      }
    }
    // Validate each final page once and require every consumed evidence row to be structurally
    // live.
    for (const auto &[page, pids] : unit.allocation_evidence) {
      const auto ref = unit.latest_refs.find(page);
      if (ref == unit.latest_refs.end()) {
        poison("op-WAL row allocation evidence has no final page image");
      }
      const SegmentOp final = reread_effect_page(ref->second, unit.generation, page);
      for (PID pid : pids) {
        const uint64_t raw = static_cast<uint64_t>(pid);
        if (raw < old_hwm || raw >= new_hwm) {
          continue;
        }
        const QGRowTrailer trailer =
            qg_read_page_trailer(reinterpret_cast<const char *>(final.bytes.data()),
                                 page_size_,
                                 npp_,
                                 pid % npp_);
        if (trailer.valid_degree > deg_ || (trailer.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
          poison("op-WAL allocation evidence final row is not structurally live");
        }
      }
    }
  }

  void consume_published_allocation_evidence(EffectCommitUnit &unit, uint64_t new_hwm) {
    for (auto page_it = unit.allocation_evidence.begin();
         page_it != unit.allocation_evidence.end();) {
      auto &pids = page_it->second;
      for (auto pid_it = pids.begin(); pid_it != pids.end();) {
        if (static_cast<uint64_t>(*pid_it) < new_hwm) {
          pid_it = pids.erase(pid_it);
        } else {
          ++pid_it;
        }
      }
      if (pids.empty()) {
        unit.latest_refs.erase(page_it->first);
        page_it = unit.allocation_evidence.erase(page_it);
      } else {
        ++page_it;
      }
    }
    unit.saw_standalone_tombstone = false;  // its same-HWM page effects were just installed
    if (unit.allocation_evidence.empty()) {
      unit.clear();
    }
  }

  void validate_effect_unit_pages(const EffectCommitUnit &unit) {
    for (const auto &[page, ref] : unit.latest_refs) {
      (void)reread_effect_page(ref, unit.generation, page);
    }
  }

  void apply_effect_unit_pages(const EffectCommitUnit &unit) {
    for (const auto &[page, ref] : unit.latest_refs) {
      const SegmentOp op = reread_effect_page(ref, unit.generation, page);
      write_at(op.offset, reinterpret_cast<const char *>(op.bytes.data()), op.bytes.size());
    }
  }

  void apply_one_replay_tombstone(PID pid) {
    mark_hidden(pid);
    mirror_deleted_insert(pid);
    replay_persist_tombstone_trailer(pid);
  }

  void apply_staged_legacy_tombstones(const EffectCommitUnit &unit) {
    for (PID pid : unit.staged_legacy_tombstones) {
      apply_one_replay_tombstone(pid);
    }
  }

  void replay_apply_or_stage_tombstone(const SegmentOp &op, EffectCommitUnit &unit) {
    if (op.pid >= row_generations_.size()) {
      return;  // preserves the existing idempotent out-of-capacity no-op
    }
    const PID pid = static_cast<PID>(op.pid);
    if (pid_generation_activated_ || enable_pid_reuse_) {
      uint64_t owner = unit.legacy_txid;
      if (owner == 0 && !unit.saw_standalone_tombstone) {
        owner = single_staged_legacy_txid();
      }
      if (owner != 0) {
        begin_or_check_effect_unit(unit, op.segment_generation, /*frame_offset=*/0);
        unit.legacy_txid = owner;
        unit.staged_legacy_tombstones.insert(pid);
        return;
      }
      begin_or_check_effect_unit(unit, op.segment_generation, /*frame_offset=*/0);
      unit.saw_standalone_tombstone = true;
    }
    // Unit-external current-generation tombstones retain the historical immediate, idempotent
    // apply.
    apply_one_replay_tombstone(pid);
  }

  void resolve_effect_unit_before_lane(EffectCommitUnit &unit, const char *lane) {
    if (!unit.active()) {
      return;
    }
    if (unit.legacy_txid != 0 || !unit.allocation_evidence.empty() ||
        (!unit.latest_refs.empty() && !unit.saw_standalone_tombstone)) {
      poison(std::string("op-WAL unresolved effect unit crosses into the ") + lane + " lane");
    }
    // A standalone tombstone was already installed idempotently. Validate its referenced final
    // pages, then discard the recovery-only refs so the lane cannot consume them accidentally.
    validate_effect_unit_pages(unit);
    unit.clear();
  }

  void commit_effect_unit_at_publish(uint64_t publish_watermark,
                                     uint64_t &committed_watermark,
                                     EffectCommitUnit &unit) {
    if (unit.legacy_txid != 0) {
      poison("op-WAL kind=5 cannot consume a legacy-owned effect unit");
    }
    if (publish_watermark > static_cast<uint64_t>(kPidMax) ||
        publish_watermark > row_generations_.size()) {
      poison("op-WAL standalone publish watermark exceeds the PID capacity");
    }
    const uint64_t old_hwm = committed_watermark;
    if (publish_watermark < old_hwm) {
      poison("op-WAL standalone publish watermark regressed below the committed HWM");
    }
    validate_allocation_evidence(unit, old_hwm, publish_watermark);
    validate_effect_unit_pages(unit);  // complete validation before the first page write
    apply_effect_unit_pages(unit);
    committed_watermark = publish_watermark;
    consume_published_allocation_evidence(unit, publish_watermark);
  }

  // Idempotently set the on-disk trailer tombstone bit for one row during replay
  // (design section 3.7). A direct disk RMW: the authoritative trailer scan in
  // rebuild_state_after_replay reads from disk, and replay_row_patch also writes
  // straight through, so this stays consistent with any later whole-page after-image.
  void replay_persist_tombstone_trailer(PID id) {
    const size_t pi = page_index(id);
    if (pi >= page_versions_.size()) {
      return;
    }
    AlignedBuf page(page_size_);
    read_at(page_offset(id), page.data(), page_size_);
    QGRowTrailer trailer = qg_read_page_trailer(page.data(), page_size_, npp_, id % npp_);
    if ((trailer.flags & kQGRowTombstone) != 0) {
      return;  // already tombstoned on disk: idempotent
    }
    trailer.flags |= kQGRowTombstone;
    qg_write_page_trailer(page.data(), page_size_, npp_, id % npp_, trailer);
    write_at(page_offset(id), page.data(), page_size_);
  }

  // Stage a label_bind by tx_id (B-04): pid_generation must be 0 and the frame
  // batch_id must equal the payload tx_id. Set-level validation happens at
  // tx_publish; a tx that never reaches tx_publish is silently dropped.
  void replay_label_bind(const SegmentOp &op, uint64_t frame_batch_id) {
    if (op.pid_generation != 0) {
      poison("op-WAL label_bind has a non-zero pid_generation");
    }
    if (frame_batch_id != op.tx_id) {
      poison("op-WAL label_bind frame batch_id != payload tx_id");
    }
    // NEW-BLOCKER-2 (leg-8, r2 blind-audit): a kind=7 bind only STAGES a binding -- it writes
    // nothing to the index -- so it is NEVER convicted on its own, even on a pid-activated base.
    // leg-7 poisoned here whenever pid_generation_activated_ && tx_id > base, but that falsely
    // killed a LEGAL crash state: a legacy bundle whose orphan kind=7 bind survived the kind=8
    // (SIGKILL before commit -- explicitly allowed by the (tx_id,row_op_id) dedup contract)
    // followed by a pid-activation checkpoint that fsynced its G+1 superblock but crashed before
    // the WAL reset. On reopen the pid-active G+1 base is selected while the orphan (tx_id > base)
    // still precedes the activation flip in the WAL; poisoning here fail-closed a recoverable
    // segment forever. The orphan is now staged and then dropped at the single end-of-replay clear
    // (I-4): a G+1 install flip re-adopts the base and clears staged_binds_ in adopt_label_state,
    // while a SAME-generation activation flip (already the selected base) early-returns without
    // re-adopting, so the explicit staged_binds_.clear() at the end of replay_and_rebuild is what
    // drops it in that case. Conviction of a forged NON-absorbed legacy transaction moves to its
    // effectful commit point (replay_tx_publish, tx_id > base), and its interleaved row_patch
    // after-images are validate-only by classify_standalone_effect -- so nothing the forged bundle
    // carries is ever applied before that commit-time poison (index stays unchanged).
    //
    // Idempotent de-dup by (tx_id, row_op_id): a same-txid retry after a torn bundle
    // whose earlier binds survived in the OS page cache (process crash on a bundle
    // larger than the WAL userspace buffer) re-appends identical binds. Collapse
    // them (same pid+label == idempotent; differing == poison) so a legal retry is
    // not falsely rejected by the count check. Torn power-loss states never expose
    // this (the forced WAL ends at the last fsync, before any bind).
    auto &staged = staged_binds_[op.tx_id];
    for (const auto &existing : staged) {
      if (existing.row_op_id == op.row_op_id) {
        if (existing.pid != static_cast<PID>(op.pid) || existing.label != op.label) {
          poison("op-WAL label_bind conflicting re-bind of the same (tx_id, row_op_id)");
        }
        return;  // idempotent duplicate
      }
    }
    staged.push_back(LabelBindStage{op.row_op_id, static_cast<PID>(op.pid), op.label});
  }

  // Promote or idempotently verify a tx_publish (B-04, semantics 4). Splits on the
  // adopted base's persisted txid: (a) a new tx to promote, (b) a tx already
  // absorbed by the base (new superblock flipped, old WAL not yet reset).
  void replay_tx_publish(const SegmentOp &op,
                         uint64_t frame_batch_id,
                         uint64_t &committed_watermark,
                         EffectCommitUnit &unit) {
    if (frame_batch_id != op.tx_id) {
      poison("op-WAL tx_publish frame batch_id != payload tx_id");
    }
    if (op.binding_count == 0) {
      poison("op-WAL tx_publish binding_count must be >= 1");
    }
    if (unit.legacy_txid != 0 && unit.legacy_txid != op.tx_id) {
      poison("op-WAL legacy tx_publish cannot consume another txid's effect unit");
    }
    const bool owns_effect_unit = unit.legacy_txid == op.tx_id;
    // NEW-BLOCKER-1/2 (leg-8): this is the COMMIT-TIME conviction point for the legacy lane. A
    // kind=8 is the effectful commit (it promotes staged binds + advances the watermark), so once
    // pid reuse is activated a legacy tx_publish (segment_generation < activation) that is NOT an
    // absorbed prefix (tx_id > base) is a forged cross-generation classifier downgrade -- poison
    // BEFORE any promotion. Together with kind=7 staging (never convicted alone) and old-generation
    // standalone row_patch being validate-only, a forged legacy bundle [kind=7,kind=1,kind=8] never
    // applies anything before this poison, while a legal torn orphan (no kind=8) is dropped -- so
    // NEW-BLOCKER-2's recoverable crash state is preserved and NEW-BLOCKER-1's forge stays closed.
    if (pid_generation_activated_ && op.tx_id > base_committed_txid_) {
      poison(
          "op-WAL post-activation legacy tx_publish is not an absorbed prefix "
          "(cross-generation classifier downgrade)");
    }
    const auto it = staged_binds_.find(op.tx_id);
    static const std::vector<LabelBindStage> kNoStaged;
    const std::vector<LabelBindStage> &staged = it != staged_binds_.end() ? it->second : kNoStaged;

    if (op.tx_id <= base_committed_txid_) {
      // (b) already absorbed by the selected base: verify idempotently and skip.
      // Do NOT apply new-range checks or re-promote.
      if (op.new_pid_watermark > base_num_points_) {
        poison("op-WAL absorbed tx_publish watermark exceeds the base num_points");
      }
      if (op.applied_collection_op_id > base_applied_op_id_) {
        poison("op-WAL absorbed tx_publish applied_op_id exceeds the base");
      }
      if (staged.size() != op.binding_count) {
        poison("op-WAL absorbed tx_publish staged bind count mismatch");
      }
      for (const auto &bind : staged) {
        if (bind.row_op_id >= op.binding_count) {
          poison("op-WAL absorbed tx_publish row_op_id out of range");
        }
        const auto found = label_working_.find(bind.pid);
        if (found == label_working_.end() || found->second.label != bind.label) {
          poison("op-WAL absorbed tx_publish binding disagrees with the persisted slot");
        }
      }
      staged_binds_.erase(op.tx_id);
      // The selected base already holds an absorbed bundle. Resolve only the effect unit explicitly
      // owned by this tx; unrelated standalone evidence remains available to its later boundary.
      if (owns_effect_unit) {
        validate_effect_unit_pages(unit);
        unit.clear();
      }
      return;
    }

    // (a) a new transaction to promote. Full B-04 validation set.
    // NEW-BLOCKER-1 (leg-7): the classifier downgrade is closed by the pid-activated absorbed-gate
    // above (a post-activation legacy tx_publish with tx_id > base is poisoned before reaching
    // here). A cursor-generation bind is deliberately NOT imposed on the legacy lane: for a
    // pid-activated base every non-absorbed legacy tx is already rejected (stronger than a cursor
    // bind), and for a v2/2A base the legacy generation field was never cursor-bound (2A frames
    // legitimately carry an older/arbitrary generation), so binding it would fail-closed on legal
    // 2A traffic and mask the malformed-bundle divergence tests.
    if (op.tx_id <= last_committed_txid_) {
      poison("op-WAL tx_publish tx_id is not strictly increasing");
    }
    if (op.applied_collection_op_id < applied_collection_op_id_) {
      poison("op-WAL tx_publish applied_collection_op_id regressed");
    }
    const uint64_t old_hwm = committed_watermark;
    const uint64_t new_hwm = op.new_pid_watermark;
    if (new_hwm != old_hwm + op.binding_count) {
      poison("op-WAL tx_publish new_pid_watermark != old_hwm + binding_count");
    }
    if (staged.size() != op.binding_count) {
      poison("op-WAL tx_publish staged bind count != binding_count");
    }
    if (new_hwm > static_cast<uint64_t>(kPidMax) || new_hwm > row_generations_.size()) {
      poison("op-WAL tx_publish new_pid_watermark exceeds the PID capacity");
    }
    // Staged set must be exactly row_op_id == {0..count-1} and pid == [old,new),
    // each unique (counts already match, so covering + unique == a bijection).
    std::vector<uint8_t> row_seen(static_cast<size_t>(op.binding_count), 0);
    std::vector<uint8_t> pid_seen(static_cast<size_t>(op.binding_count), 0);
    for (const auto &bind : staged) {
      if (bind.row_op_id >= op.binding_count ||
          row_seen[static_cast<size_t>(bind.row_op_id)] != 0) {
        poison("op-WAL tx_publish row_op_id set is not exactly {0..count-1}");
      }
      row_seen[static_cast<size_t>(bind.row_op_id)] = 1;
      const uint64_t pid_u64 = static_cast<uint64_t>(bind.pid);
      if (pid_u64 < old_hwm || pid_u64 >= new_hwm) {
        poison("op-WAL tx_publish pid outside [old_hwm, new_hwm)");
      }
      const size_t pidx = static_cast<size_t>(pid_u64 - old_hwm);
      if (pid_seen[pidx] != 0) {
        poison("op-WAL tx_publish has a duplicate pid");
      }
      pid_seen[pidx] = 1;
    }
    // I-2 / NEW-B1 (leg-9): when the reuse classifier is armed, this bundle's row after-images were
    // STAGED (not applied immediately), so apply them here -- AFTER the full B-04 validation set
    // and the NEW-BLOCKER-1 absorbed-gate above, so a forged legacy bundle poisons before anything
    // is written. Verify whole-page evidence for the new range (a phantom bound row with no
    // after-image is rejected), then apply the pages before promoting the bindings. On the legacy
    // path (classifier disarmed) kind=1 was applied immediately and `pending` is empty, so this is
    // skipped (unchanged).
    if (pid_generation_activated_ || enable_pid_reuse_) {
      if (!owns_effect_unit) {
        poison("op-WAL tx_publish has no owned effect unit for its appended PID range");
      }
      validate_allocation_evidence(unit, old_hwm, new_hwm);
      validate_effect_unit_pages(unit);  // full unit validation before the first page write
      apply_effect_unit_pages(unit);
      apply_staged_legacy_tombstones(unit);
      unit.clear();
    }
    // Promote: install the bindings, advance committed + both tx watermarks.
    for (const auto &bind : staged) {
      const auto inserted = label_working_.emplace(bind.pid, PidBinding{0, bind.label});
      if (!inserted.second && inserted.first->second.label != bind.label) {
        poison("op-WAL tx_publish rebinds an existing pid to a different label");
      }
    }
    ++label_content_revision_;  // replay promoted new bindings (design 3.1 slot-dirty tracking)
    committed_watermark = new_hwm;
    last_committed_txid_ = op.tx_id;
    applied_collection_op_id_ = op.applied_collection_op_id;
    staged_binds_.erase(op.tx_id);
  }

  // BLOCKER-3 (leg-7): validate a flip image's label tuple + slot file + pid-generation
  // summary into TEMP objects BEFORE any pwrite. adopt_label_state() runs these checks too,
  // but only AFTER write_superblock() has already persisted the (higher-generation) image;
  // a crafted flip whose only defect is in the label slot / summary would install and then
  // poison, and every later reopen would keep re-selecting the installed illegal base.
  // Rejecting here writes nothing (index bytes stay byte-for-byte unchanged). The activation
  // generation bounds/ordering are already covered by qg_superblock_supported() at the top of
  // replay_flip; this closes the remaining label-slot / summary window.
  void validate_flip_label_state(const QGSuperblockV2 &image) {
    const auto ls = read_superblock_label_state(image);
    const auto w2c = read_superblock_wal2c_state(image);
    const bool image_pid_active = image.format_version == kQGFormatVersionV3 &&
                                  (w2c.required_feature_flags & kQgFeatPidGenerationV1) != 0;
    if (ls.generation == 0 && ls.count == 0 && ls.checksum == 0) {
      if (w2c.max_pid_generation != 0 || w2c.nonzero_pid_generation_count != 0) {
        poison("flip image: empty label slot but a non-zero pid-generation summary");
      }
      if (ls.slot > 1) {
        poison("flip image: label slot index out of range (empty tuple)");
      }
      // B3 completion (leg-9, r3 section 1 point 4): the image's label bindings must equal the
      // bindings this replay reconstructed from the WAL prefix (label_working_). An install flip is
      // reached only after every preceding bundle was promoted, so an EMPTY image slot over a
      // non-empty recovered set means the flip silently drops committed labels -- reject before
      // write.
      if (!label_working_.empty()) {
        poison("flip image: empty label slot but the recovered bindings are non-empty");
      }
      return;  // canonical legacy empty: the slot file may be absent
    }
    if (ls.slot > 1) {
      poison("flip image: label slot index out of range");
    }
    // Preload + fully validate the referenced slot file into a temp binding set (checksum,
    // count*16, ascending, pid < num_points, generation-vs-pid-active) -- pure read + poison.
    auto lb = load_label_slot_bindings(label_slot_path_[ls.slot],
                                       ls.count,
                                       ls.checksum,
                                       image.num_points,
                                       image_pid_active);
    if (image_pid_active) {
      uint32_t max_gen = 0;
      uint32_t nz_count = 0;
      for (const auto &[pid, binding] : lb.bindings) {
        (void)pid;
        if (binding.pid_generation != 0) {
          ++nz_count;
          max_gen = (std::max)(max_gen, binding.pid_generation);
        }
      }
      if (max_gen != w2c.max_pid_generation || nz_count != w2c.nonzero_pid_generation_count) {
        poison("flip image: label slot pid-generation summary does not match the image");
      }
    }
    // B3 completion (leg-9, r3 section 1 point 4): the loaded slot must equal the bindings this
    // replay reconstructed from the WAL prefix (label_working_). A G+1 install flip is reached only
    // after every preceding bundle promoted into label_working_, and a legal checkpoint captured
    // exactly that map into the slot; so a self-consistent-but-STALE slot (e.g. an older,
    // still-valid label tuple with a rewound generation, or any slot that disagrees item-by-item
    // with the recovered set) is forged. This is the item-by-item equivalence closure
    // -- stricter than the generation/count monotonicity in validate_flip_transition, and
    // orphan-robust (a torn bundle's staged binds are in neither set, so both exclude them).
    if (lb.bindings != label_working_) {
      poison("flip image: label slot disagrees with the recovered bindings");
    }
  }

  // BLOCKER-3 (leg-8, r2 section 1): PURE pre-write validation of a flip TRANSITION. Poisons on any
  // violation and writes NOTHING (no pwrite/ftruncate) -- a crafted flip is rejected with the index
  // byte-for-byte unchanged. Runs before EVERY durable flip: replay_flip (before its
  // write_superblock/ftruncate) AND checkpoint_locked (before wal_append(flip,fsync)). `image` is
  // the NEXT base; `superblock_` is the current/selected base (the replay cursor); op_generation +
  // target_slot come from the flip op. qg_superblock_supported() (self-consistency, activation
  // bounds) and validate_flip_label_state() (label slot) already ran on the replay path; this
  // closes the geometry / file-length / target-slot / generation-transition / state-monotonicity
  // window they do not cover -- exactly the dimension+1 clone, the file_size=kSectorLen truncation
  // bomb, and the target_slot==active tear the r2 review reached.
  void validate_flip_transition(const QGSuperblockV2 &image,
                                uint64_t op_generation,
                                uint8_t target_slot,
                                uint64_t expected_num_points) {
    // (1) The op's declared generation must match the image it carries: a CRC-legal flip whose
    // frame generation disagrees with its payload generation is malformed.
    if (op_generation != image.generation) {
      poison("flip op generation != the image generation");
    }
    // (2) A flip MUST target the INACTIVE A/B slot. Overwriting the active slot would, on a torn
    // 512-byte pwrite, leave only the G-1 copy while the durable WAL still demands G+1 -- the next
    // replay would face a two-generation jump and poison. (active_superblock_slot_ is -1 only
    // before any base is loaded, which never reaches a flip.)
    if (active_superblock_slot_ < 0 ||
        static_cast<int>(target_slot) != 1 - active_superblock_slot_) {
      poison("flip target slot is not the inactive A/B slot");
    }
    // (3) Generation is the next legal step with NO overflow. (replay_flip already gated == +1 on
    // its path; the checkpoint's next.generation == superblock_.generation+1 -- assert both here so
    // a wraparound at UINT64_MAX can never mint generation 0.)
    if (superblock_.generation == (std::numeric_limits<uint64_t>::max)()) {
      poison("flip generation overflow");
    }
    if (image.generation != superblock_.generation + 1) {
      poison("flip generation is not the next legal step (G+1)");
    }
    // (4) Immutable geometry: dimension / node_len / node_per_page / page_size are fixed at build
    // time (load_v2_state enforces the same identity on open). A clone that bumps the dimension by
    // one would install and only fail closed on the NEXT open -- reject it before any write.
    if (image.dimension != dim_ || image.node_len != node_len_ || image.node_per_page != npp_ ||
        image.page_size != page_size_) {
      poison("flip image geometry differs from the segment (immutable geometry)");
    }
    // (5) Capacity + count bounds (kPidMax is the free/entry sentinel).
    if (image.num_points > static_cast<uint64_t>(kPidMax)) {
      poison("flip image num_points exceeds the PID capacity");
    }
    // (5a) B3 completion (leg-9, r3 section 1 point 1): the REAL handle capacity. row_generations_
    // is sized to UpdateParams::max_points, and every commit path bounds new_hwm by it (see
    // replay_tx_publish / canonical_finalize_bundle). A flip whose num_points is within kPidMax but
    // beyond this handle's configured capacity has no backing PID/page state -- reject before the
    // write_superblock/ftruncate would install a base this handle cannot address.
    if (image.num_points > row_generations_.size()) {
      poison("flip image num_points exceeds the configured handle capacity (max_points)");
    }
    // (5b) B3 completion (leg-9, r3 section 1 point 2): the expected high-water mark. A legal
    // checkpoint captures committed_ into image.num_points, and on replay the running committed
    // watermark just before the install flip equals that same value (all preceding publishes/
    // bundles/epochs were replayed). A G+1 install image whose num_points disagrees with the
    // recovered prefix (e.g. an HWM rewind with an otherwise-exact file_size + in-range label
    // tuple) is forged -- it would install a lower base and ftruncate live data pages away.
    if (image.num_points != expected_num_points) {
      poison("flip image num_points != the expected recovered watermark");
    }
    if (image.live_count > image.num_points || image.free_count > image.num_points) {
      poison("flip image live/free count exceeds num_points");
    }
    // (5c) B3 completion (leg-9, r3 section 1 point 3): count-tuple JOINT invariants (the per-field
    // ranges above are necessary but not sufficient). live + free can never exceed num_points (both
    // <= num_points <= kPidMax, so the sum cannot overflow u64), and the free-list head sentinel
    // must agree with free_count: an empty free list carries the kPidMax sentinel, a non-empty one
    // a real in-range head (already range-checked above).
    if (image.live_count + image.free_count > image.num_points) {
      poison("flip image live_count + free_count exceeds num_points");
    }
    if (image.free_list_head != kPidMax && image.free_list_head >= image.num_points) {
      poison("flip image free_list_head is out of range");
    }
    if ((image.free_count == 0) != (image.free_list_head == kPidMax)) {
      poison("flip image free_count / free_list_head sentinel disagree");
    }
    if (image.num_points != 0 && image.entry_point != static_cast<uint64_t>(kPidMax) &&
        image.entry_point >= image.num_points) {
      poison("flip image entry_point is out of range");
    }
    // (6) Exact file length: kSectorLen + ceil(num_points/npp)*page_size. A shrunk file_size (e.g.
    // kSectorLen) would ftruncate away every data page after the superblock install; a grown one
    // would leave the tail undefined. npp_ >= 1 by construction, so there is no divide-by-zero.
    const uint64_t page_num = image.num_points == 0 ? 0 : (image.num_points + npp_ - 1) / npp_;
    if (image.file_size != static_cast<uint64_t>(kSectorLen) + page_num * page_size_) {
      poison("flip image file_size != kSectorLen + ceil(num_points/npp)*page_size");
    }
    // (7) State monotonicity (no regression): format_version, the activated feature set, the
    // activation generations, the tx watermarks, and the consolidate epoch only ever move forward
    // across a durable flip. A flip that rewinds any of them is forged / an ABA rollback.
    if (image.format_version < superblock_.format_version) {
      poison("flip image format_version regressed");
    }
    const auto cur_tx = read_superblock_tx_state(superblock_);
    const auto img_tx = read_superblock_tx_state(image);
    if (img_tx.last_committed_txid < cur_tx.last_committed_txid ||
        img_tx.applied_collection_op_id < cur_tx.applied_collection_op_id) {
      poison("flip image tx watermark regressed");
    }
    const auto cur_w = read_superblock_wal2c_state(superblock_);
    const auto img_w = read_superblock_wal2c_state(image);
    if ((img_w.required_feature_flags & cur_w.required_feature_flags) !=
        cur_w.required_feature_flags) {
      poison("flip image drops a previously-activated feature bit");
    }
    if (img_w.last_completed_consolidate_epoch < cur_w.last_completed_consolidate_epoch ||
        img_w.maintenance_activation_sb_generation < cur_w.maintenance_activation_sb_generation ||
        img_w.pid_reuse_activation_sb_generation < cur_w.pid_reuse_activation_sb_generation) {
      poison("flip image maintenance/reuse activation state regressed");
    }
    // (8) B3 completion (leg-9, r3 section 1 point 4): the label-tuple TRANSITION. validate_flip_
    // label_state() proves the image's label slot is self-consistent (and, on the replay path,
    // equals the recovered bindings); this closes the monotonicity/overflow window between the
    // CURRENT base's tuple and the image's. A label content change bumps the generation by one and
    // rewrites the inactive slot, so across one flip the label generation only moves forward, the
    // binding count is append-only (never shrinks), and a same-generation tuple must be
    // byte-for-byte identical (no content change without a generation bump). A rewound label
    // generation would let a stale-but-self-consistent slot pose as the next base's bindings.
    const auto cur_label = read_superblock_label_state(superblock_);
    const auto img_label = read_superblock_label_state(image);
    if (img_label.generation < cur_label.generation) {
      poison("flip image label generation regressed");
    }
    if (img_label.count < cur_label.count) {
      poison("flip image label count regressed");
    }
    if (img_label.generation == cur_label.generation &&
        (img_label.slot != cur_label.slot || img_label.count != cur_label.count ||
         img_label.checksum != cur_label.checksum)) {
      poison("flip image same-generation label tuple differs (content changed without a bump)");
    }
    // Overflow: minting a NEW label tuple (a differing slot/count/checksum) once the label
    // generation is already saturated is impossible (the checkpoint's label_generation_ + 1 would
    // wrap to 0, which the regress check above also catches). Reject explicitly for clarity.
    if (cur_label.generation == (std::numeric_limits<uint64_t>::max)() &&
        (img_label.slot != cur_label.slot || img_label.generation != cur_label.generation ||
         img_label.count != cur_label.count || img_label.checksum != cur_label.checksum)) {
      poison("flip image label generation overflow (new tuple over a saturated generation)");
    }
  }

  enum class ReplayFlipDisposition { kStale, kRetained, kInstall };
  struct PreparedReplayFlip {
    QGSuperblockV2 image{};
    ReplayFlipDisposition disposition = ReplayFlipDisposition::kStale;
    int next_slot = -1;
  };

  // Pure/read-only replay phase for a flip. This performs every structural, label, transition and
  // unit-independent check before the caller is allowed to write either staged pages or the base.
  [[nodiscard]] PreparedReplayFlip prepare_replay_flip(const SegmentOp &op,
                                                       uint64_t committed_watermark) {
    if (op.bytes.size() != kSegmentSuperblockImageBytes) {
      poison("flip image is not 512 bytes");
    }
    PreparedReplayFlip prepared;
    std::memcpy(&prepared.image, op.bytes.data(), sizeof(prepared.image));
    QGSuperblockV2 &image = prepared.image;
    if (!qg_superblock_valid(prepared.image)) {
      poison("flip image checksum invalid");
    }
    // BLOCKER-5: fail closed on an UNSUPPORTED (or self-inconsistent) flip image BEFORE any
    // pwrite -- otherwise replay would install a superblock this build cannot reopen, and the
    // segment would only fail closed on the NEXT open (too late, already installed).
    if (!qg_superblock_supported(image, kQgSupportedRequiredFeatures)) {
      poison("flip image requires an unsupported / self-inconsistent feature set");
    }
    if (read_superblock_uid(image) != segment_uid_) {
      poison("flip image lineage mismatch");
    }
    const uint64_t disk_generation = superblock_.generation;
    const uint64_t image_generation = image.generation;
    // B3 (leg-9, r3 section 1 point 5): the op's declared generation must match the image it
    // carries for EVERY flip -- stale, same-generation retained marker, AND G+1 install. Checking
    // it only inside validate_flip_transition (reached solely on the install path) let a
    // same-generation retained flip whose frame generation is forged to G+1 slip past the
    // byte-identical early return below. A frame header generation that disagrees with its 512-byte
    // payload generation is malformed regardless of which transition it encodes.
    if (op.segment_generation != image_generation) {
      poison("flip op generation != the image generation");
    }
    if (image_generation < disk_generation) {
      prepared.disposition = ReplayFlipDisposition::kStale;
      return prepared;
    }
    if (image_generation == disk_generation) {
      // Same generation must be byte-identical (idempotent re-adopt of the base).
      if (std::memcmp(&image, &superblock_, sizeof(image)) != 0) {
        poison("flip same-generation image differs from the selected superblock");
      }
      prepared.disposition = ReplayFlipDisposition::kRetained;
      return prepared;
    }
    if (image_generation != disk_generation + 1) {
      poison("flip generation transition is not the next legal step");
    }
    // BLOCKER-3 (leg-7): fully validate the image's label slot + pid-generation summary BEFORE
    // any pwrite. A crafted flip whose label state is illegal must be rejected here so no
    // higher-generation superblock is ever installed on disk (which later reopens would keep
    // re-selecting). adopt_label_state() below re-runs the same checks and mutates the members.
    validate_flip_label_state(image);
    // BLOCKER-3 (leg-8, r2 section 1) + B3 completion (leg-9, r3 section 1): full structural
    // transition validation (immutable geometry, exact file length, inactive target slot,
    // generation step, state monotonicity, real handle capacity, expected HWM, count joint
    // invariants, label tuple transition) BEFORE any pwrite/ftruncate -- a crafted
    // geometry/short-file/active-slot/ capacity+1/HWM-regress/old-label-slot flip is rejected here
    // with the index byte-for-byte unchanged, instead of installing and only failing closed on the
    // next open. The expected HWM is the replayed committed watermark at this flip: a legal
    // checkpoint captured committed_ into image.num_points, so a G+1 install image whose num_points
    // disagrees with the recovered prefix is forged.
    validate_flip_transition(image, op.segment_generation, op.target_slot, committed_watermark);
    prepared.disposition = ReplayFlipDisposition::kInstall;
    prepared.next_slot = static_cast<int>(op.target_slot);
    return prepared;
  }

  void apply_prepared_replay_flip(const PreparedReplayFlip &prepared,
                                  uint64_t &committed_watermark) {
    if (prepared.disposition != ReplayFlipDisposition::kInstall) {
      return;
    }
    // Apply the next legal base: rewrite its slot and set the file length.
    write_superblock(prepared.next_slot, prepared.image);
    if (::ftruncate(fd_, static_cast<off_t>(prepared.image.file_size)) != 0) {
      poison("flip replay ftruncate failed");
    }
    superblock_ = prepared.image;
    active_superblock_slot_ = prepared.next_slot;
    qg_.entry_point_ = static_cast<PID>(prepared.image.entry_point);
    committed_.store(prepared.image.num_points, std::memory_order_release);
    // The new base is the replay cursor from here on; the local running watermark must track it
    // (the final committed_.store uses this local). For a legal WAL the install flip is the last
    // frame, so this equals the value validate_flip_transition just checked, but keeping it
    // explicit (not a coincidence of "no frames follow the flip") is what makes the expected-HWM
    // invariant sound.
    committed_watermark = prepared.image.num_points;
    // Re-adopt the base's label + tx state: the flip carried a (possibly new)
    // label slot tuple and tx watermarks; load the persisted bindings from it so
    // any bundles it already absorbed are reflected without re-promotion.
    adopt_label_state(prepared.image);
  }

  void replay_flip(const SegmentOp &op, uint64_t &committed_watermark) {
    const PreparedReplayFlip prepared = prepare_replay_flip(op, committed_watermark);
    apply_prepared_replay_flip(prepared, committed_watermark);
  }

  void replay_flip_with_effect_unit(const SegmentOp &op,
                                    uint64_t &committed_watermark,
                                    EffectCommitUnit &unit) {
    // Complete flip validation comes first. In particular, a malformed G+1 image cannot cause an
    // otherwise valid staged page to be written before the flip is rejected.
    const PreparedReplayFlip prepared = prepare_replay_flip(op, committed_watermark);
    if (unit.active()) {
      if (unit.legacy_txid != 0) {
        poison("op-WAL flip cannot consume a legacy-owned effect unit");
      }
      if (!unit.allocation_evidence.empty()) {
        poison("op-WAL flip cannot absorb unpublished allocation evidence");
      }
      if (prepared.disposition != ReplayFlipDisposition::kInstall && unit.has_staged_effects()) {
        poison("op-WAL retained/stale flip has an unresolved current effect unit");
      }
      if (prepared.disposition == ReplayFlipDisposition::kInstall) {
        validate_effect_unit_pages(unit);  // all refs re-read/re-CRC before the first page write
        apply_effect_unit_pages(unit);
      }
      unit.clear();
    }
    // A valid flip is an explicit orphan-bind boundary for every disposition. An install will also
    // clear this in adopt_label_state(), but clearing here covers stale/retained exits as well.
    staged_binds_.clear();
    apply_prepared_replay_flip(prepared, committed_watermark);
  }

  // The single authoritative post-replay pass (clause C): derive the whole
  // runtime state tuple from the recovered on-disk trailers and set the physical
  // length. Under enable_wal there is no PID reuse, so the free list is empty.
  void rebuild_state_after_replay(uint64_t committed) {
    allocated_points_.store(committed, std::memory_order_release);
    next_append_id_.store(static_cast<PID>(committed), std::memory_order_release);
    qg_.num_points_ = committed;
    free_list_head_.store(kPidMax, std::memory_order_release);
    free_count_.store(0, std::memory_order_release);
    {
      const std::lock_guard<std::mutex> guard(deleted_mutex_);
      deleted_.clear();
    }
    reset_hidden();
    uint64_t live = 0;
    const size_t pages = committed == 0 ? 0 : (committed + npp_ - 1) / npp_;
    // FREE reference integrity (design section 2.3). Collect the final free set so
    // the post-scan pass can reject a "pseudo-complete" recovery that would hand a
    // still-referenced row to the reuser. Under W0 no row is ever FREE (reuse is
    // gated), so free_pid stays empty and the adjacency pass is skipped.
    std::vector<uint8_t> free_pid(committed == 0 ? 0 : static_cast<size_t>(committed), 0);
    AlignedBuf page(page_size_);
    for (size_t pi = 0; pi < pages; ++pi) {
      read_at(kSectorLen + pi * page_size_, page.data(), page_size_);
      for (size_t slot = 0; slot < npp_; ++slot) {
        const size_t raw = pi * npp_ + slot;
        if (raw >= committed) {
          break;
        }
        const auto id = static_cast<PID>(raw);
        const QGRowTrailer trailer = qg_read_page_trailer(page.data(), page_size_, npp_, slot);
        if ((trailer.flags & kQGRowFree) != 0 && (trailer.flags & kQGRowTombstone) == 0) {
          poison("recovery: a FREE row is missing the TOMBSTONE flag");
        }
        if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
          mark_hidden(id);
          mirror_deleted_insert(id);
          if ((trailer.flags & kQGRowFree) != 0) {
            free_pid[static_cast<size_t>(id)] = 1;
          }
        } else {
          ++live;
        }
      }
    }
    // A live row's valid adjacency prefix must never point at a FREE PID: such an
    // edge would let the reuser overwrite a row that is still referenced (design
    // section 2.3). Only runs when the segment actually has free rows.
    bool any_free = false;
    for (uint8_t f : free_pid) {
      if (f != 0) {
        any_free = true;
        break;
      }
    }
    if (any_free) {
      for (size_t pi = 0; pi < pages; ++pi) {
        read_at(kSectorLen + pi * page_size_, page.data(), page_size_);
        for (size_t slot = 0; slot < npp_; ++slot) {
          const size_t raw = pi * npp_ + slot;
          if (raw >= committed) {
            break;
          }
          const auto id = static_cast<PID>(raw);
          const QGRowTrailer trailer = qg_read_page_trailer(page.data(), page_size_, npp_, slot);
          if ((trailer.flags & (kQGRowTombstone | kQGRowFree)) != 0) {
            continue;  // hidden rows are not searchable; their edges never route
          }
          const char *row = page.data() + slot * node_len_;
          const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
          for (size_t j = 0; j < trailer.valid_degree; ++j) {
            if (ids[j] < committed && free_pid[static_cast<size_t>(ids[j])] != 0) {
              poison("recovery: a live row's edge still targets a FREE PID");
            }
          }
        }
      }
    }
    // Post-redo free-list convergence (design section 2.1/2.2): derive the final free
    // set from the recovered trailers and canonicalize its chain in ascending PID
    // order, writing each free row's next-free pointer straight to disk. This is
    // byte-stable across repeated recovery and matches the runtime reclaim chain.
    if (any_free) {
      std::vector<PID> free_ids;
      free_ids.reserve(free_pid.size());
      for (size_t id = 0; id < static_cast<size_t>(committed); ++id) {
        if (free_pid[id] != 0) {
          // A FREE PID whose durable generation is already saturated is a crafted /
          // corrupt state: it must have stayed a permanent tombstone (design 2.3).
          const auto it = label_working_.find(static_cast<PID>(id));
          if (it != label_working_.end() &&
              it->second.pid_generation == (std::numeric_limits<uint32_t>::max)()) {
            poison("recovery: a FREE PID has a saturated (UINT32_MAX) generation");
          }
          free_ids.push_back(static_cast<PID>(id));
        }
      }
      PID head = kPidMax;
      AlignedBuf fpage(page_size_);
      for (auto it = free_ids.rbegin(); it != free_ids.rend(); ++it) {
        const PID id = *it;
        read_at(page_offset(id), fpage.data(), page_size_);
        const uint64_t next64 = head;
        std::memcpy(fpage.data() + node_offset_in_page(id), &next64, sizeof(next64));
        write_at(page_offset(id), fpage.data(), page_size_);
        head = id;
      }
      free_list_head_.store(head, std::memory_order_release);
      free_count_.store(free_ids.size(), std::memory_order_release);
    }
    live_count_.store(live, std::memory_order_release);
    for (size_t id = 0; id < row_generations_.size(); ++id) {
      row_generations_[id].store(id < committed ? 1 : 0, std::memory_order_relaxed);
    }
    const uint64_t file_size = kSectorLen + pages * page_size_;
    if (::ftruncate(fd_, static_cast<off_t>(file_size)) != 0) {
      poison("recovery ftruncate failed");
    }
    repair_routing_roots(kPidMax);
    refresh_routing_snapshot();
    free_chain_rebuild_complete_ = true;  // the free chain is now canonical (design B.2)
  }
