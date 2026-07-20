// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Class-body fragment included exactly once by qg_updater.hpp.
// Debug and test seams.

  bool is_v1_ghost_slot(const char *row, size_t slot) const {
    const auto *ids = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
    if (ids[slot] != 0) {
      return false;
    }
    const auto *fac = reinterpret_cast<const float *>(row + factor_off_bytes());
    if (fac[slot] != 0 || fac[deg_ + slot] != 0 || fac[2 * deg_ + slot] != 0) {
      return false;
    }
    // all code bits of this slot zero?
    const uint8_t *block =
        reinterpret_cast<const uint8_t *>(row + code_off_bytes()) + (slot / kBatchSize) * pd_ * 4;
    std::vector<uint64_t> bins(kBatchSize * pd_ / 64);
    unpack_codes_block(pd_, block, bins.data());
    const uint64_t *b = &bins[(slot % kBatchSize) * (pd_ / 64)];
    for (size_t w = 0; w < pd_ / 64; ++w) {
      if (b[w] != 0) {
        return false;
      }
    }
    return true;
  }

  // exposed for unit tests
  [[nodiscard]] size_t code_off_bytes() const { return qg_.code_offset_ * sizeof(float); }
  [[nodiscard]] size_t factor_off_bytes() const { return qg_.factor_offset_ * sizeof(float); }
  [[nodiscard]] size_t neighbor_off_bytes() const { return qg_.neighbor_offset_ * sizeof(float); }

  [[nodiscard]] QGRowTrailer trailer(PID id) {
    if (id >= allocated_points()) throw std::out_of_range("QGUpdater::trailer id");
    AlignedBuf page(page_size_);
    read_node_page(id, page.data());
    return row_trailer(page.data(), id);
  }

  // Test-only: copy the raw node_len bytes of a row's committed on-disk image, so an
  // oracle test can compare a WAL consolidation against the non-WAL path byte-for-byte
  // (W1 acceptance: (reclaim x bloom x r_target) == non-WAL oracle page hash / free set).
  [[nodiscard]] std::vector<char> debug_read_row(PID id) {
    if (id >= allocated_points()) throw std::out_of_range("QGUpdater::debug_read_row id");
    AlignedBuf page(page_size_);
    read_node_page(id, page.data());
    std::vector<char> row(node_len_);
    std::memcpy(row.data(), page.data() + node_offset_in_page(id), node_len_);
    return row;
  }

  [[nodiscard]] size_t debug_page_size() const { return page_size_; }
  [[nodiscard]] size_t debug_npp() const { return npp_; }

  // The valid neighbor PID list of a row (for BFS reachability tests of the bundle spine).
  [[nodiscard]] std::vector<PID> debug_row_neighbors(PID id) {
    const auto row = debug_read_row(id);
    const auto tr = trailer(id);
    const auto *ids = reinterpret_cast<const PID *>(row.data() + neighbor_off_bytes());
    return std::vector<PID>(ids, ids + tr.valid_degree);
  }

  // Full page bytes (rows + trailers + any inter-row/trailer padding) for the crash-matrix
  // fingerprint: captures the "unused page bytes" a per-row read misses (leg-7 BLOCKER-7). Reads
  // through the CACHE (query_read: never the writer overlay), so it reflects the committed
  // SEMANTIC state and is idempotent across the crash-recovery path -- unlike a raw file read,
  // which would capture non-idempotent partial-crash bytes beyond the committed watermark.
  [[nodiscard]] std::vector<char> debug_read_page(size_t page_index) {
    AlignedBuf page(page_size_);
    read_node_page(static_cast<PID>(page_index * npp_), page.data(), /*query_read=*/true);
    return std::vector<char>(page.data(), page.data() + page_size_);
  }
  [[nodiscard]] uint32_t debug_page_version(size_t page_index) const {
    if (page_index >= page_versions_.size()) {
      throw std::out_of_range("QGUpdater::debug_page_version page");
    }
    return page_versions_[page_index].load(std::memory_order_acquire);
  }
  [[nodiscard]] std::vector<char> debug_read_disk_page(size_t page_index) const {
    if (page_index >= page_versions_.size()) {
      throw std::out_of_range("QGUpdater::debug_read_disk_page page");
    }
    std::vector<char> page(page_size_);
    read_at(kSectorLen + page_index * page_size_, page.data(), page.size());
    return page;
  }
  [[nodiscard]] std::vector<char> debug_read_arena_rows(size_t page_index) const {
    if (!qg_.arena_resident()) {
      throw std::logic_error("QGUpdater::debug_read_arena_rows requires resident arena");
    }
    const size_t first = page_index * npp_;
    const size_t count = first >= committed_.load(std::memory_order_acquire)
                             ? 0
                             : std::min(npp_, committed_.load(std::memory_order_acquire) - first);
    std::vector<char> rows(count * node_len_);
    std::memcpy(rows.data(), qg_.cache_nodes_.data() + first * node_len_, rows.size());
    return rows;
  }
#if defined(ALAYA_LASER_TESTING)
  // Direct callback seam for exception-safety tests of the ordinary cached RMW
  // path. Production callers enter through the typed insert/tombstone helpers.
  template <typename Fn>
  bool debug_modify_node_page(PID id, Fn &&fn) {
    const std::lock_guard<std::mutex> guard(page_lock(id));
    return modify_node_page(id, std::forward<Fn>(fn));
  }
#endif
  // The routing medoid vectors (design section 4): a separate cache that must agree with the
  // medoid PIDs' row data, so a routing-vector divergence a per-row fingerprint would miss is
  // caught (leg-7 BLOCKER-7).
  [[nodiscard]] const std::vector<float> &debug_medoid_vectors() const {
    return qg_.medoids_vector_;
  }

  [[nodiscard]] size_t trailer_offset_in_page(PID id) const {
    return qg_page_trailer_offset(page_size_, npp_, id % npp_);
  }

  /** @brief Assemble a full node row exactly like the static builder. */
  void assemble_row(char *row,
                    const float *vec,
                    const std::vector<const float *> &nb_vecs,
                    const std::vector<PID> &nb_ids) {
    assert(nb_vecs.size() == nb_ids.size());
    assert(nb_ids.size() <= deg_);
    std::memset(row, 0, node_len_);
    std::memcpy(row, vec, full_dim_ * sizeof(float));

    const size_t cur_degree = nb_ids.size();
    auto *ids_out = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    for (size_t i = 0; i < cur_degree; ++i) {
      ids_out[i] = nb_ids[i];
    }
    if (cur_degree == 0) {
      return;
    }

    RowMatrix<float> x_pad(cur_degree, pd_);
    RowMatrix<float> c_pad(1, pd_);
    x_pad.setZero();
    c_pad.setZero();
    for (size_t i = 0; i < cur_degree; ++i) {
      std::copy(nb_vecs[i], nb_vecs[i] + dim_, &x_pad(static_cast<int64_t>(i), 0));
    }
    std::copy(vec, vec + dim_, &c_pad(0, 0));

    RowMatrix<float> x_rot(cur_degree, pd_);
    RowMatrix<float> c_rot(1, pd_);
    for (int64_t i = 0; i < static_cast<int64_t>(cur_degree); ++i) {
      qg_.rotator_.rotate(&x_pad(i, 0), &x_rot(i, 0));
    }
    qg_.rotator_.rotate(&c_pad(0, 0), &c_rot(0, 0));

    auto *fac_ptr = reinterpret_cast<float *>(row + factor_off_bytes());
    auto *code_ptr = reinterpret_cast<uint8_t *>(row + code_off_bytes());
    float *triple_x = fac_ptr;
    float *factor_dq = triple_x + deg_;
    float *factor_vq = factor_dq + deg_;
    rabitq_codes(x_rot, c_rot, code_ptr, triple_x, factor_dq, factor_vq);

    for (size_t i = 0; i < cur_degree; ++i) {
      const float *residual = nb_vecs[i] + dim_;
      float sqr_xr = 0;
      for (size_t j = 0; j < res_dim_; ++j) {
        sqr_xr += residual[j] * residual[j];
      }
      triple_x[i] += sqr_xr;
    }
  }

  /**
   * @brief Precompute one reverse-edge payload without acquiring a page lock.
   *
   * The caller supplies an immutable raw-vector snapshot and the generation
   * captured with it.  `apply_patch_intent()` validates that generation before
   * installing the payload; a stale intent transparently falls back to the
   * legacy locked encoder.
   */
  [[nodiscard]] PatchIntent prepare_patch_intent(PID target_row,
                                                 const float *target_row_snapshot,
                                                 const float *candidate_raw,
                                                 PID candidate_pid,
                                                 uint64_t row_generation) {
    if (target_row_snapshot == nullptr || candidate_raw == nullptr) {
      throw std::invalid_argument("QGUpdater::prepare_patch_intent null vector");
    }
    float candidate_res_sqr = 0;
    for (size_t j = 0; j < res_dim_; ++j) {
      candidate_res_sqr += candidate_raw[dim_ + j] * candidate_raw[dim_ + j];
    }
    thread_local std::vector<float> candidate_pad;
    thread_local std::vector<float> candidate_rot;
    candidate_pad.assign(pd_, 0.0F);
    candidate_rot.resize(pd_);
    std::copy(candidate_raw, candidate_raw + dim_, candidate_pad.begin());
    qg_.rotator_.rotate(candidate_pad.data(), candidate_rot.data());
    return prepare_patch_intent_from_rotated_candidate(target_row,
                                                       target_row_snapshot,
                                                       candidate_raw,
                                                       candidate_rot.data(),
                                                       candidate_res_sqr,
                                                       candidate_pid,
                                                       row_generation);
  }
