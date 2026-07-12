// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file qg_updater.hpp
 * @brief RESEARCH PROTOTYPE: streaming updates (insert / tombstone-delete) for
 * the LASER on-disk quantized graph.
 *
 * Feasibility basis: LASER's per-edge RaBitQ payload (1-bit sign code +
 * triple_x/factor_dq/factor_vq) for an edge u->v depends only on the fixed FHT
 * rotator and the raw (PCA-domain) vectors of u and v — there is no global
 * codebook to retrain. The FastScan 32-slot interleave is a lossless
 * permutation, so a single logical slot can be replaced by unpacking one
 * 32-slot block, swapping the code, and repacking. File addressing is pure id
 * arithmetic, so new nodes append as new pages at EOF.
 *
 * Insert = FreshDiskANN-style: beam search (capturing expanded rows) ->
 * alpha-RobustPrune over captured exact distances -> assemble + append the new
 * row -> patch one reverse edge per chosen neighbor via page RMW.
 *
 * Reverse-edge arms (UpdateParams::backlink_mode):
 *   kNone       — no reverse edges (control arm)
 *   kEvict      — fill a ghost slot, else evict the farthest edge (FastScan
 *                 estimate with the row owner as query) if the new edge is
 *                 shorter ("nearest-only replacement")
 *   kAlphaEvict — kEvict + free alpha-occlusion test against the neighbors of
 *                 v that already sit in this insert's captured search pool
 *   kFullPrune  — read all live neighbors' raw vectors, run full RobustPrune,
 *                 rewrite the whole row (quality reference, R extra reads)
 *
 * Deletes are in-memory tombstones filtered from results at search time
 * (routing still traverses deleted nodes); see QuantizedGraph::set_result_filter.
 *
 * NOT production ready: single-writer only, no WAL / torn-page protection, no
 * persistent tombstones, meta sector rewritten only in finalize(). Ghost slots
 * are detected heuristically (id==0 && code==0 && factors==0) because the v1
 * row format stores no valid-degree.
 */

#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "index/graph/laser/common.hpp"
#include "index/graph/laser/qg/qg.hpp"
#include "index/graph/laser/qg/qg_query.hpp"
#include "index/graph/laser/quantization/fastscan_impl.hpp"
#include "index/graph/laser/quantization/rabitq.hpp"
#include "index/graph/laser/space/l2.hpp"

namespace alaya::laser {

/**
 * @brief Inverse of pack_codes for exactly one 32-slot FastScan block.
 *
 * @param padded_dim  vector dimension (multiple of 64)
 * @param block       packed block, `padded_dim * 4` bytes (32 codes)
 * @param binary_out  32 * (padded_dim/64) uint64 words, slot-major
 */
inline void unpack_codes_block(size_t padded_dim, const uint8_t *block, uint64_t *binary_out) {
  const size_t num_codebook = padded_dim / 4;
  const size_t bytes_per_code = padded_dim / 8;
  std::vector<uint8_t> tmp(kBatchSize * bytes_per_code);

  // Invert pack_codes_helper: nibble columns -> per-code byte rows.
  const uint8_t *codes2 = block;
  for (size_t i = 0; i < num_codebook; i += 2) {
    std::array<uint8_t, kBatchSize> col_lo{};
    std::array<uint8_t, kBatchSize> col_hi{};
    for (size_t j = 0; j < 16; ++j) {
      const uint8_t val0 = codes2[j];
      const uint8_t val1 = codes2[j + 16];
      col_lo[kPerm0[j]] = val0 & 15;
      col_lo[kPerm0[j] + 16] = val0 >> 4;
      col_hi[kPerm0[j]] = val1 & 15;
      col_hi[kPerm0[j] + 16] = val1 >> 4;
    }
    for (size_t j = 0; j < kBatchSize; ++j) {
      tmp[j * bytes_per_code + i / 2] = static_cast<uint8_t>(col_lo[j] | (col_hi[j] << 4));
    }
    codes2 += 32;
  }

  // Invert the per-byte nibble swap.
  for (auto &b : tmp) {
    b = static_cast<uint8_t>((b << 4) | (b >> 4));
  }
  // Invert the byte reversal inside each 8-byte (64-bit) group.
  for (size_t i = 0; i < kBatchSize; ++i) {
    for (size_t j = 0; j < padded_dim / 64; ++j) {
      for (size_t k = 0; k < 4; ++k) {
        std::swap(tmp[(i * bytes_per_code) + (8 * j) + k],
                  tmp[(i * bytes_per_code) + (8 * j) + 8 - k - 1]);
      }
    }
  }
  std::memcpy(binary_out, tmp.data(), tmp.size());
}

/** @brief Per-edge RaBitQ payload for one neighbor slot. */
struct EdgePayload {
  std::vector<uint64_t> bin;  // padded_dim/64 sign words (pre-pack layout)
  float triple_x = 0;
  float factor_dq = 0;
  float factor_vq = 0;
  bool degenerate = false;
};

/**
 * @brief Compute the RaBitQ payload of a single edge u->v.
 *
 * Mirrors rabitq_codes()/rabitq_factors() for one row so a patched slot is
 * byte-identical to what the static builder would produce.
 *
 * @param c_rot        rot(u) — rotated main-dim vector of the row owner
 * @param x_rot        rot(v) — rotated main-dim vector of the new neighbor
 * @param padded_dim   main (== padded) dimension
 * @param x_res_sqr    ||v_residual||^2, pre-added to triple_x like the builder
 */
inline EdgePayload make_edge_payload(const float *c_rot,
                                     const float *x_rot,
                                     size_t padded_dim,
                                     float x_res_sqr) {
  EdgePayload out;
  out.bin.assign(padded_dim / 64, 0);

  // Degeneracy pre-check: identical main-dim vectors cannot be sign-encoded.
  double norm_sqr = 0;
  for (size_t j = 0; j < padded_dim; ++j) {
    const double r = static_cast<double>(x_rot[j]) - c_rot[j];
    norm_sqr += r * r;
  }
  if (!(norm_sqr > 0)) {
    out.degenerate = true;
    return out;
  }

  // Delegate to the builder's own kernel with a 1-row matrix so the patched
  // slot is bit-identical to a builder-written slot (same Eigen accumulation
  // order and float rounding).
  RowMatrix<float> x(1, static_cast<int64_t>(padded_dim));
  RowMatrix<float> c(1, static_cast<int64_t>(padded_dim));
  for (size_t j = 0; j < padded_dim; ++j) {
    x(0, static_cast<int64_t>(j)) = x_rot[j];
    c(0, static_cast<int64_t>(j)) = c_rot[j];
  }
  std::vector<uint8_t> block(padded_dim * 4);  // one 32-slot FastScan block
  rabitq_codes(x, c, block.data(), &out.triple_x, &out.factor_dq, &out.factor_vq);
  out.triple_x += x_res_sqr;

  std::vector<uint64_t> bins(kBatchSize * padded_dim / 64);
  unpack_codes_block(padded_dim, block.data(), bins.data());
  std::copy(bins.begin(), bins.begin() + static_cast<int64_t>(padded_dim / 64), out.bin.begin());

  if (!std::isfinite(out.triple_x) || !std::isfinite(out.factor_dq) ||
      !std::isfinite(out.factor_vq)) {
    out.degenerate = true;
  }
  return out;
}

struct UpdateStats {
  uint64_t inserts = 0;
  uint64_t search_page_reads = 0;
  uint64_t patch_page_reads = 0;   // reverse-edge RMW reads (incl. full-prune vector reads)
  uint64_t page_writes = 0;
  uint64_t free_slot_fills = 0;
  uint64_t evictions = 0;
  uint64_t est_skips = 0;    // new edge longer than current farthest -> skipped
  uint64_t alpha_skips = 0;  // occluded by an already-captured neighbor -> skipped
  uint64_t degenerate_skips = 0;
  uint64_t full_recomputes = 0;
  uint64_t forced_links = 0;  // inserts whose only backlink came from the force policy
};

struct UpdateParams {
  size_t ef_insert = 100;
  float alpha = 1.2F;
  size_t prune_pool_cap = 300;  // candidates fed to RobustPrune
  enum class Backlink { kNone, kEvict, kAlphaEvict, kFullPrune };
  Backlink backlink_mode = Backlink::kAlphaEvict;
  size_t alpha_check_max = 16;  // captured neighbors tested per reverse edge
};

class QGUpdater {
 public:
  QGUpdater(QuantizedGraph &qg, UpdateParams params)
      : qg_(qg),
        params_(params),
        num_points_(qg.num_points_),
        dim_(qg.dimension_),
        res_dim_(qg.residual_dimension_),
        full_dim_(qg.dimension_ + qg.residual_dimension_),
        pd_(qg.padded_dim_),
        deg_(qg.degree_bound_),
        node_len_(qg.node_len_),
        page_size_(qg.page_size_),
        npp_(qg.node_per_page_) {
    if (qg_.index_file_name_.empty()) {
      throw std::logic_error("QGUpdater: call load_disk_index() first");
    }
    fd_ = ::open(qg_.index_file_name_.c_str(), O_RDWR);
    if (fd_ < 0) {
      throw std::runtime_error("QGUpdater: cannot open for write: " + qg_.index_file_name_ +
                               " errno=" + std::to_string(errno));
    }
    page_scratch_.resize(page_size_);
    row_scratch_.resize(node_len_);
  }

  QGUpdater(const QGUpdater &) = delete;
  auto operator=(const QGUpdater &) -> QGUpdater & = delete;

  ~QGUpdater() {
    if (fd_ >= 0) {
      ::close(fd_);
    }
  }

  [[nodiscard]] size_t num_points() const { return num_points_; }
  [[nodiscard]] const UpdateStats &stats() const { return stats_; }
  [[nodiscard]] const std::unordered_set<PID> &deleted() const { return deleted_; }

  /** @brief Tombstone a node: routing still passes through, results filter it. */
  void tombstone(PID id) { deleted_.insert(id); }

  /**
   * @brief Insert one vector (original query domain; PCA applied internally).
   * @return the id assigned to the new node
   */
  PID insert(const float *vec) {
    // --- PCA transform (same path queries take) ---
    const float *tvec = vec;
    std::vector<float> pca_buf;
    if (qg_.pca_transform_.is_loaded()) {
      pca_buf.resize(full_dim_);
      qg_.pca_transform_.transform(vec, pca_buf.data());
      tvec = pca_buf.data();
    }

    // --- beam search with row capture ---
    std::vector<CapturedNode> pool;
    search_for_insert(tvec, pool);

    // --- alpha-RobustPrune over exact distances ---
    std::vector<size_t> sel;
    robust_prune(tvec, pool, sel);

    // --- assemble + append the new row ---
    const PID new_id = static_cast<PID>(num_points_);
    std::vector<const float *> nb_vecs;
    std::vector<PID> nb_ids;
    nb_vecs.reserve(sel.size());
    nb_ids.reserve(sel.size());
    for (size_t idx : sel) {
      nb_vecs.push_back(pool[idx].raw());
      nb_ids.push_back(pool[idx].id);
    }
    assemble_row(row_scratch_.data(), tvec, nb_vecs, nb_ids);
    append_node(new_id, row_scratch_.data());
    num_points_ = new_id + 1;

    // --- reverse edges ---
    if (params_.backlink_mode != UpdateParams::Backlink::kNone && !sel.empty()) {
      std::unordered_map<PID, const CapturedNode *> captured;
      if (params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict) {
        captured.reserve(pool.size());
        for (const auto &c : pool) {
          captured.emplace(c.id, &c);
        }
      }
      size_t links = 0;
      for (size_t idx : sel) {
        if (params_.backlink_mode == UpdateParams::Backlink::kFullPrune) {
          links += static_cast<size_t>(full_reverse_recompute(pool[idx], new_id, tvec));
        } else {
          links += static_cast<size_t>(
              patch_reverse_edge(pool[idx], new_id, tvec, captured, /*force=*/false));
        }
      }
      // Reachability guarantee: a node with zero in-edges is unsearchable, so
      // force one link onto the closest selected neighbor (evicting its
      // farthest edge unconditionally).
      if (links == 0) {
        if (patch_reverse_edge(pool[sel[0]], new_id, tvec, captured, /*force=*/true)) {
          stats_.forced_links++;
        }
      }
    }

    stats_.inserts++;
    return new_id;
  }

  /**
   * @brief Rewrite the meta sector (num_points, file size) and fsync, so a
   * fresh QuantizedGraph(num_points(), ...).load_disk_index() accepts the file.
   */
  void finalize() {
    std::vector<uint64_t> metas(kSectorLen / sizeof(uint64_t), 0);
    read_at(0, reinterpret_cast<char *>(metas.data()), kSectorLen);
    metas[0] = num_points_;
    const size_t page_num = (num_points_ + npp_ - 1) / npp_;
    metas[8] = kSectorLen + page_num * page_size_;
    write_at(0, reinterpret_cast<const char *>(metas.data()), kSectorLen);
    // Pages are written on append, but a trailing partially-filled page may
    // leave the file shorter than meta[8]; extend explicitly.
    if (::ftruncate(fd_, static_cast<off_t>(metas[8])) != 0) {
      throw std::runtime_error("QGUpdater: ftruncate failed");
    }
    if (::fsync(fd_) != 0) {
      throw std::runtime_error("QGUpdater: fsync failed");
    }
    qg_.num_points_ = num_points_;
  }

  /** @brief Ghost-slot heuristic: zero id, zero factors, zero code bits. */
  bool is_ghost_slot(const char *row, size_t slot) const {
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

 private:
  struct CapturedNode {
    PID id = 0;
    float dist = 0;  // exact full-dim squared L2 to the inserted vector
    std::vector<char> row;
    [[nodiscard]] const float *raw() const { return reinterpret_cast<const float *>(row.data()); }
  };

  [[nodiscard]] uint64_t page_offset(PID id) const {
    return kSectorLen + page_size_ * (static_cast<uint64_t>(id) / npp_);
  }
  [[nodiscard]] size_t node_offset_in_page(PID id) const { return (id % npp_) * node_len_; }

  void read_at(uint64_t off, char *buf, size_t len) const {
    size_t done = 0;
    while (done < len) {
      const ssize_t r = ::pread(fd_, buf + done, len - done, static_cast<off_t>(off + done));
      if (r < 0) {
        throw std::runtime_error("QGUpdater: pread failed errno=" + std::to_string(errno));
      }
      if (r == 0) {  // reading a page that does not exist yet -> zero-fill
        std::memset(buf + done, 0, len - done);
        return;
      }
      done += static_cast<size_t>(r);
    }
  }

  void write_at(uint64_t off, const char *buf, size_t len) {
    size_t done = 0;
    while (done < len) {
      const ssize_t w = ::pwrite(fd_, buf + done, len - done, static_cast<off_t>(off + done));
      if (w < 0) {
        throw std::runtime_error("QGUpdater: pwrite failed errno=" + std::to_string(errno));
      }
      done += static_cast<size_t>(w);
    }
    stats_.page_writes++;
  }

  /** @brief Greedy beam search over the on-disk graph, capturing expanded rows. */
  void search_for_insert(const float *tvec, std::vector<CapturedNode> &pool) {
    QGQuery q_obj(tvec, pd_);
    q_obj.query_prepare(qg_.rotator_, qg_.scanner_);
    const float *res_query = tvec + dim_;
    float sqr_qr = 0;
    for (size_t i = 0; i < res_dim_; ++i) {
      sqr_qr += res_query[i] * res_query[i];
    }
    q_obj.set_sqr_qr(sqr_qr);

    buffer::SearchBuffer sp(params_.ef_insert);
    std::unordered_set<PID> visited;
    visited.reserve(params_.ef_insert * 8);

    if (!qg_.medoids_.empty()) {
      PID best_medoid = 0;
      float best = FLT_MAX;
      for (size_t m = 0; m < qg_.medoids_.size(); ++m) {
        const float d =
            space::l2_sqr(tvec, qg_.medoids_vector_.data() + full_dim_ * m, dim_);
        if (d < best) {
          best = d;
          best_medoid = qg_.medoids_[m];
        }
      }
      sp.insert(best_medoid, FLT_MAX);
    }
    sp.insert(qg_.entry_point_, FLT_MAX);

    std::vector<float> appro(deg_);
    std::vector<char> page(page_size_);
    while (sp.has_next()) {
      const PID cur = sp.pop();
      if (visited.count(cur) != 0) {
        continue;
      }
      visited.insert(cur);
      if (cur >= num_points_) {
        continue;
      }
      read_at(page_offset(cur), page.data(), page_size_);
      stats_.search_page_reads++;
      const char *row = page.data() + node_offset_in_page(cur);
      const auto *row_f = reinterpret_cast<const float *>(row);

      float sqr_y = space::l2_sqr(tvec, row_f, dim_);
      // FastScan over this node's neighbors
      qg_.scanner_.scan_neighbors(appro.data(),
                                  q_obj.lut().data(),
                                  sqr_y,
                                  q_obj.lower_val(),
                                  q_obj.width(),
                                  q_obj.sqr_qr(),
                                  q_obj.sumq(),
                                  reinterpret_cast<const uint8_t *>(row + code_off_bytes()),
                                  reinterpret_cast<const float *>(row + factor_off_bytes()));
      const auto *nbs = reinterpret_cast<const PID *>(row + neighbor_off_bytes());
      const auto *fac = reinterpret_cast<const float *>(row + factor_off_bytes());
      for (size_t j = 0; j < deg_; ++j) {
        const PID nb = nbs[j];
        // ghost-slot fast path: zero id + zero factors (full code check not
        // needed on the search path; a real zero-distance edge to node 0 would
        // just re-enqueue an already-visited node)
        if (nb == 0 && fac[j] == 0 && fac[deg_ + j] == 0 && fac[2 * deg_ + j] == 0) {
          continue;
        }
        if (nb >= num_points_ || visited.count(nb) != 0 || sp.is_full(appro[j])) {
          continue;
        }
        sp.insert(nb, appro[j]);
      }

      if (res_dim_ > 0) {
        sqr_y += space::l2_sqr(row_f + dim_, res_query, res_dim_);
      }
      CapturedNode cap;
      cap.id = cur;
      cap.dist = sqr_y;
      cap.row.assign(row, row + node_len_);
      pool.push_back(std::move(cap));
    }

    std::sort(pool.begin(), pool.end(), [](const CapturedNode &a, const CapturedNode &b) {
      return a.dist < b.dist;
    });
    if (pool.size() > params_.prune_pool_cap) {
      pool.resize(params_.prune_pool_cap);
    }
  }

  /** @brief DiskANN-style alpha occlusion prune over the captured pool. */
  void robust_prune(const float *tvec, const std::vector<CapturedNode> &pool,
                    std::vector<size_t> &sel) {
    (void)tvec;
    sel.clear();
    for (size_t i = 0; i < pool.size() && sel.size() < deg_; ++i) {
      if (deleted_.count(pool[i].id) != 0) {
        continue;
      }
      bool occluded = false;
      for (size_t s : sel) {
        const float d_sc = space::l2_sqr(pool[s].raw(), pool[i].raw(), dim_) +
                           (res_dim_ > 0 ? space::l2_sqr(pool[s].raw() + dim_,
                                                         pool[i].raw() + dim_, res_dim_)
                                         : 0.0F);
        // squared-distance domain: alpha^2 * d2(s,c) <= d2(x,c) prunes c
        if (params_.alpha * params_.alpha * d_sc <= pool[i].dist) {
          occluded = true;
          break;
        }
      }
      if (!occluded) {
        sel.push_back(i);
      }
    }
  }

  void append_node(PID id, const char *row) {
    const uint64_t off = page_offset(id);
    if (npp_ > 1) {
      read_at(off, page_scratch_.data(), page_size_);
      stats_.patch_page_reads++;
    } else {
      std::memset(page_scratch_.data(), 0, page_size_);
    }
    std::memcpy(page_scratch_.data() + node_offset_in_page(id), row, node_len_);
    write_at(off, page_scratch_.data(), page_size_);
  }

  /**
   * @brief Cheap reverse-edge patch: fill a ghost slot, else evict the
   * farthest current edge (FastScan estimate with v as its own query) when the
   * new edge is shorter. kAlphaEvict additionally rejects the new edge when an
   * already-captured current neighbor alpha-occludes it (zero extra I/O).
   */
  bool patch_reverse_edge(const CapturedNode &v, PID x_id, const float *x_vec,
                          const std::unordered_map<PID, const CapturedNode *> &captured,
                          bool force) {
    if (deleted_.count(v.id) != 0) {
      return false;
    }
    const uint64_t off = page_offset(v.id);
    read_at(off, page_scratch_.data(), page_size_);
    stats_.patch_page_reads++;
    char *row = page_scratch_.data() + node_offset_in_page(v.id);
    const auto *row_f = reinterpret_cast<const float *>(row);
    auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());

    // residual-consistent comparison value for the new edge (v,x)
    float x_res_sqr = 0;
    float v_res_sqr = 0;
    for (size_t j = 0; j < res_dim_; ++j) {
      x_res_sqr += x_vec[dim_ + j] * x_vec[dim_ + j];
      v_res_sqr += row_f[dim_ + j] * row_f[dim_ + j];
    }
    const float est_new = space::l2_sqr(row_f, x_vec, dim_) + x_res_sqr + v_res_sqr;

    // alpha-occlusion test using neighbors already captured by this search
    if (!force && params_.backlink_mode == UpdateParams::Backlink::kAlphaEvict &&
        !captured.empty()) {
      size_t checked = 0;
      const float d_vx = space::l2_sqr(row_f, x_vec, dim_) +
                         (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_)
                                       : 0.0F);
      for (size_t j = 0; j < deg_ && checked < params_.alpha_check_max; ++j) {
        if (ids[j] == x_id || is_free_fast(fac, ids, j)) {
          continue;
        }
        auto it = captured.find(ids[j]);
        if (it == captured.end()) {
          continue;
        }
        ++checked;
        const float d_sx =
            space::l2_sqr(it->second->raw(), x_vec, dim_) +
            (res_dim_ > 0
                 ? space::l2_sqr(it->second->raw() + dim_, x_vec + dim_, res_dim_)
                 : 0.0F);
        if (params_.alpha * params_.alpha * d_sx <= d_vx) {
          stats_.alpha_skips++;
          return false;
        }
      }
    }

    // choose a slot: first ghost slot, else evict farthest by FastScan estimate
    size_t slot = deg_;
    for (size_t j = 0; j < deg_; ++j) {
      if (is_free_fast(fac, ids, j) && is_ghost_slot(row, j)) {
        slot = j;
        break;
      }
    }
    bool evicted = false;
    if (slot == deg_) {
      QGQuery vq(row_f, pd_);
      vq.query_prepare(qg_.rotator_, qg_.scanner_);
      vq.set_sqr_qr(v_res_sqr);
      std::vector<float> appro(deg_);
      qg_.scanner_.scan_neighbors(appro.data(),
                                  vq.lut().data(),
                                  0.0F,
                                  vq.lower_val(),
                                  vq.width(),
                                  vq.sqr_qr(),
                                  vq.sumq(),
                                  reinterpret_cast<const uint8_t *>(row + code_off_bytes()),
                                  fac);
      float worst = -1;
      for (size_t j = 0; j < deg_; ++j) {
        if (ids[j] == x_id) {
          return true;  // already linked (shouldn't happen for a fresh id)
        }
        if (appro[j] > worst) {
          worst = appro[j];
          slot = j;
        }
      }
      if (!force && est_new >= worst) {
        stats_.est_skips++;
        return false;
      }
      evicted = true;
    }

    if (!patch_slot(row, slot, x_id, x_vec, x_res_sqr)) {
      stats_.degenerate_skips++;
      return false;
    }
    write_at(off, page_scratch_.data(), page_size_);
    if (evicted) {
      stats_.evictions++;
    } else {
      stats_.free_slot_fills++;
    }
    return true;
  }

  /** @brief Quality-reference arm: full RobustPrune with all neighbor vectors read back. */
  bool full_reverse_recompute(const CapturedNode &v, PID x_id, const float *x_vec) {
    if (deleted_.count(v.id) != 0) {
      return false;
    }
    const uint64_t off = page_offset(v.id);
    read_at(off, page_scratch_.data(), page_size_);
    stats_.patch_page_reads++;
    char *row = page_scratch_.data() + node_offset_in_page(v.id);
    const auto *row_f = reinterpret_cast<const float *>(row);
    auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());

    // gather live neighbors + the new node as prune candidates
    struct Cand {
      PID id;
      float dist;
      std::vector<float> vec;
    };
    std::vector<Cand> cands;
    cands.reserve(deg_ + 1);
    std::vector<char> nb_page(page_size_);
    for (size_t j = 0; j < deg_; ++j) {
      const PID nb = ids[j];
      if (nb >= num_points_ || is_ghost_slot(row, j) || nb == v.id ||
          deleted_.count(nb) != 0) {
        continue;
      }
      bool dup = false;
      for (const auto &c : cands) {
        if (c.id == nb) {
          dup = true;
          break;
        }
      }
      if (dup) {
        continue;
      }
      read_at(page_offset(nb), nb_page.data(), page_size_);
      stats_.patch_page_reads++;
      const auto *nb_f =
          reinterpret_cast<const float *>(nb_page.data() + node_offset_in_page(nb));
      Cand c;
      c.id = nb;
      c.vec.assign(nb_f, nb_f + full_dim_);
      c.dist = space::l2_sqr(row_f, c.vec.data(), dim_) +
               (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, c.vec.data() + dim_, res_dim_)
                             : 0.0F);
      cands.push_back(std::move(c));
    }
    {
      Cand c;
      c.id = x_id;
      c.vec.assign(x_vec, x_vec + full_dim_);
      c.dist = space::l2_sqr(row_f, x_vec, dim_) +
               (res_dim_ > 0 ? space::l2_sqr(row_f + dim_, x_vec + dim_, res_dim_) : 0.0F);
      cands.push_back(std::move(c));
    }
    std::sort(cands.begin(), cands.end(),
              [](const Cand &a, const Cand &b) { return a.dist < b.dist; });

    std::vector<size_t> sel;
    for (size_t i = 0; i < cands.size() && sel.size() < deg_; ++i) {
      bool occluded = false;
      for (size_t s : sel) {
        const float d_sc =
            space::l2_sqr(cands[s].vec.data(), cands[i].vec.data(), dim_) +
            (res_dim_ > 0 ? space::l2_sqr(cands[s].vec.data() + dim_,
                                          cands[i].vec.data() + dim_, res_dim_)
                          : 0.0F);
        if (params_.alpha * params_.alpha * d_sc <= cands[i].dist) {
          occluded = true;
          break;
        }
      }
      if (!occluded) {
        sel.push_back(i);
      }
    }

    std::vector<float> v_vec(row_f, row_f + full_dim_);  // row buffer is rewritten below
    std::vector<const float *> nb_vecs;
    std::vector<PID> nb_ids;
    bool x_survived = false;
    for (size_t s : sel) {
      nb_vecs.push_back(cands[s].vec.data());
      nb_ids.push_back(cands[s].id);
      x_survived = x_survived || (cands[s].id == x_id);
    }
    assemble_row(row, v_vec.data(), nb_vecs, nb_ids);
    write_at(off, page_scratch_.data(), page_size_);
    stats_.full_recomputes++;
    return x_survived;
  }

  /** @brief Replace one slot's code + factors + id inside a row buffer. */
  bool patch_slot(char *row, size_t slot, PID x_id, const float *x_vec, float x_res_sqr) {
    const auto *row_f = reinterpret_cast<const float *>(row);
    std::vector<float> c_pad(pd_, 0.0F);
    std::vector<float> x_pad(pd_, 0.0F);
    std::copy(row_f, row_f + dim_, c_pad.begin());
    std::copy(x_vec, x_vec + dim_, x_pad.begin());
    std::vector<float> c_rot(pd_);
    std::vector<float> x_rot(pd_);
    qg_.rotator_.rotate(c_pad.data(), c_rot.data());
    qg_.rotator_.rotate(x_pad.data(), x_rot.data());

    EdgePayload payload = make_edge_payload(c_rot.data(), x_rot.data(), pd_, x_res_sqr);
    if (payload.degenerate) {
      return false;
    }

    const size_t block_idx = slot / kBatchSize;
    const size_t in_block = slot % kBatchSize;
    uint8_t *block =
        reinterpret_cast<uint8_t *>(row + code_off_bytes()) + block_idx * pd_ * 4;
    std::vector<uint64_t> bins(kBatchSize * pd_ / 64);
    unpack_codes_block(pd_, block, bins.data());
    std::copy(payload.bin.begin(), payload.bin.end(), bins.begin() + in_block * (pd_ / 64));
    pack_codes(pd_, bins.data(), kBatchSize, block);

    auto *fac = reinterpret_cast<float *>(row + factor_off_bytes());
    fac[slot] = payload.triple_x;
    fac[deg_ + slot] = payload.factor_dq;
    fac[2 * deg_ + slot] = payload.factor_vq;
    auto *ids = reinterpret_cast<PID *>(row + neighbor_off_bytes());
    ids[slot] = x_id;
    return true;
  }

  /** @brief Cheap free-slot pre-filter (id==0 && factors==0); confirm with is_ghost_slot. */
  [[nodiscard]] bool is_free_fast(const float *fac, const PID *ids, size_t j) const {
    return ids[j] == 0 && fac[j] == 0 && fac[deg_ + j] == 0 && fac[2 * deg_ + j] == 0;
  }

  QuantizedGraph &qg_;
  UpdateParams params_;
  int fd_ = -1;
  size_t num_points_;
  size_t dim_, res_dim_, full_dim_, pd_, deg_, node_len_, page_size_, npp_;
  std::unordered_set<PID> deleted_;
  UpdateStats stats_{};
  std::vector<char> page_scratch_;
  std::vector<char> row_scratch_;
};

}  // namespace alaya::laser
