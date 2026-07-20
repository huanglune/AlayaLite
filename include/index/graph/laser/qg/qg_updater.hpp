// SPDX-FileCopyrightText: 2025 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

/**
 * @file qg_updater.hpp
 * @brief RESEARCH PROTOTYPE: streaming updates (insert / tombstone-delete /
 * consolidation) for the LASER on-disk quantized graph.
 *
 * Feasibility basis: LASER's per-edge RaBitQ payload (1-bit sign code +
 * triple_x/factor_dq/factor_vq) for an edge u->v depends only on the fixed FHT
 * rotator and the raw (PCA-domain) vectors of u and v — there is no global
 * codebook to retrain. The FastScan 32-slot interleave is a lossless
 * permutation, so a single logical slot can be replaced by unpacking one
 * 32-slot block, swapping the code, and repacking. File addressing is pure id
 * arithmetic; inserts either reuse a consolidated free PID in place or append
 * at EOF.
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
 * Concurrency model (multi-writer inserts, batch three-phase publish):
 *   - The caller runs a parallel append batch with insert_with_id(), or lets
 *     allocate_and_insert() pop reclaimed PIDs. publish() advances the append
 *     watermark and simultaneously removes reused PIDs from the tombstone
 *     result filter after their rows/backlinks are complete.
 *   - Every row write (append, backlink patch, consolidation rewrite) runs
 *     under a striped page-lock table, and bumps a per-page seqlock version
 *     (odd = write in progress). Lock-free search reads validate the version
 *     before/after the pread and retry on a torn page.
 *   - tombstone() / consolidate() must not run concurrently with inserts
 *     (phase separation is the caller's responsibility).
 *
 * Deletes have persistent trailer flags plus a RAM result filter (routing can
 * still traverse tombstoned rows until they become free). consolidate() purges dead out-edges:
 * each dead slot is
 * spliced to the nearest live neighbor of the dead node (ranked with zero
 * extra I/O by FastScan-scanning the dead node's own row with the live row's
 * vector as query), or zeroed back into a free ghost slot (regenerating
 * update headroom) when no candidate exists.
 *
 * Format v2 stores authoritative per-row valid_degree/flags trailers and A/B
 * CRC-protected superblocks.
 *
 * Durability (UpdateParams::enable_wal, off by default): when enabled the
 * updater keeps a per-segment after-image op-WAL (`<index>.opwal`, the SEGMENT_OP
 * family of the shared WAL7 envelope, see segment_op_wal.hpp /
 * docs/design/unified-wal-vocabulary.md). A no-steal page cache appends a
 * whole-page after-image before the page is installed, publish() group-commits
 * the batch (fsync) before advancing the watermark, and checkpoint() commits an
 * A/B superblock flip. Reopen runs a dedicated recovery path (read-only base ->
 * WAL redo under a replaying_ guard -> one authoritative trailer scan that
 * rebuilds committed/allocated/next/live/hidden/deleted and the physical length
 * -> routing repair). A durable segment lineage uid in the superblock reserved
 * area rejects a stale/foreign .opwal, and any WAL/critical-index error poisons
 * the writer (fail closed). The G1 minimal safe scope forbids PID reuse/reclaim,
 * consolidate/garden, and bloom under enable_wal — those paths throw and their
 * WAL transaction formats are a later wave. Remaining caller contracts: phase
 * separation (tombstone/consolidate vs inserts) still applies, the labels
 * sidecar is not covered by the op-WAL, and a single writer per segment must be
 * enforced above (W3 handle: exclusive flock + checkpoint/mutation lane).
 *
 * With enable_wal off the paths below are byte-for-byte unchanged; the legacy
 * ghost heuristic exists only in the one-time v1 migration scan.
 */

#pragma once

#include "index/graph/laser/qg/detail/qg_updater_core.hpp"
