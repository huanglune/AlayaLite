// SPDX-FileCopyrightText: 2026 AlayaDB.AI
//
// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "index/graph/laser/qg/detail/qg_updater_api.hpp"

namespace alaya::laser::detail {

[[nodiscard]] PID qg_updater_allocate_and_insert(QGUpdater &updater, const float *vec);
[[nodiscard]] size_t qg_updater_allocated_points(const QGUpdater &updater);
void qg_updater_publish(QGUpdater &updater, size_t new_committed);
[[nodiscard]] PhysicalBundleResult qg_updater_commit_physical_bundle_tokens(QGUpdater &updater,
                                                                            uint64_t txid,
                                                                            uint64_t applied_op_id,
                                                                            const float *vecs,
                                                                            const uint64_t *labels,
                                                                            size_t n);
[[nodiscard]] std::shared_ptr<const LabelBindings> qg_updater_label_snapshot(
    const QGUpdater &updater);
void qg_updater_tombstone(QGUpdater &updater, PID id);
[[nodiscard]] uint32_t qg_updater_durable_generation(const QGUpdater &updater, PID id);
[[nodiscard]] bool qg_updater_row_is_live(const QGUpdater &updater, PID id);
[[nodiscard]] size_t qg_updater_num_points(const QGUpdater &updater);
void qg_updater_writeback(QGUpdater &updater, size_t num_threads);
void qg_updater_consolidate(QGUpdater &updater,
                            size_t num_threads,
                            size_t r_target,
                            bool reclaim_slots,
                            bool bloom_consolidate);
[[nodiscard]] uint64_t qg_updater_free_count(const QGUpdater &updater);
[[nodiscard]] bool qg_updater_pid_generation_activated(const QGUpdater &updater);
[[nodiscard]] bool qg_updater_is_poisoned(const QGUpdater &updater) noexcept;
void qg_updater_checkpoint(QGUpdater &updater);
void qg_updater_require_dual_v3_if_activated(QGUpdater &updater);
void qg_updater_ensure_readable(const QGUpdater &updater);
[[nodiscard]] std::vector<PID> qg_updater_search(QGUpdater &updater,
                                                 const float *query,
                                                 size_t k,
                                                 size_t ef,
                                                 size_t max_beam_width,
                                                 float *distances);
[[nodiscard]] uint64_t qg_updater_live_count(const QGUpdater &updater);
[[nodiscard]] uint64_t qg_updater_applied_collection_op_id(const QGUpdater &updater);
[[nodiscard]] uint64_t qg_updater_last_committed_txid(const QGUpdater &updater);
[[nodiscard]] UpdateStats qg_updater_stats(const QGUpdater &updater);

}  // namespace alaya::laser::detail
