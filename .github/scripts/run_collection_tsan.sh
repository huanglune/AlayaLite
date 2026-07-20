#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only
# Manual g05 TSan lane for collection stress and maintenance concurrency coverage.
# setarch disables ASLR; the explicit filter excludes only the documented test::Barrier fixture debt.

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCH=$(uname -m)

if [[ -r /proc/sys/fs/aio-nr && -r /proc/sys/fs/aio-max-nr ]]; then
  echo "aio quota: $(</proc/sys/fs/aio-nr)/$(</proc/sys/fs/aio-max-nr)"
fi

cmake --preset tsan -S "$ROOT"
cmake --build --preset tsan --target \
  disk_flat_segment_stress_test \
  collection_facade_stress_test \
  segmented_collection_stress_test \
  collection_maintenance_interleaving_test \
  collection_read_only_test

export TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:history_size=7}
setarch "$ARCH" -R ctest --test-dir "$ROOT/build/TSan" \
  --output-on-failure -L tsan -E '^collection_maintenance_interleaving_test$' -j 1

# Known fixture debt: test::Barrier::arrive_and_wait() double-locks under TSan.
# Keep every maintenance case that does not use that barrier in the lane; the
# four excluded cases are tracked separately from product concurrency defects.
MAINTENANCE_SAFE_FILTER='CollectionMaintenanceInterleaving.*-'
MAINTENANCE_SAFE_FILTER+='CollectionMaintenanceInterleaving.PendingWriteCompletesBeforeQueuedMaintenance'
MAINTENANCE_SAFE_FILTER+=':CollectionMaintenanceInterleaving.MaintenanceBlocksWritesButNotSearch'
MAINTENANCE_SAFE_FILTER+=':CollectionMaintenanceInterleaving.RecoveryLatchWonWhileQueuedMaintenanceRechecks'
MAINTENANCE_SAFE_FILTER+=':CollectionMaintenanceInterleaving.CheckpointAndRotateShareMaintenanceOrder'
setarch "$ARCH" -R "$ROOT/build/TSan/tests/collection/collection_maintenance_interleaving_test" \
  --gtest_filter="$MAINTENANCE_SAFE_FILTER"
