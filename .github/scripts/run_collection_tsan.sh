#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ARCH=$(uname -m)

if [[ -r /proc/sys/fs/aio-nr && -r /proc/sys/fs/aio-max-nr ]]; then
  echo "aio quota: $(</proc/sys/fs/aio-nr)/$(</proc/sys/fs/aio-max-nr)"
fi

cmake --preset tsan -S "$ROOT"
cmake --build --preset tsan --target \
  segmented_collection_stress_test qg_segment_test

export TSAN_OPTIONS=${TSAN_OPTIONS:-halt_on_error=1:history_size=7}
setarch "$ARCH" -R "$ROOT/build/TSan/tests/collection/segmented_collection_stress_test"
setarch "$ARCH" -R "$ROOT/build/TSan/tests/index/qg_segment_test" \
  --gtest_filter=QgSegmentTest.ConcurrentSearchOnlyIsReentrant
