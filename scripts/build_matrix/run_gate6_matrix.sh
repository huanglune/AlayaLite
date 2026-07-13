#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

set -u
set -o pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${ALAYA_GATE6_BUILD_ROOT:-${repo_root}/build/gate6-matrix}"
jobs="${ALAYA_GATE6_JOBS:-4}"
python="${ALAYA_GATE6_PYTHON:-${repo_root}/.venv/bin/python}"
requested_lane="${1:-all}"
lanes=(native avx2)
ctest_regex='^(disk_test_(flat_segment|vamana_segment|flat_segment_stress|vamana_segment_stress|segment_factory|collection_factory_dispatch)|test_segment_factory_(vamana|laser)|laser_segment_(test|stress_test)|manifest_v2_test|heterogeneous_segment_integration_test)$'

lane_args() {
  case "$1" in
    native)
      echo "-DALAYA_NATIVE_ARCH=ON -DALAYA_X86_AVX2_BASELINE=OFF -DALAYA_ALLOW_NATIVE_PACKAGE=ON"
      ;;
    avx2)
      echo "-DALAYA_NATIVE_ARCH=OFF -DALAYA_X86_AVX2_BASELINE=ON -DALAYA_ALLOW_NATIVE_PACKAGE=OFF"
      ;;
    *) return 1 ;;
  esac
}

run_lane() {
  local lane="$1" dir="${build_root}/$1" args
  args="$(lane_args "$lane")" || return 2
  rm -rf "$dir"
  mkdir -p "$dir"
  echo "[$lane] configure Release"
  # Matrix arguments are fixed above and intentionally word-split here.
  # shellcheck disable=SC2086
  cmake -S "$repo_root" -B "$dir" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON -DBUILD_PYTHON=ON -DALAYA_ENABLE_LASER=ON $args \
    >"$dir/configure.log" 2>&1 || return 1
  echo "[$lane] full Release build"
  cmake --build "$dir" --parallel "$jobs" >"$dir/build.log" 2>&1 || return 1
  echo "[$lane] Gate 6 CTest subset"
  ctest --test-dir "$dir" --output-on-failure -R "$ctest_regex" \
    >"$dir/ctest.log" 2>&1 || return 1
  echo "[$lane] golden generation twice"
  PYTHONPATH="${repo_root}/python/src" "$python" \
    "$repo_root/scripts/golden/generate_artifact_baseline.py" \
    --build-dir "$dir" --baseline "$dir/artifact-run1.json" --write \
    >"$dir/golden-run1.log" 2>&1 || return 1
  PYTHONPATH="${repo_root}/python/src" "$python" \
    "$repo_root/scripts/golden/generate_artifact_baseline.py" \
    --build-dir "$dir" --baseline "$dir/artifact-run2.json" --write \
    >"$dir/golden-run2.log" 2>&1 || return 1
}

mkdir -p "$build_root"
if [[ "$requested_lane" != "all" ]]; then
  lane_args "$requested_lane" >/dev/null || {
    echo "unknown Gate 6 lane: $requested_lane" >&2
    exit 2
  }
  lanes=("$requested_lane")
fi

failed=0
for lane in "${lanes[@]}"; do
  if run_lane "$lane"; then
    echo "[$lane] PASS"
  else
    echo "[$lane] FAIL (see $build_root/$lane/*.log)" >&2
    failed=1
  fi
done

if [[ "$requested_lane" == "all" && "$failed" == 0 ]]; then
  "$python" "$repo_root/scripts/build_matrix/compare_gate6_artifacts.py" \
    --baseline "$repo_root/tests/golden/artifact-baseline.json" \
    --native "$build_root/native/artifact-run1.json" \
    --native-repeat "$build_root/native/artifact-run2.json" \
    --avx2 "$build_root/avx2/artifact-run1.json" \
    --avx2-repeat "$build_root/avx2/artifact-run2.json" \
    | tee "$build_root/artifact-comparison.log" || failed=1
fi

echo "Gate 6 matrix logs: $build_root"
exit "$failed"
