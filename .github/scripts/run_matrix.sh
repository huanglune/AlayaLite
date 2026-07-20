#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
# SPDX-License-Identifier: AGPL-3.0-only

set -u
set -o pipefail
# Do not enable errexit: all requested cases must run so the final table reports every failure.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build_root="${ALAYA_MATRIX_BUILD_ROOT:-${repo_root}/build/matrix}"
jobs="${ALAYA_MATRIX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
requested_case="${1:-all}"
cases=(core-only laser-default portable-no-avx512 portable-no-avx2 python-on)
results=()

case_args() {
  case "$1" in
    core-only) echo "-DALAYA_ENABLE_LASER=OFF -DALAYA_X86_AVX2_BASELINE=ON -DBUILD_PYTHON=OFF -DBUILD_TESTING=ON" ;;
    laser-default) echo "-DALAYA_ENABLE_LASER=ON -DALAYA_X86_AVX2_BASELINE=ON -DBUILD_PYTHON=OFF -DBUILD_TESTING=ON" ;;
    portable-no-avx512) echo "-DALAYA_ENABLE_LASER=OFF -DALAYA_X86_AVX2_BASELINE=ON -DBUILD_PYTHON=OFF -DBUILD_TESTING=ON" ;;
    portable-no-avx2) echo "-DALAYA_ENABLE_LASER=OFF -DALAYA_X86_AVX2_BASELINE=OFF -DBUILD_PYTHON=OFF -DBUILD_TESTING=ON" ;;
    python-on) echo "-DALAYA_ENABLE_LASER=OFF -DALAYA_X86_AVX2_BASELINE=OFF -DBUILD_PYTHON=ON -DBUILD_TESTING=OFF" ;;
    *) return 1 ;;
  esac
}

run_case() {
  local name="$1" dir="${build_root}/$1" log="${build_root}/$1.log" args
  args="$(case_args "$name")" || return 2
  rm -rf "$dir"
  mkdir -p "$dir"
  echo "[$name] configure + build"
  # Matrix arguments are fixed above and intentionally word-split here.
  # shellcheck disable=SC2086
  if ! cmake -S "$repo_root" -B "$dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $args >"$log" 2>&1 \
    || ! cmake --build "$dir" --parallel "$jobs" >>"$log" 2>&1; then
    tail -n 40 "$log"
    return 1
  fi

  if [[ "$name" == "core-only" ]]; then
    if grep -qE 'AIO_(LIBRARY|INCLUDE_DIR):' "$dir/CMakeCache.txt"; then
      echo "[$name] unexpected libaio discovery" | tee -a "$log"
      return 1
    fi
  fi
  if [[ "$name" == "portable-no-avx512" ]] && grep -q -- '-mavx512' "$dir/compile_commands.json"; then
    echo "[$name] AVX-512 leaked into the compile baseline" | tee -a "$log"
    return 1
  fi
  if [[ "$name" == "portable-no-avx2" ]] && grep -qE -- '-mavx2|-mfma' "$dir/compile_commands.json"; then
    echo "[$name] AVX2/FMA leaked into the generic compile baseline" | tee -a "$log"
    return 1
  fi
  case "$name" in
    core-only|portable-no-avx512|portable-no-avx2)
      ctest --test-dir "$dir" --progress --output-on-failure -R 'simd_test_(cpu_features|l2_sqr|ip|fht)' >>"$log" 2>&1 \
        || return 1
      ;;
  esac
}

mkdir -p "$build_root"
if [[ "$requested_case" != "all" ]]; then
  case_args "$requested_case" >/dev/null || { echo "unknown case: $requested_case" >&2; exit 2; }
  cases=("$requested_case")
fi

failed=0
for name in "${cases[@]}"; do
  if run_case "$name"; then
    results+=("$name|PASS")
  else
    results+=("$name|FAIL")
    failed=1
  fi
done

printf '\n%-24s %s\n' CASE RESULT
printf '%-24s %s\n' '------------------------' '------'
for result in "${results[@]}"; do
  printf '%-24s %s\n' "${result%%|*}" "${result##*|}"
done
echo "Logs: $build_root/*.log"
exit "$failed"
