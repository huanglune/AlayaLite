#!/usr/bin/env bash
set -euo pipefail

# P3a mixed-load matrix. Required environment:
#   BASE_FBIN, QUERY_FBIN, GT_IBIN, BASE_PREFIX, OUT_DIR
# BASE_PREFIX is the immutable 500k index prefix (before _R64_MD128.index).
# The script reflink-copies it per cell because every mixed run mutates its copy.
# Optional overrides are listed in the defaults below.

required=(BASE_FBIN QUERY_FBIN GT_IBIN BASE_PREFIX OUT_DIR)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "missing required environment variable: $name" >&2
    exit 2
  fi
done

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BENCH=${BENCH:-"$ROOT_DIR/build/Release/benchmarks/laser/bench_laser_update_sift"}
N=${N:-500000}
FROM=${FROM:-500000}
COUNT=${COUNT:-500000}
R=${R:-64}
MAIN_DIM=${MAIN_DIM:-128}
TOPK=${TOPK:-10}
MIXED_EF=${MIXED_EF:-100}
EF_INSERT=${EF_INSERT:-200}
INSERT_THREADS=${INSERT_THREADS:-64}
BATCH=${BATCH:-4096}
MIXED_SECONDS=${MIXED_SECONDS:-5}
TOMBSTONE_FROM=${TOMBSTONE_FROM:-0}
TOMBSTONE_N=${TOMBSTONE_N:-50000}
R_TARGET=${R_TARGET:-56}
GARDEN_FRAC=${GARDEN_FRAC:-0.20}
CSV=${CSV:-"$OUT_DIR/mixed_matrix.csv"}

mkdir -p "$OUT_DIR"
: >"$CSV"

copy_cell_index() {
  local label=$1
  local cell_dir="$OUT_DIR/$label"
  local cell_prefix="$cell_dir/index"
  mkdir -p "$cell_dir"
  shopt -s nullglob
  local artifacts=("${BASE_PREFIX}_R${R}_MD${MAIN_DIM}.index"*)
  if ((${#artifacts[@]} == 0)); then
    echo "no index artifacts match ${BASE_PREFIX}_R${R}_MD${MAIN_DIM}.index*" >&2
    exit 2
  fi
  local src suffix
  for src in "${artifacts[@]}"; do
    suffix=${src#"$BASE_PREFIX"}
    cp --reflink=auto --preserve=mode,timestamps "$src" "${cell_prefix}${suffix}"
  done
  if [[ -f "${BASE_PREFIX}_pca.bin" ]]; then
    cp --reflink=auto --preserve=mode,timestamps \
      "${BASE_PREFIX}_pca.bin" "${cell_prefix}_pca.bin"
  fi
  printf '%s\n' "$cell_prefix"
}

run_cell() {
  local label=$1
  local rate_pct=$2
  local query_threads=$3
  local full_rate=${4:-0}
  local prefix
  prefix=$(copy_cell_index "$label")
  local log="$OUT_DIR/$label/run.log"
  echo "[matrix] $label rate=${rate_pct}% qthreads=$query_threads" | tee "$log"
  "$BENCH" mixed \
    --base "$BASE_FBIN" --query "$QUERY_FBIN" --gt "$GT_IBIN" \
    --prefix "$prefix" --n "$N" --from "$FROM" --count "$COUNT" \
    --R "$R" --main_dim "$MAIN_DIM" --topk "$TOPK" --mixed_ef "$MIXED_EF" \
    --ef_insert "$EF_INSERT" --arm alpha --alpha 1.2 --prune_cap 300 \
    --insert_threads "$INSERT_THREADS" --flush_threads "$INSERT_THREADS" --batch "$BATCH" \
    --query_threads "$query_threads" --mixed_seconds "$MIXED_SECONDS" \
    --mixed_rate_pct "$rate_pct" --mixed_full_rate "$full_rate" \
    --tombstone_from "$TOMBSTONE_FROM" --tombstone_n "$TOMBSTONE_N" \
    --r_target "$R_TARGET" --garden_frac "$GARDEN_FRAC" --ef_maintenance 200 \
    --pump_budget 4 --garden_policy lowdeg --reuse 0 --direct 0 --write_cache 1 \
    --stage 0 --csv "$CSV" 2>&1 | tee -a "$log"
}

# Calibrate the throttled legs from the required unlimited 64W/16Q cell.
run_cell rate100_q16 100 16 0
FULL_RATE=$(awk -F, '$1=="pure_insert" && $2==100 && $3==16 {print $12; exit}' "$CSV")
if [[ -z "$FULL_RATE" || "$FULL_RATE" == "0" ]]; then
  echo "failed to derive unlimited insert QPS from $CSV" >&2
  exit 1
fi
echo "[matrix] unlimited reference insert_qps=$FULL_RATE"

# Fixed 16-query load x insertion rate {0,25,50,100}; 100 is the calibration row.
run_cell rate0_q16 0 16 "$FULL_RATE"
run_cell rate25_q16 25 16 "$FULL_RATE"
run_cell rate50_q16 50 16 "$FULL_RATE"

# Fixed unlimited 64-writer insertion x query concurrency {4,16,32}; q16 already exists.
run_cell rate100_q4 100 4 0
run_cell rate100_q32 100 32 0

echo "[matrix] complete: $CSV"
