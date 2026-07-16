#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# LASER arms for the full-cache adjudication probe. Serializes on the shared
# AIO lock (one aligned_file_reader fleet at a time on this box).
set -euo pipefail

BENCH="$1"        # path to bench_laser_update_sift
DATA="$2"         # data/laser-update
OUT="$3"          # output dir
QUERY="$4"
GT="$5"

mkdir -p "$OUT"
LOCK="$DATA/.aio.lock"

run_arm() {
  local tag="$1" prefix="$2" dram="$3" threads="$4" rep="$5"
  echo "=== arm=$tag threads=$threads rep=$rep dram_gb=$dram ==="
  flock "$LOCK" "$BENCH" \
    eval --prefix "$prefix" \
    --n 1000000 --R 64 --main_dim 128 \
    --query "$QUERY" --gt "$GT" \
    --topk 100 --efs 40,60,100,200 --beam 16 \
    --threads "$threads" --runs 3 --dram_gb "$dram" \
    > "$OUT/laser_${tag}_t${threads}_rep${rep}.csv" 2> "$OUT/laser_${tag}_t${threads}_rep${rep}.log"
}

# Independent-process repetitions (docs §14 red line: in-process --runs alone is not enough)
for rep in 1 2 3; do
  run_arm disk      "$DATA/sift1m"                          0 1 "$rep"
  run_arm cache15   "$DATA/sift1m"                          3 1 "$rep"
  run_arm fullcache "$DATA/fullcache-20260715/sift1m-full"  4 1 "$rep"
done
for rep in 1 2; do
  run_arm disk      "$DATA/sift1m"                          0 16 "$rep"
  run_arm fullcache "$DATA/fullcache-20260715/sift1m-full"  4 16 "$rep"
done

echo "all arms done"
