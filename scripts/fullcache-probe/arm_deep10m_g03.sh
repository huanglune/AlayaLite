#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Scale arm of the arena-parity probe: deep10m (10M x 96) on g03.
# Self-contained: builds host-native binaries, stages data on /md1, runs
# MemQG-native vs LASER R32 arena (+ beam full-cache tax arm), copies CSVs
# back to the NFS results dir, touches DONE marker no matter what.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
PYV=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-update-explore/.venv/bin/python
SRC=/home/huangliang/workspace/alaya-dev/data/deep10m-fbin
OUT=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715/results/scale-dim
LOCAL=/md1/huangliang/tmp/uprobe-deep10m
BUILD=${BENCH_DIR:-$WT/build/RN-$(hostname)}
ST=$OUT/status-deep10m.log
mkdir -p "$OUT" "$LOCAL"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), cores=$(nproc), build=$BUILD"

BENCH=$BUILD/benchmarks/laser/bench_laser_update_sift
MEMQG=$BUILD/tests/index/bench_memqg_native
if [ ! -x "$BENCH" ] || [ ! -x "$MEMQG" ]; then
  note "FAIL no prebuilt binaries at $BUILD (remote home has no toolchain)"
  touch "$OUT/DONE-deep10m"; exit 1
fi
note "binaries ok"

# 1) stage data locally
for f in sift_base.fbin sift_query.fbin sift_gt.ibin; do
  [ -f "$LOCAL/$f" ] || cp "$SRC/$f" "$LOCAL/$f"
done
note "data staged"

# 2) MemQG native arm
"$MEMQG" "$LOCAL/sift_base.fbin" "$LOCAL/sift_query.fbin" "$LOCAL/sift_gt.ibin" 100 3 \
  > "$OUT/deep10m_memqg_g03.csv" 2>&1 && note "memqg ok" || note "FAIL memqg"

# 3) LASER R32 build (full main dim 96)
if [ ! -f "$LOCAL/deep10m-r32_R32_MD96.index" ]; then
  flock /tmp/.aio-uprobe.lock "$BENCH" build --base "$LOCAL/sift_base.fbin" \
    --prefix "$LOCAL/deep10m-r32" --n 10000000 --R 32 --main_dim 96 --threads 48 \
    > "$LOCAL/laser_build.log" 2>&1 || { note "FAIL laser build"; touch "$OUT/DONE-deep10m"; exit 1; }
fi
note "laser built"

# 4) identity-order full sidecar for arena
"$PYV" "$WT/scripts/fullcache-probe/gen_sidecar_full.py" \
  "$LOCAL/deep10m-r32_R32_MD96.index" "$LOCAL/deep10m-r32full_R32_MD96.index" \
  10000000 1408 4096 2 > "$LOCAL/sidecar.log" 2>&1 || { note "FAIL sidecar"; touch "$OUT/DONE-deep10m"; exit 1; }
note "sidecar ok"

# 5) LASER arms
run() { # name prefix threads dram arena
  flock /tmp/.aio-uprobe.lock "$BENCH" eval --prefix "$2" --n 10000000 --R 32 --main_dim 96 \
    --query "$LOCAL/sift_query.fbin" --gt "$LOCAL/sift_gt.ibin" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$3" --runs 3 --dram_gb "$4" --arena "$5" \
    > "$OUT/$1.csv" 2>&1 && note "$1 ok" || note "FAIL $1"
}
run deep10m_laser_arena_t1_rep1  "$LOCAL/deep10m-r32full" 1 20 1
run deep10m_laser_arena_t1_rep2  "$LOCAL/deep10m-r32full" 1 20 1
run deep10m_laser_arena_t16_rep1 "$LOCAL/deep10m-r32full" 16 20 1
run deep10m_laser_beam_t1_rep1   "$LOCAL/deep10m-r32"     1 20 0

note "all done"
touch "$OUT/DONE-deep10m"
