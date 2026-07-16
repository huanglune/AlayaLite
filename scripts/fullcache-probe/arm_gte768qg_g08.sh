#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Topology-seal probe arm: LASER m768 index packed from the *memory-QG*
# topology (bench_memqg_native dump -> bench build --reuse_graph, packed on
# the build host), evaluated on g08 node1 with the same protocol as the
# Vamana-topology m768 arm. Index files are staged into $LOCAL beforehand.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
PYV=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-update-explore/.venv/bin/python
OUT=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715/results/scale-dim
LOCAL=/tmp/huangliang-uprobe-gte768
BUILD=${BENCH_DIR:?set BENCH_DIR to the prebuilt build dir}
export LD_LIBRARY_PATH="$BUILD/shlib:${LD_LIBRARY_PATH:-}"
ST=$OUT/status-gte768qg.log
N=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), build=$BUILD"

BENCH=$BUILD/tests/laser/bench_laser_update_sift
if [ ! -x "$BENCH" ] || [ ! -s "$LOCAL/gte-m768qg_R32_MD768.index" ]; then
  note "FAIL missing bench or staged index"
  touch "$OUT/DONE-gte768qg"; exit 1
fi

"$PYV" "$WT/scripts/fullcache-probe/gen_sidecar_full.py" \
  "$LOCAL/gte-m768qg_R32_MD768.index" "$LOCAL/gte-m768qgfull_R32_MD768.index" $N 7680 8192 1 \
  > "$LOCAL/sidecar768qg.log" 2>&1 && note "sidecar ok" || { note "FAIL sidecar"; touch "$OUT/DONE-gte768qg"; exit 1; }

run() { # name threads
  $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval --prefix "$LOCAL/gte-m768qgfull" --n $N --R 32 --main_dim 768 \
    --query "$LOCAL/query.fbin" --gt "$LOCAL/gt.ibin" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$2" --runs 5 --dram_gb 12 --arena 1 \
    > "$OUT/$1.csv" 2>&1 && note "$1 ok" || note "FAIL $1"
}
run gte768_laser_m768qg_arena_t1_rep1  1
run gte768_laser_m768qg_arena_t1_rep2  1
run gte768_laser_m768qg_arena_t16_rep1 16

note "all done"
touch "$OUT/DONE-gte768qg"
