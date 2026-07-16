#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# E1 probe: resident-arena candidate-prefetch depth sweep on g08 node1.
# The arena kernel historically issued no software prefetch while the memory
# QG kernel prefetches the next-best candidate after every pool insert; P1
# profiling attributed +19.5% LLC misses to the arena side. The rebuilt bench
# reads ALAYA_ARENA_PF_LINES (0 = legacy control, 10 = memqg parity, larger =
# deeper row coverage; gte768 row = 120 lines, sift row = 24).
# Recall must be bit-identical across PF values (prefetch is semantics-free).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
BUILD=${BENCH_DIR:-$WT/build/ReleaseNative}
export LD_LIBRARY_PATH="$BUILD/shlib:${LD_LIBRARY_PATH:-}"
FC=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715
OUT=$FC/results/e1-prefetch
LOCAL=/tmp/huangliang-uprobe-gte768
ST=$OUT/status.log
N768=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
BENCH=$BUILD/tests/laser/bench_laser_update_sift
mkdir -p "$OUT"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), build=$BUILD, bench_mtime=$(stat -c %y "$BENCH")"

run() { # tag prefix main_dim n query gt dram pf threads rep
  ALAYA_ARENA_PF_LINES="$8" $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval \
    --prefix "$2" --n "$4" --R 32 --main_dim "$3" \
    --query "$5" --gt "$6" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$9" --runs 5 --dram_gb "$7" --arena 1 \
    > "$OUT/laser_$1_pf$8_t$9_rep${10}.csv" 2>&1 && note "laser_$1_pf$8_t$9_rep${10} ok" || note "FAIL laser_$1_pf$8_t$9_rep${10}"
}

for rep in 1 2 3; do
  for pf in 0 10 20 48 120; do
    run gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 "$pf" 1 "$rep"
  done
  for pf in 0 10 24; do
    run sift1m "$LOCAL/sift1m-r32full" 128 1000000 "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" 4 "$pf" 1 "$rep"
  done
done

# 16T arms: does prefetch help or hurt under bandwidth contention?
for pf in 0 10 48 120; do
  run gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 "$pf" 16 1
done

note "all done"
touch "$OUT/DONE-e1"
