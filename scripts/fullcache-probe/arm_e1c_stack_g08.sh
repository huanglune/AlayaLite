#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# E1c probe: QGScanner stack-scratch fix (per-pop heap vectors -> stack
# arrays) re-measured under the E1/E1b protocols on g08 node1.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
RN=$WT/build/ReleaseNative
SP=$WT/build/SecProf
FC=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715
OUT=$FC/results/e1c-stack
LOCAL=/tmp/huangliang-uprobe-gte768
ST=$OUT/status.log
N768=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
mkdir -p "$OUT"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname)"

run_rn() { # tag prefix main_dim n query gt dram pf threads rep
  LD_LIBRARY_PATH="$RN/shlib" ALAYA_ARENA_PF_LINES="$8" $NUMA flock /tmp/.aio-uprobe.lock \
    "$RN/tests/laser/bench_laser_update_sift" eval \
    --prefix "$2" --n "$4" --R 32 --main_dim "$3" --query "$5" --gt "$6" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$9" --runs 5 --dram_gb "$7" --arena 1 \
    > "$OUT/laser_$1_pf$8_t$9_rep${10}.csv" 2>&1 && note "rn $1 pf$8 t$9 rep${10} ok" || note "FAIL rn $1 pf$8 t$9 rep${10}"
}

for rep in 1 2 3; do
  for pf in 0 20; do
    run_rn gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 "$pf" 1 "$rep"
  done
  run_rn sift1m "$LOCAL/sift1m-r32full" 128 1000000 "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" 4 24 1 "$rep"
done
run_rn gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 10 16 1
run_rn gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 20 16 1

for pf in 0 20; do
  LD_LIBRARY_PATH="$SP/shlib" ALAYA_ARENA_PF_LINES=$pf $NUMA flock /tmp/.aio-uprobe.lock \
    "$SP/tests/laser/bench_laser_update_sift" eval \
    --prefix "$LOCAL/gte-m768qgfull" --n $N768 --R 32 --main_dim 768 \
    --query "$LOCAL/query.fbin" --gt "$LOCAL/gt.ibin" \
    --topk 10 --efs 100,200 --beam 16 --threads 1 --runs 5 --dram_gb 12 --arena 1 \
    > "$OUT/ksp_laser_gte768_pf$pf.csv" 2>&1 && note "ksp gte768 pf$pf ok" || note "FAIL ksp gte768 pf$pf"
done

note "all done"
touch "$OUT/DONE-e1c"
