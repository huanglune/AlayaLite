#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# A3 probe: hugepage-backed resident arena (E5). Three interleaved arms on
# g08 node1 under the E1c protocol:
#   old       f43692b  4K-page arena, three-pass scan   (ReleaseNative)
#   huge      edf5d3f  2MB MADV_HUGEPAGE arena, three-pass (RelHuge)
#   hugefused 3e887dc  hugepage arena + AVX-512 fused scan (RelHugeFused)
# memqg references (same node, same day): t1 ef40 37158 / ef60 30091 /
# ef100 21114 / ef200 12687; t16 ef100 286338 / ef200 177503.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WTOLD=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
WTNEW=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/fused-scan
SHLIB=$WTOLD/build/ReleaseNative/shlib
BIN_OLD=$WTOLD/build/ReleaseNative/tests/laser/bench_laser_update_sift
BIN_HUGE=$WTOLD/build/RelHuge/tests/laser/bench_laser_update_sift
BIN_HF=$WTNEW/build/RelHugeFused/tests/laser/bench_laser_update_sift
FC=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715
OUT=$FC/results/a3-hugepage
LOCAL=/tmp/huangliang-uprobe-gte768
ST=$OUT/status.log
N768=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
mkdir -p "$OUT"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname)"

run_arm() { # arm bin tag prefix main_dim n query gt dram pf threads rep
  local arm=$1 bin=$2 tag=$3 prefix=$4 md=$5 n=$6 q=$7 gt=$8 dram=$9 pf=${10} thr=${11} rep=${12}
  LD_LIBRARY_PATH="$SHLIB" ALAYA_ARENA_PF_LINES="$pf" $NUMA flock /tmp/.aio-uprobe.lock \
    "$bin" eval \
    --prefix "$prefix" --n "$n" --R 32 --main_dim "$md" --query "$q" --gt "$gt" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$thr" --runs 5 --dram_gb "$dram" --arena 1 \
    > "$OUT/laser_${arm}_${tag}_pf${pf}_t${thr}_rep${rep}.csv" 2>&1 \
    && note "$arm $tag pf$pf t$thr rep$rep ok" || note "FAIL $arm $tag pf$pf t$thr rep$rep"
}

for rep in 1 2 3; do
  for pair in "old:$BIN_OLD" "huge:$BIN_HUGE" "hugefused:$BIN_HF"; do
    arm=${pair%%:*}; bin=${pair#*:}
    run_arm "$arm" "$bin" gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 20 1 "$rep"
    run_arm "$arm" "$bin" sift1m "$LOCAL/sift1m-r32full" 128 1000000 "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" 4 24 1 "$rep"
  done
done

for pair in "old:$BIN_OLD" "huge:$BIN_HUGE" "hugefused:$BIN_HF"; do
  arm=${pair%%:*}; bin=${pair#*:}
  run_arm "$arm" "$bin" gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 20 16 1
done

note "all done"
touch "$OUT/DONE-a3"
