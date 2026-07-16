#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# E1b probe: kernel section audit (rdtsc, ALAYA_KERNEL_SECTION_PROFILE build)
# on g08 node1. Sections are aligned 1:1 across engines (prep/exact/scan/pool)
# so the remaining same-ef gap can be attributed to a specific section rather
# than guessed from instruction counters. t1 only — the accumulator is a
# plain global.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
BUILD=$WT/build/SecProf
export LD_LIBRARY_PATH="$BUILD/shlib:${LD_LIBRARY_PATH:-}"
FC=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715
OUT=$FC/results/e1b-ksp
LOCAL=/tmp/huangliang-uprobe-gte768
ST=$OUT/status.log
N768=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
BENCH=$BUILD/tests/laser/bench_laser_update_sift
MEMQG=$BUILD/tests/index/bench_memqg_native
mkdir -p "$OUT"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), build=$BUILD"

for pf in 0 20; do
  ALAYA_ARENA_PF_LINES=$pf $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval \
    --prefix "$LOCAL/gte-m768qgfull" --n $N768 --R 32 --main_dim 768 \
    --query "$LOCAL/query.fbin" --gt "$LOCAL/gt.ibin" \
    --topk 10 --efs 100,200 --beam 16 --threads 1 --runs 5 --dram_gb 12 --arena 1 \
    > "$OUT/ksp_laser_gte768_pf$pf.csv" 2>&1 && note "laser gte768 pf$pf ok" || note "FAIL laser gte768 pf$pf"
done
for pf in 0 24; do
  ALAYA_ARENA_PF_LINES=$pf $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval \
    --prefix "$LOCAL/sift1m-r32full" --n 1000000 --R 32 --main_dim 128 \
    --query "$LOCAL/sift_query.fbin" --gt "$LOCAL/sift1m_gt100_exact.ibin" \
    --topk 10 --efs 100,200 --beam 16 --threads 1 --runs 5 --dram_gb 4 --arena 1 \
    > "$OUT/ksp_laser_sift1m_pf$pf.csv" 2>&1 && note "laser sift1m pf$pf ok" || note "FAIL laser sift1m pf$pf"
done

for ef in 100 200; do
  PROFILE_EF=$ef $NUMA "$MEMQG" "$LOCAL/base.fbin" "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 100 7 \
    > "$OUT/ksp_memqg_gte768_ef$ef.csv" 2>&1 && note "memqg gte768 ef$ef ok" || note "FAIL memqg gte768 ef$ef"
done
for ef in 100 200; do
  PROFILE_EF=$ef $NUMA "$MEMQG" "$LOCAL/sift_base.fbin" "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" 100 7 \
    > "$OUT/ksp_memqg_sift1m_ef$ef.csv" 2>&1 && note "memqg sift1m ef$ef ok" || note "FAIL memqg sift1m ef$ef"
done

note "all done"
touch "$OUT/DONE-e1b"
