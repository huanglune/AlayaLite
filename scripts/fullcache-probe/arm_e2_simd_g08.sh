#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# E2 probe: fp32 pairwise l2_sqr dispatch policy A/B on g08 node1.
# PR #90 left the exact-distance kernel on AVX2 by default
# (kPreferStableThroughput) with ALAYA_SIMD_DISTANCE_POLICY=avx512 as the
# escape hatch; LASER's own integer kernels already run AVX-512BW. Both
# engines share the same dispatcher, so one env var A/Bs both arms.
# Arms: LASER R32 arena (gte768 m768qg + sift1m r32) and memqg native
# (PROFILE_EF pinned config), policies interleaved rep-major so machine
# drift cancels.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
BUILD=${BENCH_DIR:-$WT/build/ReleaseNative}
export LD_LIBRARY_PATH="$BUILD/shlib:${LD_LIBRARY_PATH:-}"
DATA=/home/huangliang/workspace/alaya-dev/data
FC=$DATA/laser-update/fullcache-20260715
OUT=$FC/results/e2-simd
LOCAL=/tmp/huangliang-uprobe-gte768
ST=$OUT/status.log
N768=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
BENCH=$BUILD/tests/laser/bench_laser_update_sift
MEMQG=$BUILD/tests/index/bench_memqg_native
mkdir -p "$OUT" "$LOCAL"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), build=$BUILD"

if [ ! -x "$BENCH" ] || [ ! -x "$MEMQG" ]; then
  note "FAIL missing binaries"; touch "$OUT/DONE-e2"; exit 1
fi

# stage sift r32 arena files to local disk (aligned reader wants O_DIRECT)
for f in sift1m-r32full_R32_MD128.index sift1m-r32full_R32_MD128.index_cache_ids \
         sift1m-r32full_R32_MD128.index_cache_nodes sift1m-r32full_R32_MD128.index_rotator \
         sift1m_gt100_exact.ibin; do
  [ -s "$LOCAL/$f" ] || cp "$FC/$f" "$LOCAL/$f" || { note "FAIL stage $f"; touch "$OUT/DONE-e2"; exit 1; }
done
[ -s "$LOCAL/sift_query.fbin" ] || cp "$DATA/sift-fbin/sift_query.fbin" "$LOCAL/sift_query.fbin"
[ -s "$LOCAL/sift_base.fbin" ] || cp "$DATA/sift-fbin/sift_base.fbin" "$LOCAL/sift_base.fbin"
note "staged"

run_laser() { # tag prefix main_dim n query gt dram pol rep
  ALAYA_SIMD_DISTANCE_POLICY="$8" $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval \
    --prefix "$2" --n "$4" --R 32 --main_dim "$3" \
    --query "$5" --gt "$6" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads 1 --runs 5 --dram_gb "$7" --arena 1 \
    > "$OUT/laser_$1_$8_rep$9.csv" 2>&1 && note "laser_$1_$8_rep$9 ok" || note "FAIL laser_$1_$8_rep$9"
}

run_memqg() { # tag base query gt pol rep
  PROFILE_EF=200 ALAYA_SIMD_DISTANCE_POLICY="$5" $NUMA "$MEMQG" "$2" "$3" "$4" 100 7 \
    > "$OUT/memqg_$1_$5_rep$6.csv" 2>&1 && note "memqg_$1_$5_rep$6 ok" || note "FAIL memqg_$1_$5_rep$6"
}

for rep in 1 2 3; do
  for pol in stable avx512; do
    run_laser gte768 "$LOCAL/gte-m768qgfull" 768 $N768 "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 12 "$pol" "$rep"
    run_laser sift1m "$LOCAL/sift1m-r32full" 128 1000000 "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" 4 "$pol" "$rep"
  done
done

for rep in 1 2 3; do
  for pol in stable avx512; do
    run_memqg sift1m "$LOCAL/sift_base.fbin" "$LOCAL/sift_query.fbin" "$LOCAL/sift1m_gt100_exact.ibin" "$pol" "$rep"
  done
done

for rep in 1 2; do
  for pol in stable avx512; do
    run_memqg gte768 "$LOCAL/base.fbin" "$LOCAL/query.fbin" "$LOCAL/gt.ibin" "$pol" "$rep"
  done
done

note "all done"
touch "$OUT/DONE-e2"
