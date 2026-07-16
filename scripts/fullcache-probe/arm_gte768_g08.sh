#!/bin/bash
# SPDX-FileCopyrightText: 2026 AlayaDB.AI
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# Dimension arm of the arena-parity probe: gte768 (1,006,717 x 768) on g08.
# NUMA node1 only (node0 is in use by others). MemQG-native (FHT pad 1024)
# vs LASER R32 arena at main_dim=448 (PCA split, production config) and
# main_dim=768 (full-dim). DONE marker always written.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

WT=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-fullcache
PYV=/home/huangliang/workspace/alaya-dev/AlayaLite/.claude/worktrees/laser-update-explore/.venv/bin/python
SRC=/home/huangliang/workspace/alaya-dev/data/emb-fbin/gte768
OUT=/home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715/results/scale-dim
LOCAL=/tmp/huangliang-uprobe-gte768
BUILD=${BENCH_DIR:-$WT/build/RN-$(hostname)}
ST=$OUT/status-gte768.log
N=1006717
NUMA="numactl --cpunodebind=1 --membind=1"
mkdir -p "$OUT" "$LOCAL"
note() { echo "[$(date +%H:%M:%S)] $*" >> "$ST"; }
note "start on $(hostname), cores=$(nproc), build=$BUILD"

BENCH=$BUILD/tests/laser/bench_laser_update_sift
MEMQG=$BUILD/tests/index/bench_memqg_native
if [ ! -x "$BENCH" ] || [ ! -x "$MEMQG" ]; then
  note "FAIL no prebuilt binaries at $BUILD (remote home has no toolchain)"
  touch "$OUT/DONE-gte768"; exit 1
fi
note "binaries ok"

# 1) stage data locally
for f in base.fbin query.fbin gt.ibin; do
  [ -f "$LOCAL/$f" ] || cp "$SRC/$f" "$LOCAL/$f"
done
note "data staged"

# 2) MemQG native arm (768 -> FhtKac pad 1024)
$NUMA "$MEMQG" "$LOCAL/base.fbin" "$LOCAL/query.fbin" "$LOCAL/gt.ibin" 100 5 \
  > "$OUT/gte768_memqg_g08.csv" 2>&1 && note "memqg ok" || note "FAIL memqg"

# 3) LASER R32 builds: main448 (PCA split) + main768 (full dim)
build_idx() { # prefix main_dim
  [ -f "$1_R32_MD$2.index" ] && return 0
  $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" build --base "$LOCAL/base.fbin" \
    --prefix "$1" --n $N --R 32 --main_dim "$2" --threads 48 \
    > "$LOCAL/laser_build_md$2.log" 2>&1
}
build_idx "$LOCAL/gte-m448" 448 && note "laser m448 built" || note "FAIL laser m448 build"
build_idx "$LOCAL/gte-m768" 768 && note "laser m768 built" || note "FAIL laser m768 build"

# 4) identity-order full sidecars (node_len: m448=5632, m768=7680; npp=1, page=8192)
if "$PYV" "$WT/scripts/fullcache-probe/gen_sidecar_full.py" \
  "$LOCAL/gte-m448_R32_MD448.index" "$LOCAL/gte-m448full_R32_MD448.index" $N 5632 8192 1 \
  > "$LOCAL/sidecar448.log" 2>&1; then
  cp "$LOCAL/gte-m448_pca.bin" "$LOCAL/gte-m448full_pca.bin"
  note "sidecar448 ok"
else
  note "FAIL sidecar448"
fi
"$PYV" "$WT/scripts/fullcache-probe/gen_sidecar_full.py" \
  "$LOCAL/gte-m768_R32_MD768.index" "$LOCAL/gte-m768full_R32_MD768.index" $N 7680 8192 1 \
  > "$LOCAL/sidecar768.log" 2>&1 && note "sidecar768 ok" || note "FAIL sidecar768"

# 5) LASER arena arms
run() { # name prefix main_dim threads dram
  $NUMA flock /tmp/.aio-uprobe.lock "$BENCH" eval --prefix "$2" --n $N --R 32 --main_dim "$3" \
    --query "$LOCAL/query.fbin" --gt "$LOCAL/gt.ibin" \
    --topk 10 --efs 40,60,100,200 --beam 16 --threads "$4" --runs 5 --dram_gb "$5" --arena 1 \
    > "$OUT/$1.csv" 2>&1 && note "$1 ok" || note "FAIL $1"
}
run gte768_laser_m448_arena_t1_rep1  "$LOCAL/gte-m448full" 448 1 9
run gte768_laser_m448_arena_t1_rep2  "$LOCAL/gte-m448full" 448 1 9
run gte768_laser_m448_arena_t16_rep1 "$LOCAL/gte-m448full" 448 16 9
run gte768_laser_m768_arena_t1_rep1  "$LOCAL/gte-m768full" 768 1 12
run gte768_laser_m768_arena_t1_rep2  "$LOCAL/gte-m768full" 768 1 12

note "all done"
touch "$OUT/DONE-gte768"
