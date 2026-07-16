<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# 拓扑保真 Seal PoC 报告

## 结论

**方向 1 可行：SIFT1M 上，冻结的 Vamana 邻接可无损物化为可由 `DiskANNIndex` 只读打开的磁盘索引；相对独立从零重建，三档 recall 差距最大仅 0.008 pp，QPS 差异在 -3.4%～+1.7%，含 PQ 的 seal 加速 1.67×，去掉共同的 PQ 训练/编码成本后纯拓扑物化加速 12.68×。**

本 PoC 未接入 Collection seal，不修改任何 segment、Vamana/DiskANN 内核或磁盘格式代码。

## 实现

新增 `benchmarks/diskann/topology_preserving_seal_poc.cpp`，目标名为
`topology_preserving_seal_poc`。路径如下：

1. 使用底层 `VamanaBuilder` 构建活动内存图。它已公开只读
   `graph()` 和 `medoid()`，因此不需要新增内核 accessor。
2. 图完成后取得稳定的邻接引用和入口点，作为 frozen snapshot。本 PoC
   没有并发 mutation，freeze 是 O(1) 的稳定视图获取。
3. 直接调用现有 `write_disk_layout(vectors, graph, params)` 写
   `diskann.index`；该 writer 已原生消费现成 graph，并会拒绝超过记录容量的度数。
4. 复用公开 `PQTable` 和 `NodeCache` writer 生成 PQ/cache sidecar；ID 使用
   identity map。
5. benchmark 层补齐 `ids.bin` 和 73-byte `meta.bin` 打包，组成完整只读
   DiskANN 目录，再通过 `DiskANNIndex::load` 打开和搜索。
6. 对照臂调用 `DiskANNIndex::build`，从相同 raw vectors、相同参数独立重建。

`VamanaMemSegment` 当前在 build 中把 builder graph 写到临时文件，再以
`VamanaReader` 形态持有，段对象本身没有邻接导出接口。简报允许使用底层
Vamana kernel，因此本 PoC 直接冻结 `VamanaBuilder` 输出，没有修改
`VamanaMemSegment`。生产接入仍需要一个段级 `FrozenGraphSnapshot` 能力，见后文。

完整 DiskANN 目录目前没有公开 `build_from_graph` 组合入口：低层 layout writer
能消费 graph，但 `DiskANNIndex` 的目录级 API 只有 raw-vector rebuild。本 PoC 的
meta/ID 打包是 benchmark-local glue，不应原样进入生产；它说明缺口位于目录发布
编排，而不是图 writer 或搜索格式。

## 实验设计

### 数据与环境

| 项目 | 值 |
|---|---|
| base | `sift1m_pca_base.fbin`, 1,000,000 × 128 float32 |
| query | `sift_query.fbin`, 10,000 × 128 float32 |
| ground truth | `sift1m_gt100_exact.ibin`, exact L2 top-100 |
| 代码基线 | `5d5bf289453b8629878f8d15c26c143e8f201704` |
| CPU | 2 × AMD EPYC 9554，256 logical CPUs（1 个 offline） |
| 内存 | 1.0 TiB |
| 数据盘 | `/dev/nvme8n1`, ext4 |
| 编译 | Release，GCC 11.4.0，C++20，AVX2/FMA，CMake 3.30.5 |
| 时间 | 2026-07-16，America/Los_Angeles |

所有正式的磁盘构建与搜索在同一个进程中执行，并全程持有：

```text
flock /home/huangliang/workspace/alaya-dev/data/laser-update/.aio.lock ...
```

两个搜索 arm 顺序打开：materialized arm 销毁并释放 reader/AIO 资源后，才打开
rebuilt arm，没有同时占用两套 AIO context。

### 构建参数

| 参数 | 值 |
|---|---:|
| R | 64 |
| build L | 100 |
| alpha | 1.2 |
| build threads | 48 |
| seed | 1234 |
| record capacity | 64 |
| PQ chunks | 32（每向量 32 bytes） |
| PQ train iterations | 15 |
| BFS cache ratio | 1% |

活动图的初次构建是 seal 之前已经支付的 ingestion/build 成本，单列记录，不计入
freeze+materialize。对照臂计时包含 Vamana 建图、layout、ID、PQ、cache 和 meta
全部步骤。

### 搜索参数与统计口径

| 参数 | 值 |
|---|---:|
| query threads | 8 |
| beam width | 4 |
| search L | 100 / 200 / 400 |
| top-k | 100 |
| PQ | enabled |
| exact rerank count | L（对整个保留 frontier 重算 full-precision L2） |
| warmup | 每 arm 100 queries，L=400 |
| repetitions | 每 arm、每 L 3 次，报告中位数 |
| deterministic barrier | off（默认异步性能路径） |

每次 top-100 搜索同时计算：返回前 10 与 exact GT top-10 的交集比例，以及返回前
100 与 exact GT top-100 的交集比例。QPS 的计时只包围
`DiskANNIndex::batch_search`，不包含 recall 计算。

## 结果

### Recall 与 QPS

差值均为 `materialized - rebuilt`；recall 差值单位为百分点（pp）。

| Search L | Materialized recall@10 | Rebuilt recall@10 | Δ pp | Materialized recall@100 | Rebuilt recall@100 | Δ pp | Materialized QPS | Rebuilt QPS | QPS ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 99.462% | 99.465% | -0.003 | 77.7497% | 77.7572% | -0.0075 | 495.74 | 487.55 | 1.017× |
| 200 | 99.836% | 99.839% | -0.003 | 95.8646% | 95.8696% | -0.0050 | 245.66 | 252.30 | 0.974× |
| 400 | 99.908% | 99.907% | +0.001 | 99.6136% | 99.6120% | +0.0016 | 121.73 | 126.00 | 0.966× |

QPS 的三个原始样本如下：

| L | Materialized QPS samples | Rebuilt QPS samples |
|---:|---|---|
| 100 | 505.05 / 495.74 / 484.54 | 482.49 / 504.74 / 487.55 |
| 200 | 245.66 / 239.97 / 248.90 | 252.30 / 256.39 / 246.83 |
| 400 | 123.87 / 121.73 / 121.05 | 125.16 / 126.00 / 126.52 |

最大 recall 差距只有 0.008 pp，没有随 L 放大的系统性损失。QPS 最大差异为
-3.4%，没有数量级或明显性能异常；L=100 的重复波动本身约 4%。L=200/400
materialized 稍慢，最可能来自两个独立并行 build 的细微拓扑/BFS-cache 差异，而非
layout 或量化差异，依据见“拓扑与 artifact 核对”。

### Seal 耗时

主实验为两臂都启用 PQ32 的完全同配置比较。

| 阶段 | Materialized | Rebuilt |
|---|---:|---:|
| 已有 active Vamana graph build（仅参考，不计入 seal） | 11.614 s | — |
| freeze stable view | <0.001 s | — |
| disk layout | 1.130 s | 包含在下列总计 |
| identity IDs | 0.005 s | 包含在下列总计 |
| PQ train + encode + save | 15.863 s | 15.958 s |
| BFS cache | 0.061 s | 0.019 s |
| meta | <0.001 s | 包含在下列总计 |
| 从零 Vamana graph build | — | 11.342 s |
| layout + IDs | — | 1.164 s |
| **freeze + materialize 总计** | **17.060 s** | — |
| **完整从零 rebuild 总计** | — | **28.545 s** |
| **加速比** | **1.673×** | baseline |

PQ train/encode 占 materialization 的 93.0%，因此它形成 Amdahl 上限并掩盖了省掉
建图的主要收益。为隔离纯 full-precision topology/layout 成本，另跑同一 SIFT1M、
同一 R/L/alpha/thread/cache，但 `pq_chunks=0` 的 build-only ablation：

| 配置 | Freeze + materialize | 完整 rebuild | 加速比 |
|---|---:|---:|---:|
| PQ32（主实验） | 17.060 s | 28.545 s | 1.673× |
| No-PQ（成本消融） | 0.995 s | 12.618 s | **12.681×** |

No-PQ 消融说明现有 graph writer 本身已经达到预期的线性布局转换成本；要让 PQ
生产 seal 接近该收益，需要复用 collection/global codebook，或至少把 PQ train 与
encode 分开治理。不能把 12.68× 直接宣称为当前 PQ32 端到端加速，主结论仍是
1.67×。

### 拓扑与 artifact 核对

| 核对项 | Materialized | Rebuilt | 结论 |
|---|---:|---:|---|
| entry point / medoid | 123742 | 123742 | 相同，确定性 centroid medoid |
| ordered-topology digest | `0x9c7e4b736eaa7449` | `0x8139a56109f1c3b1` | rebuilt 是独立并行建图 |
| edge count | 44,552,139 | 44,570,576 | rebuilt +0.041% |
| mean degree | 44.552139 | 44.570576 | 均低于 R=64 |
| zero-degree nodes | 0 | 0 | 无 orphan |
| directory bytes | 867,095,785 | 867,095,785 | 几何与 sidecar 大小相同 |
| PQ pivots digest | `0x43b101ff20a97cec` | `0x43b101ff20a97cec` | 字节一致 |
| PQ codes digest | `0x24811fce1078b8a0` | `0x24811fce1078b8a0` | 字节一致 |

冻结内存 graph 与 materialized layout 的 ordered-topology digest 均为
`0x9c7e4b736eaa7449`，edge count 也完全相同。digest 覆盖 medoid、节点 ID、degree 和
有序 neighbor IDs；物化后的全文件顺序扫描还检查了每个 degree 上限和 neighbor ID
范围。因此没有入口点变化、度数截断或邻接重排。

Materialized 与 rebuilt 的 PQ pivots/codes 字节一致，排除了量化差异。两者唯一可见
的搜索数据面差异是独立 48-thread Vamana build 因 dynamic scheduling 产生的
0.041% edge-count 差异；这足以解释万分位 recall 差和几个百分点以内的 QPS 波动。

## 质量门与复现

新增源码和本报告均带 2026 `AlayaDB.AI` / `AGPL-3.0-only` SPDX 头。只对新增 C++
文件运行了 clang-format，没有执行全树 reformat。

Release 构建目标：

```bash
export PATH="$HOME/.local/bin:$PATH"
cmake --preset release \
  -DPython_EXECUTABLE=/home/huangliang/workspace/alaya-dev/AlayaLite-rabitq-equiv/.venv/bin/python
cmake --build --preset release --target topology_preserving_seal_poc -j 48
```

相关 ctest 共 8 项，全部通过：

```text
test_diskann_layout
test_diskann_pq
test_diskann_node_cache
test_diskann_index
test_diskann_e2e
index_test_vamana_mem_segment
vamana_test_greedy_search
vamana_test_reader
```

正式结果位于源码树之外：

```text
/home/huangliang/workspace/alaya-dev/data/laser-update/topo-seal-poc/
  sift1m-r64-l100-t48-pq32/
    build_metrics.csv
    search_metrics.csv
    materialized/
    rebuilt/
  sift1m-r64-l100-t48-nopq-timing/
    build_metrics.csv
```

主实验命令：

```bash
flock /home/huangliang/workspace/alaya-dev/data/laser-update/.aio.lock \
  build/Release/benchmarks/diskann/topology_preserving_seal_poc \
  --phase all \
  --base /home/huangliang/workspace/alaya-dev/data/laser-update/sift1m_pca_base.fbin \
  --query /home/huangliang/workspace/alaya-dev/data/laser-update/drift/ctrl-data/sift_query.fbin \
  --gt /home/huangliang/workspace/alaya-dev/data/laser-update/fullcache-20260715/sift1m_gt100_exact.ibin \
  --output-dir /home/huangliang/workspace/alaya-dev/data/laser-update/topo-seal-poc/sift1m-r64-l100-t48-pq32 \
  --r 64 --build-l 100 --alpha 1.2 --build-threads 48 \
  --pq-chunks 32 --pq-train-iters 15 --cache-ratio 0.01 --seed 1234 \
  --search-threads 8 --beam-width 4 --search-l 100,200,400 \
  --repetitions 3 --warmup-queries 100 --rebuild
```

## 遗留问题与下一步

1. **定义生产 `FrozenGraphSnapshot`。** 至少包含 vectors、adjacency、entry point、
   internal-to-logical ID map、generation/watermark，并由 freeze token 保证生命周期；
   不应让 Collection 直接依赖 `VamanaBuilder`。
2. **增加目录级 `DiskANNIndex::build_from_graph`/materializer。** 复用现有
   `write_disk_layout`、PQ、cache、meta、ID、原子发布和校验逻辑，删除 PoC 中的
   benchmark-local meta/ID glue，避免格式序列化重复成为维护风险。
3. **治理 PQ Amdahl 瓶颈。** 当前 PQ train+encode 是 seal 的 93%。下一轮应分别计时
   train/encode，并评估 collection 级稳定 codebook、异步 encode 或 seal 时复用已有
   quantization state；任何复用都要保留 codebook/version/checksum 身份。
4. **做同拓扑 A/A QPS 控制。** 将同一 materialized 目录复制为第二 inode、交换 arm
   顺序并重复搜索，可把 0～3.4% 的设备/运行波动与独立 graph 差异进一步拆开。
5. **接入前补并发与恢复语义。** Collection seal 必须执行 rotate-to-successor、关闭旧
   generation 写 admission、drain、取得一致 watermark，再后台物化；staging、fsync、
   manifest publish、orphan cleanup 应复用现有 artifact transaction，而不是 PoC 的
   简单 directory rename。
6. **ID 与删除语义。** 本实验是 dense identity IDs、无 tombstone。生产 snapshot 必须
   固化 ID map 和 live-version/tombstone view，并证明 reader 不会复活旧版本。

PoC 判据已经满足；建议下一阶段先做第 1、2 项的最小 production seam，再决定 PQ
codebook 的生命周期策略，之后才接入 Collection seal。
