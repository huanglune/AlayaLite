<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# ADR: QG seal topology 与 LASER runtime 的所有权边界

- **状态**：**Accepted（用户判决 2026-07-19：D1-A、D2-A、D3-D、D4-A、D5-A）**
- **日期**：2026-07-19
- **证据基线**：`main@3d89192f`
- 起草时推荐组合为 D1-A、D2-A、D3-D、D4-C、D5-A；**D4 终判改为 A（硬切零 shim）**——C++ 头从未作为产品发布（发布线仅 Python wheel，止于 v1.0.3），零用户/零历史债原则一致适用。**执行修正：本文一切 shim 条款作废**；D4-C 中的价值项（受支持 facade 清单声明、public-header compile-closure 测试、release note 标注）全部保留照做；阶段一不建任何 forwarding header，旧 `qg/` 目录随迁移彻底消失，grep 验收无 shim 豁免。

## Context

在证据基线中，graph include 根下原 canonical `qg/` 子树有 4 个文件：3 个文件构成 Collection seal 时使用的瞬时 QG topology builder，另 1 个是搜索扩展契约。`include/index/graph/laser/` 有 29 个文件，承载活动的 LASER QG builder/search/update、量化、space、residency 与 I/O 工具。两边都出现 “QG”，但不是两个对等算法家族。

现场已把该边界钉实：

- `memory_qg::Builder` 只返回 `FrozenGraphSnapshot`；测试断言它不是 `Searchable`、`BatchSearchable`、`Saveable`、`StatsProvider` 或 `Mutable`（`tests/index/qg_builder_test.cpp:24-28`）。
- Collection 的 IP/cosine seal 路径调用该 builder、必要时截 degree，再把 snapshot 交给 `laser::QGBuilder::build_from_graph`；L2 仍走 Vamana（`include/index/collection/detail/collection_target_builder.hpp:417-445,588-624`）。
- `QgSearchExtension` 的默认 `effort=100` 且 `algorithm_id=qg`（迁移前 `qg_search_extension.hpp:13-27`）。Collection 将它校验、取 `max(100, candidate_limit, user effort)`，再翻译成原生 `LaserSegmentSearchExtension`（`include/index/collection/segmented_collection.hpp:1630-1692`）。后者已由 `include/index/disk/laser_segment.hpp:71-127` 独立定义。因此前者是逻辑 qg/Collection 契约，不是 LASER 内核类型。

### 裁决时迁移面

统计单位是受 Git 跟踪文本中的**路径命中行/文件**；排除被统计目录自身及不可改写的 `docs/reports/**`、`docs/research/**`。分类互斥：`tests/**`（包括 collection tests）统一计入 tests；collection 仅指生产 `include/index/collection/**`。

| 分布 | `qg/` 行/文件 | `laser/` 行/文件 |
| --- | ---: | ---: |
| collection | 2 / 2 | 1 / 1 |
| binding（`python/include/**`） | 1 / 1 | 0 / 0 |
| benchmark（`benchmarks/**`） | 1 / 1 | 1 / 1 |
| tests | 3 / 3 | 82 / 27 |
| 其他生产/Facade `include/**` | 1 / 1 | 14 / 9 |
| design、政策与版权清单 | 1 / 1 | 27 / 7 |
| **合计** | **9 / 9** | **125 / 45** |

证据基线的 headline 与旧提案一致，**没有数字偏差**。但提案称 qg 命中“分布在 collection/binding/benchmark/tests”若被理解为穷尽式描述则不完整：这四类只占 7 行，另有 `qg_naming.hpp` 和 `docs/design/LASER.md` 各 1 行。另一个必要口径澄清是：只数真实预处理 include 时，qg 为 **8 行/8 文件**，LASER 为 **90 行/34 文件**；125/45 不能被解释成编译依赖边，更不能单独证明公开 API。

复现命令：

```sh
git rev-parse --short=8 HEAD
for d in qg laser; do
  printf '%s lines=' "$d"
  git grep -n "index/graph/$d/" -- \
    ":!include/index/graph/$d/**" ':!docs/reports/**' ':!docs/research/**' | wc -l
  printf '%s files=' "$d"
  git grep -l "index/graph/$d/" -- \
    ":!include/index/graph/$d/**" ':!docs/reports/**' ':!docs/research/**' | wc -l
done

legacy_qg_path='index/graph/'qg'/'
git grep -n "$legacy_qg_path" -- \
  ':!include/'"$legacy_qg_path"'**' ':!docs/reports/**' ':!docs/research/**' | sort

for d in qg laser; do
  for mode in -n -l; do
    printf '%s %s\n' "$d" "$mode"
    git grep "$mode" "index/graph/$d/" -- \
      ":!include/index/graph/$d/**" ':!docs/reports/**' ':!docs/research/**' |
    awk -F: '
      function bucket(p) {
        if (p ~ /^include\/index\/collection\//) return "collection"
        if (p ~ /^python\/include\//) return "binding"
        if (p ~ /^benchmarks\//) return "benchmark"
        if (p ~ /^tests\//) return "tests"
        if (p ~ /^include\//) return "other-production"
        return "docs-policy-metadata"
      }
      { print bucket($1) }
    ' | sort | uniq -c
  done
done

for d in qg laser; do
  git grep -n -E '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]index/graph/'"$d"'/' -- \
    ":!include/index/graph/$d/**" ':!docs/reports/**' ':!docs/research/**' | wc -l
  git grep -l -E '^[[:space:]]*#[[:space:]]*include[[:space:]]*[<"]index/graph/'"$d"'/' -- \
    ":!include/index/graph/$d/**" ':!docs/reports/**' ':!docs/research/**' | wc -l
done
```

分布表由同一输出按上述六个互斥路径前缀分组；没有把文档命中重新解释为 include。

## Decision

### D1. 按生命周期建立三个所有权边界

| 选项 | 定义 | 代价 |
| --- | --- | --- |
| **D1-A 生命周期三边界（推荐）** | Collection policy/seal、瞬时 topology build、LASER persisted/runtime 三层分开，以中立 `FrozenGraphSnapshot` 交接 | 需要维护明确的依赖规则；同一逻辑 qg 会跨多个目录 |
| D1-B 按算法名分成 qg 与 LASER | 名为 QG 的代码进 `qg/`，名为 LASER 的代码进 `laser/` | 继续混淆逻辑产品、构图 scratch 与物理 runtime，无法指导新代码 |
| D1-C topology 并入 LASER | 把当前 memqg builder 当作 LASER builder 内部件 | 令独立拓扑生产者依赖可选 runtime，弱化 snapshot 中立边界，也掩盖 L2/Vamana 与 IP/cosine 的不同来源 |

**推荐 D1-A。** 三个边界如下；归属判据看对象存活到哪个生命周期，而不看类型名是否含 QG、LASER 或 RaBitQ。

| Owner | 职责与新代码归属判据 | 允许依赖 | 禁止反向依赖 |
| --- | --- | --- | --- |
| `collection/seal` | 逻辑 `qg` 产品契约、平台门、metric 预处理、seal/manifest/publication 编排、搜索扩展校验与 qg→LASER 翻译。若代码决定“Collection 要构建/公开什么”或协调多个物理实现，归这里 | 可依赖 topology facade、LASER/disk facade 与中立 core/graph types | topology 与 LASER 内核不得 include Collection internals |
| `qg topology`（构建面） | seal 期间的 QG 邻接生成、prune/beam/visited scratch、输入校验；唯一产物是已冻结拓扑，scratch 在 build 后死亡 | 只依赖 core、space、graph 构建辅助件（当前历史路径为 `detail/search_runtime`）和中立 `FrozenGraphSnapshot` | 不得依赖 Collection、disk segment 或 `laser/**`；不得新增 search/open/save/mutate/artifact 面 |
| `laser runtime`（可写实现面） | 原生 QG pack/build/open/search/update、量化、artifact layout、residency、WAL/IO 工具；若代码随 artifact reopen、query 或 mutation 存活，归这里 | 可依赖 core/platform/simd/storage/space 与中立 snapshot | 不得依赖 Collection 或 topology 实现；通过 snapshot/明确 adapter 接收拓扑 |

跨边界通用对象必须上提为中立类型；`FrozenGraphSnapshot` 保持在 `include/index/graph/`，不归任一实现树。

### D2. 将 canonical 顶层 `qg/` 政名

| 选项 | 收益 | 代价 |
| --- | --- | --- |
| **D2-A `include/index/graph/seal_topology/`（推荐）** | 直接表达“seal 期、topology-only”，最能阻止 serving/runtime 新代码误入 | 路径较长，并把目录绑定到当前唯一生产消费者；未来非 seal 消费须重新裁决 |
| D2-B `include/index/graph/qg_topology/` | 保留 QG 算法身份，同时消除 serving 暗示；认知迁移最小 | 仍以算法名为第一线索，和 `laser/qg/` 的双 QG 视觉冲突未完全消失 |
| D2-C `include/index/graph/topology_build/` | 强调 build phase，未来可容纳其他 topology producer | 过于宽泛，容易把 Vamana 或通用 build helpers 汇成新杂物间，也隐藏当前 QG/RaBitQ 约束 |

**推荐 D2-A。** 只改 canonical 路径，不改 `memory_qg` namespace、类名或算法。三个 topology 文件进入 `seal_topology/`；search extension 由 D3 独立决定。

以下是当前 9 个外部点的逐行迁移面。令 A/B/C 分别为 include 拼写 `index/graph/{seal_topology,qg_topology,topology_build}`（磁盘路径再加前缀 `include/`），E 为 D3 推荐的 `index/collection/qg_search_extension.hpp`；表中是 canonical replacement，兼容 shim 由 D4 决定。

| 当前 `file:line` | 当前目标 | D2-A | D2-B | D2-C |
| --- | --- | --- | --- | --- |
| `benchmarks/parity_lanes_benchmark.cpp:48` | search extension | E | E | E |
| `docs/design/LASER.md:329` | qg tree 文字 | `include/index/graph/seal_topology/` | `include/index/graph/qg_topology/` | `include/index/graph/topology_build/` |
| `include/index/collection/detail/collection_target_builder.hpp:32` | builder | `A/qg_builder.hpp` | `B/qg_builder.hpp` | `C/qg_builder.hpp` |
| `include/index/collection/segmented_collection.hpp:34` | search extension | E | E | E |
| `include/index/graph/qg_naming.hpp:7` | builder facade | `A/qg_builder.hpp` | `B/qg_builder.hpp` | `C/qg_builder.hpp` |
| `python/include/collection_binding.hpp:30` | search extension | E | E | E |
| `tests/collection/collection_qg_recall_floor_test.cpp:30` | search extension | E | E | E |
| `tests/disk/rabitq_format_separation_test.cpp:15` | builder | `A/qg_builder.hpp` | `B/qg_builder.hpp` | `C/qg_builder.hpp` |
| `tests/index/qg_builder_test.cpp:14` | builder | `A/qg_builder.hpp` | `B/qg_builder.hpp` | `C/qg_builder.hpp` |

此外有 2 个被迁目录内部 include（当前 `qg_builder.hpp:18-19`）必须随候选根更新，但不计入“外部 9 点”。若 D3 不选推荐项，只替换表中 4 个 E 行，不改变 D2 的独立裁决。

### D3. `qg_search_extension.hpp` 的 owner

| 选项 | 后果 | 代价 |
| --- | --- | --- |
| D3-A 留在旧 `qg/` | 4 个消费者零迁移，旧 include 最稳 | 旧 `qg/` 变成 extension-only 孤岛，canonical 改名不完整，继续把 search contract 与 topology 混为一层 |
| D3-B 移到 `laser/qg/` | 靠近最终执行 effort 的物理 QG | 将 `algorithm_id=qg` 的逻辑契约放入可选物理实现；与已有原生 `LaserSegmentSearchExtension` 重叠，flat-only 构建也会被路径名误导 |
| D3-C 移到 D2 新目录 | 一次搬完，所有旧 qg include 统一改根 | 搜索契约在 seal 之后仍存活，放进 topology-only 目录违反生命周期判据 |
| **D3-D 移到 `include/index/collection/qg_search_extension.hpp`（推荐）** | 由校验并翻译它的逻辑产品层持有；LASER 继续只持有 native extension | 新增一个明确的 Collection public header；4 个消费者需改 include，并需旧路径 shim |

**推荐 D3-D。** ⑦终报的“逐字保留”是硬约束：canonical 文件只能做纯 `git mv`，不得改 SPDX、include、namespace、字段、默认值、reserved、factory 或格式。当前 31 行文件 SHA-256 为 `ee699ce98ecf8e3e78e5c863b9e5b7f84dcb3b931b668facb7f6849155cb812c`，且与初始隔离提交 `f539e80` 相同；迁后必须仍相同。旧路径的兼容 shim 是另一个新文件，不算修改被移动正文。

### D4. 公开 include 与兼容声明

| 选项 | 后果 | 代价 |
| --- | --- | --- |
| D4-A 所有路径硬切、无 shim | 最快清零旧路径 | 无法观测仓外 C++ 消费者，可能静默破坏 source include |
| D4-B 把整个 `qg/**`、`laser/**` 永久视为 public | 最大兼容性 | 冻结 29 个 LASER 实现头及其内部拆分，今后几乎不能整理 |
| **D4-C 显式 facade + 有界 shim（推荐）** | 只承诺文档列出的 facade；旧 qg 非-detail 头过渡；LASER 本体本 ADR 冻结不搬 | 暂时保留一个只含 shim 的旧 `qg/` 目录，并增加 compile-closure 测试与 release note |

**裁决：125 行/45 文件是事实上的仓内迁移面，不构成事实公开面。** 其中 90 行/34 文件才是真 include，且大部分位于 tests；本仓 grep 既不能证明仓外消费者存在，也不能证明其为零。CMake 的 INTERFACE target 暴露整个 `include/` 根，使这些头“技术上可包含”，但不自动形成支持承诺。

建议入库声明原文：

> 受支持的 C++ include 路径仅限项目文档明确列出的 facade。包含 `/detail/` 的路径属于内部实现，可随时变更，不承诺 forwarding header。`index/graph/laser/**` 当前是 implementation-facing 路径；仓内引用或可包含性本身不构成稳定性承诺。本 ADR 的两个阶段冻结该树，不移动、不改名。未来移动任何非-detail LASER 头必须另立 ADR，并作为明确的 breaking 变更在 release note 记录。

**（D4-A 终判）阶段一不建任何 forwarding shim**：全部消费者一次性改到新 canonical 路径，graph include 根下原 canonical `qg/` 目录随迁移删除；路径变更在 CHANGELOG/release note 记录一行。

### D5. 执行切片

| 选项 | 后果 | 代价 |
| --- | --- | --- |
| **D5-A 两阶段独立落地（推荐）** | 第一阶段只做 qg canonical 改名与 extension 归属；第二阶段只整理 facade include | 两次完整验收；过渡期存在 canonical + shim 双路径 |
| D5-B 一次搬 qg 与 LASER | 表面上一次完成目录整齐 | 把 125 行路径面、29 个活动实现头、仓外未知消费者和语义变更绑成不可审计大爆炸 |
| D5-C 只写边界、不改路径 | 零迁移风险 | 目录继续给新代码错误暗示，ADR 无法形成可执行约束 |

**推荐 D5-A，并将“禁止一次性搬动、改名或 flatten `include/index/graph/laser/**`”写为本 ADR 的强制决议。**

## Consequences

- 收益：逻辑 qg 产品、瞬时 topology 与持久/可写 LASER runtime 各有唯一 owner；`FrozenGraphSnapshot` 成为清晰的中立接缝。
- 收益：extension 的 `effort=100`、versioned layout 与 qg→LASER 翻译保持不变，位置变化不夹带语义改写。
- 收益：`laser/**` 的 125 行仓内迁移风险和未知仓外风险均不进入本波。
- 代价（D4-A 终判后）：仓外若存在未知 C++ include 消费者将在升级时硬断——按零用户/零历史债原则显式接受，仅以 CHANGELOG/release note 记录。
- 代价：`memory_qg` namespace、`qg_builder.hpp` 文件名等历史命名暂留；重命名符号是另一项 API 决策，不得搭车。
- 非目标：不改算法、recall、artifact/wire format、platform gate、LASER mutable 生命周期、Vamana L2 路径或 `detail/search_runtime` 的另波改名。

## 执行合同

### 阶段一：canonical qg 改名 + extension owner

1. 单一、可回滚提交完成：3 个 topology 文件 `git mv` 到 `seal_topology/`（D2-A），只修复其 include 路径；extension 纯移动到 `include/index/collection/qg_search_extension.hpp`（D3-D）；更新上表 9 点与 2 个内部 include；**不建任何 shim（D4-A），旧 `qg/` 目录删净**。不得修改现有 29 个 `laser/**` 文件，不与 tests/docs/benchmark 其他结构调整合并。
2. 结构验收：除历史 `docs/reports/**`、`docs/research/**` 外，上述 `legacy_qg_path` 的 `git grep` 为零；graph include 根下原 canonical `qg/` 目录不复存在；`git diff --name-only -- include/index/graph/laser` 无输出。
3. 契约验收：迁后 extension SHA-256 必须仍为上述值；builder 的五个负 concept static_assert、IP/cosine snapshot→LASER bridge、effort floor/translation 测试必须通过。
4. 构建验收：Release 全 target build、全部 CTest、Python 全测、pre-commit、`git diff --check`；另跑 `ALAYA_ENABLE_LASER=OFF` 的完整 configure/build/CTest，证明 Collection-owned extension 与 topology facade 不误依赖 LASER。
5. 回滚：该提交可整体 revert 回原路径；失败时不保留半迁 shim 或双 canonical 文件。

### 阶段二：只整理 facade include

1. 独立提交显式让 `include/alaya/collection.hpp` 导出 Collection-owned qg search extension，更新/补充 public-header compile closure 与兼容声明；仓内只在消费“公开逻辑契约”时切 facade，测试 LASER 内核的文件继续直接 include 内核头。
2. **禁止移动、改名、拆分或 flatten `include/index/graph/laser/**`；禁止借 facade 整理改 LASER 类型、实现、namespace 或 artifact。** 该目录在阶段二的允许 diff 为零。
3. 重跑与阶段一相同的 Release、CTest、Python、no-LASER、pre-commit 和 diff 门禁。阶段二可单独 revert 而不撤销阶段一；若两阶段均已落地，回滚按二→一逆序执行。

两阶段必须各自从干净工作树开始、各自绿后才进入下一阶段；不得与 `search_runtime/`、tests、benchmark、docs history 或其他目录重排合并。
