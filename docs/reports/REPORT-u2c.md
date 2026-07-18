<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# U2-c: laser 作为 Collection target 接线 + 驻留选择 + OOM 修复 + sorted_rows 补测 — 执行报告

分支 `feat/u2c-collection-wiring`，基于 `main@c231b7d`。未 push，未动 main，未开 PR。

## 提交清单

| commit | 说明 |
|---|---|
| `7f03ccd` | fix(laser): clamp QGBuilder out-of-core degree and harden vector-file dimension check — W3。`qg_builder.hpp` 三处修复：①`init_from_vamana()` 逐节点 k 夹到 `degree_bound_`（robust_prune top-R 语义）；②`build()` 的向量文件维度校验从 assert 升级为硬 `throw std::invalid_argument`；③header 的 `max_observed_degree` assert 降级为非致命警告（原因见下）。新增回归测试 `tests/laser/qg/test_qg_builder_oom_regression.cpp`（同进程 12 轮连续建索引）。副作用修复：3 个既有 fixture（`test_admission_contract.cpp`、`test_unified_laser_admission.cpp`、`segmented_collection_laser_filter_test.cpp`）补上此前缺失的 `write_fbin()` 调用（见下"OOM 修复前后证据"） |
| `26ac6f9` | feat(collection): wire laser as a Collection target algorithm — W1。`collection.hpp` 的 `validate_options()` 放行 `target_algorithm=laser`（含 rabitq 量化交叉检查的必要扩展，见"清单外判断调用"）；`collection_target_builder.hpp` 新增 `laser_target_support()`/`build_laser_collection_target()`/`open_laser_collection_target()`，把 `kCollectionTargetRegistrations` 里 laser 条目从 unsupported 三件套换成真实实现。新增端到端测试 `tests/collection/collection_laser_target_test.cpp`（create→upsert→seal→plain/filter 搜索→close→reopen→rotate_to_successor→metric=inner_product 回退 flat） |
| `cb77eec` | feat(laser): wire residency selection into LaserSegment::open() — W2。`laser_residency_request()` 从 `segment_factory.hpp`（唯一调用方是死代码）搬到 `laser_segment.hpp`，成为真正的生产决策点。`LaserSegment` 的 `searcher_` 拆成 `legacy_searcher_`/`unified_searcher_` 双指针（恰好一个非空）；`admission_aware_search()` 在 unified 模式下直接走 `UnifiedLaserSegmentSearcher::search()` 自带的 `compile_admission`，legacy 旁路仅 legacy 模式用。`segment_factory.hpp::load_segment_from_manifest`（连同仅服务于它的 `throw_unsupported_engine`）确认零调用方后删除。新增两个测试：`test_unified_laser_admission.cpp` 的 `OpenDirectoryHonorsResidencyEnvOverrideAndBothModesAgree`（`LaserSegment::open` 直连两驻留对拍），`collection_laser_target_test.cpp` 的 `ResidentArenaResidencyViaEnvOverrideThroughCollection`（env 覆盖经 Collection 端到端，校验原生 manifest 落盘 `x_laser_residency=resident_arena`） |
| `69ee008` | test(laser): cover kind=sorted_rows admission on UnifiedLaserSegmentSearcher — W4。`test_unified_laser_admission.cpp` 新增 `SortedRowsFilterMatchesBitmapFilter{ResidentArena,PagedPool}`：同一个 30% 选择率的录取集合，分别用 `kind=sorted_rows`（原始有序 PID 数组）和 `kind=bitmap`（预物化位图）两种编码喂 `search()`/`batch_search()`，断言两条编码路径结果一致 + 每条命中都满足录取集合（`ResultFlag::filtered` 在此层不可表达，见下"清单外判断调用"） |
| `f2bdd85` | style: apply clang-format/cmake-format/end-of-file-fixer after the U2-c series — `uvx pre-commit run --all-files` 首轮自动改写 6 个文件后落盘，纯格式无语义变化 |

## 测试数字

`ctest --test-dir build/Release -LE performance -j8`：**80/80 全绿**（相对 `main@c231b7d` 基线的 78 个 ctest 条目，本波净增 2 个新注册：`laser_test_qg_builder_oom_regression`、`collection_laser_target_test`；W2/W4 均在既有 ctest 条目内部新增 gtest 用例，不增加 ctest 条目数）。全量运行 4 次（每个提交点各一次 + 格式化后一次），稳定无 flaky。

各二进制 gtest 用例数（本波touch过的）：

| 二进制 | 用例数 | 净新增 |
|---|---|---|
| `test_qg_builder_oom_regression`（新） | 1 | +1（内部 12 轮连续建索引） |
| `collection_laser_target_test`（新） | 3 | +3（W1 lifecycle、W2 residency、W1 fallback） |
| `test_unified_laser_admission` | 9 | +3（W2 residency 对拍 ×1、W4 sorted_rows ×2；基线 6 例不变） |
| `test_admission_contract` | 5 | +0（仅修复 fixture bug + 注释，用例数不变） |
| `segmented_collection_laser_filter_test` | 4 | +0（同上） |

`laser_segment_test` 的 `DifferentialRankOnlyManifestGateCollectionRejectionAndPerformance`：**SKIPPED**（`BUILD_PYTHON=OFF` 下 LASER python fixture 未生成，`fixture_available()` 判空后正常跳过，退出码 0），已单独用 `--gtest_filter` 确认。

`uvx pre-commit run --all-files`：**全绿**（`end-of-file-fixer`/`cmake-format`/`clang-format` 首轮自动改写 6 个文件——`segment_factory.hpp` 尾部空行、`qg_builder.hpp`/两个 `CMakeLists.txt` 的注释换行宽度、`collection_target_builder.hpp`/`laser_segment.hpp` 的函数调用参数对齐——二轮确认全绿，纯格式变更已落盘为 `f2bdd85`）。

红线核查：`git diff c231b7d --stat` 过滤 `include/space/**`、`include/simd/**`、`include/index/graph/qg/**`（memqg）、`include/index/graph/hnsw/**`、`tests/space/**`、`tests/include/utils/evaluate.hpp`、`qg_updater.hpp`/`segment_op_wal.hpp`/`mutable_laser_segment.hpp` 结果为空——全程零触碰。

## OOM 修复前后证据

**根因链**（手册已锁定，实测进一步细化）：

1. `QGBuilder::init_from_vamana()` 逐节点 out-degree `k` 从 vamana 文件读取，从不校验是否超过 `degree_bound_`，仅 `assert(max_observed_degree == qg_.degree_bound())`——而 `tests/laser/CMakeLists.txt` 的 `_laser_test_opts` 显式传 `-DNDEBUG`（不依赖整体 Release 构建类型），所以**每一个** LASER 测试目标里这个 assert 都是空操作。读 `include/index/graph/vamana/vamana_writer.hpp::save_graph()` 发现：header 的 `max_observed_degree` 字段**无条件写调用方要求的 R**，从不反映真实数据（注释原话："The writer truncates the header's max_observed_degree to R on output"）——即该字段本就不是可信来源，这也是我把 fix③ 降级为警告而非硬校验的依据。
2. `QGBuilder::build()` 的向量文件维度探测（`int n, d`）同样只有 `assert(d == dimension_+residual_dimension_)`。**关键新发现**：`tests/laser/qg/test_admission_contract.cpp`、`tests/disk/test_unified_laser_admission.cpp`（原 `SegmentFixture`）、`tests/collection/segmented_collection_laser_filter_test.cpp` 三个既有 fixture 全部调用 `QGBuilder::build()` 却从未写入其文档要求的 `"{prefix}_pca_base.fbin"` 文件——`std::ifstream vector_input` 从未成功 open，后续 `int n, d;`（未初始化局部变量）读取到不确定的栈残留值。这才是原始 "同进程建多个索引~2/5概率崩溃" 现象最直接的触发源：`build()` 用这个垃圾 `d` 计算 `vector_tmp_page_size` 来分配 `neighbor_vector_scratch_`，而 `update_qg_out_of_memory()` 用**真实**的 `dimension_+residual_dimension_` 计算 `full_page_size` 做索引偏移——当垃圾 `d` 小于真实维度，两者不一致直接导致固定大小 scratch buffer 越界写。

**复现（fix 前，100% 稳定复现）**：临时还原 `qg_builder.hpp` 到 `c231b7d` 状态 + 保留新建的多轮回归测试文件（不写 `_pca_base.fbin`，完全复刻 3 个原始 fixture 的写法）：

```
=== 8 个独立进程，各内部连续建 12 个索引 ===
exit=139（SIGSEGV）× 8/8
```

gdb 定位（`gdb --batch -ex run -ex "bt full"`）：

```
Thread N received signal SIGSEGV
#0  __memmove_avx512_unaligned_erms ()
#1  alaya::laser::QuantizedGraph::update_qg_out_of_memory(...)
#2  alaya::laser::QGBuilder::build(char const*, char const*) [clone ._omp_fn.1] ()
#3  libgomp.so.1（OpenMP worker 线程）
```

与 U2-b 报告描述（"SIGSEGV inside a memmove ... reached from QuantizedGraph::update_qg_out_of_memory"）**逐字匹配**。

用同样手段单独复现"缺 `_pca_base.fbin`、单次建索引"场景（即现有 `test_admission_contract.cpp` 原样，只建一次）：10/10 进程未崩溃——证实原始三份 fixture 之所以"能用"，纯粹是因为**只建一次**时垃圾栈值恰好落在无害区间的运气，并非真的安全；这也解释了为什么 U2-b 把"进程内只建一次"当规避手段而非巧合来记录。

**修复后（fix②的硬维度检查 + 三个 fixture 补写 `write_fbin` + fix①的 per-node degree clamp 全部落地）**：同一 8 进程 × 12 轮场景，以及额外 5 次独立进程重跑：**72 轮等效连续建索引，0 崩溃**。另外，`collection_laser_target_test.cpp` 的 `CreateUpsertSealSearchFilterReopenAndRotate` 测试本身在 `seal()` + `rotate_to_successor()`（内部 `prepare_successor()`）两步各触发一次真实 Collection 驱动的 LASER 构建，即生产路径下的同进程二次建索引——重复运行 5 次全绿，未见 flaky。

**W3 第 12 条**：三个规避 fixture 的"单次构建模式"结构未改动（仍是每 TestSuite 建一次索引给多个 TEST_F 共用，比逐用例重建快），仅补上各自缺失的 `write_fbin()` 调用并更新过时注释；同进程多次建索引的约束已解除（新回归测试 + `collection_laser_target_test.cpp` 的双建索引路径共同覆盖）。

## 清单外判断调用

以下是手册未逐字规定、由我在实现时裁决或手册假设与代码现实冲突后就地调整的点：

1. **三个既有 fixture 缺失 `write_fbin()` 是独立于手册根因链的第二个 bug，修复它是保持提交①全绿的必要条件**（见上"OOM 修复前后证据"）。发现路径：应用 fix②（维度硬校验）后，立即在 `test_admission_contract.cpp`/`test_unified_laser_admission.cpp` 上复现出 `SetUpTestSuite()` 抛出异常导致整个 suite 失败——不修就无法保持既有测试绿。修复方式与代码库里另一支已正确写法的 fixture（`qg_wal_test_support.hpp`）完全一致（同样的两 int32 header + 数据体 `.fbin` 布局）。

2. **fix③（`max_observed_degree` assert）选择"并入①的运行时逻辑"而非"同升硬校验"**：手册给了两个选项。依据是读 `vamana_writer.hpp::save_graph()` 发现该 header 字段无条件写调用方声明的 R、从不反映真实数据——硬校验它不仅无意义，还可能对本就该被 fix① 兜底的合法文件产生误杀。降级为警告（`clamped_nodes>0` 时汇总打印），保留可观测性但不再是安全屏障。

3. **`Collection::validate_options()` 的 rabitq/algorithm 交叉检查需要一并扩展，超出手册字面指定的单行（757-759）**：手册只要求给 `algorithm_valid` 加 laser，但既有代码另有一对交叉检查（"quantization=rabitq 要求 target_algorithm=qg" 与其逆命题）会让 `target_algorithm=laser + quantization=rabitq`（`laser_target_support()` 要求的确切组合，对齐 `qg_target_support()`）永远在 `Collection::create()` 阶段被拒——不改这对检查，laser 作为 target 根本无法被端到端构建出来。改为对称放行 qg 或 laser，错误文案保留 "explicit index_type=qg" 子串以兼容既有回退测试的断言。

4. **`laser_target_support()` 加了手册未逐字列出的 dim ≥128 且为 2 的幂检查**：手册文字只提metric/scalar/quantization/行数门槛。但 `LaserSegmentImporter` 构造函数硬性要求这两条，不加此检查会让不满足的 schema 在 `laser_target_support()` 判"supported"后，于 `build_laser_collection_target()` 内部才炸成 `core::Status` 错误——违反"不满足→回退flat"这一贯穿全手册的静默回退语义。

5. **`build_laser_collection_target()` 的产物落盘形态镜像 disk-flat（经 `LaserSegmentImporter` 自带的 tmp-dir+rename 事务）而非 qg 的内存态 `ArtifactManifestV2` 事务**：手册明确指向读 `build_qg_collection_target`/`open_qg_collection_target` 理解 artifact 机制，但 LASER 原生格式是磁盘多文件（如 Flat），不是可整体序列化的内存对象（如 QG）——镜像 qg 的是"registration/harvest/错误处理/CollectionTargetBuildResult 填充"这层样板，产物落盘本体照搬 `LaserSegmentImporter::import_from()` 已有的独立事务机制。实测发现该机制不像 `ArtifactControlPlaneTransaction` 那样自动建父目录（`segments/`），补了 `create_directories(seg_dir.parent_path())`。

6. **给拓扑保真 rotate 留缝的具体形态**：手册要求"结构上预留可选 source-graph 入参位"，但 `build_laser_collection_target` 的签名被 `BuildTargetFn` 函数指针类型锁死，不能literal加参数。改为在函数体内放一个带 TODO 注释、当前恒为 `nullptr` 的局部占位（指向 `FrozenGraphSnapshot`/`QGBuilder::build_from_graph()`），作为未来接入点的具体标记而非仅口头注释。

7. **`LaserSegment` 的类型工程选"双指针"而非"小接口"**：手册给了两个选项。选双指针（`legacy_searcher_`/`unified_searcher_`，恰好一个非空 + 三个私有 dispatch accessor）是因为 `UnifiedLaserSegmentSearcher` 已经内部持有一个 `LaserSegmentSearcher legacy_`，走"小接口"需要新引入一个跨两个类的抽象基类，改动面明显更大；双指针只需改 `LaserSegment` 一处。副作用：新 `searcher_labels()` accessor 必须包在 `#if ALAYA_DISK_LASER_SEGMENT_SUPPORTED` 里（不能像 `dim()`/`size()` 那样无条件——`labels()` 不在两个类共享的 `SegmentSearcher` 基类里），这是编译 `laser_segment_header_closure.cpp`/`collection_header_closure.cpp`（不开 LASER consumer flags 的 TU）实测报错后补上的，手册未提及这两个 header-closure 目标的存在。

8. **`segment_factory.hpp` 死代码删除范围包含 `throw_unsupported_engine`**：手册明确点名的是 `load_segment_from_manifest`。穷举调用方时发现 `throw_unsupported_engine` 的两处调用全部在 `load_segment_from_manifest` 内部——删除后者会让前者一并归零调用，一并清理，未单独在手册里出现但满足同样"零调用方"标准。保留了该文件顶部 `#if ALAYA_DISK_SEGMENT_FACTORY_LASER_SUPPORTED` 的 4 个 include（`<cstdlib>`/`<optional>`/两个 searcher 头），虽然此刻本文件自身不再需要，但穷举了 6 个 include 该文件的消费方后判断保守保留风险最低（其中 2 个直接用到 `LaserSegmentSearcher` 的文件已自带直接 include，不依赖此处的传递包含）。

9. **两处新增的驻留/编码对拍测试都用"重叠度下限"而非"逐位相等"**：`OpenDirectoryHonorsResidencyEnvOverrideAndBothModesAgree`（跨 legacy/unified 两实例）预期内会遇到已被 `test_admission_contract.cpp` 等文件文档化的 paged 内核 AIO 完成序非确定性，用重叠计数 >50% 通过（实测 ~93%）。`SortedRowsFilterMatchesBitmapFilterPagedPool` 更意外——最初按"同实例背靠背调用应该确定"的假设写成逐位相等，`batch_search()` 部分实测稳定失败（`search()` 部分反而稳定通过）；根因是 `batch_search()` 在 paged 模式下把 20 个 bitmap 查询全部跑完才开始 20 个 sorted_rows 查询，两次"同一 query"比较点之间的时间/状态跨度远大于逐 query 交替的 `search()` 循环，给 AIO 非确定性更多显现窗口。改为与 `search()` 部分一致的重叠下限（实测稳定 ~95%），5 次独立重跑无 flaky。

10. **W4 的 `ResultFlag::filtered` 断言未实现，且判定其在当前架构下不可行**：`DiskSearchHit`（`UnifiedLaserSegmentSearcher::search()`/`batch_search()` 的返回元素类型）只有 `{label, distance}` 两个字段，没有 result-flags；该标志只在高一层的 `LaserSegment::execute_search()` 构造 `core::SearchHit` 时才附加。而 `LaserSegment::search()` 对 `kind=sorted_rows` 直接返回 `not_supported`（既有回归测试 `LaserSegmentRejectsSortedRowsAndPredicateFilters` 已锁定此行为，`segmented_collection.hpp` 也只发 `kind=bitmap`）——sorted_rows 从未到达会打这个标志的那一层。用"每条命中都满足录取集合 + 与同集合 bitmap 编码结果一致"作为该验收点在这一层能达到的最接近证明，并在测试注释与本节都写明这一 gap，未做进一步改动（扩大 LaserSegment 支持的 filter kind 超出 W4 范围）。

## 遗留边界

- **`sorted_rows` 从未在 `LaserSegment`/Collection 层被真正发出**（`LaserSegmentRejectsSortedRowsAndPredicateFilters` 锁定拒绝行为，`segmented_collection.hpp` 只产出 `kind=bitmap`），W4 的测试只证明了内核层 `compile_admission()` 分支本身正确，不代表这条编码路径已经"可达用户"。如果未来有场景需要不经位图物化直接传有序行 ID（例如极稀疏过滤时避免整段位图分配），`LaserSegment`/`segmented_collection.hpp` 都需要新开一条分支。
- **`build_laser_collection_target()` 每次都从零跑 Vamana + QGBuilder**，即便是 rotate 场景下拓扑本可部分复用；TODO 占位已留（见判断调用 6），未实现。
- **`admission_aware_search()` 的 legacy 旁路互斥量与 `LaserSegmentSearcher::search_mutex_` 仍是两把互不感知的锁**（U2-b 报告已记录此风险，本波未改变其性质，只是在其基础上新增了 unified 分支——unified 分支自身用 `UnifiedLaserSegmentSearcher::search_mutex_`，三把锁互相独立，同一实例混用不同 residency 从架构上不可能发生，故不构成新风险，但 legacy 内部的旧风险原样保留）。
- **`segment_factory.hpp` 顶部为 laser 保留的几个 include 现在locally-unused**（见判断调用 8），未做更激进的 include 精简。

## 关键文件路径

- `include/index/graph/laser/qg/qg_builder.hpp`（W3：三处 assert→真实校验/夹逼）
- `tests/laser/qg/test_qg_builder_oom_regression.cpp`（新，W3 回归测试）
- `tests/laser/qg/test_admission_contract.cpp`、`tests/disk/test_unified_laser_admission.cpp`、`tests/collection/segmented_collection_laser_filter_test.cpp`（W3：补 `write_fbin()` + 注释更新；后两者另有 W1/W2/W4 的新增内容）
- `include/index/collection/collection.hpp`（W1：`validate_options()`）
- `include/index/collection/detail/collection_target_builder.hpp`（W1：`laser_target_support`/`build_laser_collection_target`/`open_laser_collection_target`/注册表）
- `tests/collection/collection_laser_target_test.cpp`（新，W1+W2 端到端测试）
- `include/index/disk/laser_segment.hpp`（W2：`laser_residency_request`、双指针类型工程、`admission_aware_search` 分流）
- `include/index/disk/unified_laser_segment_searcher.hpp`（W2：新增 `labels()`）
- `include/index/disk/segment_factory.hpp`（W2：删除死代码）
- `include/index/disk/laser_segment_importer.hpp`（W2：两处注释更新指向新位置）
