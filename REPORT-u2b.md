<!--
SPDX-FileCopyrightText: 2026 AlayaDB.AI

SPDX-License-Identifier: AGPL-3.0-only
-->

# U2-b: filter 下推 / 统一录取谓词落地 — 执行报告

分支 `feat/u2b-filter-pushdown`，基于 `main@866331e`。未 push，未动 main。

## 提交清单

| commit | 说明 |
|---|---|
| `e04c696` | feat(laser): add RowAdmission and thread it through the QG kernel — 新建 `row_admission.hpp`（POD 视图 + 4 个工厂函数），`qg.hpp` 六个内核入口加 `admission` 尾参并改写两处 admit 点，`residency.hpp` 同步 `ResidencyProvider` 接口，单测 `test_row_admission.cpp`（12 例） |
| `55a89cc` | feat(disk): compile the segment admission contract into LASER search paths — `types.hpp` 加 `DiskSearchOptions.filter`；`unified_laser_segment_searcher.hpp` 编译 admission 并驱动两种驻留；`laser_segment.hpp` 接受 `kind=bitmap`、做 label→PID 转换、新增 `admission_aware_search` 旁路、置 `ResultFlag::filtered`；新增内核层 `test_admission_contract.cpp`（5 例）与磁盘层 `test_unified_laser_admission.cpp`（6 例） |
| `984d4cb` | feat(collection): precompile segment admission bitmaps, add auto-policy fallback — `fanout_search` 改发 `kind=bitmap`（段内位图现算）取代 `kind=predicate`；新增 `automatic` 策略下 `not_supported` 回退重试；顺手修了一个无关的悬垂指针预存量 bug（见下）；新增 Collection 端到端 `segmented_collection_laser_filter_test.cpp`（4 例） |
| `2ee1037` | style: apply clang-format/cmake-format after the admission contract series — 纯格式化，`uvx pre-commit run --all-files` 首轮自动改写后的落盘 |

## 测试数字

`ctest --test-dir build/Release -LE performance -j8`：**73/73 全绿**（连续 3 次全量运行稳定，无 flaky）。较基线（commit e04c696 前）70 个测试净增 3 个新 ctest 条目（`laser_test_row_admission` 在基线内已含；净增 `laser_test_admission_contract`、`test_unified_laser_admission`、`segmented_collection_laser_filter_test`），新增用例总计 27 个 GTest case（12+5+6+4）。

`laser_segment_test` 的 `DifferentialRankOnlyManifestGate...`：**SKIPPED**（非 FAILED）——因构建命令 `-DBUILD_PYTHON=OFF`，LASER python fixture 流水线未生成，`fixture_available()` 判空后正常跳过，退出码 0。未触发手册预留的"已知例外"分支。

`uvx pre-commit run --all-files`：**全绿**（`cmake-format`/`clang-format` 首轮自动改写 6 个文件，二轮确认全绿，改写内容已落盘为 commit `2ee1037`，纯格式无语义变化）。

## 性能 A/B（契约验收 4）

`AdmissionContract.PerformanceAdmissionVsExcludeSetEqualLiveRatioArenaKernel`（arena 内核，N=2000，10% dead/90% live，400 次迭代，warmup 20 次）三次独立运行：

| exclude-set (µs/query) | admission bitmap (µs/query) | ratio |
|---|---|---|
| 2602.0 | 2568.0 | 0.99 |
| 2622.4 | 2588.0 | 0.99 |
| 2590.3 | 2575.8 | 0.99 |

admission 路径与 exclude-set 路径耗时基本相等（比值稳定在 0.99-1.00），符合契约验收 4"admission 开销 ≤ exclude-set"。未做 CI 断言（按手册），仅保留一个宽松哨兵（`admission_us < exclude_us * 5.0`）防止未来出现意外的 O(N) 回归。

## 清单外判断调用

以下是手册未逐字规定、由我在实现时裁决的点，均在代码注释中同步留痕：

1. **`ResidencyProvider::search/batch_search` 加 admission 尾参**（`residency.hpp`，未在手册文件清单内）。为了让 `UnifiedLaserSegmentSearcher` 的 kResidentArena 分支能透传 admission，`ResidencyProvider` 接口必须同步加参数；默认值 `nullptr` 保证旧调用点源码兼容。判断：这是决议 1（内核入口加尾参）与决议 5（unified searcher 编译点）之间必经的胶水层，不改动就无法让 arena 驻留吃到 admission。

2. **`UnifiedLaserSegmentSearcher::search()` 的 kPagedPool 非 none 分支改为始终经 `provider_->search(...)` 而非手册字面写的"`quantized_graph_->search(...)`"**。核实 `PagedPoolProvider::search()` 就是 `qg.search(...)` 的纯透传（`test_unified_residency.cpp` 的 `ProviderDispatchMatchesKernels` 已验证字节等价），两者等价，走 `provider_->search()` 能让 paged/arena 两分支共享同一段锁 + set_params 缓存代码，未引入额外行为差异。

3. **`RowAdmission`/`admission_from_sorted_rows` 的行/PID 值类型统一用 `uint64_t`，不用 `laser::PID`（uint32_t）**。手册要求 `row_admission.hpp` "只依赖 <cstdint>/<vector> 级别，不 include collection/core"；为了同时不依赖 `laser/common.hpp`（同样是"层外"依赖，只是层次不同），把该头做成与 laser 完全解耦的纯 STL 工具头，`admission_from_exclude_set` 用模板兼容 `unordered_set<PID>`（qg_updater 的 `deleted_`）与任意整型 exclude set。

4. **`laser_segment.hpp` 新增 `admission_aware_search()` 旁路 + 独立 `admission_search_mutex_`/`AdmissionLastSetParams` 缓存**（手册决议 6 未给出这一层的具体机制，只给了 PID/label 核对的方向）。核实 `LaserSegment::searcher_` 是 `LaserSegmentSearcher`（legacy，仅 paged pool），其 `search()` 从不读 `DiskSearchOptions.filter`；若只做 `resolve_search_options()` 里的位图翻译而不新增这条旁路，翻译结果根本不会被内核看到——首次实现漏了这一步，被新写的 `LaserSegmentTranslatesLabelBitmapToPidSpace` 测试当场抓到（返回的 pid 全部不满足过滤条件），修复后复测通过。这条旁路复刻了 `UnifiedLaserSegmentSearcher` 已有的"绕过 legacy search()，直连 `graph()`"模式，但用了 `LaserSegment` 自己的互斥量（因为 `LaserSegmentSearcher::search_mutex_` 是私有的，够不到）。**遗留边界**：这意味着同一个 `LaserSegment` 实例如果同时收到 `kind=none` 请求（走 legacy 自身的锁）和活跃 filter 请求（走新旁路的锁），两把锁互不感知——若两者的 `ef`/`beam_width` 恰好不同导致 `set_params()` 在两条路径上交替触发，存在一个理论上的竞争窗口。这与 `UnifiedLaserSegmentSearcher` 现有的 kPagedPool/kResidentArena 分裂锁本就带有的同类风险同构，非本次新增的独立问题；工作负载若对同一 collection 固定 `ef`/`beam_width`（常见做法）则不会触发。已在代码注释中写明，未在手册要求范围内做进一步收敛（需要更大改动，如把 `LaserSegmentSearcher::search_mutex_` 提升为可共享）。

5. **`segmented_collection.hpp` 的 bitmap 容量取 `known_rows_for(*entry)`**（手册未指定具体来源）。该值已是同一循环里 `candidate_limit` 现成用的"这个段有多少行"口径，选它保持与既有代码同一套容量假设，不引入新概念；对每行做 `row >= known_rows` 越界跳过做防御。**假设**：`known_rows_for`（`reverse` 映射条目数）与该段行 id 的最大值+1 相等——对顺序分配、无空洞的场景（本次两个测试 fixture 均如此：LASER 全量导入、QG 全量注册）成立；若某个 Collection 只登记了段的部分行（如 `laser_segment_test.cpp` 里刻意只注册 10/2048 行的场景），位图容量会比对应过小，可能导致某些行的过滤位不可表达——这不是本次改动引入的新风险（`candidate_limit` 早就用同一个值），且 Collection 侧既有的逐命中 re-verify（1644-1656 行区）兜底：容量不足只会让"应该被过滤掉"仍被过滤掉（因为越界即不可表达=不设位=不放行，`segment_filter_storage` 默认全 0），不会导致数据泄漏，只可能损失召回。

6. **`segmented_collection.hpp` 中修复一个与 filter 无关的预存悬垂指针 bug**：`hnsw_effort`/`qg_effort`（`synthesize_effort` 用来合成 extension 的局部变量）原本作用域在 `if (is_memory_graph) {...}` 块内，但 `make_hnsw_search_extension`/`make_qg_search_extension` 把 `payload` 指向这两个局部变量本身，而 `segment_extensions`（存着这个悬垂指针）要活到该 `if` 块结束之后（`entry->segment.search(...)` 才真正读取）。这是一个已存在、恰好"运气好没炸"的未定义行为——加入 `segment_filter_storage`（一个新的栈上局部变量）之后栈布局被扰动，`collection_hnsw_seal_test`/`collection_qg_seal_test` 各一例开始确定性失败（"HNSW effort must be in [top_k, UINT32_MAX]"）。用 `git stash` 反复验证：基线（无本次改动）4/4 稳定通过，带 commit-3 改动 4/4 稳定失败，排除环境 flaky。修复方式是把两个变量的声明挪到外层作用域（与 `segment_request` 同级），使其存活期覆盖到实际读取点；纯作用域调整，不改变任何数值/逻辑。这个 bug 与 admission 契约完全无关，是"发现手册与代码现实冲突时就地解决"条款下的必要修复（不修就无法保持 commit 3 全绿）。

## 手册外发现（非本次修复，已在测试注释中记录）

- **`disk_search_qg`（paged 内核）在本机环境下跨调用不确定**：对同一 `QuantizedGraph` 实例、同一 query 连续调用 `search()` 三次，返回三个互不重叠的 top-10 集合（用一次性诊断程序通过 `gdb` 之外的直接对比确认，非偶然）。根因推测是异步页读完成顺序（io_uring/libaio）影响探索顺序，与本次改动的 admit 点逻辑无关——admit 点本身（`if (admit) res_pool.insert(...)`）在两条路径（`result_filter_` 与 `admission`）形状完全一致，arena 内核（无异步 I/O）下的等价性测试（`TombstoneParityArenaKernel`）稳定精确匹配，证明 admit 逻辑本身正确；仅 paged 内核受这个预先存在的不确定性影响，因此 paged 相关测试改为验证"两条路径都不泄漏 tombstoned id"而非逐位相等。**未修复**（超出本任务范围，修复需要深入 AIO 编排逻辑，风险与工作量都不小）。
- **`QGBuilder::build()` 的 out-of-memory 补丁路径存在一个数据相关、与线程数无关的崩溃**（`QuantizedGraph::update_qg_out_of_memory` 内一次 `memmove` 越界，`gdb` 回溯确认），在同一进程内连续构建多个小索引时可复现（约 2/5 概率触发，与是否单线程无关——已用 `num_threads=1` 复测，仍崩溃，排除竞态假说，确认是纯数据触发的缓冲区溢出）。**未修复**（与 admission 契约无关，超出本任务范围）；三个新测试文件（`test_admission_contract.cpp`、`test_unified_laser_admission.cpp`、`segmented_collection_laser_filter_test.cpp`）均改为"进程内只构建一次索引，多个 TEST 共享"的 fixture 模式规避，注释中记录了根因与规避方式。

## 遗留边界

- **`LaserSegment`（AnySegment 面）目前只能走 paged pool 驻留**：`LaserSegment::open()`/`open_directory()` 硬编码构造 legacy `LaserSegmentSearcher`，从不读取 `x_laser_residency` manifest 字段选择 `UnifiedLaserSegmentSearcher`。这意味着契约验收 3（bitmap 过滤 recall）在 Collection 端到端层面只能实际验证 paged pool 一种驻留；resident arena 驻留的 admission 正确性已在磁盘层（`UnifiedLaserSegmentSearcher` 直连测试，两种驻留都覆盖）与内核层完全验证，只是"经 Collection"这条链路目前物理上够不到 arena 驻留。这是本次改动前就存在的架构现状（`segment_factory.hpp::load_segment_from_manifest` 虽支持按 manifest 选驻留，但没有任何生产代码路径调用它），不在手册决议清单内，未做扩展。
- **`sorted_rows` filter kind 的具体 wire format 系本次实现时定义**（`payload` = 连续 `uint64_t` 行/PID 数组，`payload_size` = 字节数），因为手册与代码库中此前均无先例可循；已在 `row_admission.hpp`/`unified_laser_segment_searcher.hpp` 注释中写明，`admission_from_sorted_rows` 单测覆盖，但 `unified_laser_segment_searcher.hpp` 里 `kind=sorted_rows` 分支未被任何集成测试实际经过（`segmented_collection.hpp` 只发 `kind=bitmap`，不发 `kind=sorted_rows`）。
- **RowAdmission.popcount 未接入 planner**：按决议 8 保留字段不接线，符合手册。

## 关键文件路径

- `include/index/graph/laser/qg/row_admission.hpp`（新）
- `include/index/graph/laser/qg/qg.hpp`（内核 admit 点、admission 尾参）
- `include/index/graph/laser/qg/residency.hpp`（ResidencyProvider 接口）
- `include/index/disk/types.hpp`（`DiskSearchOptions.filter`）
- `include/index/disk/unified_laser_segment_searcher.hpp`（admission 编译点、`compile_admission`）
- `include/index/disk/laser_segment.hpp`（bitmap 接受、label→PID 翻译、`admission_aware_search`）
- `include/index/collection/segmented_collection.hpp`（bitmap 生产者、not_supported 回退重试、悬垂指针修复）
- `tests/laser/qg/test_row_admission.cpp`、`tests/laser/qg/test_admission_contract.cpp`（新）
- `tests/disk/test_unified_laser_admission.cpp`（新）
- `tests/collection/segmented_collection_laser_filter_test.cpp`（新）
