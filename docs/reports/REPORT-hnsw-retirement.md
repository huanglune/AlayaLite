# HNSW 退役执行报告

分支 `feat/hnsw-retirement`,基于 `main@5dfc5a5`。五个 W 序全部完成并提交,验收清单六项全部满足(细节见下)。本报告是数据,不是散文。

## 交付物:五个 commit

| W | commit | 一句话 |
| --- | --- | --- |
| W1 | `f474425` | QG 接入 cosine 度量(复用 `L2NormalizedQuerySegment` 适配器),新增 `CollectionQgCosineSeal` |
| W2 | `1a80473` | `CollectionOptions::target_algorithm` 默认 hnsw→qg;Python `fill_none_values()` 按 dtype(+显式 quantization_type)分派 |
| W3 | `d01d127` | 删除 `include/index/graph/hnsw/` 全树 + 注册面/路由面;五个测试文件删除/迁移/改造 |
| W4 | `91a08f2` | 删除 `tools/codegen/` 全链 + 两个消费测试文件 + 三处未列入手册但确实引用了被删文件的 CI/lint/benchmark 消费点 |
| W5 | (本次) | CHANGELOG、CLIENT_USER_MANUAL.md 旗舰示例、三份 superseded 横幅、本报告;补两个开箱路径的永久测试 |

总计(W1-W4,base→W4 HEAD):51 files changed, 535 insertions(+), 4174 deletions(-)。W5 另有 6 个文档/测试文件的增量:CHANGELOG.md + 3 份 superseded 横幅 + CLIENT_USER_MANUAL.md + test_client.py(84 insertions, 5 deletions)。

```
W1  f474425   4 files changed,  241 insertions(+),   31 deletions(-)
W2  1a80473   3 files changed,   24 insertions(+),    5 deletions(-)
W3  d01d127  35 files changed,  269 insertions(+), 3698 deletions(-)
W4  91a08f2  15 files changed,   19 insertions(+),  458 deletions(-)
```

## 验收清单逐项状态(对照手册第 68-75 行)

1. **CollectionQgCosineSeal 绿(recall≥0.95+零向量+reopen);cosine 对照数字在 REPORT。** 通过,首次运行即绿(未调参)。数字见下方"cosine 对照数字"节。
2. **默认路径四条开箱行为各有测试且绿。** 四条路径均有**永久提交的**测试(非临时脚本):
   - 默认 float32 建→QG 封:`python/tests/client/test_client.py::test_collection_params`(W5 补强,新增 `index_type=="qg"` 断言)。
   - 默认 int8 建→flat:`python/tests/client/test_client.py::test_default_dtype_dispatch_int8_collection_uses_flat`(W5 新增,此前只有 W2 的临时脚本验证过)。
   - 显式 qg 无 rabitq→报错:`python/tests/client/test_collection_canonical.py::test_native_status_maps_to_versioned_python_exception`(W3 为其他原因改造,顺带覆盖此路径)。
   - 显式 rabitq 无 qg/laser→报错:C++ `TEST(CollectionFacade, RabitqRequiresExplicitQg)` 第一分支(W3 修复,强制 flat+rabitq 保证真正命中交叉校验)。
3. **全量 ctest 绿(除已知环境失败);pytest 默认口径全绿;golden 比对零漂移。** 详见下方"验证结果"。
4. **残留 grep 清单干净(仅 reserved 注释/历史文档/CHANGELOG)。** 17 行残留,逐条归类见下方"残留 grep 清单"。
5. **两处跨段裁决测试均已迁 QG 腿且绿;QG recall 下限锁测试绿。** `heterogeneous_segment_integration_test.cpp`(Hnsw+Flat→Qg+Flat)、`segmented_collection_test.cpp`(CrossSegmentHnsw→CrossSegmentQg)均迁移且绿;`collection_qg_recall_floor_test.cpp`(原 `collection_hnsw_qg_parity_test.cpp`)四个 case 全绿。
6. **CHANGELOG/手册示例/三横幅落地;REPORT-hnsw-retirement.md 完整。** 见下方"文档改动"节。

## cosine 对照数字(W1,decision 1 要求)

非严格受控 A/B(两份测试文件维度/行数不同,各自沿用自己文件的既有常量,未刻意对齐):

| 测试 | 维度 | 行数 | recall@10 |
| --- | ---: | ---: | ---: |
| `collection_hnsw_seal_test.cpp::CollectionHnswCosineSeal` | 32 | 640 | **1.0000** |
| `collection_qg_seal_test.cpp::CollectionQgCosineSeal` | 64 | 384 | **0.9700** |

与同一构建下 l2/inner_product 的既有 hnsw-vs-qg parity 数字量级一致(`collection_hnsw_qg_parity_test.cpp` 迁移前实测:l2_unit hnsw=0.9900/qg=0.9900、l2_nonunit hnsw=0.9650/qg=0.8650、ip_unit hnsw=1.0000/qg=0.9950、ip_nonunit hnsw=1.0000/qg=0.9650),QG 比 HNSW 低 0-10pp 的量级符合"RaBitQ 量化本身有代价"的预期,无异常。

`collection_qg_recall_floor_test.cpp`(W3 由 parity 测试改造)的四档下限,即取自这次 parity 测试的实测值减 10-15pp 余量:l2_unit≥0.85、l2_nonunit≥0.75、ip_unit≥0.85、ip_nonunit≥0.80。

## 验证结果

- **C++ 构建**:`cmake --preset release -DBUILD_PYTHON=ON` + `cmake --build --preset release -j 32`,W1/W2/W3/W4 后各跑一次,全部零错误(仅有与本改动无关的既存 `-Walloc-size-larger-than=` 警告)。
- **ctest**:`ctest --preset release -LE performance`,W3/W4/W5 后各跑一次全量(非仅 scoped),均只有 1 个失败——`laser_segment_test` 的 `DifferentialRankOnlyManifestGate`,手册已预先声明为 `BUILD_PYTHON=ON` 下的已知环境性失败,与本改动无关。W1/W2 后跑的是 scoped 子集(collection|hnsw|qg 相关 18 个目标),全绿。
- **pytest**:`uv run --locked pytest -q`,W1 后 176 passed/3 skipped(基线);W2 后先出现 12 个回归(定位为 dispatch 规则过窄导致,修正后归零,同样 176/3);W3 后 165 passed/11 failed/3 skipped(11 个失败全部 confined 在即将于 W4 整体删除的 `test_collection_canonical_matrix.py`);W4 后 161 passed/3 skipped/0 failed;W5 补两个永久测试后 **162 passed/3 skipped/0 failed**。总数变化可完全对账:179(W2)→179(W3,同总数只是 11 个变红)→164(W4,减少的 15 = 11 个失败 case + 2 个仍通过的 matrix case + `test_codegen_matrix_golden.py` 1 个 + `test_legacy_cleanup_lint.py` 被删的 1 个)→165(W5,+1 新测试)。
- **golden**:`uv run python3 tests/golden/generate_artifact_baseline.py --build-dir build/Release`,W3(删 hnsw 两族后)与 W4 后各跑一次,均报告 `artifact baseline matches`,零漂移。
- **lint**:所有触碰到的 Python 文件跑过 `uv run ruff check` + `uv run ruff format --check`,全绿;所有触碰到的 C++ 文件跑过 `clang-format -i` 后重新构建确认未破坏语义。

## 残留 grep 清单(`grep -rin hnsw --include='*.hpp' --include='*.cpp' --include='*.py' --include='*.yaml' --include='*.txt' --include='*.cmake' .`,17 行)

```
python/tests/golden/test_legacy_cleanup_lint.py:23:    # HNSW retirement wave: the whole hnsw-keyed dispatch codegen chain.
python/tests/client/test_client.py:94:        # to qg+rabitq (HNSW retirement wave, see CHANGELOG).
include/index/graph/graph.hpp:44:  std::unique_ptr<OverlayGraphType> overlay_graph_ = nullptr;  ///< the overlay graph of HNSW
include/index/graph/graph.hpp:233:   * @return Graph<Id> The final graph of the HNSW .
tests/include/utils/evaluate.hpp:25:// mirrors collection_hnsw_seal_test.cpp's exact_score:249-252.
tests/collection/collection_qg_seal_test.cpp:647:  // HNSW, NSG, and Fusion are all retired: their algorithm ids stay reserved
tests/collection/collection_qg_seal_test.cpp:650:  // (HNSW+rabitq used to be the invalid_argument "explicit index_type=qg"
tests/collection/collection_qg_seal_test.cpp:653:       {core::algorithm::hnsw, core::algorithm::nsg, core::algorithm::fusion}) {
tests/collection/collection_qg_recall_floor_test.cpp:4:// QG recall floor lock (HNSW retirement wave, decision 4). This file used to
tests/collection/collection_qg_recall_floor_test.cpp:5:// build one HNSW segment Collection and one QG segment Collection side by
tests/collection/collection_qg_recall_floor_test.cpp:7:// collection_hnsw_qg_parity_test.cpp, and REPORT-u4-preflight.md item 3, for
tests/collection/collection_qg_recall_floor_test.cpp:8:// the original HNSW-vs-QG numbers). HNSW is retired in this same wave, so
tests/collection/collection_qg_recall_floor_test.cpp:99:// hnsw_seal's (now retired) make_cosine_dataset style: unit vectors with a
include/core/algorithm_registry.hpp:17:inline constexpr AlgorithmId hnsw =
scripts/fullcache-probe/bench_memqg.py:45:        index_type="hnsw",
benchmarks/size_map/generate_size_map.py:53:        # Retired along with the hnsw-keyed dispatch codegen matrix
benchmarks/size_map/generate_size_map.py:54:        # (HNSW retirement wave): there is no longer a canonical identity
```

归类:
- **reserved 注释(1 行)**:`algorithm_registry.hpp:17`,与 nsg/fusion/vamana/diskann 同款措辞。
- **必要的保留 id 功能性引用(3 行)**:`collection_qg_seal_test.cpp:647/650/653`,测试"请求 hnsw 现在命中 not_supported 而非旧的 invalid_argument 交叉校验"这一行为本身,必须引用 `core::algorithm::hnsw` 这个保留 id,与 nsg/fusion 同款处理。
- **受保护文件,按手册指令原样不动(3 行)**:`include/index/graph/graph.hpp:44,233`(共享基建红线,手册禁止删除且未要求改注释,两处是描述"overlay graph"历史命名来源的通用注释);`tests/include/utils/evaluate.hpp:25`(手册明文"不改 evaluate.hpp")。
- **既存死脚本,手册明文不修不删(1 行)**:`scripts/fullcache-probe/bench_memqg.py:45`。
- **本波自己写的说明性注释,指向已完成的历史事实,不误导(9 行)**:其余全部,包括 `collection_qg_recall_floor_test.cpp` 文件头(解释此文件从何而来)、`generate_size_map.py` 的字段退役说明、两条 CHANGELOG 指路注释。

无一条落在"活代码里断言 HNSW 仍存在"的危险区间。

## 判断调用(按 W 编号,偏离手册处均已记录)

**W1**
1. `make_l2_normalized_query_segment`(`collection_normalized_segment.hpp`)的内层前置检查从"仅接受 `preprocessing==none`"泛化为"接受 `{none, engine_quantized}`,只拒绝 `l2_normalized`"。必要而非可选:`QgSegment::descriptor()` 硬编码返回 `engine_quantized`(RaBitQ 是 QG 唯一的量化方式),与"是否已做外部 L2 预归一化"是正交的两件事;不泛化则 W1 完全无法编译通过语义正确性(QG cosine 包装必然触发这条检查失败)。
2. 用两个已有的、各自独立提交的 cosine 测试(`CollectionHnswCosineSeal`/`CollectionQgCosineSeal`)的打印数字作为"hnsw-vs-qg cosine 对照",未另建手册字面意义上的"临时脚手架"脚本——两个数字已经是同一构建下的真实测量,额外脚手架不会带来更多信息,只会增加维护面。
3. 顺手删除了 `CollectionQgFallback` 测试里已经变假的"cosine 触发 flat 回退"子用例(W1 让 cosine+qg 转为成功路径,该子用例的前提不再成立)。手册未列出此文件,但该改动是 W1 语义变更的直接必然后果。

**W2**
4. `collection_facade_test.cpp` 的 `options()` 默认 helper 修复推迟到 W3 执行(手册原文把它归在 W2 描述里)。理由:该 helper 显式设置 `target_algorithm=hnsw`(不依赖 C++ 结构体默认值),W2 仅翻转默认值对它零影响;它只在 W3 把 hnsw 从 `validate_options` 白名单剔除后才会真正报错。W2 阶段动它是无意义的空转。
5. Python `fill_none_values()` 的 dispatch 规则从手册字面的"仅按 dtype"扩展为"按 dtype **和**显式 `quantization_type` 联合判断":float32+`quantization_type∈{None,"rabitq"}`→qg;float32+显式非 rabitq(`sq8`/`sq4`/`none`)→hnsw(W2 过渡态,W3 改判 flat,已在 W2 commit message 标注留痕并在 W3 兑现);非 float32→flat。必要原因:仓库里存在大量"显式设置 `quantization_type` 但不设置 `index_type`"的既有测试(sq8 恢复测试、多个 `quantization_type="none"` 的 hybrid_query 测试),纯按 dtype 分派会把它们的 `index_type` 误判成 qg 并被 qg 的 rabitq-only 交叉校验拒绝。首次实现遗漏此点,导致 12 个既有 pytest 回归;定位后按此规则修正,回归清零。
6. 未翻转 C++ 结构体 `CollectionQuantization quantization` 字段的默认值(手册也未要求)。穷举了全仓库 10 个 `CollectionOptions` 构造点,无一依赖裸默认值组合(全部显式设置 `target_algorithm`+`quantization` 或至少其中真正相关的一个),故"裸默认 qg+none 组合不自洽"是无害的文档性事实,不需要额外改动来消除。

**W3**
7. `heterogeneous_segment_integration_test.cpp`:把文件级共享常量 `kDim`(8→64)、`kRowsPerSegment`(24→40)一并上调以满足 QG 的 rotator dim 下限与 build 行数下限。核实过文件内另一个纯 flat 测试仅以符号方式引用这两个常量(从未硬编码字面量),上调对它无影响,已跑绿确认。
8. `segmented_collection_test.cpp` 的 `CrossSegmentHnswUpsertDeleteSuppressesOldVersions` 迁移时**没有**采用同样的"上调 dim"策略,而是保留 dim=2,显式向 `RaBitQSpace` 构造函数传入 `RotatorType::MatrixRotator`(而非默认的 `FhtKacRotator`)。原因:该测试与 `FakeMutableSegment`(`fake_mutable_segment.hpp`,同时被 3 个测试文件共享)配对,后者的线格式与距离计算把 dim=2 写死在多处(`std::array<float,2>`、`payload.vector.dim != 2` 校验、`query[0]/query[1]` 距离公式),泛化它超出本波范围且风险高;`MatrixRotator` 恰好没有 `FhtKacRotator` 特有的 `floor_log2(dim)∈[6,11]` 下限,构造出的 QG 段功能完整,已跑绿确认。
9. `search_test.cpp` 的 `build_hnsw_graph` fixture 换成手写的暴力 k-NN 图构造(`build_knn_graph`),而非改用 QG 构建再抽取图。理由:该 fixture 的唯一用途是给 `GraphSearchJob` 提供一个"连通性合理的图"用于测试搜索算法本身,与用什么引擎构建无关;暴力 k-NN 让 fixture 彻底与任何图引擎解耦,不引入对 QG 内部结构的新依赖。实测:1000 节点数据集下入/出度 100% 覆盖(远超原有 ≥90% 门槛),recall 0.99(远超原有 ≥0.5 门槛)。顺带删除了因此变成孤儿的 `max_thread_num()` 及其 `thread_config.hpp` include;文件里另两个**W3 之前就已死**的函数(`selective_score_threshold`/`sparse_id_threshold`,经 `git show` 原始版本确认从未被调用)未动,不在本波范围。
10. 顺带把 `SearchHNSWTest`/`SearchHNSWTestSQSpace` 两个测试用例名改成 `SearchGraphTest`/`SearchGraphTestSQSpace`——它们已经不再使用 HNSW,保留旧名会误导。
11. 手册未列出但代码现实要求必须改的两处:`collection_facade_stress_test.cpp` 里一个从未 `.seal()` 的测试仍用 `target_algorithm=hnsw`(改为 `flat`,配置本就是惰性的,无需 quantization 配对);`collection_qg_seal_test.cpp` 的 "foreign-rabitq-hnsw" 子用例(手册 decision 5 已预判此场景,但未在文件清单里点名)按 decision 5 精神并入 nsg/fusion 的"已退役算法"循环。
12. 清理了三处不在手册文件清单内、但直接因删除 `collection_hnsw_seal_test.cpp` 而变成悬空引用的注释(`collection_normalized_segment.hpp`、`collection_qg_ip_norm_test.cpp`、`rabitq_space_test.cpp`),均只涉及注释文本,不涉及逻辑。

**W4**
13. 手册只列了 6 个文件/目标(dispatch.yaml、gen.py、生成物、两个消费测试、CMake 目标),实际发现还有 5 处真实依赖这条链路且会被我的删除直接搞坏的下游,一并处理并记录:
    - `.github/workflows/code-checker.yaml` 的 `codegen-drift-check` CI job(跑 `gen.py` 并 diff 生成物,两者都已不存在)——整个 job 删除。
    - `python/tests/ci/test_workflow_caching.py` 里对该 job 名字的引用——不删就会在下次跑测试时对着一个不存在的 job key 抛 `KeyError`,已同步更新。
    - `python/tests/golden/test_legacy_cleanup_lint.py` 的 `test_codegen_schema_has_no_runtime_or_rollback_fields`——直接读 `dispatch.yaml`,文件已删会抛 `FileNotFoundError`;删除该测试函数,把四个新退役路径按该文件自己的既有惯例(`REMOVED_PATHS` "stays absent" 列表)加进去,而不是留一个测函数覆盖一个不存在的文件。
    - `.pre-commit-config.yaml` 里三条排除 `_dispatch_matrix_params.py` 的 `exclude` 规则——文件已删,规则变成死配置,清理。
    - `benchmarks/size_map/generate_size_map.py` 的 `canonical_identity_rows` 字段(靠读 `dispatch.yaml` 算出来)——退役为固定值 0,与同一份 JSON 里已有的 `legacy_dispatch_rows_linked: 0`(另一次历史退役留下的先例)完全同款处理;同步更新已提交的 `baseline.json` 和 `README.md`。核实过这个脚本/baseline 不在任何 CI job 里被硬性比对,改动风险极低。
    - `cmake/AlayaPython.cmake` 文件头注释里"codegen"这一 Python_EXECUTABLE 消费方列举项——已不成立,删除。

**W5**
14. `docs/user/CLIENT_USER_MANUAL.md` 的 "Pure ANN Search" 一节(连同 "Persistence"/"Choosing Index Or Collection"/"Notes" 里的部分内容)调用 `client.create_index(...)`——**该方法在当前 `Client` 类里根本不存在**(`grep -rn "def create_index" python/src/alayalite/` 零命中),是 1.1.0 版遗留 API 整体移除后从未同步的文档债,与本波 HNSW 退役无关且规模远超本波范围,不在本次修复。仅对其中出现的 `index_type="hnsw"`/`quantization_type="none"` 做了最小字面替换(→`"qg"`/`"rabitq"`)以保持 grep 清洁,不改变该节示例本身已经跑不通的事实。**建议后续单独开一个文档修复任务**,把这几节要么删除要么改写成 `Collection` 用法。
15. 新增 `### Changed` CHANGELOG 小节承载默认值翻转/报错行为变化——手册原文只说"记录默认变更"/"记录行为变更",未指定放进哪个既有小节;鉴于内容性质是"行为变了"而非单纯"加了什么"或"删了什么",按 Keep-a-Changelog 惯例新开 Changed 更贴切。
16. `collection_qg_recall_floor_test.cpp` 四档下限阈值(l2_unit≥0.85、l2_nonunit≥0.75、ip_unit≥0.85、ip_nonunit≥0.80)取自**本次会话自己重新跑出的** parity 实测值减 10-15pp,而非解析 `REPORT-u4-preflight.md` 里的旧存档数字——两者在能交叉核对的档位(ip_unit/ip_nonunit)一致,新跑的数据与本次改动后的确切代码状态直接绑定,更可信。
17. W5 补了两个此前只有临时脚本验证过、没有永久测试覆盖的开箱路径(见验收清单第 2 项),不在手册文件清单内,但直接服务于手册自己写的验收标准第 2 条。

## 文档改动(W5)

- `CHANGELOG.md`:`[Unreleased]` 下新增 1 条 Added(QG cosine)、新开 `### Changed` 小节 2 条(默认翻转、报错行为变化)、`### Removed` 新增 2 条(HNSW 引擎本体、codegen 矩阵链)。已有的 nsg/fusion/vamana/diskann 历史条目原文未动(含其中"flat、hnsw、qg 不受影响"措辞,按手册要求保留);新条目一律用"flat、qg"措辞。
- `docs/user/CLIENT_USER_MANUAL.md`:"Common Index Parameters" 旗舰示例(原 :695-711 区域)重写为 `index_type="qg", metric="cosine", quantization_type="rabitq"`,新示例已用 `.venv` 手工脚本端到端验证跑通(create_collection→insert→search→options() 全部符合预期);"Common values" 的 `index_type` 枚举同步去掉 `"hnsw"`。":66 legacy 示例"检查结果见判断调用 14。
- 三份 superseded 横幅:`docs/design/memory-segment-migration-pattern.md`、`docs/design/ann-sealed-target.md`、`docs/design/memory-qg-legacy-dispatch.md`,均在标题正下方插入 `> **Superseded / historical (2026-07-17).**` 块引用,格式对齐仓库里已有的 `collection-canonical-facade.md`/`legacy-cleanup.md` 先例,注明日期并指向 `CHANGELOG.md` 的 `[Unreleased]` 条目;正文一字未动。

## 红线遵守情况

- 未触碰 `include/index/graph/laser/**`、`include/index/disk/mutable_laser_segment.hpp`、`include/wal/frame.hpp`、`tests/laser/**`——全程零访问,cosine 工作完全限定在 collection 层,未发现任何"必须动 laser"的情况。
- `graph.hpp`/`graph_search_job.hpp`/`vector_iterator.hpp`/`evaluate.hpp`/`max_neighbors` 字段/`L2NormalizedQuerySegment`/legacy `index.py` 墓碑均未删除;`L2NormalizedQuerySegment` 反而被 QG 复用继续服役。
- `include/space/**`、`include/simd/**` 全程只读,零改动。
- 算法 id=2(hnsw)已标记 reserved 永不复用;kind 词汇/WAL 相关零触碰。
- git 纪律:五个 commit 均无 `Co-Authored-By`;全程未 push;未开 PR;未使用 `git stash`。

## 未解决事项

1. `docs/user/CLIENT_USER_MANUAL.md` 的"Pure ANN Search"及相关小节存在与本波无关的更大范围文档债(调用不存在的 `client.create_index`),已在判断调用 14 记录,建议另开任务修复,本波未处理。
2. `scripts/fullcache-probe/bench_memqg.py:45` 仍用 `index_type="hnsw"`——按手册明文指令不修不删,现在这行代码如果真的执行会在 Python 层直接被 `ValueError` 拒绝(`common.py` 的合法集合已不含 `"hnsw"`),脚本本身处于"从未被本波验证、可能早已不可运行"的状态,维持手册裁决原样不动。
