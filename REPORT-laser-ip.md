# REPORT — laser IP 距离核并行线 (feat/laser-ip-kernel)

memqg 退役前置路线①:laser 内核 IP 距离支持。基线 wave2-integration@cf67680,与 WAL-2C 并行。
codex 终审(STOP,9 BLOCKER)后**路线改按其焦点 F**:本分支只交付 **W0 修正** + **W1 休眠内核件**;
一切生产接线(qg metric dispatch / factor pack / searcher / segment / importer / target build+open /
构图 / cosine / admission)推迟到 **2C 合入后的原子阶段**(§5)。

## 状态

- **W0(oracle 基座):已修正并绿。** 真官方 fixture(离线 generator 真链官方库)+ score-domain 规范 +
  退化输入契约。`laser_ip_oracle_test` **11/11**(6 oracle + 5 degenerate)。旧 golden 自举分支
  (`GoldenFactorsStable` + `ip_oracle_golden.tsv` + `ALAYA_IP_ORACLE_REGEN`)已删除。
- **W1(ip.hpp 休眠内核):已落地并绿。** `include/index/graph/laser/space/ip.hpp`(精确 wrapper +
  full-sign IP factor helper),**不被任何生产 TU include**,只被 `ip_kernel_test.cpp` 消费。
  `laser_ip_kernel_test` **7/7**(含 factor_dq==official.f_rescale/2 锁 + 一维反例 + factor_vq 映射 +
  零残差策略 + ISA 覆盖)。
- **原子阶段(生产接线):未执行,计划见 §5(焦点 F 六步 + 焦点 E 反假绿七件套)。** 2C 合入后单原子提交启用。

---

## 1. 关键坐标与架构事实

- laser 内核距离分两条:**精确路径**(`space/l2.hpp::l2_sqr` 包 `simd::l2_sqr`;本程新增 `space/ip.hpp::ip`
  包 `simd::ip_sqr` 返回 `-<a,b>`)与**因子路径**(量化期估计器修正因子)。fastscan 位内核
  `space/bitwise.hpp` metric 无关,W1 不碰。
- L2 因子实体在 `include/space/quant/rabitq_core.hpp::RaBitQCore::laser_l2_factors`(红线 `include/space/**`)。
  laser 侧 `include/index/graph/laser/quantization/rabitq.hpp::rabitq_factors`(:99-122)硬调 L2;
  `factor_vq = factor_dq * (2·popcount − dim) = factor_dq·Σs` 在此下游派生,**metric 无关**。
- memqg 侧 `RaBitQCore::memory_factors(..., metric)` 已含 IP 分支(:114-125),与官方
  `rabitq_impl.hpp:127-131 METRIC_IP` 逐字同式。**IP 半符号 f_rescale = −l2_sqr/⟨r,h⟩ = −2A**。
- laser 估计器(`include/simd/laser_dispatch.hpp:117`,metric 无关 SIMD):
  `appro_dist = sqr_y + triple_x + fac_dq·width·result + fac_vq·vl + sqr_qr`。metric 特化只在因子三元组
  (build 侧 rabitq.hpp)、query 侧 `sqr_y`(`qg.hpp:1116`)与 `sqr_qr`、g_add 语义。
- **AlayaLite IP-as-distance = `-<x,y>`**(`simd/distance_ip.ipp:39` "smaller = more similar")。

---

## 2. 判断调用(numbered)

前程 J1–J7(见 git 历史 91d8c0b 版 REPORT):三方 oracle re-derivation 定位、约定无关锚定、界脚手架、
排序契约两层、laser factor-2 推导、golden 落盘、oracle target 链接。本程按 codex 焦点 F 修正/接续:

**J8 — fixture 用"共享确定映射 + 存 seed + 存输入 FNV",不落原始向量。**
`make_row_inputs(seed,dim)`(整数 LCG `x=x·1664525+1013904223` → `((x>>24 & 0xFF)−128)/16`,全 IEEE 精确)
被离线 generator 与 CI 测试**逐字节共用**;fixture 存每行 seed + `input_fnv`(FNV-1a64 over data++centroid)。
故:(a) 跨平台位精确复现(消除 `normal_distribution` 跨 stdlib 漂移,B-LIP-08 item4);(b) fixture 紧凑;
(c) FNV 锁使任何映射漂移**硬失败**。"输入落盘"以 (seed→确定映射) + FNV 证明形式满足。

**J9 — 手写重推降级为 `algebra_ip_factors`(第四代数 oracle),官方权威值只来自不可变 fixture。**
fixture 由 `tools/gen_ip_official_fixture.cpp` 离线**真链官方库 @ b1f613d** 调
`one_bit_code_with_factor(..., METRIC_IP)` 生成(源码 + fixture + commit SHA + 命令一并提交,generator 不进 CI 构建)。
CI 只读 fixture:**缺失/截断/commit 不符/输入被篡改**全 hard fail(5 模式实证,§8)。删除自举 golden 与 REGEN 分支。

**J10 — 非有限=拒绝(位模式谓词),非传播。**
实证 Release 用 `-Ofast`(⇒ `-ffast-math`/finite-math-only):`std::isnan/isfinite` 被折叠、NaN 既不传播也不可 FP 检测。
故诚实契约:内核对非有限输入是 UB,**上游 importer/descriptor 须在量化前 fail-closed 拒绝**,谓词=整数指数位检查
(`(bits & 0x7F800000)==0x7F800000`,免疫 fast-math,同 `rabitq_factor_equivalence_test` NaN 位惯例)。本程锁该谓词。

**J11 — laser_ip_factors 零残差策略=NaN-free `{base=1, dq=0}`(⇒ vq=0, error=0),背离 L2 保留 NaN。**
检测 `⟨r,s⟩==0`(⟺ r==0,因 `⟨r,s⟩=Σ|r_i|≥0`)。o=c 时该策略给**精确** target `1−⟨q,c⟩`。B-LIP-09。

**J12 — laser_ip_factors 用 A-form(`A=||r||²/⟨r,s⟩`)双精度累加实现,非镜像 laser_l2_factors 的 Eigen float。**
签名与 laser_l2_factors 完全一致(原子阶段可直接 drop-in 到 rabitq_factors:112 的 metric 分派);A-form 与之代数恒等、
更可审计(与手册公式 1:1)、命名空间安全;`fac_norm` 在 A-form 中约掉,`[[maybe_unused]]` 保留仅为签名兼容。
等价性:vs 官方 fixture rel<1e-3、vs memqg 估计 rel<5e-4。

**J13 — factor_vq 映射以 fastscan query 量化分解锁定(非平凡恒等)。**
令 `q_i=vl+delta·qcode_i`,则 `factor_dq·⟨s,q⟩ = factor_dq·delta·Σ(s_i·qcode_i) + factor_vq·vl`,
`factor_vq=factor_dq·Σs`。证 factor_vq **恰好吸收 vl 偏移**(对齐 laser_dispatch.hpp:118 的 `fac_vq·vl` 项)。

**J14 — ip.hpp 由构造保持休眠:只被 kernel 测试 include,不加进 `test_laser_compile.cpp`。**
后者显式枚举 laser 头(非 glob,实证不含 ip.hpp),且属共享 laser 测试 infra(本程禁碰)。kernel 测试自身即证 ip.hpp 在工具链下编译。

### 2a. Score-domain 规范表(B-LIP-03;写死进 `ip_fixture_common.hpp` 头注 + 测试)

| 量 | 值 | 用途 | 返回用户? |
|---|---|---|---|
| `d_exact` | `−<q,o>` | AlayaLite IP 距离(`simd::ip_sqr`);min-heap 样值 | 是(exact rerank 回填) |
| `d_est` | `1 − <q,o>` | RaBitQ 估计 native target(官方 METRIC_IP + memqg IP 分支) | 否(仅遍历) |
| 误差比较 | `d_est − 1` vs `d_exact` | 两者皆 `−<q,o>`;比 `|est − target|` | — |
| SymQG 完整路径(**本程未用**) | `2 − <q,o>` | 若跑官方完整 QG 需先减常量 2 | 否 |
| 排序 | 仿射 `+1`(或 `+2`)候选无关 | 双精度排序 + **统一候选 id tie-break**;仅在整数下标上判恒等 | — |

**红线**:删一切 float 逐位相等断言(SIMD/FMA/Eigen 归约序不同),全改容差(参 `tests/simd/ip_test.cpp:53`)。

---

## 3. W1 已落地:laser IP 因子(`include/index/graph/laser/space/ip.hpp`)

full-sign 推导(codex B-LIP-02):`r=o−c`、`s_i=2·bit_i−1∈{−1,+1}`、`A=||r||²/⟨r,s⟩`。单级估计
`est = g_add + base + factor_dq·⟨s,q⟩`,g_add=`−⟨q,c⟩`,target `1−⟨q,o⟩` ⇒

| 量 | 值 | vs L2 |
|---|---|---|
| base(triple_x) | `1 − ⟨r,c⟩ + A·⟨c,s⟩` | L2 是 `||r||² + 2·x_x0·nx1`(单交叉项,系数 2→**1**) |
| factor_dq | `−A` | L2 是 `−2·x_x0·fac_norm=−2A`(即 IP 取**半**) |
| factor_vq | `factor_dq·Σs`(下游派生) | 同结构 |

**半符号 vs 全符号 factor-2**:官方/memqg 用 `h=s/2`,配 `f_rescale=−2A`;laser 用全符号 `⟨s,q⟩=2⟨h,q⟩`,
故 **`laser.factor_dq(−A) == official.f_rescale(−2A)/2`**。一维反例 c=2,o=3,q=4:正确 `−11`,照抄官方
`−2A` 不减半得 `−15`(已做成 executable 反例 W1-d)。base **不减半**(代数上 `laser.base == official.f_add`)。
零残差策略见 J11。

---

## 4. 新老兼容策略(缺口 10;原子阶段落地,B-LIP-05 修正)

- **不加第二个 manifest metric 键、不设缺省**:`SegmentManifest.metric` 已是必需键(segment_manifest.hpp:251)。
  旧 L2 段本有 `metric=L2` 原样打开;缺失/未知 → **fail-closed**(缺省会放行损坏 manifest,已否决前程"缺省 l2")。
- **原生 kernel metric/preprocessing 证明**:QG 文件头无 metric;importer 须与 reader **双向核对**原生证明;
  无证明的旧产物只能解释为 L2;IP/COS 缺证明或不一致 fail-closed。证明若写 superblock 保留区**须等 2C 分配字段**(禁抢)。

---

## 5. 原子阶段计划(焦点 F 六步,2C 合入后单提交)

1. **修正 W0**:真官方 fixture + score-domain 规范 + 退化输入。**[本程已交付]**
2. **W1 只落 dormant ip.hpp**(精确 wrapper + full-sign factor helper + 纯核测试),不开任何生产 gate。**[本程已交付]**
3. **等 2C 合入。**
4. **原子启用**`qg metric dispatch + factor pack + searcher/segment/importer + target build/open`(**一个提交内**):
   - `rabitq.hpp:112` metric 分派 `laser_ip_factors`(drop-in);`qg.hpp:1116` sqr_y 按 metric 走 `space::ip`;
     `sqr_qr` IP 侧归零;`qg.hpp:667/1116/1276` medoid/精确/邻居 factor 的 L2 硬编码全解;
   - `laser_segment.hpp`(:330/:550/:824/:867/:1049/:1057)、`laser_segment_searcher.hpp:197` metric gate、
     `laser_segment_importer.hpp:249` 拒绝、`collection_target_builder.hpp:640/909` build/reopen 全同批放开;
   - **2C 前 importer/opener/target admission 一律 fail-closed**(禁中间"能打开但物理 L2 的假 IP 段")。
5. **IP 构图过渡**:用 memqg `QgSegment::export_graph_snapshot()`(FrozenGraphSnapshot)作 metric-aware topology oracle
   → `QGBuilder::build_from_graph()` 打包(B-LIP-06;不给 Vamana 直接传负 IP)。**cosine**:构建期归一化 harvested
   向量**副本**再构图/量化 + build/reopen 均包 normalized-query adapter + 持久化并校验 `l2_normalized` 证明 +
   **高范数错误候选**测试(B-LIP-07;active mutable adapter 属 2C 独占,留后)。
6. **最后放行 admission**:需引擎身份 + native metric proof + paged/arena 两路径 + **端到端 IP recall floor**。

### 5a. 反假绿七件套(焦点 E;admission 放行前全断言)

1. receipt:`built_algorithm==laser`、`flat_fallback==false`;
2. Collection manifest:sealed entry 的 algorithm/factory、`required_features=="disk_laser_segment"`;
3. native manifest:`metric==IP/COS`;
4. descriptor 与 Collection schema 的 dim/metric/scalar 一致;
5. 原生 kernel metric/preprocessing 证明一致;
6. reopen 后再次搜索;
7. golden/fixture 缺失**不得生成并通过**(本程 W0 已按此:hard fail,无 regen 路径)。

`target_implementation_key()` 单独**不是**证明(按请求 registration 返回)。

---

## 6. 与 2C 的交集声明(B-LIP-01,**不再宣称零交集**)

- **已声明、已机械化的 CMake 交集**:`tests/laser/CMakeLists.txt` 与 2C 共享。本程只在 **EOF 追加**一段带
  `# laser-ip` 标记的区块(oracle/kernel 两 target);集成时**单一集成人**在 2C 合入后的树上机械 append-合并。
- **生产头编译耦合**是原子阶段推迟的根本原因:`laser_segment→searcher→qg.hpp`、`importer→residency→qg.hpp`、
  `collection_target_builder→qg_builder` 且被 2C 独占的 `collection.hpp`/`types.hpp` 包含。故本程产物
  (**休眠头 + 测试**)在 91d8c0b(cf67680 派生)判绿即可;**集成时主会话须在 2C 合入后的树上全量重编** LASER/
  Collection/header-closure 终验。
- **2C 独占文件**(qg.hpp / qg_updater.hpp / segment_op_wal / include/wal/** / mutable_* / segmented_collection /
  collection.hpp / index/collection/types.hpp / routing_snapshot / tests/{collection,wal,laser/qg}/**):**一字节未碰**。
- 本程新增文件全在 laser 保留区:`include/index/graph/laser/space/ip.hpp`、`tests/laser/space/**`。
  `test_laser_compile.cpp` 等共享 laser 测试 infra **未碰**。

`git diff --name-status cf67680 HEAD`(集成前):
```
M  tests/laser/CMakeLists.txt                    (EOF append-only,带 # laser-ip 标记)
A  include/index/graph/laser/space/ip.hpp        (休眠内核)
A  tests/laser/space/ip_fixture_common.hpp       (共享:输入映射/fixture reader/algebra oracle)
A  tests/laser/space/ip_official_fixture.tsv     (不可变官方 fixture)
A  tests/laser/space/ip_oracle_test.cpp          (W0)
A  tests/laser/space/ip_kernel_test.cpp          (W1)
A  tests/laser/space/tools/gen_ip_official_fixture.cpp  (离线 generator,不进 CI 构建)
D  tests/laser/space/ip_oracle_golden.tsv        (删:自举 golden 被 fixture 取代)
```

---

## 7. 测试结果汇总

`laser_ip_oracle_test`(**11/11**):
- **OfficialFixtureMatchesMemqgAndAlgebra**:24 fixture 行,输入 FNV 复现锁 + memqg base/rescale/popcount ==
  官方(rel<1e-3)+ algebra 全项(含 f_error)== 官方。
- EstimatorLocksToOneMinusDotAtQEqualsC:q=c ⇒ est==1−⟨c,o⟩(rel<2e-3),dims 64/128/192/256/1024。
- **OneBitBoundIsHighProbability**(LUT bypass,4 seed × 2400 样本):frac_within=**1.0**、max(err/bound)=**0.196**、
  mean_signed=**0.057** vs mean_bound=58.5(≈无偏,高概率界非逐样本绝对界)。
- AffineOrderIdentityAndRecall:双精度 + id tie-break 仿射恒等逐位 ASSERT;未量化召回@10=**0.591**(地板 0.45)。
- MemqgFastscanPathTracksExactIp(真 LUT 量化):dim128=**0.617**、dim768(→1024 pad)=**0.597**(地板 0.45)。
- SimdIpConventionIsNegated:`simd::ip_sqr==−<a,b>` 锁。
- Degenerate(5):NegativeInnerProduct / ZeroQuery / **ExtremeNonUnitNormOrdersByIpNotNorm**(证伪 ||o||=1 假设)/
  TieBreakAndAllEqualSign / **NonFiniteRejectionPredicate**(位模式拒绝谓词)。

`laser_ip_kernel_test`(**7/7**):
- EstimateMatchesMemqgAndAlgebra:laser==memqg==algebra 单级估计(rel<5e-4),码逐位相等,dims 64..1024×24。
- EstimateLocksToOneMinusDotAtQEqualsC:q=c ⇒ 1−⟨c,o⟩。
- **FactorDqIsHalfOfficialRescale**:24 fixture 行,`laser.base≈official.f_add`、
  `laser.signed_query_scale≈official.f_rescale/2`(rel<1e-3)。
- **OneDimensionalCounterexample**:c=2,o=3,q=4 ⇒ 正确 −11、照抄官方得 −15,二者显著相异。
- **FactorVqCapturesQuantizationOffset**:query 量化分解重构 == `factor_dq·⟨s,q⟩`(rel<1e-3),dims 64/128/256。
- **ZeroResidualPolicyIsNanFree**:base==1、dq==0、vq==0、无 NaN;o=c 估计==1−⟨q,c⟩。
- **ExactWrapperAcrossIsaLevels**:`ip()`/generic/AVX2/AVX512 == −<a,b>(rel<1e-3,含非 8 整除尾;AVX 按 CPU feature 门控)。

**fixture 生成记录**:commit `b1f613d7412a041000d1e71aaa323d3e7554e733`;命令
`g++ -std=c++17 -O2 -fopenmp -I <RaBitQ-Library>/include -I tests/laser/space tests/laser/space/tools/gen_ip_official_fixture.cpp -o /tmp/gen_ip_fixture && /tmp/gen_ip_fixture > tests/laser/space/ip_official_fixture.tsv`;
24 行(dims 64/128/192/256 × 6 trial);列 `dim trial seed input_fnv popcount f_add f_rescale f_error`。

**fixture hard-fail 实证(5 模式)**:缺失 / commit 不符 / 截断行 / 输入篡改(FNV 不符)全 FAILED;还原后 PASS 且 fixture 无改动。

**l2 全家回归(零红)**:`rabitq_factor_equivalence_test` 2/2(L2 因子等价 + 零残差 NaN 策略)、
`laser_simd_dispatch_test` 10/10、`rabitq_space_test` 9/9(IP recall 兄弟)、`tests/simd/ip_test` 26/26、
`test_laser_compile` exit 0(laser 头树编译,证 include 图未破)。

**cosine 放行**:未放行(原子阶段第 5 步)。

---

## 8. 遗留边界

1. **原子阶段接线**(§5 第 4–6 步)全部未做,是 codex 焦点 F 的核心推迟项;2C 合入后单提交启用,并在合并树上全量重编终验。
2. ip.hpp `laser_ip_factors` 是 **drop-in**,但 rabitq_factors(rabitq.hpp:112)的 metric 分派、`qg.hpp` 两级布局残差项
   (sqr_y/sqr_qr 分解)属原子阶段;dormant 期未接。
3. 非有限输入=UB(-ffast-math);拒绝谓词已锁,但**执行拒绝的 importer/descriptor gate 属原子阶段**(本程禁碰)。
4. superblock 保留区(native metric proof 落点)由 2C 占用,字段分配须等 2C 合入,禁抢。
5. cosine 归一化存储副本 + normalized-query adapter + `l2_normalized` 证明属原子阶段(active mutable adapter 属 2C 独占)。
