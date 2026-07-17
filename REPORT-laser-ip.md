# REPORT — laser IP 距离核并行线 (feat/laser-ip-kernel)

memqg 退役前置路线①:laser 内核 IP 距离支持 + 非 QG 侧 metric 面。
基线 wave2-integration@cf67680。与 WAL-2C 并行,文件硬切分。

## 状态

- **W0(oracle 基座,纯测试零生产代码):已完成并 commit(5cb0ca0),7 测试全绿。**
- **W1(ip.hpp 内核):等 codex 审查放行。** 审查修正未到期间已完成设计推导(§3)+ 骨架测试 `tests/laser/space/ip_kernel_test.cpp`(2/2 绿,数值证实 §3 推导:参考 laser IP 因子的单级估计 == memqg == 官方,rel<5e-4,dims 64/128/192/256/1024;q=c 锁定 1-<c,o>)。**生产 ip.hpp 未落**——ip.hpp 落地时把骨架里的 `reference_laser_ip_factors` 换成 `laser::space::laser_ip_factors` 即成正式 W1 单测。
- **W2(非 QG 侧 metric 面):计划见 §5,未执行(等 W1)。**
- **W3(qg.hpp/qg_updater.hpp 搜索路径分派):不在本分支执行,交接线备忘见 §6。**

---

## 1. 关键坐标与架构事实(动手前侦察所得)

- laser 内核距离分两条:**精确路径**(`space/l2.hpp::l2_sqr` 包 `simd::l2_sqr`)与**因子路径**(量化期估计器修正因子)。fastscan 位内核 `space/bitwise.hpp` metric 无关(只 pack_binary),**W1 不碰它**。
- L2 因子实体在 **`include/space/quant/rabitq_core.hpp::RaBitQCore::laser_l2_factors`**(canonical,红线 `include/space/**`,**不可改**)。laser 侧 `include/index/graph/laser/quantization/rabitq.hpp::rabitq_factors` 调它,硬编码 L2。
- memqg 侧 `RaBitQCore::memory_factors(..., metric)` **已含 IP/cosine 分支**(rabitq_core.hpp:73-125),与官方库 `rabitq_impl.hpp:127-131 METRIC_IP` 逐字同式(f_add=`1-<r,c>+...`,f_rescale=`-l2_sqr/ip_resi`,f_error=`1*tmp_error`)。
- laser 估计器公式(`include/simd/laser_dispatch.hpp:117`,metric 无关的 SIMD):
  `appro_dist = sqr_y + triple_x + fac_dq*width*result + fac_vq*vl + sqr_qr`。
  metric 特化只在:(a) 每邻居因子 triple_x/fac_dq/fac_vq(build 侧,rabitq.hpp),(b) query 侧 `sqr_y`(`qg.hpp:1116` = `l2_sqr(query,cur_data,dim)`)与 `sqr_qr`(残差范数),(c) g_add 语义。
- **AlayaLite IP-as-distance 约定 = `-<x,y>`**(`simd/distance_ip.ipp:39` "Negative for distance metric, smaller=more similar")。min-heap 距离序下,IP 相似度取负即距离样值。

---

## 2. 判断调用(numbered)

**J1 — 三方 oracle 中"官方库参考"用 re-derivation 而非 #include 官方头。**
官方库 `baselines/RaBitQ-Library` 是 header-only + 自带 Eigen/命名空间,直接 include 进 AlayaLite 构建有依赖/flag 冲突风险。采用**从源码逐行 re-derive**(`official_ip_factors()`,带 rabitq_impl.hpp:80-136 行级引用)。更强:若 re-derivation 同时对齐 repo memqg 与暴力精确,公式即被三角定位。T1 实测 official==memqg 因子(bits 逐位相等,f_add/f_rescale rel<1e-3)通过,re-derivation 忠实。

**J2 — oracle 只锚定"约定无关真值",不预置 laser 因子约定。**
laser 因子约定(signed_x/fac_norm 形态)是 W1 ip.hpp 产出物、且与 memory 约定差一个 factor-2(见 J5),属审查标的。W0 只固化官方/memqg/暴力三方(约定无关),laser 约定的验证留 W1,**保持审查门开放**。

**J3 — 估计误差界脚手架用"未量化估计"隔离因子代数。**
`unquantized_ip_estimate()` = `f_add + g_add + f_rescale*<half_signed,q>`(无 LUT 字节量化),隔离二值码误差。官方界 `|est-(1-<q,o>)| ≤ f_error*g_error`(g_error=`||q-c||`,query.hpp:105;eps=1.9,rabitq_impl.hpp:17)。实测 2000 样本 **frac_within=1.0、max(err/bound)=0.34、mean_signed_err=-0.22 vs mean_bound=59(≈无偏)**。量化路径另由 T5 走真实 RaBitQSpace 覆盖。

**J4 — 排序契约拆成"解析恒等"+"统计召回"两层。**
manifest "IP 序与负距离序全同" 是**解析恒等**:target=`1-<q,o>` 与 `-<q,o>` 差候选无关常数 +1,升序排列同一置换(T4a 对 32 候选逐位 ASSERT_EQ,零阈值)。单批 RaBitQ 估计的统计召回是另一回事(单 fastscan 批、无图重排,天然粗),软地板 0.45(对齐兄弟测试 RaBitQSpaceIpNormTest),实测未量化 0.591、memqg 路径 0.60/0.587,均 >>随机 0.31。原始 0.75 阈值系我早期误标定,已改。

**J5 — laser IP 因子推导(factor-2 已对账,待审查确认)。见 §3。**
`RaBitQQuantizer::batch_quantize`(rabitq.hpp:81-82)对 memory 因子做 `metric==l2 ? 2*scale : scale`;laser `rabitq_factors` 不后缩放。故 laser L2 `factor_dq == 0.5*memory_L2_f_rescale`(equivalence 测试实证),对应 `laser_inner=<signed_x,q>=2*memory_inner=2*<half_signed,q>`。IP 分支 memory 不做 2×,故 laser IP `signed_query_scale` 须取 L2 形态的**一半**。数值以官方库/memqg 估计等价为准并记录于此。

**J6 — golden 落盘定位用编译期宏 `ALAYA_IP_ORACLE_GOLDEN_DIR` 指向源树。**
tolerance 比较(rel 1e-3)容跨平台 float 抖动;`ALAYA_IP_ORACLE_REGEN=1` 或文件缺失时自举重写。已提交 `ip_oracle_golden.tsv`(16 行 dim128/256)。

**J7 — oracle target 走默认 AlayaLite 链接(非 BARE/非 LASER),但注册在 tests/laser/ 下。**
oracle 只碰 core-space(rabitq_core/rabitq_space/simd),不含任何 laser 内部头;默认 `GTEST` target 即链 AlayaLite。放 tests/laser/space/(受 `ALAYA_ENABLE_LASER` 门控)遵 manifest,与未来 ip.hpp 就近。

---

## 3. W1 设计:laser IP 因子推导(待审查确认)

复用 `laser_l2_factors` 记号(rabitq_core.hpp:136-154):`signed_x=2·bits-1`,`fac_norm=1/√dim`,`R=||residual||`,`S=residual·signed_x=2·ip_resi`,`normalized_x0=fac_norm·S/R`,`normalized_x1=fac_norm·(centroid·signed_x)`,`x_x0=R/normalized_x0=R²/(fac_norm·S)`。

L2(现状)→ IP(目标),照官方 L2→IP delta(rabitq_impl.hpp:124-131)映射:

| 量 | L2(laser 现状) | IP(推导目标) |
|---|---|---|
| base(=triple_x) | `R² + 2·x_x0·normalized_x1` | `(1 − residual·centroid) + 1·x_x0·normalized_x1` |
| signed_query_scale(=factor_dq) | `−2·x_x0·fac_norm` | `−1·x_x0·fac_norm`(即 L2 之半) |
| factor_vq | `signed_query_scale·(2·popcount−dim)` | 同结构,用 IP 的 signed_query_scale |

对账:`x_x0·normalized_x1 = R²·ip_cent/ip_resi`,`x_x0·fac_norm = R²/S = R²/(2·ip_resi)`;代入即 base_IP=`1−<r,c>+R²·ip_cent/ip_resi`、scale_IP=`−R²/(2·ip_resi)`——与 memory_factors IP 分支经 batch_quantize(IP 不 2×)后、且 laser_inner=2×memory_inner 的估计**逐点等价**。误差项 `f_error_IP=1·tmp_error`(vs L2 的 2×)仅供界分析,不入 factor 三元组。

**落点(W1,等审查):**
- 新文件 `include/index/graph/laser/space/ip.hpp`:命名空间 `alaya::laser::space`;`inline float ip(v0,v1,dim)` 包 `simd::ip_sqr`(返回 `-<v0,v1>`,精确路径);`laser_ip_factors(residual,centroid,sign_bits,dim,fac_norm)->RaBitQCoreFactors<float>`(上表 IP 列;注释给字面量 1 上游出处 rabitq_impl.hpp:128,131)。结构/ISA 纪律照抄 l2.hpp,复用 bitwise.hpp。
- **不改** rabitq_core.hpp(红线);laser_ip_factors 落在 laser/space/ip.hpp,可 include rabitq_core.hpp 只读复用 `RaBitQCoreFactors` 与 `dot_product`。
- W1 单测(见骨架):laser IP 因子 vs W0 三方 oracle,断言**估计等价**(official==memqg==laser,约定无关)+ 码逐位相等 + 排序解析恒等;而非裸因子相等(factor-2 约定差)。仿 `tests/laser/rabitq_factor_equivalence_test.cpp` 的估计等价范式。

---

## 4. 新老兼容策略(缺口 10:manifest metric 键)

- **只加不改语义**:磁盘 manifest 现有键语义零变(红线)。metric 作为**新增键**,缺省 `l2`——老段(无 metric 键)读作 l2,行为零变化。
- **fail-closed**:非法/未知 metric 值在 descriptor/importer 解析处显式抛错,不静默回退。
- **feature 字符串**:现存 `disk_laser_segment`(available_features)不动;IP 段是否加子标记(如 `disk_laser_segment_ip`)待 W2 定——倾向**不新增 feature 字符串**,metric 走 manifest 键即可,reader 老版本遇未知键按缺省 l2 处理(前向兼容)且 IP 段因 metric 键存在而被新 reader 正确识别。此点 W2 落定时复核并记此处。
- CHANGELOG/golden:memory_qg golden 家族与 laser_fixture 独立,W1/W2 不碰既有 golden;新增仅 `ip_oracle_golden.tsv`(测试内)。

---

## 5. W2 计划(非 QG 侧 metric 面,未执行)

1. `laser_segment.hpp` descriptor 去 l2 硬编码(:330,:550 一带):metric 从 manifest 读,缺省 l2,非法 fail-closed。
2. importer(`laser_segment_importer.hpp`)增 metric 字段透传+校验;**维度门(:219-226 pow2≥128)不动**(路线②)。
3. `collection_target_builder.hpp::laser_target_support`(:270 一带):admission 扩 `{l2, ip}`;cosine 走既有 `L2NormalizedQuerySegment` 归一化包装(wave-1 已泛化前置条件,collection 层引擎无关)——确认 laser 路径可挂后放行 cosine,挂不上记判断调用留 W3。
4. 反假绿(缺口 9):新测试断言**引擎身份**(磁盘 manifest/implementation feature 字符串),非只断言 algorithm id;fallback flat 显式失败。
5. 全链 metric 打通后:ip 段经非 QG 路径可建可查的部分做端到端小测。

---

## 6. W3 接线备忘(交后续波;qg.hpp/qg_updater.hpp 被 2C 独占,本分支零字节触碰)

**精确落点(file:line)+ 最小分派方案:**

1. **`include/index/graph/laser/qg/qg.hpp:1116`** — `scan_neighbors` 内 `float sqr_y = space::l2_sqr(q_obj.query_data(), cur_data, dimension_);`
   IP:改为按 metric 分派 `sqr_y = (metric==ip) ? space::ip(query,cur_data,dim) : space::l2_sqr(...)`。`space::ip` 返回 `-<q,c>`,即 memory g_add(IP) 语义。**这是最小分派点**(单行三元)。
2. **`qg.hpp:656-660` 与 :830-834** — query prep 的 `sqr_qr = Σ residual_query²`。IP 无 `||q_r||²` 项:metric==ip 时 `set_sqr_qr(0)`(或等价常数),使 `appro_dist` 公式的 `+sqr_qr` 项归零。
3. **`qg.hpp:697/706`、:910/919** — 两处 beam-search 主循环内 `sqr_y += space::l2_sqr(cur_data+dimension_, residual_query, residual_dim)` 的残差补偿:IP 侧改 `-ip(...)` 或依两级布局重定义(见风险)。
4. **build 侧 `rabitq.hpp::rabitq_factors`** — 现硬调 `laser_l2_factors`;W1 落地 ip.hpp 后,按 metric 分派到 `laser_ip_factors`。此项介于 W2/W3:建 IP 段需要,可随 W2 metric 面一起接,或 W3 统一。倾向 W2(build 侧属 metric 面)。
5. **`qg_updater.hpp`** — 流式更新的量化调用同样需 metric 分派到 IP 因子;与 #4 同源。

**排序契约前提(已由 W0 T4a 证):** IP 估计 = affine(`-<q,o>`)+ 常数,正斜率,min-heap 零改动即正确;下游 exact_rerank(segmented_collection.hpp:1825)用 `space::ip` 精确回填。

**测试计划(wave-3 集成后):**
- 端到端 IP recall 下限锁(仿 laser recall floor lock);IP 段 build→search→rerank 全链。
- 与 memqg IP 平价重测(同并发语义,缺口 5:batch 定义须统一后再"打平")。
- 引擎身份断言(缺口 9)。

**风险/未决:**
- qg.hpp 两级向量布局(main `dimension_` + residual `residual_dimension_`)下 IP 的残差项分解未完全展开(#3);2C 改 qg.hpp 时需与本备忘对齐,exact IP 用 `space::ip` 全维度即可兜底正确性。
- cosine:collection 层归一化后即 IP,内核复用 IP 路径;放行判定在 W2。

---

## 7. 与 2C 零交集自证

`git diff --name-status cf67680(base) HEAD`:
```
M	tests/laser/CMakeLists.txt        (append-only,0 删除行)
A	tests/laser/space/ip_oracle_golden.tsv
A	tests/laser/space/ip_oracle_test.cpp
```
红线碰撞扫描(qg.hpp/qg_updater.hpp/segment_op_wal.hpp/include/wal/**/mutable_*/segmented_collection.hpp/collection.hpp/index/collection/types.hpp/routing_snapshot.hpp/tests/collection/**/tests/wal/**/tests/laser/qg/**/include/space/**/include/simd/**/laser/utils/**/bitwise.hpp):**无命中(clean)**。
tests/laser/ 下只新增 tests/laser/space/ 子目录;tests/laser/CMakeLists.txt 只追加(集成 append-append 可机械合)。

---

## 8. 测试结果汇总

`laser_ip_oracle_test`(7 测试全绿):
- OfficialFactorsMatchMemqg:官方 re-derivation == memqg 因子(bits 逐位、f_add/f_rescale rel<1e-3),dims 64/128/192/256/1024。
- EstimatorLocksToOneMinusDotAtQEqualsC:q=c 时 est == 1-<c,o>(rel<2e-3)。
- UnbiasedAndBounded:2000 样本,frac_within=**1.0**,max(err/bound)=**0.34**,mean_signed_err=**-0.22** vs mean_bound=59(≈无偏)。
- RankingMatchesExactNegIp:仿射排序契约逐位 ASSERT_EQ 通过;未量化召回 0.591(地板 0.45)。
- MemqgFastscanPathTracksExactIp:dim128=**0.60**、dim768(→1024 pad)=**0.587**(地板 0.45)。
- SimdIpConventionIsNegated:`simd::ip_sqr==-<a,b>` 锁。
- GoldenFactorsStable:golden 自举+比较双路径通过。

`laser_ip_kernel_test`(W1 骨架,2 测试全绿):
- EstimateMatchesMemqgAndOfficial:参考 laser IP 因子单级估计 == memqg == 官方(rel<5e-4),码逐位相等,dims 64/128/192/256/1024 × 24 trial。
- EstimateLocksToOneMinusDotAtQEqualsC:q=c 时 laser IP 估计 == 1-<c,o>(rel<2e-3)。

l2 全家回归(相邻既有测试,W0 零生产代码变更故 by-construction 不受影响,抽验确认):
- `tests/space/rabitq_space_test`:9/9 绿(含 IP recall 兄弟测试 unit=0.636/nonunit=0.71)。
- `tests/laser/rabitq_factor_equivalence_test`:2/2 绿(含 L2 因子等价 + 零残差 NaN 策略)。

**cosine 放行:** 未放行(W2 事项)。依据:内核仅需 IP;cosine=collection 层 L2NormalizedQuerySegment 归一化后即 IP,引擎无关(缺口 1/dossier)。W2 确认 laser 路径可挂归一化包装后放行,挂不上记判断调用留 W3。

**遗留边界:** (1) W1 ip.hpp 待审查放行;(2) laser IP 因子推导(§3)待审查数值确认;(3) qg.hpp 两级布局残差项(§6 #3)待 2C 集成对齐;(4) 维度门 pow2≥128 属路线②不动。
