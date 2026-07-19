# LASER 盘上量化图原地更新：VLDB/SIGMOD 工业轨论文故事线终报

> 定位：论文叙事架构 + adversarial review。证据基线为
> `docs/LASER_UPDATE_EXPLORATION.md`（下称“探索文档”）、
> `docs/LASER_UPDATE_NEXT_PLAN_YI.md`（下称“Yi 对照”）与
> `include/index/graph/laser/qg/qg_updater.hpp`。
>
> 总判断：这项工作已经越过“可行性 demo”，具备一篇强系统论文的机制、反例、实现和
> 长 churn 闭环；但以 VLDB/SIGMOD 工业轨标准，当前实验面仍不足。最危险的缺口不是
> WAL，而是**没有证明逐边免码本在分布漂移下带来可观察优势，也没有与 fresh-layer
> 系统做同资源实测**。若不补这两项，论文应收缩为“SymphonyQG/LASER 的首个原地更新
> 实例与工程经验”，不能把标题和摘要写成广义动态量化 ANN 的优越性结论。

## 1. 一句话主张与叙事弧

### 1.1 最有力的一句话

**LASER 把看似令图冻结的边上量化布局，反过来变成了局部更新原语：拓扑改一条边，
量化状态也只改这一条边；定宽空槽吸收新回链，行内打包码直接为驱逐和删除重连提供
零额外 I/O 的距离信号。**

这句话比“量化不是障碍而是杠杆”更强，也更可守：它明确了杠杆是什么、影响半径多大、
收益落在哪些操作上。它不声称所有量化索引都如此，也不错误暗示使用全局 PQ 码本的系统
每次插入都必须重训。

备选标题式主张：

1. **Quantization as an Update Primitive:** edge-local codes turn a packed on-disk graph from an
   immutable artifact into a locally repairable data structure.
2. **The Row Is the Update Unit, but the Edge Is the Validity Radius:** in-place updates preserve
   LASER's read-optimized layout without a delta graph or global re-encoding.
3. 较保守的工业轨版本：**A read-optimized quantized graph need not choose between static
   search performance and sustained in-place updates.**

不建议把“无全局码本”单独写成主 claim。稳定分布的 SIFT churn 尚未显示它相对固定 PQ
码本的实测优势；它目前是数学闭包依据，而非已证的端到端性能优势。

### 1.2 动机张力：需要击破的假两难

工业盘上 ANN 常被迫在两条路线中选择：一条保住 SOTA 读路径，把图压成定宽、量化、
SIMD 交织的不可变段，再依赖 fresh/delta 层和周期 merge；另一条保留可更新邻接表与
raw-vector 维护路径，却支付额外内存、寻址、双路查询或重建成本。LASER 看上去是最不可能
原地更新的一端：每行同时编码源点 raw PCA 向量、`degree_bound` 条边的 1-bit RaBitQ 码、
SoA 因子和 PID，32 槽 FastScan 交织打包，位置又由 id 直接算出。插入一个点需要改变许多
老行，直觉上会触发“解压—重编码—重写”和全局量化一致性问题。

论文的转折不能是“我们给静态文件加了写 API”，而应是三个反直觉结构事实：

1. 边 `(u→v)` 的码和因子只依赖 `u,v` 两端 raw 向量；FHT rotator 数据无关，故拓扑变化
   的量化有效性半径恰好是一条边（探索文档 §2.1）。
2. FastScan 打包只是可逆置换；替换一个 slot 只需解包并重打包一个 32-slot block，不会
   传播到其他边（探索文档 §2.2、§4）。
3. 行固定宽且 id 算术寻址；预付的空槽不是浪费，而是可无驱逐吸收 backlink 的更新余量，
   新点仍可追加到文件尾（探索文档 §2.3、§5 E1b）。

因此叙事弧应按下面推进：

1. **冻结直觉。** 为读优化的量化、打包和定宽布局看似排除了细粒度更新。
2. **通用方法撞墙。** 只 append 的新点不可达；照搬 raw-vector DiskANN 的协议遗漏了
   LASER 特有的边载荷维护；fresh/merge 与重建能工作，但牺牲单图、单路径与即时可见性，
   且未利用已经在行内付费的数据。
3. **结构反转。** edge-local quantization、可逆打包和 headroom 让“改图”和“修量化”
   在一个页级 RMW 中闭合；同一行内的码还是本地距离 oracle。
4. **三个原语。** append + backlink patch 完成插入；tombstone 完成即时删除；dead-row
   FastScan splice + gardening 完成长期整形。
5. **系统化。** 跨批驻留 buffer pool 利用 hub 偏斜，把许多逻辑边更新合并成少量物理页写；
   inline patch 与搜索自然重叠，达到 13.2k–14.7k inserts/s。
6. **长期判决。** 100% 换血时 consolidation 提升 8.6pp；更深插入搜索解决主要边质量欠账，
   gardening 修入度尾并在 50 轮复利，最终 recall 平台约 0.988。
7. **更深洞见。** 同样是持续 churn，DiskANN 对照以入度年龄衰减为主，而 LASER 的 evict
   backlink 持续刷新老行，残差反而主要在“出生时边质量”。**物理量化布局改变了动态图的
   失效模式，也改变了维护预算的最优投向。**

## 2. “通用方法失效”的分类学

这里的“失效”分三类：正确性/可达性失效、仍可工作但没有利用 LASER 结构、以及尚未实测
不能下结论。论文必须严格区分，避免把“我们没做”写成“别人不行”。

| 通用方案 | 在 LASER 上的机理问题 | 已有证据 | 可下的判断 |
|---|---|---|---|
| 只做 append-only | 新行只有出边，没有任何老点指向它；从旧 entry 出发的搜索无法进入新点 | no-backlink 臂的插入点 recall 恒为 0（探索文档 §5 E1） | **硬失效。** backlink 不是优化，而是可达性条件 |
| 直移植 DiskANN/Yi 原地协议 | 搜索、剪枝、反向 reconnect、页缓存可复用；但 raw-vector/PQ 邻接更新不负责 LASER 的逐边码、三因子和 32 槽交织，也不能直接利用 dead row 的边码做 splice | patch 与 builder 重算等价；LASER splice 使用死行 FastScan 零额外 I/O；Yi 对照 §1–§3 | **协议骨架可移植，数据面不可照抄。** LASER 需要 quantization-aware patch，但也因此得到免费估距 |
| 全局码本量化索引的更新观 | 固定码本可编码新点，所以不能声称“每次更新都重训”；真正区别是分布漂移时码本代表性可能下降，且拓扑边修改与点码维护不是同一个局部闭包 | 只有结构论证；SIFT 自然顺序、固定分布，无 drift 实验（探索文档 §2、§7 局限） | **当前仅是机制差异，非性能胜负。** 必须补 drift 才能升级为优势 claim |
| fresh 内存层 + merge | 可规避前台随机写，但引入双路搜索/top-k merge、delta 内存、后台资源隔离、manifest 和 merge；merge 时最终仍要为新拓扑边生成 LASER payload。它延后而非消灭边码写 | 四个触发门均未触发；纯 in-place 已达 14.7k/s 与 0.9878 稳态（探索文档 §8.4、§9） | **不是失效，而是当前工作负载下不划算。** 无 FreshDiskANN 实测，不能声称全面优于 |
| tombstone + 周期全量重建 | 轻删短期有效，但死边和路由 ballast 随 churn 增长；全量重建恢复质量却造成资源峰值、可见性和运维窗口问题 | 10% 删除几乎不掉 recall；无整形 100% 换血降到 0.8887；局部 consolidation 恢复到 0.9748（探索文档 §5 E2/E3、§7.2） | **纯 tombstone 长期失效；“tombstone+重建”是可用但粗粒度的基线。** 尚缺同资源重建成本曲线 |
| 每次反向边做完整 RobustPrune | 每个被改老行需读约 R 个邻居 raw 向量，放大随机读；完整 α 剪枝还会降低新点入度 | full 约 2303 patch reads/insert、205 inserts/s；evict recall 0.967 vs full 0.965，成本约 9×（探索文档 §5 E1） | **过度维护。** 行内近似距离足以选择最差边，churn 要入度而非保守剪枝 |
| 通用写路径调优：更多线程、逐 patch O_DIRECT、批末 flush、异步相位流水 | 热点 backlink 造成重复页写；同步 DIO 把设备往返压进临界路径；批末 flush 破坏搜索/写重叠；线程扩展还暴露短命分配导致的 `mmap_sem` 串行点 | DIO 5.9k < buffered 7.4k；批末缓存 3.4k；跨批驻留池写 35.8→8.6；TLS scratch 后终局 13.2k/14.7k（探索文档 §9.1–§9.3） | **写次数才是首要敌人。** 正解是跨批驻留合并 + inline patch，不是 syscall 口味 |
| 变长邻接/通用预留容量 | 变长结构可更新，但会引入分配、间接寻址或搬迁；在 LASER 中槽字节已经预付，少建 8 条边不会省空间 | R=56/容量 64 的 low-ef 插入点 recall 0.9618，超过 R=64 静态 0.9565；驱逐仅 3.9/insert（探索文档 §5 E1b） | **LASER 的固定宽不是包袱而是 headroom。** 但超静态结果仍需跨 R/数据集验证 |

核心措辞纪律：fresh layer、全量重建、固定 PQ 码本都不是“不正确”；论文要证明的是，
它们要么没有解决 backlink 可达性，要么付出了 LASER 本可避免的架构/维护成本。

## 3. 设计与贡献点

### C1：局部闭合的 quantization-aware update primitive

设计出 append + backlink page-RMW：从两端 row 中的 raw 向量重算单边 RaBitQ payload，
只解包/重打包所在 32-slot block，并保持 id 算术寻址不变。

证据锚点：三个结构事实（探索文档 §2）；pack round-trip、edge payload、assemble row、
npp=3 E2E 五类单测（§4）；实现接口与注释（`qg_updater.hpp` 文件头、
`insert_with_id`、`patch_reverse_edge`）。其中“逐字节一致”应严谨写为码位/ID逐字节一致，
因子在 `-Ofast` 跨调用点允许 `1e-5` 相对舍入；不要在正文偷换成所有 bytes 无条件一致。

### C2：利用行内量化状态的低成本拓扑维护

把源行自身作为查询运行 FastScan，选择最远边驱逐；删除整形时，用死节点行内、以死节点
为中心的码估计 `||u-n||`，召回候选后少量 raw 精排。量化载荷同时服务搜索与维护。

证据锚点：evict 与 9× 成本 full RobustPrune 曲线重合（探索文档 §5 E1）；splice 不增加
候选扫描 I/O，100% churn recall 0.8887→0.9748（§7.2）；实现 `consolidate_row` 注释与路径。

### C3：固定宽 headroom 是零空间成本的更新余量

在 `degree_bound=64` 下以 R=56 构图，保留至少 8 槽优先吸收 backlink，减少驱逐和旧图
破坏，而不改变 row size。

证据锚点：探索文档 §5 E1b，R56 基图 low-ef recall 不降，插入点 low-ef recall
0.9618 > 静态 R64 的 0.9565，且 34.2 fills 对 3.9 evictions。

### C4：面向 hub skew 的驻留写合并系统

逻辑更新在统一 buffer pool 内跨批驻留，读路径透视最新页，物理写推迟到水位逐出、整形或
finalize；inline patch 保留搜索 I/O 与补丁 CPU 的自然重叠，并用 thread-local scratch
去除堆伸缩串行点。

证据锚点：探索文档 §9.1–§9.3。物理写 35.8→8.6/insert；64T 13,216 inserts/s，
128T 14,691，较 7,439 基线 1.78–1.97×；质量逐点持平；11/11 单测通过。

### C5：持续 churn 的局部质量闭环及失效模式诊断

删除先 tombstone，再以 purge/splice 再生 headroom；插入加深搜索解决出生时边质量欠账；
低入度 gardening + 有预算 backlink pump 修复慢变量。由此把无界下坠变为高平台有界收敛。

证据锚点：探索文档 §7.2、§9.4。consolidation +8.6pp；`ef_insert 100→200` 单项
+0.92pp；入度 p1 0→15/16 的 garden 短程约 +0.4pp；终选 10×50k 为 0.9865、
50×10k 为 0.9878，尾段约 −0.005pp/轮。

### C6：量化布局改变动态退化的主导模式

与同仓 DiskANN/Yi 战役对照，LASER 的 evict backlink 在每轮约 30 万次驱逐中持续刷新
老行，入度年龄不是首要残差；最有效投入是提高插入候选质量，再以 gardening 修尾。

证据锚点：探索文档 §9.4 与 §8；Yi 对照 §3.2。此贡献目前是**单数据集经验性洞见**，
必须用多数据集、边年龄/存活率曲线加强，否则放在 discussion 而非摘要。

## 4. 假想审稿人攻击：按杀伤力排序

### 1. “免码本是全文核心，但实验根本没测漂移”——致命

攻击：SIFT 自然序 churn 近似同分布。固定 PQ 码本也能持续编码新点；论文观察到的收益
可能全来自图 patch，而非 edge-local quantization。当前不能声称优于全局码本更新。

防御/补实验：构造时间分段 drift：（a）聚类中心平移/旋转/尺度变化；（b）先按 coarse
cluster 或 PCA 第一主轴排序，只建前若干簇再逐簇注入；（c）至少一个真实时间戳数据集。
比较 LASER 固定 rotator、固定码本 DiskANN/PQ、允许周期重训/重编码的 oracle，报告新点
recall、老点 recall、码本重训停顿、重编码字节与更新吞吐。若结果不显著，删掉“免码本
质量优势”，只保留“边更新闭包”。

### 2. “没有 FreshDiskANN/fresh-layer 实测，通用方法失效只是稻草人”——致命

攻击：fresh layer 正是工业动态 DiskANN 的主流答案；触发门未触发是作者自设标准。
原地方案的 14.7k/s、内存和查询放大必须与 delta+merge 在同硬件同 durability 下比较。

防御/补实验：至少实现或接入一个可审计的 FreshDiskANN-style baseline：同 base graph、
相同总 RAM/CPU、相同更新可见性，扫 delta 容量和 merge 周期；同时报告 update throughput、
query QPS/p99、双路 recall、merge debt、写放大、内存和峰值后台资源。若无法复现官方系统，
把它明确标为“fresh-layer design point”，开源实现与参数，避免冒充原论文复现。

### 3. “1M、单 NVMe、SIFT 单 seed，系统与结论都可能是缓存玩具”——致命

攻击：4GB pool 能装下 500k churn 图，轮内 89k→118k/s 实际变成内存实验；hub 合并率、
8.6 writes/insert 和收敛平台都可能不随规模成立。

防御/补实验：至少 SIFT100M/Deep100M 或能显著超过 RAM 的 50M+；RAM cap 扫描
0.5/1/4/16GB，确保 working set > pool；两种 SSD；报告 cache hit、唯一脏页率、设备读写、
每阶段 wall time。数据集至少增加 Deep/GIST 或高维 embedding，≥3 seeds。

### 4. “没有并发查询—更新干扰曲线，工业轨核心场景缺席”——很高

攻击：当前搜索与写更新共享 buffer pool、shard locks、CPU 和设备，但只给隔离吞吐与
更新后 QPS。“搜索无锁”不等于 tail latency 不受影响。

防御/补实验：固定 query load 扫 update rate，及固定 update load 扫 query concurrency；
报告 recall/visibility lag、query QPS、p50/p95/p99/p999、update p99、maintenance backlog、
CPU/SSD 饱和点。分别测 insert、delete、consolidate、garden；以纯静态和 fresh-layer 为线。

### 5. “没有 crash consistency，不能叫生产系统”——很高但工业轨可诚实降级

攻击：一次 insert 修改新行和几十个 backlink 页，当前 `finalize()+fsync` 不是事务；外部
reader 也只在 finalize 后一致。掉电可留下半发布图、meta/tombstone 不一致和复用风险。

防御：论文定位为 production-derived prototype，并明确单进程 epoch/批次可见性，不声称
durable online update。最好补 full-page redo WAL + A/B superblock，并在 append、各 backlink、
WAL sync、data flush、meta flip 注入 kill -9。若来不及，必须把 durability 排除在贡献外，
提供检测/离线恢复策略，并在 limitations 首段写明。

### 6. “evict 用近似码做局部贪心，误差会不会长期累积？”——高

攻击：单次 E1 与 full 重合不证明长期不偏；误估的最差边可能系统性保留坏边，随后这些边
又成为 splice 候选，形成反馈环。最终 recall 可能被 gardening 掩盖。

防御/补实验：记录每次 FastScan argmax 与 raw exact argmax 的命中率、rank regret、被驱逐边
真实距离分布和新边存活时间；增加 exact-evict oracle 与 random-evict，对 300%+ churn 和
drift 跑长程；分别关闭 garden/consolidation 看误差独立累积。报告每轮累积错误而非只报终点。

### 7. “headroom 超静态只是 R=56/R=64 参数巧合”——高

攻击：R56 恰好正则化了 Vamana，静态 R64 可能参数未调优；只在 ef=60 超过，ef≥100 仍低。
“免费”也只指字节，不指构建/查询与有效出度。

防御/补实验：二维 sweep `build_R ∈ {40,48,56,60,64}` × `degree_bound ∈ {56,64,72,80}`，
在相同行大小预算和相同 build tuning 下比较 0/5/10/25/50% 增量；多 seed/数据集，报告
fills、evictions、入度、QPS、row bytes。主 claim 应降为“headroom improves the
quality–update tradeoff at zero incremental row bytes”，不要主打“普遍超过静态”。

### 8. “baseline 不公平：full RobustPrune 实现方式制造了 9×”——中高

攻击：full 臂逐邻居随机读取 raw vector，而成熟系统会缓存、batch、预取或维护 RAM PQ/raw
cache；205 inserts/s 不是算法固有成本。

防御/补实验：给 full 相同驻留 buffer pool、batch/staging 和并发；分解逻辑读与物理读；
加入 exact-evict（只精确判最远，不做全 RobustPrune）拆开“估距误差”和“剪枝复杂度”。
论文可保留逻辑 I/O 复杂度，但 9× 只能描述当前实现。

### 9. “量化闭包只在 SIFT 128D、免 PCA/残差路径上实测”——中高

攻击：真正 LASER 配置可能有 PCA residual、非 2 次幂维度、npp>1 和不同 row geometry；
大实验绕开了最易错路径。

防御/补实验：选择至少一个非 2 次幂维度数据集，强制 `main_dim < dim`；规模实验覆盖
npp>1 与跨页边界；patch-vs-builder 随机 property test 覆盖维度、degree、slot、重复向量、
id=0 和编译选项。把 bitwise/epsilon 语义说准确。

### 10. “固定 id 算术与 append-only 文件在长期 churn 下空间无界”——中高

攻击：100% 换血只是逻辑窗口，物理文件继续增长；没有槽复用时不能称 sustained update。

防御/补实验：实现 free-list + tombstone 持久化、暗槽 publish、入度为零后复用；跑 10×
数据量 turnover，证明文件大小平台化。若不做，标题避免“space-bounded”，并把 collection
compaction 作为必要组件而非可选优化。

### 11. “与 DiskANN 失效模式不同的结论缺乏受控因果证据”——中

攻击：两个系统的 R、prune、更新率、维护预算、缓存和数据顺序都可能不同，不能把差异归因
于量化布局。

防御/补实验：同数据、同更新 trace、相近 recall/QPS 点，统一记录边年龄、入度分位、每轮
驱逐率；给 LASER 加 no-evict/fill-only 对照，给 DiskANN 加相同 nearest-only backlink
策略。只有交互项显著，才在正文写“layout changes failure mode”；否则作为观察。

### 12. “比较点与度量选择可能掩盖质量损失”——中

攻击：E1 用 recall@10、churn 用 recall@100；自然顺序、单 seed；静态上界和 masked GT 的
定义不完全一致。低 ef 的 pp 差可能来自参数点选择。

防御/补实验：统一报告 recall@10/100、QPS–recall Pareto、插入点/老点/全体三组、masked
live GT；所有曲线置信区间 ≥3 seeds；以固定 QPS 和固定 latency 双重比较，不挑单一 ef。

## 5. Claim → Evidence → Gap 矩阵

优先级：P0 = 投稿前不可缺；P1 = 强烈建议；P2 = 可明确列局限。成本为研究工程粗估，
不含大规模机器排队和第三方 baseline 复现的不确定性。

| Claim | 已有证据锚点 | 主要缺口 | 优先级 / 预估成本 |
|---|---|---|---|
| 单边拓扑修改的量化有效性局部闭合 | 探索文档 §2；§4 单测；`qg_updater.hpp` 文件头与 patch 接口 | 大实验未覆盖 PCA residual、非 2 次幂、多 geometry | P1 / 3–5 天 |
| backlink 是 append-only LASER 的可达性条件 | 探索文档 §5 E1，none 的 new recall=0 | 仅 SIFT/一种 entry 与插入顺序 | P1 / 2–3 天，随多数据集一起补 |
| evict 近似 full RobustPrune 的质量而显著更便宜 | 探索文档 §5 E1，0.9667 vs 0.9649，逻辑读 81+40 vs 82+2303 | full 未同等优化；缺 exact-evict；缺长期误差轨迹 | P0 / 1–2 周 |
| 行内 FastScan 码是无需额外扫描 I/O 的维护信号 | 探索文档 §7.2；实现 `consolidate_row` | “零额外 I/O”是相对候选扫描，不含 dead row/raw rerank 读；缺消融 | P1 / 3–5 天 |
| headroom 以不增加 row bytes 改善更新质量 | 探索文档 §5 E1b | R56/R64 单点、单 seed；“超静态”可能巧合 | P0 / 1 周 |
| tombstone 适合短期删除，splice 使长期 churn 有界 | 探索文档 §5 E2/E3、§7.2；0.8887→0.9748 | 仅到 100% turnover；无空间复用；缺全量重建成本 baseline | P0 / 1–2 周（质量）；复用另 2–3 周 |
| 驻留 buffer pool 利用 hub skew 降低物理写 | 探索文档 §9.1–§9.3；35.8→8.6 writes/insert | 图可入池；缺规模/RAM cap/设备扫描 | P0 / 1–2 周 + 大机 |
| 原地路径达到 13.2k–14.7k inserts/s 且 recall 不降 | 探索文档 §9.3 | 单机隔离吞吐；无 p99、durability、并发 query | P0 / 2 周 |
| 边质量出生欠账是 LASER churn 主残差 | 探索文档 §9.4；ef_insert 100→200 +0.92pp | 单数据集因果外推；冷池成本只估计未实测 | P1 / 1 周 |
| gardening 修入度尾并产生长期复利 | 探索文档 §9.4；p1 0→16；50 轮 +0.46pp | 单 seed；维护成本在 resident 图上被低估 | P1 / 1–2 周 |
| 最终质量在 100% turnover 上有界收敛到约 0.988 | 探索文档 §9.4，10×50k/50×10k | 未到 300%+，无多 seed/真实时间 trace | P0 / 1–2 周计算 |
| 无全局码本使 LASER 对分布漂移更适合更新 | 探索文档 §2 的机制推导 | **完全缺端到端 drift 证据** | **P0 / 2–3 周** |
| 纯 in-place 比 fresh layer 更合适 | 探索文档 §8.4 四触发门；Yi 对照表明 Yi 也纯 in-place | 无 FreshDiskANN/fresh baseline，同资源成本未知 | **P0 / 2–4 周，高不确定性** |
| 量化格式改变 churn 失效模式 | 探索文档 §8、§9.4；Yi 对照 §3.2 | 非同配置受控对照 | P1 / 1–2 周 |
| 运行时读写可见性正确 | 探索文档 §7、§9.2；seqlock/overlay；11/11 单测 | 无混合负载 stress/TSAN/线性化测试；外部 reader 边界弱 | P0 / 1 周 |
| 可用于生产级 durable update | 当前没有；探索文档 §8/P3 明确未做 | WAL、双 superblock、persistent tombstone、recovery 全缺 | P2（若工业轨强主张则 P0）/ 3–5 周 |

## 6. 相关工作撞车检查

> 按用户要求，本节基于已有知识，不联网核验。论文定稿前必须逐篇核对版本、正式题名、
> 年份和具体机制。以下对较新的 IP-DiskANN、CleANN、OasisANN 细节存在记忆不确定性，
> 已显式标注，不应直接复制进 related-work 定稿。

### 6.1 逐项定位

- **FreshDiskANN。** 用内存 fresh index 吸收更新、删除列表屏蔽旧数据，并周期性 streaming
  merge 到磁盘图；其核心是 immutable base + mutable delta 的生命周期管理。LASER 本文
  直接修改同一盘上量化图，保持单搜索路径，并把边码维护与页 RMW 合并。它是**架构路线
  最近的反面基线，也是投稿时最需要实测的竞品**。

- **SPFresh（SOSP'23，面向 SPANN）。** 通过局部增量重平衡/分裂实现 near-real-time
  向量索引更新，重点是分区/posting 的局部演化和更新期间服务，而非 edge-resident
  quantized graph 的单边 payload 修复。共同点是避免全局重建；差异是更新单元分别为
  partition/posting 与 packed graph row/edge。

- **IP-DiskANN / Yi 线。** 纯 in-place DiskANN：搜索、写新节点、反向 reconnect、删除二跳
  旁路/修复，依靠用户态 buffer pool、dirty writeback 和 staged edges。它是**更新协议与
  系统实现上最近的正面竞品**。本文的独特性不应写“首次盘图原地更新”，而应写“首次把
  该协议闭合到 edge-resident RaBitQ/FastScan 布局，并从该布局派生 evict/splice/headroom
  原语”；Yi 对照也显示两者长期退化主因不同。

- **SymphonyQG。** LASER 格式的直接技术祖先：边局部 RaBitQ、FastScan 与定宽盘行服务
  高吞吐查询。本文不是新的查询量化方法，而是揭示并实现该布局此前未被利用的可更新性。
  必须清楚区分继承的格式/核与新增的 update protocol、maintenance、buffering 和 churn
  诊断。

- **RaBitQ 系。** 提供随机旋转、1-bit quantization、距离估计及其理论/实现基础；本文不
  贡献新的量化误差界，而是把“端点决定的边码”和 FastScan 排列用于动态图维护。避免把
  RaBitQ 本身的性质包装成系统原创。

- **Starling。** 面向盘上图 ANN 的高效搜索/布局与 I/O 优化（具体版本机制投稿前核验）；
  更接近静态读路径竞品，而不是持续原地更新协议。本文应比较静态 QPS/recall 是否维持，
  但独特性在动态边载荷维护。

- **CleANN。** 记忆中属于动态/可更新 ANN 或面向删除清理的工作，但其精确架构与发表信息
  **不确定**。定稿前核查它是否已有“局部清理 + 图修复”以及是否支持量化盘图；若是，差异
  必须落到 edge-local payload、dead-row FastScan splice 与 fixed-row headroom，而不能只说
  “我们也避免重建”。

- **OasisANN。** 记忆中属于面向在线/动态磁盘 ANN 的近期系统，但 exact mechanism
  **不确定**。需要重点核对它是否已有 in-place update、冷热层、后台重组织或 crash-safe
  服务。如果它也是 in-place 磁盘图，则最近邻竞品排序可能上升，本文独特性仍须限定为
  quantized edge-resident format 的维护闭包与量化辅助修图。

- **其他动态图 ANN（如 HNSW 动态插入/懒删除、在线 Vamana 变体）。** 它们证明图算法层
  可以局部插入/删除，但通常在 RAM 或显式邻接表上操作，不承担 packed edge-code 的更新、
  4KiB RMW 与盘上写放大。它们是算法 baseline，不是格式/系统等价物。

### 6.2 谁是最近邻竞品

不能只选一个：

1. **系统协议最近：IP-DiskANN/Yi。** 同为纯 in-place 磁盘图和 backlink reconnect。
2. **产品架构最近：FreshDiskANN。** 解决相同动态 DiskANN 需求，但用 delta+merge。
3. **数据格式最近：SymphonyQG。** 相同 edge-resident quantized/FastScan layout，但静态。
4. **局部在线维护思想最近：SPFresh。** 都以局部修复代替全量重建，但数据结构不同。

论文的 novelty 是这三条坐标的交点，不是任一坐标单独首次：

> **To our knowledge, this is the first in-place update design for a fixed-row,
> edge-quantized on-disk graph that makes topology repair quantization-aware without a delta
> graph or global re-encoding, and reuses the packed edge codes themselves for eviction and
> deletion splice.**

这条声明站得住的前提是文献核查确认 CleANN/OasisANN 等没有同类 edge-resident 设计。
“first dynamic quantized ANN”“first in-place DiskANN”都过宽，不能写。

### 6.3 最值得强调的差异

相对 Yi，不是“我们也能原地更新”，而是：Yi 的更新协议在 LASER 上必须多维护每条边的
quantized payload；这个负担因为 payload 的端点局部性而可闭合，又因为 payload 常驻行内而
反过来消除了 evict/splice 的候选扫描 I/O。**额外状态同时是额外义务和额外信息。** 这是
最像论文 insight 的比较句。

## 7. 论文骨架建议

### 7.1 Abstract：五句足够

1. 背景/矛盾：高性能盘上 ANN 把图压成定宽 edge-quantized rows，通常因此被视为静态。
2. 反直觉洞见：在 LASER/SymphonyQG 中，单边 payload 只依赖两端点，packed codes 又能
   直接为图维护估距，所以量化把更新限制在一个 row-RMW，而非制造全局失效。
3. 系统：介绍 append+backlink patch、headroom、tombstone+FastScan splice、gardening 和
   hub-aware resident buffer pool。
4. 结果：写 35.8→8.6/insert，13.2k–14.7k inserts/s；evict 与 full prune 质量重合；
   100% churn 从 0.8887 修到 0.9865/0.9878。
5. 意义与边界：保留单图读路径且无需 fresh layer/global re-encoding；如果 drift/fresh
   baseline 尚未补齐，摘要不要声称全面胜过两者。

### 7.2 Introduction：三段式

**第一段：假两难。** 以线上 embedding collection 的持续插入、撤回和分布变化开场；盘上
索引需要低 RAM、高 QPS，但把邻接、量化码与 SIMD 布局冻结后，工业系统往往加 delta 层或
重建。指出两者的双路查询、merge debt 和资源峰值，但不贬低其正确性。

**第二段：看似最坏的 LASER，恰是突破口。** 用一张 row anatomy 图解释 raw vector、
edge codes、SoA factors、PID 和 32-slot interleave；提出 edge validity radius、reversible
packing、arithmetic addressing 三事实。立即给 no-backlink recall=0 与 patch correctness，
建立问题不是“append”，而是 quantization-aware backlink repair。

**第三段：系统和结果。** 概括三个局部维护原语、跨批 buffer pool 与长期 scheduler；给
四个最硬数字。最后列 C1–C5，不把尚弱的 C6 放在首要贡献。

### 7.3 正文章节切分

1. **Background and Problem Formulation**：LASER row anatomy、查询路径、更新语义、目标与
   非目标（durability 若未做必须在此声明）。
2. **Why Generic Updates Miss the Structure**：append-only 反例、raw-vector reconnect 的
   payload gap、fresh/merge design point、写路径反例。用分类学而非 related-work 式罗列。
3. **Locally Closed Quantized Updates**：三结构事实；单边 payload 生成；block patch；
   append/backlink/publish；并发不变量。给形式化不变量：每个可见 edge payload 与其当前
   endpoints 一致；读者见完整旧页或新页；published high-watermark 不越过未完成 backlink。
4. **Quantization-Assisted Graph Maintenance**：ghost/valid slot、evict、headroom、tombstone、
   dead-row splice。解释为什么死行码仍估计 `||u-n||`。
5. **Sustaining Churn**：consolidation/headroom 闭环、ef_insert、indegree gardening/pump、
   scheduler 与预算；将 DiskANN failure-mode 对照放在末节。
6. **Write Path**：先呈现失败的 O_DIRECT、batch flush 和 pipeline，再给 resident pool、
   overlay/version、hub coalescing、inline patch、TLS scratch。工业轨很看重这组负结果。
7. **Evaluation**：按研究问题组织，不按 E0/E1 编号照搬实验日志。
8. **Related Work / Limitations / Production Path**：WAL、slot reuse、collection integration；
   不要把未实现功能伪装成设计贡献。

### 7.4 实验研究问题与图表清单

| 图/表 | 内容 | 要证明什么 |
|---|---|---|
| Fig. 1 | LASER row anatomy + 单 slot patch 范围 | 量化/打包看似冻结，但实际影响半径局部 |
| Fig. 2 | generic append、fresh layer、LASER in-place 三路径示意 | 本文保留单图读路径，明确成本位置而非画 strawman |
| Table 1 | 三结构事实与操作复杂度/I/O | 设计为何数学与布局上闭合 |
| Fig. 3 | none/evict/exact-evict/full 的 recall–update throughput Pareto | backlink 必需；cheap evict 是否逼近质量上界 |
| Fig. 4 | FastScan evict argmax 命中、rank regret、长期累积 | 回答近似误差会否累积 |
| Fig. 5 | build_R × degree_bound × churn 比例 heatmap | headroom 不是 R56/R64 巧合 |
| Fig. 6 | 逻辑 patch、unique dirty page、physical write/insert，随 batch/RAM/规模 | hub skew 和跨批 pool 才是写合并来源 |
| Fig. 7 | threads vs inserts/s，分解 search/patch/allocator/writeback | 解释 13.2k–14.7k 扩展与剩余瓶颈 |
| Table 2 | buffered、per-patch DIO、batch flush、resident staged、inline+pool | 系统负结果及最终设计选择 |
| Fig. 8 | 无维护、purge、splice、+deep insert、+garden 的 300% churn 曲线 | 各质量组件的独立贡献和有界性 |
| Fig. 9 | recall 与入度 p1/p5、边年龄、新边存活率的时间序列 | 支撑失效模式因果解释 |
| Fig. 10 | 10×50k vs 50×10k，含 maintenance cost | 修复频率与 garden 复利，而非只报终点 |
| Fig. 11 | query QPS/p99 vs update rate，多维护阶段 | 工业混合负载下是否真正可服务 |
| Fig. 12 | 规模和 RAM cap sweep，working set 明显超内存 | 排除 1M/全缓存玩具效应 |
| Fig. 13 | 同分布与多种 drift 下 LASER/PQ/fresh baseline 的新旧点 recall | 证明或否定“免码本”优势；全文最关键新增图 |
| Table 3 | LASER in-place、Yi/IP-DiskANN、FreshDiskANN-style：RAM、QPS、p99、更新、写放大、merge debt | 同资源端到端竞争力 |
| Table 4 | correctness/stress/crash matrix | 明确当前运行时与持久化保证 |

建议 evaluation 的研究问题顺序：

1. RQ1：量化 row 能否被局部、正确地修改？
2. RQ2：量化辅助的 evict/splice 是否以更低成本保持质量？
3. RQ3：headroom 在多参数与多数据集上是否改善更新 Pareto？
4. RQ4：跨批 buffer pool 在超内存规模和混合负载下能否扩展？
5. RQ5：持续 churn 是否有界，残差究竟来自出生边质量还是年龄/入度？
6. RQ6：相对 fresh-layer、raw-vector in-place 与全量重建，何时胜、何时不胜？
7. RQ7：edge-local/no-codebook 在 distribution drift 下是否产生实测收益？

## 8. 投稿决策与最小补实验包

当前版本直接投稿，严苛审稿人的合理结论会是：**机制新颖、工程诊断扎实，但 evaluation
像一个成功的 SIFT1M 原型，尚不足支撑工业轨的广义主张。** 最小可投稿包应按以下顺序：

1. **P0：drift + PQ/fixed-codebook 对照。** 决定主 thesis 能写多强。
2. **P0：FreshDiskANN-style 同资源 baseline。** 决定“无需 fresh layer”是结果还是偏好。
3. **P0：超 RAM 的 50M/100M、多数据集、≥3 seeds。** 决定 buffer pool 与 churn 结论能否外推。
4. **P0：并发 query/update 的 QPS–p99–update-rate 曲面。** 工业轨不可缺。
5. **P0：exact-evict 与长期误差 telemetry；headroom 二维 sweep。** 封住两个最明显的机制攻击。
6. **P1：300% churn、空间增长/复用、统一 recall 指标。** 强化“sustained”。
7. **P2 或明确降级：WAL/crash injection。** 若标题/摘要出现 production-ready，则升级为 P0；
   若定位 production-derived research prototype，可作为最醒目的 limitation。

最终建议的论文标题：

> **Quantization as an Update Primitive: In-Place Evolution of a Packed On-Disk Graph**

较保守、在 drift 实验不支持强优势时使用：

> **In-Place Updates for LASER's Edge-Quantized On-Disk Graph**

最重要的写作纪律是：把已经证实的“局部闭包、免费维护信号、写合并、长期质量闭环”写得
坚决；把尚未证实的“对全局码本漂移更强、全面优于 fresh layer、生产级 durability”明确
标成待验证或非目标。这样故事不是缩弱，而是让真正独特的贡献不被一个过宽 claim 拖垮。
