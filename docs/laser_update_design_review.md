# AlayaLite LASER 动态更新：设计判断与实验方案批判

> 独立评审终报，2026-07-11。代码判断以本仓库当前实现为准；相关工作链接列于文末。

## 结论先行

**LASER 在数学和行布局上可以更新，但当前格式和运行时还不能安全更新。** “逐边局部量化”这一核心论证成立：边 `u -> v` 的 sign code 和三个因子均可仅由固定 rotator、`u`、`v` 以及配置维度重算，不依赖全局码本或其他邻边。FastScan 的 32 路交织也是双射式重排，替换一个逻辑 slot 后重打包该 32-slot block 即可。真正的阻碍不是量化失效，而是有效槽语义、页级原子性、可增长元数据、缓存一致性和删除修复。

明确建议如下。

1. **论文主线：LASER-aware 原地增量图更新，外加异步质量整形。** 核心贡献应是 edge-resident RaBitQ 的局部可维护性、一个无需读取旧邻居向量的低 I/O 反向边更新器，以及质量/写放大边界；不能把“最远边替换”直接等同于 RobustPrune。
2. **产品主线：混合架构。** 小批实时更新进入可写 active LASER segment（或先经短暂 memtable），原地 patch；达到更新比例、删除比例或质量阈值后，后台构造新不可变段并通过 collection manifest 原子切换。不要把“永不合并的单个可写大文件”作为第一版生产承诺。
3. **原型前置条件：新增显式 `valid_degree`/valid bitmap。** 不能用 `triple_x == 0` 判空，也不应继续把 `id=0` 兼作空槽。否则已有幽灵边会被搜索当成到 0 号点的真实边，新增/删除语义也不可靠。
4. **4a 只适合作为最低成本基线。** 仅按 `d(v, ·)` 驱逐最远项会系统性破坏 Vamana 的角向多样性和长边，长期流式更新下风险为中到高；建议最低成本版本采用“精确的新边 + 小预算代表邻居 α 检验 + lazy RobustPrune”，并定期重建。

## 1. 结构判断与数学核验

### 1.1 三个因子是否逐边局部

令中心点为 `c = rot(u)`，邻点残差为 `o = rot(v) - c`，`s = sign(o) in {-1,+1}^D`，`fac_norm = 1/sqrt(D)`。当前 `rabitq_factors` 对每个邻点独立计算：

- `x = ||o||`；
- `x0 = <o, s/sqrt(D)> / ||o||`；
- `x1 = <c, s/sqrt(D)>`；
- `x_x0 = x/x0`；
- `triple_x = x^2 + 2*(x/x0)*x1`；
- `factor_dq = -2*(x/x0)/sqrt(D)`；
- `factor_vq = factor_dq * (2*popcount(bin)-D)`。

构建函数随后再给该 slot 的 `triple_x` 加上邻点原始 residual dimensions 的 `||v_r||^2`。因此结论是：**一个 slot 的 code、`triple_x`、`factor_dq`、`factor_vq` 只依赖 `(u,v)`、固定 rotator、主/残差维度，不依赖其他邻居或全局数据分布。** `fac_x1` 虽依赖中心 `u`，但 patch 的目标行正是 `u`，其 raw vector 在行内；这不破坏局部性。残差项依赖 `v` 的 residual raw vector：插入反向边时若新点 raw vector 已在内存，也不产生额外读。

需要加三条实现约束：

- patch 必须使用 **PCA 后完整向量**；主维用于 rotation，residual 维只把邻点的 `||v_r||^2` 加到 `triple_x`，查询侧另加 `||q_r||^2`。不能误加 `||v_r-u_r||^2`；当前 residual 近似并未存储或计算交叉项 `-2<q_r,v_r>`。
- `v == u` 或主维旋转残差为零时，代码出现 `0/0`；极近重复向量也可能因 `x0` 很小放大因子。这是静态构建已有的数值边界，动态路径应拒绝 self-loop、定义 duplicate-vector 策略，并做 `isfinite`/epsilon 防护。
- 若更新的是节点 `u` 的 raw vector 而非只新增不可变 ID，则所有 `u -> *` 和 `* -> u` 的边量化均失效。故 v1 API 应是 append/immutable-vector semantics；“update(label, vector)”应实现为 delete-old + insert-new。

### 1.2 FastScan block 是否可局部补丁

`pack_codes` 依次进行 64-bit 内字节反序、每字节高低 nibble 交换、`kPerm0` 行置换，以及把第 `j` 与 `j+16` 个 code 的 nibble 合入一个字节。这些都是无损重排/拼接；32 个 code 的一个 block 占 `32*D/8 = 4D` bytes，scanner 也恰以 `padded_dim << 2` 递增 block 指针。

所以：

- 逻辑 code 可逆；一个 slot 的 bit 不与另一 slot 做算术混合，只共享物理字节的不同 nibble。
- 安全的最小实现是解包一个 32-slot block、替换一个 code、重打包整个 block；三个因子和 ID 的 SoA slot 可直接替换。
- 不建议首版手写散点 byte patch；虽可行，但端序/nibble/16-lane 配对容易错。必须以“全行 `rabitq_codes` 结果 == block patch 结果”的逐字节 property test 覆盖 slot 0/15/16/31、多 block 和随机维度。

### 1.3 零填充不是无害空槽

当前 `scan_neighbors` 总是扫描 `degree_bound` 并将每个 neighbor ID 插入候选池；没有 per-row degree/valid bitmap。零尾槽产生 `id=0`、code=0、factors=0，近似距离并非无穷大，而近似为中心精确距离加查询 residual norm，可能很有吸引力。visited 去重只能减少重复插入，不能阻止第一次错误的 0 号边；而 0 本身又是合法节点。

`triple_x == 0` 也不能判空：浮点表达式可以合法为零或接近零，重复向量会生成 NaN，合法邻居 0 更无法区分。推荐按优先级选择：

1. native format v2 每行增加 `uint16/uint32 degree`（最简单）；
2. 或 degree-bound bits 的 valid bitmap（支持洞，但 scanner 后仍需 mask）；
3. 若短期不能改行长，则在 meta 中声明一个越界 sentinel `UINT32_MAX`，搜索显式过滤；这仍要重写旧索引尾槽，不如 v2 干净。

此外必须过滤 neighbor ID `>= committed_num_points`、self-loop 和重复 ID。此项不是优化，而是更新正确性的格式门槛。

## 2. 原地、LSM 与混合路线

| 路线 | 实时可见性/查询 | 写放大与质量 | LASER 适配判断 |
|---|---|---|---|
| 单大段原地 | 单图查询，新增点即时成为路由点 | 每插入约 `1 + R` 个页 RMW；随机写、锁和恢复复杂；长期拓扑漂移 | 很适合证明局部量化可更新，不宜独自承担首版生产 SLA |
| collection LSM | 写 memtable/新段快；查询 fan-out、跨段 top-k merge | compaction 昂贵但顺序写、恢复简单；段间没有路由边 | 符合现有 segment/manifest 架构，但当前 LASER hit 距离为 NaN、只按段内 rank 合并，先天无法正确做跨段全局 top-k，必须先返回可比距离 |
| 混合 | 热增量低延迟，段数受控 | 小增量随机写，周期批构建恢复质量；可控峰值资源 | **推荐产品主线** |

建议把 active segment 限制在例如 base 的 1%--5% 或以实测质量/写放大触发，不把比例写死。旧 immutable segments 可继续无锁查询；active segment 单写、多读。后台 merge 读取一致快照，生成完整 LASER artifacts，再以现有“临时目录 + fsync + manifest 原子发布”范式切换；旧段延迟回收。

论文可将三条路线都作为对照，但主命题应是：LASER 的 edge-resident 量化虽然增加每条拓扑边的 patch 数据，却免除了 PQ/codebook 重训，使局部拓扑变化可立即得到量化一致的边。

## 3. 反向边 evict-only 的风险与低成本改进

“对 `v` 的所有邻边用 `v` 作为 query 跑 FastScan，若新点更近就替换最远者”保持的是局部短边，不保持 α-RobustPrune 的覆盖/多样性。风险尤其集中于：

- 被驱逐的最远边可能是跨簇长边或唯一方向性桥梁，恰是低 ef 导航所需；
- 热点区域连续插入会让邻接表被近重复点占满，产生 crowding；
- FastScan 排序本身是近似的，最远边边界可能误判；
- 只向选中的 `R` 个邻居加反向边，入度分布会愈发偏斜；删除后更脆弱；
- 单次 10% 插入可能看似可接受，100% turnover 或局部/时间相关插入会暴露累积退化。

低成本改进按收益/成本排序：

1. **先填有效空槽，不驱逐。** 新边因子用 `(v,new)` 精确计算。
2. **保护边规则。** 至少保留每行若干最长/低冗余边，或限制同一新批次占据的 slot 数；这比无条件最远替换稳健。
3. **近似 α 检验。** 候选 `p` 是否被已保留 `q` 遮蔽需要 `d(p,q)`，仅有“以 v 为中心”的 `d(v,p)` 不足以完成严格 RobustPrune。可从 `q` 的行对 `p` 构造查询扫描，但 `p` 未必是 `q` 的邻 slot；所以所谓“只用 v 行现有距离做完整 α 检验”数学上不可行。可采用三角/角度启发式，但须标为 heuristic。更可靠的折中是仅读取一个小预算 `B << R` 的代表邻居 raw vector，精确检查新点与它们的冗余。
4. **lazy repair queue。** 为每行累计 patch 次数、近似冗余率、被驱逐长边、查询热度；超过阈值后批量读该行 R 个邻居 raw vectors，执行完整 RobustPrune 并整行重写。相邻脏行按页聚合，降低 RMW。
5. **周期 merge/rebuild。** 它是质量上界和碎片清理机制，不是失败。

实验中 4a 应命名为 `nearest-only replacement`，并与 `no-backlink`、精确 RobustPrune、预算 `B={4,8,16}`、lazy repair 四类对照；否则无法判断收益来自反向边本身还是剪枝策略。

## 4. 实验设计批判与补全

### 4.1 必需对照臂

- 900k 静态、1M 静态、900k+100k **仅新行无反边**、4a、完整 RobustPrune、预算化/lazy 版本；再加 collection 多段/LSM 基线。E0 的 900k 不是仅做 sanity，而是拆分“增加数据难度”和“更新拓扑损失”。
- 若声称接近 FreshDiskANN，至少加入其公开实现或同语义 DiskANN update baseline；否则只称内部 ablation。
- 新增 **full-row rewrite vs 32-block patch**：衡量 CPU、device write amplification 和 recall 是否逐字节等价。
- 删除需有：只过滤结果、过滤且不展开、保留路由但过滤结果、splice/local repair、周期 consolidation；“路由穿 tombstone”应是明确实验臂。
- 加 steady-state churn：10% insert/delete 循环至 100%--500% turnover，而非只做一次 10%。报告 recall 随累计更新量的曲线。

### 4.2 工作负载与顺序

- 随机顺序至少 3--5 seeds，并报告置信区间；另测按原始 ID、按一个主成分/簇排序、局部热点、分布漂移和 adversarial near-duplicate。
- 分清 bulk catch-up 与逐条实时；测 batch size `1/10/100/1000`。逐条插入时页写合并机会很少，不能拿批量吞吐代表在线吞吐。
- 混合读写测搜索线程数、一个/多个 writer、读写比、p50/p95/p99 搜索延迟、更新可见延迟、更新 p99，且固定 CPU affinity/queue depth。

### 4.3 指标和正确 GT

- masked recall 的分母必须是 live 集合上重新定义的 top-k；从 full GT 列表简单删除 tombstone，若不足 k，必须读取更深 GT 或重算，否则指标偏差。
- 报 recall@10--QPS 曲线及固定 recall 下 QPS，不只报同 ef；同时固定 ef 的差值有助定位图质量。
- I/O 统计区分 logical page reads/writes、实际 device bytes、fsync/WAL bytes、cache hits、写放大、queue depth；插入的 beam-search reads 与 backlink RMW 分开。
- 加 indegree 分布、连通分量/从 entry 可达率、平均路径 hops、长边保留率、邻居角向多样性、脏行比例；这些是解释 recall 的中间证据。

### 4.4 容易遗漏的实现坑

- **cache：** 默认主结果应禁用 `caches_`，否则 patch 后读到构建期快照；另做真实 cache 臂并实现 write-through/失效。删点、entry/medoid 也要处理缓存。
- **medoid/entry：** 同时报“保留旧值”和“重算/追加 medoid”两臂。随机 10% 插入可能掩盖漂移；聚簇新增会放大差异。删除 entry/medoid 必须选 live replacement。
- **workspace：** `set_params/init_workspace` 按当时 `num_points_` 构造 `HashBasedBooleanSet(min(N/10, ef^2))`。它看起来是容量受限 hash set而不是 N-bit bitset；增长后不一定越界，但装载/碰撞行为和容量假设需单测，并在 committed N 或 ef 改变后重建 thread data。所有 neighbor range check 使用原子 committed N 的查询快照。
- **页面共享：** `node_per_page > 1` 时不同节点 patch 同一页会 lost update；锁粒度必须是物理页而非节点。新点落在当前末页时不是简单“追加新页”，可能是补写已有部分页，仍是 RMW。
- **O_DIRECT：** buffered 写与 direct read 混用、对齐、fdatasync 后可见性、同一 fd/不同 fd coherence 均需验证；不要用 OS page cache 偶然行为解释结果。
- 固定 rotator seed、编译 ISA、SSD 型号/预置状态、热/冷 cache、NUMA、PCA artifacts；SIFT128 免 PCA 不能覆盖非 2 次幂/PCA+residual 的更新正确性，至少加一个该路径的小数据测试。
- 0 号幽灵边、重复 ID、self-loop、NaN factors、末页 padding、跨 32-slot block、ID 达 `UINT32_MAX` 前的上限都要做格式级测试。

## 5. 生产化最小闭环

### 5.1 并发与可见性

最低可接受模型是单 writer、多 reader、**页锁 + committed high-water mark**：

1. 为新 ID 准备 raw vector/row，写入新节点所在页；
2. 以递增页号获取 backlink 页锁，页 RMW，避免死锁和共享页 lost update；
3. 更新/失效对应 cache entry；
4. durable 模式下刷数据/WAL；
5. 原子发布 `committed_num_points`，查询只接受其快照以下 ID。

仅用页锁仍不能防进程崩溃产生 torn page。首个生产版本建议 **WAL + physiological full-page image**（before/after 或 redo，含 page id、generation、checksum），恢复时幂等 replay；或者 copy-on-write/shadow page + page table。现有纯算术寻址没有 page table，shadow paging 会把格式改得更大，因而 WAL full-page redo 更贴近当前布局。每页增加 checksum/generation；单个插入跨 R+1 页不能假装天然原子，恢复协议要么 redo 到提交完成，要么用 visibility/事务记录忽略未提交新 ID并修复 backlinks。

### 5.2 meta/file-size 升级

当前加载断言构造时 `num_points_ == metas[0]` 且 `metas[8] == actual file_size`，追加数据后不更新 meta 会拒载；只先更新 meta 又会在崩溃窗口指向未完成页。应引入 native format v2：

- 双 superblock（可用 4KB meta 内 A/B record），字段含 magic、version、generation、committed N、allocated N/pages、expected logical size、geometry、entry、WAL LSN、checksum；
- 数据页/WAL durable 后，原子选择更高 generation 的有效 superblock；恢复时 actual size 可大于 committed logical size，尾部是可截断/可复用的未提交分配，不能一概拒载；小于 committed size 才是损坏；
- `QuantizedGraph` 从 header 获得 N，而不是要求调用者先以完全相同 N 构造；importer/segment manifest 的 count 与 **committed N** 对齐；
- rotator/PCA/geometry 不可在线改变。cache snapshot 标注 generation，过旧则拒载或逐页失效。

这同时解决追加末页、崩溃后 orphan bytes、assert 在 release build 消失等问题；所有格式校验应为显式异常。

### 5.3 DiskCollection API

建议外部 API 保持 label 语义、内部 segment/local ID 隔离：

- `add_batch(vectors, labels, WriteOptions{durability, visibility}) -> WriteResult`；LASER 路由到 active segment，允许微批页合并；
- `delete_batch(labels, DeleteOptions) -> WriteResult`；持久 tombstone/label-version map，幂等；
- `upsert` 明确定义为新 version 插入 + 旧 version tombstone；
- `flush()` 只保证 memtable/active WAL durable，不等于全量 rebuild；`compact()/optimize()` 后台合并并返回 job handle；
- `snapshot/search` 读取 collection generation，跨段按真实可比距离 merge，再按 tombstone/version 过滤。当前 LASER 返回 NaN 距离、按段内 rank tie-break 的行为必须先修，否则 LSM 路线没有语义正确的全局 top-k。

manifest 发布继续复用临时目录、fsync、atomic rename；collection tombstone 也必须版本化并和 manifest generation 一起发布。内存反向表若作为删除加速器，应可由 WAL/图重建而非唯一事实来源；完整 N*R 反向表可能违背磁盘索引内存预算，可用最近更新 delta reverse map + 后台扫描/compaction。

## 6. 相关工作与 LASER 的差异

- **FreshDiskANN** 将实时更新吸收到内存中的临时索引，并周期性做 streaming merge/consolidation，把删除边清理和图修复摊到批次；其关键启示是前后台资源隔离、删除列表、长期 turnover 而非一次增量评测。对 LASER 而言，它支持推荐的混合架构，但 merge 时每条新拓扑边仍需生成 edge code/factors。
- **SPFresh/SPANN** 不是 proximity graph 逐边 patch 的直接模板。其 LIRE 在 posting/partition 边界做局部 split 和邻近分区重分配，避免全局重建；可借鉴的是局部失衡触发器、只移动边界对象、后台资源预算。LASER 的固定度有向图没有天然 posting ownership，不能直接套用 LIRE。
- **IP-DiskANN** 针对单链出边图缺乏 in-neighbor 的删除难题，目标是每次 insert/delete 原地完成并避免 FreshDiskANN 式 batch consolidation。最值得借鉴的是显式维护/定位受影响拓扑、逐删除 repair 和长时间更新稳定性评测；LASER 每改一条边还必须同步改 edge-resident code/factors，页写成本更高。
- **OdinANN**（FAST 2026）强调 billion-scale graph 的 direct insert 与持续稳定性能，可作为“直接原地更新”路线的最新系统对照；其调度/局部修复思想可借鉴，但不能省略 LASER 的 32-block repack、SoA factor patch 和 cache coherence。若论文实验无法获得其实现，应只做机制定性对照，不声称性能胜出。

LASER 的**独特困难**是：拓扑边不是 4-byte ID，还是 `D` bits code + 12 bytes factors 的一致性单元；FastScan 交织令一个 slot 的 code patch 触碰共享 block；固定度无 valid bit；高入度缓存复制整行；随机 backlink 修改以页为最小 I/O/锁/恢复单元。

LASER 的**独特便利**是：raw vector 已随节点落盘，rotator 数据无关且固定，因子逐边局部，无全局 PQ/codebook、训练统计或残差码本需要失效。只要拿到中心行与新邻点，就能生成与静态 builder 数学相同的新边。这正是值得验证的论文命题。

## 7. 建议的判定门槛

在声称“支持更新”前，至少满足：

1. format v2 有显式有效槽、committed N、generation/checksum 和恢复测试；
2. 随机 slot block patch 与整行重算逐字节一致，包含 residual/PCA/duplicate 边界；
3. 单 writer + 并发 search 下无 torn/lost update，cache 开关结果一致；kill -9 注入覆盖每个持久化阶段；
4. SIFT1M 一次 10% 增量之外，完成至少 100% turnover、多顺序、多 seeds；
5. 4a 相对 full RobustPrune 的 recall--QPS、写放大和长期退化被量化；
6. collection 跨段返回可比距离，tombstone/version 过滤有快照语义。

若原型只能先做一件事，优先顺序是：**valid-degree 格式修复 -> 无 cache 单写原地 insert -> 4a/full-prune ablation -> tombstone -> WAL/并发 -> collection hybrid。**

## 参考资料

- [FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for Streaming Similarity Search](https://arxiv.org/abs/2105.09613)
- [SPFresh: Incremental In-Place Update for Billion-Scale Vector Search](https://arxiv.org/abs/2410.14452)
- [IP-DiskANN: In-Place Updates of a Graph Index for Streaming ANN Search](https://arxiv.org/abs/2502.13826)
- [OdinANN: Direct Insert for Consistently Stable Performance in Billion-Scale Graph-Based Vector Search](https://www.usenix.org/conference/fast26/presentation/guo)
