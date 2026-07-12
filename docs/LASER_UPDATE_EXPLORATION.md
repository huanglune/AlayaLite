# LASER 动态更新可行性探索(研究原型 + 实验)

> 课题:AlayaLite 集成的 LASER(on-disk 量化图)当前纯静态,能否支持 insert/delete?
> 结论、机制、原型实现(`include/index/graph/laser/qg/qg_updater.hpp`)、SIFT1M 实验与生产化差距。
> 配套独立设计评审见 `docs/laser_update_design_review.md`(codex 并行评审,结论互相印证)。

## TL;DR

**可以。** LASER 的 edge-resident RaBitQ 量化(每条边的 1-bit 码 + 3 因子)只依赖固定 FHT rotator
和边两端的 raw 向量——没有全局码本可失效,这使得**原地局部更新在数学上闭合**。本原型证明:

1. **插入**:追加新行(纯 id 算术寻址,文件尾追加新页)+ 每个被选邻居一次 4K 页 RMW
   反向边补丁,单写者即可流式插入;`patch == 整行重算` 已被逐字节测试锁定
   (码位与邻居 id 完全一致,因子仅有 -Ofast 跨调用点的 ≤1e-5 相对舍入差)。
2. **删除**:内存 tombstone + 结果过滤(路由仍穿过删除点)零成本可用;拓扑修复
   (splice/repair)是后续工作。
3. 质量/成本数字见下文实验表(SIFT1M,R=64,main_dim=128 免 PCA)。

生产化门槛(格式 valid-degree、页原子性/WAL、并发控制、collection 集成)在文末与
评审文档中列明——那是工程量问题,不是结构可行性问题。

## 1. 为什么此前不支持更新

- 静态管线:`Index.fit` = (可选 PCA) → Vamana 建图 → `QGBuilder::build` 量化落盘;
  没有任何写路径。
- `DiskCollection` 对 LASER 只有 `import_laser_segment`(整段导入),`add_batch` 直接
  throw,tombstone 是 v1 桩。
- 表面上的"硬"障碍:每个节点的 row 里存的是**其全部 `degree_bound` 个邻居的量化码**
  (SymphonyQG 风格边上量化)。插入一个点要给 R 个已有节点加反向边 ⇒ 改 R 个节点的
  码区;直觉上"牵一发动全身"。

## 2. 为什么其实可以:三个结构事实

1. **逐边局部量化**。边 (u→v) 的码 = sign(rot(v)−rot(u)),三个因子
   (`triple_x/factor_dq/factor_vq`)只是 (u,v) 的函数(rabitq.hpp),rotator 是固定
   随机 FHT、数据无关;残差维只以 `||v_r||²` 形式预加进 `triple_x`。
   ⇒ 加/换一条边只需要边两端 raw 向量,而 raw 向量就存在各自 row 的头部。
2. **FastScan 打包是无损重排**。32 邻居一块的交织(`pack_codes`)由字节反序、nibble
   交换、`kPerm0` 置换组成,完全可逆(`unpack_codes_block`,往返测试锁定)。
   ⇒ 换一个 slot = 解包该 32-slot 块 → 换码 → 重打包,块内其它 slot 不动。
3. **寻址纯 id 算术**。`page_offset(id) = 4096 + page_size·(id/npp)`,无分配器、无
   page table。⇒ 新节点即文件尾新页(npp>1 时是末页 RMW)。

## 3. 原型设计(qg_updater.hpp)

插入 = FreshDiskANN 语义在 LASER 布局上的实例化:

```
insert(x):
  tvec = PCA(x)                    # 与查询同路径;SIFT 配置为恒等
  pool = beam_search(tvec, ef)     # 同步 pread 版磁盘搜索;捕获每个展开节点的
                                   #   {id, 精确距离, 整行字节}(raw 向量随行免费得到)
  sel  = alpha_robust_prune(pool)  # 精确距离 + 捕获的 raw 向量,纯内存,α=1.2
  row  = assemble_row(tvec, sel)   # 与 builder 完全同构(复用 rabitq_codes)
  append_page(new_id, row)
  for v in sel:                    # 反向边:每邻居一次 4K 页 RMW
    patch_reverse_edge(v, new_id)
  if 没有任何反向边成功: 强制在最近邻居上驱逐一条(可达性保底)
```

反向边四臂(实验对照):

| 臂 | 行为 | 每反向边额外 I/O |
|---|---|---|
| `none` | 不加反向边(下界对照,新点几乎不可达) | 0 |
| `evict` | 先填幽灵空槽;否则以 v 自身为查询跑一遍 FastScan 估计边长,新边更短则驱逐最远边("nearest-only replacement") | 1 读 + 1 写 |
| `alpha` | evict + **免费 α 检验**:v 的现有邻居若已在本次搜索捕获池中,用其 raw 向量做 α-遮蔽检查,被遮蔽则放弃加边(零额外 I/O 的近似 RobustPrune) | 1 读 + 1 写 |
| `full` | 读回 v 全部活邻居 raw 向量,完整 α-RobustPrune,整行重写(质量参照上界) | ~R 读 + 1 写 |

关键实现点:

- **幽灵槽**:v1 行格式无 valid-degree;Vamana 欠满节点尾部 slot 是 0 填充
  (id=0、码=0、因子=0)。判空用三条件联合(id、三因子、该 slot 全部码位均为零),
  真实的"到 0 号点的边"不会误判(除非严格重复向量,那条边本身可弃)。这是启发式,
  格式 v2 应加显式 degree —— 见评审文档 §1.3。
- **因子位级一致**:单边因子不手写公式,而是调 1 行矩阵的 `rabitq_codes`,与 builder
  的 Eigen 归约逐位对齐(-Ofast 下跨调用点仍可能有最后一位舍入差,单测以码位/ID
  逐字节 + 因子 1e-5 相对误差锁定)。
- **写路径**:buffered `pwrite` 整页 RMW;搜索读路径是 O_DIRECT libaio,Linux 语义下
  direct read 会先回写脏页缓存范围,`finalize()` 再统一 fsync + 重写 meta sector
  (num_points、file size),重载校验通过。
- **删除**:`QuantizedGraph::set_result_filter(&tombstones)`——结果过滤但路由穿透
  (三行侵入改动)。修复(splice/图整形)未实现。

## 4. 正确性证据(单元测试,tests/laser/qg/test_qg_updater_unit.cpp,5/5 通过)

1. `UnpackRoundTrip`:pack→unpack 恒等;补丁一个 slot 后重打包 == 直接打包修改后的码。
2. `EdgePayloadMatchesRabitqCodes`:单边 payload 与 builder 内核输出位级一致。
3. `AssembleRowMatchesBuilder`:对静态构建的真实索引逐行重组,`memcmp == 0`。
4. `InsertPatchTombstoneEndToEnd`(2000 点真实小索引,npp=3 共享页路径):
   插入 64 点 → 被补丁行与整行重算等价;重载后 ≥90% 插入点 top1 命中自身;
   tombstone 后结果零泄漏。
5. `GhostSlotDetection`:真实边/幽灵槽全部判定正确。

## 5. SIFT1M 实验(E0/E1/E2)

配置:SIFT1M(128 维,2 的幂 ⇒ main_dim=128、免 PCA、残差 0),R=64,L=200,
ef_indexing=200;900k 基础索引 53s 构建、1M 87s(48 线程)。评估:10k 查询,
recall@10,GT 深度 100,masked recall(live 集上重取 top-10 真值);QPS 为
16 线程 batch_search 三轮最优;beam=16;缓存关闭(dram_budget=0)。
插入:单写者,ef_insert=100,α=1.2。

### E0 静态基线(参照)

| ef | 1M 全建 recall@10 | QPS(16T) | 900k 全建 recall@10(masked) |
|---|---|---|---|
| 60 | 0.9713 | 10715 | 0.9712 |
| 100 | 0.9905 | 8865 | 0.9909 |
| 200 | 0.9978 | 6249 | 0.9977 |
| 300 | 0.9987 | 4858 | 0.9986 |

构建成本:900k 53s / 1M 87s(48 线程,Vamana+量化全程)。

### E1 插入(900k 建 + 单写者流式插入 100k,对照 1M 静态全建)

整体 recall@10(全量 1M GT):

| ef | 静态 1M | none | evict | alpha | full |
|---|---|---|---|---|---|
| 60 | 0.9713 | 0.8793 | **0.9667** | 0.9675 | 0.9649 |
| 100 | 0.9905 | 0.8952 | **0.9883** | 0.9885 | 0.9891 |
| 200 | 0.9978 | 0.9003 | **0.9971** | 0.9971 | 0.9971 |
| 300 | 0.9987 | 0.9010 | **0.9983** | 0.9983 | 0.9985 |

仅统计落在插入段(id ≥ 900k,9824 条 GT 条目)的 recall:

| ef | 静态 1M(上界) | none | evict | alpha | full |
|---|---|---|---|---|---|
| 60 | 0.9565 | 0.0 | **0.9506** | 0.9348 | 0.9421 |
| 100 | 0.9864 | 0.0 | **0.9800** | 0.9728 | 0.9778 |
| 200 | 0.9983 | 0.0 | **0.9944** | 0.9927 | 0.9945 |
| 300 | 0.9994 | 0.0 | **0.9969** | 0.9963 | 0.9975 |

插入成本(单写者,页缓存热;4K 页 I/O 逻辑计数):

| 臂 | inserts/s | 搜索读/插入 | 补丁读/插入 | 页写/插入 | 反向边构成 |
|---|---|---|---|---|---|
| none | 3441 | 81 | 0 | 1 | — |
| evict | 1809 | 81 | 40.0 | 35.6 | 19.6 填空槽 + 15.0 驱逐 + 5.4 skip |
| alpha | 1841 | 81 | 40.3 | 28.9 | 16.0 填空槽 + 11.9 驱逐 + 8.2 α-skip + 4.2 skip |
| full | 205 | 82 | **2303** | 41.8 | 40.8 整行重算 |

**读数**:

1. `none` 臂 new_recall 恒为 0 —— 只追加行、不维护反向边,插入点完全不可达。
   反向边补丁是唯一命脉;这就是"把 LASER 当普通索引追加"不行的实证。
2. **廉价结构补丁(evict)与 9× 昂贵的完整 RobustPrune(full)recall 曲线重合**
   (差异 ≤0.2pp,在噪声内;full 在插入点上甚至略低——过度剪枝压低了新点入度)。
   单次 10% 插入时,填空槽 + FastScan 估边长驱逐最远,就足以逼平质量上界。
3. 流式插入后的索引离静态全建只差:低 ef 0.46pp / ef≥150 约 0.1pp;插入点
   自身与"生而在库"的差距 0.6pp(ef60)→ 0.25pp(ef300)。
4. 每插入总 I/O ≈ 157×4K(evict),其中搜索占一半;QPS 侧无衰退。

### E2 删除(1M 建,tombstone 10%,对照 900k 静态重建)

| ef | tombstone(路由穿透+结果过滤) | 900k 重建(理想删除) |
|---|---|---|
| 60 | 0.9660 | 0.9712 |
| 100 | 0.9887 | 0.9909 |
| 200 | 0.9974 | 0.9977 |
| 300 | 0.9986 | 0.9986 |

10% 懒删除几乎免费:低 ef 差 0.3-0.5pp,ef≥200 与完全重建重合,QPS 不变。
(高删除比例 / 持续 churn 下的空间与路由 ballast 需要 consolidation,见 E3。)

### E1b 追加臂:headroom(利用定宽行的免费更新余量)

LASER 行固定 `degree_bound` 槽、字节先付;Vamana 用 R=56 建、degree_bound=64
⇒ 每行天然 ≥8 个空槽,反向边优先填空、零驱逐零质量损失。

| ef | hr 静态 900k(R56) | hr 插入后整体 | hr 插入点 new_recall | 静态 1M 上界(new) |
|---|---|---|---|---|
| 60 | 0.9728 | 0.9695 | **0.9618** | 0.9565 |
| 100 | 0.9907 | 0.9894 | **0.9837** | 0.9864 |
| 200 | 0.9974 | 0.9973 | **0.9962** | 0.9983 |
| 300 | 0.9985 | 0.9985 | **0.9981** | 0.9994 |

- R56 欠满建图本身不掉点(0.9728@60 ≥ R64 的 0.9712),QPS 还略高;
- 插入后整体离静态 1M 只差 0.18pp(ef60),**插入点 recall 在低 ef 超过静态上界**
  (0.9618 vs 0.9565)——新点通过填空槽获得高入度,而旧图几乎不被驱逐破坏
  (每插入 34.2 次填空槽 vs 仅 3.9 次驱逐);插入后 QPS 为所有臂最高(12.9k@60)。
- 这是定宽行布局特有的红利:普通变长度索引"少建边"=省存储但伤质量,LASER 的
  空槽字节反正已付,欠满建图即免费预留更新余量。

### E3 持续 churn(滑动窗口,原库 100% 换血)

500k 基础,10 轮 ×(tombstone 最老 50k + 插入新 50k),ef_eval=100,evict/alpha 两臂。
注意本实验是**无 consolidation 的最坏情形**:tombstone 只做结果过滤、从不清除,
10 轮后路由图里一半节点是墓碑 ballast;衰减同时来自边质量侵蚀与墓碑稀释。

| 轮(换血%) | 0 | 2 (20%) | 4 (40%) | 6 (60%) | 8 (80%) | 10 (100%) |
|---|---|---|---|---|---|---|
| evict | 0.9919 | 0.9815 | 0.9661 | 0.9494 | 0.9242 | 0.8887 |
| alpha | 0.9916 | 0.9797 | 0.9618 | 0.9380 | 0.9088 | 0.8636 |

- 每轮 −0.5→−2.0pp,**缓降且加速,不趋平**:墓碑从不清除 ⇒ 活节点的出边越来越多
  指向死点(有效出度衰减)、搜索 beam 浪费在死区;这与 DiskANN 侧 fig13 战役的
  "入度年龄衰减"是同一类机制,解法也相同——周期 consolidation(清墓碑 + splice +
  重剪枝),其行级原语(`full_reverse_recompute`)本原型已具备,缺的只是调度。
- alpha 臂在 churn 下**更差**(−12.8pp vs evict −10.3pp):α 检验拒掉 ~20% 反向边
  ⇒ 新节点入度更低,在老节点持续死亡的环境里雪上加霜。单次增量里无所谓的
  多样性保险,在持续换血下是负资产——**churn 场景要的是入度,不是保守剪枝**。
- 插入吞吐在 churn 中稳定(2.1k→2.8k inserts/s,墓碑使剪枝池变小反而略快)。

## 6. 结论

**回答课题:LASER 能支持更新,且不是靠通用方法,而是靠它自己的结构。**

1. **结构可行性(定性)**:逐边局部量化(无全局码本)+ FastScan 可逆置换 + 纯 id
   算术寻址,三者使"追加行 + 页级反向边补丁"成为闭合操作;patch 与整行重算逐字节
   等价已被单测锁定。相较之下,PQ 系磁盘索引(DiskANN 等)插入新点要么忍受码本
   漂移、要么重训——LASER 的 RaBitQ 结构在这点上**天生适合更新**。
2. **质量(单次 10% 增量)**:廉价补丁(每插入 ~157 次 4K I/O,1.8k inserts/s 单写者)
   与 9× 成本的完整 RobustPrune 打平,离静态全建 0.1-0.5pp;headroom 欠满建图后,
   插入点 recall 甚至**超过**静态上界,整体差距缩到 0.18pp。
3. **删除**:10% 懒删除(路由穿透 + 结果过滤)几乎免费。
4. **持续 churn 是真正的边界**:无 consolidation 时 recall 每轮缓降且加速
   (100% 换血后 −10.3pp,见 E3 表),与 DiskANN 侧 fig13 的经验同类——流式更新的
   长期质量必须靠周期性整形(清墓碑/splice/合并重建)兜底。这与"能否支持更新"是
   两个问题:更新原语本身已经闭合,整形是在原语之上的调度策略
   (`full_reverse_recompute` 即整行整形原语)。附带教训:α-保守剪枝在 churn 下
   是负资产(alpha 臂 −12.8pp 更差),持续换血场景要优先保证新点入度。

### 生产化差距(按优先序;详见 laser_update_design_review.md §5-7)

1. **格式 v2**:显式 valid-degree(替代幽灵槽启发式)、双 superblock(committed N /
   generation / checksum,取代"meta 断言拒载")、追加页崩溃一致性。
2. **并发**:单写者 + 页锁 + committed high-water mark;O_DIRECT 读与缓冲写的
   混合路径要么统一 O_DIRECT 整页写、要么 WAL full-page redo。
3. **删除修复**:delta 反向表 + splice/consolidation(原语已有,缺调度);
   墓碑持久化。
4. **DiskCollection 集成**:推荐混合路线——小增量进可写 active segment 原地
   patch(本原型),阈值触发后台合并成不可变段;跨段 top-k 需先修 LASER 段
   NaN 距离问题。
5. **PCA/medoid 陈旧**:固定线性映射保证正确性,分布漂移只降质量;medoid/入口
   可周期重算(本实验静态入口在 10% 增量下无可见损失)。

## 7. v2:并行插入 + consolidation(回应"性能/效果都差")

v1 原型的两处硬伤及 v2 修复(设计经 codex 并发评审修正:批次三阶段发布替代票据自旋、
页 seqlock 版本号消除撕页读、splice 用 FastScan 召回 + raw 精排):

### 7.1 性能:多写者批式插入

批次三阶段:并行(搜索/构行/反向边补丁,页锁保护写)→ barrier → 每批发布一次
committed 水位;批内搜索一律用批前快照(mini-batch 语义)。搜索读无锁,靠每页
seqlock 版本校验重试。

| 配置 | inserts/s | 页写/插入 | 对 v1 串行加速 |
|---|---|---|---|
| v1 串行(evict) | 1809 | 35.6 | 1× |
| T=32(evict,batch 4096) | 7203 | 35.8 | 4.0× |
| T=64(evict) | 7329 | 35.8 | 4.05× |
| T=64(evict,捕获裁剪后) | 8015 | 35.8 | 4.4× |
| **T=64(none 臂 = 每插入 1 写,搜索侧上限)** | **17654** | 1.0 | 9.8× |
| churn 内(500k 图,T=32) | 10.6k–12.9k | — | — |

质量零损失:T=32 并行后 recall 0.9661/0.9879/0.9969(@60/100/200)与串行
0.9667/0.9883/0.9971 重合;headroom+并行 0.9688@60、插入点 0.9583(仍超静态上界)。

**扩展性瓶颈已定位并量化在写路径**:evict 臂 T=32/T=64/裁剪后总页写速率同为
~260–287k pwrites/s(平台期),而把写降到 1/插入的 none 臂立刻到 17.7k inserts/s
——ext 系文件系统上 buffered `pwrite` 走排他 inode 锁,单文件多写者天然串行。
工程出路(未实现,已量化收益上限):(a) 对齐 O_DIRECT 写(ext4/XFS 走共享
ilock,可并行,但牺牲读侧页缓存);(b) 用户态写合并(批内同页补丁合并一次写,
热点页收益大);(c) 分段文件。按 none 臂上限,写路径解锁后 evict 臂应到
~15k+ inserts/s(64T)。

### 7.2 效果:consolidation 拉平 churn 曲线

`consolidate(threads, r_target)`:并行扫活行,死邻居槽 = 用死节点自己的行做
FastScan 召回(码以死点为中心,估计量仍是 ||u−n||,零额外 I/O)→ top-4 读 raw
精排 → patch;无候选或已达 r_target 则清零成幽灵空槽(**headroom 再生**)。
r_target=56 保持 8 槽更新余量,不回填满 64。每 churn 轮(删 50k 后)执行一次,
32 线程 4–5 秒。

| 轮(换血%) | 0 | 2 | 4 | 6 | 8 | 10 (100%) |
|---|---|---|---|---|---|---|
| 无整形(v1) | 0.9919 | 0.9815 | 0.9661 | 0.9494 | 0.9242 | 0.8887 |
| **+consolidation** | 0.9918 | 0.9873 | 0.9804 | 0.9802 | 0.9752 | **0.9748** |

- 无整形曲线加速下坠(尾轮 −2.0pp);consolidation 后**第 4 轮起趋平**
  (尾 6 轮共 −0.6pp 且减速,第 5→6 轮回升)——有界收敛稳态,+8.6pp@100% 换血。
  与 DiskANN fig13/garden 战役"churn 自平衡 + 整形提平台"结论同构。
- 累计填空槽 13.8M vs 驱逐 2.0M:consolidation 清出的空槽被后续插入吸收,
  purge↔headroom 闭环成立。
- 不变量测试:consolidate 后所有活行零死边引用(单测覆盖)。

### 实验局限

单数据集(SIFT1M)、单次序(自然顺序)、单 seed、128 维免 PCA 路径;残差维
(main_dim < dim)与 npp>1 大规模路径仅有单测覆盖;并发读写、崩溃注入未做。
这些是论文级 battery 的下一步,不改变可行性结论。

## 8. 参考 Yi 的对照与下一步计划

Yi(原版 `~/workspace/alaya-dev/Yi`,SPDK/io_uring 全异步)与其在 AlayaLite 的
迁移版(`include/index/graph/diskann/`,经 fig13 复现与 delete-repair 战役检验)
是同一套 in-place 更新协议的两个实现。读码结论如下。

### 8.1 Yi 更新架构画像(读码证据)

- **纯 in-place,无 fresh 层**:Yi 没有 FreshDiskANN 式内存增量图与 StreamingMerger;
  插入/删除直接改磁盘图(经 buffer pool)。更新任务是协程
  (`QueryContext::co_update` / `co_update_ipdiskann`),轮次 = 批量懒删 → 批量插入
  → 并发搜索评估(`Yi/src/runner/update_runner.cpp`)。
- **写路径 = 用户态脏页缓存 + 延迟写回**:更新在 buffer pool 页内 RMW 并标脏,
  热路径零 pwrite;写回移出更新路径(迁移版 `disk_page_io.hpp` 明确注释
  "Yi defers write-back out of the update path"),flush 时每脏页只写一次,
  O_DIRECT 对齐写(独立 `O_DIRECT|O_RDWR` fd,绕开 buffered 写的排他 inode 锁)。
- **反向边暂存(Yi `_inserted_edges` / 迁移版 `StagedEdges`,
  diskann_index.hpp:1574)**:插入把 `选中邻居 → 新槽` 的回链塞进按邻居分桶的暂存表,
  哪个 reconnect 任务先到该邻居就 drain 全部暂存边,一次 RMW 落多条回链——
  天然去重 + 图层面的写合并。
- **批内无 barrier**:端到端插入协程(搜索→分配→写→publish→暂存→reconnect),
  每 chunk 一个 when_all;并发靠每节点互斥锁 + 暗槽协议。迁移版注释明确记录:
  早期四段 barrier 相位流水线让 32 线程比 8 线程还慢(与我们 v2 的三阶段 barrier
  是同一族设计,Yi 的经验是 barrier 越少扩展性越好)。外部 API 层面更新调用串行
  (`update_serial_mutex_`),并行度全部在批内。
- **槽位生命周期(设计 D5,`slot_allocator.hpp`)**:LIFO free-list + tombstone
  位图一体;`alloc()` 复用最近释放槽,**暗至 publish**(数据落盘/入缓存前搜索
  不可见,防复用槽旧字节在新 label 下泄漏);`free()` 入 free-list 并置 tombstone;
  `save()/load()` 一个文件持久化完整分配状态。
- **删除 = 懒删 + 二跳旁路**:`remove()` 缓存死节点邻居表(Yi 用容量 4% 的 LRU,
  迁移版 `DiskUpdateContext::removed_node_nbrs_`),搜索遇 tombstone 经缓存二跳
  绕行(每死邻居最多取 5 个活二跳候选,重连池 heap 上限 degree+32);修图推迟到
  下次插入触达或 safety-net(tombstone 比例 ≥5% 且久无插入时主动重连,
  `disk_update_context.hpp:35`)。Yi 另有更激进的 IP-DiskANN 删除变体
  `co_delete_ipdiskann`:以被删向量搜索候选,对全部受影响活点 fan-out 修复任务
  (活旧边 + 搜索候选最近 3 个再剪枝)——这是 Yi 里最接近"删除即整形"的路径。
- **写回调度细节**(codex 读 Yi 原版证实):更新只改内存页版本号,脏页进 dirty
  LRU;worker 事件循环在队列深度允许时逐页写回,脏页积压超 30k 一次调度 32 页;
  io_uring 后端 `O_RDWR|O_DIRECT`、每线程深度 256 的 ring,**未用** registered
  buffer/SQPOLL——Yi 的收益来自"同页多次修改合成一次设备写",不是 io_uring 高级特性。
- **崩溃一致性**:Yi 与迁移版都没有 WAL;研究系统定位,与我们相同。

### 8.2 与 LASER 原型逐项对照

| 机制 | Yi/迁移版 diskann | LASER v2 现状 | 差距 |
|---|---|---|---|
| 写路径 | 脏页缓存 RMW + flush 每页一次 + O_DIRECT | 每补丁立即 buffered pwrite(35.8 写/插入),287k pwrites/s inode 锁墙 | **主要吞吐差距** |
| 反向边 | StagedEdges 按邻居聚合,一次 RMW 落多条 | 每条回链独立 解包→改槽→打包→写 | 写放大 + CPU 放大 |
| 批内同步 | 无 barrier,节点锁 + 暗槽 | 三阶段 barrier + 页锁 + seqlock | 扩展性上限 |
| 槽位复用 | free-list+tombstone 一体,暗至 publish,可持久化 | append-only + 幽灵槽回填,无复用、不持久化 | 空间回收缺失 |
| 删除路由 | 懒删 + 二跳旁路缓存 | tombstone 结果过滤 + 路由穿透(行字节还在,免费) | 我们更简单且够用(购槽复用前) |
| 整形 | 插入驱动 reconnect + safety-net + gardening(P3 战役:入度刷新砍 40% 衰减) | 手动 consolidate(purge+splice) | 触发策略 + 质量杠杆 |
| 量化 | PQ 码在 RAM(`encode_pq_slot`),修图不碰量化 | 码在行内逐边,回链 RMW 要重打包 | 结构差异:LASER 免 RAM 码表,代价可被页缓存+暂存吸收 |

关键判断:**Yi 用纯 in-place 跑通了持续 churn,没有 fresh 层**;我们的 v2 数据
(0.9748 平台)也说明 in-place 质量够。fresh 层(内存增量图+双路搜索+后台 merge)
只在要求 >20k/s 持续插入或插入 P99 延迟敏感时才值得,现阶段不做。

### 8.3 计划

**P0 写路径改造(照抄 Yi 两件套,目标 evict 臂 8k → 12–15k/s 或 ≥none 上限的 70%)**
1. **P0.1 O_DIRECT 归因臂(2–4 人日)**:最小改动——更新 fd 换
   `O_DIRECT|O_RDWR` + 对齐页 buffer,其余语义不变。若 T=32→64 不再同卡,
   证实 287k 墙 = buffered inode 锁;若无改善,先 lockstat/perf 重新归因再动大刀。
   附带修正一个现存隐患:LASER 搜索读本来就是 O_DIRECT
   (`utils/aligned_file_reader.hpp:359`),而更新 fd 是 buffered O_RDWR
   (`qg_updater.hpp:253`)——buffered 写对 O_DIRECT 读的可见性靠内核先回写脏区,
   有隐藏回写税,两侧统一 direct 才干净(迁移版 `disk_page_io.hpp` 头注同一结论)。
2. **P0.2 分片脏页缓存(1–1.5 人周)**:抽取 `diskann/disk_page_cache.hpp` 的
   分片 LRU/dirty 语义,QGUpdater 所有行写改为缓存页内 RMW+标脏,批末
   `flush_dirty_pages()`(每脏页一次 O_DIRECT 对齐写)→ 再 `publish()`。
   flush 先行保证搜索只见已落盘状态,搜索侧无需查更新缓存;若后续把 flush
   进一步解耦(跨批合并/后台写回),读路径须加页级 overlay snapshot(不抄
   diskann 的整节点 NodeCache override——LASER 行大,按页 overlay 才不产生双份行)。
   同时上 StagedEdges 反向边暂存:按邻居聚合回链,drain 时该邻居行**解包一次、
   改 k 槽、打包一次、写一次**(现在是 k 次全套)。
3. 验证:none/evict 臂 T=1/8/16/32/64 吞吐;三层写计数
   `逻辑页更新 / 唯一脏页 / 物理写`(收益不足时用唯一页比率解释,而不是盲调线程);
   "同页 N 次 patch 只 1 次 flush"、并发 patch 不丢更新、cache 开关字节等价单测;
   单次 10% 插入 recall 与 v2 持平;ctest 全绿。

**P1 质量闭环(移植 delete-repair 战役判决,目标稳态 0.9748 → ≥0.985)**
1. 更深维护搜索:再插入式刷新用独立 `ef_maintenance ∈ {100,200,400}` 消融
   (oracle 判决:残余是边质量欠账,深搜是杠杆;garden 侧 128→200 同理)。
   注意仅调大 `splice_rerank` 无效——splice 候选只来自死点旧邻域,改善的是
   候选内精排,扩不了候选来源;深 beam 才是扩源。
2. gardening:RAM 入度 delta 计数器(uint16/槽,patch/zero/插入时增减,
   checkpoint 可扫 id SoA 重建,不必持久化)+ 每轮对最低入度分位(bottom 5%)
   + 高年龄/高驱逐行做再插入式深搜刷新、整行 `assemble_row` 重写 +
   densification(欠满行填满,headroom 臂已证明满行更优)。**反向泵要有预算**
   (B∈{0,4,8},优先填幽灵槽):LASER 每条回链要重算 1-bit 码+三因子,
   不能像 DiskANN 无预算 fan-out;同页泵靠 P0 页缓存合并。diskann 判决:
   入度定向刷新砍 ~40% 老化衰减。
3. 触发策略:tombstone 比例 + ops-since-insert 的 safety-net(照抄
   `needs_safety_net_reconnect` 语义),替代手动每轮调用。
4. 验证:六臂消融(purge-only / 现 splice / +浅刷 / +深刷 / 深刷+泵 /
   **等预算随机刷新对照**),≥3 seeds,跑到 100–300% 换血;同时报结构指标
   (入度 p1/p5/中位、entry 可达率、活出度、边年龄、新边存活率)。
   长稳态复用 g03 协议(稳态回收 + 永不删固定队列)。
   门槛:recall@100 ≥0.985 且维护摊销 ≤前台更新 CPU/IO 的 20%。
   工程量:观测+消融 4–6 人日(可与 P0.2 并行),胜出臂做成带预算的
   正式 scheduler 再 1–1.5 人周。

**P1.5 插入流水线协程化(1–2 人周,P0 判决后,可与 P1 scheduler 并行)**
- 动机:协程基建在仓已全(C++20 全项目、coro 库、`storage/io/uring_reactor.hpp`,
  diskann 更新路径已跑在其上);none 臂 17.7k/s 上限含可观 IO 等待,libaio 每线程
  一个 io context(MAX_EVENTS=1024,全机 ≤64 线程,实测撞过两次);协程 + reactor
  少数 ring 服务几百并发,并发数与线程数解耦。
- 做法:给 qg 插入搜索做异步变体,照 `diskann/beam_search_async.hpp` 的契约
  (cache 命中在挂起前内联吸收、锁不跨挂起点、结果与同步版逐字节一致);bench 的
  OpenMP parallel-for 换 `coro::thread_pool` + `when_all`
  (`batch_insert_locked_with_pool` 现成模式);搜索 peek P0.2 分片缓存并按版本
  协议回填——分片缓存升级为 Yi 式统一 buffer pool。
- 边界:协程不解决写路径 inode 锁(P0 的事),不加速 FastScan(CPU bound)。
- 验证:none/evict 吞吐 × 线程数扫描,并发>线程数的扩展性,io context 计数 O(1)。

**P2 槽位复用与格式 v2(空间回收,长 churn 文件不再无限涨;1–2 人周)**
1. 移植 SlotAllocator:purge 清净入边的死槽进 free-list(入度计数器守门:入度>0
   不得复用),插入优先复用,暗至 publish;分配状态(free-list+tombstone)随
   finalize 持久化(照抄 `slot_allocator.hpp` 的 save/load 格式)。
2. 格式 v2:行头 valid-degree(或槽位位图)替代幽灵槽三条件启发式;A/B 双
   superblock 存 geometry / allocated-committed N / generation / checksum,给后续
   WAL 提供锚点;v1 只读兼容 + 离线迁移工具。
3. 验证:长 churn 下文件大小有界;复用槽无旧字节泄漏、id=0 真边、欠满行、
   npp>1、旧格式兼容单测。

**P3 生产化(按需)**
- WAL/崩溃一致性(2–3 人周):full-page redo WAL(batch 级 begin / 页
  after-image+checksum / commit,先 durable WAL 再写数据页,A/B superblock 翻转
  提交;**不用 shadow page**——它会把纯 id 算术寻址逼成 page table)。
  tombstone 位图与 meta 纳入同一事务。Yi 也没做,原型阶段标注 non-durable 不阻塞。
- io_uring 异步 flush(可选,5–8 人日):页缓存合并之后的写回引擎,有界 QD +
  固定对齐 buffer 池,隐藏 O_DIRECT 写延迟;profile 证明有收益再上 registered
  buffer(Yi 自己也没用)。
- collection 集成:活跃可更新 LASER 段 + 持久 label 映射/tombstone + 后台把老段
  重建为不可变段、原子发布 manifest(周期全量重建作为质量/空间兜底)。

### 8.4 明确不做(及重启条件)

- **fresh 内存增量层**:Yi 本身就是纯 in-place;引入双图要付出双搜索路径合并、
  merge 时逐边重生成码/因子、manifest 发布、崩溃恢复四份复杂度,而且它不消灭
  LASER 边码写,只把写延后成更大的 merge。设为触发式备选,满足任一才立项:
  (a) P0 后 evict 仍 <10k 且瓶颈已被证明是设备随机 IOPS 而非软件;(b) 更新可见
  延迟 + fsync 成本使按批 in-place 提交不可接受;(c) collection 本就要做多段
  merge,可复用其框架;(d) 更新分布高度局部而页缓存合并率仍低。即使触发,也做
  在 collection 层(内存 delta + 不可变 base 段),不把 QGUpdater 改成双图。
- **分段文件**:与纯 id 算术寻址冲突,留给 collection 层的段管理,不进单段格式。
- **SPDK/全协程调度移植**:Yi 的 io_uring reactor 读路径迁移版已有,LASER 搜索有
  自己的 IO 栈;写侧只需要 O_DIRECT flush(+可选 io_uring 写回),不需要 reactor。

### 8.5 执行顺序与判决点

1. P0.1 归因臂先行:O_DIRECT-only 若 T=32→64 仍同卡 ~287k 写/s,暂停页缓存
   大改,先重新归因;否则立即进 P0.2。
2. P0.2(写路径)与 P1 观测/消融并行——写路径先解决,gardening 的反向泵成本
   结论才不被旧墙污染。P1.5 协程化排在 P0 判决之后:若 P0 后吞吐已贴 none 上限,
   它是抬上限的下一根杠杆。
3. P0/P1 末 go/no-go:吞吐 ≥12k 且深刷稳态 ≥0.985 → 锁定纯 in-place,进入
   P2 格式/复用与 P3 WAL;未达标按唯一页 IOPS / 结构指标分别定位,不直接跳 fresh。

> 详细版(逐文件/行号证据、Yi 并发协议逐段拆解、各方案风险表、验证门槛全文)
> 见 `docs/LASER_UPDATE_NEXT_PLAN_YI.md`——codex 独立调研终报,与本节交叉核对
> 后一致;本节采纳了其两处修正:P0 拆归因臂+页缓存两步、工程量估计以其为准。

## 9. P0 执行记录

### 9.1 P0.1 O_DIRECT 归因臂:判决与意外(2026-07-12)

实现:双 fd——读保持 buffered(OS 页缓存伺服 ~82 页读/插入),写路由到独立
`O_RDWR|O_DIRECT` fd(`UpdateParams::direct_io`,bench `--direct`);页缓冲全部
换 4K 对齐 `AlignedBuf`,非对齐公共调用者走线程本地 bounce。seqlock 版本序
(写完 pwrite 返回后才翻偶,DIO 写自带页缓存失效)保证 buffered 读 + DIO 写
进程内一致。质量 eval 与 buffered 完全重合(0.965@60 / 0.988@100 / 0.997@200)。

吞吐(SIFT1M 900k+100k,evict 臂 35.8 页写/插入,NVMe ext4):

| 配置 | inserts/s |
|---|---|
| buffered(对照,当日复现) | 7439 (64T) |
| 每补丁同步 DIO,克隆未 sync | 2030 (64T) / 2269 (32T) |
| 每补丁同步 DIO,克隆后 sync | 5934 (64T) |
| none 臂 DIO(1 追加写/插入) | **18559** (64T) |

**判决:每补丁同步 O_DIRECT 写不可行,且推翻了"inode 锁是主敌"的原始归因。**
1. 未 sync 坍塌 3.7×:DIO 写命中脏页缓存区间强制同步回写
   (`filemap_write_and_wait_range`)——克隆刚 cp 完 3.4GB 全脏。教训:凡 DIO
   写实验,克隆后必须 `sync`。
2. sync 后仍输 buffered ~20%:同步设备往返(~25-100µs)压在页锁临界区与插入
   关键路径里,替代的是 buffered 的 ~µs 级页缓存 memcpy——**内核页缓存本来就是
   写回缓存**,287k pwrites/s 平台是 buffered 写路径(页拷贝+ilock+脏页记账)的
   软件成本,不是设备。
3. 真正的解法与 Yi 一致:P0.2 用户态脏页缓存——热路径零 syscall、批内同页
   RMW 去重、批末受控 flush(DIO fd 留作 flush 通道)。none 臂 18.6k/s 说明
   搜索侧上限还在,追加写(含 DIO)无碍。

默认值已回翻 buffered(`direct_io=false`);P0.2 进行中(实现已按简报派发 codex,
DIO flush 模式选型另有并行诊断)。

### 9.2 P0.2 → P0.2b:从批末写回到驻留 buffer pool(2026-07-12)

**第二判决:批末写回两种口味都输,写次数才是敌人。** P0.2 初版(批内脏页缓存、
批末每唯一页一次写回、flush→publish)写次数 35.8→28.3/插入,但吞吐反降至
3.4k/s:相位计时显示 flush 21s(buffered+sync_file_range 只有 134k 页/s——
批末回写与下一批打架)、DIO flush 更慢(70k/s,同步 completion)。inline 基线
的 287k/s 本质是"写与搜索重叠"的产物,拆成独立相位后怎么写都亏。

**解法 = Yi 的 buffer pool 终局形态(P0.2b)**:缓存跨批驻留
(`cache_cap_pages` 水位,默认 4GB),回链目标高度偏斜(hub 行),跨批合并把
物理写压到 **8.6/插入(−4.15×)**,且全部推迟到水位驱逐/consolidate/finalize;
批内 flush=0。搜索改为**穿透池读**(`read_node_page` overlay,命中 shard 锁内
memcpy),缓存页变异纳入既有每页 seqlock(改前奇改后偶),批内安全性由快照守卫
(`cur/nb >= snapshot`)保证;外部进程 reader 的边界仍是 `finalize()+fsync`。

**第三判决:第二个串行点是 glibc 堆伸缩,不是锁也不是设备。** 池化后 drain
(补丁应用)阶段成为主项且 T32≈T64 不随线程扩展——存在 ~330k patches/s 全局顶
(v2 inline 的 266k patches/s 是同一个顶,当时被搜索重叠掩盖)。perf 归因:
每条补丁数次短命对齐分配(Eigen RowMatrix、QGQuery 的 lut/rotate/quantize
scratch、patch_slot 的 pad/rot/bins)→ glibc memalign 反复 grow/shrink 堆 →
`mprotect` 压 `mmap_sem`(内核全局)。修复:五处 thread_local scratch +
`QGQuery::rebind()` 每线程复用 + `EdgePayload` 按 thread_local 引用返回 +
bench 侧 `mallopt(M_TRIM_THRESHOLD/M_TOP_PAD/M_MMAP_THRESHOLD)`。
drain 10.9s→4.8s(64T),恢复线程扩展。

**P0 终局数字**(SIFT1M 900k+100k,evict,batch=4096):

| 配置 | inserts/s | 物理写/插入 | 备注 |
|---|---|---|---|
| inline 立即写(基线) | 7439 (64T) | 35.8 | 写搜重叠,287k 写/s 顶 |
| 池化 P0.2b | 7126 (32T) / **8319 (64T)** | **8.6** | 含 finalize 一次性写回;循环净速率 ~10.3k/s |
| 池化 none 臂 | **20271** (64T) | 1.0 | 搜索侧上限较 buffered 版 +15%(overlay 读) |

质量逐点持平(0.9658@60 / 0.9881@100 / 0.9969@200,new_recall 0.9763@100);
churn 100% 换血 + consolidation:0.9740(≈v2 0.9748),轮内 QPS 略升。
单测新增:同页去重断言、缓存开/关文件逐字节一致;laser 全套 11/11。

**下一杠杆**(P0 收尾/P1.5 衔接):drain(4.8s)与下一批搜索(4.9s)目前是
串行相位——bench 级双批流水(drain(i) 与 batch i+1 搜索重叠,隔离度等效
batch×2,forced-links 语义有一处批间分裂的小注意点,实测 forced=0 风险可忽略)
估计可到 ~14k/s;再往上是 P1.5 协程端到端流水线。教训入档:**测并行写路径前
先划相位计时;"不随线程扩展"优先怀疑全局串行点(本例 mmap_sem),而不是加锁
粒度。**

### 9.3 P0 收官:inline 补丁 + 驻留池 = 13.2k–14.7k inserts/s(2026-07-13)

**双批流水验证为负结果**:drain(i) 与 batch i+1 搜索重叠(--pipeline)仅 +2%
(8489 vs 8319,64T;flush_threads=16 反而 6229)——两相位共享池/锁/分配器结构,
重叠被相互拖慢抵消。这个负结果指回正解:**staging 的写合并收益本来就来自池
(同页补丁在池内天然合并,与相位无关),相位分离才是 2× 的代价**。

**终解(--stage 0,现默认)**:恢复 v2 的 inline 反向边补丁,但补丁落池不落盘
——搜索 IO 与补丁 CPU 在同线程自然重叠(v2 已证),零写 syscall,kAlphaEvict
的捕获池检查也回来了。分配器串行点已修(§9.2),补丁不再有 330k/s 顶:

| 配置 | inserts/s | vs 基线 7439 |
|---|---|---|
| inline+池 evict 64T | **13216** | 1.78× |
| inline+池 evict 128T | **14691**(仍在扩展) | 1.97× |
| inline+池 alpha 64T | 14280 | 1.92× |

物理写维持 8.6/插入(−4.15×);质量逐点持平(0.9666@60 / 0.9882@100 /
0.9968@200,new_recall 0.975@100;alpha 臂 new_recall@60 低 0.8pp,与 E1 一致,
默认仍 evict)。**P0 目标带 12–15k 达成。**

**意外红利:池即统一 buffer pool。** churn(500k 索引 1.9GB < 池容 4GB)中,
补丁与 consolidate 逐轮把全图带进池,搜索读退化为 shard 锁内 memcpy——轮内
插入速率 89k→118k/s 递增(50k 插入 <0.5s),consolidate 从 6.3s 降到 ~2.7s/轮,
churn recall 轨迹不变(0.9742@100%)。索引装得下内存时,更新路径自动获得
Yi 式统一缓冲池;装不下时按 cache_cap_pages 水位驱逐。后续知识点:读缺失
目前不入池(只有写路径入池),"populate-on-read" 可作为 P1.5 的一个便宜旋钮。

**StagedEdges 路径保留为 --stage 1 对照臂**(相位分离变体,含双批流水),
研究价值:分相位的 barrier 设计在共享结构上会互相拖慢,Yi 的"批内无 barrier"
经验在 LASER 上二次验证。

### 9.4 P1 质量消融:插入深度是主杠杆,gardening 是配角(2026-07-13)

机制(codex 按简报实现,已审查合并):RAM 入度计数器(可选,增量维护 + 单测
锁定"增量==重扫");`garden()` = 低入度分位(或等预算随机对照)选点 →
ef_maintenance 深搜 → 精确 RobustPrune → 整行重写 → 预算反向泵。
churn 序:tombstone → consolidate → garden → insert → eval。

SIFT 500k 滑窗 churn,10 轮 100% 换血,recall@100(第 10 轮):

| 臂 | recall | Δ | 入度 p1/p5(活点) |
|---|---|---|---|
| consolidate-only(ef_insert=100) | 0.9736 | — | 0 / 0 |
| + garden lowdeg ef100-400,无泵,frac .05 | 0.9750–0.9756 | +0.2pp | 8 / 14 |
| + garden lowdeg ef200/400 + 泵4,frac .05 | 0.9774 / 0.9776 | +0.4pp | 15-16 / 18-19 |
| + garden **random** ef200 + 泵4(等预算对照) | 0.9760 | +0.24pp | 8 / 13 |
| + garden lowdeg,frac **.20**(4× 剂量) | 0.9831 | +0.95pp | 15 / 23 |
| **仅 ef_insert 100→200,无 garden** | **0.9828** | **+0.92pp** | 0 / 0 |
| ef_insert 200 + garden frac .05 | 0.9842 | +1.06pp | 17 / 21 |
| **ef_insert 200 + garden frac .20(终选)** | **0.9865** | **+1.29pp,门槛 0.985 通过** | 16 / 24 |

**判决:**
1. **oracle"深搜是杠杆"在 LASER 上以 `ef_insert` 的形式成立**:100→200 一步
   +0.92pp,是所有单项里最大的;残差主体是**插入时的边质量欠账**,不是老化。
2. **入度尾巴修复成功但只值 +0.4pp**:gardening+泵把活点入度 p1 从 0 拉到 15+,
   recall 增益却有限——与 DiskANN 战役"入度刷新砍 40% 衰减"不同构。结构解释:
   LASER 的 evict 反向边补丁在每次插入时持续换血老节点(每轮 ~30 万次驱逐),
   入度年龄衰减本来就弱;R64+headroom 下 p50=37 不饥饿。
3. 定向 > 随机(+0.14pp)、泵 > 无泵(泵贡献 ≈ 无泵增益的一倍)、剂量近似
   线性(frac .05→.20:+0.4→+0.95pp),且 garden 剂量与插入深度可互换
   (f.20@ins100 ≈ ins200)——预算该花在哪取决于插入吞吐要求。
4. 成本:churn 内(池驻留,搜索读为 RAM)ef_insert 200 与 garden 的代价都
   可忽略(轮内插入 ~10 万/s 量级不变);**单次冷池插入**场景 ef_insert 200 预计
   吞吐减半(搜索读翻倍),是显式的质量/速度旋钮。

**50 轮长时域确证**(500k 基座,50 轮 × 10k = 100% 换血,ef_insert=200):

| 轮 | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| consolidate-only | 0.9915 | 0.9894 | 0.9855 | 0.9862 | 0.9837 | 0.9831 |
| + garden(f.20/ef200/泵4) | 0.9918 | 0.9906 | 0.9893 | 0.9899 | 0.9881 | **0.9878** |

- 两臂都有界收敛(与 diskann g03 稳态判决同构);cons-only 尾段仍缓渗
  −0.03pp/轮,garden 臂尾段 −0.005pp/轮 ≈ 全平。
- **garden 的价值随时域放大**:第 10 轮 +0.12pp → 第 50 轮 +0.46pp——老化是
  慢变量,短 churn 低估维护收益;这与"入度尾巴只值 +0.4pp"并不矛盾,长时域下
  尾巴不断再生,定期修剪的复利显现。
- 小轮次(2%/轮)比大轮次(10%/轮)整体平台更高(0.9878 vs 0.9865):修复
  节奏越接近流式,质量越好。
- 局限照旧:单数据集、单 seed;100 轮 × 大轮次需要数据集侧回收机制
  (diskann 侧的稳态回收协议),留待 collection 集成后补。

**P1 定案**:ef_insert 200 + 每轮 garden(lowdeg,frac 0.20,ef_maintenance 200,
泵 4)= 100% 换血 recall@100 **0.9865(10×50k)/ 0.9878(50×10k)**,门槛 0.985
通过;质量预算的分配次序:先插入深度,后 garden 剂量,最后泵。

## 10. 下一阶段路线图(2026-07-12 定,故事线定稿后;证据基线 = §9 + LASER_UPDATE_STORY.md)

> 原则:**先赌注后装修**。漂移实验决定论文命题上限,最先做;工程项按"论文刚需"
> 排序而非工程趣味(协程降级为条件项)。预计全包 6–10 周(并行化后)。

### Phase A(1–2 周,机时便宜)— 下注 + 封便宜攻击

1. **分布漂移实验 v0(全计划最重要一项)**:cluster-holdout 建图(80% 簇)+
   held-out 簇 OOD 插入流;eval 分新簇/老簇查询两组;对照臂 = 自家 DiskANN 线
   (冻结 PQ 码本 + 原地更新,同 harness 同 trace)。判决:免码本是否拉开差距
   → 直接决定标题/摘要档位(激进版 vs 机制版)。
2. exact-evict 臂 + evict 误差遥测(FastScan argmax vs 精确 argmax 命中率、
   rank regret、按轮累积曲线);full 臂同池化公平化。堵 A7/A11。
3. R56 建的 1M 静态上界对照(堵 A6 "headroom 参数巧合")。
4. 口径脚本化:≥3 seeds、recall@10/@100 双报、插入点/老点/全体三组分层。

### Phase B(2–3 周)— 工程双项,全是论文刚需

5. **P2 槽复用 + 格式 v2**:free-list + 墓碑持久化 + 零入度回收 + valid-degree +
   A/B superblock。解锁:300%+ 换血、文件大小平台化(A13 "sustained" claim)、
   大轮次 churn 的数据集回收。
6. **P3a 读写路径统一 + 并发干扰曲面**:统一读(现 O_DIRECT)写(现 buffered)
   语义;测 query QPS/p50/p99 × update rate 二维曲面,分 insert/delete/
   consolidate/garden 四相。堵 A4(系统论文必答)。
- (P1.5 协程 = **条件项**:若并发曲线难看或 deep10m 吞吐缩水过狠再上——它抬
  none 上限 20.3k 与 IO 重叠,但不在论文关键路径。)

### Phase C(2–4 周,机时大头)— 规模与基线

7. deep10m 主表复跑 + RAM-cap 扫描(0.5/1/4/16GB,working set > pool,报
   cache hit/唯一脏页/设备读写);100m 视机时(g03 数据须搬本地 /md1)。堵 A3。
8. fresh-layer 基线:最小实现(内存 fresh 图 + 周期 merge 全量重打包)或严格
   成本核算 + FreshDiskANN 文献数字。堵 A2。
9. 非 2 次幂维度数据集一个(强制走 PCA 残差路径 + npp>1 几何)。堵 A12。

### Phase D — 写作与判决点

10. 按 RQ1–7 重组评测(STORY §7);WAL 设计章 + 微基准(研究轨 scope out 完整
    恢复);文献精读核查(CleANN/OasisANN/PipeANN)→ novelty 句定稿。
11. PVLDB rolling(每月 1 号)投稿制,无悬崖:P0 包做完即投。

### 判决点与资源交织

- **JP-1(Phase A 末)**:漂移正向 → 激进标题+免码本主 claim;负向 → 机制主张,
  论文仍成立。
- **JP-2(Phase C 初)**:deep10m 吞吐缩水程度 → 是否启动 P1.5 协程。
- **与工业轨交织**:工业轨冻结线 2026-10(SIGMOD)/12(VLDB);本线 Phase C 机时
  重头应避开工业轨战役期;Phase A 便宜且决定赌注,建议立即执行。

## 11. Phase A 执行记录

### 11.1 A6 对照判决:headroom"反超静态"不成立,主张降级(2026-07-12,T3)

R56(容量 64)静态 1M 全建 vs 已有数字,new 段 = GT id≥900k(与 E1b 同口径):

| ef | R64 静态 new 段 | **R56 静态 new 段** | hr 插入点 new(E1b) | hr − R56 静态 |
|---|---|---|---|---|
| 60 | 0.9569 | **0.9644** | 0.9618 | −0.26pp |
| 100 | 0.9860 | **0.9872** | 0.9837 | −0.35pp |
| 200 | 0.9975 | **0.9975** | 0.9962 | −0.13pp |
| 300 | 0.9994 | **0.9989** | 0.9981 | −0.08pp |

- E1b 的"插入点反超静态上界(0.9618>0.9565)"**只对 R64 基线成立**;换成同参数
  (R56 建/容量 64)静态上界后,反超消失——hr 插入点距上界 0.08–0.35pp。
- 附带观察:R56 欠满建的静态索引本身在低 ef 优于 R64(总 0.9726 vs 0.9705@60,
  new 段 +0.75pp@60),与 E1b 在 900k 上的观察一致。
- **主张定稿(预判被证实,见 STORY §5 A6)**:headroom 的正确卖点是
  "零增量行字节改善质量/更新 tradeoff——欠满建图不损静态质量,流式插入点逼近
  同参数静态上界(≤0.35pp)",不是"反超静态"。E1b 原文的"反超"表述以本节为准修正。
- 复核:R64 复建与既有数字偏差 ≤0.08pp,实验链可信。产物 data/laser-update/r56upper/。

### 11.2 JP-1 分布漂移判决:正向,免码本敏感度小 2–4×(2026-07-12,T1/T4 + LASER 臂)

设计:SIFT1M 聚 64 簇,外围 12 簇 198,588 点做 held-out OOD 插入流(按簇心距升序
渐进加剧),建图集 801,412;null 对照 = 随机同量划分。查询按 top-10 GT 落 held-out
比例分组(g0=old 6303 / g1=mixed 1493 / g2=new 2204;ctrl 组下 g2=298,噪声 ±0.6pp)。
两臂同机同 trace:LASER(evict,ins200) vs 自家 DiskANN 原地线(PQ32 建图时训练,
冻结证据=pivots SHA-256 建前后一致 + 插入仅 encode_one,L=25)。

**干净口径**(两腿终态同为全量 1M;与同底座静态 1M 全建对照,同查询/GT/分组,
唯一差异 = 尾段 198k 流式 vs 建入):

| @ef100 | drift 腿 g2 流式惩罚 | drift g0 | ctrl g2 | ctrl g0 | 漂移特异超额(双差) |
|---|---|---|---|---|---|
| LASER | −1.17pp | −0.06 | −0.37 | −0.38 | **−1.12pp**(@ef60:−1.67) |

DiskANN(pre→post 双差,掩码人群漂移在双差中对消):drift 腿 g2 −4.92pp/g0 −0.81,
ctrl 两组均匀 −1.9 → **漂移特异超额 −4.16pp**。

**判决**:
1. 冻结码本在 OOD 区域确凿开裂(−4.2pp 特异超额,剂量响应单调);LASER 同口径
   敏感度小 2–4×(−1.1~−2.1pp,视口径)。**JP-1 正向**,免码本主张可升级为
   实测优势(带量级,不带"免疫")。
2. LASER 残余惩罚的归因:LASER 无任何训练产物(FHT 数据无关、免 PCA 配置),
   其 −1.1pp 只能是**拓扑冷启动**(新簇首批点无既有邻域);DiskANN 的 −4.2pp =
   拓扑冷启动 + 码本开裂两项之和 ⇒ 码本特异成分 ≈ −2~−3pp。要坐实分解,需补
   DiskANN 同底座静态 1M 参照腿(工具已备,便宜)。
3. LASER 漂移腿 g2 随簇外围度渐进变差(ckpt0 0.9883 → ckpt3 0.9817),ctrl 平
   ——剂量响应支持因果。
4. 论文级还需:DiskANN 静态参照腿、L 扫描(当前 L=25 单点)、≥3 seeds、
   周期重训 oracle 臂(升级敏感度上界)。
数据:data/laser-update/drift/(runs/、diskann-runs/);事故记录:首轮 LASER 臂
脚本漏传 --from(默认 0)误插建图集头部向量,已废弃重跑——**bench insert 模式
--from 无默认化,永远显式传**。

### 11.3 A7/A11 判决:估距误差无累积、零质量代价;full 直移植在现代配置下更差(2026-07-12)

单次 900k+100k(并行,ins200,驻留池,同配置三臂):

| 臂 | inserts/s | new_recall@ef60 | @ef100 | @ef200 |
|---|---|---|---|---|
| evict(FastScan 估距) | **10473** | **0.9654** | 0.9861 | 0.9956 |
| exact-evict(精确距离) | 3775(2.8×慢) | 0.9613 | 0.9845 | 0.9954 |
| full(完整 RobustPrune) | 2125(4.9×慢) | **0.8836** | 0.9345 | 0.9701 |

churn 10×50k 双臂(evict vs exact,同配置,无 r_target):逐轮 recall 双向抖动
±0.4pp,终点 0.9361 vs 0.9348——精确距离全程无优势。遥测(p=0.05):argmax
一致率 57%→47%(行内近平手边随轮增多),但 rank regret p50 = 0→1、相对误差
1.67%→1.69% 全程平——**分歧发生在平手场景,误差既不增长也不转化为 recall 差**。

**判决**:(1) A7 封口:估距误差不累积(10 轮遥测平 + exact 臂零增益 + 50 轮
recall 平台三重证据);(2) A11 封口:同池同并行下 exact 2.8×/full 4.9×,均无质量
增益——"9×"修正为"2.8–4.9×(并行池化口径)",且代价换不来质量;(3) 新发现:
full 直移植在 ins200 并行下 new_recall 崩 8pp(E1 单写者 ef100 时代仅 −0.85pp)
——深候选池加剧 α-过度剪枝压新点入度,叠加行粒度重写对批内并发的敌意
(读-剪-写跨锁窗口 vs evict 的槽粒度原子补丁);机理拆分待查,headline 不变:
**G2"直移植不只贵,还更差"从 0.2pp 加强到 8pp**。exact-evict 臂保留为诊断工具。
注意:本轮 churn 未传 --r_target,绝对值不可与 §9.4(0.9878)横比,臂间对照有效。

## 12. P3a 设计契约:查询走池的并发读写统一(待 P2 落地后实施)

**问题**:搜索读是 O_DIRECT libaio(aligned_file_reader),更新写是 buffered+驻留池
——并发下 (a) DIO 读绕过页缓存看到陈旧盘态;(b) DIO 读命中脏页缓存范围触发同步回写
(§9.1 实证的性能干扰);(c) 4K 页回写对并发 DIO 读无原子性保证 ⇒ 撕裂行。

**P3a 方案(bench 级,不动 QuantizedGraph)**:混合负载模式让查询线程**读穿驻留池**
——QGUpdater 增查询路径(search_for_insert 的 top-k 返回变体,同一 read_node_page
+ seqlock 协议),新鲜度语义 = published 水位 + tombstone 过滤(与插入搜索一致)。
池即 Yi 式统一 buffer pool 的过渡形态;QuantizedGraph 原生读者的池对接留给
collection 集成(P3 本体)。

**测量矩阵(bench --mixed 模式)**:固定查询负载(16T)× 插入速率
{0, 25%, 50%, 100% 无限速}(限速用令牌桶);固定插入(64T)× 查询并发 {4,16,32}。
每格报:query QPS / p50 / p99 / recall(vs 当前 live GT)、insert 吞吐、
维护相位(consolidate/garden)期间的查询尾延迟单列。四相分开:纯插入、
插入+删除、+consolidate、+garden。

**验收**:干扰曲面成图(A4 封口的数据);查询 p99 在 100% 插入负载下的膨胀系数
量化;seqlock 重试率计数(读写热页冲突的直接观测)。

### 11.2b JP-1 判决修正:v0 = 零结果(2026-07-12,T5 静态参照腿落地后)

**§11.2 的"正向,−4.2pp"作废**。DiskANN 同底座静态 1M 参照腿(PQ 训练于全量 1M)
落地后,干净口径(流式终态 − 静态 1M,同查询/GT/分组)显示:

| @L25 | drift g2 | drift g0 | ctrl g2 | ctrl g0 | 交互项 |
|---|---|---|---|---|---|
| DiskANN 流式惩罚 | −0.95pp | −0.88pp | +0.87pp | −0.30pp | **−1.24pp** |
| LASER(@ef100,§11.2) | −1.17pp | −0.06pp | −0.37pp | −0.38pp | **−1.12pp** |

两系统交互项统计上不可区分(ctrl g2 仅 298 查询,±0.6–0.9pp 噪声带)。

**为什么 §11.2 的 −4.2pp 是伪影**:pre→post 口径里,新区查询 pre 时只在稀疏存活
GT 上打分(掩码虚高 97.8%),而其真实 GT 在 L=25 下即使静态全建也只有 93.8%
——"题目变难"被计入"索引退化";且 drift/ctrl 两腿 g2 人群的硬度轨迹不同
(−4.0 vs −2.8pp),双差消不掉。**方法论教训(论文级):masked-recall 的
pre→post 双差在人群构成变化的组上不可用;必须用同终态的静态参照口径。**

**修正后判决**:
1. SIFT1M 温和簇漂移(held-out/内部 NN 距离比 1.29×)+ 全精度 rerank 兜底下,
   冻结 PQ **无可检出的码本特异退化**——rerank 把中等码本误差全部吸收。
2. 免码本主张退回**机制优势**(永无重训练义务、无全局耦合产物);recall 优势
   的兑现条件收窄为:更强漂移(跨域/真实时序)、rerank 受限或 PQ 预算更小的
   工况。激进档标题不解锁,主命题维持 codex 收窄版(§STORY.1)。
3. 附带正面读数:LASER 流式对老区几乎零损(g0 −0.06pp vs DiskANN −0.88pp),
   且 LASER 的 OOD 流式惩罚(−1.17pp)与 DiskANN(−0.95pp)同量级——
   "量化打包行的原地更新不因格式付出额外漂移代价"这个防守性结论成立。
4. v1 实验设计(若追求兑现 recall 优势,P1 级):跨域插入流(如 big-ann T2I
   clustered runbook / 异源 embedding)、holdout 加大、PQ chunks 16 压力臂、
   candidates-only(无 rerank)召回遥测直接观测码本开裂;或接受零结果,
   论文以机制主张 + 防守性结论定稿。

### 11.4 多 seed 批:顺序不敏感成立;shuffle 发布语义混杂待修(2026-07-12)

3 seeds × 最优配置 churn(10×50k,cons+r_target56+garden f.20+ins200,--shuffle_seed):
r10 recall = 0.9640 / 0.9629 / 0.9627,**种子间散布 ±0.06pp**;free_fills/evictions
逐 seed 几乎恒等(18.91M/5.45M)——插入顺序不敏感 + 维护行为确定性,防守性证据。

**但绝对值比 seed=0 协议(§9.4 的 0.9878)低 2.4pp——口径混杂**:shuffle≠0 走
"整段快照 + 段末统一发布"语义(T2 实现权衡:乱序 id 无法推进连续水位),轮内新点
互不可见,intra-cohort 边缺失。**本批不可用作 0.9878 的误差棒**。修法(Phase C 前任
一):(a) 批内洗牌(水位语义保留,随机性弱化);(b) 数据级洗牌(drift_prep 式预重排
底座,运行时语义原封,随机性来自数据)——推荐 (b),随 Phase C 规模批一起做。
