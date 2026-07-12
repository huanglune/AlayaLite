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

### 实验局限

单数据集(SIFT1M)、单次序(自然顺序)、单 seed、128 维免 PCA 路径;残差维
(main_dim < dim)与 npp>1 大规模路径仅有单测覆盖;并发读写、崩溃注入未做。
这些是论文级 battery 的下一步,不改变可行性结论。
