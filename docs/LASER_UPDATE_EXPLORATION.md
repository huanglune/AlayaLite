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

### E0 静态基线

<!-- FILL: eval_static1m.csv / eval_static900k.csv -->

### E1 插入(900k 建 + 流式插入 100k,对照 1M 静态全建)

<!-- FILL: insert_*.log 吞吐/IO + eval_ins_*.csv -->

### E2 删除(1M 建,tombstone 10%,对照 900k 静态重建)

<!-- FILL: eval_delete_tombstone.csv vs eval_static900k.csv -->

## 6. 结论与生产化差距

<!-- FILL after experiments -->
