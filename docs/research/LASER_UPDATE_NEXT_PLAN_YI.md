# 对照 Yi 更新实现：LASER 动态更新下一步计划

> 调研范围：Yi 的 in-place DiskANN 更新与 I/O 栈、AlayaLite 的 DiskANN 迁移实现、
> LASER `QGUpdater` v2 和 `docs/LASER_UPDATE_EXPLORATION.md` §7。
>
> 结论先行：继续走 LASER **纯 in-place 数据面**，但把 Yi/迁移版 DiskANN 的
> “4 KiB 用户态页缓存 + 脏页合并 + O_DIRECT 整页写回”移植过来；先不要引入
> FreshDiskANN 式双图。质量侧把现有 consolidation 从“只处理死边”升级为
> “死边 splice + 低入度/高年龄活点 gardening”，并把更深的维护搜索作为首要质量杠杆。

## 1. Yi 更新架构画像

### 1.1 调度：外层小批次，节点内协程化，插入后 fan-out reconnect

Yi 的更新 runner 按轮执行“先删后插”。`UpdateRunner::update_a_round()` 和
`IPDiskANNRunner::update_a_round()` 都以 32 为 batch，给每个请求建立 `QueryContext`，
再用 `coro::when_all` 等待一批完成（Yi `src/runner/update_runner.cpp:70-147`、
`src/runner/ipdiskann_runner.cpp:66-127`）。这不是 FreshDiskANN 的内存增量图，也没有
前后台 merge；更新直接落到同一份磁盘图。

插入的实际链路在 `QueryContext::co_insert_update()`：

1. `co_insert()` 搜索并选择新节点出边，写新节点；
2. 对 `new_neighbors` 中每个点创建一个 `UpdateQuery`；
3. `when_all` 并发执行这些反向 reconnect（Yi
   `src/scheduler/query_context.cpp:500-531`）。

`co_update()` 的候选池由三部分组成：本轮插入暂存到 `_inserted_edges[node_id]` 的
反向边、仍存活的旧邻居、每个已删旧邻居最多 5 个缓存的二跳邻居；候选溢出时先取
`build_k` 近邻再 RobustPrune，最后重写该节点的邻接表（Yi
`src/scheduler/query_context.cpp:991-1107`）。`IDVecMap` 的 accessor 让多个插入向同一
目标积累的反向边可由第一个 updater 一次 drain，天然合并部分重复 reconnect 工作。

普通删除 `co_delete()` 很轻：先读取被删点旧邻居，放入 `_removed_nodes[id]`，再从
`_current_list` 移除；它不立刻扫描所有入邻居（Yi
`src/scheduler/query_context.cpp:674-682`）。Yi 的 IP-DiskANN 删除变体更积极：
`co_delete_ipdiskann()` 以被删向量搜索候选，把“被删点旧邻居 + 搜索候选池”去重后，
为所有存活受影响点 fan-out `UpdateQueryIPDiskANN`；每个 updater 保留活旧边，再加搜索
候选中距离最近的 3 个后重剪枝（Yi `src/scheduler/query_context.cpp:451-499`、
`1117-1222`）。这就是 Yi 代码中最接近“删除 splice/整形”的路径。

Yi 允许搜索与更新共存。`UpdateRunner::run()` 可另起 search thread，在更新轮运行期间
持续提交搜索协程（Yi `src/runner/update_runner.cpp:158-178`）；worker 事件循环同时轮询
I/O completion，并依次从 local/search/delete 等队列取协程（Yi
`src/scheduler/co_scheduler/co_worker.cpp:43-90`）。

### 1.2 并发协议：页内单 updater，读者在指针切换点等待

Yi 的并发单位是 `Buffer`（即 4 KiB block），不是节点：

- `BufferManager::async_read()` 先合并同一页的并发 miss；读盘完成后，如果
  `update_ready_` 已置位，读协程挂到 `ReadUpdateAwaitable`，否则取得 shared read lock
  （Yi `src/buffer/buffer_manager.cpp:76-101`）。
- `Buffer::prepare_update()` 用 `is_updating_` CAS 保证同一页只有一个 updater；竞争者
  挂入 `update_awaiting_list_`。成功者复制整页到 `data_copy`（Yi
  `src/buffer/buffer.cpp:136-166`）。
- `Buffer::finish_update()` 先设置 `update_ready_`，等待旧读者释放，再递增
  `memory_version_`、唤醒读者，最后只唤醒一个等待 updater（Yi
  `src/buffer/buffer.cpp:168-311`）。

因此 Yi 保证的是页内更新串行、搜索读到完整旧版或新版；它没有跨页事务原子性。
协程 awaitable 和 lock-free waiter list 是性能实现，协议本身可移植为“页锁/页版本 +
读者重试”。LASER v2 已用 4096 条页锁和 per-page seqlock 实现了后一种等价协议：
`write_node_page()` 写前后把版本改成奇/偶，`read_node_page()` 在 `pread` 前后校验版本
（`qg_updater.hpp:550-583`）。

### 1.3 写路径：用户态 buffer pool 合并脏页，再异步整页直写

Yi 的关键不是“把每次 pwrite 换成异步 syscall”，而是先把逻辑更新吸收到页缓存：

- `Buffer::finish_update()` 只递增 `memory_version_`，不在更新临界路径落盘；
- `Replacer::release_buffer()` 发现 `memory_version_ != disk_version_` 就把页放入 dirty LRU
  （Yi `src/buffer/replacer.cpp:441-497`）；同一页再次被修改时继续修改同一内存页，多个
  逻辑边更新被合成一次设备写；
- worker 每轮检查 completion，并在队列深度允许时调用 `writeback_one()`；脏页超过
  30k 后一次调度 32 页，最终 `writeback_remaining()` 清空（Yi
  `src/scheduler/co_scheduler/co_worker.cpp:43-65`、`src/buffer/replacer.cpp:267-330`）；
- `BufferManager::async_write()` 提交整页 I/O，completion 才把 `disk_version_` 追到
  `memory_version_`（Yi `src/buffer/buffer_manager.cpp:211-256`）。

I/O 后端统一以固定 block 寻址。io_uring 后端用 `open(..., O_RDWR | O_DIRECT)`，每线程
一个深度 256 的 ring，提交 `block_size` 的 positional read/write（Yi
`src/io/iouring_engine.cpp:10-29,39-84`）；SPDK 后端每线程一个 NVMe qpair，直接提交
namespace read/write（Yi `src/io/spdk_engine.cpp:63-103`）。二者都绕过 buffered file
write 路径，因此没有 LASER 当前 buffered `pwrite` 的 ext4 排他 inode 锁热点。

这里没有写合并成“大于 4 KiB 的连续 extent”：合并发生在**同一逻辑页的多次修改**，
后台仍按 4 KiB 页分别提交。io_uring 代码也未使用 registered file/buffer、SQPOLL 或
linked write；不能把 Yi 的收益归因于这些特性。

### 1.4 一致性与恢复：运行时可见性有协议，崩溃原子性没有

Yi 在进程内通过 buffer 版本和锁保证搜索/更新共存，但调研范围内没有 WAL、page
checksum、事务 commit record 或恢复 replay。runner/benchmark 结束显式调用
`writeback_remaining()`、持久化 bitmap、写不完整 block（Yi
`inc/runner/insert_bench.hpp:155-158`）。这更像实验系统的 checkpoint，不是可证明的
crash consistency：新节点本体、bitmap/PQ、多个 backlink 页中途崩溃可以不一致。

SPDK 特有部分是 hugepage/DMA-safe 内存、每线程 qpair、主动 completion polling 和裸
namespace LBA；可移植部分是固定 4 KiB 页、用户态 buffer pool、脏版本、页内更新串行、
异步队列深度控制、批内并发以及延迟 writeback。Linux 文件版可用 O_DIRECT + io_uring
实现同一模型。

## 2. 与 LASER 原型逐项对照

| 维度 | Yi 原版 | AlayaLite DiskANN 迁移版 | LASER `QGUpdater` v2 | 判断 |
|---|---|---|---|---|
| 更新形态 | 同一磁盘图 in-place；32 请求一批 | 同一磁盘图 in-place；chunk 内端到端协程 | 同一 QG 文件 in-place；调用者并行 batch | 路线一致，无 fresh 图 |
| 插入出边 | search → prune → 写 node | `select_insert_neighbors_async` → alloc/reuse → 写 | `search_for_insert` → `robust_prune` → `assemble_row` | LASER 已对齐 |
| 反向边 | `_inserted_edges` 聚合，fan-out `co_update` | `StagedEdges` 分 64 stripe，首个任务 drain 后 reconnect | 每个 selected neighbor 独立 `patch_reverse_edge` | LASER 缺同页/同目标 staging 合并 |
| 删除 | tombstone/current-list 移除；缓存旧邻居 | slot tombstone + 复用；缓存旧邻居，insert/safety-net 修 | 内存 tombstone，路由穿透 | LASER 尚无持久 tombstone/复用 |
| 删除修复 | 普通路径懒修；IP 变体搜索候选并 fan-out | live old + 每死邻居最多 5 个二跳；轻 safety-net 只 purge | consolidation 用死点行 FastScan 召回，top-4 raw 精排 | LASER splice 更贴合 edge-code 布局 |
| 写缓存 | 全局 BufferManager + dirty LRU | 64 shard `DiskPageCache`，dirty eviction/flush | 无；每个成功 patch 立刻 buffered pwrite | 287k 墙的直接根因 |
| I/O | SPDK 或 io_uring + O_DIRECT | `O_DIRECT|O_RDWR`；读可 io_uring wave，写回同步 pwrite | updater fd 为普通 `O_RDWR` buffered；搜索另走 direct I/O | 必须统一 LASER 写路径 |
| 写合并 | 同页多次更新合成一次 writeback | cache 内覆盖同一 `page_off`，dirty 位 OR；逐出/flush 一写 | 无；约 35.8 页写/insert | 最大可移植收益点 |
| 运行时并发 | Buffer 级 updater 串行，读者切换等待 | 64 shard mutex + per-node reconnect lock；外层 update 串行 | 4096 stripe 页锁 + per-page seqlock；批后 watermark | LASER 协议更适合无锁读 |
| 发布 | 没有显式跨页事务；内存 current-list | 写 node 后 `publish_slots`，NodeCache override 保脏页可见 | 全 batch backlink 完成后一次 `committed_.store(release)` | LASER 发布语义更清楚 |
| 搜索见脏页 | 直接读 Buffer | O_RDONLY 搜索读看不到 dirty page，靠 NodeCache override | 每次 pwrite 已进内核页缓存；seqlock 防进程内 torn read | 引入延迟写回后必须加 overlay/cache coherence |
| 崩溃恢复 | 未发现 WAL/replay | 未发现 WAL；`flush()` 是 checkpoint | 明示无 WAL，`finalize()` 才改 meta + fsync | 三者都未达生产级 |

迁移版的具体证据尤其重要：`DiskPageIO` 以 64 shard mutex 保护页缓存和 RMW scratch，
`write_node_neighbors()` 在锁内读页、改邻接区、调用 `write_page_locked()`；缓存启用时后者
只把 `DiskPageCache` entry 标脏，逐出或 `flush_dirty_pages()` 才用对齐 bounce buffer 做
O_DIRECT `pwrite`（`disk_page_io.hpp:432-458,479-487,918-984`）。`DiskPageCache::write()`
对同一 `page_off` 直接覆盖 bytes 并 OR dirty，对 LRU dirty victim 才 flush
（`disk_page_cache.hpp:37-79`）。这套代码已经回答“LASER 应抄什么”，无需从 Yi 重新搬
一整套 SPDK BufferManager。

## 3. 四个问题的判断

### 3.1 287k pwrites/s 墙：推荐“两阶段”而非四选一

#### 推荐方案

**第一阶段：O_DIRECT 对齐整页写 + 分片 write-back page cache，一起做。**

只换 O_DIRECT 能避开 buffered ext4 inode 写路径，让不同页的写可并行，是最快验证
瓶颈归因的实验；但 LASER 每插入约 35.8 次成功 backlink 页写，若不合并，设备 IOPS、
CPU 与写放大仍然存在。只加 buffered 用户态 cache 虽能减少 syscall，却在 eviction
时仍回到同一 inode 锁。因此生产方向应二者组合，直接复用/抽取迁移版
`DiskPageCache` 和 `DiskPageIO` 的 shard + aligned scratch 设计。

LASER 需要额外处理搜索可见性：延迟写回后，搜索不能只 `pread` 旧文件。建议 updater
内提供 `read_page_snapshot(page_id)`：先在对应 shard 下复制 cache 中的最新页；miss 才
O_DIRECT 读盘。现有 `read_node_page()` 的 seqlock 包住这次 snapshot copy/miss read，
保留 lock-free reader 语义。不要复制 DiskANN 的全节点 `NodeCache override`，因为 LASER
一行大、还包含 edge codes/factors；以页为 overlay 才不会产生双份行状态。

写回采用 epoch/version：flush 先在 shard lock 下复制 `(page, dirty_version)` 到对齐
bounce buffer，解锁后 O_DIRECT 写；完成后仅当当前 version 仍等于提交 version 才清 dirty，
否则保留脏状态等待下一轮。这样不会把并发中新修改误标为已落盘。相邻 page batching
可后加，不是 P0。

#### 备选方案评估

| 方案 | 能否破 inode 锁墙 | 能否降逻辑写次数 | 主要风险 | 估计工程量 |
|---|---:|---:|---|---:|
| 仅 O_DIRECT + 对齐写 | 是 | 否 | 对齐、末页/metadata、direct read coherence；随机 IOPS 仍高 | 2–4 人日（验证臂） |
| 用户态页缓存 + O_DIRECT writeback | 是 | 是，同页多 patch 合一 | 搜索必须读 dirty overlay；eviction/version 正确性 | 1.5–2.5 人周 |
| io_uring 普通/注册写 | 若 fd 为 O_DIRECT 才是 | 否 | completion 生命周期、固定 buffer 池；注册写不解决写放大 | 普通写 3–5 人日；注册资源 1–2 人周 |
| 分段文件/多 inode | 部分，是绕路 | 否 | 破坏纯 id 算术；需 segment map、跨段寻址、merge/recovery | 3–5 人周以上 |

io_uring 应作为第二阶段的 writeback engine：在页缓存已经把写合并后，用固定对齐 buffer
池和有限 QD（例如 64/128）隐藏设备延迟。先做 registered write 的收益上限较小；Yi 的
io_uring 后端本身也没有注册资源。分段文件与 LASER 当前“一个 id 直接算一个 offset”
的最大优点冲突，不应为绕 inode 锁先引入 page table/segment directory。

性能预期应分层报告，不能直接承诺 15k：

- O_DIRECT-only 若 287k 墙确由 ext4 buffered inode lock 主导，应让 T=32→64 不再同卡；
- 页缓存收益取决于 batch 中 backlink page 重合率，需直接统计
  `logical_page_updates / unique_dirty_pages / physical_writes`；
- 最终上界仍是 none 臂的搜索侧 17.7k inserts/s。比较现实的 P0 目标是 evict 达
  12–15k inserts/s，或至少达到 none 上限的 70%，同时 physical writes/insert 明显低于
  35.8；若 unique-page 比例过高，则 O_DIRECT 并行而非合并贡献主要收益。

### 3.2 churn 残余 1.7pp：从 dead-edge repair 升级为质量 gardening

现有 consolidation 已解决“死边和墓碑 ballast”，并以 `r_target=56` 再生 8 个 headroom
槽；100% 换血后 recall@100 从 0.8887 拉到 0.9748，且第 4 轮趋平。剩余相对静态重建
约 1.7pp，说明问题已不主要是死边，而是活图的边质量/入度年龄欠账。这与 DiskANN
gardening 的 oracle 判决一致：入度定向刷新有效，但真正杠杆是维护搜索深度。

LASER 上建议这样落地：

1. **维护入度与年龄的轻量 delta。** 在成功 `patch_slot/zero_slot` 时对 old/new id 做
   `indegree_delta--/++`，并记录 `last_refresh_epoch`、本行累计 eviction、死边比例。
   不必持久化完整反向表；checkpoint 可扫 neighbor-id SoA 重建。
2. **候选行选择。** 每轮 consolidation 后，从活点中取低入度分位（如 bottom 5%）+
   高年龄/高 eviction 行；随机对照同样数量，避免把全扫成本误判为策略收益。
3. **再插入式深搜刷新。** 以该行 raw vector 为 query，用独立
   `ef_maintenance={100,200,400}` 做更深的 `search_for_insert`，候选池与当前活邻居合并，
   exact raw 距离排序后 RobustPrune 到 `r_target=56`，一次 `assemble_row` 重写整行。
4. **反向泵要有预算。** 对刷新后新增的前 `B={0,4,8}` 个邻居，向对端补 backlink；
   优先填 ghost slot，必要时才 evict。LASER 每条 backlink 还要重算 1-bit code + 三因子，
   所以不能像 DiskANN 一样无预算 fan-out。先依靠 P0 页缓存把同页泵合并。

为何不是继续把 splice top-4 调大：`consolidate_row()` 的候选只来自死节点自己的旧
邻域，局部性强，能清 ballast，但无法主动发现更远、更新鲜的导航边；提高
`splice_rerank` 只改善候选内精排，不能扩大候选来源。更深维护 beam 才对应 oracle 的
“搜索深度是杠杆”。

验证必须同时测 recall 与结构：每轮 recall@100、低入度分位/p1/p5/median、从 entry
可达率、平均活出度、边年龄、刷新行的新边存活率、maintenance 页读/写和搜索 tail
latency。核心消融为：purge-only、现 splice、splice+低入度浅刷、低入度深刷、深刷+
反向泵、等预算随机刷新。成功门槛是把稳态 0.9748 至少提升到 0.985，且维护摊销不超过
前台 update CPU/I/O 的 20%；若 `ef=400` 仍无收益，再考虑扩大候选来源或局部重建。

### 3.3 fresh 层取舍：当前不做 FreshDiskANN 式双图

Yi 的实际做法是纯 in-place，并没有用 fresh 内存图。LASER 也有三个继续 in-place 的
结构优势：固定宽行和纯 id 算术、逐边 RaBitQ 无全局码本、headroom 可廉价吸收 backlink。
现原型已证明 10% 增量几乎追平静态、consolidation 后 churn 有稳态；引入内存图会新增：

- 两套距离/搜索路径和 top-k merge；
- fresh→disk merge 时为每条新拓扑边重新生成 code/factors；
- id/segment/manifest 发布与删除版本；
- 内存图容量、后台资源隔离以及 merge crash recovery。

它并不消灭最终的 LASER 边码写，只把写延后成更大的 merge。当前已知瓶颈可以用页缓存+
direct I/O 直接处理，已知质量缺口可用 in-place gardening 验证，因此现在做 fresh 层会把
两个可独立回答的问题绑成一次架构重写。

建议把 fresh 层设为 P2 的**触发式备选**，满足任一条件才立项：

- P0 后 evict 仍低于 10k inserts/s，且设备/随机 unique-page IOPS 而非软件锁成为瓶颈；
- 严格更新可见延迟要求与 WAL/fdatasync 成本使每批 in-place 提交不可接受；
- 多段 collection 本就需要 immutable segment merge，能复用 manifest/compaction 框架；
- 更新分布高度局部，fresh 图在 merge 前能显著改善查询，而页缓存合并率仍低。

即使触发，也建议“内存 delta + 不可变 base segment”的 collection 层实现，不要把
`QGUpdater` 本身改成双图；保持 QG 文件格式和单段搜索器简单。

### 3.4 生产一致性：不能照抄 Yi 的空白

LASER 当前 batch barrier + committed watermark 只保证进程内发布顺序；seqlock 只防并发
torn read，进程崩溃后版本表消失。`finalize()` 直接改 meta、`ftruncate`、最后 `fsync`
（`qg_updater.hpp:415-435`），无法恢复“新行已写、部分 backlink 已写、meta 未提交”等
中间态。P0 性能实验可以继续标记 non-durable；生产 P1 必须补 WAL。

最贴合纯算术布局的是 physiological full-page redo WAL：记录 tx/batch id、page id、
page generation、after-image、checksum；先 durable WAL，再写数据页，最后 durable commit
和 A/B superblock 的 committed N/LSN。恢复时 replay committed batch 的 after-image，
忽略/截断未提交尾部。tombstone bitmap 和 meta 也必须进入同一事务。不要用 shadow page，
因为它会迫使纯 id 算术寻址变成 page table。

## 4. P0 / P1 / P2 计划

### P0：先解除已量化的性能墙，并建立质量可解释性（2–3 周）

#### P0.1 O_DIRECT-only 写路径实验臂（2–4 人日）

- **动机：** 用最小改动验证 ext4 buffered inode lock 是 287k 平台的主因。
- **做法：** 在 `qg_updater.hpp` 抽出 `LaserPageIO`；数据页 fd 用
  `O_DIRECT|O_RDWR`，页 buffer 用 sector/page 对齐分配；meta 4 KiB 也对齐写。保留现有
  页锁、seqlock、每 patch 一写，暂不改变语义。
- **验证：** 复跑 SIFT1M T=1/8/16/32/64 的 none/evict；记录 inserts/s、physical
  pwrite/s、CPU、device IOPS、p95/p99；用 npp>1 单测覆盖末页 RMW。
- **预期：** T=32/64 不再同卡；evict 明显超过 8k。若无改善，perf/lockstat 与块层指标
  应能否定 inode-lock 假设。

#### P0.2 分片 write-back page cache（1–1.5 人周）

- **动机：** 把同一页多个 backlink 和 append/RMW 合成一次物理写，降低写放大。
- **做法：** 抽取 `disk_page_cache.hpp` 的 LRU/dirty 语义，按 page id 分 64/256 shard；
  增加 page version、aligned flush scratch、dirty high/low watermark 和显式
  `flush_batch()`。`append_node`、`patch_reverse_edge`、`consolidate_row` 全部走同一页
  overlay；搜索 `read_node_page` 先读 overlay snapshot，miss 才 direct pread。
- **验证：** 新增“同页 N 次 patch 只产生 1 次 flush”、flush 与并发 patch 不丢更新、
  dirty search 可见、eviction 后重载一致、cache on/off 字节等价测试；基准报告 logical
  update/unique dirty/physical write 三层计数。
- **预期：** evict 12–15k inserts/s 或达到 none 上限 70%；physical writes/insert 低于
  35.8。收益不足时用 unique-page 比率解释，而不是继续盲调线程。

#### P0.3 gardening 观测和深度消融（4–6 人日，可与 P0.2 并行）

- **动机：** 判定残余 1.7pp 是否同 DiskANN oracle 一样由维护搜索深度主导。
- **做法：** 在 `QGUpdater` 加 indegree delta、row age/eviction 统计；扩展
  `bench_laser_update_sift.cpp` 支持 maintenance ef、refresh fraction、pump budget 和
  selection policy；先不做复杂持久化。
- **验证：** 上述六臂消融，至少 3 seeds，跑到 100%–300% turnover；同时输出结构指标。
- **预期：** 低入度深刷显著优于等预算随机/浅刷，目标 recall@100 ≥0.985。

### P1：把有效方案做成可维护、可恢复的 in-place 引擎（4–7 周）

#### P1.1 格式 v2：valid degree + durable metadata（1–2 人周）

- **动机：** 删除幽灵槽启发式，并给恢复提供 generation/LSN/checksum。
- **做法：** 行头加入 `valid_degree` 或 slot bitmap；A/B superblock 保存 geometry、
  allocated/committed N、generation、WAL LSN、checksum；提供 v1 只读和离线迁移工具。
- **验证：** id=0 真边、重复向量、欠满行、npp>1、旧格式兼容和损坏 checksum 单测。
- **预期：** 空槽判定确定化，格式可演进，后续 WAL 有可靠锚点。

#### P1.2 WAL + checkpoint/recovery（2–3 人周）

- **动机：** Yi 未解决、LASER 生产化不可绕过的崩溃一致性。
- **做法：** full-page redo WAL；batch 级 begin/page/commit；data/WAL fdatasync 顺序；
  checkpoint 回收；tombstone/meta 纳入事务。page cache flush 按 WAL durable LSN 限制。
- **验证：** 在 append、每个 backlink、WAL sync、data flush、superblock flip 前后 kill -9；
  恢复后检查 committed 范围、无死/越界边、查询与参考状态一致。
- **预期：** 已提交批次必恢复，未提交批次不可见且尾部可安全复用/截断。

#### P1.3 正式 gardening scheduler（1–1.5 人周）

- **动机：** 将 P0 胜出臂从 benchmark 脚本变成有预算的后台维护。
- **做法：** maintenance queue 按低入度/年龄/死边率打分；独立 CPU/I/O token bucket；
  深搜重构整行，有限 backlink pump；与 insert/consolidate 用 epoch phase 或同一页协议共存。
- **验证：** 混合搜索+更新+维护，报告搜索 p50/p95/p99、更新可见延迟、维护 backlog；
  300% turnover 不持续衰减。
- **预期：** 收回大部分 1.7pp，维护开销可控且不造成搜索 tail spike。

#### P1.4 io_uring writeback（可选，5–8 人日）

- **动机：** page cache 合并后进一步隐藏 O_DIRECT write latency。
- **做法：** 有界 QD、固定 aligned buffer pool、completion 后按 version 清 dirty；先普通
  io_uring write，profile 证明 buffer registration 有收益后再注册。
- **验证：** QD sweep、flush latency、并发覆盖、短写/错误注入。
- **预期：** 提高 flush 带宽并降低 updater 被同步 writeback 阻塞的尾延迟；不预设它会
  改善逻辑写放大。

### P2：collection 集成与条件式架构升级（4–8 周，取决于触发条件）

#### P2.1 Collection/API 集成与在线 compaction（2–4 人周）

- **动机：** 把研究 updater 接入 label/tombstone/segment 生命周期。
- **做法：** `DiskCollection` 支持 active updatable LASER segment、持久 label map、
  tombstone、flush/optimize job；后台可将老段重建为 immutable segment并原子发布 manifest。
- **验证：** API e2e、重启、并发 query/update、段替换和空间回收。
- **预期：** 用户可用的动态 LASER，同时保留周期全量重建作为质量/空间兜底。

#### P2.2 Fresh 层决策门（1 周实验；若通过再 4–8 周实现）

- **动机：** 只在 in-place 已被设备随机 IOPS、durability latency 或 collection 多段需求
  证明不够时承担双图复杂度。
- **做法：** 先用小型内存 Vamana delta + 现有 base 做 query dual-route 原型，测增量驻留
  时间、merge 写放大、tail latency和总内存；不先改永久格式。
- **验证：** 与 P1 in-place 在相同 durability、CPU、内存、更新率下比较；必须在吞吐或
  tail latency 至少改善 2 倍且 merge 可持续时才进入产品实现。
- **预期：** 得到有数据的 go/no-go；默认 no-go，继续 in-place。

## 5. 建议的执行顺序与判决点

1. 先完成 P0.1。若 O_DIRECT-only 仍在约 287k 写/s 同卡，暂停 page-cache 大改，先用
   lockstat/perf/设备队列重新定位；否则立即进入 P0.2。
2. P0.2 与 P0.3 并行推进。写路径先解决，gardening 的 backlink pump 才不会被旧墙
   污染成本结论。
3. P0 结束做一次 go/no-go：吞吐达到 ≥12k 且深刷达到 ≥0.985，则锁定纯 in-place，
   进入格式/WAL；未达到时分别按 unique-page IOPS 和质量结构指标定位，不直接跳 fresh。
4. P1 顺序为 format v2 → WAL → scheduler；io_uring writeback 是 profile 驱动的可选项。
5. Fresh 层只在 P2 决策门通过后实现。分段文件也留在 collection/manifest 层，不破坏
   单个 LASER segment 的纯 id 算术寻址。

这条路线最大限度复用了已经验证的代码：并发发布继续用 `QGUpdater::insert_with_id()` /
`publish()`，页缓存/I/O 复用 `disk_page_cache.hpp` / `disk_page_io.hpp` 的成熟模式，质量
维护复用 `consolidate_row()`、`search_for_insert()`、`assemble_row()` 和 `patch_slot()`；
新增工程集中在页 overlay 可见性、持久格式和 WAL，而不是重写搜索与量化内核。
