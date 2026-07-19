# O_DIRECT 写路径实验诊断与 P0.2 flush 模式选型

## 结论先行

1. **未 `sync` 时从 7439/s 坍塌到 2030/s，主因就是 DIO 写前的页缓存协调。** 更精确地说，Linux iomap DIO 写会进入 `kiocb_invalidate_pages()`；非 NOWAIT 路径先对本次写的字节范围调用 `filemap_write_and_wait_range()`，再做 `invalidate_inode_pages2_range()`。因此，克隆留下的 3.4 GB dirty page cache 使随机 4 KiB patch 高概率变成“先把该范围旧的 dirty folio 同步写回并等待，再失效，再提交新的 DIO 写”。它**不是每个 patch 刷整个 3.4 GB**，而是每次只强制处理重叠范围。`sync` 后 5934/s 的强反事实对照使这个归因置信度为 **0.97**。
2. **干净缓存下剩余约 20% 吞吐差距的主导项是同步 DIO 的 I/O completion 等待，以及等待被包在应用页 mutex / seqlock 奇数窗口内造成的放大。** ext4 对已分配、已初始化 extent 内的对齐 overwrite 走 shared `i_rwsem`，不是主要损失；页缓存 pre/post invalidation、folio/xarray 争用以及由失效带来的 buffered read miss 是明确的第二项。该项无法仅凭四个吞吐数精确分账，下面给出区间估计，归因置信度 **0.75**。
3. **P0.2 明确选择 b：每个唯一脏页一次 buffered `pwrite`，批末 `sync_file_range(..., SYNC_FILE_RANGE_WRITE)` 异步启动回写；publish 不等待回写完成，`finalize()` 仍以 `fsync()` 作为外部 reader 的可见性/完成边界。** 这是四个候选中唯一同时保留内核页缓存读局部性、允许 publish 后继续异步排水、又避免 c 的无界 dirty backlog/随机节流的方案。推荐置信度 **0.86**。
4. **b/c 下 seqlock 仍能保证本进程的 buffered `pread` 不接受撕页结果**，条件是版本奇数必须覆盖完整 buffered `pwrite` syscall，且同页写者仍串行。内核 buffered write 的 folio 内 memcpy 与 buffered read 的 copy-to-user 即使存在重叠，读者也会因为前后版本不同或读到奇数而重试。seqlock 不让尚在用户态 cache、尚未 `pwrite` 的数据自动可见；发布前必须先把待发布页复制到内核 page cache。置信度 **0.93**。

## 1. 未 sync 时 2030/s 的坍塌

### 内核调用链

以 Linux 6.8 的 iomap/ext4 路径为准，4 KiB 对齐 `pwrite(O_DIRECT)` 的关键链路是：

```text
ext4_file_write_iter
  -> ext4_dio_write_iter
     -> ext4_dio_write_checks
     -> iomap_dio_rw / __iomap_dio_rw
        -> kiocb_invalidate_pages
           -> filemap_write_and_wait_range(offset, offset + 4095)
              -> __filemap_fdatawrite_range(..., WB_SYNC_ALL)
              -> __filemap_fdatawait_range(...)
           -> invalidate_inode_pages2_range
        -> submit_bio
        -> 同步 kiocb 等 completion
        -> iomap_dio_complete
           -> kiocb_invalidate_post_direct_write
              -> invalidate_inode_pages2_range（再次清理由并发读重新装入的页）
```

源码锚点：

- `fs/ext4/file.c`: `ext4_file_write_iter()`、`ext4_dio_write_iter()`、`ext4_dio_write_checks()`、`ext4_overwrite_io()`；
- `fs/iomap/direct-io.c`: `__iomap_dio_rw()` 在写提交前调用 `kiocb_invalidate_pages()`，`iomap_dio_complete()` 在完成后调用 `kiocb_invalidate_post_direct_write()`；
- `mm/filemap.c`: `kiocb_invalidate_pages()`、`filemap_write_and_wait_range()`、`invalidate_inode_pages2_range()`。

两个 fd 指向同一 inode/address_space；读 fd 是否带 `O_DIRECT`、是否是另一次 `open()` 都不隔离 page cache。因此克隆产生的 dirty folio 与 `wfd_` 的 DIO 写直接冲突。

### 为什么能慢 3.7 倍

吞吐换算为每插入墙钟时间：

| 场景 | us/insert | 相对 clean-DIO 增量 |
|---|---:|---:|
| buffered 基线 7439/s | 134.4 | -34.1 |
| clean DIO 5934/s | 168.5 | 0 |
| dirty DIO 2030/s | 492.6 | **+324.1** |

dirty-DIO 约为 `2030 × 35.8 = 72.7k` 次 patch DIO/s。对初始 dirty 范围中的第一次触达，每次 patch 除了新的 DIO write，还同步触发一次旧 dirty page 的 writeback；即该 patch 在关键路径上有两次数据写的工作量，并夹着 folio writeback wait 和 invalidation。随机触达把内核原本可以聚合、排序、后台完成的 3.4 GB writeback 拆成大量 4 KiB 的前台同步等待。

代价又被三层放大：

- `write_node_page()` 的应用 seqlock 从 `pwrite` 前保持奇数直到 syscall 返回；对应 page stripe mutex 也由 `patch_reverse_edge()` 持有。等待旧页 writeback 和新 DIO completion 的时间都落在临界区内，冲突写者排队、读者 yield/retry。
- 64 个同步写线程把随机旧页 writeback和新 DIO 同时压到 NVMe，排队延迟不是单次 NAND program latency的常数；吞吐下降后搜索/CPU 与 I/O 的流水重叠也变差。
- 每次 pre-invalidation 都扔掉 buffered 搜索可能复用的页；后续搜索要重新从盘装入，读流量又与双重写流量竞争。

因此“3.7 倍”不能解释成某个 syscall 固定多了 3.7 倍指令；它是 write-back cache 被逐页强制同步化、额外物理写、设备排队、应用临界区拉长和读 cache 破坏的非线性合成。`sync` 预先把 dirty folio 清掉，直接恢复到 5934/s，正是这条机制的强验证。

**判定：是，但措辞应为‘每个 DIO 写对其重叠范围先 `filemap_write_and_wait_range`’，不是‘每次刷新整 inode 的全部脏页’。**

## 2. sync 后 5934 仍输 buffered 7439

clean DIO 相对 buffered 多 34.1 us/insert，等价于总时间增加 25.4%、吞吐降低 20.2%。不能把 34.1 us 再简单除以 35.8 当作单次 NVMe latency：64 线程并发隐藏了大部分设备等待，所得 0.95 us/page 只是**聚合墙钟摊销**。

### 2.1 主导：同步 DIO completion + 临界区放大

普通 `pwrite` 生成的是同步 kiocb；`__iomap_dio_rw()` 的 `wait_for_completion` 为真，提交 bio 后调用线程等待 I/O completion 才返回。这里的“完成”是块 I/O completion，**不是持久化保证**：fd 没有 `O_SYNC/O_DSYNC`，每次写不会额外保证 NVMe volatile write cache 已 flush/FUA。

clean-DIO 在 evict 臂达到约 `5934 × 35.8 = 212k` 次随机 4 KiB DIO write/s，已经把设备/块层 completion 置于每个 patch 的关键路径；buffered 基线的约 266k pwrite/s 只需把数据复制到 page cache，并可由内核之后合并/排序写回。none 臂只有约 18.6k 次追加 DIO/s仍达 18559 insert/s，说明“有 DIO 就慢”不成立；真正触发差距的是 evict 臂的高频同步随机 completion。

应用层还把该等待包含在：

```text
page stripe mutex held
version: even -> odd
    synchronous DIO pwrite + completion wait
version: odd -> even
page stripe mutex released
```

热点页/同一 stripe 的写者会排队；无锁搜索读在奇数窗口自旋或在版本变化后重读。这个放大属于同步 DIO 主项，而不是 ext4 inode lock 主项。

**估计贡献：吞吐差距中的 55%–75%，即单独消除后约可追回 11–15 个百分点。置信度 0.78。** 要精确验证，应看 `block:block_rq_issue/complete` 延迟、`iomap_dio_rw_*` tracepoint、page mutex wait 和 seqlock retry，而不是只看 CPU profile。

### 2.2 非主导：ext4 overwrite 的 inode lock 模式

`ext4_dio_write_checks()` 先调用 `ext4_overwrite_io()`。当请求：

- 完全位于 `i_size/i_disksize` 内；
- `ext4_map_blocks()` 证明请求覆盖的全部块已分配；
- 是已初始化 mapped extent；
- 4 KiB offset/length/buffer 对齐；

则 `overwrite=true`、`unwritten=false`，保留 shared inode `i_rwsem`，并选用 `ext4_iomap_overwrite_ops`。题中 reverse-edge patch 正是这个 fast path。不会因为同 inode 的多个 overwrite DIO 而在排他 ilock 上串行。

例外是每插入一次的新页 append/文件扩展、未分配洞、unwritten extent 的特殊不对齐写；它们可升级 exclusive lock并处理分配/元数据。但每插入只有约一次 append，而 35.8 次中的绝大多数是旧页 overwrite；none 臂的 18559/s也证明 append 分配路径没有形成当前上限。

**估计贡献：0%–10%，通常低于 5%；主导项判定为否。置信度 0.90。** 若预分配没有覆盖新增 100k 页，exclusive 分支主要影响那一次 append，而不是 35.8 个 backlink patch。

### 2.3 次要但确定：page-cache invalidation、folio/xarray 争用和读 miss

这不是仅在“恰好与 buffered read 同时”的罕见事件：patch 自己先用 buffered `read_at()` 做整页 RMW，确保目标页刚进入/命中 page cache；紧接着同范围 DIO 写必走 pre-invalidation，完成时再做 post-invalidation。因此 clean 场景下每个成功 patch 仍近似有：

1. xarray 查找/摘除 clean folio；
2. folio lock/refcount/LRU 操作；
3. 与并发 buffered pread/readahead 对同一 folio及 mapping xarray 的争用；
4. 丢失该图页的缓存局部性，使后续搜索产生真实 NVMe read 或 readahead。

约 212k 次 patch/s意味着同量级的强制 page-cache 驱逐。相对题给约 1.5M buffered read/s，这足以产生可见 cache churn；post-invalidation 还专门处理 DIO 期间被并发读重新装入的页。通常真正昂贵的是**失效后的设备 read 与读写队列干扰**，纯 xarray 锁指令本身不是全部。

**估计贡献：吞吐差距中的 25%–45%，即约 5–9 个百分点；其中纯 xarray/folio 锁更可能只有 1–3 个百分点，其余来自 cache miss/read I/O。置信度 0.68。** 此区间与上一项有交互，不能机械相加；应以 `workingset_refault`、`filemap:mm_filemap_delete_from_page_cache`、块层 read IOPS/latency 和 pread cache-miss率拆账。

### 主导机制排序

```text
同步随机 DIO completion + 应用 mutex/seqlock 窗口拉长
    > DIO invalidation 导致的 buffered-read cache churn / read I/O
    > folio/xarray 的纯 CPU 锁争用
    > ext4 exclusive inode lock（旧页 overwrite 基本不走）
```

## 3. P0.2 flush 模式比较与明确推荐

前提是批内逻辑更新已在用户态页缓存中按唯一页合并。比较的是数万到十余万次**唯一 4 KiB 页** flush，而非原实验每插入 35.8 次即时写。批内搜索若故意只看旧 committed snapshot，可以不查用户态 dirty cache；发布之前必须完成“新快照可被正常读路径读取”的转换。

| 模式 | flush/发布吞吐 | 对并发 buffered 搜索读 | 外部 O_DIRECT reader 可见性 |
|---|---|---|---|
| a. 并行同步 O_DIRECT `pwrite` | 同页合并已拿到最大收益；但每页仍一个 syscall并等待 completion，必须等全批完成后才能安全 publish。同步线程数过多会抬高尾延迟 | 每页 pre/post invalidation；破坏热点图页缓存，flush 与搜索争设备读写队列 | 每个 DIO completion 后该页可见；批级一致性仍须 barrier。`finalize()+fsync` 后满足题设 |
| **b. buffered `pwrite` + `sync_file_range(WRITE)`** | `pwrite` 只复制到内核 page cache；range call 用 `WB_SYNC_NONE` 启动后台写回，不等完成。可在 page-cache 接管全部页后 publish并继续排水。受约 287k buffered pwrite/s 的 exclusive `i_rwsem` 平台限制，但唯一页合并后 syscall 数已大幅下降 | **最好**：不失效 page cache，发布后 buffered pread直接读取新版 cache；后台 writeback会占设备带宽，但不制造 DIO 驱逐/refault | publish 后若外部 DIO 强行读，内核 DIO read 会先对重叠 dirty range write-and-wait，能协调但代价高且无批快照。题设只在 `finalize()+fsync` 后读，届时完全满足 |
| c. buffered `pwrite` 后不主动回写 | 单批前台时间最低，与 b 的进程内可见性相同 | 短期最好；长期 dirty 页累积到阈值后，`balance_dirty_pages`/内存回收会在不可控位置节流，并与搜索突发争盘 | `finalize()+fsync` 后满足；此前同 b，无跨页批级保证 |
| d. io_uring 批量 DIO | 减少提交 syscall、用有界 QD隐藏单次 latency，优于 a 的同步等待形态；但所有 CQE 完成前不能把“磁盘快照已发布”当真。64 同步线程已经提供较高 QD，故收益不是数量级 | 与 a 使用同一 iomap DIO 协调：**同样 pre/post invalidation和 cache churn**，且高 QD flush 更容易挤压搜索读 | 全部 CQE 完成后页可见；`finalize()+fsync` 后满足 |

### 推荐：b

具体发布边界应是：

```text
批内用户态页修改/合并
  -> 对每个唯一 dirty page 做一次 buffered pwrite
  -> 所有 pwrite 返回（数据已经由 inode page cache 接管）
  -> sync_file_range(data range, SYNC_FILE_RANGE_WRITE)
  -> publish(release)，不等待 writeback completion
  -> 后台回写与下一批计算重叠
...
finalize:
  -> 刷净仍在用户态 cache 的页
  -> 更新 meta / ftruncate
  -> fsync(fd_) 返回
  -> 才允许外部 eval 进程重开并 O_DIRECT 读取
```

选择 b 而不是 a/d 的核心不是 buffered syscall 单项一定比满 QD DIO 快，而是本工作负载**搜索读始终依赖同 inode page cache**。a/d 每写一页都主动摧毁搜索缓存，恰好与 1.5M buffered reads/s 对冲。P0.2 已通过“每唯一页一次”消掉最大写放大，此时没有必要再为了绕过 buffered inode lock，重新引入每页 DIO coherence 税。

选择 b 而不是 c，是为了控制长期运行的 dirty backlog。单批可能产生约 40 MiB–400+ MiB dirty data；完全依赖全局回写阈值会使后续某批的 `pwrite` 或内存分配突然承担节流。`SYNC_FILE_RANGE_WRITE` 明确开始排水、但不要求批发布等待。内核源码也明确说明它是 `WB_SYNC_NONE` 的 asynchronous flush，不适合数据完整性；这恰好符合“publish 只要求进程内可见、最终才 fsync”的语义。它在设备 request 资源耗尽时仍可能阻塞一段时间，所以要监控调用耗时和 dirty backlog；这不改变模式选择。

287k buffered pwrite/s 对十余万唯一页的 flush 仍可能成为约 0.35 s 级阶段上限，故后续优化优先级应是：连续 offset 排序、合并相邻页为较大 buffered pwrite/pwritev、与批计算流水重叠；不是先上 io_uring DIO。若未来搜索读也改为统一用户态 buffer pool/O_DIRECT，或测得 buffered inode lock在“每唯一页一次”后仍是明确 P0 瓶颈，才重新评估 d。

### 外部 reader 的边界

- fd 是不是另一次 `open()` 不影响内核 coherent page cache；同 inode 的 DIO read会与 dirty buffered range 协调。
- 但 finalize 之前不应把这一点当成 API：reader 可能看到一部分已发布/一部分未发布页，也可能为每个范围触发同步写回；seqlock还是进程私有，不能保护独立进程。
- `finalize()` 完成所有用户态 cache 到 page cache 的复制、写 meta，并让 `fsync()` 返回后，再启动/重开 eval，b/c 与 a/d 在题设的可见性要求上等价。

## 4. b/c 下 seqlock 能否防 buffered 撕页

### 能，原因是检测并重试，而不是假设内核 memcpy 原子

整页 buffered `pwrite` 最终在内核 page cache folio 内执行 copy；buffered `pread` 从 folio copy 到用户 buffer。不能把 4 KiB copy 当成原子操作，也不应把正确性寄托于读写双方一定共用同一把 folio lock。允许如下重叠：

```text
reader: v1=even(old) ---- pread/copy_to_user(old+new mixture) ---- v2
writer:             version++ -> odd; pwrite/memcpy; version++ -> even(new)
```

只要 writer 的 odd/even 覆盖完整 `pwrite` syscall，reader 的第二次 acquire load必然看到版本变化，丢弃混合 buffer并重试。其余交错也安全：

- reader 看到 odd：不发起/不接受 pread，等待重试；
- reader 两次都看到同一 even：期间没有成功的本页 writer，接受旧版或新版；
- writer 在 reader 第一次版本读取之后、pread 之前完成：第二次版本不同，仍重试；
- 多 writer：同页 mutex/缓存 shard lock必须把它们串行，否则 seqlock本身不提供 writer-writer互斥。

现有 `fetch_add(acq_rel)` 置 odd、完成 syscall 后 `fetch_add(release)` 置 even，读者前后 acquire load 的结构满足该检测协议。严格形式上 `uint32_t` 存在约 2^31 次完整页写后版本绕回的 ABA；不可能在一次 4 KiB pread 窗口内发生，可视为工程上安全。

### 三个必须保留的条件

1. **不能在 buffered `pwrite` 返回前置 even。** 返回表示 page-cache copy 已完成；b 的 `sync_file_range` 和后续物理 writeback不需要继续保持 odd，因为 buffered reader读的是已经完整更新的 folio内容。
2. **用户态 dirty cache 不属于 buffered fd 的可见数据。** 若批中搜索按设计只读旧 committed snapshot，可以等批末再逐页 `pwrite`；但 publish 必须在所有待发布页的 buffered `pwrite` 返回之后。seqlock不能弥补“数据尚未复制进内核 page cache”。若要让批中读看脏页，必须走用户态 cache overlay，并对 overlay 本身使用同样的版本/锁协议。
3. **保护只覆盖本进程、走 `read_node_page()` 的读者。** 独立进程没有 `page_versions_`；其安全边界是停止并发更新后的 `finalize()+fsync`，不是此 seqlock。

由此，b/c 不降低当前进程内无撕页保证；相反，它们把 odd 窗口从“等待 NVMe completion”缩短为“向 page cache 做一次内存 copy”，会显著减少 reader retry 和热点页 writer 排队。

## 最终选型

**P0.2 flush 使用 b：buffered pwrite（每唯一脏页一次）+ 批末 `sync_file_range(SYNC_FILE_RANGE_WRITE)` 异步下发；pwrite 全部返回后即可 publish，只有 `finalize()` 才等待 `fsync()`。**

这是当前“buffered 搜索读 + 同 inode 高频随机页更新 + 无批级崩溃一致性要求 + eval 仅在 finalize 后启动”约束下的明确最优选择。a 被同步 completion和 cache invalidation拖累；d 只改善提交/等待形态、不消除同样的 coherence 税；c 缺少长期 dirty backlog控制。
