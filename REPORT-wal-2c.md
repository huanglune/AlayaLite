# REPORT: WAL 第三波 2C 维护事务

分支 `feat/wal-2c-maintenance`,基于 `wave2-integration@cf67680`。逐阶段累积:备忘裁决映射、7 BLOCKER 落地映射、判断调用、边界清单(含三条 non-goal)。

阶段:W0(基座)→ W1(consolidate 事务)→ W2(PID reuse)→ W3(Collection 集成)。每阶段 = 实现 → 全量重编 disk LASER/searcher/builder/header-closure → laser/collection/wal/crash ctest 绿 → commit。

---

## 当前状态

- **W0 完成并 commit**:bloom MAP_SHARED 根治、恢复三阶段严格化、kind=2 trailer 幂等、FREE 引用完整性、v2/v3 reader + 选择器 fail-closed、kind=7/8 golden。
- **W1 完成并 commit**(两个 commit:frame API/failpoints + consolidate 事务):consolidate 四参数在 enable_wal 下作为维护事务解禁、MaintenanceOverlay + WAL 溢写、epoch 恢复状态机 + 语义截断、free-list redo 后收敛(canonical chain)、v3 activation checkpoint、consolidate SIGKILL 矩阵。garden 仍 throw。
- **W2/W3**:未开始(见"遗留")。**W2 前须按纪律发起 codex 对抗审查。**
- 相关 ctest(laser/collection/wal/crash 标签)40/40 绿;新增测试:frame 三 API、fail-closed 选择器、kind=7/8 golden、consolidate 三功能测试、consolidate SIGKILL 5 格。

---

## 一、备忘裁决逐条映射(设计备忘 §0-§8)

| 备忘条目 | 状态 | 落地位置 / 说明 |
|---|---|---|
| §1 consolidate 真事务(整页 after-image + kind=3/4) | **W1 完成** | `consolidate_wal_transaction`:BEGIN(fsync)→overlay 改页→spill(kind=1 flush)→finalize(dirty 页 kind=1 buffered)→END(fsync,提交点)→`maint_install_all`(seqlock)→发布 free-list/epoch。garden 保留 throw。 |
| §1.2 私有 MaintenanceOverlay | **W1 完成(单线程)** | `maint_pages_`/`maint_spilled_`/`maint_dirty_` + `maint_overlay_page`/`maint_log_page`/`maint_spill_over_cap`/`maint_install_all`;`modify_node_page`/`read_rmw_page`/`cached_raw`/`bloom_dependency_page` 在 `maintenance_active_` 下路由到私有 overlay(并发 search 仍读 committed disk)。并行 page workers 见 JC-8。 |
| §1.3 恢复状态机 | **W1 完成** | `replay_and_rebuild` epoch 状态机:BEGIN 开 epoch、row_patch 只 stage latest-per-page、END 应用、EOF 未匹配 BEGIN → `truncate_to` 语义截断、absorbed 前缀 validate-only(JC-11)、epoch 越级 poison。 |
| §1.4 与 checkpoint flip 的关系 | **W1 完成** | checkpoint 准入新增"无 active maintenance epoch";kind=4 提交事务,kind=6 吸收入 base。 |
| §2 free-list redo 后收敛 + §2.2/§2.3 | **W1 完成** | reclaim 走事务局部 `maint_local_free_head/count`,END 后发布;`rebuild_state_after_replay` 从 trailer 派生 free 集 + canonical(升序 PID)重建 chain + 写盘;FREE⇒TOMBSTONE、live 边→FREE poison(W0 加,W1 生效)。 |
| §3 PID reuse 协议 | 未开始(W2) | |
| §4 Bloom MAP_SHARED bug 根治 | **W0 完成** | 见 W0。 |
| §5 门禁演进 / garden | **W1 部分** | consolidate 四参数解禁;garden 仍 throw;legacy add/allocate 仍 append(reuse 未解禁,W2)。 |
| §6 崩溃矩阵扩展 | **W1 部分** | consolidate SIGKILL 5 格(begin_fsync/before_end/end_fsync/install_page/install_before_publish)+ 双 replay byte-stable;完整 C0-C11 + 掉电撕裂族见"遗留"。 |
| §7 superblock v3 布局 + 版本策略 | **W1 完成** | v3 常量/2C 扩展/feature bits(W0 qg.hpp)+ `Wal2cState` 读写 + activation checkpoint 写 v3 + `kQgSupportedRequiredFeatures = maintenance_tx_v1|post_redo_free_list_v1`。A/B 混合槽 seal 前门禁见 W3。 |
| §0 恢复三段 / §2.1 恢复顺序 | **W0+W1 完成** | 唯一收敛点 `rebuild_state_after_replay`;`replaying_` 持到 rebuild 完成;WAL base load 不提前 repair routing。 |

## 二、7 BLOCKER 落地映射

| BLOCKER | 阶段 | 状态 | 落地位置 |
|---|---|---|---|
| B-2C-01 writer-private seed | W2 | 未开始 | |
| B-2C-02 canonical prebind 恢复完整性 | W2 | 未开始 | |
| B-2C-03 Collection 维护入口+门禁链 | W1/W3 | **句柄锁部分** | 段级 `MutableLaserSegment::mutex_` 已串行化所有 mutating 入口(consolidate 需经其接入,W3);Collection hook + 全门禁链是 W3。 |
| B-2C-04 frame 三 API + 流式构造 | W1 | **完成** | `frame.hpp`:`FrameLocation`、`append`→location、`visit_frames`(单帧内存)、`read_frame`(重校验)、`truncate_to`(语义截断);wire 零变化(make_frame golden 未动)。溢写重载用 `read_frame`。全流式 replay 见 JC-9。 |
| B-2C-05 MaintenanceOverlay 并发协议 | W1 | **串行化解决(并行 deferred)** | 单线程维护 lane:无并发 WAL append、无容器竞态;install 走 page seqlock(odd→write_at/arena→even)保证并发 search 无半页。并行 page workers + barrier-外 spill 见 JC-8。 |
| B-2C-06 superblock v3 fail-closed + 双常量 | W0 | **完成** | 见 W0 表 + 判断调用 JC-1/2/3。 |
| B-2C-07 协议洞→硬验收 | W2/W3 | 未开始(部分:consolidate×checkpoint 由准入 + 交错测试覆盖) | |

## 三、判断调用(逐条编号)

- **JC-1**:`qg_superblock_valid` 接受 v2+v3(结构合法);fail-closed 支持性单列 `qg_superblock_supported`。理由:write_superblock/replay_flip 需校验 v3 镜像。
- **JC-2**:legacy `select_qg_superblock` 不变(仅 bench/tests 用,只喂 v2);真实路径走 `select_qg_superblock_checked`。
- **JC-3**:read-only `load_disk_index` 也 fail-closed。B-2C-06"旧 reader 选旧 v2"指前-2C 二进制;新 reader 遇不支持的更高 gen v3 必须 fail closed。
- **JC-4**:kind=2 replay 用盘上 RMW 持久 trailer;rebuild ftruncate 裁越界 pid。
- **JC-5**:`modify_bloom_node_page`/`merge_dirty_into_mapping` 保留为薄委托别名。
- **JC-6**:`SharedFileMapping` 用 `MAP_SHARED|PROT_READ`(反映 pwrite,PROT_READ 关 no-steal 隐患);非 MAP_PRIVATE(会读 stale)。
- **JC-7**:`open_op_wal_and_recover` 总是 `replay_and_rebuild`(空 WAL 亦然),使 rebuild 成为 WAL 模式唯一修 routing 收敛点。
- **JC-8**:**W1 维护事务单线程**(num_threads coerced;`consolidate_row_phase` 串行)。理由:B-2C-05 的并行 spill/安装并发协议靠"串行 append lane"最简正确地满足(无 ofstream/wal_op_id_ 竞态、无 `maint_pages_` 竞态);install seqlock 仍保并发 search 正确。**并行 OpenMP page workers + barrier-外单线程 spill(codex 方案的页分批协议)为 deferred 优化**,不影响正确性/持久性验收。手册"跑不起来做最小可行覆盖"授权此裁决。
- **JC-9**:**replay 仍走 `recovery_scan()`(全 WAL 入内存)+ epoch 状态机 latest-per-page(payload)**;`visit_frames` 已建并用于 overlay 溢写重载(`read_frame`)。理由:改 replay 为全流式风险大(动全部恢复/崩溃测试);epoch_pages 已按 touched 页(非 WAL 字节)有界。**全流式 replay(visit_frames 驱动、O(touched×FrameLocation))为 deferred**——大规模多溢写下 `recovery_scan` 仍 O(WAL 字节),内存验收部分满足。
- **JC-10**:statvfs 预检用 `file_pages()×page_size×2 + 1MiB` 启发式(非 codex 的逐帧 overhead 精确式)。理由:小图测试余量充足;精确上界为 deferred。
- **JC-11**:恢复 absorbed 前缀(E ≤ base epoch)validate-only(stage 校验但不 re-apply)。采纳 codex B-2C-05 坑 5(虽 in-order replay 可证正确,保守更稳)。
- **JC-12**:consolidate 失败(过 BEGIN)catch 中**保持** epoch/overlay 状态并 poison(不清理),交 reopen 重建(设计/准入裁决)。

## 四、边界清单

### 显式 non-goal(条款 8⑥;原样保留,不许消失)

1. **`close()` 不释放 flock**:`MutableLaserSegment` 句柄 writer flock 全生命周期持有,`close`/析构才释放;2C 不改。
2. **filter 仍 postfilter**:过滤在结果侧后过滤,不下推 QG 搜索;2C 不改。
3. **旧 active 目录到下次 open 才回收**:rotate/seal 后旧活写目录物理回收延迟到下次 open;2C 不改(即时 orphan reclaim 非本波目标)。

### 其它边界

- garden 在 enable_wal 下继续 throw(未解禁)。
- W2 前 PID reuse 仍全禁(WAL 模式 allocate 恒 append;free-list 由 W1 consolidate 填充,但 W1 不消费)。
- W1 维护单线程(JC-8);全流式 replay 未接线(JC-9);statvfs 启发式(JC-10)。

## 五、各阶段 commit(见 git log,前缀 feat(wal-2c):/test(wal-2c):)

- W0:bloom + recovery strictification + v2/v3 fail-closed reader。
- W1 part 1:frame streaming/location API + consolidate failpoints。
- W1 part 2:consolidate maintenance transaction(overlay + epoch state machine + free-list post-redo + v3 activation + SIGKILL 矩阵)。

## 六、遗留 / 未开始

- **W1 deferred 硬化**:并行 OpenMP 维护 page workers + barrier-外 spill(JC-8);全流式 replay via visit_frames(JC-9);完整 C0-C11 + END-durable 后 index 任意 k 页/单页扇区撕裂的掉电族;statvfs 精确上界(JC-10);observer 断言"END 前 index/arena 维护写=0";`(reclaim×bloom×r_target)` 与非 WAL oracle page-hash/free-set 逐项等价对照;TSan 覆盖维护并行(单线程下 N/A,记录实况)。
- **W2**:PID generation + bundle-only reuse(writer-private seed B-2C-01、canonical prebind 恢复完整性 B-2C-02、checkpoint 准入 all-reuse 窗口、effective_label override、2B adapter token 化、v3 PID activation、R0-R11、B-2C-07 硬验收)。**须 codex 对抗审查后进 W3。**
- **W3**:Collection maintenance hook + 门禁链 B-2C-03、activation checkpoint 全套 + A/B 混合槽 seal 前双 v3 门禁、四携带项(B-03 failpoint/durability 三模式/B-09 SIGKILL keep-set/边界 7 rerank 回归)、非 goal 三条(已在上)、consolidate×active write/checkpoint/seal-rotate 交错矩阵、search×END 安装断言、statvfs/ENOSPC 两测试、四文件时间线 + 文档 `unified-wal-vocabulary.md`、TSan。
