# REPORT: WAL 第三波 2C 维护事务

分支 `feat/wal-2c-maintenance`,基于 `wave2-integration@cf67680`。本报告逐阶段累积:备忘裁决映射、7 BLOCKER 落地映射、判断调用、边界清单(含三条 non-goal)。

阶段推进:W0(基座)→ W1(consolidate 事务)→ W2(PID reuse)→ W3(Collection 集成)。每阶段 = 实现 → 全量重编 disk LASER/searcher/builder/header-closure → laser/collection/wal/crash ctest 绿 → commit。

---

## 当前状态

- **W0 已交付并 commit**(bloom MAP_SHARED 根治 + 恢复三阶段严格化 + kind=2 trailer 幂等 + FREE 引用完整性 + v2/v3 reader + 选择器 fail-closed + kind=7/8 golden)。40/40 相关 ctest 绿。
- W1/W2/W3:未开始(见文末"遗留")。

---

## 一、备忘裁决逐条映射(设计备忘 §0–§8)

| 备忘条目 | 状态 | 落地位置 / 说明 |
|---|---|---|
| §4 Bloom MAP_SHARED bug 根治 | **W0 完成** | `SharedFileMapping` 改 `MAP_SHARED|PROT_READ`(只读,JC-6);删 `mutable_data()`;`modify_bloom_node_page`→`modify_node_page`;`merge_dirty_into_mapping`→`flush_dirty`。所有 Bloom RMW 统一走 `write_at`(镜像 arena)。 |
| §2 free-list 唯一恢复收敛点 | **W0 部分** | `rebuild_state_after_replay` 唯一收敛点;`replaying_` 持到 rebuild 完成;WAL base load 不再提前 repair routing。free-list 事务发布是 W1。 |
| §2.3 额外完整性检查(FREE 引用) | **W0 完成** | `rebuild_state_after_replay`:FREE⇒TOMBSTONE、live 边→FREE 目标 poison(W0 下无 FREE 行,守卫惰性)。 |
| §3.7 kind=2 replay 幂等设 trailer | **W0 完成** | `replay_persist_tombstone_trailer`:replay 时盘上 RMW 设 tombstone 位。 |
| §7.1/§7.2 superblock v3 布局 + 版本策略 | **W0 部分** | v3 常量 + 2C 扩展布局 + feature bits 定义于 `qg.hpp`;选择器 fail-closed 分类完成。activation checkpoint 写 v3 是 W1。 |
| §0 恢复三段 / §2.1 恢复顺序 | **W0 完成** | 见"恢复严格化"。 |
| §1 consolidate 真事务 | 未开始(W1) | |
| §3 PID reuse 协议 | 未开始(W2) | |
| §5 门禁演进 / garden 门禁 | garden throw 保留(未动);consolidate/reuse 解禁在 W1/W2 | |
| §6 崩溃矩阵扩展 | 未开始(W1/W2) | |

## 二、7 BLOCKER 落地映射

| BLOCKER | 阶段 | 状态 | 落地位置 |
|---|---|---|---|
| B-2C-01 writer-private seed | W2 | 未开始 | |
| B-2C-02 canonical prebind 恢复完整性 | W2 | 未开始 | |
| B-2C-03 Collection 维护入口+门禁链 | W1/W3 | 未开始 | |
| B-2C-04 frame.hpp 三 API + 流式构造 | W1 | 未开始 | |
| B-2C-05 MaintenanceOverlay 并发协议 | W1 | 未开始 | |
| B-2C-06 superblock v3 fail-closed + 双常量 | W0 | **完成** | `qg.hpp`:`kQGFormatVersion`(v2)+`kQGFormatVersionV3`;`qg_superblock_supported`+`select_qg_superblock_checked`(-2=fail closed);`create_empty` 仍产 v2;读路径(read-only loader + QGUpdater recovery)均 fail-closed;单测 `QGSuperblockSelector.FailsClosedOnUnsupportedNewerSlot`。 |
| B-2C-07 协议洞→硬验收 | W2/W3 | 未开始 | |

## 三、判断调用(逐条编号)

- **JC-1**:`qg_superblock_valid` 改为接受 v2 **和** v3(结构合法=magic+version∈{2,3}+checksum),而非新增独立函数。理由:`write_superblock` 前置与 `replay_flip` 镜像校验在 W1+ 需校验 v3 镜像;fail-closed 的"支持性"单列为 `qg_superblock_supported`。既有 v2-only 测试不受影响。
- **JC-2**:legacy `select_qg_superblock`(三值返回)保持不变(现会把 v3 视为结构合法),仅供 bench/tests(只喂 v2)使用;真实恢复路径与 read-only loader 走 `select_qg_superblock_checked`。理由:最小化对冻结调用者扰动。
- **JC-3**:read-only `QuantizedGraph::load_disk_index` 也接入 fail-closed 选择器。理由:B-2C-06"旧 superblock-only reader 选旧 v2"描述的是**前-2C 二进制**(冻结产物)行为;新的 2C-aware reader 遇不支持的更高 generation v3 必须 fail closed,不得静默降级到 stale v2。
- **JC-4**:kind=2 replay 用直接盘上 RMW 持久 trailer,而非在 rebuild 追踪独立"kind=2-hidden"集。rebuild 末尾 ftruncate 裁掉越界 pid 页,无需额外守卫。
- **JC-5**:`modify_bloom_node_page`/`merge_dirty_into_mapping` 保留为薄委托别名,而非删除并改写全部调用点。理由:最小化 churn,语义等价。
- **JC-6**:`SharedFileMapping` 用 `MAP_SHARED|PROT_READ` 而非 `MAP_PRIVATE`。理由:只读共享映射仍反映 `write_at` pwrite(clean 页正确);`MAP_PRIVATE` 对"写前 fault-in、写后 evict"的页读到 stale 私有副本;no-steal 隐患由 PROT_READ(绝不 dirty 映射)关闭。
- **JC-7**:`open_op_wal_and_recover` 非 fresh 路径现**总是**调 `replay_and_rebuild`(空 WAL 亦然),使 `rebuild_state_after_replay` 成为 WAL 模式下唯一修 routing 的收敛点——移除 wal_mode 提前 `repair_routing_roots` 的必要配套。

## 四、边界清单

### 显式 non-goal(条款 8⑥;原样保留,不许消失)

1. **`close()` 不释放 flock**:`MutableLaserSegment` 句柄 writer flock 全生命周期持有,`close`/析构才释放;2C 不改。
2. **filter 仍 postfilter**:过滤在结果侧后过滤,不下推 QG 搜索;2C 不改。
3. **旧 active 目录到下次 open 才回收**:rotate/seal 后旧活写目录物理回收延迟到下次 open;2C 不改(即时 orphan reclaim 非本波目标)。

### 其它边界

- garden 在 enable_wal 下继续 throw(未解禁)。
- W0 下 PID reuse 仍全禁(free-list 在 WAL 模式恒空),FREE 完整性守卫因此惰性。

## 五、各阶段 commit

- W0:见 git log(前缀 `feat(wal-2c):` / `test(wal-2c):`)。

## 六、遗留 / 未开始

- W1:consolidate 真事务(MaintenanceOverlay + WAL 流式/FrameRef API B-2C-04 + epoch 状态机 + 语义截断 + overlay 并发协议 B-2C-05 + seqlock 安装 + 句柄锁 + v3 activation checkpoint + 四参数解禁 + 新 failpoint/SIGKILL/掉电矩阵)。
- W2:PID generation + bundle-only reuse(writer-private seed B-2C-01 + canonical prebind 恢复完整性 B-2C-02 + checkpoint 准入 + effective_label override + 2B adapter token 化 + v3 PID activation + B-2C-07 硬验收)。
- W3:Collection maintenance hook + 门禁链 B-2C-03 + activation checkpoint 全套 + 四携带项 + 边界 7 rerank 回归 + 全矩阵 + TSan。
