# REPORT: WAL 第三波 2C 维护事务

分支 `feat/wal-2c-maintenance`,基于 `wave2-integration@cf67680`。逐阶段累积:备忘裁决映射、7 BLOCKER 落地映射、判断调用、边界清单(含三条 non-goal)。

阶段:W0(基座)→ W1(consolidate 事务)→ W2(PID reuse)→ W3(Collection 集成)。每阶段 = 实现 → 全量重编 disk LASER/searcher/builder/header-closure → laser/collection/wal/crash ctest 绿 → commit。

---

## 当前状态

- **W0 完成并 commit**:bloom MAP_SHARED 根治、恢复三阶段严格化、kind=2 trailer 幂等、FREE 引用完整性、v2/v3 reader + 选择器 fail-closed、kind=7/8 golden。
- **W1 完成并 commit**(两个 commit:frame API/failpoints + consolidate 事务):consolidate 四参数在 enable_wal 下作为维护事务解禁、MaintenanceOverlay + WAL 溢写、epoch 恢复状态机 + 语义截断、free-list redo 后收敛(canonical chain)、v3 activation checkpoint、consolidate SIGKILL 矩阵。garden 仍 throw。
- **W2a plumbing 完成并 commit**(续程,e363c39):`pid_generation` label-slot 字段端到端激活(PidBinding/PidToken/find_binding、serialize 写真 gen、effective_label 显式-binding-优先),复用未开启⇒40 绿。**W2 复用核(runtime+replay 原子同落)designed-with-runway,未 landed**(见"七、W2 复用核 runway";JC-15 战略停点)。
- **W1 补账完成并 commit**(续程,2e85883):完整 C0-C11 SIGKILL 崩溃矩阵(13 格,含 tiny-cap 强制 spill 的 C4/C5)+ consolidate 掉电族 roll-forward 测试(END durable → S_new over 全丢/全留 install);`(reclaim×bloom×r_target)` 5 格与非 WAL oracle 的 live-row 逐字节 + tombstone/free 集合一致;statvfs 精确上界(design §1.3,取代 JC-10);write_at BUILD-phase steal 断言("END 前 index/arena 维护写=0")。
- **W2/W3**:未开始(见"遗留")。**W2 前须按纪律发起 codex 对抗审查。**
- 相关 ctest(laser/collection/wal/crash 标签)40/40 绿;新增测试:frame 三 API、fail-closed 选择器、kind=7/8 golden、consolidate 三功能测试、consolidate SIGKILL C0-C11(13 格)、consolidate 掉电 roll-forward、oracle 等价 5 格。

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
- **JC-13**(W1 补账,取代 JC-10):`maint_statvfs_preflight(reclaim_slots)` 用 design §1.3 精确上界。推导:repair/reclaim 各按页序处理、完成页本相位内不再触碰,故每页每相位至多 log 一次 ⇒ kind=1 帧数 ≤ repair_pages + reclaim_pages(混合页计两次=其至多两 log)。每页帧 overhead = WAL7(kHeaderBytes 36 + kTrailerBytes 4) + SEGMENT_OP 头 19 + row_patch 定长 20 = 79;marker 帧 = 40+19+8 = 67;另加 CoW 文件系统每装入页 ×page_size 的保守项 + max(16MiB, base/20) 余量。全程 saturating 乘加(溢出即 UINT64_MAX 拒绝)。repair_upper=reclaim_upper=committed_pages 是安全上界(bloom 模式候选页 ≤ committed_pages)。
- **JC-14**(W1 补账,量化 JC-9 内存验收缺口):全流式 replay 仍**deferred**。量化依据:单个未 checkpoint 的维护 epoch 其 WAL 帧数 ≤ repair_pages+reclaim_pages ≤ 2×committed_pages(见 JC-13),故 `recovery_scan()` 峰值 ≈ 2× 索引 live 页字节(与加载索引同阶,非无界);epoch 状态机的 `epoch_pages`(latest-per-page)独立地 ≤ committed_pages×page_size(不随 spill 次数 S 涨)。真正的 O(WAL 字节)只在"多 epoch 长期不 checkpoint"下累积,而 checkpoint 的 WAL reset 在实践中封顶。将 `recovery_scan()` 改为 `visit_frames` 驱动(API 已建、已用于 overlay 溢写重载)可把 replay 峰值降到 O(max-frame + touched×page)——因动全部恢复/崩溃路径,风险大,列为 deferred 优化。"END 前维护写=0" 由 write_at 的 BUILD-phase 硬断言直接保证(任何隐蔽 steal 即 poison,C0-C7 崩溃格与 oracle 测试全绿即其反证)。
- **JC-15**(战略停点,接续 W2a 绿边界):W2 复用核=B-2C-01 writer-private 可见性 + canonical prebind 重写 + staged-replay + B-2C-02 恢复完整性 + §3.4 mixed-HWM 代数 + checkpoint 准入 + v3 pid 激活 + 2B adapter token 化 + R0-R11,≈800 行本波最高风险协议,且**必须 runtime+replay 原子同落**(备忘 §3.5:交错 reuse 的 torn bundle 会覆盖 <committed 的真实地址⇒corruption;无更小正确切片)。因剩余预算须覆盖强制 codex 对抗审查 + W3,不冒"按测试绿但协议不成立"(手册设 codex 中审的正因)之险 rush 落地。裁决:代码停在 W2a 绿边界(前程先例=W0/W1 后停并诚报),把复用核作 runway 冻结进本 REPORT §七 + 后台 codex 参考实现(brief/report 在 job tmp:wal2c-w2-codex-brief.md / wal2c-w2-codex-report.md),供续程按 runway 直接落地。已验证前提:robust_prune/patch_reverse_edge_impl/full_reverse_recompute 均写者独占(公开 search 走 FastScan 不经它们)⇒writer-private `bundle_reveal_` 判据成立。

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

- **W1 deferred 硬化**(续程后现状):完整 C0-C11 崩溃矩阵 ✅、掉电 roll-forward 族 ✅、statvfs 精确上界 ✅(JC-13)、`(reclaim×bloom×r_target)` 非 WAL oracle 等价 ✅、"END 前维护写=0" write_at 硬断言 ✅。**仍 deferred**:并行 OpenMP 维护 page workers + barrier-外 spill(JC-8,单线程串行 lane 是 minimal-viable-correct);全流式 replay via visit_frames(JC-9/JC-14,已量化内存上界);TSan 覆盖维护并行(单线程下无并发维护写,tsan preset 亦 ALAYA_ENABLE_LASER=OFF,记录实况)。
- **W2**:PID generation + bundle-only reuse(writer-private seed B-2C-01、canonical prebind 恢复完整性 B-2C-02、checkpoint 准入 all-reuse 窗口、effective_label override、2B adapter token 化、v3 PID activation、R0-R11、B-2C-07 硬验收)。**须 codex 对抗审查后进 W3。**
- **W3**:Collection maintenance hook + 门禁链 B-2C-03、activation checkpoint 全套 + A/B 混合槽 seal 前双 v3 门禁、四携带项(B-03 failpoint/durability 三模式/B-09 SIGKILL keep-set/边界 7 rerank 回归)、非 goal 三条(已在上)、consolidate×active write/checkpoint/seal-rotate 交错矩阵、search×END 安装断言、statvfs/ENOSPC 两测试、四文件时间线 + 文档 `unified-wal-vocabulary.md`、TSan。

---

## 七、W2 复用核 runway(designed, not landed — 续程直接落地蓝图)

坐标基于 527154c + 续程 commit(2e85883/e363c39)。已 landed 的地基(W2a):`PidBinding{gen,label}`/`PidToken{pid,gen}`、`LabelBindings::find_binding`、serialize 写真 gen、`label_working_` 改 PidBinding、`effective_label` 显式-binding-优先、`pid_generation_activated_`(adopt_label_state 从 v3 pid feature bit 推)、load 的 gen!=0 poison 由该 flag 门控。**以下未 landed**,须 runtime+replay 原子同落(否则 corruption):

### 1. 激活门(v3 pid 特征)
- `UpdateParams` 加 `bool enable_pid_reuse=false`;ctor `enable_pid_reuse_ = params.enable_pid_reuse && enable_wal_`(init 点 qg_updater.hpp:579 邻域)。
- `ensure_pid_reuse_activated()`(仿 `ensure_maintenance_activated` @3281):若 base 非 v3-pid,持 checkpoint_mutex_ 调 checkpoint 写 v3,`required_feature_flags |= kQgFeatPidGenerationV1|kQgFeatCanonicalPrebindV1|kQgFeatMutableLabelSlotV1`(三位捆绑,qg.hpp `qg_required_features_self_consistent` 强制),写 `pid_reuse_activation_sb_generation=next.generation`。checkpoint 的 v3 分支 @1779 需扩:activating pid reuse 时 OR 三位 + 写 activation gen。
- **红线雷**:`qg.hpp:196 kQgSupportedRequiredFeatures` 现=maint|postredo,**必须同步扩** pid_gen|canonical|mutable_label,否则 selector fail-closed 打不开自己刚激活的库(B-2C-06)。
- checkpoint(1712)admission 加拒 active bundle/reserved reuse;label slot 判定 @1757 `current_count>label_count_` 改 content-revision/dirty(复用换绑不增 count,否则丢新 gen/label)——用 snapshot 内容 revision 或脏标志。

### 2. reserve_bundle_pids(free-list first)
- 新 `std::vector<PidToken> reserve_bundle_pids(size_t n)`:先 pop free-list 前 k(≤n)个(pop_free_slot @2641,或直接按 canonical 升序取以与 rebuild @5495 一致),每个 token={pid, old_gen+1}(old_gen 来自持久 binding:`label_snapshot()->find_binding(pid)`,无则 0⇒新 gen 1);余 n-k 个 append(next_append_id_),gen=0。`row_generations_[id]>=UINT32_MAX` 的 PID 跳过复用(wrap 禁,永久 tombstone)。设 `bundle_active_=true`、`bundle_reservation_count_=n`(checkpoint 准入读)。

### 3. writer-private 可见性(B-2C-01;已验证前提成立)
- 成员:`std::unordered_set<PID> bundle_reveal_`(reserved 且已写入本 bundle 的 PID)、`PID bundle_private_entry_=kPidMax`。
- helper `bool writer_hidden(PID id) const { return is_hidden(id) && bundle_reveal_.count(id)==0; }`(空时等价 is_hidden⇒现有测试不变)。
- 替换**写者路径** is_hidden→writer_hidden 四处:robust_prune @3907、patch_reverse_edge_impl @3998、full_reverse_recompute @4200 与 @4225。**勿动**查询/beam-frontier/consolidate 的 is_hidden(898/912/consolidate 区)。
- search_for_insert @3841 邻域加 writer-private 播种:`if (bundle_private_entry_!=kPidMax && bundle_private_entry_<snapshot) sp.insert(bundle_private_entry_,FLT_MAX);`(仅写者经此函数,查询走 FastScan⇒不泄漏)。appended 行仍靠 `insert_visible_override_` 地板;reused 行靠 bundle_reveal_ 揭示 + inline backlink 达 reachability;delete-all→all-reuse 空 live 图靠 private_entry=首个已写 reserved PID。

### 4. canonical prebind commit_physical_bundle(§3.5;新分支,保留 2A 交错给未激活)
分支:`enable_pid_reuse_ && pid_generation_activated_` 走 canonical;否则现交错路径(@1094-1211 原样,现有测试不激活⇒不变红)。canonical 序:
1. reserve_bundle_pids(n) → tokens。
2. 预建 next snapshot:对每 token `bindings[pid]=PidBinding{token.gen, labels[i]}`(复用换绑=替换,非 emplace)。
3. append **全部** kind=7(row_op 序,`encode_label_bind(...pid, token.gen, label...)` @1155,gen 传真值)。
4. 全部行写入:对每行 `insert_with_id(vec, token.pid)`(caller-assigned PID,只出 kind=1 行/反向边);写前把已写 reserved PID 加 bundle_reveal_、appended 抬 insert_visible_override_;首行设 bundle_private_entry_。
5. `wal_append(kind=8, fsync)`(new_hwm=old+append_count,binding_count=n)。
6. 发布序(§3.6,严格):store_label_snapshot → routing/新入口 refresh → 对 reused PID `mirror_deleted_erase+clear_hidden`(reuse 可见点)→ committed_.store(new_hwm) → live_count += n。清 bundle_reveal_/private_entry/insert_visible_override_/bundle_active_。
- 失败(过首个 kind=7)一律 poison(保状态),reopen 重建。

### 5. replay canonical bundle lane(§3.5 + W1 设计 §3;与 maintenance epoch lane 互斥)
- replay_and_rebuild(5066)加第三态 `in_canonical_bundle`(仿 in_epoch):
  - kIdle 见 kind=7 且 `op.segment_generation>=pid_reuse_activation_gen`⇒开 canonical lane(记 txid、清 staged);否则 legacy(现 stage-by-txid + kind=1 立即)。
  - lane 内:kind=7(同 txid)累积 token/gen;kind=1 stage latest-per-page(**不**立即写);kind=8(同 txid)做验证后 apply+promote;其它 op⇒poison;EOF⇒丢弃 staged(torn)。
- kind=8 验证(**B-2C-02 + §3.4**,任一败即 poison):
  - 每 bound PID 在本 bundle 有最终 staged page(FrameRef);从该页 image 验 trailer 为 live;
  - reused PID(gen>0):事务前状态(base/前缀)确为 FREE 且 `new_gen==old_gen+1`;
  - gen=0 PID 恰覆盖 dense `[old_hwm,new_hwm)`;`append_count+reuse_count==binding_count`;`new_hwm==old_hwm+append_count`;all-reuse⇒`new_hwm==old_hwm && binding_count>0`;txid 严格增、applied 不退。
  - 通过⇒apply staged pages(write_at)→label_working_ 换绑{gen,label}→committed/txid/applied 前进。
- rebuild_state_after_replay(5416)已 canonical free-chain + FREE 引用完整性(W1 完),复用后仍复用之。

### 6. 2B adapter token 化(§3.7;mutable_laser_segment.hpp + adapter)
- `label_to_pid_`(@464)升 `unordered_map<uint64_t, PidToken>`;commit 用返回的 `PhysicalBundleResult.rows`(需把 QGUpdater::commit_physical_bundle 返回改 `PhysicalBundleResult{vector<PidToken> rows, old_hwm, new_hwm}`,adapter 不推 dense range)。
- `tombstone(PidToken expected)`:锁内比当前 gen——更大⇒幂等 no-op,更小⇒corruption。reverse binding 仅 token 全等才删(勿误删新 incarnation)。previous 先满足 2B 同 segment id/gen 再 logical 反查 token。
- effective_label 已显式-binding-优先(W2a)。

### 7. R0-R11 + B-2C-07 硬验收(测试)
writer-private seed 两族(delete-all→all-reuse N>1 / mixed N>1,逐行同进程 + 首次 reopen + 二次 reopen 可搜);canonical prebind 三 poison(bind+publish 无 row_patch / 漏一绑定 PID 页 / 最终 trailer 仍 FREE-TOMBSTONE);checkpoint 准入两窗口(all-reuse 已 reserve、kind=7 前/后并发 checkpoint 不写 kind=6);三类首复用(base/legacy identity/explicit gen0)/连续复用 gen2/wrap 禁/stale token 无害/复用 PID 旧 label 被替换专项;R0-R11 SIGKILL 切点(备忘 §6.2)。

**最高风险 5 坑**:①激活忘扩 kQgSupportedRequiredFeatures⇒自锁库;②writer_hidden 漏替一处或误替查询路径⇒reused 行泄漏给并发查询/或写者链不到;③replay canonical lane 与 maintenance/legacy 三态互斥没写严⇒交错 poison 漏判;④B-2C-02 少验一条(尤其"reused 事务前确为 FREE")⇒crafted WAL 把仍被引用行交复用器;⑤checkpoint label-slot 仍用 count 判定⇒all-reuse 后新 gen/label 丢。**原子同落**:1-6 必须同一 commit(runtime 无 replay=reopen 丢数据;replay 无 runtime=死码但激活位已写会 fail-closed 老 reader)。
