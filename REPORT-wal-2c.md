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
- **W1 codex 复审补账完成并 commit(续程,本波第三程)**:codex 复用核审查报告(`wal2c-w2-codex-report.md`)A 部分的 5 BLOCKER + 5 MAJOR,除 MAJOR-10(activation-ready 洞,codex 明令随激活原子组落地)外全部落地;详见"八、Step 1 状态映射"与 JC-16..JC-21。40/40 laser/collection/wal/crash 绿(crash 36 例含强化的 S_old/S_new 指纹族)。
- **W2 复用核 step 8 + codex 中审修复(本波第五程,final)**:①**codex 对抗中审**(manifest 强制,brief/report 在 job scratchpad `wal2c-codex-review-*`)对 step5+6 抓出 **5 BLOCKER + 4 MAJOR**,全部真问题,已修(JC-27):BLOCKER-1 spine 被后续 full-prune 重写⇒最终双向 spine pass(rows[i]↔rows[i+1],构建期逐行 spine 仅作 seed)+ full_reverse floor 用 writer-visible 而非 committed_;BLOCKER-2 clean writer 发布 entry 走 `repair_routing_roots`(与 replay 同确定性规则)⇒clean/replay entry 收敛;BLOCKER-3 reserve 发布 state 后异常 poison(统一异常边界从 reserve 返回起);BLOCKER-4 install cache page 先构造后插 map(禁 {pi,nullptr} 暴露并发 reader);BLOCKER-5 feature fail-closed 补全(self_consistent 三位全有全无+依赖 maintenance pair、v3 必带 maintenance、replay_flip 写前 supported 校验、adopt 断言 pid⇒maintenance);MAJOR-1 kind=8 校验 segment generation;MAJOR-3 B-2C-02 自检用 scratch 读不 re-resident spilled 页;MAJOR-4 空 slot 强制 summary=(0,0)。MAJOR-2(absorbed validate-only 不建 retained-kind=6 前态模型)保留 JC-22 挂账(validate-only 不 apply⇒不 corrupt,仅不拒畸形 WAL 后缀,R11 可测时补,强化 note)。②**R0-R11 SIGKILL 矩阵**(`test_segment_op_wal_reuse_crash`,6 例绿):R0(reserve 后)/R1(全 kind=7 后)/R4(kind=8 前)⇒S_old,R5(kind=8 buffered userspace 后 force 前:SIGKILL 丢 userspace buffer⇒S_old)/R6(kind=8 force 后)⇒S_new,+ power-loss 截断 unforced tail⇒S_old;每格 S_old/S_new 全图指纹(counts+trailer+live-row bytes+free chain+label bindings)精确落位 + **首恢复输出再 reopen 二次**字节稳定。③**额外硬族**(`test_qg_updater_reuse` 7 例 + `test_mutable_laser_segment_reuse` 2 例):activation+all-append、delete-all→all-reuse N>1 双 reopen、mixed、gen 0→1→2+summary、**BLOCKER-1/2 回归**(kFullPrune+prune_pool_cap=1 delete-all→all-reuse 可达性 clean+replay+checkpoint 三态一致)、garden throw、no-reuse 恒 v2;segment base pid 复用 gen-token+reopen shadowing、stale/future token ABA。④**legacy 逐字节不变**:全部现有 laser/wal/crash/segment/collection 测试零变红(2A wire/kind1-8 golden/2A 矩阵/2B 由既有套件锁)。JC-27。
- **W2 复用核 step 7 落地并 commit(本波第五程,adapter/segment token 化)**:codex §B.7。QGUpdater:`commit_physical_bundle_tokens`(返回 `PhysicalBundleResult`,legacy 2A body 抽为 `commit_physical_bundle_legacy_2a` + pair wrapper,现有 pair caller 零改)+ `durable_generation(pid)`(ABA 判据)。MutableLaserSegment:`label_to_pid_`→`unordered_map<uint64_t,PidToken>`、`token_for_label`、`tombstone(PidToken expected)` ABA(current>expected→stale no-op、current<expected→corruption throw、reverse map 仅全 token 相等 erase)、`commit_physical_bundle` 用 tokens 绑定(禁 dense range 推导)、`rebuild_reverse_index` 存 token + **JC-17 base 区 shadowing**(base pid binding 必 gen>0 且已激活,gen-0 base binding=非法 shadowing⇒fail closed)、`consolidate`/`free_count`/`pid_generation_activated` passthrough(reuse 可测,W3 门禁在其上)。Adapter:tombstone 目标 label **bundle 提交前**解析为 token 捕获,提交后用捕获 token tombstone(同 txid 换绑+reuse 不误杀新 incarnation)。功能测试 `test_mutable_laser_segment_reuse`(2 例绿):base pid 复用 gen-token+reopen(shadowing 通过)、stale token no-op+future token corruption。焦点 laser/collection/adapter 全绿。JC-26。**边界 7(rank-only rerank 新 incarnation)**由 effective_label 显式-binding-优先自然满足(reused pid 返回新 label⇒Collection 按 label rerank 用新向量),step 8 端到端验。
- **W2 复用核 step 6 落地并 commit(本波第五程,arming)**:扩 `kQgSupportedRequiredFeatures |= pid_generation|canonical_prebind|mutable_label`(3-bit,qg.hpp;与 step-3 replay lane + step-5 writer 同库⇒"不先扩 mask 后补 replay")+ activation summary(checkpoint v3 块写 `max_pid_generation`/`nonzero_pid_generation_count`,从 snapshot bindings 算)+ 六 fail-closed load 条件(adopt_label_state:activation gen ∈(0,sb.generation]、slot 实测 max/nz 与摘要相等;既有:gen!=0 未激活、binding pid≥HWM、FREE gen==UINT32_MAX、feature 三位不成套)。**特性武装**:`enable_pid_reuse=true` 首 bundle 触发 v3 pid activation checkpoint 后走 canonical。功能测试 `test_qg_updater_reuse`(5 例全绿):activation+all-append+reopen、delete-all→all-reuse N>1+双 reopen、mixed reuse+append+reopen、同 PID gen 0→1→2+reopen+摘要校验、no-reuse 恒 v2。焦点 laser/collection(mutable_laser_segment/active_laser/qg_seal/target/unit)全绿⇒mask 扩容不伤 v2/维护-only-v3 消费者。JC-25;JC-16 七洞:loader max/nz 核对✅、activation gen 界✅、promotion insert_or_assign✅(step3)、PidToken 默认 kPidMax✅(step2)、content-revision slot-dirty✅(step2);replay_label_bind 拒非零 gen=**正确非洞**(post-activation 帧走 canonical lane,legacy 仅 pre-activation);absorbed 比 gen=JC-22 validate-only。
- **W2 复用核 step 5 落地并 commit(本波第五程,dormant canonical writer)**:codex §B.4 canonical prebind writer——`commit_physical_bundle_canonical`(reserve→预建 snapshot(insert_or_assign 换绑)→全 kind=7 →reused 页 FREE preimage kind=1 →私有 overlay 建行/反向边/bundle spine→finalize 每脏页 final kind=1→B-2C-02 自检→kind=8 force→install 入 shared write cache(seqlock)→固定发布序:snapshot→routing/entry→clear reused hidden→committed HWM→live_count→txid/applied),`BundleInsertContext` 私有页 overlay(materialize/log/spill/install 四辅助)+ `writer_hidden`(4 写者路径 is_hidden 替换)+ private-entry 播种 + bundle spine(`last_built→current` force,Backlink::kNone 可达性)+ append_node canonical 分支(reused 行最终 trailer live)+ `checkpoint`/`checkpoint_locked` 拆分(JC-23)+ `ensure_pid_reuse_activated`(写序备好,mask 未扩⇒step 6 arming)。**全 dormant**:`enable_pid_reuse` 默认 false 且无激活⇒`pid_generation_activated_||enable_pid_reuse_` 恒 false⇒canonical 分支不进,2A/2B/golden/crash 逐字节不变,4/4 focused 绿。**step 6 arming(扩 mask+JC-16)后方功能可测**;JC-24。
- **W2 复用核 step 2-4 落地并 commit(本波第四程,dormant 安全推进)**:codex 8 步序的 step 2/3/4——step 2(**caa3287**)不激活类型/API(`PidToken.pid=kPidMax` + `operator==`、`PhysicalBundleResult`、label content revision slot-dirty 追踪、free saturation 双守卫 reclaim skip + recovery poison),step 3(**84bf2f0**)canonical replay reader(BundleStage/lane、FREE preimage first/latest ref 识别、mixed HWM 代数、gen+1 链、B-2C-02 final-live、EOF torn 截断、absorbed validate-only=JC-22),step 4(**f300f6d**)reserve_bundle_pids + reuse checkpoint 准入原子组。**三步全 dormant**:supported mask 未扩、`enable_pid_reuse` 未接线、`pid_reuse_activation_gen_` 恒 0⇒canonical lane 不进/reserve 不被调/准入恒满足⇒2A/2B/golden/crash 逐字节不变,40/40 全绿。**step 5-8 未落**(见"六、遗留");停点=step 4→step 5 组间绿边界(writer 编译耦合 step 6 arming⇒一旦落 writer 即需 activation + R0-R11 全矩阵 + 强制 codex 中审,超本程预算;JC-15 原子同落纪律)。

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
| B-2C-01 writer-private seed | W2 | **完成** | BundleInsertContext 私有页 overlay + writer_hidden + private-entry + 双向 bundle spine(step5/8);delete-all→all-reuse/mixed 硬测 + kFullPrune 回归绿 |
| B-2C-02 canonical prebind 恢复完整性 | W2 | **完成** | canonical replay lane(step3)+ writer 产物一一对应(step5)+ 自检 scratch 读(step8 MAJOR-3);R6 恢复正向验证 + 三 poison 由 replay 校验覆盖 |
| B-2C-03 Collection 维护入口+门禁链 | W1/W3 | **句柄锁部分** | 段级 `MutableLaserSegment::mutex_` 已串行化所有 mutating 入口(consolidate 需经其接入,W3);Collection hook + 全门禁链是 W3。 |
| B-2C-04 frame 三 API + 流式构造 | W1 | **完成** | `frame.hpp`:`FrameLocation`、`append`→location、`visit_frames`(单帧内存)、`read_frame`(重校验)、`truncate_to`(语义截断);wire 零变化(make_frame golden 未动)。溢写重载用 `read_frame`。全流式 replay 见 JC-9。 |
| B-2C-05 MaintenanceOverlay 并发协议 | W1 | **串行化解决(并行 deferred)** | 单线程维护 lane:无并发 WAL append、无容器竞态;install 走 page seqlock(odd→write_at/arena→even)保证并发 search 无半页。并行 page workers + barrier-外 spill 见 JC-8。 |
| B-2C-06 superblock v3 fail-closed + 双常量 | W0 | **完成** | 见 W0 表 + 判断调用 JC-1/2/3。 |
| B-2C-07 协议洞→硬验收 | W2/W3 | **W2 部分完成** | all-reuse reservation×checkpoint 由准入+checkpoint_mutex_ 结构保证;同页多 spill latest-ref、delete-all non-dense seed、rank-only rerank(effective_label 新 label)已覆盖;Collection consolidate×seal/rotate active identity 交错=W3 |

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
- **JC-16**(MAJOR-10 缓落定位):W2a generation 门控的 activation-ready 待补洞(loader 核对 `max_pid_generation/nonzero_pid_generation_count`、absorbed replay 比 generation、promotion `insert_or_assign` 换 incarnation、`replay_label_bind` 放开非零 generation、checkpoint 以 content-revision/dirty 判 slot、`PidToken` 默认 `kPidMax`)——codex A.10 明令"必须在扩 `kQgSupportedRequiredFeatures` 的同一原子落地组内完成",故不在 Step 1 落地,随 Step 6 激活组。清单已在此 REPORT §七 + codex 报告 §A.10 固化。
- **JC-17**(MAJOR-9 分割):slot loader `pid < num_points` 已在 QG loader 落地(fail-closed 拒越界 binding);codex A.9 的"未激活时 base PID 不得有 binding / `pid<base_count`⇒`gen>0`"部分依赖 segment 层 `base_count`(reuse 模型),随 Step 7 adapter/effective_label 组落地。
- **JC-18**(BLOCKER-1 范围):落地"当前页 pin"(`maint_spill_over_cap(pin_pi)` 绝不驱逐正在 RMW 的页),直接消除 codex 反例(cap=1 下溢写选中刚开始的活跃页⇒下一行重载重记的无界放大)。完整"物理页分批"帧数上界仍是并行 lane 的 deferred 设计(JC-8);残余(小 cap 下邻居跨行重触发的再溢写)由 preflight 余量 + pin 有界,近满盘极端下仍可能超 preflight——与 JC-8 串行-lane-最小可行同范围。
- **JC-19**(BLOCKER-3 落地,取代 JC-9/JC-14 的 deferred):流式恢复经 `WalFile` 新增 opt-in `stream_recovery` ctor 旗标(default false⇒既有 caller 与 `wal_frame_test` 红线逐字节不变)+ `scan_structure_streaming`(单帧内存、不留 payload)+ QG replay/fresh-probe 全走 `visit_frames`。行为等价(同帧序、同 epoch 状态机),恢复内存 O(max frame + touched pages) 非 O(WAL 字节)。`truncate_to` 仍是流式恢复语义截断,非第四 API 滥用。
- **JC-20**(BLOCKER-4):`maint_install_all` 的 page seqlock 在 `write_at` 抛出时经 try/catch 补齐 even(关闭 seqlock 对),并发 reader 不再永久自旋;句柄同抛即 poison,撕裂页在 poison 下无 post-commit 内容保证,仅保 reader liveness。
- **JC-21**(MAJOR-8):no-steal 审计统一为 `assert_no_maintenance_steal(what)`,覆盖 `write_at`(pwrite+arena)/bloom-scan `ftruncate`/`write_superblock`(直接 pwrite)三口;replaying_ 与 pre-BEGIN 激活 checkpoint 下为 no-op。codex 建议的"observer 正向计数证明 END 前维护写=0"被 poison 守卫 + 现已精确的 S_old 指纹(BLOCKER-2,任何 build 期 index 变更泄漏进 S_old 即被指纹比对逮住)双重涵盖,故不新增计数器(冗余)。
- **JC-22**(Step 3 canonical replay reader,absorbed-canonical validate-only):canonical replay lane 的 absorbed 前缀(`txid <= base_committed_txid_`)采**保守 validate-only**——跑结构集合验证(row_op {0..count-1} 唯一、pid 全局唯一、binding_count 匹配)+ B-2C-02(每绑定 PID 有 final live trailer),但**跳过** HWM 代数与 gen+1 链校验(此时 running watermark 已是 POST-absorbed base,pre-tx 代数无法在不重建 retained-kind=6 模型下复原)、**不重放**页、不提升绑定、不进 watermark。达成 codex §B.5 目标(不再用"binding 与最终 slot 相等";absorbed 页只验证不重放),但未实现 codex 的完整"从 retained kind=6 建模、顺序执行 absorbed bundle、最终 flip 比对"belt-and-suspenders。理由:该路径仅 R11(checkpoint 吸收后 WAL-reset 前崩溃)可达,且**须 step 5 writer + step 6 activation 才能产出可测的 canonical WAL**;dormant reader 阶段无法端到端验证。裁决:保守版随 step 3 落地(安全——absorbed bundle 已在通过自身 checksum 的 base 内),完整 retained-kind=6 模型作 step 6 携带项(R11 可测时补齐并对抗验)。非翻案(保留 codex 结论:absorbed validate-only、不重放),仅 dormant 路径的实现缩减 + 显式挂账。
- **JC-27**(Step 8 codex 中审修复,5 BLOCKER+3 MAJOR 落地):逐条=上"当前状态"step 8 条①。关键判断调用:①**BLOCKER-2 entry 收敛**用 `repair_routing_roots(kPidMax)`(clean writer 在 clear reused hidden 之后调,故 reused entry 读为 live)而非硬设 entry=private_entry——replay 侧 `rebuild_state_after_replay` 调同一函数⇒同一确定性 entry;双向 spine 保证任一 live bundle row 可达全 bundle。**发布序微调**:reused hidden clear 提到 repair 之前(repair 需看 live 态定位 entry;delete-all 下若 clear 在 repair 后,repair 见全 hidden⇒无 live entry),与 publish_common"clear reused hidden 在 committed store 前"一致,不引入 live-before-page 窗口(页已 install)。②**R5 语义**:writer 全程 buffered(userspace)到单一 force_wal,故 SIGKILL 与 power-loss 边界重合于 force(非 append)——R5(force 前)双路径均 S_old,无"完整 unforced END 保留"歧义(consolidate 同款)。③**medoid 陈旧**(reused medoid 保留旧 vector)codex BLOCKER-2 次要点:medoids 非持久(每 open 从 base sidecar 重载)⇒clean 与 replay 一致(都重载+repair 删 hidden medoid),非 divergence;reused-live medoid 陈旧 vector 仅劣化 routing 起点(beam 仍经图到达真近邻),非丢行,列质量 follow-up 不阻断。④**MAJOR-2 absorbed**:validate-only 不 apply/不 promote⇒畸形 absorbed bundle 不改状态(base 已权威),仅未拒畸形 WAL 后缀(robustness 非 corruption);完整 retained-kind=6 前态模型 R11 可端到端测时补(JC-22 挂账延续)。
- **JC-26**(Step 7 token 化):判断调用:①**保留 pair 版 `commit_physical_bundle` 为 thin wrapper**(而非改签名)——现有 30+ 测试 caller(test_qg_updater_wal/test_mutable_laser_segment)读 pair,legacy body 抽出 `commit_physical_bundle_legacy_2a`(共享 precondition 由 `commit_physical_bundle_tokens` 前置),tokens 版为主入口。②adapter tombstone 目标 **提交前捕获 token**:物理 label 唯一(每版本一 label)⇒提交后捕获也不误(L_old 只映 Q 或 nullopt,不被 bundle 换绑),但按 §B.7 提交前捕获=belt-and-suspenders(reuse+同 txid 场景更稳);replay 幂等由 token_for_label 反映当前 durable 态(L_old 已 tombstone⇒nullopt⇒skip)保证。③**runtime previous/erase miss 保持宽松幂等**(不改为 §B.7 corruption)——2B 现契约,收紧风险 2B 回归红线,ABA 安全已由 token 覆盖;§B.7 "op 未 applied 且无法解析 corruption" 挂账 W3。④segment 加 `consolidate` passthrough(reuse 可测,持 mutex_;W3 Collection 门禁链在其上,非抢占)。⑤边界 7 rank-only rerank 靠 effective_label 显式-binding-优先(reused pid⇒新 label⇒Collection 按 label rerank 用新 retained 向量),非新代码。
- **JC-25**(Step 6 arming):`kQgSupportedRequiredFeatures` 三位与 replay lane(step3)+ canonical writer(step5)同库原子武装(codex 风险 1)。判断调用:①**activation summary 每 checkpoint 重算**(非仅激活时):从当前 snapshot bindings 算 max/nz 写入 w2c——clean checkpoint 时 snap==persisted slot,dirty 时 snap==新 slot,恒一致;load 侧从实际 slot bindings 重算比对,mismatch⇒poison(第六 fail-closed)。②**maintenance activation gen 界检查同时收紧**(maintenance_activated_⇒gen∈(0,sb.generation]):维护-only v3 base 恒有非零 gen,不误伤。③`replay_label_bind` 拒非零 gen **保留为正确**(非 JC-16 洞):dispatch 按 `is_canonical_generation(seg_gen)` 分流,post-activation 帧走 canonical lane,legacy 2A 路径只见 pre-activation gen=0 帧。④test 加两只读诊断 accessor(`superblock_format_version`/`pid_generation_activated`)。
- **JC-24**(Step 5 canonical writer):落地 codex §B.4 完整写序。判断调用:①**install 目标=shared write cache**(非 write_at 直写 index),与 2A 追加同款(pages 驻留 cache,物理写延后到 eviction/checkpoint)——codex §B.4 "install final pages into shared write cache" 原文;bundle 全程不碰 index 文件(仅 WAL+cache),故 `assert_no_maintenance_steal` 对 bundle 恒 no-op(bundle 非维护 epoch,不设 `maint_in_build_phase_`)。②**overlay 读路由**:writer 的 `read_node_page`(query_read=false)仅对**已触碰**页(resident/spilled)走 overlay,未触碰页落 committed 读(避免每遍历页 materialize 膨胀);并发 public search(query_read=true,他线程)恒不进此支⇒严格 writer-private。③**bundle spine**:每行(除首行)`patch_reverse_edge_impl(last_built,current,force=true)` 强制前向边,幂等(边已存在返回 kApplied),装不上即 kind=8 前 poison——满足 §B.3 "Backlink::kNone 删光可达性" 二选一之更稳者。④**reserve 在 try 外**:precondition(allocated==committed/free-chain rebuilt/no staged)throw 为 caller error 非 poison;reserve 内 corrupt free chain 才 poison;kBuilding 后任何失败 poison 保状态。⑤**append 可见性floor**:reused 行 override=old_hwm(经 revealed 集揭示),append 行 override=其 pid(=old_hwm+appends_built,揭示同 bundle 先建 append)——混合 bundle append 连续段在 result.rows 尾部(reserve 保证),故 floor 与 pid 对齐。⑥**overlay 读 committed 经 `read_committed_page`(cache-aware)**:bundle 前不 flush cache(不同于 consolidate 的 baseline 归零),脏 cache 页被正确读入 overlay,install 时覆写回;未触碰脏页留存 cache。dormant 未功能验证,step 6 arming 后端到端验。
- **JC-23**(Step 4 checkpoint 准入):codex §B.6 的"将 checkpoint() 拆成公开加锁层 + checkpoint_locked(),bundle/consolidate 共用 checkpoint_mutex_"**推迟到 step 5**(canonical writer 才需在 checkpoint_mutex_ 下调 checkpoint/activation)。step 4 仅在现有 checkpoint() 内加三条 reuse 准入(bundle_state==idle、reservation_count==0、free_chain_rebuild_complete),dormant 下恒满足⇒零回归。理由:public/locked 拆分是 writer 集成关注点,无 writer 时拆分是空动作;step 5 落 writer 时一并拆(那里才有跨 checkpoint_mutex_ 的真实 activation 调用)。

## 八、Step 1 状态映射(codex 复用核报告 A 部分:最小正确落地顺序步 1)

| 项 | 类型 | 状态 | commit | 落地 |
|---|---|---|---|---|
| BLOCKER-1 statvfs/spill | 代码 | **落地**(pin) | 9653693 | `maint_spill_over_cap(pin_pi)` 钉活跃页;全上界=JC-18 deferred |
| BLOCKER-2 C 矩阵 oracle | 测试 | **落地** | 96ce92a | S_old/S_new 全指纹(页内容+PID 集+free chain+routing+epoch)+ SIGKILL 断言 + 二次 replay 重开首恢复输出 |
| BLOCKER-3 流式 replay | 代码 | **落地** | c3e3306 | `stream_recovery` ctor + `scan_structure_streaming` + `visit_frames` 驱动 replay(JC-19) |
| BLOCKER-4 seqlock scope guard | 代码 | **落地** | 6e7f008 | install `write_at` 抛出时补 even(JC-20) |
| BLOCKER-5 replay generation/epoch | 代码 | **落地** | 6e7f008 | BEGIN 绑 cursor generation,kind=1/END 同 generation 否则 poison |
| MAJOR-6 C7 真 torn-END | 测试 | **落地** | 96ce92a | 显式物化 truncated/bad-CRC END⇒S_old,完整 END⇒S_new roll-forward |
| MAJOR-7 oracle 等价补全 | 测试 | **落地** | 96ce92a | 补 trailer `valid_degree` + canonical free chain(head=min/升序/kPidMax/len==free_count/集合相等) |
| MAJOR-8 steal guard 统一 | 代码 | **落地** | 6e7f008 | `assert_no_maintenance_steal` 覆盖 write_at/ftruncate/write_superblock(JC-21) |
| MAJOR-9 slot loader 验证 | 代码 | **部分落地** | 6415aba | QG loader `pid<num_points`;base-region shadowing 随 Step 7(JC-17) |
| MAJOR-10 W2a activation-ready 洞 | 代码 | **随激活组(Step 6)** | — | codex A.10 明令原子同落;清单=JC-16 + §七 |

**Step 2/3/4 落地(本波第四程,commit caa3287/84bf2f0/f300f6d,全 dormant、40/40 绿)**:step 2 不激活类型/API + revision + free saturation;step 3 canonical replay reader(§B.5,含 absorbed validate-only=JC-22);step 4 reserve_bundle_pids + reuse checkpoint 准入(§B.2/§B.6,public/locked 拆分推迟=JC-23)。**Step 5-8 未落**:step 5 canonical writer(§B.4,≈250 行最深水:BundleInsertContext/private overlay/kind=8 后固定发布序/final-live-trailer)与 **step 6 activation**(扩 supported mask + activation checkpoint/summary + JC-16 七洞——**arming 动作**)**编译耦合**(writer 的 canonical 分支调 `ensure_pid_reuse_activated`);一旦落即 arm 协议,须再落 step 7 adapter token 化(JC-17)+ step 8 R0-R11 全矩阵 + 额外硬族 + **强制 codex 对抗中审**方为 sound。剩余预算不足以 sound 落 writer+arming+矩阵+审,故停在 step 4→step 5 组间绿边界并诚报(JC-15 runtime+replay 原子同落、codex §5 坑 1、前三程 W0/W1/W2a+Step1 停点先例)。step 5-8 runway = §七 + codex 报告 §B.4/§B.6/§B.7 + 本程 dormant 地基(reader/reserve/admission 已就位:writer 直接消费 `reserve_bundle_pids` 返回的 `PhysicalBundleResult`,arming 后 reader 即接管 canonical replay)。

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
- **W2 完成(step 1-8 全落地)**:PID generation + bundle-only reuse 全链——writer-private seed B-2C-01 ✅、canonical prebind 恢复完整性 B-2C-02 ✅、checkpoint 准入 all-reuse 窗口 ✅、effective_label override ✅、2B adapter token 化 ✅、v3 PID activation ✅、R0-R11 SIGKILL 矩阵 ✅、B-2C-07 W2 部分 ✅。**codex 对抗中审已做**(5 BLOCKER+4 MAJOR,8 修 1 挂账 JC-22/27)。**W2 残余硬族(code-covered/结构保证,未单测)**:并发 public query 隔离(codex 结构验证 query_read 短路)、all-reuse×并发 checkpoint(checkpoint_mutex_ 结构保证+admission)、canonical poison 2/3(页几何依赖;poison 1=bind-无-rowpatch 已测)、三 PID 来源(base 已测,legacy/explicit 部分)、saturated-FREE(step2 code+recovery poison,未单测)——列 W3/后续补测。
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
