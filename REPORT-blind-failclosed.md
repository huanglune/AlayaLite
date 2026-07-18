# Wave 3 blind fail-closed 修复报告

## 结论

盲审的 F1（BLOCKER）、F2（MAJOR）和 F3（MAJOR）均确认生产可达，已按
F1 → F3 → F2 的硬排序修复并分别提交。修复没有改变 WAL wire format，也没有改变
未激活、`enable_pid_reuse=false` 的 legacy 落盘格式。Release 配置下全量 94/94
CTest、reuse 59/59、reuse 崩溃格 31/31、B04 8/8、kind 1-8 golden 2/2
全部通过，所有测试进程退出码均为 0。

基线：`feat/wave3-integration@9fdc6e0`

修复提交：

1. `97180fd fix(laser): fail closed on baseline flush errors`
2. `4db820f fix(laser): enforce maintenance cap during reclaim`
3. `0b6bde3 fix(laser): bound maintenance WAL spill growth`

## F1：baseline flush 失败时 fail closed

结论：发现成立，采用 poison + recovery-required 语义。

修法：

- 用 `parallel_for_catch()` 包住本文件全部 OpenMP worker 区域。worker 捕获并发布首个
  `exception_ptr`，隐式 barrier 结束后才由调用线程重抛；异常不再越过 OpenMP structured
  block。
- 用 `write_page_versioned()` 统一物理页写的 seqlock 生命周期。写失败时先发布 poison，
  再用 release store 闭合 odd → even，最后重抛。`write_node_page()`、`flush_dirty()` 和
  maintenance install 都走该路径。
- 将 statvfs admission 移到 baseline writeback 之前，并把当前共享 cache 中的实际 dirty
  页数计入可能分配块的 index CoW 预算。
- baseline flush 发生异常时，consolidate 在 BEGIN 前停止；由于 pwrite 可能已经部分完成，
  handle 保持 poisoned。既有 segment/adapter/Collection recovery-required 传播链因此可见，
  不能把该 handle 当作普通可重试失败继续使用。
- 增加仅用于故障注入的 `before_index_write_hook`，没有增加 WAL kind 或 payload 字段。

证据：

- `QgMaintenanceConcurrency.BaselineFlushFailurePoisonsClosesEvenAndPrecedesBegin`：注入
  baseline pwrite ENOSPC；确认 admission 已发生、无新增 BEGIN、handle poisoned、版本为
  even，且绕过 poison gate 的异步 raw page read 在 1 秒内完成。
- `QgMaintenanceConcurrency.DirectPageWriteFailurePoisonsAndClosesEven`：覆盖无 write cache 的
  `write_node_page()` 失败路径，确认 poison 且无永久 odd。
- `MutableLaserAdapter.BaselineFlushFailureIsRecoveryRequired`：确认 segment 与 adapter 都报告
  recovery-required，adapter latch 生效，后续 checkpoint 被拒绝。
- `QgMaintenanceConcurrency.StatvfsFailureBeforeBeginDoesNotLatch` 继续确认 admission 本身失败
  是无写入、无 BEGIN、可重试的干净失败。

## F3：reclaim maintenance overlay 资源上界

结论：发现成立，reclaim 已改为按物理页处理并在每页后执行 cap enforcement。

修法：

- existing free-chain walk 在读出 next 后立即释放不再需要的 clean dependency page；跨页时
  不让整条链常驻 overlay。
- eligibility 扫描按物理页聚合且只读，读完一页立即释放。
- 得到最终 free set 后按 PID/物理页排序；同一页内一次完成新 FREE trailer 和 canonical
  next 指针改写，随后执行 `maint_spill_over_cap()`。
- 暴露 `maintenance_peak_overlay_pages` 测试统计。统计口径是在 overlay page 实际
  materialize 后记录 resident `maint_pages_` 峰值；允许当前 pin/RMW 转换产生一个常数页，
  因而验收界为 `cache_cap_pages + 1`。

证据：

- `QgMaintenanceConcurrency.ReclaimOverlayHonorsPageCap`：256 行全 tombstone、
  `cache_cap_pages=1`、reclaim 全开，确认全部 256 PID 被回收且峰值不超过 2 页；第二个
  epoch 还覆盖已有 256-PID free-chain 的 dependency walk，峰值仍不超过 2 页。
- reuse 及 reuse crash 套件继续覆盖 canonical free-chain、重开和崩溃恢复，分别 59/59、
  31/31 通过。

## F2：reload → respill 放大与 admission 上界

结论：发现成立；选择“消除纯放大并给出可证明 BEGIN-time 上界”，无需增量 statvfs 复查，
无需修改 wire format。

修法：

- 将 overlay 状态拆成三部分：`maint_dirty_` 表示 epoch 内曾修改、最终必须 install 的页；
  `maint_resident_dirty_` 表示 resident bytes 比最新 WAL after-image 更新；
  `maint_spilled_` 始终保留该页最新 kind=1 frame location。
- 从最新 spill 重载仅供依赖读取时不再删除 `maint_spilled_`。该 resident copy 没有再修改，
  evict 时可直接丢弃，不追加重复 after-image；再次修改后才进入 resident-dirty 并合法追加
  新的最新 frame。finalize 也只记录 resident-dirty 页。
- replay 原有 same-page latest-frame-wins 行为保持不变；install 按 `maint_dirty_` 唯一页集合
  取 resident 或最新 spill image。
- row repair 以物理页顺序处理，并 pin 当前 target；历史脏页作为依赖重载后是 clean-to-latest-
  spill，可免费释放。因此每个物理页至多产生一个 repair frame。F3 的 combined reclaim
  pass 又保证每页至多一个 reclaim frame。
- admission 使用可证明上界 `repair_pages + reclaim_pages`：repair 最多为全部 committed
  physical pages，启用 reclaim 时再加同样一份。WAL byte 上界由相同 kind=1 encoder 开销和
  两个 marker 计算；baseline dirty page 与 post-END install 的 index CoW 另行预算，并保留
  16 MiB/5% 安全余量和饱和算术。
- 暴露实际 page frame 数/字节及本次 preflight frame/WAL-byte 上界统计，供回归直接比较。

证据：

- `QgMaintenanceConcurrency.DependencyReloadFramesStayWithinPreflightBound`：使用盲审探针同形
  的 256 行/128 页、R64/d64、`cache_cap_pages=1`、偶数 PID tombstone、reclaim；确认
  marker 分隔的 maintenance kind=1 扫描数等于内部 frame 计数，preflight frame 上界为
  `128 * 2 = 256`，maintenance frame 与事务字节都不超过对应 admission 上界。全文件窗口
  另按 baseline tombstone/row-patch 与 maintenance 事务组成的合成上界检查。
- `QgMaintenanceConcurrency.EnospcAfterBeginPoisonsAndReopenRollsBack` 与 crash grids 继续确认
  BEGIN 后异常仍 poison，重开按事务边界恢复。

## 判断调用（JC）

- **JC-1（F1 错误语义）**：选择 poison + recovery-required，而不是普通可重试。理由是
  `pwrite()` 报错不能证明零字节落盘，继续使用当前映射/cache 可能服务部分页。
- **JC-2（poison/even 发布顺序）**：失败路径必须先 poison、后发布 even。被解除自旋的
  reader 在 exit gate 观察 recovery-required，不会把部分状态当成正常页。
- **JC-3（OpenMP）**：全部同型 worker 区统一首错捕获；只有 barrier 后的串行调用线程重抛。
- **JC-4（pre-BEGIN admission）**：statvfs rejection 保持干净可重试；admission 必须早于
  baseline pwrite，并预算 baseline dirty CoW。
- **JC-5（F3 cap 口径）**：统计真实 resident overlay map；当前 pin/RMW 允许 cap + 1 的
  短暂常数余量，不允许随 free page 数增长。
- **JC-6（reclaim 布局）**：先只读确定 eligibility，再把 FREE trailer 与 next pointer 合并到
  每物理页的一次写 pass。这同时收紧内存和 WAL frame 上界。
- **JC-7（F2 overlay 状态）**：ever-dirty、resident-newer、latest-spill 必须分离。仅重载读取
  不构成新的 durable state，不能再次记 frame；再修改则必须记最新 frame。
- **JC-8（F2 admission 策略）**：剩余行为可证明为每页每阶段最多一个 frame，故采用精确
  BEGIN-time 上界；没有引入每次 append 前 statvfs 复查，也没有扩大事务/wire 设计。
- **JC-9（兼容性）**：不修改 `segment_op_wal.hpp`、WAL kind、payload encoder/decoder 或
  golden bytes。legacy 双 false 路径继续由 reuse 等价/字节稳定测试锁定。

## 验证记录

构建配置：Release、`BUILD_TESTING=ON`、`BUILD_PYTHON=ON`。`ctest -N` 注册 94 项；全量
build 为 203/203。全量 CTest 使用 `-j 4 --output-on-failure`，直接保存 `$?`，未经过管道。

| 验证 | 结果 | 退出码 |
|---|---:|---:|
| 全量 CTest | 94/94 | 0 |
| `test_qg_updater_reuse --gtest_brief=1` | 59/59 | 0 |
| `test_segment_op_wal_reuse_crash --gtest_brief=1` | 31/31 | 0 |
| `test_qg_updater_wal --gtest_filter='Divergence/*'` | 8/8 | 0 |
| kind 1-8 两项 golden filter | 2/2 | 0 |
| `git diff --check 9fdc6e0..HEAD` | clean | 0 |

legacy 兼容性包含 `NoReuseFlagKeepsLegacyV2Base`、
`LegacyV2PrefixUnderReuseFlagRecoversEquivalentToNoReuse`、三项 reuse on/off 等价测试和
`LegacyFullPruneBundleStaysV2AndByteStable`，均包含在 59/59 结果中。

Python 验证环境说明：配置指定的解释器来自主 checkout 的 `.venv`，其
`_alayalite_editable` meta finder 会抢先加载主 checkout 扩展；fixture 现有过滤条件只匹配
`_editable_` 前缀。因 `tests/disk/**` 不在本任务领地，没有改该脚本。验证时仅在 ignored
`build/Blind/validation_python/sitecustomize.py` 移除该 finder，并通过 `PYTHONPATH` 传给
build/CTest；fixture 随后明确打印并校验加载的是本工作树
`build/Blind/python/_alayalitepy.cpython-313-x86_64-linux-gnu.so`。该辅助文件未提交。

## CI coverage lane F2 回归追修

### 裁决：测试窗口可见性（分析 1），不是生产双记（分析 2）

Codecov 的 `542,976` 字节差值可以精确分解，而不只是近似为 130 个 page frame：

`542,976 = 128 * [(4096 + 79) row_patch frame + 67 tombstone frame]`

也就是 128 个偶数 PID tombstone 在 maintenance BEGIN 之前各自产生的一条 kind=1
whole-page after-image 和一条 kind=2 tombstone。证据链如下：

- `tombstone()` 先以 `Sync::buffered` 追加 kind=2，再由普通（非 maintenance）
  `modify_node_page()` 追加 kind=1，并把对应 cache page 标为 dirty。
- `WalFile::Sync::buffered` 只写入 `ofstream` 的进程内缓冲；该缓冲配置为 1 MiB。
  外部 `visit_frames()` 和 `filesystem::file_size()` 能看到多少已追加记录取决于
  libstdc++ 何时把 stream buffer 推给内核。gcc-11 本地与 gcc-13 coverage lane 因此得到
  不同的开窗前可见前缀。
- consolidate 的 baseline `flush_dirty()` 首先调用 `force_wal()`，所以此前已逻辑追加但尚未
  对另一个 fd 可见的 128 对记录会在原测试窗口内一次显现。`flush_dirty()` 自身没有任何
  WAL append；随后只执行 index pwrite。
- BEGIN fsync 后才设置 `maintenance_active_=true`。row repair/reclaim 的每个
  `modify_node_page()` 都在函数首部进入 private overlay 并提前返回，不可能落到 legacy
  `log_page_after_image()` 分支。
- `maintenance_page_frames` 只在 `maint_log_page()` 成功 append 后增加；按 BEGIN/END marker
  扫描出的 maintenance kind=1 数与该统计增量相等，排除了同页 legacy + maintenance 双记。

因此没有生产代码或 WAL wire 变更；`maintenance_last_preflight_wal_bytes` 继续表示
marker-delimited maintenance 事务的专属上界。把含有更早 baseline 操作的全文件可见增量
直接与它比较，是原回归的口径错误。

### 确定性回归

`DependencyReloadFramesStayWithinPreflightBound` 现在在 tombstone 之前开启测量窗，明确把
baseline 与 maintenance 两部分都纳入，而不再依赖 stream buffer 是否提前可见：

- 每物理页只 tombstone 一个 PID；consolidate 前断言 resident pool 为 128 页，完成后通过
  `flush_unique_pages` 增量断言入口确有 128 个 dirty 页。
- 全窗口精确断言新增 128 条 kind=2，且 kind=1 总数等于 128 条 baseline row-patch 加
  maintenance 专属统计。
- 扫描最后一个完整 BEGIN/END epoch，断言其中 kind=1 数等于 maintenance 专属统计，事务
  字节不超过 `maintenance_last_preflight_wal_bytes`。
- 用冻结 codec 实际计算 frame 大小，断言全文件增量精确等于
  `128 * (row_patch_frame + tombstone_frame) + maintenance_window_bytes`，并检查相同组成的
  合成上界。

### 追修验证

| 配置/套件 | 结果 | 退出码 |
|---|---:|---:|
| Release、Python ON、全量 CTest | 94/94 | 0 |
| Debug、gcc-11、coverage ON、Python OFF、全量 CTest | 94/94 | 0 |
| coverage 配置下确定性 F2 单测 | 1/1 | 0 |
| reuse | 59/59 | 0 |
| reuse crash | 31/31 | 0 |
| B04 | 8/8 | 0 |
| kind 1-8 golden | 2/2 | 0 |

本机 Debug+coverage 下既有 `qg_builder_oom_regression` 需要 192-249 秒，超过源码设置的
120 秒 CTest timeout。首次串行验证因此是 93/94 + 纯超时；只在 ignored build tree 的
generated `CTestTestfile.cmake` 将该目标 timeout 调为 300 秒后，全量复跑 94/94，最终
`COVERAGE_CTEST_RETRY_EXIT_CODE=0`。没有修改其源码、CMakeLists 或 tracked 文件。

## 领地与兼容性审计

相对 `9fdc6e0` 的 tracked 修改只有：

- `include/index/graph/laser/qg/qg_updater.hpp`
- `tests/collection/mutable_laser_adapter_test.cpp`
- `tests/laser/qg/test_qg_maintenance_concurrency.cpp`
- 本报告

未触碰 `segment_op_wal.hpp`、`docs/design/**`、`include/simd/**`、`include/platform/**`、
`include/space/**`、`include/index/collection/**` 或 `.github/**`；没有 push。
