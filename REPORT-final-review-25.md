# #25 终审记账项当场修报告

## 结论

终审清单中的两项均成立并已修复：A 将普通 write-cache RMW 的任意回调纳入与
maintenance 物理写相同的 updater 级 fail-closed seqlock 守卫；B 新增 Linux-only
mutable LASER CMake capability，只在能力存在时注册 updater/op-WAL/reuse/active
Collection 测试。WAL wire、生产宏和 legacy 双 false 落盘字节均未改动。

基线为 `main@c27dafd`，分支为 `fix/final-review-25`。任务 A 的独立提交是
`286e9e8 fix(laser): close cached RMW seqlock on callback throw`；任务 B 与本报告由
第二个独立提交承载。未 push。

## 任务 A：write-cache RMW 回调异常闭合 seqlock

### 裁决

问题成立，poison 粒度采用与 maintenance 一致的 `QGUpdater` 级 `poison_latch`。

普通缓存 RMW 的回调拿到共享 `cached->bytes` 后可能已经改写部分页才抛异常，调用方
无法证明页面仍是旧像，也没有可靠、零额外成本的通用回滚。继续允许当前 handle 读写
会把半修改缓存作为正常状态服务。因此异常路径必须按以下顺序 fail closed：

1. updater 级 poison/recovery-required；
2. release 闭合该页 odd -> even；
3. 原样重抛当前异常。

poison 必须先于 even：被 seqlock 放行的 reader 会在读出口观察 recovery-required，不能
把半修改页当成成功写入。非异常路径仍是原来的两次 version bump、同一把
`bytes_mutex` 和同一个回调；只增加零成本异常表，不增加原子、锁或 failpoint 分支。

### 证据

- `modify_node_page()` 的普通 `write_cache=true` 分支改用既有
  `write_page_versioned()` 执行任意 `fn(cached->bytes.data())`。该守卫已实现
  catch-all -> `poison_latch()` -> even -> `throw;`，异常类型不被替换。
- `write_page_versioned()` 的契约注释扩展为同时覆盖 cached RMW callback、pwrite 和
  post-write failpoint。
- 故障注入没有进入生产热路径：仅在 `ALAYA_LASER_TESTING` 下暴露
  `debug_modify_node_page()`，maintenance concurrency 测试目标单独定义该宏。
- `CachedModifyCallbackThrowPoisonsClosesEvenAndReopens` 先修改共享缓存页的一个字节，再
  抛 `std::bad_alloc`，证明异常确实发生在 odd 窗口里的任意回调，而不是更早的分配边界。
- 同类站点全文件复核：
  - WAL scratch after-image 安装站点只在 odd/even 间执行定长 `std::memcpy`；
  - canonical bundle post-commit cache 安装站点也只执行定长 `std::memcpy`；
  - 两处均保留裸配对并新增“不得加入可抛工作”的 no-throw 不变量注释；
  - 其余 page-version 物理写均已通过 `write_page_versioned()`。对
    `page_versions_[...].fetch_add` 的全文件扫描未发现第四个裸站点。
- 本提交没有改动 `segment_op_wal.hpp`、codec 或任何 WAL payload/kind；B 仅改 CMake。
  reuse 的 legacy byte-invariance 用例包含在 59/59 专项中。

### 验证

新增回归逐项断言：

- `std::bad_alloc` 向调用方原样传播；
- updater 立即 poisoned，`ensure_readable()` 报 recovery-required；
- `debug_page_version(0)` 为 even；
- 绕过 poison gate 的 raw cache read 在 1 秒界内完成，且观察到回调留下的局部改写，
  证明不是靠 poison gate 掩盖永久 odd；
- 析构 poisoned handle 后重开，handle 可读且页面恢复为 durable base bytes。

| 验证 | 结果 | 退出码 |
|---|---:|---:|
| 新增 callback-throw 回归 | 1/1 | 0 |
| maintenance concurrency 套件 | 11/11 | 0 |
| `test_qg_updater_reuse --gtest_brief=1` | 59/59 | 0 |
| `test_segment_op_wal_reuse_crash --gtest_brief=1` | 31/31 | 0 |
| `test_qg_updater_wal --gtest_filter='Divergence/*'` | 8/8 | 0 |
| `test_segment_op_wal --gtest_filter='SegmentOpCodec.*Golden*'` | 2/2 | 0 |

## 任务 B：macOS mutable LASER 测试注册门

### 裁决

问题成立。新增 cache option `ALAYA_ENABLE_MUTABLE_LASER`：默认值当且仅当
`ALAYA_ENABLE_LASER=ON` 且 `CMAKE_SYSTEM_NAME` 为 `Linux` 时为 ON。这个可用条件与
`collection.hpp` 中生产宏 `ALAYA_COLLECTION_HAS_ACTIVE_LASER` 的
`ALAYA_ENABLE_LASER && __linux__` 条件对齐。

Linux 可显式设为 OFF，作为只缩减 writable 测试/工具注册的验证 lane；因此能力为 ON
必然意味着生产宏为真，默认配置下两者同真同假，而显式 OFF 是需求允许的单向测试抑制。
依赖不成立却强开不会静默降级：LASER=OFF 时强开为 FATAL，非 Linux 强开也为 FATAL，
错误信息明确提示保留 sealed LASER 并关闭 mutable capability。

### 证据

- `cmake/AlayaOptions.cmake` 集中定义默认值、option 注释和 FATAL validation；
  `cmake/PrintSummary.cmake` 显示最终 capability 值。
- `tests/laser/CMakeLists.txt` 用新门包住 updater、op-WAL、maintenance、crash、
  `MutableLaserSegment`、residency write mirror、update bench 和 PID reuse 目标。
- `tests/collection/CMakeLists.txt` 的外层仍为 `ALAYA_ENABLE_LASER`；其中 sealed filter、
  target 和 recall tests 留在外层，active adapter/active Collection 目标进入新内层门。
- `tests/CMakeLists.txt` 与 `tests/disk/CMakeLists.txt` 的 sealed LASER 注册保持原条件；
  `laser_segment_test` 没有被新 capability 约束。
- 未改任何生产源码、生产宏或 backend 实现。

显式 OFF 后消失的 16 个 build targets 为：

1. `bench_laser_update_sift`
2. `collection_active_laser_maintenance_test`
3. `collection_active_laser_recovery_test`
4. `collection_active_laser_test`
5. `mutable_laser_adapter_header_closure`
6. `mutable_laser_adapter_test`
7. `test_mutable_laser_segment`
8. `test_mutable_laser_segment_reuse`
9. `test_qg_maintenance_concurrency`
10. `test_qg_updater_reuse`
11. `test_qg_updater_unit`
12. `test_qg_updater_wal`
13. `test_segment_op_wal`
14. `test_segment_op_wal_crash`
15. `test_segment_op_wal_reuse_crash`
16. `test_unified_residency`

其中 update bench 与 header closure 不是 CTest；其余对应 14 个消失的 CTest。默认树仍为
94 项，OFF 树为 80 项，差集没有额外目标。

### 验证

| lane / 检查 | 结果 | 退出码 |
|---|---:|---:|
| 默认 Linux configure，LASER=ON / MUTABLE=ON | 通过，94 项注册 | 0 |
| 默认 Linux all-target build | 173/173 work units | 0 |
| 默认 Linux CTest（直接进程退出码，无管道） | 94/94 | 0 |
| MUTABLE=OFF configure | 通过，80 项注册 | 0 |
| MUTABLE=OFF all-target build | 完成（authoritative continuation 187/187） | 0 |
| MUTABLE=OFF CTest（直接进程退出码，无管道） | 80/80 | 0 |
| LASER=OFF configure 抽查 | 通过，MUTABLE 自动 OFF | 0 |
| 模拟 Darwin 显式 MUTABLE=ON option validation | 预期 FATAL | 1（预期） |

OFF lane 中下列保留面均实际编译并通过：

- `laser_segment_test`；
- `laser_test_threadpool_file_reader`；
- `laser_test_ip_oracle`；
- `laser_test_ip_kernel`；
- sealed Collection 的 filter、target 和 recall tests。

验证环境说明：新 OFF 树第一次 all-target build 在 27/214 处被主 checkout `.venv` 的
`_alayalite_editable` meta finder 拦截；请求的是本工作树扩展，实际被抢先加载主 checkout
扩展，fixture provenance 因而按设计拒绝。随后复用既有盲审验证方法，在 ignored build
目录放置 `sitecustomize.py` 移除该 finder，并通过 `PYTHONPATH` 传给 build/CTest。fixture
明确打印加载本树扩展后，最终 all-target build 与 80/80 CTest 均退出 0。该辅助文件未提交。

## 最终质量门

- `git diff --check`：clean。
- `uvx pre-commit run --all-files`：全部 hooks Passed，0 Failed，退出码 0。
- 未 push。
