# Claude × Codex 协作工作流 v1.2

## 【项目变量区】（每个项目按需替换）

```yaml
PROJECT_LANG: "C++20"
PROJECT_STYLE: "CamelCase 类, lower_case 函数/变量, 成员变量后缀 _ (如 data_), kConstantName 常量"
PROJECT_CONSTRAINTS:
  - "include/ 下 header-only，不创建 .cpp（测试和工具代码不受此限）"
  - "优先模板 + Concepts 静态多态，仅 hot path 使用 CRTP"
  - "coroutine-based 并发，不用裸线程"
  - "所有 header 修改必须考虑编译时间和 ODR（One Definition Rule）风险"
BUILD_CMD: "make build"
TEST_CMD: "make test"
LINT_CMD: "make lint"
CODEX_MODEL_HEAVY: "gpt-5.3-codex xhigh"
CODEX_MODEL_LIGHT: "gpt-5.3-codex high"
MAX_PARALLEL: 3
MAX_RETRY: 2
PERF_BASELINE: "当前主分支运行 3 次取中位数（正式发布前 5 次）"
PERF_THRESHOLD: "latency ≤ 1.05x baseline, throughput ≥ 0.95x baseline"
```

---

## 【执行流程】

### Phase 0: Preflight

- 检查 codex-plugin-cc 可用性、模型可选项
- 检查 openspec 工件 / 任务来源是否存在
- **最小能力要求**：
  - codex-plugin-cc 可响应调用
  - 至少一个 CODEX_MODEL 可用
  - 项目可执行 BUILD_CMD
- **降级策略**（非硬停止）：
  - Codex 不可用 → 降级为 Claude-only 串行模式，标注 `[degraded: no-codex]`
  - CODEX_MODEL_HEAVY 不可用 → 全部使用 CODEX_MODEL_LIGHT
  - openspec 工件缺失 → 从用户指令 / issue 直接构建 DAG

### Phase 1: Planning（Claude 主导）

- 阅读任务来源（openspec 计划 / issue / 用户指令），输出子任务 DAG
- 每个子任务用固定字段：

| 字段 | 说明 |
|------|------|
| `task_id` | 唯一标识，如 T1, T2 |
| `goal` | 一句话目标 |
| `write_set` | 该任务会修改的文件列表 |
| `read_set` | [高风险任务必填] 该任务需读取但不修改的文件 |
| `deps` | 依赖的 task_id 列表，无依赖填 `[]` |
| `interface` | 输入/输出接口签名（代码块） |
| `acceptance` | 验收标准（编译通过 / 测试名 / 性能阈值） |
| `complexity` | `low` / `medium` / `high`，用于 Phase 2 模型选择 |
| `recommended_model` | `HEAVY` / `LIGHT`，基于 complexity 推荐 |
| `risk` | [可选] 仅 hot path 或高风险任务标注 |

- 识别可并行组，需**同时满足全部条件**：
  - `write_set` 完全不重叠
  - `read_set` ∩ 其他任务的 `write_set` = 空（无读-写冲突）
  - 无共享接口变更（含模板签名、公共常量、宏定义、inline 变量）
  - `deps` 无交叉
- **强制串行规则**：涉及公共头文件模板签名 / inline 变量 / constexpr 常量变更的任务，一律串行
- **DAG 输出格式（必须严格遵守）**：

```json
{
  "tasks": [
    {
      "task_id": "T1",
      "goal": "...",
      "write_set": ["file1.hpp"],
      "read_set": [],
      "deps": [],
      "interface": "...",
      "acceptance": "...",
      "complexity": "high",
      "recommended_model": "HEAVY",
      "risk": ""
    }
  ],
  "parallel_groups": [
    { "group": "A", "tasks": ["T1", "T3"], "reason": "write_set 不重叠，无共享接口" },
    { "group": "B", "tasks": ["T4", "T5"], "reason": "..." }
  ],
  "serial_order": ["group_A", "T2", "group_B"],
  "conflict_analysis": [
    "T1 和 T2 共享 space.hpp 模板签名，必须串行",
    "T4 和 T5 write_set 无交叉，read_set 无冲突"
  ]
}
```

- 高风险任务（`complexity: high` 或 `risk` 非空）额外输出 Self-Critique：检查依赖遗漏、接口不一致、并行冲突

### Phase 2: Dispatch（并行调用 Codex）

- 按 DAG 拓扑序，对每个并行组同时发起 codex-plugin-cc 调用（上限 `MAX_PARALLEL` 路）
- **模型选择**：优先使用任务的 `recommended_model`，未标注时按规则：
  - `complexity: high` → `CODEX_MODEL_HEAVY`
  - `complexity: low/medium` → `CODEX_MODEL_LIGHT`
- **上下文策略**（分级，避免 token 爆炸）：
  - 小文件（≤300 行）→ 附完整内容
  - 大文件（>300 行）→ 附相关函数/类片段 + 行号锚点 + 文件结构摘要
  - 核心头文件（标注 `[full_context]`）→ 即使超 300 行也附完整内容（上限 500 行）
  - 公共接口 → 始终附完整签名
- **每次调用必须包含**：
  - 目标文件内容（按上下文策略）
  - 相关接口片段（被调用方 / 调用方签名）
  - `PROJECT_STYLE` + `PROJECT_CONSTRAINTS` 摘要
  - 项目风格备忘（从历史风格错误沉淀，见【风格规则积累区】）
  - 明确禁止：不改动 write_set 外的文件、不重命名公共符号、不手动重排 include（允许 formatter 自动调整）
- **强制输出格式**：unified diff + 修改文件清单 + write_set 内未改动文件清单（用于范围审计：确认未越界修改）
- **重试策略（progressive prompting）**：
  - 第 1 次重试：附完整编译/接口错误信息 + 相关上下文
  - 第 2 次重试：附上次 Codex diff + 错误信息 + 最小修正提示

### Phase 3: Integrate & Cross-Review

```
┌─────────────────────────────────────────────────┐
│           Cross-Review Protocol                 │
│                                                 │
│  Codex 写的代码 ──→ Claude 审核                 │
│  Claude 写的代码 ──→ Codex 审核                 │
│  谁写的代码都不能自审自过                        │
│                                                 │
│  降级例外：Codex 不可用时 Claude 自写自审，      │
│  标注 [unreviewed]，纳入人工复查清单             │
└─────────────────────────────────────────────────┘
```

**3a. Claude 审核 Codex 产出：**

- 按依赖序合并
- 审查：编译正确性、接口一致性、风格合规、边界条件、lifetime/dangling reference
- 通过 → 合入
- 不通过 → **按错误分类处理**：
  - `编译错` → 附完整错误信息 + 相关上下文，Codex 重新生成（progressive prompting）
  - `接口错` → 附正确接口签名，Codex 重新生成
  - `风格错` → Claude 直接修复，不消耗重试次数，并将修正规则沉淀到【风格规则积累区】
- 编译错/接口错最多重试 `MAX_RETRY` 次

**3b. 重试耗尽后的升级路径：**

- Claude 自己写该子任务代码
- 将代码发送给 Codex 审核（使用 `CODEX_MODEL_LIGHT`），审核提示：

```
审核以下代码，检查：
1. 与现有接口的一致性（附上相关接口签名）
2. 语言惯用法是否正确使用
3. 性能隐患（不必要的拷贝、缓存不友好的布局）
4. 边界条件遗漏
5. lifetime / dangling reference 风险
6. Concepts 约束完整性（是否遗漏必要约束或过度约束）
输出严重级别 + 修改建议：
- BLOCKER: 必须修复才能合入（附 unified diff）
- MAJOR: 强烈建议修复（附 unified diff）
- MINOR: 可选优化（文字说明即可）
最终结论：PASS / FAIL（仅存在 BLOCKER 时 FAIL）
```

- Codex 返回 PASS 或仅 MAJOR/MINOR → 合入（MAJOR 由 Claude 酌情修复）
- Codex 返回 FAIL (BLOCKER) → Claude 应用 BLOCKER diff 后合入
- **终止条件**：若 Codex 审核本身也失败（超时/不可用/输出无法解析）→ Claude 自主判断合入，标注 `[unreviewed]`，加入人工复查清单

**3c. 合并冲突处理：**

- 并行产出文本可合并但语义冲突（如模板签名不兼容）→ 编译验证捕获
- 并行产出冲突 → 回退为串行，保留先完成的结果，冲突任务重新 Dispatch

### Phase 4: Verify & Report

- **分层验证**：
  - 子任务级：每个子任务完成后立即跑最小相关验证（单个测试 / 编译检查）
  - Sprint 级：全部子任务完成后跑全量门禁
- **完成标准（DoD）**：
  - [ ] `BUILD_CMD` 编译通过
  - [ ] 相关单元测试通过
  - [ ] `LINT_CMD` 风格检查通过
  - [ ] 性能验证：按 `PERF_BASELINE` 方法测量，结果满足 `PERF_THRESHOLD`（无基准时跳过并标注）
- **输出**：
  - 最终 diff 或完整文件
  - 验证命令 + 实际执行结果
  - 已完成任务表
  - `[unreviewed]` 标记清单（如有）
  - 下一轮任务队列（标注并行组）

---

## 【Dispatch 子提示模板】

每次调用 Codex 时使用此结构：

```
你是一个严格遵守项目风格和约束的 {PROJECT_LANG} 专家，只输出要求的格式，不添加额外解释。

## Task
{task_id}: {goal 一句话}

## Context Files
<!-- 小文件(≤300行): 完整内容 -->
<!-- 大文件(>300行): 相关片段 + 行号锚点 + 文件结构摘要 -->
<!-- 核心头文件([full_context]): 完整内容，上限500行 -->
{文件内容}

## Related Interfaces
{被调用/调用方的接口签名片段，始终完整}

## Constraints
- Language: {PROJECT_LANG}
- Style: {PROJECT_STYLE}
- Project rules: {PROJECT_CONSTRAINTS 逐条}
- Style memo (from past corrections): {风格规则积累区内容，如有}
- DO NOT modify files outside write_set: {write_set}
- DO NOT rename public symbols
- DO NOT manually reorder includes (formatter may adjust)

## Acceptance Criteria
{从 Phase 1 DAG JSON 复制该任务的 acceptance}

## Output Format
1. Unified diff (```diff 代码块)
2. Modified files list
3. Unmodified files list (within write_set only, for scope audit)
```

---

## 【Review 子提示模板】

Codex 审核 Claude 代码时使用此结构：

```
你是一个严格的代码审核者。审核以下由 Claude 编写的代码。

## Code Under Review
{Claude 编写的代码，unified diff 格式}

## Related Interfaces
{相关接口签名片段}

## Project Constraints
- Language: {PROJECT_LANG}
- Style: {PROJECT_STYLE}
- Rules: {PROJECT_CONSTRAINTS 逐条}

## Review Checklist
1. 与现有接口的一致性
2. 语言惯用法是否正确使用
3. 性能隐患（不必要的拷贝、缓存不友好的布局）
4. 边界条件遗漏
5. lifetime / dangling reference 风险
6. Concepts 约束完整性

## Output Format
Per finding:
- Severity: BLOCKER / MAJOR / MINOR
- Location: file:line
- Description
- Fix: unified diff (BLOCKER/MAJOR) or text (MINOR)

Final verdict: PASS (no BLOCKER) / FAIL (has BLOCKER)
```

---

## 【风格规则积累区】

从历史风格错误中沉淀的短规则（上限 5 条，超出时淘汰最旧的）。
每次 Claude 修复 Codex 的风格错后，将修正规则追加到此区，并在后续 Dispatch 的 Constraints 中引用。

```
1. (待积累)
2. (待积累)
3. (待积累)
4. (待积累)
5. (待积累)
```

---

## 【快速参考】

| 场景 | 动作 | 备注 |
|------|------|------|
| 子任务独立、写集+读集+符号不重叠 | 并行 Dispatch（上限 `MAX_PARALLEL`） | |
| 涉及公共头模板签名/inline/constexpr 变更 | **强制串行** | 即使 write_set 不重叠 |
| 有依赖或共享接口 | 严格串行 | |
| Codex 产出：风格错 | Claude 直接修 + 沉淀规则 | 不计重试 |
| Codex 产出：编译错/接口错 | progressive prompting 重试 | ≤ `MAX_RETRY` 次 |
| 重试耗尽 | Claude 写 → Codex 审 | 分 BLOCKER/MAJOR/MINOR |
| Codex 审核也失败 | Claude 合入 + `[unreviewed]` | 纳入人工复查 |
| 并行产出冲突 | 回退串行，保留先完成的 | |
| `complexity: low/medium` | `CODEX_MODEL_LIGHT` | |
| `complexity: high` | `CODEX_MODEL_HEAVY` | |
| Codex 完全不可用 | 降级 Claude-only + `[degraded]` | |
| 小文件 ≤300 行 | 完整内容 | |
| 大文件 >300 行 | 片段 + 行号锚点 | |
| 核心头文件 `[full_context]` | 完整内容（上限 500 行） | |

---

## 【v1.1 → v1.2 变更记录】

| 变更 | 来源 | 说明 |
|------|------|------|
| `PROJECT_STYLE` 明确 `成员变量后缀 _` | Grok #7 | 消除歧义 |
| 新增 ODR 风险约束 | Grok #15 | header-only 项目必备 |
| 新增 `PERF_BASELINE` 定义 | Grok #4 | 快速 3 次 / 正式 5 次中位数 |
| Phase 0 新增最小能力要求 | Grok #10 | 避免降级逻辑歧义 |
| 新增 `read_set` 字段（高风险必填） | Grok #1 | 增强并行安全，但不强制全量填写 |
| 新增 `complexity` + `recommended_model` | Grok #13 | 减少模型选型漂移 |
| 公共头/模板签名变更强制串行 | Grok #1 + Claude/Codex | 规则兜底，比全量 read_set 更实际 |
| DAG 输出强制 JSON 格式 | Grok #3 | 避免格式漂移，只要一种格式 |
| 高风险任务 Self-Critique | Grok #11 | 非默认全开，仅高风险时触发 |
| 上下文策略新增 `[full_context]` 标记 | Grok #2 | 核心头文件可放宽到 500 行 |
| Review prompt 补 lifetime/Concepts 检查 | Grok #5 | 消除审核不对称 |
| 新增独立 Review 子提示模板 | Grok #5 | 与 Dispatch 模板同等详细度 |
| 风格规则积累区 | Grok #6 | 短规则沉淀，上限 5 条 |
| Dispatch 模板新增 Role reminder | Grok #12 | 一行角色提醒 |
| Progressive prompting 重试 | Grok #14 | 附上次 diff + 错误 + 最小修正提示 |
| Unmodified files list 注明用途 | Grok #8 | 范围审计 |
| 拒绝 explicit CoT 指令 | Grok #9 | Claude + Codex 一致拒绝 |
