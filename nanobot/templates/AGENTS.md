# Agent Instructions

## Workspace Guidance

Use this file for project-specific preferences, recurring workflow conventions, and instructions you want the agent to remember for this workspace. Keep durable facts about the user in `USER.md`, personality/style guidance in `SOUL.md`, and long-term memory in `memory/MEMORY.md`.

## Scheduled Reminders

Before scheduling reminders, check available skills and follow skill guidance first.
Use the built-in `cron` tool to create/list/remove jobs (do not call `nanobot cron` via `exec`).
Get USER_ID and CHANNEL from the current session (e.g., `8281248569` and `telegram` from `telegram:8281248569`).

**Do NOT just write reminders to MEMORY.md** — that won't trigger actual notifications.

## File Writing — User-Specified Paths

When the user explicitly specifies a target file path (e.g., "写入这个文件", "写回原文件", "保存到 X", "save to X"):

- **Write directly to that path** using `write_file`. Do NOT create an intermediate file (e.g., `_out.txt` in the workspace) and then ask for confirmation.
- Do NOT ask "需要写回原文件吗" — the user already told you where to write.
- The user's intent is the final destination. Honor it immediately.

## Interactive Choice Prompts (`ask_user`)

需要用户做明确选择时（不是要确认，是要选项），用 `ask_user` 工具弹出选择 popup。

### When to use

**应使用**：
- 用户请求模糊，需要在多个解释之间选（"用方案 A 还是 B？"）
- 实施前需要用户决定关键参数（"工作目录是 ./output 还是 ./dist？"）
- 多步流程中需要用户确认下一步方向

**不应使用**：
- 普通确认（"我可以开始了吗？"）— 直接做
- 仅一个明显的合理选项
- LLM 自己能合理推断的细节
- 一次会话超过 3 次提问（用户会烦）

### 调用规范

- `questions`: 1–10 个问题列表，每个：
  - `question`: 完整问句
  - `header`: 短标签（≤12 字符），如 "存储位置"
  - `options`: 2–6 个选项，每个 `{label, description}`
- 返回 JSON: `{answers: {问句: 选中label}}` 或 `{cancelled: true}` 或 `{error: ...}`
- `cancelled: true` 时立即改用纯文本提问，不要重试 ask_user

### 频道支持

仅 CLI（Textual TUI）支持 popup。其他频道（telegram/slack/email 等）返回 `{error: "...not supported..."}`，应改用纯文本问。

## In-Session Tasks (`todo_write`)

会话内的多步任务用 `todo_write` 工具跟踪。该工具维护一个结构化 todo 列表，自动注入 system prompt 顶部，重启后从 session 文件恢复。

### When to use `todo_write` vs Plan Tree

| 场景 | 用 `todo_write` | 用 Plan Tree (MEMORY.md) |
|------|----------------|-------------------------|
| 持续时长 | 单次会话内（数小时） | 跨会话、跨天/周 |
| 步骤数 | 3–20 步 | 任意（含子步骤） |
| 暂停-恢复语义 | 不需要 | 需要（明确暂停某计划恢复另一个） |
| 与外部权威文件对齐 | 不需要 | 可能（`> 来源:`） |
| 子步骤嵌套 | 不支持 | 支持递归子步骤 |

**经验法则**：默认用 `todo_write`；只有当任务跨会话、需暂停恢复、有外部来源结构、或有嵌套子步骤时才升级到 Plan Tree。两者可并存——长期 Plan Tree 下，每次执行某一大步骤时用 `todo_write` 拆出当前会话的小步骤。

### `todo_write` 调用规则

- **全量覆盖**：每次调用都要传完整 items 列表，不是增量。
- **状态值**：`pending` / `in_progress` / `completed`。
- **At most one `in_progress`**：同一时刻最多一个项处于 `in_progress`，代码强制校验。
- **每完成一步立即更新**：不要批量。开始下一步前先把上一步标 `completed`、新一步标 `in_progress`。
- **失败处理**：步骤失败时**不自动标 completed**，向用户报告失败原因，询问：重试 / 跳过 / 取消。
- **清空**：传 `items=[]` 清空列表。

### When to Create a Todo List

**应建**：任务有 ≥3 个有序步骤 / 需跨多轮完成 / 用户明确要求"列出步骤"
**不建**：一次性任务 / ≤2 步且当轮可完成 / 纯查询

## Cross-Session Plans (Plan Tree)

跨会话的长期、阶段性任务 — 学习计划、跨周重构、feature rollout、investigations — 用 Plan Tree 记到 MEMORY.md。当任务仅在单次会话内完成时，**优先用 `todo_write`**（见上节）。

### Format

```
## 计划名称  [🔄进行中]  2026-05-19
> 简介: 一句话说明（可选）
> 来源: 外部权威文件路径（如有，严格遵循其结构，不得自创步骤或重新编号）
> 前置: 依赖的其他计划名（如有）
- [✅] 步骤1
- [🔄] 步骤2  ← 已完成 a.py，剩余 b.py（中断位置备注写在这里）
  - [✅] 子步骤2a
  - [⬜] 子步骤2b
- [⬜] 步骤3
```

**Plan-level status**（写在 `##` 标题行）：`[⬜未开始]` · `[🔄进行中]` · `[⏸暂停]` · `[✅完成]` · `[❌已取消]`  
**Step-level markers**：`[⬜]` 未开始 · `[🔄]` 进行中 · `[✅]` 已完成 · `[❌]` 失败/跳过  
同一话题同一时刻**最多一个** `[🔄进行中]` 计划。

### When to Reference a Plan

**只在**用户明确表达继续/查询意图时才查计划——如"继续"、"下一步"、"还剩什么"、"计划进度"、"继续重构"、"恢复计划B"等。

> ⚠️ **严格禁止**：在话题切换、会话启动、打招呼（"你好"、"hi"等）时主动提及任何计划——即使 MEMORY.md 中有活跃计划，也**绝对不能**主动说"你还有个计划要继续吗"。这条规则的优先级高于一切"主动帮助"的倾向。

### Disambiguation: Which Plan to Continue

用户说"继续"（未指明计划）时，按优先级处理：

1. 用户点名（"继续重构" / "恢复计划B"）→ 直接定位该计划
2. 恰好一个 `[🔄进行中]` → 继续它，无需确认
3. 无 `[🔄]`，恰好一个 `[⏸暂停]` → 恢复它，简要说明
4. 无 `[🔄]`，多个 `[⏸暂停]` → 列出所有，让用户选择
5. 无活跃/暂停计划 → 告知用户，询问是否开启新计划

### Interrupting & Resuming

- **新计划打断当前计划**：将当前 `[🔄进行中]` 改为 `[⏸暂停]`，在被中断步骤旁注明中断位置；新计划标为 `[🔄进行中]`
- **恢复暂停计划**：目标计划改回 `[🔄进行中]`，原活跃计划改为 `[⏸暂停]`

### Executing Steps

- 找活跃计划的第一个 `[⬜]` 步骤执行；执行中标 `[🔄]`，完成后标 `[✅]`
- 每完成一步**立即增量更新** MEMORY.md，不批量写入
- 子步骤递归适用相同状态标记
- 有 `来源` 文件的计划：严格遵循该文件结构，不得自创步骤或重新编号

### Failed Steps & Completion

- 步骤失败：标 `[❌]` 附原因，**不自动跳至下一步**，询问用户：重试 / 跳过 / 取消计划
- 计划完成：计划头改为 `[✅完成]`，告知用户，**不自动开始下一个计划**，等待指令

### When to Create a Plan

**应建**：用户明确要求 / 任务有 ≥3 个有序步骤 / 需跨多轮完成  
**不建**：一次性任务 / ≤2 步且当轮可完成 / 纯查询  
**主动建议时**：先呈现计划草稿供用户确认，批准后再执行

## Heartbeat Tasks

`HEARTBEAT.md` is checked on the configured heartbeat interval. Use file tools to manage periodic tasks.

- Use `apply_patch` for normal task-list updates, especially when adding, removing, or changing multiple lines.
- Use `edit_file` only for small exact replacements copied from the current `HEARTBEAT.md`.
- Use `write_file` for first creation or intentional full-file rewrites.

When the user asks for a recurring/periodic task, update `HEARTBEAT.md` instead of creating a one-time cron reminder.
