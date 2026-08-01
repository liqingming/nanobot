# Fork 变更清单与 upstream 同步手册

本文档记录 `liqingming/nanobot` 相对 `HKUDS/nanobot` 的本地 fork 增量。后续同步 `upstream/main` 时，先读本文档，再处理冲突。

当前同步背景：

- 当前日常使用分支：`main`，最近 fork 提交 `384c602a 增加模型错误后的任务级自动恢复`。
- 当前 upstream 对比点：`upstream/main` 已更新到 2026-07-08。
- 当前隔离同步分支：`sync-upstream-20260709`，worktree 位于 `E:\learn\nanobot-sync-upstream`。
- 当前 merge-base：`3d3ef586e791894d07daf4b108cca907d59c082d`。

同步总原则：

1. 保留 upstream 全量能力，不做裁剪式同步。
2. 保留本 fork 的 CLI/TUI、学习记忆、日志、模型恢复、运行体验等增量。
3. `nanobot/fork/*` 是 fork 自有模块。upstream 删除该目录时，合并冲突应选择保留本地。
4. 真正需要手工处理的是 core 接入点：upstream 重构后，fork 模块必须重新接回新结构。
5. 合并时不能简单全选 ours 或 theirs。大多数 core 文件应以 upstream 为主体，再把 fork 接入点移植回去。

## 1. 必须保留的 fork 自有目录

这些文件 upstream 没有，合并时删除它们通常都是错误：

- `nanobot/fork/__init__.py`
- `nanobot/fork/agent/learning.py`
- `nanobot/fork/agent/topic_memory.py`
- `nanobot/fork/agent/tools/ask_user.py`
- `nanobot/fork/agent/tools/memory_search.py`
- `nanobot/fork/agent/tools/skill.py`
- `nanobot/fork/agent/tools/todo.py`
- `nanobot/fork/cli/tui.py`
- `nanobot/fork/cli/tui_base.py`
- `nanobot/fork/cli/tui_factory.py`
- `nanobot/fork/cli/tui_keys.py`
- `nanobot/fork/cli/tui_textual.py`
- `nanobot/fork/providers/claude_ai_oauth_provider.py`
- `nanobot/fork/utils/tool_hints.py`
- `tests/fork/*`

保留策略：

- 若 upstream merge 显示这些文件被删除，选择保留本地。
- 若 upstream 新结构移动了同类能力，不直接删除 fork 版本，先确认 fork 是否仍承担本地 CLI/TUI 行为。
- `tests/fork/*` 是同步后验收 fork 行为的最低保护网，也要保留。

## 2. 核心接入点总览

这些是 fork 模块接入 upstream core 的位置。冲突处理时必须逐项确认。

| 接入点 | 文件 | 丢失后果 |
| --- | --- | --- |
| fork bootstrap | `nanobot/__init__.py` | `nanobot.fork` 不再导入，fork 工具自注册失效 |
| fork tool registry | `nanobot/agent/tools/registry.py` | `todo`、`ask_user`、`search_history`、`load_skill` 消失 |
| TUI 入口 | `nanobot/cli/commands.py` | `nanobot agent` 回退到普通交互，Textual TUI 失效 |
| TUI 后端配置 | `nanobot/config/schema.py` | `tui_backend`、`enable_learning` 等配置失效 |
| 学习上下文 | `nanobot/agent/loop.py`、`nanobot/agent/context.py` | TurnSummary、topic memory、history search 断链 |
| per-topic memory | `nanobot/agent/memory.py`、`nanobot/session/manager.py` | 话题隔离记忆和历史压缩行为异常 |
| tool hints | `nanobot/agent/progress_hook.py`、`nanobot/fork/utils/tool_hints.py` | CLI/TUI 工具痕迹显示退化 |
| Claude.ai OAuth | `nanobot/providers/factory.py`、`nanobot/providers/registry.py`、`nanobot/cli/commands.py` | `claude_ai` provider 退化或登录不可用 |
| runtime logs | `nanobot/session/manager.py`、`nanobot/utils/session_runtime_log.py`、`nanobot/agent/loop.py` | 会话目录下不再写运行日志，报错定位能力丢失 |
| task auto recovery | `nanobot/agent/loop.py` | LLM server error 后不能自动发起任务级恢复 |
| slash commands | `nanobot/command/*`、`nanobot/cli/commands.py`、`nanobot/fork/cli/*` | 命令选择可能被当普通消息发给 LLM |

## 3. 5 月基线能力

### 3.1 CLI/TUI 双后端

主要文件：

- `nanobot/fork/cli/tui_base.py`
- `nanobot/fork/cli/tui_factory.py`
- `nanobot/fork/cli/tui.py`
- `nanobot/fork/cli/tui_textual.py`
- `nanobot/fork/cli/tui_keys.py`
- `nanobot/cli/commands.py`

能力：

- `nanobot agent` 使用全屏 TUI，支持边输出边输入。
- 默认后端是 Textual，可通过 `tui_backend` 或 `NANOBOT_TUI` 切换。
- 支持话题选择、新话题命名、`/new`、`/resume`、`/todos`、`/continue`、`/commit_memory`、`/exit`。
- 支持工具痕迹、todo bar、ask_user 弹窗、reasoning 隐藏/缓冲、流式输出 spinner。
- 支持鼠标选择复制、滚轮滚动、历史恢复。

同步注意：

- upstream 如果重写 `nanobot/cli/commands.py`，应保留新的 command 功能，但必须重新接入 `create_tui()`。
- `commands.py` 的 outbound consumer 是 TUI 的核心分发点，不能丢掉 `_ask_user`、todo diff、tool events、reasoning 拦截。
- `tui_keys.py` 是 prompt_toolkit 和 Textual 共用的按键决策层，命令弹窗和新话题输入都依赖它。

### 3.2 学习、话题记忆、历史检索

主要文件：

- `nanobot/fork/agent/learning.py`
- `nanobot/fork/agent/topic_memory.py`
- `nanobot/fork/agent/tools/memory_search.py`
- `nanobot/agent/loop.py`
- `nanobot/agent/context.py`
- `nanobot/agent/memory.py`

能力：

- 每轮结束生成 `TurnSummary`，下轮按重要性注入上下文。
- `PatternStore` 统计跨会话工具序列，识别重复模式。
- 每个 topic 有独立 `MemoryStore`，目录位于 data dir 下的 topic memory 子目录。
- `search_history` 支持全局和话题历史检索，包含 CJK token 和中英同义扩展。
- `MemoryStore(memory_dir_override=...)` 是 topic memory 的关键底层钩子。

同步注意：

- upstream 如果重构 `ContextBuilder`，要重新确认 global memory、topic memory、todos、learning ctx 的注入顺序。
- upstream 如果重构 `AgentLoop`，要重新接 `_capture_turn_summary`、`_build_learning_ctx`、`clear_session_learning`。
- `enable_learning=False` 时不能注册 search_history，也不能注入 learning ctx。

### 3.3 fork 工具

主要文件：

- `nanobot/fork/agent/tools/todo.py`
- `nanobot/fork/agent/tools/ask_user.py`
- `nanobot/fork/agent/tools/memory_search.py`
- `nanobot/fork/agent/tools/skill.py`
- `nanobot/agent/tools/registry.py`

能力：

- `todo_write`：结构化任务计划，最多一个 `in_progress`，支持 TUI todo bar。
- `ask_user`：CLI/TUI 中的交互式提问，支持 metadata 回填和 timeout。
- `search_history`：历史检索。
- `load_skill`：按 skill 名加载 `SKILL.md`，失败时列出可用 skill。

同步注意：

- `nanobot.fork.__init__` 通过导入工具模块触发自注册。
- `ToolRegistry` 必须保留 `register_fork_tool()` / `iter_fork_tool_factories()` 这类接入机制，或迁移到 upstream 新的插件机制。
- 如果 upstream 已新增同名/同类工具，先比较行为，不直接替换 fork 版本。

### 3.4 Claude.ai OAuth provider

主要文件：

- `nanobot/fork/providers/claude_ai_oauth_provider.py`
- `nanobot/providers/factory.py`
- `nanobot/providers/registry.py`
- `nanobot/cli/commands.py`

能力：

- 使用 Claude.ai 订阅账号 OAuth token 调 Claude。
- token 优先读取 `~/.nanobot/credentials/claude_ai.json`，可回退 Claude Code 凭证。
- `providers.factory` 中必须有 `claude_ai_oauth` 分支。

同步注意：

- upstream provider factory 变化时，不能让 `claude_ai` 静默退化为 OpenAI-compatible provider。
- CLI provider login/logout 也要保留 `claude_ai` 分支。

### 3.5 tool trace 和 tool result summary

主要文件：

- `nanobot/agent/tools/base.py`
- `nanobot/agent/tools/filesystem.py`
- `nanobot/agent/tools/web.py`
- `nanobot/agent/tools/shell.py`
- `nanobot/fork/utils/tool_hints.py`

能力：

- 工具执行时显示短 trace，例如 `read_file("./src/main.py")`。
- 路径优先显示 workspace-relative，Windows 路径统一成 `/`。
- shell 输出做 UTF-8/OEM/locale 解码降级，避免中文 Windows 控制台乱码。
- 工具结果可以生成摘要，供 TUI 展示。

同步注意：

- upstream 如果已有新的 runtime events/tool events，应把 fork 的格式化能力接到新事件管线，而不是保留旧重复通道。

## 4. 7 月新增和近期必须补保的能力

### 4.1 CLI 默认行为：`nanobot` 等同 `nanobot agent`

主要文件：

- `nanobot/cli/commands.py`
- `tests/cli/test_commands.py`

能力：

- 不带参数运行 `nanobot` 时进入 agent，而不是只显示 help。
- help 仍然可通过 `nanobot --help` 使用。

同步注意：

- upstream 如果改 Typer callback 或 command 分发，必须保留无参启动 agent 的默认路径。

### 4.2 运行日志写入话题目录

主要文件：

- `nanobot/session/manager.py`
- `nanobot/utils/session_runtime_log.py`
- `nanobot/agent/loop.py`
- `tests/session/test_session_runtime_log.py`

能力：

- 每个会话/话题目录写 `runtime.log.jsonl`。
- 日志包括 turn start/end、state enter/exit、异常、runner error、auto recovery 调度等。
- 日志路径与 session key 绑定，例如 `sessions/cli_<topic>/runtime.log.jsonl`。
- 异常字段保留较长 traceback，避免只看到 `Sorry, I encountered an error.`。

同步注意：

- upstream 如果删除 `nanobot/utils/session_runtime_log.py`，应保留 fork 版本，或迁移到 upstream 新的 run records/logging 机制。
- `AgentLoop` 状态机重构时，要把 `append_session_runtime_log()` 接回每个关键节点。

### 4.3 模型错误可见化和任务级自动恢复

主要文件：

- `nanobot/agent/runner.py`
- `nanobot/agent/loop.py`
- `nanobot/cli/commands.py`
- `tests/agent/test_loop_runner_integration.py`
- `tests/agent/test_runner_errors.py`

能力：

- provider/model server error 不再只吞成通用提示，CLI/TUI 要能明显看到错误。
- server error 类中断会触发任务级恢复，最多 `_AUTO_RECOVER_MAX_ATTEMPTS = 2`。
- 自动恢复通过内部消息重新进入同一 session，metadata 标记 `_auto_recovery`、`_auto_recover_attempt`。
- 恢复提示为“请继续上次因模型服务错误中断的任务。”。

同步注意：

- upstream 如果引入自己的 retry/fallback，需要区分 provider 单次请求重试和 fork 的“任务级恢复”。
- 如果 upstream 改 `LLMResponse.finish_reason` 或 runner return shape，要重新验证 `_should_auto_recover_model_error()`。

### 4.4 slash command 统一交互

主要文件：

- `nanobot/command/builtin.py`
- `nanobot/command/router.py`
- `nanobot/cli/commands.py`
- `nanobot/fork/cli/tui.py`
- `nanobot/fork/cli/tui_textual.py`
- `nanobot/fork/cli/tui_keys.py`
- `tests/command/test_router_dispatchable.py`
- `tests/cli/test_tui_textual_pilot.py`

能力：

- `/model`、`/goal`、`/dream`、`/dream-log`、`/dream-restore`、`/pairing` 等命令进入统一 palette。
- 有参数的命令选择后进入编辑模式，不立即提交。
- 无参数直接执行类命令可以立即提交。
- `/new` 新建话题时，话题名不能被当普通用户消息发给 LLM。
- 选择命令不能触发 pre-submit spinner，也不能污染输入历史。
- mid-turn 时可识别 dispatchable slash command，避免排进 LLM 注入队列。

同步注意：

- upstream 如果新增命令，必须补到 TUI palette。
- command metadata 或 arg hint 是判断“edit/submit”的关键，不要只按字符串匹配。
- `CommandRouter.is_dispatchable_command()` 是 mid-turn 命令处理的关键守卫。

### 4.5 模型 preset 和 reasoningEffort 显示/切换

主要文件：

- `nanobot/config/schema.py`
- `nanobot/agent/model_presets.py`
- `nanobot/agent/loop.py`
- `nanobot/command/builtin.py`
- `nanobot/cli/commands.py`
- `nanobot/fork/cli/tui_textual.py`
- `tests/config/test_model_presets.py`
- `tests/command/test_model_command.py`

能力：

- 顶层 `modelPresets` 支持 `gpt55-minimal`、`gpt55-low`、`gpt55-medium`、`gpt55-high` 等模式。
- `/model` 可查看当前模型和 preset，`/model <preset>` 运行时切换，`/model default` 回到默认。
- 切换只影响后续 turn，不重写 `config.json`。
- 欢迎界面显示 model、preset、reasoningEffort。

同步注意：

- upstream 如果也有 model preset，需要保留 fork 的 CLI/TUI 显示和 `/model` command 行为。
- provider snapshot 切换后必须同步 `loop.context_window_tokens`、`runner.provider`、`subagents`、`consolidator`、`dream`。

### 4.6 TUI 粘贴和输入框增强

主要文件：

- `nanobot/fork/cli/tui_textual.py`
- `nanobot/fork/cli/tui.py`
- `tests/cli/test_tui_textual_pilot.py`

能力：

- 支持 bracketed paste。
- 多行粘贴显示为 `[pasted N lines]` token，提交时恢复原始 payload。
- 大于阈值的单行粘贴显示为 `[pasted N chars]` token，避免输入框被大文本刷屏。
- 如果用户编辑 paste token，则按可见文本提交，避免错发隐藏 payload。
- 多个 paste token 按显示顺序恢复。

同步注意：

- upstream 如果重写 composer/input widget，必须保留 hidden payload 机制。
- 这个能力修复了 Windows/终端大粘贴先弹窗导致无法继续复制的问题。

### 4.7 TUI 滚动行为和流式输出

主要文件：

- `nanobot/fork/cli/tui_textual.py`
- `nanobot/fork/cli/tui.py`
- `tests/cli/test_tui_textual_pilot.py`
- `tests/cli/test_tui_prompt_e2e.py`

能力：

- 用户手动向上查看历史时，流式输出不能强制拉回底部。
- 如果用户本来在底部，则新输出自动跟随到底部。
- 用户发送新消息或新 stream 开始时，应恢复跟随最新回复。
- stream delta debounce 尊重手动滚动窗口，避免粘滞感。
- Textual output 支持列选择、鼠标拖选复制。

同步注意：

- `_OutputLog.write(..., follow=...)`、`is_at_bottom()`、`user_is_scrolling()`、`mark_user_scroll()` 是关键点。
- 不要用简单 `scroll_end()` 覆盖所有输出路径。

### 4.8 文件编辑 diff 展示

主要文件：

- `nanobot/fork/cli/tui_base.py`
- `nanobot/fork/cli/tui_textual.py`
- `nanobot/cli/commands.py`
- `nanobot/utils/file_edit_events.py`
- `nanobot/session/webui_turns.py`
- `webui/src/lib/types.ts`
- `tests/utils/test_file_edit_events.py`

能力：

- CLI/TUI 能展示文件编辑 diff 进度。
- file edit events 同时服务 CLI 和 WebUI。

同步注意：

- upstream 近期也新增了 file edit diff progress view。合并时应优先复用 upstream 事件结构，再保留 TUI 展示层。
- 冲突文件 `nanobot/utils/file_edit_events.py` 和 `webui/src/lib/types.ts` 需要重点对齐字段名。

### 4.9 workspace 显示和话题栏

主要文件：

- `nanobot/cli/commands.py`
- `nanobot/fork/cli/tui_factory.py`
- `nanobot/fork/cli/tui_textual.py`

能力：

- CLI topic bar 显示当前 workspace。
- 非默认 workspace 的 session/cache 仍写入对应 `~/.nanobot/caches/<workspace>_<hash>/`。

同步注意：

- upstream 如果更改 workspace/project 机制，需确认 fork 的 data dir 和 workspace 分离仍然成立。

### 4.10 加载 Claude 指令文件

主要文件：

- `nanobot/agent/context.py`
- `tests/agent/test_context_builder.py`

能力：

- 支持读取 Claude 风格指令文件并注入上下文。

同步注意：

- upstream 如果重构 prompt template/context builder，要保留该注入位置。
- 指令注入不能破坏 learning_ctx、runtime_ctx、topic memory 的相对顺序。

### 4.11 会话持久化增强

主要文件：

- `nanobot/agent/loop.py`
- `nanobot/agent/runner.py`
- `nanobot/session/manager.py`
- `nanobot/utils/atomic_write.py`
- `nanobot/utils/session_runtime_log.py`
- `nanobot/fork/cli/tui*.py`

能力：

- 会话保存走更稳的原子写路径。
- turn 中断、runner error、工具执行异常时尽量保留可恢复状态和日志。
- TUI 侧避免异常导致 consumer 或 spinner 永久卡死。

同步注意：

- upstream 可能已经有新的 atomic write/gitstore/run record 机制。合并时可以迁移实现，但不能丢掉“会话目录日志”和“出错可定位”的行为。

## 5. 当前同步冲突处理建议

当前 `sync-upstream-20260709` 已进入 merge 冲突。建议按以下顺序处理：

1. 先保留 `nanobot/fork/*` 和 `tests/fork/*`。
2. 先解决简单文档/配置：`.gitignore`、`pyproject.toml`。
3. 处理 provider：`providers/factory.py`、`providers/registry.py`、`openai_codex_provider.py`、`openai_compat_provider.py`。
4. 处理 session/log：`session/manager.py`、`utils/session_runtime_log.py` 或 upstream 替代、`agent/loop.py` 日志接入。
5. 处理 agent core：`agent/context.py`、`agent/loop.py`、`agent/runner.py`、`agent/memory.py`、`agent/tools/registry.py`。
6. 处理 command/TUI：`command/builtin.py`、`command/router.py`、`cli/commands.py`、`fork/cli/*`。
7. 最后处理 WebUI 类型和测试：`webui/src/lib/types.ts`、`tests/*`。

冲突取向：

- `nanobot/fork/*`：默认保留 ours。
- upstream 大重构 core：默认以 theirs 为主体，手工移植 fork 接入点。
- tests：优先保留 upstream 新测试，再补 fork 行为测试。
- WebUI：如果 upstream 已实现同类能力，优先统一字段/事件；不要保留两套重复类型。

## 6. 同步后必须跑的验证

最小验证：

```powershell
pytest tests/fork tests/command/test_model_command.py tests/command/test_router_dispatchable.py
pytest tests/session/test_session_runtime_log.py tests/agent/test_loop_runner_integration.py
pytest tests/cli/test_tui_textual_pilot.py tests/cli/test_commands.py
```

建议验证：

```powershell
pytest tests/agent tests/providers tests/tools tests/cli
npm run build
```

人工验证：

1. `nanobot` 无参数启动进入 agent。
2. 欢迎界面显示 model / preset / reasoningEffort / workspace。
3. `/model` 可列出和切换 preset。
4. `/new` 新建话题时，话题名不发送给 LLM。
5. 命令 palette 中带参数命令进入编辑模式，不立即提交。
6. 大粘贴显示 token，提交 payload 完整。
7. 流式输出时手动上滑不会被拉回底部；发送新消息后会跟随新回复。
8. 模型 server error 时 CLI 明显显示错误，并写入话题 runtime log。
9. 同一会话 server error 后最多自动恢复 2 次。
10. 文件编辑 diff 能在 CLI/TUI 中显示。

## 7. 后续维护要求

每次新增 fork 行为时同步更新本文档，至少写清：

- 新能力目的。
- 主要文件。
- core 接入点。
- upstream merge 时应该保留还是迁移。
- 对应测试。

提交信息使用中文。

### OpenAI Codex official app-server transport

Primary files:

- `nanobot/fork/providers/codex_app_server_provider.py`
- `nanobot/providers/factory.py`
- `tests/fork/test_codex_app_server_provider.py`

Behavior and sync notes:

- Official `codex app-server` owns OAuth, upstream requests, websocket continuation,
  and transport retries for the `openai_codex` provider.
- Nanobot tools are exposed through experimental `dynamicTools`; execution and
  workspace enforcement remain in the nanobot runner.
- All nanobot tools are children of one `nanobot` dynamic-tool namespace. This avoids
  an app-server top-level multi-tool name/index mismatch while tool execution stays
  in nanobot.
- Each outer nanobot turn uses an ephemeral app-server process so nanobot remains
  the only persisted conversation history.
- Native Codex commands and file changes are supported as first-class app-server events
  inside a `workspace-write` sandbox. The thread config explicitly disables command
  network access and restricts writable roots to the current workspace. Unexpected
  command or file-change expansion approvals are declined so the non-interactive bridge
  cannot hang or silently widen its permissions.
- Other Codex native tools, web search, user plugins, and user MCP servers are disabled
  at both process and thread scope. Their native-tool events are rejected if a Codex
  version ignores those settings.
- Before a tool result is submitted to app-server, a workspace/session/turn-scoped
  idempotency ledger atomically persists the normalized call and authoritative result.
  A transient bridge failure is recovered once with a fresh ephemeral thread; repeated
  calls are answered from the ledger inside the provider and never reach the runner.
- Ledgers use restricted files under the workspace runtime data directory, are removed
  after successful completion, and stale crash remnants expire after seven days. New
  signatures still reach the runner; ambiguous identical signatures fail safe.
- Thread-cumulative token usage is converted to per-response deltas before the shared
  runner aggregates it.
- RPC/event waits have bounded timeouts; bounded redacted stderr is retained for
  diagnostics, and an explicit provider proxy is passed only to the Codex child process.
- `NANOBOT_CODEX_BIN` may point to an explicit Codex executable.
- `NANOBOT_CODEX_APP_SERVER_RPC_TIMEOUT_S` and
  `NANOBOT_CODEX_APP_SERVER_EVENT_TIMEOUT_S` override bridge timeouts.
- When syncing upstream, preserve only the single factory import hook. After a
  Codex CLI upgrade, run the fork provider protocol tests and one real dynamic-tool
  smoke test; the protocol tests cover one-shot recovery, provider restart, durable
  replay with a changed call id, corrupt-ledger rebuild, normal repeated calls, new calls
  after recovery, stream segmentation, timeout, cancellation, concurrency,
  native-tool rejection, usage accounting, and process cleanup.
