# upstream 同步记录 2026-07-09

同步目录：`E:\learn\nanobot-sync-upstream`
同步分支：`sync-upstream-20260709`
upstream：`HKUDS/nanobot` 的 `upstream/main`

原则：以 upstream 新流程为主体，把 fork 功能按 `nanobot/fork/FORK_CHANGES.md` 的接入点重新接回去。两边功能尽量都保留。

## 当前阶段

- [x] 创建隔离 worktree
- [x] 执行 `git merge upstream/main`
- [x] 补全 `nanobot/fork/FORK_CHANGES.md`
- [x] 建立本同步记录文档
- [x] 解决简单配置冲突
- [x] 解决 provider 与 session/log 冲突
- [x] 解决 agent core 与工具注册冲突
- [x] 解决 CLI/TUI/WebUI/测试冲突
- [x] 跑验证并整理剩余风险

## 已处理决策

### D001: 保留 fork 文档和 fork 目录

`nanobot/fork/*` 和 `tests/fork/*` 是本 fork 自有能力，合并时默认保留。upstream 删除这些文件不代表我们应删除。

### D002: `.gitignore` 合并两边新增忽略项

保留 fork 的 `_sync_tmp/`，同时加入 upstream 的 `.playwright-mcp/` 和 `bridge/node_modules/`。

### D003: `nanobot/__init__.py` 同时保留 upstream lazy exports 和 fork bootstrap

采用 upstream 更完整的 `__all__`，在其前保留 `import nanobot.fork` 的 silent no-op bootstrap。


### D004: provider/facade 冲突以 upstream 新流程为主体，补回 fork 兼容点

- `azure_openai_provider.py`：合并 docstring，保留 upstream AAD 认证说明，也保留 fork 的 `CancelledError` 重抛说明；代码体内取消重抛已保留。
- `openai_codex_provider.py`：采用 upstream 的 `_codex_error_response()` 结构化错误响应，保留更完整的 retry/error metadata，满足 fork 的错误可见化需求。
- `openai_compat_provider.py`：合并 reasoning 解析，保留 fork 的 `model_extra.reasoning_content` 和 final message fallback，同时保留 upstream 的 Mistral thinking content 与 text tool-call 解析。
- `nanobot.py`：保留 upstream SDK stream cleanup、`aclose()` 和 async context manager 支持。

### D005: session/log/file-edit 冲突按上游结构合并，保留 fork 可观测性

- `session/manager.py`：保留 upstream 的 collision-resistant session 文件名、legacy 路径删除和 `fork_session_before_user_index()`，同时保留 fork 的 per-session artifact/runtime log 目录，并在删除会话时一并清理该目录。
- `session/webui_turns.py`：保留 upstream 对 `nanobot.bus.progress.build_bus_progress_callback()` 的委托；已确认 bus progress 支持 fork 的 `file_edit_events`，不再恢复旧内联实现。
- `utils/file_edit_events.py`：以 upstream 的结构化 `diff: { format, context, truncated, text }` 为主，保留 fork 的 live emit 常量、start 阶段近似统计、失败/不可读时基于参数预测 after 文本，以及 legacy `diff_text/diff_total_lines/diff_truncated` 兼容字段。
- `utils/helpers.py`：空 reasoning 不写入，避免 provider 兼容问题；但有 `thinking_blocks` 时保留 upstream 需要的空 `reasoning_content` 配套字段。
- `cron/service.py`：同时保留 upstream 的 `is_bound_cron_job` 绑定检查和 fork 的 `replace_file_with_retry` 原子替换。

### D006: 工具层冲突保留 upstream 结构化错误和 fork TUI 摘要

- `tools/base.py`：同时保留 fork 的 `summarize_result()` 扩展点和 upstream 的 `Tool.error()` 便捷方法。
- `tools/registry.py`：同时保留 upstream 的 JSON 参数解包/稳定 schema 排序和 fork 的 `_FORK_TOOL_FACTORIES` 扩展注册点。
- `tools/filesystem.py`、`tools/web.py`：错误结果按 upstream 返回 `ToolResult.error()`，并保留 fork 的 TUI 摘要方法。
- `tools/exec_session.py`：进程完成时保留 fork 的 reader drain + transport close，运行中保留 upstream 的 buffered output 等待。

### D007: hook/runner 生命周期同时保留单工具 hook 与批量 hook

- `agent/hook.py`：保留 upstream 的 `before_execute_tool/after_execute_tool/on_execute_tool_error`，同时保留 fork 的 `after_execute_tools` 批量观察点。
- `agent/runner.py`：`AgentRunSpec` 同时保留 fork `event_logger` 和 upstream 的 goal continuation/finalization 字段。
- 模型响应日志移动到 `raw_usage` 计算之后，避免引用未定义变量；工具完成日志与 upstream `tools_used` 统计并存。
- 旧 file-edit 直发逻辑不再恢复，统一通过 upstream hook 生命周期触发 `FileEditActivityHook`，避免重复发送文件编辑事件。

### D008: context/memory 以 upstream runtime context 为骨架，补回 fork 话题记忆与指令文件

- `agent/context.py`：保留 upstream 的 workspace-scoped runtime lines、MCP/CLI runtime control、recent history token cap 和模板化 identity/skills section。
- 同时补回 fork 的 `data_dir`、`topic_memory_factory`、`project_skill_roots()`、`~/.claude/CLAUDE.md`/workspace `CLAUDE.md` 加载、`learning_ctx`、todos、pending summary 和 skill-match reminder。
- 用户消息拼接采用 upstream 的 runtime context 末尾追加方式，fork 的 learning/reminder/skill hint 放在用户正文前，兼顾 prompt cache 和 fork 指令顺序。
- `agent/memory.py`：同时保留 fork 的 `MEMORY.md` mtime 缓存/原子替换/Consolidator runner 依赖，以及 upstream 的 dream prompt oversize 标记和 append lock。

### D009: AgentLoop 以 upstream 状态机和 runtime events 为主，补回 fork 学习/日志/自动恢复

- `agent/loop.py`：保留 upstream 的 workspace scope、runtime event、turn continuation、ephemeral hook、turn hook factory 和 goal continuation 流程。
- 补回 fork 的 learning context、empty-after-tools nudge retry、turn summary、per-session runtime log、dispatch exception log 和 transient server error 自动恢复。
- AgentLoop 不再直接持有 `WebuiTurnCoordinator`；WebUI 标题与 run status 走 upstream runtime events，由 CLI/gateway 装配层订阅。
- `_run_agent_loop` 中统一使用 upstream `build_agent_turn_hook()`，fork runtime log 通过 `AgentRunSpec.event_logger` 接入，避免重复 progress/file-edit 通道。
### D010: CLI commands 保留 upstream 事件流，补回 fork TUI 交互能力

- `cli/commands.py`：保留 upstream 的 typed outbound event、`StreamRenderer`、heartbeat/重启提示、WebUI/API 启动流程和 optional feature 命令。
- 同时补回 fork 的 TUI 命令面板、`/new`/`/resume`/`/continue`/`/todos`/`/commit_memory`、话题切换清理 learning state、发送后流式显示、任务错误显式展示和 stale todo 隐藏逻辑。
- 模型展示统一返回 `(model, preset_tag, reasoning_effort)`，欢迎/API/status/TUI 均显示 reasoningEffort；`status` 使用已加载配置对象，避免把 CLI 参数字符串误传入展示函数。
- 交互消费路径同时兼容 upstream typed events 与 fork 旧 metadata，降低 WebUI/TUI/测试逐步迁移时的断裂风险。
### D011: 测试冲突以 upstream 覆盖为主体，补回 fork 回归点

- `webui/src/lib/types.ts` 与 `tests/utils/test_file_edit_events.py`：确认 file-edit 事件使用 upstream 结构化 `diff`，并保留 fork legacy `diff_text/diff_total_lines/diff_truncated` 字段。
- agent/context/subagent 测试：以 upstream runtime context、history、tool toggle、runner hook 测试为主体，补回 fork skill reminder、`.claude/skills` 和错误 metadata 兼容断言。
- CLI 测试：以 upstream gateway/webui/provider/trigger 覆盖为主体，追加 fork TUI 命令面板、话题缓存大小、stale todo、root 默认启动和 Claude AI OAuth provider 路由回归测试。
- Codex provider 测试：保留 upstream 结构化 timeout/http/proxy/retry/diagnostic log 测试，补一条 fork 空异常类型可见化断言。
### D012: 验证阶段修正真实合并拼接和兼容缺口

- 修复 `agent/memory.py`、`agent/runner.py`、`agent/loop.py` 中冲突合并导致的语法拼接问题。
- `bus/outbound_events.py` 在 typed `ProgressEvent` 外继续回填 `_progress/_file_edit_events/_tool_events` 等 legacy metadata，保证 fork TUI/旧插件在迁移期不丢事件。
- `cli/commands.py` 补回 `safe_filename`，并让 `_print_enable_options()` 同时兼容 fork 旧两参输出和 upstream 四参 feature table。
- `utils/file_edit_events.py` 保留 start 阶段近似增删预测，但 end 阶段遇到 oversized/binary/unreadable 时不再用预测值伪造计数。
- 目标测试组已通过：`247 passed, 10 skipped`；命令为 `PYTHONPATH=E:\learn\nanobot-sync-upstream pytest tests/utils/test_file_edit_events.py tests/agent/test_loop_progress.py tests/agent/test_loop_runner_integration.py tests/agent/test_loop_consolidation_tokens.py tests/agent/test_context_prompt_cache.py tests/agent/test_subagent.py tests/cli/test_commands.py tests/providers/test_openai_codex_provider.py tests/test_openai_api.py tests/tools/test_mcp_probe.py`。
## 待回顾点

- [x] upstream 已新增 `nanobot/agent/turn_hooks.py`，已保留单工具 hook 和 fork `after_execute_tools` 批量 hook。
- [x] upstream 已新增/重构 runtime events；file edit 已走 hook，session runtime log 通过 runner event_logger 和 AgentLoop 状态机接入。
- [x] upstream 删除/替换部分 heartbeat/cron 结构，需要确认 fork `/continue`、topic session、runtime log 不被破坏：CLI/agent/gateway 相关目标测试已覆盖。
- [x] upstream WebUI 已新增 file edit diff progress view，需要与 fork CLI/TUI 的 file edit diff 字段对齐：核心事件、WebUI 类型和目标测试已对齐 `diff` 对象 + legacy `diff_text` 字段。
- [ ] `FORK_CHANGES.md` 是 UTF-8 文档；PowerShell 默认编码显示乱码时使用 `Get-Content -Encoding UTF8`。