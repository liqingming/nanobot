# Fork 改动总账 / 上游同步防丢清单

> **用途**:这是本 fork(`liqingming/nanobot`)相对上游(`HKUDS/nanobot`)的全部改动账本。
> 下次从上游 `merge` 时,**逐条对照本清单**,确认每项 fork 功能在合并后仍然存在、挂载点未被上游覆盖。
>
> **基线(merge-base)**:`3d3ef586`(对任一文件 `git diff 3d3ef586 HEAD -- <file>` 即得 fork 对该文件的全部原创改动)。
> **生成时间**:2026-05-27 北京时间。**末次 fork commit**:`200ec157`。
> **维护约定**:每次给 fork 加功能 / 改 core patch / 调默认值,顺手更新本文件对应小节。

---

## 0. 同步模式与当前仓库状态(先读这条)

**本 fork 的同步模式 = 上游全量 + fork 增量。上游与 fork 两边功能都要保留,从不裁剪上游。**

> 早期 commit(阶段 0-2,`74e61716`→`f434c5e8`)曾探索性删除过上游大量模块(signal/msteams、webui、部分 provider、tests/docs),但那只是早期探索,**并非 fork 的意图**。随后的 merge `97d268b4` 已把它们全部恢复——这**符合预期,不是问题**,无需处理。

**结论**:当前仓库 ≈ **完整上游代码(3d3ef586 时点)+ fork 新增功能**,上游能力一个不少。
- 因此本清单的重点是第 1-3 节:fork **新增/修改**了什么,确保下次 merge 时这些增量不被上游覆盖丢失。
- 第 4 节仅作历史备注,不代表任何待办。

---

## 1. 挂载机制总览(最易在 merge 中丢失的"接缝")

fork 通过 4 类接缝接入 core。**这些接缝一旦被上游覆盖,对应 fork 功能会静默消失**,且往往不报错,最难发现:

| # | 接缝 | 位置 | 丢失后果 |
|---|------|------|----------|
| S1 | bootstrap import | `nanobot/__init__.py` 一行 `import nanobot.fork`(`try/except ImportError` 包裹) | 所有 fork 自注册工具全部消失 |
| S2 | fork 工具自注册表 | `agent/tools/registry.py`:`_FORK_TOOL_FACTORIES` + `register_fork_tool()` + `iter_fork_tool_factories()`;`fork/__init__.py:22-27` import 四个工具模块触发自注册 | todo/ask_user/search_history/load_skill 工具消失 |
| S3 | core 薄 patch import | `loop.py`(learning/topic_memory/tool_hints)、`context.py`(format_todos)、`commands.py`(create_tui/format_todos/deliver_reply/oauth)、`nanobot.py`(oauth provider) | 对应功能断链 |
| S4 | config 开关 | `config/schema.py`:`enable_learning`/`tui_backend` 等 | 学习/检索/TUI 后端回退 |

---

## 2. Fork 新增模块(`nanobot/fork/` 目录)

> 这些文件上游完全没有,merge 不会冲突,但要确认 S1/S2/S3 接缝仍把它们接住。

### 2.1 TUI 双后端体系(`fork/cli/`)
- **`tui_base.py`** — `TUIBase` 抽象基类,`commands.py` 只依赖此接口。`flush_accumulator`/`set_todos`/`add_tool_result`/`show_question_popup` 为默认 no-op,仅 Textual 后端覆盖。
- **`tui_factory.py`** — `create_tui()` 运行时选后端。**默认 `textual`**,环境变量 `NANOBOT_TUI` 覆盖;后端模块延迟 import。被 `commands.py` 调用。
- **`tui_keys.py`** — 两后端共享的按键决策**纯函数**(Enter/Up-Down/Tab → 返回 `EnterAction`/`PopupAction` dataclass,无副作用)。new_topic 模式优先级最高。
- **`tui.py`**(PromptTUI,prompt_toolkit,~1137 行) — 全屏分栏 TUI,手动管理滚动与 ANSI 行切片。
  - 滚动靠 `_scroll_offset`(距底行数);`_get_output_text`(`:593`)join 全部行后按终端高度切片。
  - reasoning 只渲染尾部 N 行(`_reasoning_tail_lines=5`,`:498`);`markup=False` 防 rich 把 `[1]` 当标记崩溃。
  - 别名 `SplitTUI`。
- **`tui_textual.py`**(TextualTUI,~2117 行) — Textual 后端(默认),RichLog 原生滚动 + 鼠标拖选复制 + 工具 trace 折叠/展开 + 流式 spinner。
  - **跨上下文调用约定**(`_safe_call`,`:583`):非 Textual 线程调用建 Timer 的方法须经 `call_later`,否则 shutdown 报 `LookupError`。
  - 流式/spinner **行级原位重写**(改写 `_tool_placeholder_line` 单行 + `truncate_to`),不重写整块。
  - idle thinking 调度(`:1731`):工具完成后 500ms 无 delta 才显示"思考中"。
  - Windows 剪贴板用 ctypes 直调 user32/kernel32(`:139`),`restype=c_void_p` 防 64 位句柄截断。
  - 中间内容块超 `_COLLAPSE_THRESHOLD=6` 行折叠为可点击摘要。

### 2.2 学习 / 记忆 / 话题(`fork/agent/`)
- **`learning.py`** — 回合级学习上下文:下条用户消息前注入上一回合结构化元数据(工具序列、错误数、压力、重复模式)。`PatternStore` 跨会话工具序列计数持久化到 `<data_dir>/memory/patterns.json`;`detect_user_delta` 关键词分类 correction/new_topic/continuation;`TurnSummary.is_significant` 控制是否注入。被 `loop.py` import。
- **`topic_memory.py`** — `TopicMemoryFactory`:每个 `session_key` 独立 MemoryStore,根目录 `<data_dir>/memory/topics/<safe_key>/`;复用上游 `MemoryStore.memory_dir_override` 钩子白嫖原子写/Dream/迁移;SOUL.md/USER.md 仍共享。被 `loop.py` 注入 `context.py`。

### 2.3 Fork 工具(`fork/agent/tools/`,经 S2 自注册)
- **`todo.py`** — `TodoWriteTool` + `format_todos`/`format_todo_diff`。会话级多步待办,每次全量覆盖,经 bus 实时推 diff 给 TUI。强不变量:最多一个 in_progress、completed 不可回退、≤50 项。`format_todos` 被 `commands.py`/`context.py` import。
- **`ask_user.py`** — `AskUserTool` + `deliver_reply`。交互式多选提问(仅 CLI channel);bus `OutboundMessage` metadata(`_ask_user_id`)+ `asyncio.Future` 异步等回复,300s 超时。`deliver_reply` 被 `commands.py` 调用。
- **`memory_search.py`** — `SearchHistoryTool`。对 HISTORY.md 的 BM25 检索 + 中英双语同义词扩展,合并 global + 所有 topics 历史;CJK 按字符拆分;**仅 `enable_learning` 为真才注册**。依赖外部 `rank_bm25` 包。
- **`skill.py`** — `LoadSkillTool`。按名加载 `<root>/<name>/SKILL.md`,trace 显示 `load-skill("name")`;加载失败列出可用 skill 助 LLM 自纠。

### 2.4 其它
- **`providers/claude_ai_oauth_provider.py`** — `ClaudeAIOAuthProvider`:用 claude.ai 订阅 OAuth Bearer token(非 API key)。token 优先 `~/.nanobot/credentials/claude_ai.json`,回退 Claude Code 库 `~/.claude/.credentials.json`;绕过 `AnthropicProvider.__init__` 直接用 `auth_token=` 建 client。被 `commands.py`/`nanobot.py` import。
- **`utils/tool_hints.py`** — `format_tool_hint`/`relativize_path`:在上游基础上加 workspace 相对路径显示(`write_file("./src/main.py")`)。被 `loop.py` 注入 `AgentProgressHook(formatter=...)`,且 `tui_textual.py` 历史回放间接用。
- **`extensions/__init__.py`、`hooks/__init__.py`** — 仅 docstring 占位,无逻辑,保留目录即可。

---

## 3. Core 文件 Fork Patch(按文件,附挂载关系)

> 这些是上游已有、fork 改过的文件。**merge 冲突高发区**,逐个核对。

### 3.1 Agent 主链路
- **`agent/loop.py`**(+309,改动最集中) — 挂载点之王。
  - 新增 `data_dir`(与 `workspace` 分离:workspace 给 agent,data_dir 存 nanobot 元数据),`ContextBuilder`/`SessionManager` 改挂 data_dir。
  - 挂载 learning:`enable_learning` 开关、`PatternStore`、per-session 三个状态字典(`_last_turn_summary`/`_prev_consolidated`/`_last_user_input`)、turn 结束 `_capture_turn_summary`、下轮 `_build_learning_ctx` 注入。
  - **nudge 重试**:LLM 在 tool 结果后吐空 assistant(`_empty_after_tools`)时,注入追问 user 消息重跑一次,二次结果缝合回消息列表。
  - fork 工具自注册:`iter_fork_tool_factories()` 遍历注入。
  - 中文/双语 `max_iterations_message`(提示 `/continue`);`TopicMemoryFactory` 在 learning 开启时挂上。
  - **同步注意**:`_state_run` 被大幅重写(包了 `_invoke` 闭包 + nudge)。fork 注释明确依赖 upstream `_run_agent_loop` 的 **5-tuple 返回**;`_invoke` 在 await 后立即快照 `_last_usage`/`_last_tool_events`(防并发污染,commit 5003cd59)。`loop.py:1651` runtime-context 剥离从 `startswith` 改 `in`(因 prefix 顺序变了)。上游若改返回结构必须重新对齐解包。
- **`agent/context.py`**(+312) — 话题感知记忆注入。
  - `__init__` 加 `data_dir` + `topic_memory_factory`;`build_system_prompt` 拼接 global memory + topic memory + `todos` 段(`format_todos`)+ `LEARNING_RULES.md`。
  - `build_messages` 把 learning_ctx / runtime_ctx / pending-summary reminder / skill 软提示**统一拼成 prefix 放到 user content 之前**(上游是拼在之后)。
  - `_get_identity` 整段从模板改成 fork 内联 f-string。新增 `_build_skill_match_reminder`(关键词重叠推荐 load_skill)。
  - **同步注意**:prefix **前置**改变了 prompt 前缀,必须与 `loop.py` 的剥离逻辑严格配对;`_get_identity` 已脱离上游 `identity.md`/`platform_policy.md` 模板,上游改身份模板 fork 不自动受益。
- **`agent/runner.py`**(+12) — 仅 `runner.py:357` 一行行为变更:tool 执行后插 `await hook.after_execute_tools(context)`(learning 观察钩子点);其余 docstring。
- **`agent/hook.py`**(+12) — `AgentHook` + `CompositeHook` 新增 `after_execute_tools` 钩子(基类空实现,Composite 遍历+异常隔离)。纯扩展。
- **`agent/memory.py`**(+15) — `MemoryStore.__init__` 加 `memory_dir_override`(per-topic 子目录的关键 hook);SOUL/USER 仍挂 workspace(话题间共享)。
- **`agent/skills.py`**(+47) — `build_skills_summary` 输出改 `<skills><skill>` XML,**刻意省略 `<location>`**(fork 用 `load_skill(name=)` 按名加载,暴露路径浪费 token + 诱导 read_file 误用)。**同步注意**:上游若改 skills_section 模板格式会冲突。
- **`agent/subagent.py`** — 仅顶部 docstring 标注,无代码变更。
- **`agent/progress_hook.py`**(+7) — 新增 `tool_hint_formatter` 注入点,默认回退上游,fork 注入 workspace 相对路径格式化器。

### 3.2 Session / 入口
- **`session/manager.py`**(+34) — `Session` 加 `todos` 与 `pending_consolidation_summary` 字段(加载/保存/clear 三处同步)。`pending_consolidation_summary` 是 **cache-safe consolidation** 核心:摘要先缓冲,下轮以 system-reminder 注入而非重渲染 system prompt。**同步注意**:metadata_line JSON schema 增了两个 key。
- **`nanobot/__init__.py`**(+9) — `import nanobot.fork` 触发自注册(S1),`except ImportError` 静默 no-op 保持裸上游可用。
- **`nanobot/nanobot.py`** — SDK facade(`Nanobot.from_config` / `run`)。**注**:原有一个放错地方的死代码 `_make_provider`(无人调用且缺 return),已删除;provider 构造统一回 `factory._make_provider_core`,其 `claude_ai_oauth` 分支已补到 factory(见 3.3)。
- **`bus/events.py`** — 无 fork 改动(diff 为空)。

### 3.3 Providers
- **`providers/base.py`**(+51) — `max_tokens` 自适应降级三件套(`_is_max_tokens_error`/`_extract_max_tokens_upper_bound`/`_downgrade_max_tokens`):API 报 max_tokens 超限时从错误文本正则解析上界,永久下调本会话默认值并重试一次。挂在 `_safe_chat` 非 transient 分支,排在去图重试之前;用 `dataclasses.replace` 改 frozen `GenerationSettings`;兜底 8192。
- **`providers/anthropic_provider.py`**(+62) — (1) `_strip_prefix` 接受 `claude-ai/`/`claude_ai/`(OAuth provider 复用本类);(2) `redacted_thinking` 块双向往返;(3) temperature-deprecated 自动去 temperature 重试;(4) **`except asyncio.CancelledError: raise`** 四处(TUI 取消即时生效的前提,必须排在 `except Exception` 前)。
- **`providers/openai_compat_provider.py`**(+76) — DeepSeek thinking 历史回填(缺 `reasoning_content` 的 assistant 补空格占位)/ 反向删空 `reasoning_content`(Anthropic 兼容端点);reasoning 解析三处加 `model_extra` 兜底来源;CancelledError 重抛。**与 `utils/helpers.py` 那条空 reasoning 治理是一对**。
- **`providers/azure_openai_provider.py`**(+9) — `chat`/`chat_stream` 加 CancelledError 重抛。
- **`providers/registry.py`**(+15) — 新增 `claude_ai` ProviderSpec(`is_oauth=True`,backend `claude_ai_oauth`)。
- **`providers/factory.py`**(+9) — `_make_provider_core` 新增 `claude_ai_oauth` 分支,构造 `ClaudeAIOAuthProvider`。这是 CLI+SDK+fallback **唯一**的 provider 工厂,缺此分支会让 claude.ai OAuth backend 静默退化成 `OpenAICompatProvider`。**同步注意**:上游文件,merge 时此分支易被覆盖丢失(已加回归测试 `tests/cli/test_commands.py::test_make_provider_uses_claude_ai_oauth_backend` 守卫)。
- 无改动:`openai_codex_provider.py`。

### 3.4 Tools
- **`agent/tools/shell.py`**(+85) — (1) `_decode_console_bytes`:子进程输出先试 UTF-8,失败按 OEM 代码页/cp936/locale 逐个降级(中文 Windows 控制台 GBK 乱码);(2) 子进程环境注入 `PYTHONIOENCODING=utf-8`+`PYTHONUTF8=1`(Win+POSIX 两分支),防 LLM 代码 `print("📊")` 在 cp936 下崩溃;(3) `summarize_result`(tool-trace 摘要)。UTF-8 是快路径,正常零开销。
- **`agent/tools/registry.py`**(+29) — fork 工具注册机制(S2),见第 1 节。
- **`agent/tools/base.py`**(+10) — `Tool` 基类加 `summarize_result(args,result)->str` 默认 `""`,各工具按需 override,约定绝不抛异常。
- **`agent/tools/summaries.py`**(新增 +81) — tool-trace 摘要共享 helper(`line_count`/`truncate`/`extract_error_summary`/`summarize_tool_result`,best-effort 吞异常)。**放 core 而非 fork**:太多上游工具消费它,移 fork 会倒置依赖(docstring 说明)。
- **`agent/tools/filesystem.py`**(+59) — Read/Write/Edit/List 四工具加 `summarize_result`;`WriteFileTool` description 要求 LLM 结尾附绝对路径。
- **`agent/tools/web.py`**(+53) — WebSearch/WebFetch 加 `summarize_result`。
- 无改动:`tools/mcp.py`、`tools/spawn.py`。

### 3.5 Config / CLI
- **`config/schema.py`**(+34) — `AgentDefaults` 新增 `enable_learning`(默认 True)、`tui_backend`(默认 textual)、`promote_pending_on_restart`(默认 False)、`nudge_after_empty_tools_message`;`max_tool_iterations` 默认 **200→1000**;`ProvidersConfig` 加 `claude_ai`(exclude)。
- **`config/paths.py`**(+23) — `get_workspace_cache_dir(workspace)`:非默认 workspace 元数据存 `~/.nanobot/caches/<safe>_<sha1[:8]>/` 保持项目目录干净。
- **`cli/commands.py`**(+633,**冲突最高**) —
  - 新增 `cache` 子命令组:`cache migrate <old> <new>`、`cache list`。
  - 新增 `_login_claude_ai`(优先静默导入 Claude Code 凭证)。
  - **交互模式整体重写为 split-pane TUI**(原单行循环删除,改 `create_tui` + `run_async`)。新增交互:
    - 斜杠命令:`/new [name]`、`/resume [name]`(无参弹 picker)、`/todos [clear]`、`/continue`(恢复因 iteration 上限中断的任务)、`/commit_memory [show]`、`/exit`。
    - 启动话题选择弹窗 `_startup_picker`(默认 `topic_YYYYMMDD_HHMMSS`)。
    - **outbound consumer 重写** `_consume_outbound`:按 metadata 分发到 TUI 各方法,处理 `_ask_user` 弹窗回填、`_system_message`(todo diff)。**reasoning 拦截**:`_reasoning_delta`/`_reasoning_end` 在 `add_progress` 前显式 `pass` 丢弃(思考链显示已屏蔽,TUI 侧方法保留以便恢复);整个 consumer 包 `try/except` 防单条异常冻结 UI。
    - **取消机制** `_cancel_current`:cancel `agent_loop._active_tasks[session_key]`,配合 provider CancelledError 重抛。
    - **pre-submit** `_send_message` 里 `asyncio.sleep(0.015)`(超过 prompt_toolkit max_render_postpone,保证 thinking 动画先到屏)。
    - 消息排队 `pending_queue`;非默认 workspace data_dir;`SafeFileHistory`(清代理字符,修 Windows emoji 输入崩溃 #2846);响应 header 时间戳。
  - **同步注意**:上游改交互主循环/outbound 分发/登录注册几乎必冲突,需逐块手工合并。
- **`cli/stream.py`**(+5) — 流式 header 加时间戳。
- **`utils/helpers.py`**(+4) — `build_assistant_message` 仅在 `reasoning_content` 真值时写字段(与 openai_compat 空 reasoning 修复成对)。

---

## 4. 早期裁剪探索(历史备注 — 已全部恢复,无需关注)

> 这些是早期探索遗留,**非 fork 意图**。阶段 0-2 曾删除,merge `97d268b4` 已全部恢复,当前代码完整包含。
> 列在此仅为追溯历史,**不是待办,也不要重新删除**。

早期曾删除约 13.4 万行(试图收敛为 CLI/TUI + 核心 agent),涉及:渠道(signal/msteams/websocket)、provider(bedrock/github_copilot/image_generation 等)、部分 tools、整个 WebUI、API server、部分 skills/templates/docs。

**同步原则**:始终保留上游全量 + fork 增量,不再做任何裁剪。

---

## 5. ⚡ 性能 / 响应速度审计

> 基于三个 subagent 对当前 HEAD 的**静态阅读**,均**未实测 profiling**。严重度按"是否在请求/渲染热路径 + 开销是否随规模(会话长度/skill 数/历史大小)增长"定性评估。

### 5.1 Fork 主动做的优化(merge 时务必连注释一起保留)
| 位置 | 优化 |
|------|------|
| `context.py` / `session/manager.py` | **cache-safe consolidation**:摘要缓冲到 `pending_consolidation_summary`,下轮以 system-reminder 注入而非重渲染 system prompt → 保护 Anthropic 前缀缓存命中(commit 64db5574/5003cd59) |
| `fork/cli/tui.py:498` | reasoning 只渲染尾部 N 行,避免每 0.1s tick 重渲染长 buffer |
| `fork/cli/tui.py:1081` | stream 渲染按 `len(stream_buf)` 做缓存键,长度不变跳过 rich 重渲染 |
| `fork/cli/tui.py:1119` | `run_async` 预加载 FileHistory,避免首次 Enter 卡顿(commit 68947c08) |
| `fork/cli/tui_textual.py:1789` | `stream_delta` 仅当用户在底部才 truncate+rewrite,滚上去时跳过 |
| `fork/cli/tui_textual.py:1681` | 中间内容块超 6 行折叠,减少渲染行数 |
| `commands.py:_send_message` | `asyncio.sleep(0.015)` 用 15ms 延迟换 thinking 动画先到屏(权衡,非问题) |
| `shell.py` | `_decode_console_bytes` UTF-8 快路径,正常输出零额外解码开销 |
| learning | `_invoke` await 后立即快照 usage/events,防并发污染(正确性,无额外开销) |

### 5.2 性能开销点 — 处理状态(2026-05-27 本轮优化)

> 责任归属分两层:**纯 fork**(代码本身是 fork 新增/改的)与 **fork 触发 + 上游底层**
> (fork 提高了调用频率,但底层慢实现是上游原有)。所有 ✅ 项均已跑测试验证。
> ⚠️ 已优化项的缓存失效机制(mtime / `len` / 写时失效)和 `Fork(perf)` 注释,merge 时务必保留。

| 严重度 | 开销点 | 责任归属 | 状态 |
|--------|--------|----------|------|
| 高 | P1 skill_match 每轮 2N 次读盘+YAML(`skills.py` `get_skill_metadata`) | fork 触发(每轮 skill-match) + 上游底层(无缓存) | ✅ `get_skill_metadata` 加 mtime 缓存(`_resolve_skill_path`/`_parse_skill_metadata`/`_meta_cache`) |
| 高 | H1 `tui.py:_get_output_text` 每帧 join 全历史 | 纯 fork(TUI) | ✅ 按 `len(_output_lines)` 缓存 join 结果,append/clear 自动失效 |
| 高 | H3 `memory_search` 合并 BM25 索引每次重建 | 纯 fork | ✅ 合并索引按"所有 HISTORY 文件 (path,mtime)"签名缓存 + 回归测试 `tests/fork/test_memory_search.py` |
| 高 | H2 `tui.py:_animate_thinking` 0.1s tick 全量渲染 | 纯 fork(TUI) | ⏸ 评估后暂不改:spinner 动画本质,降帧→卡顿、改渲染生命周期高风险,收益<风险 |
| 中 | M1 `PatternStore` 每 turn 同步写盘 | 纯 fork | ✅ 30s 节流 + `flush()`;内存 `_counts` 始终精确,崩溃最多丢节流窗口(非关键统计) |
| 中 | P3 双 `read_memory` 每轮读盘(`memory.py`) | fork 触发(topic 那次) + 上游底层(无缓存) | ✅ `read_memory` 加 mtime 缓存 + `write_memory` 写时失效 |
| 中 | M3/M4 `tui_textual` 折叠/渲染 O(n) 全行扫描 | 纯 fork(TUI) | ⏸ 评估后暂不改:仅折叠/滚动**交互时**触发(非每帧),边际收益、中风险 |
| 中 | nudge 重试 / provider max_tokens·temperature 重试 | 纯 fork | ❌ 不改:错误恢复/功能逻辑,非性能 bug,改了会改变行为 |
| 低 | popup 每键重算、启动单次读盘、`summarize_result` 单次解析、consumer 1s 轮询、DeepSeek reasoning 回填 | 混合 | ❌ 不改:合理设计,开销可忽略 |

**已优化项的关键不变量**(merge 时检查):
- `skills.py`/`memory.py`/`memory_search.py` 的 mtime 缓存依赖"文件改动→mtime 变";若上游引入原地写不改 mtime 的逻辑需重新评估。
- `tui.py` H1 缓存依赖"`_output_lines` 只 append / clear,不原地改元素"(已在注释声明)。
- `PatternStore` 提供 `flush()` 但**尚未挂到 shutdown**;如需零丢失,在 loop 关闭处调用。

### 5.3 已修复的阻塞问题(正面)
- outbound consumer `except Exception: log+continue`(`commands.py`):修复"单条消息异常杀死 consumer → UI 永久冻结在 spinner"。
- 取消路径 `_cancel_current` + provider CancelledError 重抛:长任务可即时中断,不让用户干等。

### 5.4 审计局限
未做实际 profiling;高严重度项的真实影响取决于运行期规模(skill 数 / 会话长度 / HISTORY 大小),Windows + 杀软可能放大文件读开销。`Consolidator`/`auto_compact` 内部未纳入本轮审计。

---

## 6. 已知问题 / 待办

| 问题 | 位置 | 说明 |
|------|------|------|
| ✅ 已修复:`_make_provider` 死代码 + claude_ai_oauth 接入 | `nanobot.py` / `factory.py` | 原 `nanobot.py:_make_provider` 是无人调用、缺 return 的死代码;真正隐患是 `factory._make_provider_core` 缺 `claude_ai_oauth` 分支(claude.ai OAuth 静默退化成 OpenAICompatProvider)。已删死代码 + factory 补分支 + 加回归测试。 |
| ✅ 已优化(本轮):后端 + TUI 热路径 | 见 5.2 | P1 skills 缓存 / P3 memory 缓存 / M1 PatternStore 节流 / H3 search 索引缓存 / H1 TUI 历史 join 缓存,均带测试验证。H2、M3/M4(TUI 渲染)评估后暂不改(理由见 5.2)。 |
| ✅ 已清理:fork 目录 pre-existing lint(29→0) | `fork/` 全目录 + `tests/cli/test_commands.py` | E402:自注册 import 加 `# noqa: E402`(保留底部自注册模式);`tui.py` 重构把 `_FilteredFileHistory`/`_HISTORY_SKIP` 移到所有 import 之后。I001:`ruff --fix` 自动排序。N806:Win32 常量(`CF_UNICODETEXT`/`GMEM_MOVEABLE`)、阈值(`_COLLAPSE_THRESHOLD`)、标签(`_RUNTIME_TAG`/`MAX`)加 `# noqa: N806`(常量语义,大写有意)。F401:删未用 `subprocess`。`ruff check nanobot/fork/` 现 0 errors。 |

---

## 7. Merge 上游时的操作 Checklist

1. **同步模式固定为"上游全量 + fork 增量"**——保留上游所有功能,只确保 fork 增量不丢,不做任何裁剪。
2. merge 后**逐项核对第 1 节 4 个接缝(S1-S4)**——它们被覆盖时不报错,最易静默失效:
   - `grep "import nanobot.fork" nanobot/__init__.py`(S1)
   - `grep "iter_fork_tool_factories\|register_fork_tool" nanobot/agent/tools/registry.py`(S2)
   - 确认 `loop.py`/`context.py`/`commands.py`/`nanobot.py` 的 fork import 仍在(S3)
   - 确认 `config/schema.py` 的 `enable_learning`/`tui_backend` 仍在(S4)
3. 核对第 3 节"同步注意"标记的高冲突文件:`loop.py`(5-tuple 返回 + 剥离逻辑)、`context.py`(prefix 前置)、`commands.py`(交互主循环)、`skills.py`(XML 输出)。
4. 跑测试:`pytest tests/fork/ tests/cli/ tests/agent/ tests/providers/ -q`(覆盖 fork 工具、TUI、本轮性能优化涉及的 skills/memory/factory)。重点确认 fork 相关 + claude_ai backend(`tests/cli/test_commands.py::test_make_provider_uses_claude_ai_oauth_backend`)、H3 缓存(`tests/fork/test_memory_search.py`)。
5. 启动 `nanobot agent` 实测:话题切换、todo、ask_user 弹窗、流式渲染、取消(Ctrl+C)是否正常。
6. 更新本文件:记录本次 merge 的上游基线 commit,刷新"基线/末次 commit"。
