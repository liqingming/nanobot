# nanobot 执行流程图

## 一、总览：一条消息的完整旅程

```mermaid
sequenceDiagram
    participant 👤 as 👤 用户
    participant TUI as TUI (prompt_toolkit)
    participant CMD as commands.py
    participant Bus as MessageBus
    participant Loop as AgentLoop<br/>loop.py
    participant Ctx as ContextBuilder<br/>context.py
    participant Runner as Runner<br/>runner.py
    participant Hook as AgentHook<br/>hook.py
    participant LLM as LLM Provider
    participant Tools as ToolRegistry<br/>tools.py
    participant Sub as SubagentManager<br/>subagent.py
    participant Mem as Memory<br/>memory.py
    participant Skills as SkillsLoader<br/>skills.py
    participant Sess as SessionManager

    Note over 👤,Sess: ═══ 阶段 0: 启动 ═══

    TUI->>Loop: AgentLoop(config, session)
    Loop->>Ctx: ContextBuilder(skills_loader, workspace)
    Loop->>Skills: 双目录扫描（内置 + 自定义）
    Loop->>Mem: MemoryStore(session_dir) → 加载 MEMORY.md
    Loop->>Sess: 加载历史会话消息
    Loop-->>TUI: 欢迎界面 + 话题选择弹窗

    Note over 👤,Sess: ═══ 阶段 1: 用户输入 → 动画启动 ═══

    👤->>TUI: 输入消息 + Enter
    TUI->>CMD: _pre_submit() 同步触发
    CMD->>TUI: stream_start() → thinking spinner 🎬
    TUI->>CMD: _on_submit() 异步调度
    CMD->>Bus: publish_inbound(msg, wants_stream=True)

    Note over 👤,Sess: ═══ 阶段 2: AgentLoop 消息调度 ═══

    Bus->>Loop: _dispatch(msg)
    Loop->>Loop: _process_message(msg)
    
    Note over 👤,Sess: ═══ 阶段 3: System Prompt 五层组装 ═══

    Loop->>Ctx: build_system_prompt()
    Note over Ctx: Layer 1: 身份声明 (SOUL.md 运行时)<br/>Layer 2: Bootstrap (AGENTS/SOUL/USER/TOOLS.md)<br/>Layer 3: MEMORY.md 长期记忆<br/>Layer 4: always skills 完整注入<br/>Layer 5: skills summary XML 清单

    Loop->>Ctx: build_messages(history, user_msg)
    Note over Ctx: merged user message + 工具结果拼接

    Note over 👤,Sess: ═══ 阶段 4: Runner 调用 LLM ═══

    Loop->>Runner: run_stream(system, messages, tools, hooks)
    Runner->>Hook: on_run_start(ctx)
    Runner->>LLM: send_message(...)

    loop 流式响应
        LLM-->>Runner: stream delta
        Runner->>Hook: on_stream_delta(delta)
        Hook-->>Bus: _stream_delta → TUI 实时渲染
    end

    Note over 👤,Sess: ═══ 阶段 5: 工具调用循环 ═══

    alt LLM 返回 tool_use
        LLM-->>Runner: tool_use blocks
        Runner->>Hook: on_stream_end(resuming=True)
        Hook-->>Bus: _stream_end → TUI: tool_phase_start()

        loop 每个工具
            Runner->>Hook: on_tool_start(name, params)
            Runner->>Tools: execute(name, params)

            alt spawn 工具
                Tools->>Sub: SubagentManager.spawn(task)
                Note over Sub: 独立 AgentLoop<br/>工具子集（无 message/spawn）<br/>max_iterations=15<br/>asyncio.create_task 后台运行
                Sub-->>Tools: task_id
            else 普通工具
                Tools-->>Runner: tool_result
            end

            Runner->>Hook: on_tool_end(name, result)
        end

        Runner->>LLM: send_message(带工具结果继续)
    end

    Note over 👤,Sess: ═══ 阶段 6: 最终响应 ═══

    LLM-->>Runner: 最终文本
    Runner->>Hook: on_stream_end(resuming=False)
    Hook-->>Bus: _streamed → TUI: pop_stream() + add_response()
    Runner->>Hook: on_run_end(ctx)
    Runner-->>Loop: 完整响应

    Note over 👤,Sess: ═══ 阶段 7: 持久化与记忆压缩 ═══

    Loop->>Sess: _save_turn(user_msg, assistant_msg)
    Note over Sess: 截断/剥离/去重

    Loop->>Mem: maybe_consolidate_by_tokens()
    Note over Mem: budget = ctx_window - max_completion - 1024<br/>pick_consolidation_boundary() user-turn 分割<br/>compress() → LLM 调 save_memory<br/>→ MEMORY.md + HISTORY.md 双写<br/>last_consolidated 递增

    Bus-->>TUI: _turn_complete() → 更新上下文用量 %
    Note over TUI: ctx: 2% ▌...
```

---

## 二、ContextBuilder 五层组装

```mermaid
graph TD
    subgraph "ContextBuilder.build_system_prompt()"
        direction TB
        
        L0["📥 输入<br/>SOUL.md · AGENTS.md · TOOLS.md · USER.md<br/>MEMORY.md · skills_loader"]
        
        L1["<b>Layer 1: 运行时上下文</b><br/>You are nanobot...<br/>当前时间 / 平台 / 工作区<br/>_RUNTIME_CONTEXT_TAG 防注入"]
        
        L2["<b>Layer 2: Bootstrap 模板</b><br/>AGENTS.md → 全局指令<br/>SOUL.md → 人格与价值观<br/>TOOLS.md → 工具使用说明<br/>USER.md → 用户画像"]
        
        L3["<b>Layer 3: 长期记忆</b><br/>MEMORY.md → 重要事实<br/>用户偏好 · 项目上下文 · 关系"]
        
        L4["<b>Layer 4: Always Skills</b><br/>cron/memory 等高频技能<br/>完整 SKILL.md 内容直接注入"]
        
        L5["<b>Layer 5: Skills 清单</b><br/>build_skills_summary() → XML<br/>name · available · requires · location<br/>按需 read_file 懒加载"]
        
        OUT["📦 完整 System Prompt<br/>→ Runner → LLM"]
    end

    L0 --> L1 --> L2 --> L3 --> L4 --> L5 --> OUT
```

---

## 三、AgentLoop 核心管线

```mermaid
graph TD
    subgraph "AgentLoop.run() — 消息调度"
        MSG["📨 消息到达<br/>Bus → _dispatch()"]
        LOCK{"_active_tasks<br/>并发锁<br/>WeakValueDictionary"}
        LOCK -->|"同一 session<br/>排队等待"| WAIT["await 前一个 task"]
        LOCK -->|"新 session"| START["创建 asyncio.Task"]
        WAIT --> START
        
        START --> PROCESS["_process_message(msg)"]
        
        PROCESS --> BUILD_PROMPT["ContextBuilder<br/>build_system_prompt() + build_messages()"]
        BUILD_PROMPT --> RUNNER["Runner.run_stream()<br/>循环 LLM 调用 + 工具执行"]
        RUNNER --> SAVE["Session._save_turn()<br/>截断 · 剥离 · 去重"]
        SAVE --> CONSOL["Memory.maybe_consolidate_by_tokens()<br/>条件触发压缩"]
        CONSOL --> DONE["✅ 处理完成"]
    end
```

---

## 四、工具系统全路径

```mermaid
graph TD
    CALL["LLM 返回 tool_use block"]
    
    CALL --> REG["ToolRegistry.execute(name, params)"]
    REG --> FIND{"_registry.get(name)"}
    
    FIND -->|"命中"| CAST["tool.cast_params(params)"]
    FIND -->|"未命中"| HINT["⚠️ 错误提示 + 建议可用工具列表<br/>返回给 LLM 自行纠正"]
    
    CAST --> VALID["tool.validate_params(casted)"]
    VALID -->|"通过"| DISP{"工具类型?"}
    VALID -->|"失败"| ERR["ValidationError → LLM"]
    
    DISP -->|"filesystem"| FS["<b>文件系统工具</b><br/>_FsTool.resolve_workspace()<br/>路径安全检查"]
    DISP -->|"spawn"| SPAWN["<b>子代理</b><br/>SubagentManager.spawn()<br/>→ 独立 AgentLoop 运行"]
    DISP -->|"exec"| EXEC["<b>命令执行</b><br/>安全过滤 + timeout"]
    DISP -->|"其他"| OTHER["<b>web_search / fetch / cron / message</b><br/>各自独立实现"]
    
    FS --> R["read_file"]
    FS --> W["write_file"]
    FS --> E["edit_file<br/>三层匹配引擎:<br/>① 精确匹配<br/>② 缩进容错<br/>③ 行首/行尾截断"]
    FS --> L["list_dir"]
```

---

## 五、Memory 双层架构

```mermaid
graph TD
    subgraph "MemoryStore — 文件 I/O"
        MS_READ["read_memory()<br/>→ MEMORY.md 内容<br/>→ 注入 system prompt Layer 3"]
        MS_WRITE["write_memory(content)<br/>→ 追加到 MEMORY.md"]
        MS_HIST["append_history(entry)<br/>→ 追加到 HISTORY.md<br/>格式: [YYYY-MM-DD HH:MM] ..."]
    end

    subgraph "MemoryConsolidator — 调度策略"
        TRIGGER["_save_turn() 每次触发"]
        CHECK["maybe_consolidate_by_tokens()"]
        BUDGET["计算 budget<br/>= ctx_window - max_completion - 1024<br/>目标: 压缩到 budget // 2"]
        
        CHECK --> EST["estimate_session_prompt_tokens()"]
        EST --> OVER{"超过 budget?"}
        OVER -->|"否"| SKIP["跳过"]
        OVER -->|"是"| BOUNDARY["pick_consolidation_boundary()<br/>按 user-turn 边界安全分割"]
        
        BOUNDARY --> COMPRESS["compress() → LLM"]
        COMPRESS --> LLM_CALL["LLM 被引导调用 save_memory 工具"]
        LLM_CALL --> WRITE["写 MEMORY.md + HISTORY.md"]
        WRITE --> INC["last_consolidated += N<br/>后续 API 调用只发未压缩消息"]
    end
    
    MS_WRITE -.->|"被 Consolidator 驱动"| COMPRESS
    MS_HIST -.->|"被 Consolidator 驱动"| COMPRESS
```

---

## 六、Subagent 隔离机制

```mermaid
sequenceDiagram
    participant Main as 🔵 主 Agent
    participant SM as SubagentManager<br/>subagent.py
    participant SA as 🟢 子 Agent
    participant Bus as MessageBus
    participant Context as 主 Agent Context

    Main->>SM: spawn(task, label)
    SM->>SM: task_id = uuid4()[:8]
    SM->>SM: 注册 done_callback
    SM->>SA: asyncio.create_task(_run_subagent())
    
    Note over SA: 🔒 严格隔离<br/>— 工具子集 (无 message/spawn)<br/>— max_iterations = 15<br/>— fail_on_tool_error = True<br/>— 独立 ContextBuilder + Runner<br/>— 与主 Agent 并行运行

    SA->>SA: 独立 build_system_prompt()
    SA->>SA: 独立 run_stream() 循环

    alt 正常完成
        SA-->>SM: 完整响应文本
    else 超时/异常
        SA-->>SM: 错误信息
    end

    SM->>SM: done_callback 清理
    SM->>Bus: _announce_result(task_id, result)
    Note over Bus: 包装为 system 消息<br/>注入主 Agent context<br/>带使用说明指令

    Bus->>Context: 主 Agent 下轮 _process_message<br/>看到子代理结果
```

---

## 七、Hook 策略模式

```mermaid
graph TD
    subgraph "Hook 接口 (AgentHook)"
        H1["on_run_start(ctx)<br/>→ LLM 调用开始"]
        H2["on_stream_delta(delta)<br/>→ 每个流式 token 到达"]
        H3["on_stream_end(resuming)<br/>→ 流结束 (resuming=T: 还有工具; =F: 最终)"]
        H4["on_tool_start(name, params)<br/>→ 工具执行前"]
        H5["on_tool_end(name, result)<br/>→ 工具执行后"]
        H6["on_run_end(ctx)<br/>→ LLM 调用结束"]
        H7["finalize_content(content)<br/>→ 内容后处理 pipeline"]
    end

    subgraph "CompositeHook 错误隔离"
        ASYNC["<b>async 方法:</b> H1-H6<br/>一个 hook 出错不影响其他<br/>错误被 log 吞掉"]
        PIPELINE["<b>pipeline 方法:</b> H7 (finalize_content)<br/>前一阶段输出 → 后一阶段输入<br/>不隔离，任一失败影响全链路"]
    end

    subgraph "BusHook 实现"
        BUS_H["将 hook 事件转换为 Bus outbound 消息<br/>→ TUI 渲染 / Channel 通知"]
    end

    H1 & H2 & H3 & H4 & H5 & H6 --> ASYNC
    H7 --> PIPELINE
    ASYNC --> BUS_H
    PIPELINE --> BUS_H
```

---

## 八、TUI 渲染状态机

```mermaid
stateDiagram-v2
    [*] --> Idle: 启动
    
    Idle --> Thinking: Enter 键<br/>_pre_submit() → stream_start()
    
    Thinking --> Streaming: 首条 LLM delta<br/>stream_delta() → cancel thinking
    
    Streaming --> ToolExec: LLM 返回 tool_use<br/>flush_stream() + tool_phase_start()
    
    ToolExec --> Streaming: LLM 继续响应<br/>stream_delta() → cancel tool
    
    Streaming --> Done: 最终响应结束<br/>pop_stream() + add_response()
    
    Done --> Idle: _turn_complete()<br/>更新上下文用量

    note right of Thinking: ⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏<br/>100ms/帧 braille spinner
    note right of ToolExec: ◐◓◑◒<br/>150ms/帧 半圆 spinner<br/>+ progress hint
    note right of Streaming: 流式文本实时追加<br/>_stream_buf 累积
```

---

## 九、模块职责速查

| 模块 | 文件 | 核心职责 | 关键概念 |
|:--|:--|:--|:--|
| **ContextBuilder** | `context.py` (202行) | 五层 prompt 组装 | SOUL/AGENTS/USER/TOOLS/MEMORY skills |
| **AgentLoop** | `loop.py` | 消息调度管线 | WeakValueDictionary 并发锁、_process_message |
| **Runner** | `runner.py` | LLM 调用编排 | 工具循环 → send_message 直到无 tool_use |
| **AgentHook** | `hook.py` (108行) | 7 方法生命周期 | CompositeHook 错误隔离、BusHook 实现 |
| **ToolRegistry** | `tools/` | 工具注册/查找/执行 | cast_params → validate_params → execute |
| **FilesystemTools** | `tools/filesystem.py` | 4 个文件工具 | _FsTool 路径安全、EditFile 三层匹配 |
| **MemoryStore** | `memory.py` | 文件 I/O | MEMORY.md / HISTORY.md 双文件 |
| **MemoryConsolidator** | `memory.py` | 自动压缩调度 | budget 计算、user-turn 边界、LLM 驱动压缩 |
| **SubagentManager** | `subagent.py` (263行) | 后台隔离代理 | 工具子集、max_iterations=15、并行运行 |
| **SkillsLoader** | `skills.py` (228行) | 技能发现/加载 | 双目录扫描、依赖检查、XML summary |
| **Provider** | `providers/` | LLM API 适配 | Anthropic/OpenAI/DeepSeek/OAuth |
| **TUI** | `cli/tui.py` | 分屏终端 UI | prompt_toolkit、spinner 动画、滚动 |
| **MessageBus** | `bus/` | 消息路由 | inbound/outbound、handler 注册 |

---

## 十、数据流一图总览

```mermaid
graph LR
    subgraph "📥 输入"
        U["👤 用户<br/>CLI / Telegram / Discord"]
    end

    subgraph "🐈 nanobot"
        B["🔀 Bus<br/>路由"]
        L["🔄 Loop<br/>调度"]
        C["📋 Context<br/>组装"]
        R["🏃 Runner<br/>编排"]
        T["🔧 Tools<br/>执行"]
        M["🧠 Memory<br/>压缩"]
        S["🎯 Skills<br/>发现"]
        A["🤖 Subagent<br/>并行"]
        H["🪝 Hook<br/>拦截"]
    end

    subgraph "☁️ 外部"
        P["LLM<br/>DeepSeek / Claude / GPT"]
    end

    subgraph "📤 输出"
        O["TUI / Channel<br/>流式渲染"]
    end

    U --> B --> L
    L <--> C
    L <--> R
    C <--> M
    C <--> S
    R <--> P
    R <--> T
    R <--> A
    R --> H
    H --> B
    B --> O
    O -.->|"下一轮"| U
```
