# nanobot 源码阅读指南

## 第一层：入口与骨架（理解"怎么跑起来的"）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 1 | `__main__.py` | 程序入口，很短，只是调用 CLI |
| 2 | `cli/commands.py` | CLI 命令定义（`agent`、`config`、`channels` 等），重点看 `agent` 命令的交互循环 |
| 3 | `nanobot.py` | `Nanobot` 核心类，组装所有子系统（配置→provider→tools→agent loop→channels） |

## 第二层：配置系统（理解"怎么配的"）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 4 | `config/schema.py` | 配置的数据结构定义（provider、model、tools、channels 等字段） |
| 5 | `config/loader.py` | 配置加载逻辑（YAML 解析、环境变量替换、默认值合并） |

## 第三层：LLM Provider（理解"怎么调用模型的"）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 6 | `providers/base.py` | Provider 抽象基类，定义了 `chat`、`chat_stream`、重试逻辑、消息清理 |
| 7 | `providers/openai_compat_provider.py` | 最核心的 provider，OpenAI/DeepSeek/OpenRouter 等都用它，包含流式解析 `_parse_chunks` |
| 8 | `providers/registry.py` | Provider 注册表，内置了 20+ 提供商的预设 |

## 第四层：Agent 核心（理解"怎么思考和执行的"）⭐ 重点

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 9 | `agent/runner.py` | Agent 运行器，管理"调 LLM → 解析 tool_call → 执行工具 → 再调 LLM"的循环 |
| 10 | `agent/hook.py` | Hook 机制，在 runner 各阶段插入回调（流式输出、进度通知、内容过滤） |
| 11 | `agent/loop.py` | Agent 主循环，把 runner + tools + memory + skills 串起来，是 `_run_agent_loop` 的所在 |
| 12 | `agent/context.py` | 上下文管理（system prompt 拼接、token 预算控制） |
| 13 | `agent/memory.py` | 记忆系统（短期/长期记忆的读写） |

## 第五层：工具系统（理解"能做什么"）

| 顺序 | 文件 | 看什么 |
|------|------|--------|
| 14 | `agent/tools/base.py` | Tool 基类，定义工具的标准接口 |
| 15 | `agent/tools/registry.py` | 工具注册表 |
| 16 | `agent/tools/shell.py` | 最常用的工具——执行 shell 命令 |
| 17 | `agent/tools/web.py` | Web 搜索/抓取工具 |
| 18 | `agent/tools/filesystem.py` | 文件读写工具 |

## 第六层：辅助系统（按需阅读）

| 文件 | 用途 |
|------|------|
| `bus/` | 消息总线，解耦入站/出站消息 |
| `cli/stream.py` | 终端流式渲染（spinner、Live、Markdown） |
| `utils/helpers.py` | 工具函数（`strip_think`、`build_assistant_message` 等） |
| `templates/` | System prompt 模板（SOUL.md、TOOLS.md 等） |
| `channels/` | 各 IM 平台适配器（Telegram、Discord、微信等） |

## 核心数据流

```
用户输入 → CLI → bus → agent loop → runner → provider(LLM) → tool 执行 → 响应 → bus → CLI 输出
```