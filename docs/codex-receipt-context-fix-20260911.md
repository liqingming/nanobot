# FGUI技能测试24：子任务回执触发全量上下文重建

## 现场与原因

2026-09-11 14:59:32，话题 `cli:session_2ef88dfadcb14e189070c43fb4b1e63d` 收到子任务完成回执。下一次模型请求在 14:59:33 被本地输入预算拦截：`278209 > 258400`。此前原生线程最近报告的上下文输入为 121274 tokens，自动压缩阈值为 206720。

`ContextCheckpoint.needs_rebase` 会标记新增 user 消息，要求 provider 显式同步。provider 原先仅在出现原生文件/命令事件时使用 `turn/steer`，纯动态工具执行的回执则关闭旧线程并走全量历史启动检查，因此累计的本地历史过大时中断。

## 修复

- `codex_app_server_provider.py` 以历史前缀及模型、工具等配置是否稳定为条件：可以追加时，直接在原线程执行 `turn/steer`，随后提交已落盘的待处理工具结果。
- 保留新增输入和工具结果的预算检查，不重新计算已在原线程中的完整历史；没有扩大预算或删除历史。
- 历史/配置确实变化时仍走原有受保护的重建路径；存在原生副作用时继续拒绝不安全重建。
- 在发送 steer 前标记该线程已有输入注入尝试。若注入或后续传输失败，禁止自动恢复重放，保留幂等账本并返回不可自动重试的错误，避免重复注入。

## 回归

新增确定性预算估算 + 真实 stdio 模拟，复现“每次增量合法、累计历史超过启动预算”的情况。旧逻辑以 `1116 > 1000` 失败；修复后始终只有一个 thread/start、一个 turn/start，回执仅 steer 一次，7 个工具结果各提交一次并最终完成。

扩展用例覆盖有/无原生副作用时的正常回执、多次回执、steer 拒绝、确认 turnId 不匹配、注入时断线、注入后断线及新增回执自身超预算。旧历史治理测试仍覆盖真正重建时的严格幂等保护，普通新增回执测试改为验证增量同步。

完整相关回归 **205 passed in 23.93s**：

```powershell
python -m pytest tests/fork/test_codex_app_server_provider.py tests/fork/test_codex_context_rebase.py tests/fork/test_codex_native_context.py tests/fork/test_codex_collaboration_boundary.py tests/fork/test_subagent_dispatch_wait.py tests/agent/test_runner_injections.py tests/agent/test_runner_goal_continue.py -q
```

本次涉及的 3 个 Python 文件通过 `ruff check`、`git diff --check`。

## 生效与现场状态

只修复 nanobot 源码与测试，未提交、未重启、未修改 GOT_PC 工程或历史会话。运行中的实例需重新加载修复版本。已中断线程不会因源码变更自动恢复；原 GOT_PC 冲突合并任务仍需后续恢复，真实模型现场尚未复验。
