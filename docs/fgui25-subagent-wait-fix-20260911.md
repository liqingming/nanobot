# FGUI技能测试25：等待、重复轮询与恢复指引

## 现场

话题 `cli:session_54bc7aaea44d471988271e29925c9b17` 于 15:47:54 报 Codex 事件等待 240 秒超时，自动恢复后 15:56 完成。此前主线程反复执行 `subagent_control status → chapter_orchestration next-action`，触发重复循环警告。长停顿在模型请求阶段，不是工具后的队列检查。

## 修复行为

- 只有单独调用 `subagent_control(action='wait')`、成功回执确认同一 task_id 仍在运行时，runner 在工具结果落盘后进入宿主队列等待。等待期间不发模型请求，真实用户输入、子任务回执或取消可以打断。沿用最长 300 秒的宿主等待检查点，到期交回模型重新判断。
- 工具的 `timeout_s` 仍限定 0～30 秒；该时间控制工具状态等待，随后宿主可能继续等消息。工具说明已明确这一区别。立即查询使用 status。
- spawn、绑定、status/list/cancel、未知/完成任务、错误回执、混合工具批次不会触发这条新增等待路径。原有正常回复收尾等待保持不变。
- spawn、subagent_control 描述与 running 状态提示引导先完成绑定和独立工作，再调用 wait，避免重复查询未变化的计划或用模型思考/sleep 等待。
- 显式 wait 豁免重复工具循环警告；status 和 cancel 仍受保护。包含子任务控制的循环警告给出转入宿主等待的具体指引，不豁免任意 exec 命令。
- 新增 `runner.subagent.wait.start/done` 日志，便于区分宿主等待与模型传输停顿。
- 自动恢复检测到本话题仍有子任务时，提示复用已有任务 ID、不得重复派发或重做操作，并使用宿主等待。其他话题的子任务不触发提示。此处没有新增自动派发动作，也未扩大自动恢复次数。
- `my` 对 tools 查询失败给出 tool_names 指引；tool_names 为只读并出现在概览中，工具注册表继续禁止访问和修改。

## 验证

包含真实 AgentLoop 队列 + AgentRunner 的测试验证：spawn 后先 bind，单独 wait 后模型调用次数不增长；回执续跑、用户插话、取消均正常；工具结果只保留一次。另测等待匹配边界、恢复话题隔离和工具名称只读保护。

本轮完整相关组合 **341 passed in 30.11s**：

```powershell
python -m pytest tests/fork/test_subagent_dispatch_wait.py tests/fork/test_subagent_control.py tests/agent/tools/test_self_tool.py tests/agent/test_runner_injections.py tests/agent/test_runner_goal_continue.py tests/agent/test_loop_runner_integration.py::test_dispatch_auto_recovers_transient_model_error tests/fork/test_codex_app_server_provider.py tests/fork/test_codex_context_rebase.py tests/fork/test_codex_native_context.py tests/fork/test_codex_collaboration_boundary.py -q
```

本次 8 个修改的 Python 文件通过 ruff check 与 git diff --check。

额外扩展检查并非全部通过，以下未按本次范围修改：

- `tests/agent/test_runner.py` 为 18 passed、3 failed。两个旧用例持续设置活动目标，却预期普通回复直接结束，与当前持续目标续跑行为冲突；`test_subagent_max_iterations_announces_existing_fallback` 使用普通 MagicMock 模拟异步清理，出现 await TypeError。
- 对整个 `test_loop_runner_integration.py` 的组合运行长时间未完成后主动中止，未取得该文件全量结论；与本次修改直接相关的自动恢复用例已单独通过并纳入上述 341 项。

## 限制与生效

本次解决等待子任务时仍依赖模型轮询的执行路径，提供超时后的恢复指引，不能保证外部模型服务永不超时。没有延长模型超时或绕过原生副作用重放保护。

未提交、未重启、未修改 GOT_PC 工程或历史会话。运行实例需要重新加载源码；真实模型下的后续话题尚未复验。
