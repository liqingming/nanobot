# FGUI技能测试24：主线程等待与工具参数提示修复

## 范围与原因

话题 `cli:session_2ef88dfadcb14e189070c43fb4b1e63d` 的主日志显示，spawn、查询、绑定等操作之间反复相隔约 300 秒。原因是工具后的消息检查与回复收尾共用 `_drain_pending`，只要本话题仍有子任务且队列为空，就进入最长 300 秒的等待。

另外出现 `my(action="check", key="subagent")` 和 `subagent_control(action="wait", timeout_s=120)` 参数错误。正确键是 `subagents`，wait 的既有上限为 30 秒。

## 本次改动

- `runner.py`、`loop.py`：透传 `wait_for_subagents`。工具后默认只读取已到达消息；非空正常回复收尾时才允许等待本话题子任务。等待可被真实用户消息和取消打断；显式等待用户输入时不额外等待子任务。
- 兼容无参数、仅接受 `limit` 的旧消息回调。保留等待超时、消息顺序和持续目标续跑机制。
- `tools/self.py`：描述列出只读键 `subagents`；未知键错误提示可用无 key 查询概览，对 `subagent` 额外给出正确键及 `subagent_control(action='list')` 指引。已有同名 scratchpad 数据仍可读取。
- `fork/agent/subagent_control.py`：说明整数 0～30 秒范围、默认 10 秒、合法 wait 示例，以及超时仅返回状态。仍拒绝 120，不扩大上限或静默截断。

## 验证

新增真实 AgentLoop 消息回调 + AgentRunner 的异步调度回归：spawn 后必须在工作者完成之前执行 bind；仅在阶段回复后等待；覆盖回执续跑、用户插话后继续等回执、取消、模型错误、显式等待用户。修复前首批四项有效回归均因提前阻塞超时失败，修复后通过。

首次修复分批测试共 **259 个不同用例通过**（跨批重复项已扣除）：

- 调度、my、subagent_control、runner 注入：154 passed；补充旧回调等待标志参数化后，runner 注入 34 项在后续批次全部通过。
- 原 `_drain_pending` 阻塞、非阻塞、超时用例：3 passed，8 deselected。
- 目标续跑、子任务回执、上下文预算、治理和持久化：70 passed。
- runner 注入、目标续跑、取消、turn continuation 扩展批次：64 passed，1 failed，1 deselected。
- 本次涉及的 9 个 Python 文件 `ruff check` 通过；`git diff --check` 通过。

首次扩展检查发现的问题（后续已按用户授权修复，见下文）：

1. `test_subagent_announces_error_when_tool_execution_fails` 的模拟模型始终返回同一工具调用，而当前默认 `fail_on_tool_error=False`、迭代上限 1000；长时间未结束后主动中止，在后一批中排除。这一用例未取得通过结果。
2. `test_cancel_by_session_cancels_running_subagent_tool` 在执行清理时失败：`execution_scope.py` 对模拟 provider 的 `aclose_execution` 执行 await，但旧用例使用普通 `MagicMock`，报 `TypeError: object MagicMock can't be used in 'await' expression`。该子任务 spec 没有 injection_callback，不进入本次新增等待分支。


## 扩展测试后续修复

用户授权修复上述两个问题后，仅调整 `tests/agent/test_task_cancel.py`，没有修改生产运行逻辑：

- 严格工具失败用例显式设置 `fail_on_tool_error=True`，模拟模型只提供两次不同 ID 的工具调用，迭代上限为 3，并设置 5 秒测试等待上限；断言恰好调用两次、以 `tool_error` 结束、错误回执保留已完成步骤。
- 两个用例均使用带 `LLMProvider` 接口约束的 mock、真实生成参数和异步 `aclose_execution`，并断言清理方法被 await 一次。
- 取消用例为等待启动与取消添加超时，用 finally 回收任务；保留工具已取消、任务处于 cancelled 状态、没有误发结果回执的断言。

两个原问题用例单独验证 **2 passed**。以下完整扩展回归 **80 passed in 9.17s**，无排除用例：

```powershell
python -m pytest tests/agent/test_runner_injections.py tests/agent/test_runner_goal_continue.py tests/agent/test_task_cancel.py tests/session/test_turn_continuation.py tests/fork/test_subagent_dispatch_wait.py tests/fork/test_subagent_control.py -q
```

修改后的测试文件通过 `ruff check` 和 `git diff --check`。以上 80 项包含此前通过的回归，不与首次 259 项直接相加。

## 生效范围

这是工作区源码及离线回归结果；未提交、未重启运行中的 nanobot，也未改 GOT_PC 工程和历史会话。现有进程需要重启并加载此工作区源码才会使用修复；真实模型下再次运行 FGUI 流程尚未验收。
