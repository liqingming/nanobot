# 体验优化4：上下文修复交付记录

更新：2026-09-11。状态：三项修复的实现核验、组合回归和交付说明已完成。

## 完成范围

原话题目标：修复 FGUI技能测试23 暴露的 nanobot 历史证据裁剪与 read_file 去重不协调、子任务缺少治理日志、实际上下文占用与累计治理统计混淆。

接手时实现及专属回归已在工作区，待办停在“执行组合回归并更新交付说明”。本次核对实际接入，并重新执行当时中断的完整 21 文件组合，不依赖中断进程的测试结果。收尾仅新增此文档及原治理文档的交付入口，没有新增运行时代码修改。

沿用原边界：完成代码与离线验证并明确未验收范围；不提交、不重启、不调用真实模型，不改配置、FGUI 技能/工程产物/占用、现场历史或幂等账本，保留既有未提交改动。

## 三项实现与验收

### 1. 历史证据与文件重读

- `nanobot/fork/agent/native_context.py` 在首次投影或需要重新治理时使用配置副本 `inflight_start_index=0`，取消固定只保留 64 个旧工具交换的裁剪，仍受软/硬预算约束。普通追加保持已发送前缀，不重写源会话或跨过 Provider 预算门禁。
- `nanobot/fork/agent/read_visibility.py` 的 `require_read_evidence` 已接入 `AgentRunner.run`。运行期间通过 ContextVar 要求重读返回正文，正常返回、异常、取消后复位。
- `nanobot/agent/tools/filesystem.py` 在该模式内不以“磁盘未变”代替正文；独立工具调用保留旧去重行为。路径权限、读取范围、force 和读取后外部修改的哈希保护保持有效。
- 测试覆盖超过 64 个旧正文保留、稳定前缀、软/硬预算、无法缩减时启动前拒绝、连续重读、嵌套/并发隔离、异常/取消复位、路径限制与哈希保护。

这是保守重读策略，不声称能得知原生压缩后远端模型实际保留了哪些正文。重读可能增加输入量，预算门禁仍然生效。

### 2. 子任务治理日志

- `nanobot/fork/agent/subagent_diagnostics.py` 已通过 SubagentManager 的 `event_logger` 和独立 `turn_id` 接入真实 runner。
- 日志位于 `<data_dir>/subagent-runtime/<父会话哈希>/<task_id>/<run_id>/runtime.log`；无 data_dir 时使用工作区。每次执行有新 run_id，关联父会话与 Provider 执行身份。
- 记录治理预算、投影变化、模型 usage、原生压缩诊断、开始/结束及异常/取消。新统计字段进入白名单，不记录任务正文、工具参数、模型原始输出或任意嵌套字段。
- 测试覆盖真实 Manager/runner 接入、普通/原生模式、实际/估算用量、并发身份隔离、初始化失败、错误/取消终态、重执行身份和日志写入失败不改变任务结果。

日志为 best effort：磁盘写入失败可能缺记录，不保证进程强杀后的完整审计。新增白名单不代表所有既有调试日志均已脱敏。

### 3. 占用与累计统计

`nanobot/fork/agent/governance_metrics.py` 已在每次 runner 执行内独立使用：

| 字段 | 含义 |
| --- | --- |
| `context_input_peak_tokens` | 实际上下文占用样本峰值 |
| `context_input_estimated_peak_tokens` | 估算占用样本峰值，单独统计 |
| `context_input_*_samples` | 实际、估算、未知样本计数 |
| `local_projection_reduction_tokens` | 当前本地投影相对源记录的 token 差值 |
| `local_projection_reduction_peak_tokens` | 本地投影差值的观察峰值 |
| `local_projection_reduction_token_observations` | 各次差值观察累加，允许重复计入同一段历史 |
| `response_prompt_usage_peak_tokens` | 已观察响应的 prompt usage 峰值 |
| `projection_omitted_tool_results` / `projection_changed_tool_results` | 当前投影中移除或改变的工具结果数 |

没有实际/估算样本时对应峰值为 null，不用 0 冒充测量值。兼容字段 `governance_saved_total`、`prompt_peak_tokens` 保留，并增加语义说明字段。累计用量保持原口径；本地差值和重复观察总和不代表远端压缩量或费用节省。

测试以重复观察、混合实际/估算/未知值及真实 runner 重读验证统计，覆盖普通旧策略、普通事务策略和原生模式，验证子任务日志保留新字段且不泄漏原文。

## 两种中断

- 09:41 阶段汇报仍称整体未完成，执行却以 completed 结束。现有 runner 已在正常 stop/end_turn 回复后检查持续目标：活动且未等待用户时继续；完成、等待用户、异常、取消及迭代上限仍按对应分支处理。接续测试包含在本次组合中。
- 10:02 CodexIdempotencyLedgerError 来自正常检查点续跑再次轮询进程。此前已在本对话修复：仅新调用 ID、无输入、无关闭 stdin、无终止操作的 write_stdin 可读取新结果；同 ID、带副作用操作和真实断线恢复仍遵守重放保护。桥接模拟覆盖连续轮询、旧 ID 拒绝、副作用拒绝及续跑后真断线恢复。

以上是磁盘代码与离线测试结论，不证明旧进程当时已加载新实现。

## 最终验证

2026-09-11 重跑原定 21 文件组合：**423 passed，34.62 秒，退出码 0**。没有失败、跳过或筛除项。模型响应均为本地替身或 stdio 模拟服务。

```powershell
python -B -m pytest tests/fork/test_governance_metrics.py tests/fork/test_read_visibility.py tests/fork/test_native_read_visibility.py tests/fork/test_subagent_diagnostics.py tests/fork/test_codex_native_context.py tests/fork/test_codex_context_rebase.py tests/fork/test_codex_app_server_provider.py tests/fork/test_codex_collaboration_boundary.py tests/fork/test_execution_scope.py tests/fork/test_subagent_context_budget.py tests/fork/test_subagent_control.py tests/fork/test_summary_transaction.py tests/fork/test_transactional_context.py tests/fork/test_context_usage.py tests/fork/test_goal_execution_continuation.py tests/agent/test_runner_governance.py tests/agent/test_runner_injections.py tests/agent/test_runner_persistence.py tests/agent/test_runner_goal_continue.py tests/agent/test_subagent.py tests/agent/test_subagent_lifecycle.py -q -p no:cacheprovider
```

三项实现及轮询修复的 13 个相关 Python 文件 `ruff check` 通过：runner.py、subagent.py、tools/filesystem.py、fork/agent 下的 native_context.py、read_visibility.py、subagent_diagnostics.py、governance_metrics.py、fork/providers/codex_app_server_provider.py，以及 tests/fork 下的 test_governance_metrics.py、test_read_visibility.py、test_native_read_visibility.py、test_subagent_diagnostics.py、test_codex_context_rebase.py。

`git diff --check` 通过。未运行全仓库测试，不声称全仓库全绿。此前 176 项桥接/进程工具测试和 49 项接续测试有重叠，不与 423 相加。

## 生效与剩余边界

- 原任务限定的实现、离线组合验收和交付说明已完成，没有该范围内的收尾待办。
- 原话题仍保存中断时的 active 目标和 in_progress 待办，本次未改写。它们是历史状态，本次完成记录以本文为准；没有在原进程执行 complete_goal。
- 运行中的 Python 进程不会自动加载磁盘修改，需重新加载后使用；本次未重启。
- 真实模型长链、原生自动压缩、跨用户轮线程复用、真实异常恢复、FGUI 全流程及实际 token/费用收益仍属于原文档阶段④，未在本次范围内验收。
- 原文档记录的范围外问题没有顺手修复，例如旧输入回执 media=None、部分旧测试前提与当前接口不一致等；本次未重新验证这些范围外结论。

背景见 [上下文治理重构执行记录](context-governance-refactor.md) 和 [Codex 协作边界修复](codex-collaboration-boundary-fix.md)。
