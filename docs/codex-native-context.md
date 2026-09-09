# Codex 原生上下文治理（最小闭环）

实现范围：仅 `nanobot/fork/providers/codex_app_server_provider.py`。
普通 Provider 和旧版 Codex SSE Provider 沿用原治理流程。

## 分工与执行边界

- 首次请求仍由 nanobot 修复消息结构、约束历史及工具结果规模。
- 同一次 runner.run 内，已发送的模型副本前缀保持稳定，仅对新增工具结果进行整形。
- 历史在 Codex 线程内部由原生自动压缩管理。不在等待动态工具回执时调用
  `thread/compact/start`，也不把 RPC 接受请求当成压缩完成。
- 不修改用户的 Codex 配置文件。通过临时线程的 `config` 设置自动压缩阈值和
  `model_auto_compact_token_limit_scope = "total"`。
- 用户真实插话、历史编辑、模型/工具/预算变化仍走原检查点逻辑，不能为避免重建而忽略。
- 仍是一轮 nanobot 外层 turn 对应一个临时 Codex 进程/线程；没有实现跨用户轮持久线程。

## 预算和证据

- 输入预算来自模型有效窗口；显式 `context_block_limit` 只能进一步收紧，不能放大。
- 原生自动压缩触发阈值设为输入预算的 80%，为单批新回执预留 20%。
- 新回执不重复计入已发送的工具定义、调用参数和历史。小回执原样保留；
  超大文本需要压缩时，先保存可重新读取的原文，摘要携带完整输出路径。
- Provider 再检查首次/恢复输入及待提交回执。不能满足本地估算预算时明确停止，
  不靠删除幂等账本、重建重跑或无限重试绕过。
- 80% 是原生压缩的触发阈值，不是服务端总占用的硬上限。内部工具、原生提示词、
  图像和估算误差均可能使真实占用与本地不同。没有声称“绝不超窗”。
- 没有窗口信息的直接 Provider 调用不强行构造虚假预算，使用 Codex 自身默认配置。
- 可恢复的动态工具断线仍受原幂等账本保护；冻结副本过大、不能安全恢复时停止。

## 压缩事件与异常

跟踪同线程的 `contextCompaction` item started/completed 事件，以 item ID 去重。
诊断字段：

- `context_management = codex_native_auto`
- `native_auto_compact_token_limit`
- `native_compactions_started / native_compactions_completed`
- `native_compaction_in_progress`
- `native_recovery_suppressed`（异常时）

压缩事件使旧的上下文输入读数失效，等待同线程新的 tokenUsage.last。
缺少新读数时使用明确标为 estimated 的本地估算，不把旧读数伪装成压缩后实际值。
runner 治理日志的 `native_checkpoint_copy` 仅指本地检查点，不代表服务端上下文大小或费用节省。

原生压缩尚未完成时断线，或线程已有原生命令/文件副作用时发生异常，
不自动重建重放。保留账本并清理当前连接。取消操作同样只清理对应执行者。
原生模式的上下文超限错误不再触发 runner 的“本地裁剪后重试”。

## 已验证与未验证

- 本机实际执行程序：Codex CLI/App Server 0.153.4。
- 使用该版本导出的 JSON Schema 核对配置和压缩 item 事件。
- 实际 App Server 完成 initialize 和只读 ephemeral thread/start，
  接受 64000 自动压缩阈值与 total 计数方式；没有发送 turn/start。
- 回归覆盖：同线程多次压缩、已执行原生命令/文件操作后的正常续传、
  回执完整性、压缩失败、原生副作用后断线、接口拒绝配置、取消清理、
  超大输入拒绝、超大证据落盘、小回执保护、主子运行状态隔离。
- 压缩生命周期与长链行为使用真实 stdio 模拟服务测试。
- 原生 imageView 的既有兼容桥接仍会关闭线程、携带受控 read_file 结果恢复；
  本次未接管该路径，也未实现插话原地 steer。因此不是消除所有线程重建。
- 尚未在真实模型调用中触发自动压缩，尚未运行真实 FGUI 全流程；
  因而不承诺实际耗时、费用或长流程成功率已经改善。
- 未改变 FGUI 技能、任务产物、既有失败测试，也未自动重启应用。

## 官方依据

- https://developers.openai.com/codex/config-reference
- https://developers.openai.com/codex/app-server

本地代码：`nanobot/fork/agent/native_context.py`、
`nanobot/fork/providers/codex_native_context.py`。
测试：`tests/fork/test_codex_native_context.py`。
