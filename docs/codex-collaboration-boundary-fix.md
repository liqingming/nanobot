# Codex 协作边界修复（2026-09-10）

## 授权范围
修复 nanobot 宿主协作接入与线程事件归属；保留已完成阶段①～③上下文治理改动。不修改 FGUI 技能、工程产物、历史会话、章节占用或用户配置，不提交、推送、重启或运行真实模型。

## 问题证据
FGUI技能测试20 的 Codex 底层记录表明，14:12:42 原生执行者 /root/search_441_resume 已创建，随后绑定并执行读取；14:13:07 nanobot 才因 collabAgentToolCall 结束本轮。因此事件拦截不是执行前审批，不能认定子任务从未启动，也不能自动释放旧占用。
官方配置参考：https://learn.chatgpt.com/docs/config-file/config-reference
当前协作开关为 agents.enabled；此前桥接仅设置 features.multi_agent=false。

## 改动
- codex_app_server_provider.py：进程参数增加 agents.enabled=false，thread/start 配置增加 agents.enabled=false，保留旧开关。守卫明确只用宿主 spawn 和 subagent_control；不可用则报告阻断，不替换为原生协作。
- 保存 turn/start 返回的真实 turn ID；在任何 item/、turn/、线程用量事件影响工具、文本、用量、压缩或完成状态前校验 threadId 和 turnId（含嵌套 turn.id）。
- 缺失或冲突的归属安全停止，诊断记录预期和实际身份；不把子线程结果归入父任务。
- 原生协作事件与归属错误均标记 unsafe_protocol_event，不依赖是否启用原生压缩预算，禁止自动纠偏或恢复重放；失败保留幂等账本。
- fork/agent/subagent_control.py：新增当前话题及工作区范围内的 list/status/wait/cancel。等待有界，超时或取消等待不取消执行者；显式取消等待任务清理，重复取消不打断同一清理。
- 在 SubagentManager 创建和结束点登记回执，保留最多 256 个终态，不保存完整任务正文或模型输出。未知 ID、跨话题、跨工作区、进程重启、回执淘汰均不证明已停止；清理异常也不确认停止。
- spawn 描述明确 fresh/不继承父历史、只绑定返回的实际 ID。使用完整 UUID，终态与业务验收分开。原有独立执行身份、结果回原话题及 provider 清理机制保留。
- 核心只增加创建/清理接入和工具发现入口，管理实现放在 fork。未修改 FGUI 技能。

## 验证
主回归：183 passed，覆盖以下 9 文件：
- tests/fork/test_codex_collaboration_boundary.py
- tests/fork/test_codex_app_server_provider.py
- tests/fork/test_codex_native_context.py
- tests/fork/test_codex_context_rebase.py
- tests/fork/test_subagent_control.py
- tests/fork/test_execution_scope.py
- tests/fork/test_subagent_context_budget.py
- tests/agent/test_subagent_lifecycle.py
- tests/agent/test_subagent.py

覆盖缺失/异线程/异 turn 事件、父线程正常响应、协作泄漏不重启、提交工具回执后失败不回放且保留账本、线程配置、宿主双子任务执行身份隔离、取消等待清理、跨话题/工作区拒绝、超时、未知状态和终态回执。

旧 stdio 模拟服务补齐标准 threadId/turnId；生产校验没有为旧测试放宽。
本机 Codex 0.153.4 使用修复后的实际命令执行 initialize + config/read：agents.enabled=false、features.multi_agent=false；未执行 thread/start 或 turn/start，模型调用数为 0。
本次涉及 Python 文件 ruff check 通过；git diff --check 通过。

扩大工具加载/子任务集成回归：47 passed、1 failed、1 deselected。
- 失败：tests/agent/tools/test_subagent_tools.py::test_drain_pending_blocks_while_subagents_running；既有 fork/agent/input_evidence.py 的 tuple(msg.media) 遇到 media=None 抛 TypeError。该输入取证路径本次未修改，属于原授权范围外，未修复。
- 未运行：test_spawn_tool_rejects_when_at_concurrency_limit；其测试任务“first task”不满足现有委派范围校验，测试意图与既有校验不匹配，未调整。

## 生效与验收限制
- 已完成代码修复与离线回归，不等于阶段④真实运行验收完成，也不宣称全仓库测试全绿。
- 正在运行的 nanobot 进程不会自动加载这些修改；需重启后验证。此次未重启用户进程。
- 新控制接口只管理 nanobot 自己创建的任务，无法接管此前的 Codex 原生执行者或跨进程历史。不要把 unknown 当成 stopped。
- 本次只确认本机配置解析和模拟协议行为；未通过真实模型验证 agents.enabled 对运行时工具暴露的最终效果，未验证真实原生压缩。
- FGUI技能测试20 的残留占用保留。须独立核实原执行者停止及文件断点，再由用户授权续跑；不清账本、不自动释放占用、不从头生成。


## 增量：FGUI技能测试21 的检查点续行冲突（2026-09-10）
- 证据：主话题 cli:session_a9289cb7a18f44cdbd9708f8bcafcfa4 在 15:47:36 因 CodexIdempotencyLedgerError 停止；回执注入/治理后发生上下文重建，账本游标为 0，预期 exec，实际为重新查询 subagent_control。不是原生协作越界。
- 修改范围仅为 fork/providers/codex_app_server_provider.py、tests/fork/test_codex_context_rebase.py 和本节说明；保留此前未提交改动。本次没有读取后再改写现场账本，也没有修改 FGUI 技能、任务、占用或工程产物。
- 最小修复：正常检查点重建的桥接标记 checkpoint_continuation，并在同一桥接的后续轮次保留。仅 subagent_control 的 list/status 且调用 ID 未在账本出现时作为新观察交回宿主，读取最新状态，不回放旧 running 回执，也不移动副作用回放游标。
- 其他操作仍执行原有顺序校验：cancel、wait、exec 及未知工具没有新观察例外。旧调用 ID 不获得例外；已有写入顺序回放仅返回原缓存，不重做副作用。原生副作用阻断、缺失待提交回执阻断及 write-once 账本不变。
- 真正断线恢复/纠偏重新创建的桥接不继承该标记，恢复原有严格校验。增加 checkpoint_continuation 诊断字段，区分正常续行策略。
- 回归先复现：新状态查询的 4 个组合全部出现与现场一致的 index=0 顺序冲突。修复后新增 7 个用例覆盖回执注入、治理重建、list/status 最新结果、多轮保留、流式通知仅一次、旧 ID/取消不放行，以及正常续行后真断线、模型换新 ID 仍须严格回放。
- 最终离线验证：上文相同的 9 个主回归文件共 190 passed；本次两份 Python 文件 ruff check 通过，git diff --check 通过。未运行全仓库测试；上文已记录的范围外旧问题不在本次修复中。
- 生效限制：未重启 nanobot，未运行真实模型或 FGUI 生成，尚未验收到真实任务全流程。需先核实执行者停止、重启加载新代码，再核验原断点后续跑；不要删除账本、释放未知占用或从头生成。
- 原诊断已确认 TestPkg2 子任务 01～06 完成、Search 原位修改未执行；这是诊断时快照，本次没有重新核验现场变化，不能据此直接清占用或宣布整体完成。


## 增量：FGUI技能测试22 的原生操作后回执同步（2026-09-10）
- 现场诊断：17:07:59 原生 exec_command 执行 chapter_orchestration.py next-action；17:10:24 子任务完成回执进入上下文，同时旧工具结果重新治理，触发重建并命中 native side effects 保护。该命令的只读性质不是通用放行依据。
- 本次修复不重建有原生操作历史的线程，也不关闭重建保护。对于设置与已发送前缀完全不变、仅追加普通用户/回执消息的情况，在待处理工具结果已经落盘的边界调用 turn/steer，核验 expectedTurnId 及响应 turnId，随后只提交本次待处理工具结果。保留原线程、原生事件和幂等账本；诊断 context_sync=steered。
- NativeContextPreparation 对纯追加用户消息保留已发送投影，只约束新增证据，避免回执到达导致旧结果重新治理；真实历史改写、模型/工具设置变化或追加 system/developer 仍不能走此路径。修改均在 fork：agent/native_context.py、providers/codex_context_checkpoint.py、providers/codex_app_server_provider.py。
- 保留安全停止：缺失待处理回执、历史改写、模型/高优先级指令变化、输入超限、steer 拒绝/确认不匹配/断线均不重建重放。原生操作或压缩进行中禁止故障自动恢复，不再依赖原生预算是否非零。没有通过命令名称猜测副作用，没有把原生结果冒充动态工具账本记录。
- 协议依据：OpenAI 官方文档 https://developers.openai.com/codex/app-server#steer-an-active-turn 。该接口向正在运行的 turn 追加用户输入，不新建 turn，要求 expectedTurnId，不能修改模型等 turn 级设置。
- 新回归先复现 native side effects 阻断；修复后验证真实 stdio 模拟中的原生命令/文件事件、连续两次回执各送达一次、七次工具结果不重复、原生压缩后仍同线程，以及拒绝/错误确认/断线（含确认成功后断线）不回放、预算先检查、缺失结果/历史和设置变化安全拒绝。
- 最终离线验证：前述 9 个主回归文件，加 tests/agent/test_runner_governance.py、tests/agent/test_runner_injections.py，共 272 passed。本次 5 个 Python 文件 ruff check 通过，git diff --check 通过。不是全仓库测试结论。
- 本次未改 FGUI 技能、工程、占用或现场账本；未提交、重启或调用真实模型，未核验真实任务全流程。模拟协议通过不代表本机实际模型等待工具时的 steer 行为已验收。旧错误轮次不会由本修复自动恢复；先确认执行者停止及断点一致，再加载新代码并授权续跑，禁止从头生成或直接释放未知占用。


## 增量：主任务接续与摘要失败诊断（2026-09-10）
- 范围：仅 nanobot 执行和上下文问题，保留原有未提交改动；不修改 FGUI 技能、工程产物、占用、现场历史/账本或配置，不提交、不重启、不调用真实模型。
- 主任务根因：runner 已有持续目标续跑机制，但最终响应分支未启用 allow_goal_continue。原有 8 项目标测试中 5 项先复现失败。现在只对 stop/end_turn 正常响应启用该接入；真实插话优先，目标活动且未等待用户时继续，目标完成或等待用户时停止。仍遵守迭代上限、错误与取消边界，不将阶段汇报视作业务完成。
- 接续回归：新增 tests/fork/test_goal_execution_continuation.py，使用真实目标工具和临时会话测试章节回执、动态创建目标、最终完成、等待授权、未完成 TODO 阻止提前收口、流式续接及取消。未改变工具授权策略、桥接账本或停止凭据规则。
- 摘要现场：FGUI技能测试22 的 runtime.log 1312–1313 行记录 17:29:47 BUILD 阶段泛化 SummaryTransactionError，同分钟 history.jsonl 第617行有该会话 RAW 70 messages。只能确认底层失败信息丢失；无法从旧记录确定具体是输入预算、原生摘要隔离拒绝还是其他异常。已有原生 Provider 分流保持不变，不新增跳过摘要。
- 摘要实现：memory.py 最小接入新 fork/agent/summary_diagnostics.py，输出 context.summary.failed 结构化诊断，记录阶段、稳定原因码、异常类型、预算、结束原因、状态码和白名单错误码，不记录请求/响应正文或异常文本。失败仍保留原文和覆盖游标；取消、保存回滚及诊断落盘失败安全规则保持。
- 验证：针对性接续/摘要组合 63 passed；扩大 22 文件组合为 468 passed、4 failed（43.56秒）。四项均在既有恢复回执的 getpass.getuser()：执行环境 LOGNAME/USER/LNAME/USERNAME 全部缺失，Windows 无 pwd 模块；独立 getpass 调用已复现，未修改范围外恢复代码。仅在临时测试进程使用 WindowsIdentity 获取系统用户名后，回执文件 11 passed；另跨轮接续 6 passed。本次五个 Python 文件 ruff check 和 git diff --check 通过。不能称原环境全绿或全仓测试通过。
- 扩大测试生成的 MagicMock 目录内10个 tool_corrections.json 已按路径白名单核验并清理；未清理任何既有产物。
- 未验证：没有在真实模型上重跑 FGUI，没有证明实际业务端到端自动收口或旧摘要故障已消失；正在运行的旧进程不会自动加载修复。新诊断的作用是后续失败可定位，不是对历史根因的追认。
