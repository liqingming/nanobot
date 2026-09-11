# 上下文治理重构执行记录

更新：2026-09-10。状态：已完成本次授权的阶段①～③及离线验收；阶段④另行授权。

## 授权与边界

用户已授权按方案实施阶段①～③；阶段④跨用户轮 Codex 线程复用单独设验收门槛。
不提交或推送，不修改用户配置、历史会话、FGUI 技能或任务产物，不自动重启。
实现优先放在 nanobot/fork，核心仅接入；不使用 worktree 或 ruff format。
保留 Provider 权限边界、工具配对、主子身份隔离及 write-once 幂等账本。
原生线程不因本地归档、Dream 或异常而自动重建重放。

## 执行计划

- [✅] 阶段①：统一预算、诊断与基线测试
- [✅] 阶段②：可靠任务恢复、证据与摘要覆盖事务
  - [✅] 定位三条危险路径：回放窗口归档、token 归档、空闲归档
  - [✅] 摘要失败与原文归档分离；失败不得推进摘要覆盖游标
  - [✅] 摘要、覆盖边界和版本在同一次原子保存中提交；保存失败按实际落盘结果恢复一致状态
  - [✅] 复用 ContextState/goal/todo 和 EvidenceRef，建立机械可校验恢复包；未核验来源不升级为用户授权
  - [✅] 原始证据可靠保存；文本回执绑定实际字节与行范围，恢复时受权限和预算约束核验源文件，其余明确未核验
  - [✅] 完成入口来源和恢复安全出口审查：插话原文引用与失败整批保留；未接凭据入口保守未核验，不推断身份/授权
- [✅] 阶段③：普通 Provider 事务式摘要策略及显式旧策略回退
  - [✅] 显式配置、普通/原生治理责任分派
  - [✅] 完整工具批次后异步摘要，验证后切换模型上下文版本
  - [✅] 移除新策略下无条件 24k→14k、64 个交换淘汰和静默裁历史
  - [✅] 摘要失败或关键状态无法容纳时明确停止，不降级绕过
- [✅] 组合回归、迁移/回退文档、阶段④真实验证门槛

## 阶段①交付

新增 nanobot/fork/agent/context_budget.py：

- ContextBudget 区分未知窗口（input_tokens=None）和已耗尽预算（0）。
- 复用 provider_input_token_budget，以 Provider 窗口语义及输出预留为准。
- 显式 context_block_limit 与 Provider 预算取较小值，不能扩大窗口。
- 无窗口的直接调用仍不构造虚假预算。
- bool 和 mock 值不当成合法整数输出预留。
- 诊断只含数值，不写提示词、回执内容或证据路径。

接入点：

- ContextGovernor.input_budget
- AgentLoop._replay_token_budget
- Consolidator._input_token_budget（新增可选 context_block_limit）
- Codex native_input_budget
- runner.context.governance 日志增加预算、compaction_owner、metrics_source

已知零输入预算时 runner 直接返回明确错误，不调用模型，不走最小修复兜底；
Consolidator 不在零预算下归档。切换 Provider 时清空旧后台估算缓存。

本阶段尚未改变非零预算下的软压缩、历史 64 个工具交换保留、摘要归档失败、
空闲压缩或普通上下文超限后裁剪策略。不能把阶段①视为完整安全闭环。

## 验证

修改前选定基线：69 passed。
修改后组合：156 passed，覆盖以下 10 个文件：

- tests/fork/test_context_budget.py
- tests/agent/test_runner_governance.py
- tests/agent/test_consolidation_ratio.py
- tests/fork/test_codex_native_context.py
- tests/fork/test_subagent_context_budget.py
- tests/agent/test_consolidator.py
- tests/agent/test_loop_consolidation_tokens.py
- tests/agent/test_context_governance_fresh_results.py
- tests/fork/test_codex_context_rebase.py
- tests/fork/test_context_usage.py

改动 Python 文件 ruff check 及 git diff --check 通过。
不调用真实付费模型。Codex 生命周期验证仍是 stdio 模拟，不声称真实自动压缩已验证。

测试适配说明：若干旧测试使用小于默认输出/安全预留的窗口，却依赖显式限制或
旧零预算分支继续执行。已仅调整这些测试的预算前提（显式零输出/测试安全预留，
或提高窗口并同比调整测试估算），保留原本的裁剪和调用顺序断言。

## 阶段②起点风险清单（历史基线）

1. memory.py 的 _consolidate_replay_overflow 在 archive 返回 None 后仍推进游标。
2. maybe_consolidate_by_tokens 同样把有界 raw_archive 当作摘要覆盖成功。
3. compact_idle_session 在摘要失败后仍保留最近尾部并移除其他消息。
4. 游标和续接摘要分多次保存，存在中途失败后状态不一致的风险。
5. archive 输入会先截断；不能把未进入摘要请求的原文也标为被摘要覆盖。
6. 回放消息数和 token 裁剪可能隐藏尚未成功归档的消息；只修游标不够。
7. ContextState 当前对部分关键字段作字符/条数截断；必须避免把截断后的约束
   当成完整任务契约，也不能把模型推断升级为用户授权。
8. normalize_tool_result 的落盘失败后截断可能丢掉唯一原文。
9. runner 的通用治理异常兜底会继续发原请求；新策略的安全错误必须单独停止。

以上是阶段②开始时的基线。第 1～5 项已处理摘要覆盖的机械安全边界，详见下节；
第 6 项仅保证必要摘要失败时前台不会继续回放裁剪，其余回放预算策略仍待阶段③。
第 7～9 项尚未完成，不应把本次增量视为全套治理已重构。

## 公开机制依据

采用公开原则而非复制闭源内部实现：

- https://code.claude.com/docs/en/context-window
- https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
- https://developers.openai.com/codex/app-server

阶段④需真实模型压缩、多轮恢复、插话、取消和原生副作用后断线验证通过，
再考虑跨用户轮持久线程；不能凭 RPC 接受配置就判定成功。


## 阶段②增量：摘要覆盖事务

新增 nanobot/fork/agent/summary_transaction.py，核心 memory.py 只接入事务：

- 回放窗口、token、空闲三条归档路径仅在有效摘要后提交覆盖。
- 空摘要、(nothing)、残缺续接标签、超长摘要、模型错误/输出截断、工具调用返回均不作为有效覆盖。
- 摘要请求计入完整系统指令、旧续接摘要和所选消息；不再截断输入后声称全部已覆盖。
- 不因摘要失败移除消息或推进 last_consolidated；有界 [RAW] 记录只作为审计线索。
- 前台必要归档失败抛出 SummaryTransactionError，停止进入历史回放/模型请求；
  不借用既有布尔返回值表达失败，避免旧调用方和测试把“不需压缩”误判成错误。
- 保存前检查消息、metadata、todo、时间及会话对象是否变化；拒绝过期摘要覆盖新状态。
- 校验覆盖消息与实际移除消息匹配；空闲合法尾部可能非连续，按消息多重集检查无遗漏。
- 摘要、新游标、覆盖范围哈希和版本号由同一次 SessionManager.save(fsync=True) 原子保存。
  每轮提交后再估算，下一轮能读取刚提交的摘要，不再最后单独保存续接文本。
- 保存失败时检查实际落盘标记：替换前失败回滚内存，替换后失败保留已提交版本；
  无法确认则清除缓存并明确停止，不自动重放。
- 取消不提交；后续轮次失败保留之前已成功提交的摘要版本。

测试已覆盖三路径成功/失败/取消、前台停止、重载恢复、保存前后失败、
消息/授权 metadata/todo 并发变化、对象替换、覆盖错配、无效摘要和多轮摘要链。
既有测试的布尔“摘要成功” mock 改为真实字符串；仅边界算法测试显式清除测试安全预留。
旧“摘要失败仍截断”及“输入先截断再摘要”断言按新安全契约替换。
测试不调用真实付费模型，不修改实际用户会话。

本次最终组合回归：224 passed（原阶段①的 10 个文件，加
 tests/fork/test_summary_transaction.py 与 tests/agent/test_autocompact_unit.py）。
全部改动 Python 文件 ruff check、git diff --check 通过；无遗留测试进程。
未提交、推送或重启。

限制与后续：

- 有效摘要校验目前只是结构与覆盖边界验证，不证明业务约束被语义完整保留。
- 超预算的完整摘要输入会停止；尚未实现自动安全分段/分层摘要。
- (nothing) 当前采取保守策略，不据此删除原文；即使普通闲聊也可能暂不压缩。
- 本节完成时恢复包、约束防截断和证据保护尚未实施；现已推进，见下节。
  当前源文件指纹对照和可核验用户授权来源仍未完成。
- [RAW] 失败记录尚未增加跨重启去重；保留原文后持续失败可能重复记录，需在后续处理。
- 当前覆盖元数据记录版本和哈希，尚不是持久化的多版本摘要仓库。
- Codex App Server 的 thread/start 仍使用既有原生执行配置；
  单传 tools=None 并不证明关闭原生工具。摘要专用能力隔离仍需单独接入和协议验证，
  不能据 mock 测试宣称已具备无执行能力的摘要沙箱。
- 固定软阈值、64 个旧工具交换限制及普通 Provider 的静默裁剪仍未替换。


## 阶段②增量：完整恢复包与工具回执快照

新增 fork 模块 recovery_packet.py、tool_evidence.py；核心保留 ContextState、
TaskContract、EvidenceRef、ToolDigest，只接入完整性校验、存取与保护停止。

恢复包：

- 目标、约束、验收条件、等待用户原因、未完成 todo 和完整续接摘要不再静默截断。
- 活跃决定不受旧 12 条上限淘汰；仍被活跃决定或摘要引用的证据不受近期 40 条上限淘汰。
- 恢复包带版本与确定性 SHA-256；摘要覆盖事务在提交前验证包可完整表示，并记录其指纹。
- 64,000 字符为完整性保护上限，不是模型输入窗口或软压缩目标。超限/关键字段无效则停止，
  不删减原字段。实际模型 token 预算和阶段③请求策略仍须另行校验。
- source=user、confidence=authoritative 仅显示为“声明但未核验”；工具回执不能证明用户授权。
  goal/todo/续接摘要明确标为代理记账，不自动授予权限。
- 历史上已经被截断丢失的字段不能通过本次更新重建，不宣称语义完整性已获得证明。

证据：

- 有缓存根时，动态工具回执（含短回执、read_file 和多模态结构）保存为内容寻址快照；
  文本按原始 UTF-8 字节保存，结构化结果采用稳定 JSON，摘要指纹与落盘字节一致。
- 默认位于 data_dir/sessions/<会话键哈希>/evidence/<内容哈希>.txt；
  无 data_dir 时位于 workspace/.nanobot/tool-evidence/<会话键哈希>/evidence/。
  目录参数只来自运行时配置，不用工具参数或回执中的路径。检查 junction/symlink 根外跳转。
- 使用临时文件、fsync 和 Windows 重试式原子替换；已存在的同指纹文件先校验，
  损坏则停止，不覆盖；相同 call_id 的不同回执不会复用旧内容。
- 大文本显示完整快照引用加有界预览；read_file 仍完整内联，避免“读取引用再卸载”循环。
  混合多模态回执也保留内联，不把图像降级为纯文本引用。
- 没有存储根的 SDK 调用保留内联原文，不伪称已落盘，也不截断唯一副本。
- EvidenceRef 保留 read_file 请求的 path/offset/limit/pages，明确覆盖范围仅为返回回执，
  不是整个源文件；快照 SHA-256 不是源文件 SHA-256。
- ToolDigest 的软/硬表示都保留直接可读取的快照路径，当前轮不必等待下一轮 metadata 注入。
- 来源路径只取运行时真实保存结果，不能被工具输出里的 Full output saved to 文本伪造。
- 保存/登记失败会保留整批已执行工具回执并写完成检查点，停止下一次模型调用，不重跑工具、
  不接受注入消息绕过错误。若磁盘同时故障，不能保证检查点落盘，原回执至少保留于运行时消息。
- Codex 仅替换新增回执的快照保存方式，不改前缀冻结、原生预算、线程/幂等生命周期。
- 安全错误不进入通用“最小修复后继续”兜底；带 context_safety_failure 标记，
  已完成 goal 的展示兜底也不能把安全错误改成成功。

尚未完成：

- 本节记录时尚无源文件自动对照；后续已补文本读取字节与当前状态核验，见下节。
  非文本、旧记录和无读取权限的场景仍保守标记未核验。
- 可机器核验的原始用户授权引用链；现采取保守未核验标记，不从代理摘要推导权限。
- 新快照尚无引用感知清理/容量配额，不复用旧按时间删除逻辑，以免删掉活跃证据。
- 更早的运行时检查点仍为 best effort；不能把本次证据写入视为所有崩溃/磁盘故障窗口已闭合。
- 阶段③无工具权限摘要能力隔离、普通 Provider 策略和显式回退未动，阶段②仍进行中。


本轮扩大验证（不混入定向通过数）：

- ContextBuilder、prompt cache、stop、loop tool context：102 passed、1 deselected；
  排除的是未改动模板的既存文案断言 test_tool_contract_balances_general_and_coding_workflows，
  仍期待旧短语 Batch independent bounded reads。另有一个未 await 协程警告，未顺手修复。
- loop_save_turn 的检查点、崩溃恢复、todo 恢复和工具结果保存子集：7 passed、46 deselected。
- 扩大组合在 120 秒超时；拆分后 loop_runner_integration 前 9 项通过，
  在既存 test_subagent_max_iterations_announces_existing_fallback 处达到单独 45 秒上限。
  该子代理 fallback 路径原已列为范围外问题，未修改，不能宣称完整集成套件通过。
- 扩大测试的旧 MagicMock ContextBuilder 会生成仓库内的伪路径临时产物；
  新证据模块已排除非真实路径类型，旧工具纠正模块不在本次修复范围内。


本轮最终定向组合：265 passed、1 skipped（16 个文件，包含新增恢复包/证据测试及
runner_persistence/context_artifacts）；跳过项为当前 Windows 账号无符号链接创建权限。
取消后快照保留、登记失败、伪造路径、内容损坏、硬摘要直接定位、goal 展示兜底不掩盖安全错误
均有用例。改动 Python 文件 ruff check 通过。扩大测试生成的 MagicMock 临时目录已验证后清理；
无用户历史数据或配置变更，未提交、推送或重启。

继续位置：阶段②证据的当前源文件状态核验、原始用户授权引用及剩余失败/恢复边界；
不能直接跳到阶段③，也不能把未核验来源视为授权。已有快照与恢复包实现无需重复。


## 阶段②增量：文本源文件状态核验

新增 fork 模块 source_evidence.py，核心仅在 read_file 的文本成功出口和 ContextBuilder
恢复包构建处接入；EvidenceRef 增加 source_snapshot，保留原有工具回执快照及指纹语义。

- FileReadResult 是兼容字符串的运行时回执，只接收工具实际解码的 raw 字节指纹，
  不再将读取完成后重新读文件所得的哈希冒充返回内容来源。
- 记录规范化源路径、源字节 SHA-256、字节数、实际返回行范围和全文行数；
  字符上限截掉的行不算已返回，部分读取不标为全文。空文件有独立完整空内容证据。
- source_snapshot 与工具回执 sha256 分开；CRLF 源字节指纹不等于编号后回执文本指纹。
- 只有直接内置 read_file 的带类型回执能生成来源字段，不解析工具文本中自称的
  source_evidence；普通字符串、读去重提示、图片和文档回执不会自动获得来源证明。
- EvidenceRef 序列化和重载保留来源快照；恢复构建时检查当前源文件，只生成瞬时视图，
  不修改历史 metadata，也不持久化“最新”状态，摘要覆盖事务不带文件读取能力。
- 核验必须同时满足当前工作区包含检查、真实 ReadFileTool 路径解析和当前请求
  tool_policy。禁用全部工具、禁用 read_file 或禁止该路径时不读取；额外读取根在此
  不自动获得核验权限。SDK 无真实读取工具时保持未核验。
- 当前内容哈希相等只显示 unchanged_at_check，不声称以后仍新鲜、业务有效或获得授权。
  哈希变化显示 changed；文件不存在显示 missing。权限不足、非法来源、非普通文件、
  检查过程中可检测的修改、超预算等均为 unverified。
- 单文件最多核验 8 MiB，每次恢复包最多读取 16 MiB；同一次构建内同路径复用检查结果。
  预算耗尽不据 mtime 猜测内容。mtime 未变但字节变化仍识别为 changed，
  仅 touch 且字节相同仍可显示核验时未变。
- 对比打开前后路径与文件描述符的设备、文件号、大小和 mtime；ctime 只在同一 API 内
  比较，避免 Windows Python 3.12 stat/fstat 的 ctime 语义差异产生错误失效。
  这不是文件锁，也不承诺检测所有对抗性并发修改或核验后的变化。
- Codex 已发送前缀、原生压缩和幂等生命周期未改；核验结果只用于新构建的恢复视图，
  不回写活跃线程已发送消息。

边界与继续位置：

- 当前源指纹能力限直接内置文本 read_file；图片、PDF/Office、外部工具和历史旧回执
  仍为未核验，不从请求参数或旧 FileStates 的读后哈希补造证明。
- 本节交付时尚未实现原始用户输入引用链，后续增量见下节。InboundMessage 的 channel/sender/metadata、持久化
  role=user 都不能单独证明授权：内部续跑、自动恢复、自动化触发和 SDK 输入须明确区分。
  后续先从真实接收入口保留可核验原文引用，再把代理声明关联到原文；
  即使原文指纹吻合也不自动判定授权语义或允许提交。
- 本节交付时阶段②仍进行中，后续原始输入引用增量见下节，不重复本节文件快照核验。
  阶段③普通 Provider 摘要策略及显式回退尚未开始。

本轮验证：

- source_evidence 新增定向测试：23 passed、1 skipped。
- 最终组合 19 个文件：330 passed、2 skipped，包含此前 16 文件、source_evidence 和
  tests/tools/test_filesystem.py、tests/tools/test_filesystem_tools.py；两项跳过均为 Windows
  无符号链接创建权限，未将其当作已执行通过。
- ContextBuilder/prompt cache：97 passed、1 deselected；只排除前轮已确认的旧模板文案断言，
  未修改该测试或模板，未重新运行已知超时的子代理 fallback 套件。
- 全部累计改动 Python 文件 ruff check、git diff --check 通过；未生成新的 MagicMock 伪目录。
- 未调用真实付费模型；未改用户配置或历史会话，未提交、推送、重启。


## 阶段②增量：原始输入引用与来源分类

新增 fork 模块 input_evidence.py，复用 EvidenceRef 和现有内容寻址快照，
不新增权限授予器，也不把来源校验等同于用户授权。

入口与原文：

- 交互 CLI 在 IDE 附件拼接前捕获不可变 InputReceipt：原始正文、入口类型、
  接收时间和路由字段。排队携带各自的凭据；发送后路由若不一致，降为未核验。
- /continue 保留用户提交的原始命令，生成的“继续上次任务”正文是派生输入，
  不是一条新的用户授权。未带提交凭据的发送不默认为交互用户。
- SDK process_direct 明确标为 sdk；即使 channel=cli、sender_id=user，
  或 metadata 自称 cli_interactive/已授权，也不会升级成真实交互输入。
- 其他尚未接入口凭据的渠道只在 loop 媒体加工前捕获接收到的正文，来源为 unverified。
  不根据角色、渠道名称、sender_id 或模型摘要推断真人身份。
- ReceivedInput 是 fork 内的 InboundMessage 子类，无需修改消息总线数据结构；
  dataclasses.replace 保留不可变原文。媒体提取后的正文标为 derived_input，
  文档中的文字不作为新的用户原话。媒体仅保存定位符，不声称附件字节经过授权核验。
- 内部续跑/跳过用户持久化的消息不生成新的输入证据；自动化/隐藏消息和系统消息
  只作否定性来源分类。已有子代理消息及其他内部路径没有凭据时不能补造用户来源。

保存与引用：

- 每次接收生成独立 receipt_id；同一凭据重复保存得到同一引用，相同文本的两次输入
  仍是不同接收事件。原文不截断，不从摘要反推或恢复已丢失的旧输入。
- 将原文和来源记录保存为独立 JSON 快照；EvidenceRef.kind=input_snapshot，
  input_source 存来源索引，sha256 校验整个快照，text_sha256 单独校验正文。
- 历史消息的 _input_evidence_id 指向该证据，DecisionEntry 复用 evidence_ids 关联。
  不自动推断“哪个原文授权了哪个决定”；关联只是引用，原声明仍 authorization=unverified。
- 输入引用不参与最近 40 条工具证据淘汰，即使模型尚未登记决定也不静默删掉原文入口。
  摘要替换、物理移除被覆盖消息、会话重载后，仍能沿索引定位独立原文。
- 快照先 fsync 保存，历史消息和引用索引再由一次 SessionManager.save(fsync=True) 保存。
  快照失败不登记引用，真实 loop 测试确认不会继续请求模型。
  两个文件不是跨文件事务：快照成功而索引保存前崩溃可留下未登记快照；
  不宣称已解决全部崩溃窗口，也不自动重跑或从孤立文件推断授权。

恢复与能力边界：

- 恢复核验仅从运行时 data_dir、当前 session_key 和内容哈希计算缓存路径，
  不直接追随 metadata 指定的任意路径。核对整个快照、正文、来源索引、凭据 ID 和会话。
- 原文不自动重注入为 system/user 指令。恢复包只显示引用与完整性结果；
  缺失、内容/索引不一致、跨会话错配、路径异常、不可用或超预算都有显式状态。
- 单份快照最多核验 1 MiB，每次恢复视图累计最多读取 4 MiB；大原文完整保存，
  但超预算时标为未核验，绝不截掉原文后声称完整核验。
- snapshot_matches_reference 仅证明读取的快照符合当前索引，不是签名/身份认证，
  不防拥有本地代码或缓存写权限的攻击者同时改写文件和索引。
- 来源记录、原文完整性、授权语义三者分离。诊断不等于修改授权，
  历史同意不等于最近一条消息的 Git 提交授权；本轮没有改变任何执行权限规则。
- 输入索引尚无引用感知回收策略，累计记录仍受恢复包 64,000 字符完整性保护上限约束；
  超限会明确停止，不用删掉旧授权/约束索引来继续。这是当前容量限制，不是无限记忆。

本轮验证：

- 新增 input_evidence 定向测试：32 passed。
- 最终组合 20 个文件：362 passed、2 skipped（前轮 19 文件加 test_input_evidence.py）。
  两项跳过仍是 Windows 符号链接权限限制，未当作已验证通过。
- 输入证据与 IDE bridge/TUI 组合在最后增加排队路由用例前为 61 passed、1 skipped；
  最后新增的排队用例已包含在上述定向及 20 文件组合中。未实测交互式 CLI 界面。
- ContextBuilder/prompt cache：97 passed、1 deselected，沿用已确认的旧模板文案排除项。
- loop_save_turn 的自动化、检查点、崩溃、todo/待处理用户子集：8 passed、45 deselected。
- 全部累计改动 Python 文件 ruff check、git diff --check 通过；
  未运行已知超时的子代理 fallback 全套，未调用真实付费模型。

继续位置：

- 阶段②仍进行中：审查未接入口凭据的渠道、内部恢复/插话等路径的来源边界，
  以及安全失败后是否还有继续模型请求的出口；不把上述增量当作全渠道身份认证。
- 原始输入引用、文本源文件指纹、工具快照、摘要覆盖事务已存在，不重复实现。
- 阶段③仍未开始：普通 Provider 的事务式摘要策略、显式回退和摘要能力隔离待实施。
- 本轮没有修改用户配置、历史会话、FGUI 技能或产物，未提交、推送、重启。


## 最终增量：阶段②收口与阶段③交付

本节是最新状态；前面各增量的“尚未完成/继续位置”仅记录当时断点，不代表最终状态。

阶段②收口：

- 交互 CLI、SDK 的既有入口凭据保持不变。总线插话入队捕获原文，派生媒体文本不取代原文；
  已捕获输入保存不可变凭据，其他渠道仍明确 unverified，绝不根据 sender/role 升级为真人授权。
- 中途插话保存 input_snapshot 和索引；重启可以沿引用核对原文。内部续跑不生成新用户凭据，
  子代理结果保留隐藏系统事件标记；自动化仍与用户输入区分。
- 插话转换或保存失败会将整批输入按原顺序放回队列（早于原有队尾），异常向上停止，
  不再吞成空列表。这里只保证运行时队列和已落盘证据，不宣称所有断电窗口的队列持久性。
- 新策略不合并写回已发送的相邻 user 行，保留独立输入事件及其引用；
  已有凭据的用户原文即使包含 Runtime Context 标记，也不会被当成内部前缀剥离。
- 已确认持续目标只在 max_iterations 时安排内部续跑，安全错误不会自动安排下一轮；
  摘要、原文和证据的失败也不通过“goal 已完成”展示兜底变成成功。

阶段③实现：

- 新增 fork 模块 transactional_context.py，独立拥有普通 Provider 的模型副本与版本。
  每个 runner/子任务实例独立，原始消息列表和会话保存边界不因模型副本缩短而变化。
- AgentDefaults.context_strategy 显式声明 transactional / legacy，默认 transactional；
  通过 AgentLoop.from_config、AgentRunSpec 传入，并同步给子任务。原生 Provider 单独分派，
  仍由 NativeContextPreparation 和 Provider 原生压缩管理活跃线程；不走普通摘要替换。
- 普通模型副本估算超过有效输入预算的 70% 时尝试摘要。没有旧材料可替换时，
  只要完整请求仍在硬预算内就保留原文；关键内容或最新完整批次超限则明确停止。
- 仅覆盖最新完整工具交换之前的 assistant/tool 历史；所有 system/developer/user 消息原样保留，
  最新完整工具交换和其后的插话原样保留。不会把用户原话改写成模型推断的“授权摘要”。
- 摘要请求是一个异步等待屏障：工具批次完成后 await，不与下一批执行工具并行。
  请求包含完整覆盖内容与受保护上下文，输入超预算不先截断。
- 普通摘要请求没有动态工具定义、没有工具执行循环；输出只接受完整 JSON：
  summary + 与输入相同的 source_sha256。拒绝空/超长摘要、错误/截断/未知完成原因、
  工具调用、错指纹和无法减少到硬预算内的候选。
- 先把覆盖原文可靠保存，再请求摘要；提交前重新检查消息、任务 metadata/todo、
  工具定义有无并发变化。版本文档记录原文定位、输入/副本指纹、前一版本引用，
  使用既有内容寻址原子保存与落盘校验；成功后才同步切换模型副本。
- 保存失败、取消、超时、无有效摘要及状态变化均不推进版本，不自动改用 legacy。
  已提交的前一版本不会被后续失败回滚；保存成功但尚未切换时进程退出，只留下可审计版本，
  原始会话仍保留，重启不自动采用孤立摘要或重放工具。
- 摘要等待期间到达的插话在下一次正常模型请求前再次处理，随后重做预算检查。
  摘要调用实际返回的 token usage 纳入本轮累计，不声称摘要是免费操作。
- transactional 下退出固定 24k→14k 软压缩、历史 64 个交换淘汰和超限静默 snip/retry；
  Session.get_history 的 preserve_unconsolidated 路径不按条数/token 丢弃尚未覆盖的消息。
- 新策略保存阶段不再截短工具文本；多模态仍沿用安全持久化表示，完整结构回执已有独立快照。
  大回执按已有证据快照规则卸载，不伪称预览是原文；read_file 保留内联防止引用读取循环。
- 正常请求、纠偏请求和最终回答请求均检查硬输入预算；Provider 返回上下文超限时安全停止，
  不通过插话或最终回答分支偷偷裁剪重试。未知窗口不编造预算，仍由 Provider 错误显式停止。
- 原生 Agent Provider 目前没有经验证的“不可执行摘要”能力，因此本地 Consolidator 的
  tokens/replay/idle 三条摘要路径在原生 Provider 上均拒绝发起摘要调用；必要归档失败保留原文并停止。
  这不是声称 tools=None 关闭了 Codex 原生工具，也没有修改原生线程的执行沙箱。

## 迁移与显式回退

不需要迁移或删除历史数据。本次没有修改用户配置，也没有自动重启。

- 新进程在配置未填写时采用 transactional；示例配置位置：
  agents.defaults.contextStrategy = "transactional"。
- 如确需临时恢复旧的模型副本整形，可由用户显式把同一字段设为 "legacy"，
  然后按正常流程重启或重新创建实例。SDK 可传 AgentLoop(context_strategy="legacy")
  或 AgentRunSpec(context_strategy="legacy")。未知值由配置校验/运行前门禁拒绝。
- legacy 是显式兼容回退，不是出错自动降级。它恢复旧软整形/旧回放预算，
  不关闭阶段①预算诊断、阶段②证据保存、摘要覆盖事务和安全错误保护。
- Codex 原生治理优先于普通策略选择；不会因为改成 legacy 就获得隔离摘要能力，
  也不会改成跨用户轮持久线程或放松 write-once、工具配对、主子执行身份边界。
- 如遇关键约束/最新批次过大、源引用无法验证或无隔离摘要能力，应缩小下一次读取范围、
  切换到经过验证的普通 Provider，或由用户明确调整预算/流程。独立摘要 Provider 接入尚未实现。
  不以清账本、删历史、伪造来源或自动扩大权限恢复。

## 容量、可靠性与验收边界

- 机械指纹、完整字段及工具配对校验不等于业务摘要的语义正确性。
  用户原文、约束与证据引用保留用于人工或后续模型核对，不构成身份认证/授权判定器。
- 暂无自动分段/分层摘要；完整摘要输入无法容纳时停止，而不是截断后宣称覆盖。
  70% 是当前普通副本的触发比例，不是额外固定 token 上限。
- 版本快照是每次 runner 的审计链，不是跨用户轮自动恢复模型副本的持久分支库。
  重启按完整持久会话和已提交会话摘要重建；阶段④持久线程不在这里实现。
- 输入索引/工具快照尚无引用感知垃圾回收与磁盘配额。恢复包 64,000 字符保护上限、
  单原文 1 MiB / 每次核验 4 MiB 上限仍保留；超限标记未核验或停止，不声称无限记忆。
- 原始运行时检查点仍有 best-effort 保存窗口；进程/磁盘同时故障时不保证所有未落盘队列和
  工具回执均可恢复。这不允许自动重放有副作用的原生执行。
- 阶段④未来必须单独授权并验证：真实付费长链、多用户轮线程复用、真实原生自动压缩、
  图片/插话、取消、断线及副作用后的无重复执行、FGUI 全流程验收、实测窗口/费用收益。
  本次只跑离线 mock/stdio 协议模拟，不把累计消耗或检查点大小当作窗口和费用证据。


## 最终验证与交付

- 主组合 23 个文件：480 passed、2 skipped，40.00 秒；在此前 20 文件基础上加入
  test_transactional_context.py、test_loop_save_turn.py、test_runner_injections.py。
  包含普通策略、会话覆盖事务、源文件/输入/工具证据、真实 loop 保存/插话、
  原生上下文模拟、失败/取消/重启、预算和文件工具回归。
- Codex Provider 模拟协议、原生/rebase、会话历史/fsync、配置错误与模型预设组合：
  170 passed。没有调用真实付费模型。
- 扩大 runner 的核心、错误、回退、hook、进度、诊断、reasoning、安全及工具执行：
  123 passed、4 deselected。排除的 4 项均单独使用 git show HEAD 的对应模块在内存中加载，
  不修改磁盘源码，复现相同失败；不是本次修复后的新增回归：
  - test_runner.py::test_subagent_max_iterations_announces_existing_fallback：
    既存 execution_scope 对 MagicMock.aclose_execution 执行 await。
  - test_runner_hooks.py::test_runner_calls_run_level_hooks_on_success：
    旧期望未包含已有 context_input_tokens / context_input_estimated。
  - test_runner_progress_deltas.py::test_runner_emits_write_file_diff_from_tool_execution_snapshots。
  - test_runner_progress_deltas.py::test_runner_marks_file_edit_activity_failed_when_tool_errors。
  这些范围外代码和断言未顺手修复，不能把本次结果称作整个仓库全绿。
- ContextBuilder、prompt cache、IDE bridge/TUI：127 passed、1 skipped、1 deselected。
  仍排除前轮已确认的旧 tool_contract 文案断言；未修改该模板或断言。
- Session history/内部续跑与普通事务/插话组合曾单独验证 107 passed；
  与上述组合存在重叠，以上数字不累加充当唯一用例数。
- Windows 跳过项为符号链接创建权限限制；未视为已验证通过。未实测交互 CLI 界面。
- 全部累计新增/改动 Python 文件 ruff check、git diff --check 通过。
  旧整形/合并/裁剪回归显式选择 legacy，新策略另有失败不推进及原文保护用例；
  成功摘要 mock 明确返回 stop，不用未知完成状态冒充成功。
- 扩大测试生成 MagicMock/.../memory/tool_corrections.json 临时目录，
  已确认启动时不存在于未跟踪清单、仅含测试种子文件、没有跟踪文件或重解析点，
  核验绝对路径后清理；没有修改真实用户缓存、历史数据或工具纠正语料。
- 本次未提交、推送、重启；未修改用户配置、FGUI 技能及任务产物。
  本次授权目标结束，阶段④和上述范围外问题保留为独立后续事项。

## 2026-09-11 后续交付

[体验优化4：证据重读、子任务审计与统计修复](context-evidence-fix-20260911.md) 已完成三项修复的组合验收与收尾（423 passed）；不改变本文阶段④的授权与验收边界。
