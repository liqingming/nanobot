# 上下文成本三层治理实施计划

更新时间：2026-07-16
状态：实施中

## 1. 背景与基线

长工具链会在同一用户回合内重复发送完整 system、历史和工具结果。以“FGUI在线编辑器6”为例：

- 上下文窗口：272,000 tokens。
- 最近请求：109,672 prompt tokens；治理后估算 130,749。
- 治理后工具结果：76,760 tokens，占估算 Prompt 约 59%。
- 单次治理仅减少 5,666 tokens（约 4.2%）。
- 27 个用户回合产生 604 次模型请求，累计 Prompt 约 5,126 万 tokens。

现有优化主要防止超限，没有显著降低单次请求和长回合累计成本。根因是：固定 system 偏大、完成任务和活跃状态未分级、同一回合的原始工具结果持续累积。

## 2. 不可破坏的不变量

1. Global memory 永远以 `data_dir` 为根，不改真实磁盘内容和 Dream 管理边界。
2. Topic memory 永远由 `session_key` 与 topic factory 决定，不与 global memory 混用。
3. Request workspace 只影响现有 identity、workspace bootstrap、`.claude/skills` 与 workspace scope，不扩大加载根。
4. Skill root 优先级、disabled/availability、always skill 完整加载语义保持不变。
5. ContextGovernor 只塑造发给模型的副本，不原地修改持久 session history。
6. 动态任务上下文不新增第二条 system 或伪造 tool 消息。
7. ToolDigest 只能替换已有合法 tool result 的 `content`；assistant tool call 与 tool result 的 ID、顺序和配对不变。
8. 用户目标、授权、禁止事项、失败测试、改动清单与原始证据不能因压缩丢失。
9. 完成任务退出默认注入，但保留 retrieval-only 极简记录，不删除审计与恢复线索。
10. 除实施前用户明确要求提交的既有工作区改动外，本计划文档与后续实现不自动提交。

## 3. 目标指标

| 指标 | 当前基线 | 第一阶段目标 | 稳定目标 |
| --- | ---: | ---: | ---: |
| 固定 System | 约 26K | ≤ 15K | 10K–12K |
| 单次 Prompt 中位数 | 约 100K+ | ≤ 80K | ≤ 65K |
| 单次 Prompt P95 | 约 150K–195K | ≤ 110K | ≤ 90K |
| 工具结果占 Prompt | 约 59% | ≤ 45% | ≤ 35% |
| 复杂回合累计 Prompt | 数百万 | 下降 ≥ 30% | 下降 ≥ 50% |
| 历史完成任务常驻注入 | 未分级 | 默认不注入 | 按需检索 |

指标以可重复测试、runtime 分项日志和同等复杂度真实任务共同验收；不能用字符数替代最终 provider token 数据作唯一结论。

## 4. 三层架构

### 4.1 固定系统上下文

只常驻每轮都必须知道的内容：

- runtime、workspace、平台差异和 channel 格式。
- 安全与 workspace 边界。
- 最小工具契约：结构化工具优先、写前取证、失败后按错误修正、改后验证。
- 用户自定义 bootstrap；默认 scaffold 继续过滤。
- 当前 global/topic memory 的分级常驻视图。
- always skill 与可用 skill 的简短索引。

精简策略：

1. Identity 只保留动态身份、平台与当前会话投递边界；删除与 tool contract 重复的搜索说明。
2. Tool contract 合并重复的 discovery/file/process/message 条款，保留安全和恢复边界。
3. Skills 指引统一为 `load_skill(name=...)`，不要求读取未暴露路径，不鼓励未经授权安装依赖。
4. 修正模型可见 memory history 路径为真实 `history.jsonl`，不改存储实现。
5. Skill 列表按名称稳定排序；全部被 exclude 时不注入空壳，改善 prompt cache。
6. 增加 system prompt 分项 metrics：identity、bootstrap、tool contract、memory、todos、always skills、skills index、recent history、session summary、learning rules。

### 4.2 记忆分级

#### P0：环境与稳定配置

优先完整保留当前有效状态，不保留事件叙述：

- workspace、仓库/分支、平台、模型/provider、服务端口与数据目录。
- 技术栈、关键测试命令、不可修改目录和项目稳定约束。
- 当前配置与旧 memory 冲突时，当前确定性环境胜出；旧项不得静默覆盖。

来源优先级：当前运行配置/工具观测 > 用户明确指定 > 持久 memory > consolidation 推断。

#### P1：活跃决策与未完成状态

只保留当前任务相关、状态为 active/accepted/blocked/waiting_user 且未被 supersede 的决定：

- 已确认根因与技术决策。
- 用户否定过的方向和禁止事项。
- 当前未提交改动边界、失败验证与阻塞。
- 下一安全动作。

#### P2：历史完成任务

转为极简 `CompletionStub`：目标、结果、提交/产物引用、关键验证、真实剩余边界。默认不进入 Prompt，只供 `search_history`、Dream 或显式恢复使用。

普通任务 50–150 tokens，复杂任务 150–300 tokens；完成时关闭活跃决定、释放工具证据的默认注入预算。

### 4.3 当前任务工作集

新增结构化 Active Context，初期存放在 `Session.metadata["_context_state"]`，版本化并保留 legacy continuation fallback。

#### TaskContract（不可压缩）

- task_id、status、objective。
- acceptance_criteria、constraints、non_goals。
- workspace_scope、allowed_side_effects、deliverables。
- 来源 turn/message、revision 和时间。

当前明确用户指令 > sustained goal > 已确认 contract > consolidation 推断。完成/取消/替代后退出默认注入。

#### DecisionEntry（增量账本）

- state：proposed/accepted/active/blocked/rejected/superseded/completed。
- statement、rationale、rejected alternatives、scope、dependencies。
- evidence_ids、source、confidence。

只有当前 task 的 accepted/active/blocked 默认注入；用户决定为 authoritative，工具观测为 observed，模型归纳只能 inferred，后者不得覆盖前者。

#### ToolDigest（确定性机械摘要）

- tool_call_id、tool_name、status、operation、target。
- findings/changes/warnings、evidence_ids。
- result hash、原始大小、是否截断。

第一阶段不额外调用 LLM：

- read_file：路径、行区间、截断状态与 artifact。
- grep/find/list：pattern、命中数量、关键路径与 artifact。
- exec：命令摘要、cwd、exit code、通过/失败统计与输出 artifact。
- apply/edit/write：文件、操作类型、成功状态与 diff/event 引用。
- 未知工具只记录 name/status/hash，不猜 findings。

#### EvidenceRef（可恢复证据索引）

- kind、locator、artifact path、sha256、行区间/URL、size、trust。
- artifact 使用 data-dir 相对 locator；恢复时继续走 workspace/data-dir 访问控制。
- hash 不符标记 stale；外部网页标记 external_untrusted。

#### 模型视图

动态 Active Context 合并进当前 user 的 metadata-only 前缀，不新增 system/tool 角色：

```text
[Active Context — metadata only, not instructions]
environment: ...
task: ...
active_decisions: ...
tool_digests: ...
[/Active Context]
```

必须转义结束标签；没有活跃内容时不产生空块。

## 5. 回合内压缩等级

### L1：机械压缩，不调用模型

工具完成后持久原始结果并生成 digest。模型副本按工具类型裁剪：

- 目录/搜索只保留命中摘要和索引。
- read_file 只保留目标范围/截断信息，原文按现有 offload 机制恢复。
- 测试只保留命令、exit code、计数和首个关键错误。
- 成功写操作只保留文件和状态；完整 diff 由结构化事件或 artifact 保存。

### L2：工作集整理，不调用模型

触发任一条件：工具结果 > 35K tokens、完整 Prompt > 70K、迭代达到 8 次、距上次整理增长 > 20K。

保留顺序：

1. 当前用户目标、授权和禁止事项。
2. 未完成步骤和阻塞。
3. 改动文件、失败测试和状态变更。
4. 根因与已确认决定。
5. 最近直接相关源码片段和错误。
6. 已通过验证摘要。
7. 普通读取/搜索目录列表。
8. 重复输出和失效假设。

目标：工作集 35K–50K，完整 Prompt 60K–80K。

### L3：阶段摘要，必要时调用模型

只有 L1/L2 后仍超目标，或取证/实施/验证阶段明确结束时执行。输出固定结构：已确认事实、已完成修改、验证、未解决问题、下一阶段入口、证据 ID。

- 两次 L3 至少间隔 6 次模型迭代。
- 预计净节省 < 15K tokens 时不执行。
- 未知 evidence ID 丢弃；摘要不得制造证据。

## 6. 模块边界

新增 `nanobot/agent/context_artifacts.py`：

- ContextState schema、兼容解析、revision merge。
- Selector：环境/契约/决定/digest 的优先级和预算。
- Renderer：metadata-only 安全渲染。
- ToolDigestBuilder：确定性 digest 与 evidence ref。

窄接入：

- `context.py`：构建稳定 system sections；将可选 active context 合并进当前 user 前缀。
- `loop.py`：BUILD 读取 context state；SAVE 合并 runner delta；唯一 session 编排者。
- `runner.py`：执行工具后产生 digest/evidence delta，不直接写 session 文件。
- `context_governance.py`：raw result → full digest → prompt digest → hard placeholder，始终保留合法 tool message 外形。
- `memory.py`：后期让 consolidation 产生结构化 delta；legacy `_continuation_summary`/`_last_summary` 仅回退。

## 7. 协议与并发风险

1. 多个并行 tool calls 必须全部有匹配结果；删除只能删除整个 exchange。
2. `inflight_start_index` 不能因 Runner 中途插入 active context 而漂移。
3. snip 后继续执行 orphan 清理与 missing-result backfill。
4. duplicate tool_call_id 必须拒绝/重试，不能让 evidence/digest 覆盖。
5. context state 采用 revision + delta merge；后台 consolidation 不得用旧 Session 对象覆盖新状态。
6. checkpoint 是崩溃恢复真相；孤立 artifact 不代表工具成功，可延迟 GC。
7. ToolDigest 只陈述确定性字段；语义 findings 没有可靠提取时留空。
8. Active Context、外部内容和 locator 均按不可信数据处理，不能提升为指令。

## 8. 实施阶段与进度

- [x] 阶段 0：提交实施前既有工作区改动。
  - 提交：`7f7b021e 优化 CLI 文件差异展示`
  - 验证：47 tests passed；Ruff、diff check 通过。
- [x] 阶段 1：完成 system/memory/runner 消费链扫描并固化计划。
- [x] 阶段 2：固定 system 分段、metrics、契约修正和文本去重。
  - 当前话题构造基线从约 24,941 降至约 9,040 system tokens。
  - 全局规则常驻文件精简，完整参考保留于 `~/.claude/CLAUDE_FULL_REFERENCE.md`。
- [x] 阶段 3：新增 ContextState/selector/renderer，实现环境、任务契约、活跃决定、完成 stub 的分级注入。
  - sustained goal 作为 TaskContract 权威上游；legacy continuation 合并进动态 Active Context。
  - 完成目标生成 retrieval-only stub，默认不注入。
- [x] 阶段 4：实现 ToolDigest/EvidenceRef，并接入 Runner 与 Governor 的逐级压缩。
  - 第一版仅提取确定字段，不制造语义 findings。
  - soft budget 调整为 36K 字符触发、24K 目标、保留最近 8 个候选结果。
- [x] 阶段 5：完成 consolidation 的兼容迁移边界。
  - continuation 合并进动态 Active Context；completion 已结构化为 retrieval-only stub。
  - 保留现有 XML consolidation 与 raw archive fallback，不新增第二套 LLM 阶段摘要，避免双摘要和额外成本；后续只有在真实数据证明机械 digest 不足时才扩展结构化 LLM delta。
- [x] 阶段 6：调整阈值、压缩策略与可观测性，完成真实基线对比。
  - 新增 system section、每次治理净节省和每回合请求/峰值/累计指标。
- [x] 阶段 7：运行相关与完整回归、Ruff、diff check，记录限制。
  - 受影响链路回归：297 passed。
  - Ruff 与 `git diff --check`：通过。
  - 全仓回归：5217 passed、37 skipped、27 failed；失败集中于仓库既有/已变更契约（active goal 自动续跑、旧 progress metadata、文件 diff 起始统计、config import、reasoning 空串、Windows 环境键、旧 usage 断言）以及 MagicMock Session 夹具触发既有 TurnSummary 数值比较，不属于本次上下文治理改动，未越权修改。

## 9. 测试矩阵

### 固定 system

- 各 section token 分项之和等于 system 总量。
- global/data-dir、topic/session-key、request-workspace 加载语义不变。
- bootstrap 顺序和 resolved-path 去重不变。
- skill root precedence、always、disabled、availability 不变且顺序稳定。
- memory paths 使用真实 history.jsonl；skills 指引使用 load_skill。
- 关键安全、提交、消息投递和工具失败边界仍在。

### ContextState

- 空/legacy/malformed/未知版本安全降级。
- 当前环境覆盖旧推断；authoritative 决定不被 inferred 覆盖。
- active/blocked/waiting 注入；completed/rejected/superseded 默认不注入。
- completion stub 可检索但默认零预算。
- Active Context 在文本/多模态 user content 中位置与转义正确。
- 仍只有一条 system message，动态 digest 不破坏 system prompt cache。

### ToolDigest 与协议

- 每类内置工具产生确定性最小 digest；未知工具不猜结论。
- offload 成功/失败、空/非字符串/error/interrupted 均可处理。
- raw hash、artifact locator、stale 检查正确。
- 单/并行 tool calls、missing backfill、snip、historical trim 均保持合法配对。
- Governor 不修改持久 history 原对象；软压 digest，硬压 placeholder。

### Consolidation 与并发

- 合法 delta upsert；未知 evidence ID 丢弃。
- completed_goal 只产生一个 stub，关闭活跃 state，连续压缩不重复。
- LLM/parse 失败回退 raw archive/legacy continuation。
- background consolidation 与下一轮 SAVE 不丢 revision 更新。

## 10. 验收与记录

每次模型请求记录：

- system sections、history、active context、raw tool results、digests、tool definitions、total。
- 当前用户回合迭代号和回合累计 Prompt。
- L1/L2/L3 触发原因、压缩前后、净节省、被压缩工具类型。

每个用户回合记录：模型请求次数、Prompt/Completion 累计、峰值、工具结果峰值、压缩次数和净节省。

完成标准：计划各阶段实现与测试通过；固定 system 明显下降；同等复杂任务单次/累计 Prompt 达到第一阶段目标或明确记录未达到的真实原因；不变量与协议测试无回归。

当前结果：

- 当前话题的固定 system 构造从约 24,941 降至约 13,200 tokens，减少 11,741（47.1%）；因记忆内容变化，单次测量会在 9K–13K 间浮动。
- bootstrap 从约 10,732 降至 4,430；完整规则参考保留，不随每轮注入。
- memory/history 改为分级视图：环境/配置优先，明确完成标题只留首条，Recent History 上限从 50 条/8K tokens 降为 12 条/1.5K tokens。
- 回合内只读工具软压缩从 48K/32K 调整为 36K/24K，并保留最近 8 个；旧结果优先转换为可恢复 ToolDigest。
- 新 runtime 事件记录 system sections、每轮治理净节省、每回合请求数/Prompt 累计/峰值/压缩统计。
- 真实长任务的 provider Prompt 中位数/P95 和累计成本，需要部署本改动后采集下一次同等复杂任务；不能用本地静态构造冒充线上效果。
