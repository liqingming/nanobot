---
name: upstream-sync
description: 从上游原始仓库同步代码到 fork 仓库。先全局分组判断，再按"相同功能/新功能/bug 修复"分类应用对应原则，逐提交审查并合入。
version: 1.1.0
---

# Upstream Sync — 上游代码同步

从 fork 的上游仓库（upstream）将新提交合入本地 fork，逐提交审查变更内容，选择合入策略，处理冲突。

## When to Use

- 用户说"同步上游""合入上游代码""merge upstream""sync upstream"
- 用户说"看看上游有什么新东西"
- 用户说"把上游的某个提交 cherry-pick 过来"

## Prerequisites

- Git 仓库已配置 `upstream` remote（`git remote add upstream <url>`）
- 工作区干净（无未提交的改动）或已 stash
- 知道上游主分支名（通常是 `main` 或 `master`）

## 核心概念

```
upstream/main  ──●──●──●──●──●──●──●  (上游最新，1096 个新提交)
                 \
local main       ●──●──●──...──●──●  (你的 fork，28 个自定义提交)
                       merge-base ─┘
```

- **merge-base**：两个分支最后的公共祖先
- **落后提交数**：`git rev-list --count main..upstream/main`
- **你的自定义提交**：`git log upstream/main..main`（不会被覆盖）

## 同步原则（核心决策框架）

逐提交审查时，先把每个上游提交归入下面 4 类，再按对应原则处理。**这是 sync 的核心决策依据，不要跳过这一步直接 cherry-pick**。

### 分类 1：相同功能的不同实现（你已经做过这件事）

**检测**：提交改的功能 / 文件，你的 fork 也改过。用 `git log --oneline upstream-merge-base..main -- <file>` 检查。

**处理**：
1. **逐行比较两边实现**：把上游版本和我们版本都给 AI 评估优劣（性能、健壮性、可读性、覆盖边界等）
2. **如果我们明显更好** → **自动保留我们的实现**（`git checkout --ours <file>`），然后看上游同提交里**其它文件**的改动是不是要顺带改我们的代码以维持一致（比如改 signature、改调用点）
3. **如果上游明显更好** → **自动用上游版本**（`git checkout --theirs <file>`），然后检查这个改动是否破坏了我们其它代码的依赖
4. **如果差不多** → **不要自动决定，给用户对比表让用户选**：
   ```
   文件: nanobot/agent/X.py
   功能: Y
   上游版本: <要点 3 条>
   我们版本: <要点 3 条>
   建议: <两边各自的 trade-off>
   选哪个？ A=上游 / B=我们 / C=我整合
   ```

### 分类 2：全新功能（我们没做过）

**检测**：提交引入新文件 / 新模块 / 新工具，我们的 fork 历史里完全没有同类改动。

**处理**：
1. **直接 cherry-pick 合入**
2. **合入后立即检查接口兼容性**：grep 新功能的入口符号（class / function），看我们之前的代码是不是要适配
   - 例：上游新加了 `LoadSkillTool`，我们的 `format_tool_hint` 是否要识别它？我们的 `_HINT_KEY_PRIORITY` 是否要加 "skill_name"？
3. **如果有兼容问题** → 在 sync 分支上加适配 commit（不要改 upstream 提交本身）
4. **不兼容也不要直接放弃** → 跟用户报告：上游 X 改动需要适配我们的 Y、Z 才能用，是否做适配？

### 分类 3：bug 修复（fix / bug / repair / 修复）

**检测**：提交 message 含 fix/bug/修复/regression 等关键词；或 diff 看着是修小问题（条件改、边界值改、错误处理加固）。

**处理（同分类 1）**：
1. **我们没修过这个 bug** → 直接合入
2. **我们已经独立修过** → 比较两边实现，按分类 1 规则处理（我们好就留我们，差不多让用户选）
3. **修的是上游代码里我们没动过的部分** → 直接合入，但跑测试确认不影响我们其它改动

### 分类 4：无关变更（文档 / CI / 重构与我们无重叠）

**处理**：
- 文档 / CI / lint 配置 → 直接 cherry-pick（一般安全）
- 重构了我们没碰过的模块 → 直接 cherry-pick
- **唯一例外**：如果改动量大且影响接口（如大规模 rename），即使我们没碰过也要扫一下我们的代码是否引用了被改名的符号

## Procedure

### 第 0 步：创建同步分支（必须，不可跳过）

**同步操作永远不直接在主干分支上进行。** 先在独立分支上完成所有合入和测试，验证通过后再覆盖主干。

```bash
# 从当前主干创建一个同步工作分支
git checkout main
git checkout -b sync-upstream-YYYYMMDD

# 确认当前在同步分支上
git branch --show-current
# 输出: sync-upstream-YYYYMMDD
```

**为什么必须这样做**：
- 主干分支保持不动，随时可以回退
- 如果合入过程搞砸了，`git checkout main && git branch -D sync-upstream-YYYYMMDD` 即可重来
- 可以在同步分支上随意 force push 测试，不影响主干
- 只有用户确认"没问题"后，才执行最后的覆盖操作

### 第 1 步：Fetch & 评估差异

```bash
# 拉取上游最新
git fetch upstream

# 确认远程配置
git remote -v

# 查看落后多少提交
git rev-list --count main..upstream/main

# 找到分叉点
git merge-base main upstream/main
```

**输出给用户看**：落后提交数、merge-base、上游最新提交的摘要。

### 第 1.5 步：全局判断与分组（落后 >20 提交时必做）

在逐提交之前先做整体扫描，把上游所有新提交按类型/影响范围分组，给用户全局视图。这一步是同步原则分类 1-4 的前置准备。

**扫描 1：按提交 message 关键词分类**

```bash
# 全新功能
git log --oneline main..upstream/main --grep="^feat"

# bug 修复
git log --oneline main..upstream/main --grep="^fix\|bug\|修复"

# 重构
git log --oneline main..upstream/main --grep="^refactor"

# 文档/CI/lint
git log --oneline main..upstream/main --grep="^docs\|^chore\|^ci\|^style"

# 测试
git log --oneline main..upstream/main --grep="^test"
```

**扫描 2：按改动目录分组**

```bash
# 列出本次同步涉及的所有目录（按改动量排序）
git diff --stat main..upstream/main | awk '{print $1}' | grep "/" | \
  awk -F'/' '{print $1"/"$2}' | sort | uniq -c | sort -rn
```

**扫描 3：标识跟我们改动重叠的提交（高优先级审查）**

```bash
# 我们的自定义文件（必看清单）
git diff --name-only upstream/main..main > /tmp/our_files.txt

# 找出上游提交里也碰过这些文件的（高冲突风险）
for f in $(cat /tmp/our_files.txt); do
  count=$(git log --oneline main..upstream/main -- "$f" | wc -l)
  [ "$count" -gt 0 ] && echo "$count commits touch $f"
done | sort -rn
```

**输出分组报告给用户**，类似：

```
本次同步 1096 个提交，按维度统计：

按类型：
  - feat:    523 (其中 47 改了我们改过的文件 → 高冲突风险，需详细审查)
  - fix:     312 (其中 18 重叠 → 重点比较实现)
  - refactor: 89 (其中 12 重叠)
  - docs:    104 (基本无重叠)
  - test:     68

按影响目录（重叠度倒序）：
  - nanobot/providers/         (我们改过 8 个文件，上游改了 23 个文件，6 个重叠)
  - nanobot/agent/             (我们改过 12 个文件，上游改了 41 个文件，9 个重叠)
  - nanobot/cli/               (我们改过 5 个文件，上游改了 17 个文件，3 个重叠)
  - nanobot/webui/             (我们没改过，上游引入大量新代码 → 分类 2 直接合)

建议合入顺序（风险倒序）：
  1. 先处理 docs/CI/test（低风险，建立信心）
  2. 再处理我们没碰过的目录的 feat/fix（分类 2/3）
  3. 最后处理重叠目录的提交（分类 1，最耗时）

你想先从哪一组开始？
```

**注意**：必须让用户先确认分组策略再开始逐提交审查，不要自己挑顺序开干。

### 第 2 步：列出待审查的提交

按需选择展示范围：

```bash
# 列出所有上游新提交（数量大时很慢，建议分批）
git log --oneline main..upstream/main

# 只看最近 N 个
git log --oneline main..upstream/main -20

# 按日期范围
git log --oneline --since="2026-05-01" main..upstream/main

# 按文件/目录过滤（只看影响某模块的提交）
git log --oneline main..upstream/main -- nanobot/cli/
```

**重要**：如果落后提交很多（>50），**先向用户展示摘要**，让用户决定审查范围：
- 按时间分批（最近一周/最近一月/全部）
- 按模块过滤（只看影响特定目录的提交）
- 按关键词搜索（只看包含某个关键词的提交）

### 第 3 步：逐提交审查 + 分类决策

对每个候选提交，按下面顺序判断：

**步骤 A：看提交基本信息**

```bash
git show --stat <commit-hash>     # 文件列表 + 改动量
git show <commit-hash>            # 完整 diff
```

**步骤 B：判断归类**（按"同步原则"4 大分类之一）

```bash
# 检查这个提交改的文件，我们的 fork 是否也改过
files=$(git show --name-only --pretty="" <hash>)
for f in $files; do
  our_commits=$(git log --oneline upstream/main..main -- "$f" | wc -l)
  [ "$our_commits" -gt 0 ] && echo "OVERLAP: $f (we have $our_commits commits)"
done
```

判断决策：
- 输出有 `OVERLAP` → **分类 1 或 3**（相同功能 / bug 修复重叠）
- 全新文件（我们没有）+ 提交 message 是 feat → **分类 2**（全新功能）
- 提交 message 是 fix/bug 但我们没改过 → **分类 3**（无重叠的 bug 修复，直接合）
- 都不是 → **分类 4**（无关变更）

**步骤 C：按分类应用原则**

**对分类 1（相同功能重叠）和分类 3（重叠的 bug 修复）**：

```bash
# 对每个 OVERLAP 文件，并排显示两边版本片段供 AI 评估
git show <upstream-hash>:<file> > /tmp/upstream_version.txt
cat <file> > /tmp/our_version.txt
diff -u /tmp/our_version.txt /tmp/upstream_version.txt
```

AI 评估时**必须输出明确判断**：
- "我们的实现明显更好（理由：xxx）" → 自动 `git checkout --ours`
- "上游明显更好（理由：xxx）" → 自动 `git checkout --theirs`
- "两边差不多（各有优缺点）" → 输出对比表，**让用户选 A/B/C**

**对分类 2（全新功能）**：

```bash
# 直接 cherry-pick
git cherry-pick <hash>

# 立即扫接口兼容性
new_symbols=$(git show <hash> | grep "^+.*def \|^+.*class " | awk '{print $3}' | cut -d'(' -f1)
echo "新引入符号: $new_symbols"
echo "在我们的代码里有无相关依赖需要适配："
for sym in $new_symbols; do
  grep -rn "$sym" nanobot/ --include="*.py" | head -3
done
```

如果发现需要适配，**在 sync 分支单独加适配 commit**，不要改 upstream 提交本身。

**对分类 4（无关变更）**：

```bash
git cherry-pick <hash>     # 一般安全
```

**步骤 D：冲突预判**

```bash
# 模拟 cherry-pick 不实际执行
git cherry-pick --no-commit <hash> 2>&1; git cherry-pick --abort
```

CONFLICT 输出后回到步骤 C 按分类决策。

### 第 4 步：选择合入策略

按第 3 步的分类决策，对应到具体命令：

| 分类 | 策略 | 命令 | 是否自动 |
|------|------|------|---------|
| **2（全新功能）** | cherry-pick + 检查兼容性 | `git cherry-pick <hash>` 然后 grep 新符号 | 自动 |
| **3（无重叠 bug 修复）** | cherry-pick | `git cherry-pick <hash>` | 自动 |
| **4（无关变更）** | cherry-pick | `git cherry-pick <hash>` | 自动 |
| **1/3 重叠 + 我们好** | 留我们的实现 | `git checkout --ours <file>` 然后 `git add` + `--continue` | 自动 |
| **1/3 重叠 + 上游好** | 用上游 + 适配 | `git checkout --theirs <file>` + 适配 commit | 自动 |
| **1/3 重叠 + 差不多** | **问用户** | 输出对比表，等用户选 A/B/C | **必须人工** |
| **彻底跳过** | 不操作 | 不需要的上游功能 | 报告原因，等用户确认 |

**重要**：
- "差不多"和"彻底跳过"两类**必须先报告等用户决定**，不要自己拍板
- 其它情况可以连续自动处理，但**每 5-10 个提交向用户汇报一次进度**，让用户随时叫停
- **批量合入**（`git cherry-pick <hash1>..<hashN>`）只有用户明确说"这一组全合"才用

### 第 5 步：处理冲突

当 cherry-pick 报 CONFLICT 时：

```bash
# 查看冲突文件列表
git diff --name-only --diff-filter=U

# 查看冲突内容
git diff

# 用编辑器手动解决冲突后：
git add <resolved-files>
git cherry-pick --continue

# 如果搞砸了，放弃本次 cherry-pick：
git cherry-pick --abort
```

**冲突解决原则**（提醒用户）：
1. **用户自定义改动的优先级 > 上游改动**（除非上游修复了严重 bug）
2. 如果上游新增了功能而你改了同一区域 → 两边保留，手动整合
3. 如果只是格式/注释差异 → 用上游版本保持一致性
4. 不确定时 → 保留冲突标记，让用户手动决定

### 第 6 步：验证

合入完成后，在同步分支上验证：

```bash
# 检查提交历史
git log --oneline -10

# 确认工作区干净
git status

# 如果项目有测试，运行测试
# （根据项目实际情况执行对应测试命令）

# 如果有构建，试构建
# （根据项目实际情况执行对应构建命令）
```

⚠️ **此阶段务必等待用户确认"测试通过"**，不要自动执行第 7 步。

### 第 7 步：覆盖主干分支（用户确认后才能执行）

用户确认同步分支上的改动没问题后，用同步分支覆盖主干：

```bash
# 切回主干
git checkout main

# 用同步分支强制覆盖主干（保留同步分支上的所有提交历史）
git reset --hard sync-upstream-YYYYMMDD

# 确认主干已更新
git log --oneline -5

# 推送到远程（如果需要）
# git push origin main --force-with-lease
```

**如果用户说"还不行"或"有问题"**：
- 切回主干继续在同步分支上修：`git checkout sync-upstream-YYYYMMDD`
- 修完后重复第 6 步验证，再回到第 7 步
- 如果彻底放弃：`git checkout main && git branch -D sync-upstream-YYYYMMDD`

### 第 8 步：记录同步状态

更新 MEMORY.md：

```markdown
## 上游同步记录
- 2026-05-22: 同步 upstream/main (HKUDS/nanobot)，审查并合入 X 个提交，跳过 Y 个，解决 Z 个冲突
- 上次 merge-base: <hash>
```

## Pitfalls

- ⚠️ **不要用 `git merge upstream/main`**：会把上游所有改动一次性合入，失去逐提交审查的机会。除非用户明确要求且落后提交很少。
- ⚠️ **不要用 `git rebase upstream/main`**：会改写你所有自定义提交的 hash，如果已经 push 过会导致 force push 问题。只在用户明确要求且理解后果时使用。
- ⚠️ **cherry-pick 保持顺序**：Cherry-pick 应按提交时间顺序操作，否则可能产生不必要的冲突。
- ⚠️ **大数量分批**：如果落后 > 100 个提交，不要一次性全列出来。先按时间/模块分组，让用户选择审查批次。
- ⚠️ **Windows 换行符**：Windows 上 `git diff` 可能显示大量 CRLF 变更，实际只是换行符差异。用 `git diff --ignore-cr-at-eol` 过滤。
- ⚠️ **不要跳过全局分组（第 1.5 步）**：落后 >20 提交时直接逐提交是低效且容易漏视野。先分组再决策能 5 倍提速。
- ⚠️ **"差不多"不要自己拍板**：分类 1/3 重叠时如果两边各有 trade-off，**必须输出对比表给用户选**。AI 自作主张选一边后用户事后看到对手版本可能后悔，回头改成本高。
- ⚠️ **接口适配单独 commit**：分类 2 合入新功能后做的适配改动，必须单独 commit，不要 squash 进 upstream 提交。否则后续再 sync 时 diff 看不清谁是上游、谁是我们加的。
- ⚠️ **每批汇报进度**：连续自动 cherry-pick 5-10 个后停下汇报，否则一旦中间出错用户不知道在哪一步，回退成本大。

## 快捷场景

### 场景 A：只想看上游最近改了啥（不合入）

```bash
git fetch upstream
git log --oneline main..upstream/main -20
```

### 场景 B：只想合入某个特定提交

```bash
git fetch upstream
git show <hash>              # 审查
git cherry-pick <hash>       # 合入
```

### 场景 C：落后太多，想分批处理

```bash
# 第一轮：最近一周的提交
git log --oneline --since="2026-05-15" main..upstream/main
# 逐提交审查...

# 第二轮：下一周
git log --oneline --since="2026-05-08" --until="2026-05-15" main..upstream/main
# ...
```

### 场景 D：只想看某模块的变更

```bash
# 只看 nanobot/cli/ 目录的提交
git log --oneline main..upstream/main -- nanobot/cli/

# 只看某文件的变更历史
git log --oneline main..upstream/main -- nanobot/agent/memory.py
```

## Verification

- `git status` 干净
- `git log --oneline -5` 能看到新合入的提交
- 测试通过（如有）
- MEMORY.md 已更新同步记录
