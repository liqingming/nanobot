---
name: upstream-sync
description: 从上游原始仓库同步代码到 fork 仓库。逐提交审查变更、选择合入策略、解决冲突。
version: 1.0.0
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

### 第 3 步：逐提交审查

对每个候选提交，执行：

```bash
# 查看提交详情（完整 diff）
git show <commit-hash>

# 只看文件列表（快速评估影响范围）
git show --stat <commit-hash>

# 只看 diff（不含提交信息）
git diff <commit-hash>^..<commit-hash>
```

**审查要点**（帮用户判断）：
- 🔴 是否影响了你自定义过的文件？（`git diff --name-only <hash>^..<hash>` 和你的改动文件对比）
- 🟡 是不是纯粹的新功能/修复，跟你改的东西不重叠？
- 🟢 是不是文档/注释/格式类改动？（通常可以安全合并）
- ⚠️ 是不是重构了你改过的代码？（高冲突风险，需仔细审查）

**检查冲突预判**：

```bash
# 模拟 cherry-pick，不实际执行（Windows 上用 PowerShell）
git cherry-pick --no-commit <hash> 2>&1; git cherry-pick --abort
```

如果输出有 "CONFLICT"，说明这个提交跟你的改动冲突。

### 第 4 步：选择合入策略

审查完一个提交后，向用户给出建议并确认：

| 策略 | 命令 | 适用场景 |
|------|------|---------|
| **cherry-pick** | `git cherry-pick <hash>` | 提交独立、与你改动不冲突 |
| **跳过** | 不操作 | 不相关（纯上游文档/CI/你不需要的功能） |
| **手动合并** | 先 cherry-pick，解决冲突后 `git add` + `git cherry-pick --continue` | 提交跟你有冲突，需要保留两边改动 |
| **直接用你的版本** | 冲突时 `git checkout --ours <file>` | 上游改动不如你的好 |
| **直接用上游版本** | 冲突时 `git checkout --theirs <file>` | 你的改动不再需要 |

**批量合入**：如果一组连续提交都安全，可以用 `git cherry-pick <hash1>..<hashN>` 批量操作。但**只有用户明确要求且确认安全时才用**。

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
