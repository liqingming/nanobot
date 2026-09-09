# Rider → nanobot 代码上下文

实现方式：Rider 前端插件主动获取选区快照 → 当前用户的本机回环接口 → CLI 待发送附件 → 用户提交普通问题时加入模型上下文。不需要 ReSharper 后端、网关或额外 Python 依赖。

## 安装与使用

1. 使用本仓库更新后的 nanobot，重启旧 CLI 进程。
2. Rider：Settings → Plugins → 齿轮 → Install Plugin from Disk，选择
   `integrations/rider/build/nanobot-context-0.1.1.zip`，按 IDE 提示重启。
3. 在 Rider 的项目目录内打开内嵌终端，启动 `nanobot`。输入 `/ide on`。
   可先用 `/rename 名称` 给会话命名，便于区分多个终端。
4. 在编辑器选中连续代码段，右键“添加到 nanobot 上下文”，或按 Ctrl+Alt+N。
   快捷键冲突时在 Settings → Keymap 搜索 Nanobot 修改。
5. 只有一个符合当前项目及文件范围的有效接入会话时直接添加，不弹窗；多个候选时选择目标终端/话题。
   弹窗显示话题名、会话键、实例短 ID、工作区和待发送附件数量。
6. 终端输入框上方出现文件及行号，例如 `玩家.cs:10–12`。
   输入“分析这里的问题”再按 Enter，问题和快照一起发送。

不会自动聚焦终端、自动提交问题、读取整个文件或上传整个项目。
代码来自编辑器 Document，可包含尚未保存的修改；模型看到的是选取时的快照。
本机已使用 Rider 2024.2.6 SDK 编译，要求 IntelliJ Platform 242+ / Java 21。
其他 Rider 版本未作实机验证。插件仅使用平台 API，未声明其他产品实测兼容性。

## 终端命令

- `/ide`：接入状态、附件编号及命令帮助。
- `/ide on`：开启当前 CLI 实例的本机接收接口，重复执行不会创建第二个服务。
- `/ide show 1`：查看第 1 个待发送快照全文。
- `/ide remove 1`：移除第 1 个附件。
- `/ide clear`：清空附件。
- `/ide off`：关闭接入、删除登记文件并清空附件。

默认关闭，每次新启动需主动开启。关闭和重新开启会更换凭证。
Textual 后端有持续可见的附件栏；旧 prompt_toolkit 后端用系统提示和 `/ide` 列表展示。

## 生命周期与边界

- 每个终端有随机实例 ID、独立端口、令牌和内存附件队列。
- 插件只列出 IDE 项目根目录或其子目录的 CLI 工作区，且所选文件必须在该工作区内。
  如果终端工作区是 IDE 项目根目录的父目录或其他目录，不会列出；请切到项目目录后启动。
- 切换话题或清空上下文会清空待发送附件；不会把旧话题的附件带到新话题。
- 弹窗打开后话题变化、附件被清空或已提交：旧选择返回 HTTP 409，要求重新添加。
- 普通问题在提交时消费附件。忙碌时的问题先冻结快照再排队，不会混入随后新增的选区。
- 所有斜杠命令均不消费附件。空 Enter 不发送附件。
- 已发送快照按普通用户消息保存在 nanobot 会话历史中；`/ide clear` 只清除尚未发送的附件。
- 用户取消当前请求时，沿用 CLI 原有的排队消息清理规则；已排队问题及其快照可能被丢弃。
- 待发送附件仅存在内存中，终端退出/崩溃后不会恢复。插件取消目标选择不会传送代码。
- 单选区最多 64 KiB UTF-8，最多 8 个附件，总计 256 KiB；不支持多光标/矩形选区。
- 支持已有本地路径的未保存修改，不支持没有本地文件路径的临时 Scratch/远程文件。
- 拒绝目录越界、路径中的控制字符及指向工作区外的符号链接。
- 请求 ID 在同一待发送代次去重；发送失败不会自动重试或悄悄改投其他会话。

## 安全设计

- 仅监听 `127.0.0.1` 随机端口；不监听局域网地址。
- 所有接口校验随机 Bearer 凭证、固定 Host；拒绝 Origin，不提供 CORS。
- Java 客户端禁用系统代理和重定向，凭证不会进入目标标签。
- 登记位于 `~/.nanobot/ide-bridge/<实例ID>.json`，包含本机端口、工作区和令牌。
  不要分享或提交这些文件。Unix 新文件权限 0600、目录 0700；
  Windows 继承用户目录 ACL，需确保其他用户不能读取该目录。
- 不把本机同一用户下的恶意进程当作隔离边界；它们通常已能读取相同项目及凭证。
- Python 接口请求头上限 8 KiB、请求体上限 400 KiB、处理时限 5 秒、最多 16 个连接。
- 正常退出自动清理登记。崩溃残留文件由插件探测并忽略；必要时可在所有 CLI 退出后手动清理。
- 快照以 JSON 编码作为用户消息的数据区传入，标明不是系统指令；这不替代模型侧的不可信内容防护。

## 离线构建

从仓库根目录执行（替换实际 Rider 路径）：

```powershell
python integrations/rider/build.py --rider-home "C:\Program Files\JetBrains\JetBrains Rider 2024.2.6"
```

直接用 Rider 自带 JBR 的 javac 和 lib/*.jar 编译，再按标准插件 ZIP 结构打包，
不下载 Gradle 或 SDK，不向安装目录写入，也不包含 JetBrains SDK JAR。
若 JBR 不含 javac，额外传 `--jdk-home <JDK21目录>`。
产物在插件目录的 `build/nanobot-context-0.1.1.zip`，已由仓库 build/ 忽略规则排除。
每次编译使用独立临时 classes 目录，不混入旧 class。

## 自动验证与人工验收

```powershell
python -m pytest tests/fork/test_ide_bridge.py tests/fork/test_ide_context_tui.py tests/cli/test_tui_textual_pilot.py -q
python integrations/rider/smoke_test.py --rider-home "C:\Program Files\JetBrains\JetBrains Rider 2024.2.6"
```

Java 联调使用真实插件 BridgeClient 和真实 Python TCP 服务，验证发现、中文快照、
请求重传去重、过期话题拒绝、跨项目过滤。不会调用模型或打开 Rider 窗口。

人工验收：
1. 修改 C# 文件但不保存，选区添加后用 `/ide show 1` 检查路径、闭区间行号和内容。
2. 添加后不提问，确认模型不会开始响应；提交问题后确认模型得到快照。
3. 同项目开两个终端并分别命名，确认只有选中的终端收到附件。
4. 打开目标弹窗后在终端切换话题，旧选择应被拒绝。
5. 测试移除、清空、切换话题、斜杠命令和退出清理。
6. Rider 的菜单、快捷键与模型最终接收效果仍需安装插件后实测；
   自动测试不等同于这部分 GUI 验收。

## 参考依据

- JetBrains 官方编辑器选区 API：
  https://plugins.jetbrains.com/docs/intellij/working-with-text.html
- JetBrains 官方 Rider 插件开发与前后端职责：
  https://plugins.jetbrains.com/docs/intellij/rider.html
- 实际编译依据：本机 Rider 2024.2.6 的 IntelliJ Platform SDK。
