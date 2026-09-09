"""CLI 与 IDE 桥接的生命周期和本地命令，避免扩展 Agent 核心。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nanobot.fork.cli.ide_bridge import IDEBridge
from nanobot.fork.cli.ide_context import PendingIDEContext


class IDEIntegration:
    def __init__(self, workspace: Path, tui: Any) -> None:
        self.tui = tui
        self.pending = PendingIDEContext(workspace, self._changed)
        self.bridge = IDEBridge(self.pending, lambda: getattr(tui, "_topic", "") or "未命名")

    def _changed(self, labels: list[str]) -> None:
        setter = getattr(self.tui, "set_ide_context", None)
        if callable(setter):
            setter(labels)
        elif labels:
            self.tui.add_system("IDE 待发送附件：" + "；".join(labels))

    async def command(self, text: str) -> bool:
        parts = text.split()
        if not parts or parts[0] != "/ide":
            return False
        arg = parts[1:]
        try:
            if arg == ["on"]:
                await self.bridge.start()
                self.tui.add_system(
                    f"IDE 接入已开启（实例 {self.bridge.instance_id[:8]}）。"
                    "在 Rider 选中代码后，右键添加到 nanobot 上下文。"
                )
            elif arg == ["off"]:
                await self.bridge.close()
                self.pending.clear()
                self.tui.add_system("IDE 接入已关闭，待发送附件已清空。")
            elif arg == ["clear"]:
                self.pending.clear()
                self.tui.add_system("IDE 待发送附件已清空。")
            elif len(arg) == 2 and arg[0] == "remove":
                self.pending.remove(int(arg[1]))
            elif len(arg) == 2 and arg[0] == "show":
                index = int(arg[1])
                if not 1 <= index <= len(self.pending.items):
                    raise ValueError("附件编号不存在")
                item = self.pending.items[index - 1]
                self.tui.add_system(item.label + "\n" + item.text)
            else:
                state = "开启" if self.bridge.server is not None else "关闭"
                labels = "\n".join(
                    f"{i}. {item.label}" for i, item in enumerate(self.pending.items, 1)
                ) or "无待发送附件"
                self.tui.add_system(
                    f"IDE 接入：{state}\n{labels}\n"
                    "/ide on | off | clear | show 编号 | remove 编号\n"
                    "附件仅随下一条普通问题发送，斜杠命令不消费附件；切换话题清空。"
                )
        except (OSError, ValueError) as exc:
            self.tui.add_system(f"IDE 操作失败：{exc}")
        return True
