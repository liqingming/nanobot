"""宿主子任务管理与持久回执；未知任务通过核验恢复，不能推断已停止。"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.context import ContextAware, RequestContext
from nanobot.agent.tools.schema import IntegerSchema, StringSchema, tool_parameters_schema
from nanobot.fork.agent.subagent_receipts import SubagentReceiptStore
from nanobot.security.workspace_access import current_workspace_scope


class SubagentControlState:
    """归属绑定在创建时；终态回执有界保存，不保留完整任务或模型输出。"""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.store = SubagentReceiptStore(data_dir) if data_dir is not None else None
        self.active: dict[str, tuple[str, str, asyncio.Task, Any]] = {}
        self.finished: OrderedDict[str, tuple[str, str, dict]] = OrderedDict()

    def register(self, task_id: str, session: str, workspace: Path, task: asyncio.Task,
                 status: Any) -> None:
        if self.store is not None:
            self.store.register(task_id, session, workspace, self._receipt(task_id, task, status))
        self.active[task_id] = (session, str(workspace.resolve()), task, status)

    def finish(self, task_id: str) -> None:
        entry = self.active.get(task_id)
        if entry is None:
            return
        session, workspace, task, status = entry
        receipt = self._receipt(task_id, task, status)
        if not task.done():
            raise ValueError("子任务尚未结束，不能提交停止回执")
        if self.store is not None:
            receipt = self.store.finish(task_id, receipt)
        self.active.pop(task_id, None)
        self.finished[task_id] = (session, workspace, receipt)
        while len(self.finished) > 256:
            self.finished.popitem(last=False)

    @staticmethod
    def _receipt(task_id: str, task: asyncio.Task, status: Any) -> dict:
        stopped = task.done()
        reason = status.stop_reason
        if task.cancelled():
            reason = "cancelled"
        elif stopped and task.exception() is not None:
            reason = "error"
        # 清理异常会从执行作用域抛出；不能把 Python 协程结束等同资源已释放。
        cleanup_unverified = stopped and (
            (not task.cancelled() and task.exception() is not None)
            or (status.phase == "error" and status.stop_reason is None)
        )
        return {
            "task_id": task_id,
            "label": status.label,
            "state": "unknown" if cleanup_unverified else "stopped" if stopped else "running",
            "worker_stopped": stopped and not cleanup_unverified,
            "stop_reason": reason,
            "phase": status.phase,
            "iteration": status.iteration,
            "business_success": "unverified",
        }

    async def execute(self, action: str, session: str, workspace: Path,
                      task_id: str | None, timeout_s: int) -> dict:
        owner = (session, str(workspace.resolve()))
        if action == "list":
            for tid, (sess, root, task, _status) in list(self.active.items()):
                if (sess, root) == owner and task.done():
                    self.finish(tid)
            entries = [
                self._receipt(tid, task, status)
                for tid, (sess, root, task, status) in self.active.items()
                if (sess, root) == owner
            ]
            entries.extend(value for sess, root, value in self.finished.values()
                           if (sess, root) == owner)
            if self.store is not None:
                merged = {row["task_id"]: row for row in entries}
                for row in self.store.list(session, workspace):
                    if row["task_id"] not in self.active:
                        merged[row["task_id"]] = row
                entries = list(merged.values())
            return {"tasks": entries, "scope": ("workspace_persistent" if self.store
                                               else "current_process_only")}
        entry = self.active.get(task_id)
        if entry is not None and entry[:2] == owner:
            _, _, task, status = entry
            if action == "cancel" and not task.done() and not task.cancelling():
                task.cancel()
            if action in {"wait", "cancel"} and not task.done():
                # 超时或工具被取消都不能隐式取消目标；仅 cancel 显式发出一次请求。
                await asyncio.wait({task}, timeout=timeout_s)
            if task.done():
                self.finish(task_id)
                return self.finished[task_id][2]
            return self._receipt(task_id, task, status)
        finished = self.finished.get(task_id)
        if finished is not None and finished[:2] == owner:
            if self.store is not None:
                return self.store.inspect(task_id, session, workspace) or finished[2]
            return finished[2]
        if self.store is not None:
            receipt = self.store.inspect(task_id, session, workspace)
            if receipt is not None:
                return receipt
        return {
            "task_id": task_id, "state": "unknown", "worker_stopped": False,
            "recovery_required": True,
            "detail": "没有匹配的宿主回执；核实旧执行者及外部进程后使用恢复命令。",
        }


def should_wait_for_subagent_result(calls: list[Any], results: list[str], events: list[dict]) -> bool:
    """仅单个、成功的显式 wait 且目标仍运行时，把等待交给宿主队列。"""
    if len(calls) != 1 or len(results) != 1 or len(events) != 1:
        return False
    call = calls[0]
    if (call.name != "subagent_control" or call.arguments.get("action") != "wait"
            or events[0].get("status") != "ok"):
        return False
    try:
        result = json.loads(results[0])
    except (TypeError, ValueError):
        return False
    return (
        isinstance(result, dict) and result.get("state") == "running"
        and result.get("worker_stopped") is False
        and bool(call.arguments.get("task_id"))
        and result.get("task_id") == call.arguments["task_id"]
    )


@tool_parameters(tool_parameters_schema(
    action=StringSchema("list/status/wait/cancel", enum=["list", "status", "wait", "cancel"]),
    task_id=StringSchema("Actual task id returned by Nanobot spawn; required except for list"),
    timeout_s=IntegerSchema(description="Integer 0..30 seconds (default 10); timeout returns current status without cancelling the task",
                            minimum=0, maximum=30),
    required=["action"],
))
class SubagentControlTool(Tool, ContextAware):
    """查询和取消仅作用于当前话题的宿主子任务。"""

    def __init__(self, manager: Any):
        self._manager = manager
        self._session: ContextVar[str | None] = ContextVar("subagent_control_session", default=None)

    @classmethod
    def create(cls, ctx: Any) -> Tool:
        return cls(ctx.subagent_manager)

    def set_context(self, ctx: RequestContext) -> None:
        self._session.set(ctx.session_key or f"{ctx.channel}:{ctx.chat_id}")

    @property
    def name(self) -> str:
        return "subagent_control"

    @property
    def description(self) -> str:
        return (
            "List, inspect, wait for, or cancel Nanobot spawn tasks in this session/workspace. "
            "Use the actual returned task id, never a native Codex agent path. "
            "Spawn starts fresh without parent conversation history. "
            "worker_stopped=true is a host cleanup receipt or an explicitly operator-verified "
            "recovery receipt; inspect confirmation_source. Receipts survive restart. "
            "unknown does NOT prove termination. For recovery_required, inspect old worker and "
            "external processes, then with explicit user authorization use "
            "python -m nanobot.fork.cli.subagent_recovery --data-dir <runtime-data-dir> "
            "--workspace <project> --session <session-key> --task-id <id> --confirm-stopped "
            "--reason <verification> --evidence-file <saved-evidence>. "
            "Never assert stopped merely because list is empty; do not delete claims. "
            "Stopped/completed is not business success; verify artifacts separately. "
            "list includes recent terminal receipts; timeout leaves the task running. "
            "For wait/cancel, timeout_s is an integer from 0 to 30 seconds (default 10). "
            "Example: subagent_control(action='wait', task_id='<returned-id>', timeout_s=30). "
            "Use status for an immediate snapshot. Use wait only after binding the task and "
            "finishing independent work: if still running, the host pauses model requests "
            "until a receipt, user input, or a host waiting checkpoint. "
            "Do not repeatedly alternate status with unchanged plan queries, use sleep "
            "commands, or spend model thinking time waiting. A timeout is not task failure."
        )

    async def execute(self, action: str, task_id: str | None = None,
                      timeout_s: int = 10, **kwargs: Any) -> str:
        session = self._session.get()
        if not session:
            return "Error: no session context; cannot manage subagents."
        if action not in {"list", "status", "wait", "cancel"}:
            return "Error: unsupported subagent action."
        if action != "list" and not task_id:
            return "Error: task_id is required."
        if type(timeout_s) is not int or not 0 <= timeout_s <= 30:
            return "Error: timeout_s must be an integer from 0 to 30."
        scope = current_workspace_scope()
        root = scope.project_path if scope is not None else self._manager.workspace
        result = await self._manager._task_control.execute(
            action, session, root, task_id, timeout_s,
        )
        if action == "status" and result.get("state") == "running":
            result["waiting_hint"] = (
                "任务仍在执行。完成绑定和独立工作后调用 wait，由宿主等回执；"
                "不要反复查询未变化的计划，也不要用模型思考或 sleep 命令等待。"
            )
        if result.get("recovery_required"):
            store = self._manager._task_control.store
            result["recovery_scope"] = {
                "data_dir": str(store.root.parent) if store else None,
                "workspace": str(root.resolve()), "session": session, "task_id": task_id,
            }
        if action == "list":
            result["available_slots"] = max(
                0, self._manager.max_concurrent_subagents - self._manager.get_running_count(),
            )
            result["total_slots_including_parent"] = self._manager.max_concurrent_subagents + 1
        return json.dumps(result, ensure_ascii=False)
