"""Todo tool for tracking multi-step tasks within a session."""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.tools.context import RequestContext
    from nanobot.bus.queue import MessageBus
    from nanobot.session.manager import SessionManager


_STATUS_SYMBOLS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
_VALID_STATUSES = ("pending", "in_progress", "completed")
MAX_TODO_ITEMS = 50


def format_todos(items: list[dict[str, Any]]) -> str:
    """Render todo list as plain-text checklist."""
    if not items:
        return "(no active todos)"
    lines = []
    for item in items:
        sym = _STATUS_SYMBOLS.get(item.get("status", ""), "[?]")
        lines.append(f"{sym} {item.get('content', '')}")
    return "\n".join(lines)


def format_todo_diff(
    prev: list[dict[str, Any]],
    curr: list[dict[str, Any]],
) -> str | None:
    """Render a human-readable description of how the todo list changed.

    Returns None when there is no meaningful change (caller should not display
    anything). Cases handled:

      * empty → non-empty  : "📋 计划 (N 步)" + full checklist
      * non-empty → empty  : "📋 计划已清空"
      * all newly completed: "✅ 全部 N 步完成"
      * partial changes    : per-item lines (✅ done / ⚡ started / ➕ added /
                             ↩ reverted / ➖ removed) + "📊 进度 done/total"
    """
    prev_by_content = {
        (it.get("content") or ""): it.get("status")
        for it in prev
        if it.get("content")
    }
    curr_by_content = {
        (it.get("content") or ""): it.get("status")
        for it in curr
        if it.get("content")
    }

    # First creation: show the full plan so the user sees what was decided
    if not prev and curr:
        lines = [f"📋 计划已制定 ({len(curr)} 步):"]
        for item in curr:
            sym = _STATUS_SYMBOLS.get(item.get("status", ""), "[?]")
            lines.append(f"  {sym} {item.get('content', '')}")
        return "\n".join(lines)

    if prev and not curr:
        return "📋 计划已清空"

    if not curr:
        return None

    # Whole-plan completion: show summary instead of per-item diffs
    all_done_now = all(it.get("status") == "completed" for it in curr)
    all_done_before = bool(prev) and all(
        prev_by_content.get(c) == "completed" for c in curr_by_content
    )
    if all_done_now and not all_done_before:
        return f"✅ 全部 {len(curr)} 步完成"

    # Per-item diff
    lines: list[str] = []
    for item in curr:
        content = item.get("content") or ""
        if not content:
            continue
        curr_status = item.get("status")
        prev_status = prev_by_content.get(content)
        if prev_status is None:
            lines.append(f"  ➕ 新增: {content}")
        elif prev_status != curr_status:
            if curr_status == "completed":
                lines.append(f"  ✅ {content}")
            elif curr_status == "in_progress":
                lines.append(f"  ⚡ {content}")
            elif curr_status == "pending":
                lines.append(f"  ↩ 重置: {content}")

    for content in prev_by_content:
        if content and content not in curr_by_content:
            lines.append(f"  ➖ 删除: {content}")

    if not lines:
        return None

    done = sum(1 for it in curr if it.get("status") == "completed")
    total = len(curr)
    lines.append(f"  📊 进度: {done}/{total}")
    return "\n".join(lines)


class TodoWriteTool(Tool):
    """Overwrite the session's active todo list.

    The LLM should call this whenever the task plan changes — adding new steps,
    marking one in_progress, or completing a step. Each call replaces the entire
    list, so always pass the full current state.

    Enforced invariants:
      * at most one item may be ``in_progress`` at a time;
      * ``completed`` is terminal — a completed item cannot be reverted to
        pending/in_progress (delete & re-add if needed);
      * each item's ``content`` is the identity key for matching across calls;
      * items count must not exceed ``MAX_TODO_ITEMS`` (default 50).

    Timestamps are managed by the tool, not the LLM:
      * ``created_at`` — first time the content appears in the list;
      * ``started_at`` — first transition into ``in_progress``;
      * ``completed_at`` — first transition into ``completed``.
    """

    def __init__(self, sessions: "SessionManager", bus: "MessageBus | None" = None):
        self._sessions = sessions
        self._bus = bus
        self._channel = "cli"
        self._chat_id = "direct"
        self._session_key = "cli:direct"
        # Track per-session "done count delta" between consecutive execute()
        # calls so summarize_result can show e.g. "+2" newly completed.
        self._done_count: dict[str, int] = {}
        self._last_done_delta: dict[str, int] = {}

    def set_context(self, ctx: "RequestContext") -> None:
        self._channel = ctx.channel
        self._chat_id = ctx.chat_id
        self._session_key = ctx.session_key or f"{ctx.channel}:{ctx.chat_id}"

    @property
    def name(self) -> str:
        return "todo_write"

    @property
    def description(self) -> str:
        return (
            "Manage a structured todo list for multi-step work. "
            "Each call OVERWRITES the entire list — always pass the full current state "
            "of all todos, not just the ones that changed. "
            "Status values: 'pending' (not started), 'in_progress' (active now), "
            "'completed' (done, terminal — cannot be reverted). "
            "At most one item may be in_progress at a time. "
            f"Up to {MAX_TODO_ITEMS} items per list. "
            "Use when a task has 3+ ordered steps, spans multiple turns, or when the "
            "user explicitly asks for a plan. Pass an empty list to clear the plan."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Complete list of todos in execution order. Pass [] to clear.",
                    "maxItems": MAX_TODO_ITEMS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Short, imperative task description.",
                                "minLength": 1,
                            },
                            "status": {
                                "type": "string",
                                "enum": list(_VALID_STATUSES),
                                "description": "Current state of this task.",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["items"],
        }

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        """Produce a compact one-line summary for the tool-trace UI.

        Examples:
          * 6 items, 2 done, +1 this call → "6 todos · 2/6 done · +1"
          * 6 items, 6 done, +2 this call → "all 6 done · +2"
          * 6 items, no change            → "6 todos · 2/6 done"
          * cleared                       → "cleared"
          * error                         → "Error: ..."
          * unchanged / no-op             → ""  (skip the tail)
        """
        if not isinstance(result, str):
            return ""
        if result.startswith("Error"):
            from nanobot.agent.tools.summaries import extract_error_summary
            return extract_error_summary(result)
        if result == "Todo list cleared.":
            return "cleared"
        items = args.get("items") if isinstance(args, dict) else None
        if not isinstance(items, list):
            return ""
        total = len(items)
        if total == 0:
            return ""
        done = sum(1 for it in items if isinstance(it, dict) and it.get("status") == "completed")
        delta = self._last_done_delta.get(self._session_key, 0)
        if done == total:
            # "all N done" already implies completion; +N would be redundant
            # (and looks odd when the LLM batches everything into one call).
            return f"all {total} done"
        delta_suffix = f" · +{delta}" if delta > 0 else ""
        return f"{total} todos · {done}/{total} done{delta_suffix}"

    async def execute(self, items: list[dict[str, Any]], **_kwargs: Any) -> str:
        if len(items) > MAX_TODO_ITEMS:
            return (
                f"Error: todo list exceeds limit ({len(items)} > {MAX_TODO_ITEMS}). "
                "Split the work into multiple plans or consolidate completed items."
            )

        in_progress = [it for it in items if it.get("status") == "in_progress"]
        if len(in_progress) > 1:
            return (
                "Error: at most one todo may be in_progress at a time "
                f"(found {len(in_progress)}). Demote the others to pending or completed."
            )

        session = self._sessions.get_or_create(self._session_key)
        prev_todos = [dict(t) for t in session.todos]
        prev_by_content = {
            (it.get("content") or "").strip(): it
            for it in session.todos
            if (it.get("content") or "").strip()
        }
        now = datetime.now().isoformat(timespec="seconds")

        new_items: list[dict[str, Any]] = []
        errors: list[str] = []
        seen_contents: set[str] = set()

        for idx, item in enumerate(items):
            content = (item.get("content") or "").strip()
            if not content:
                errors.append(f"item[{idx}] has empty content")
                continue
            if content in seen_contents:
                errors.append(f"item[{idx}] duplicate content {content!r}")
                continue
            seen_contents.add(content)

            status = item.get("status")
            if status not in _VALID_STATUSES:
                errors.append(f"item[{idx}] invalid status {status!r}")
                continue

            prev = prev_by_content.get(content)
            prev_status = prev.get("status") if prev else None

            if prev_status == "completed" and status != "completed":
                errors.append(
                    f"item[{idx}] {content!r}: completed→{status} not allowed "
                    "(completed is terminal; delete & re-add if needed)"
                )
                continue

            new_item: dict[str, Any] = {"content": content, "status": status}
            new_item["created_at"] = prev.get("created_at") if prev else now
            if prev and "started_at" in prev:
                new_item["started_at"] = prev["started_at"]
            if status == "in_progress" and "started_at" not in new_item:
                new_item["started_at"] = now
            if prev and "completed_at" in prev:
                new_item["completed_at"] = prev["completed_at"]
            if status == "completed" and "completed_at" not in new_item:
                new_item["completed_at"] = now
            new_items.append(new_item)

        if errors:
            return "Error: " + "; ".join(errors)

        session.todos = new_items
        self._sessions.save(session)

        # Record done-count delta vs the previous execute() call for this
        # session, so summarize_result can render "+N" newly completed.
        new_done_count = sum(1 for it in new_items if it.get("status") == "completed")
        prev_done_count = self._done_count.get(self._session_key, 0)
        self._last_done_delta[self._session_key] = new_done_count - prev_done_count
        self._done_count[self._session_key] = new_done_count

        # Push a live diff to the user via the message bus so the TUI can
        # show progress immediately, not just when the whole turn completes.
        if self._bus is not None:
            diff = format_todo_diff(prev_todos, new_items)
            if diff:
                try:
                    from nanobot.bus.events import OutboundMessage
                    await self._bus.publish_outbound(OutboundMessage(
                        channel=self._channel,
                        chat_id=self._chat_id,
                        content=diff,
                        metadata={"_system_message": True},
                    ))
                except Exception:
                    pass  # never break tool execution because of a UI side-effect

        if not new_items:
            return "Todo list cleared."
        return "Updated todo list:\n" + format_todos(new_items)


# ── self-registration ────────────────────────────────────────────────────

from nanobot.agent.tools.registry import register_fork_tool  # noqa: E402

register_fork_tool(lambda loop: TodoWriteTool(sessions=loop.sessions, bus=loop.bus))
