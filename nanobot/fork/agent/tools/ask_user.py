"""Interactive question tool — let the LLM prompt the user for explicit
choices via the TUI (similar to Claude Code's AskUserQuestion).

Flow:
  1. LLM calls ``ask_user(questions=[...])``.
  2. Tool publishes an ``OutboundMessage`` with ``metadata={"_ask_user": True,
     "_ask_user_id": <uuid>, "_ask_user_questions": [...]}``.
  3. The TUI ``_consume_outbound`` loop in ``commands.py`` shows the popup
     and, after the user picks, publishes a reply with
     ``metadata={"_ask_user_reply": True, "_ask_user_id": <same uuid>,
     "_ask_user_answers": {...}}``.
  4. A small dispatcher in this module routes the reply to the awaiting
     ``asyncio.Future`` keyed by the correlation id.
  5. The tool resolves the future and returns the JSON answers to the LLM.

Only the CLI channel currently supports popups; other channels (telegram,
slack, etc.) return an "unsupported" error so the LLM can fall back to
plain-text questions.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.tools.context import RequestContext
    from nanobot.bus.queue import MessageBus


# ── Reply dispatcher (module-level so commands.py can register replies) ────


_PENDING_FUTURES: dict[str, asyncio.Future] = {}


def deliver_reply(correlation_id: str, answers: dict[str, str] | None, cancelled: bool = False) -> None:
    """Resolve the awaiting future for ``correlation_id`` with the user's
    answers (or a cancellation). Called by ``commands.py`` when the popup
    receives a reply or is dismissed. Safe to call for unknown ids — a stale
    reply (e.g. arriving after timeout) is simply ignored.
    """
    fut = _PENDING_FUTURES.pop(correlation_id, None)
    if fut is None or fut.done():
        return
    if cancelled:
        fut.set_result({"cancelled": True})
    else:
        fut.set_result({"answers": answers or {}})


class AskUserTool(Tool):
    """Ask the user a structured question with selectable options."""

    DEFAULT_TIMEOUT_SEC = 300

    def __init__(self, bus: "MessageBus | None" = None) -> None:
        self._bus = bus
        self._channel = "cli"
        self._chat_id = "direct"

    def set_context(self, ctx: "RequestContext") -> None:
        self._channel = ctx.channel
        self._chat_id = ctx.chat_id

    @property
    def name(self) -> str:
        return "ask_user"

    @property
    def description(self) -> str:
        return (
            "Ask the user one or more multiple-choice questions and wait for "
            "their selection. Use ONLY when (a) you need a decision the user "
            "must make (e.g. choose between approaches), or (b) the request "
            "is ambiguous and you want to clarify before committing. Do NOT "
            "use to confirm trivia or seek permission for normal actions. "
            "Each question is presented in a single popup; user picks one "
            "option per question. If the user cancels (ESC), the tool returns "
            "{cancelled: true} — fall back to a plain-text follow-up. "
            "Only supported on interactive CLI channels."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "description": "One to five questions to ask in sequence.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The full question text to show the user.",
                            },
                            "header": {
                                "type": "string",
                                "description": "Optional short label (≤12 chars) shown as a tag.",
                            },
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 6,
                                "description": "2–6 distinct options for the user to pick from.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": "Concise option text (1–5 words).",
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": "What this option means or implies.",
                                        },
                                    },
                                    "required": ["label", "description"],
                                },
                            },
                        },
                        "required": ["question", "options"],
                    },
                },
            },
            "required": ["questions"],
        }

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        """Show "answered N/M" or "cancelled" in the tool-trace UI."""
        if not isinstance(result, str):
            return ""
        try:
            data = json.loads(result)
        except Exception:
            return ""
        if data.get("cancelled"):
            return "cancelled"
        if data.get("error"):
            return f"error: {data['error'][:60]}"
        answers = data.get("answers") or {}
        questions = args.get("questions") or [] if isinstance(args, dict) else []
        return f"answered {len(answers)}/{len(questions)}"

    async def execute(self, questions: list[dict[str, Any]], **_kwargs: Any) -> str:
        if not isinstance(questions, list) or not questions:
            return json.dumps({"error": "questions must be a non-empty list"})

        if self._channel != "cli":
            return json.dumps({
                "error": (
                    f"interactive popups not supported on channel "
                    f"'{self._channel}'. Ask the user in plain text instead."
                ),
                "channel": self._channel,
            })

        if self._bus is None:
            return json.dumps({"error": "ask_user requires a bus, none configured"})

        correlation_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        _PENDING_FUTURES[correlation_id] = fut

        try:
            from nanobot.bus.events import OutboundMessage
            await self._bus.publish_outbound(OutboundMessage(
                channel=self._channel,
                chat_id=self._chat_id,
                content="",
                metadata={
                    "_ask_user": True,
                    "_ask_user_id": correlation_id,
                    "_ask_user_questions": questions,
                },
            ))
        except Exception as exc:
            _PENDING_FUTURES.pop(correlation_id, None)
            return json.dumps({"error": f"failed to publish question: {exc}"})

        try:
            result = await asyncio.wait_for(fut, timeout=self.DEFAULT_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            _PENDING_FUTURES.pop(correlation_id, None)
            return json.dumps({"error": "user did not respond within timeout", "timeout_sec": self.DEFAULT_TIMEOUT_SEC})
        except asyncio.CancelledError:
            _PENDING_FUTURES.pop(correlation_id, None)
            raise

        return json.dumps(result, ensure_ascii=False)


# ── self-registration ────────────────────────────────────────────────────

from nanobot.agent.tools.registry import register_fork_tool  # noqa: E402

register_fork_tool(lambda loop: AskUserTool(bus=loop.bus))
