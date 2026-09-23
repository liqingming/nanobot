"""Tests for Feishu SDK callback diagnostics."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.bus.queue import MessageBus
from nanobot.channels.feishu import FeishuChannel, FeishuConfig


def _make_channel() -> FeishuChannel:
    channel = FeishuChannel(
        FeishuConfig(
            enabled=True,
            app_id="cli_test",
            app_secret="secret",
            allow_from=["*"],
        ),
        MessageBus(),
    )
    channel.logger = MagicMock()
    return channel


def _event(message_id: str = "om_test") -> SimpleNamespace:
    message = SimpleNamespace(
        message_id=message_id,
        chat_id="oc_test",
        chat_type="group",
        message_type="text",
    )
    return SimpleNamespace(event=SimpleNamespace(message=message))


class TestFeishuCallbackDiagnostics:
    def test_callback_is_scheduled_and_completion_is_logged(self):
        channel = _make_channel()
        channel._loop = MagicMock()
        channel._loop.is_running.return_value = True
        channel._on_message = AsyncMock()
        completed = Future()

        with patch(
            "nanobot.channels.feishu.asyncio.run_coroutine_threadsafe",
            return_value=completed,
        ) as schedule:
            channel._on_message_sync(_event())
            completed.set_result(None)

        schedule.assert_called_once()
        channel.logger.debug.assert_any_call(
            "Feishu SDK callback scheduled message_id={}", "om_test"
        )
        channel.logger.debug.assert_any_call(
            "Feishu SDK callback completed message_id={}", "om_test"
        )
        schedule.call_args.args[0].close()

    def test_callback_logs_when_main_loop_is_not_running(self):
        channel = _make_channel()
        channel._loop = MagicMock()
        channel._loop.is_running.return_value = False

        with patch("nanobot.channels.feishu.asyncio.run_coroutine_threadsafe") as schedule:
            channel._on_message_sync(_event("om_dropped"))

        schedule.assert_not_called()
        channel.logger.error.assert_called_once_with(
            "Feishu SDK callback not scheduled message_id={} reason=main_loop_not_running",
            "om_dropped",
        )

    def test_callback_failure_is_logged(self):
        channel = _make_channel()
        failed = Future()
        failed.set_exception(RuntimeError("boom"))

        channel._on_message_scheduled_done("om_failed", failed)

        channel.logger.error.assert_called_once_with(
            "Feishu SDK callback failed message_id={} error={}",
            "om_failed",
            failed.exception(),
        )
