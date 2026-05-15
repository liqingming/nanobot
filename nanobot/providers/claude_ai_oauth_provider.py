"""Claude.ai OAuth provider — uses subscription Bearer token instead of API key."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.providers.base import LLMProvider

NANOBOT_CRED_FILE = Path.home() / ".nanobot" / "credentials" / "claude_ai.json"
CLAUDE_CODE_CRED_FILE = Path.home() / ".claude" / ".credentials.json"


def load_oauth_token() -> str:
    """Return OAuth Bearer token, checking nanobot store then Claude Code store."""
    if NANOBOT_CRED_FILE.exists():
        data = json.loads(NANOBOT_CRED_FILE.read_text(encoding="utf-8"))
        token = data.get("access_token", "")
        if token:
            return token

    if CLAUDE_CODE_CRED_FILE.exists():
        data = json.loads(CLAUDE_CODE_CRED_FILE.read_text(encoding="utf-8"))
        token = (data.get("claudeAiOauth") or {}).get("accessToken", "")
        if token:
            return token

    raise RuntimeError(
        "No Claude.ai OAuth token found. Run: nanobot provider login claude-ai"
    )


def save_oauth_token(token: str) -> None:
    """Persist OAuth token to nanobot's credential store."""
    NANOBOT_CRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    NANOBOT_CRED_FILE.write_text(
        json.dumps({"access_token": token}, indent=2),
        encoding="utf-8",
    )


class ClaudeAIOAuthProvider(AnthropicProvider):
    """Claude provider authenticated via claude.ai OAuth (subscription account)."""

    def __init__(
        self,
        default_model: str = "claude-sonnet-4-20250514",
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        # Bypass AnthropicProvider.__init__; call the base to set api_key/api_base/generation.
        LLMProvider.__init__(self, api_key=None, api_base=api_base)
        self.default_model = default_model
        self.extra_headers = extra_headers or {}

        oauth_token = load_oauth_token()

        from anthropic import AsyncAnthropic

        client_kw: dict[str, Any] = {"auth_token": oauth_token}
        if api_base:
            client_kw["base_url"] = api_base
        if self.extra_headers:
            client_kw["default_headers"] = self.extra_headers
        self._client = AsyncAnthropic(**client_kw)
