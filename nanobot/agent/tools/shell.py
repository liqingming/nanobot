"""Shell execution tool."""

import asyncio
import locale
import os
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool


def _decode_console_bytes(data: bytes) -> str:
    """Decode subprocess stdout/stderr robustly across platforms.

    Modern tooling tends to emit UTF-8, but on Windows many native programs
    (cmd, python's stdout on a CJK system, etc.) emit data in the system's
    legacy console codepage (cp936 / cp1252 / sjis, …). Trying UTF-8 first
    and falling back to the OEM/locale encoding fixes the common case where
    a Chinese Windows console returns GBK bytes and naive utf-8 decoding
    produces mojibake.
    """
    if not data:
        return ""
    # Fast path: try strict UTF-8 first.
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # On Windows, the legacy console codepage is the usual source of GBK/cp936.
    candidates: list[str] = []
    if sys.platform == "win32":
        for env_var in ("PYTHONIOENCODING",):
            value = os.environ.get(env_var)
            if value:
                candidates.append(value.split(":", 1)[0])  # strip ":errors" suffix
        try:
            import ctypes
            cp = ctypes.windll.kernel32.GetOEMCP()
            candidates.append(f"cp{cp}")
        except Exception:
            pass
        candidates.append("cp936")  # very common on Chinese Windows
    # Fallback to the process locale's preferred encoding (Linux: utf-8;
    # Windows when no console: cp1252 or similar).
    candidates.append(locale.getpreferredencoding(False))
    seen: set[str] = set()
    for enc in candidates:
        if not enc or enc.lower() in seen:
            continue
        seen.add(enc.lower())
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort — never raise.
    return data.decode("utf-8", errors="replace")


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                finally:
                    if sys.platform != "win32":
                        try:
                            os.waitpid(process.pid, os.WNOHANG)
                        except (ProcessLookupError, ChildProcessError) as e:
                            logger.debug("Process already reaped or not found: {}", e)
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(_decode_console_bytes(stdout))

            if stderr:
                stderr_text = _decode_console_bytes(stderr)
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        from nanobot.security.network import contains_internal_url
        if contains_internal_url(cmd):
            return "Error: Command blocked by safety guard (internal/private URL detected)"

        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return "Error: Command blocked by safety guard (path traversal detected)"

            cwd_path = Path(cwd).resolve()

            for raw in self._extract_absolute_paths(cmd):
                try:
                    expanded = os.path.expandvars(raw.strip())
                    p = Path(expanded).expanduser().resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return "Error: Command blocked by safety guard (path outside working dir)"

        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command) # POSIX: /absolute only
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command) # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths

    def summarize_result(self, args: dict[str, Any], result: Any) -> str:
        from nanobot.agent.tools.summaries import (
            extract_error_summary, line_count, summarize_error_or,
        )
        if not isinstance(result, str):
            return ""
        if result.startswith("Error"):
            return extract_error_summary(result)
        lines = result.rstrip("\n").split("\n")
        exit_marker: str | None = None
        body = lines
        for prefix in ("[exit code:", "exit code:", "[exit:"):
            if lines and prefix in lines[-1].lower():
                exit_marker = lines[-1].strip()
                body = lines[:-1]
                break
        n = len(body)
        suffix = f", {n} line{'s' if n != 1 else ''}" if n else ""
        return (exit_marker or "done") + suffix
