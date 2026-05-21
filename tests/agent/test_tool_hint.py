"""Tests for format_tool_hint argument-selection priority + path relativization."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nanobot.agent.loop import _smart_truncate, format_tool_hint, relativize_path


def _tc(name: str, arguments) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


# ── argument priority ─────────────────────────────────────────────────────


def test_write_file_picks_path_not_content() -> None:
    """LLM JSON ordering can put 'content' first; hint must still show path."""
    args = {"content": '"""\n双均线策略\n"""\n', "path": "main.py"}
    assert format_tool_hint([_tc("write_file", args)]) == 'write_file("main.py")'


def test_exec_picks_command() -> None:
    args = {"command": "ls -la", "cwd": "/tmp"}
    assert format_tool_hint([_tc("exec", args)]) == 'exec("ls -la")'


def test_web_search_picks_query() -> None:
    args = {"max_results": 5, "query": "python textual"}
    assert format_tool_hint([_tc("web_search", args)]) == 'web_search("python textual")'


def test_read_file_picks_path() -> None:
    args = {"limit": 100, "offset": 0, "path": "/etc/hosts"}
    assert format_tool_hint([_tc("read_file", args)]) == 'read_file("/etc/hosts")'


def test_edit_file_picks_path_not_old_string() -> None:
    args = {"old_string": "x", "new_string": "y", "path": "foo.py"}
    assert format_tool_hint([_tc("edit_file", args)]) == 'edit_file("foo.py")'


def test_long_value_is_truncated() -> None:
    args = {"path": "x" * 100}  # non-path-looking (no slashes) → leading truncation
    out = format_tool_hint([_tc("write_file", args)])
    assert out.endswith('…")')
    assert len(out) < 60


def test_long_path_keeps_head_and_tail() -> None:
    """Path-like strings preserve both leading directory and trailing filename."""
    args = {"path": "/very/deep/nested/path/to/some_long_filename.py"}
    out = format_tool_hint([_tc("write_file", args)])
    # The filename (or a recognizable suffix of it) should survive truncation
    assert "filename.py" in out or "_filename.py" in out
    # Some leading portion should also survive
    assert "/very" in out
    # Ellipsis somewhere in the middle
    assert "…" in out


def test_smart_truncate_short_text_unchanged() -> None:
    assert _smart_truncate("short", 40) == "short"


def test_smart_truncate_long_nonpath() -> None:
    out = _smart_truncate("a" * 100, 40)
    assert len(out) == 40
    assert out.endswith("…")


def test_smart_truncate_long_path_preserves_filename() -> None:
    text = "/a/b/c/d/e/f/g/h/i/j/k/very_long_file_name.py"
    out = _smart_truncate(text, 40)
    assert "very_long_file_name.py" in out or "file_name.py" in out
    assert "/a/" in out or "/a" in out
    assert "…" in out
    assert len(out) <= 40


def test_smart_truncate_windows_path() -> None:
    text = r"C:\very\deep\nested\windows\path\to\file.py"
    out = _smart_truncate(text, 40)
    assert "file.py" in out
    assert "…" in out


def test_no_string_args_returns_name_only() -> None:
    assert format_tool_hint([_tc("custom_tool", {"count": 5, "verbose": True})]) == "custom_tool"


def test_empty_args_returns_name_only() -> None:
    assert format_tool_hint([_tc("ping", {})]) == "ping"


def test_falls_back_to_first_string_when_no_priority_key() -> None:
    args = {"mystery_input": "alpha", "count": 5}
    assert format_tool_hint([_tc("unknown_tool", args)]) == 'unknown_tool("alpha")'


def test_priority_skips_empty_string() -> None:
    args = {"path": "  ", "command": "real command"}
    assert format_tool_hint([_tc("custom", args)]) == 'custom("real command")'


def test_multiple_tool_calls_joined() -> None:
    calls = [
        _tc("read_file", {"path": "a"}),
        _tc("write_file", {"path": "b", "content": "x"}),
    ]
    assert format_tool_hint(calls) == 'read_file("a"), write_file("b")'


def test_list_arguments_takes_first() -> None:
    tc = _tc("read_file", [{"path": "x.txt"}])
    assert format_tool_hint([tc]) == 'read_file("x.txt")'


def test_non_dict_non_list_arguments_returns_name() -> None:
    assert format_tool_hint([_tc("foo", "raw string")]) == "foo"


# ── path relativization ───────────────────────────────────────────────────


def test_relativize_path_inside_workspace(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "file.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    out = relativize_path(str(target), tmp_path)
    # Always uses forward slashes + ./ prefix, regardless of platform
    assert out == "./sub/file.py"


def test_relativize_path_top_level(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x")
    out = relativize_path(str(target), tmp_path)
    assert out == "./main.py"


def test_relativize_path_outside_workspace_keeps_absolute(tmp_path: Path) -> None:
    # Use a path that is absolute but definitely outside tmp_path
    if Path("/etc").exists():
        outside = "/etc/hosts"
    else:
        outside = "C:\\Windows\\System32\\cmd.exe"
    out = relativize_path(outside, tmp_path)
    assert out == outside


def test_relativize_path_relative_input_unchanged() -> None:
    assert relativize_path("foo/bar.py", "/some/workspace") == "foo/bar.py"


def test_relativize_path_no_workspace_unchanged() -> None:
    assert relativize_path("/abs/path.py", None) == "/abs/path.py"


def test_tool_hint_uses_workspace_for_path(tmp_path: Path) -> None:
    target = tmp_path / "src" / "main.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x")
    args = {"path": str(target), "content": "anything"}
    out = format_tool_hint([_tc("write_file", args)], workspace=tmp_path)
    # Expect ./relative path, no absolute prefix
    assert "./src/main.py" in out
    assert str(tmp_path) not in out


def test_tool_hint_outside_workspace_keeps_absolute(tmp_path: Path) -> None:
    if Path("/etc").exists():
        outside = "/etc/hosts"
        # Absolute path with no backslashes — stays as-is
        out = format_tool_hint([_tc("read_file", {"path": outside})], workspace=tmp_path)
        assert outside in out
    else:
        # Windows absolute path: kept as absolute but backslashes normalized
        outside = "C:\\Windows\\System32\\drivers\\etc\\hosts"
        out = format_tool_hint([_tc("read_file", {"path": outside})], workspace=tmp_path)
        # Forward slashes for visual consistency in the trace
        assert "C:/Windows" in out
        assert "\\" not in out


def test_tool_hint_normalizes_backslashes_for_path_keys(tmp_path: Path) -> None:
    """Even absolute Windows paths outside workspace should show forward slashes."""
    args = {"path": "D:\\some\\external\\dir\\file.py"}
    out = format_tool_hint([_tc("write_file", args)], workspace=tmp_path)
    assert "\\" not in out
    assert "D:/some" in out or "D:/some/external" in out
    assert "file.py" in out


def test_tool_hint_no_relativization_for_non_path_keys() -> None:
    """url / query / command must NOT be touched by path logic."""
    args = {"url": "/some/path/looking/url"}
    out = format_tool_hint([_tc("web_fetch", args)], workspace="/some")
    # url not in _HINT_PATH_KEYS → no relativization
    assert "/some/path/looking/url" in out
