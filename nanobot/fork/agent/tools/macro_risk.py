"""Cheap, deterministic risk checks for conditional C# compilation."""

from __future__ import annotations

import json
import re
from pathlib import Path

from nanobot.fork.agent.tools.code_intelligence_models import RiskAssessment

_PATH_SEGMENTS = frozenset({
    "editor", "build", "buildpipeline", "platform", "platforms", "channel",
    "channels", "sdk", "thirdparty", "plugins", "framework", "packages",
})
_PREPROCESSOR_RE = re.compile(r"^\s*#\s*(if|elif|else|endif|define)\b", re.MULTILINE)
_CONDITIONAL_RE = re.compile(r"\[\s*Conditional\s*\(")


def find_asmdef(path: Path, workspace: Path) -> Path | None:
    current = path.parent
    root = workspace.resolve()
    while True:
        matches = sorted(current.glob("*.asmdef"))
        if matches:
            return matches[0]
        if current.resolve() == root or current.parent == current:
            return None
        current = current.parent


def assess_macro_risk(path: Path, workspace: Path, *, intent: str = "") -> RiskAssessment:
    result = RiskAssessment()
    try:
        scoped_path = path.resolve().relative_to(workspace.resolve())
    except ValueError:
        scoped_path = path
    relative_parts = {part.casefold() for part in scoped_path.parts}
    matched = sorted(relative_parts & _PATH_SEGMENTS)
    if matched:
        result.add(3, f"path contains macro-sensitive segment: {matched[0]}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if _PREPROCESSOR_RE.search(text):
        result.add(2, "definition file contains conditional compilation directives")
    if _CONDITIONAL_RE.search(text):
        result.add(2, "definition file contains ConditionalAttribute")
    asmdef = find_asmdef(path, workspace)
    if asmdef is not None:
        try:
            data = json.loads(asmdef.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            data = {}
        constrained = [
            key for key in ("includePlatforms", "excludePlatforms", "defineConstraints", "versionDefines")
            if data.get(key)
        ]
        if constrained:
            result.add(4, f"assembly definition constrains: {', '.join(constrained)}")
    lowered = intent.casefold()
    exhaustive_words = (
        "all references", "all callers", "safe to delete", "impact", "全部引用",
        "所有调用", "能否删除", "影响范围",
    )
    if any(word in lowered for word in exhaustive_words):
        result.exhaustive_intent = True
        result.add(6, "query intent requests exhaustive coverage")
    elif any(word in lowered for word in ("平台", "渠道", "海外", "国服", "构建", "打包")):
        result.add(3, "query intent is platform-sensitive")
    return result.finish()
