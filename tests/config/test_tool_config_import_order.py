"""Tool-first imports must leave public configuration types usable."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize("tool_module", ["image_generation", "shell", "web"])
def test_tool_first_import_resolves_public_config(tool_module):
    script = (
        f"import nanobot.agent.tools.{tool_module}; "
        "from nanobot.config.schema import Config, ToolsConfig; "
        "assert ToolsConfig().web.search is not None; "
        "assert Config.model_validate({}).tools.exec is not None; "
        "from nanobot.config.schema import WebSearchConfig, ImageGenerationToolConfig; "
        "assert WebSearchConfig is not None and ImageGenerationToolConfig is not None"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
