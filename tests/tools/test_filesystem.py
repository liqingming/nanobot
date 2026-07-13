import pytest

from nanobot.agent.tools.context import RequestContext, bind_request_context, reset_request_context
from nanobot.agent.tools.filesystem import ReadFileTool


@pytest.mark.asyncio
async def test_read_file_rejects_subpath_from_request_tool_policy(tmp_path):
    target = tmp_path / "Assets" / "ResourcesAssets" / "Prefabs" / "Test.prefab"
    target.parent.mkdir(parents=True)
    target.write_text("resource", encoding="utf-8")
    tool = ReadFileTool(workspace=tmp_path, allowed_dir=tmp_path)
    token = bind_request_context(RequestContext(
        channel="api", chat_id="default", session_key="api:review_123_code_review",
        metadata={"tool_policy": {"blocked_read_file_paths": ["Assets/ResourcesAssets"]}},
    ))
    try:
        result = await tool.execute(path="Assets/ResourcesAssets/Prefabs/Test.prefab")
    finally:
        reset_request_context(token)
    assert "blocked by the request tool_policy for read_file" in str(result)
