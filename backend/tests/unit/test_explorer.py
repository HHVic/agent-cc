import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.requirement_clarifier.explorer import Explorer
from app.agents.base.agent import WorkflowState


class TestExplorer:
    @pytest.mark.asyncio
    async def test_process_no_plan(self):
        state: WorkflowState = {}
        agent = Explorer("explorer", llm_client=None, tools={})  # type: ignore
        result = await agent.process(state)
        # Should not crash, exploration_rounds should be empty or unchanged
        assert "exploration_rounds" not in result or len(result.get("exploration_rounds", [])) == 0

    @pytest.mark.asyncio
    async def test_process_with_plan(self):
        state: WorkflowState = {
            "exploration_plan": {
                "modules_to_explore": ["auth"],
                "search_queries": ["User"],
                "priority_order": [0],
                "depth_hint": "deep",
            }
        }
        mock_glob = MagicMock()
        mock_glob.execute = AsyncMock(return_value=MagicMock(content=""))
        mock_grep = MagicMock()
        mock_grep.execute = AsyncMock(return_value=MagicMock(content=""))
        mock_file_read = MagicMock()
        mock_file_read.execute = AsyncMock(
            return_value=MagicMock(content="def foo(): pass", is_error=False)
        )
        agent = Explorer(
            "explorer",
            llm_client=None,
            tools={
                "glob_search": mock_glob,
                "grep_search": mock_grep,
                "file_read": mock_file_read,
            },
        )  # type: ignore
        result = await agent.process(state)
        assert "code_snippets" in result
        assert "exploration_rounds" in result
        assert len(result["exploration_rounds"]) == 1
