import pytest
from unittest.mock import AsyncMock, patch

from app.agents.requirement_clarifier.planner import Planner
from app.agents.base.agent import WorkflowState


class TestPlanner:
    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_summarize_requirement(self, mock_call_llm):
        mock_call_llm.return_value = "A user registration feature with email verification."
        agent = Planner("planner", llm_client=None)  # type: ignore
        result = await agent._summarize_requirement("Long doc about registration...")
        assert result == "A user registration feature with email verification."

    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_generate_plan(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"modules_to_explore": ["auth", "user_model"], "search_queries": ["User", "Registration"], "priority_order": [0, 1], "depth_hint": "deep"}'
        )
        agent = Planner("planner", llm_client=None)  # type: ignore
        plan = await agent._generate_exploration_plan("A registration feature")
        assert plan["modules_to_explore"] == ["auth", "user_model"]
        assert plan["depth_hint"] == "deep"

    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_process_without_summary(self, mock_call_llm):
        mock_call_llm.side_effect = [
            "A registration feature",  # first call: summarize
            '{"modules_to_explore": ["auth"], "search_queries": ["User"], "priority_order": [0], "depth_hint": "shallow"}',
        ]
        state: WorkflowState = {"requirement_doc": "A registration feature doc"}
        agent = Planner("planner", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert "requirement_doc_summary" in result
        assert "exploration_plan" in result
        assert result.get("exploration_round") == 1
