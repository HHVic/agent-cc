import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.workflow.graph import ClarificationWorkflow
from app.agents.base.agent import WorkflowState


class TestClarificationWorkflow:
    @pytest.mark.asyncio
    async def test_run_complete_flow(self):
        """Test the full workflow with mocked agents."""
        # Mock agents to return predictable state
        mock_planner = MagicMock()
        mock_planner.process = AsyncMock(side_effect=lambda s: s)

        mock_explorer = MagicMock()
        mock_explorer.process = AsyncMock(side_effect=lambda s: s)

        mock_analyzer = MagicMock()
        mock_analyzer.process = AsyncMock(
            side_effect=lambda s: s  # First call: needs_more_exploration=True
        )

        mock_question_gen = MagicMock()
        mock_question_gen.process = AsyncMock(
            side_effect=lambda s: s
        )

        mock_critic = MagicMock()
        async def critic_side_effect(state):
            state["critic_result"] = {
                "filtered_questions": [{"text": "Good question"}],
                "tech_questions": [],
                "quality_feedback": "ok",
                "passed": True,
            }
            state["questions"] = [{"text": "Good question"}]
            state["messages"] = state.get("messages", [])
            return state

        mock_critic.process = AsyncMock(side_effect=critic_side_effect)

        queue = asyncio.Queue()
        queue.put_nowait({"type": "placeholder"})

        state: WorkflowState = {
            "session_id": "test-session",
            "requirement_doc": "A test requirement",
            "requirement_doc_summary": "Test summary",
            "target_repos": [],
            "base_dir": ".",
            "_message_queue": queue,
            "settings": {
                "max_exploration_rounds": 2,
                "max_clarification_rounds": 2,
                "min_questions": 3,
                "coverage_threshold": 0.7,
            },
        }

        workflow = ClarificationWorkflow(
            planner=mock_planner,
            explorer=mock_explorer,
            analyzer=mock_analyzer,
            question_gen=mock_question_gen,
            critic=mock_critic,
            max_exploration_rounds=2,
            max_clarification_rounds=2,
            min_questions=3,
            coverage_threshold=0.7,
        )

        result = await workflow.run(state)
        assert "messages" in result
        assert "clarification_rounds" in result
        assert len(result["clarification_rounds"]) >= 1
