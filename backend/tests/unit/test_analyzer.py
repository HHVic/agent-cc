import pytest
from unittest.mock import AsyncMock, patch

from app.agents.requirement_clarifier.analyzer import Analyzer
from app.agents.base.agent import WorkflowState


class TestAnalyzer:
    @pytest.mark.asyncio
    async def test_process_no_content(self):
        state: WorkflowState = {}
        agent = Analyzer("analyzer", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["analysis"]["coverage_score"] == 0.0
        assert result["analysis"]["needs_more_exploration"] is True

    @pytest.mark.asyncio
    @patch.object(Analyzer, "call_llm", new_callable=AsyncMock)
    async def test_process_with_content(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"coverage_score": 0.7, "covered_points": ["user_model exists"], '
            '"uncovered_gaps": ["no email verification"], '
            '"needs_more_exploration": true, '
            '"exploration_feedback": "Explore the auth module more deeply"}'
        )
        state: WorkflowState = {
            "requirement_doc_summary": "User registration with email verification",
            "code_snippets": [{"path": "models/user.py", "content": "class User: pass"}],
        }
        agent = Analyzer("analyzer", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["analysis"]["coverage_score"] == 0.7
        assert result["analysis"]["needs_more_exploration"] is True
        assert "no email verification" in str(result["analysis"]["uncovered_gaps"])
