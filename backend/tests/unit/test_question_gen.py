import pytest
from unittest.mock import AsyncMock, patch

from app.agents.requirement_clarifier.question_gen import QuestionGen
from app.agents.requirement_clarifier.critic import Critic
from app.agents.base.agent import WorkflowState


class TestQuestionGen:
    @pytest.mark.asyncio
    async def test_no_gaps(self):
        state: WorkflowState = {
            "requirement_doc_summary": "Simple feature",
            "analysis": {"uncovered_gaps": [], "covered_points": []},
            "code_snippets": [],
        }
        agent = QuestionGen("question_gen", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["questions"] == []

    @pytest.mark.asyncio
    @patch.object(QuestionGen, "call_llm", new_callable=AsyncMock)
    async def test_generate_questions(self, mock_call_llm):
        mock_call_llm.return_value = (
            '[{"id": "abc-123", "text": "How should we handle duplicate emails?", '
            '"category": "data_model", "severity": "critical", '
            '"source_code_refs": [], "requirement_ref": "Section 2", '
            '"suggested_options": ["reject", "merge"]}]'
        )
        state: WorkflowState = {
            "requirement_doc_summary": "User registration",
            "analysis": {
                "uncovered_gaps": ["duplicate email handling"],
                "covered_points": ["User model exists"],
            },
            "code_snippets": [],
        }
        agent = QuestionGen("question_gen", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert len(result["questions"]) == 1
        assert result["questions"][0]["text"] == "How should we handle duplicate emails?"


class TestCritic:
    @pytest.mark.asyncio
    async def test_no_questions(self):
        state: WorkflowState = {"questions": []}
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["critic_result"]["passed"] is False

    @pytest.mark.asyncio
    @patch.object(Critic, "call_llm", new_callable=AsyncMock)
    async def test_pass(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"filtered_questions": [{"text": "Good question"}], '
            '"tech_questions": [], "quality_feedback": "Good questions", "passed": true}'
        )
        state: WorkflowState = {
            "questions": [{"text": "Good question"}],
            "analysis": {"coverage_score": 0.8},
        }
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["critic_result"]["passed"] is True
        assert len(result["filtered_questions"]) == 1

    @pytest.mark.asyncio
    @patch.object(Critic, "call_llm", new_callable=AsyncMock)
    async def test_separate_tech_questions(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"filtered_questions": [{"text": "Should we use OAuth2?"}], '
            '"tech_questions": [{"text": "Should we use Redis or Memcached?"}], '
            '"quality_feedback": "One technical question filtered", "passed": true}'
        )
        state: WorkflowState = {
            "questions": [
                {"text": "Should we use OAuth2?"},
                {"text": "Redis or Memcached?"},
            ],
            "analysis": {"coverage_score": 0.8},
        }
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert len(result["critic_result"]["filtered_questions"]) == 1
        assert len(result["critic_result"]["tech_questions"]) == 1
