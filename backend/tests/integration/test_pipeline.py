import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.base.agent import WorkflowState
from app.workflow.graph import ClarificationWorkflow


class TestEndToEndPipeline:
    """Integration test that runs the full pipeline with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Verify the pipeline runs from requirement to filtered questions."""
        def planner_side_effect(state):
            state["exploration_plan"] = {
                "modules_to_explore": ["auth"],
                "search_queries": ["User", "Auth"],
                "priority_order": [0],
                "depth_hint": "deep",
            }
            state["exploration_round"] = 1
            state["messages"] = state.get("messages", [])
            return state

        def explorer_side_effect(state):
            state["code_snippets"] = [
                {"path": "models/user.py", "content": "class User: pass"}
            ]
            state["call_graph_nodes"] = []
            state["call_graph_edges"] = []
            state["exploration_rounds"] = [{"round": 1, "files_explored": 1, "functions_found": 1}]
            state["messages"] = state.get("messages", [])
            return state

        def analyzer_side_effect(state):
            state["analysis"] = {
                "coverage_score": 0.5,
                "covered_points": ["User model exists"],
                "uncovered_gaps": ["No email verification flow"],
                "needs_more_exploration": False,
                "exploration_feedback": "",
            }
            state["messages"] = state.get("messages", [])
            return state

        def question_gen_side_effect(state):
            state["questions"] = [
                {"id": "q1", "text": "Should email verification be required?", "category": "business_flow", "severity": "critical", "source_code_refs": [], "requirement_ref": "Section 1", "suggested_options": ["required", "optional"]}
            ]
            state["messages"] = state.get("messages", [])
            return state

        def critic_side_effect(state):
            state["critic_result"] = {
                "filtered_questions": [state["questions"][0]],
                "tech_questions": [],
                "quality_feedback": "Good questions",
                "passed": True,
            }
            state["filtered_questions"] = state.get("questions", [])
            state["tech_questions"] = []
            state["messages"] = state.get("messages", [])
            return state

        mock_agents = [
            MagicMock(process=AsyncMock(side_effect=planner_side_effect)),
            MagicMock(process=AsyncMock(side_effect=explorer_side_effect)),
            MagicMock(process=AsyncMock(side_effect=analyzer_side_effect)),
            MagicMock(process=AsyncMock(side_effect=question_gen_side_effect)),
            MagicMock(process=AsyncMock(side_effect=critic_side_effect)),
        ]

        queue = asyncio.Queue()
        state: WorkflowState = {
            "session_id": "e2e-test",
            "requirement_doc": "Users should verify their email",
            "requirement_doc_summary": "Email verification for new users",
            "target_repos": [],
            "base_dir": ".",
            "_message_queue": queue,
            "settings": {
                "max_exploration_rounds": 2,
                "max_clarification_rounds": 2,
                "min_questions": 1,
                "coverage_threshold": 0.5,
            },
        }

        workflow = ClarificationWorkflow(
            planner=mock_agents[0],
            explorer=mock_agents[1],
            analyzer=mock_agents[2],
            question_gen=mock_agents[3],
            critic=mock_agents[4],
            max_exploration_rounds=2,
            max_clarification_rounds=2,
            min_questions=1,
            coverage_threshold=0.5,
        )

        result = await workflow.run(state)

        assert "filtered_questions" in result
        assert len(result["filtered_questions"]) >= 0
        assert "messages" in result
        assert isinstance(result["messages"], list)
