import asyncio
import logging
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState
from app.agents.requirement_clarifier.planner import Planner
from app.agents.requirement_clarifier.explorer import Explorer
from app.agents.requirement_clarifier.analyzer import Analyzer
from app.agents.requirement_clarifier.question_gen import QuestionGen
from app.agents.requirement_clarifier.critic import Critic

logger = logging.getLogger(__name__)


class ClarificationWorkflow:
    """LangGraph-compatible workflow for requirement clarification.

    Uses a manual state machine approach (not LangGraph's StateGraph) to avoid
    coupling agents to the orchestration framework. Each agent is a plain function.
    """

    def __init__(
        self,
        planner: Planner,
        explorer: Explorer,
        analyzer: Analyzer,
        question_gen: QuestionGen,
        critic: Critic,
        max_exploration_rounds: int = 3,
        max_clarification_rounds: int = 3,
        min_questions: int = 3,
        coverage_threshold: float = 0.7,
    ):
        self.planner = planner
        self.explorer = explorer
        self.analyzer = analyzer
        self.question_gen = question_gen
        self.critic = critic
        self.max_exploration_rounds = max_exploration_rounds
        self.max_clarification_rounds = max_clarification_rounds
        self.min_questions = min_questions
        self.coverage_threshold = coverage_threshold

    async def run(self, state: WorkflowState) -> WorkflowState:
        """Execute the full clarification workflow."""
        queue: asyncio.Queue = state["_message_queue"]

        # Outer loop: clarification rounds
        for clar_round in range(1, self.max_clarification_rounds + 1):
            state.setdefault("clarification_rounds", []).append({
                "round": clar_round, "questions": [], "filtered_questions": [], "passed": False,
            })

            queue.put_nowait({"type": "round_transition", "content": f"Clarification round {clar_round}"})

            # Inner loop: exploration
            for exp_round in range(self.max_exploration_rounds):
                feedback = state.get("exploration_feedback", "")
                if not feedback and exp_round == 0:
                    feedback = "Start with a fresh exploration."

                state["exploration_feedback"] = feedback

                # Planner
                state = await self.planner.process(state)
                await self._flush_queue(queue)

                # Explorer
                state = await self.explorer.process(state)
                await self._flush_queue(queue)

                # Analyzer
                state = await self.analyzer.process(state)
                await self._flush_queue(queue)

                analysis = state.get("analysis", {})
                if not analysis.get("needs_more_exploration", False):
                    break

                # Check if we've hit the exploration limit
                if exp_round >= self.max_exploration_rounds - 1:
                    break

            # Generate questions
            state = await self.question_gen.process(state)
            await self._flush_queue(queue)

            # Critic
            state = await self.critic.process(state)
            await self._flush_queue(queue)

            result = state.get("critic_result", {})
            round_idx = clar_round - 1
            if state.get("clarification_rounds"):
                state["clarification_rounds"][round_idx]["questions"] = state.get("questions", [])
                state["clarification_rounds"][round_idx]["filtered_questions"] = result.get("filtered_questions", [])
                state["clarification_rounds"][round_idx]["passed"] = result.get("passed", False)

            if result.get("passed", False):
                break

            # Set feedback for next exploration round
            state["exploration_feedback"] = result.get("quality_feedback", "More exploration needed.")

        # Finalize
        state["filtered_questions"] = result.get("filtered_questions", [])
        state["tech_questions"] = result.get("tech_questions", [])
        return state

    async def _flush_queue(self, queue: asyncio.Queue) -> None:
        """Drain all pending messages from the queue."""
        while not queue.empty():
            try:
                queue.get_nowait()
            except Exception:
                break
