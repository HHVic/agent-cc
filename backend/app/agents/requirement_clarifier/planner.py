import json
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState
from app.core.config import get_settings


class Planner(BaseAgent):
    """Parses requirement documents and generates exploration plans.

    Breaks down the requirement into modules to explore, search queries,
    and priority ordering. Can accept feedback from Critic to refine plans.
    """

    async def process(self, state: WorkflowState) -> WorkflowState:
        requirement_doc = state.get("requirement_doc", "")
        requirement_summary = state.get("requirement_doc_summary", "")
        feedback = state.get("exploration_feedback", "")

        if not requirement_summary:
            summary = await self._summarize_requirement(requirement_doc)
            state["requirement_doc_summary"] = summary

        plan = await self._generate_exploration_plan(
            state["requirement_doc_summary"], feedback
        )

        state["exploration_plan"] = plan
        state["exploration_round"] = state.get("exploration_round", 0) + 1

        self.add_message(state, "plan_generated", plan)
        return state

    async def _summarize_requirement(self, doc: str) -> str:
        prompt = (
            "Summarize the following product requirement document in 2-3 sentences. "
            "Focus on: what features are described, what user flows exist, "
            "and what key decisions are implied.\n\n"
            "Return ONLY the summary, no extra text.\n\n"
            f"=== DOCUMENT ===\n{doc}"
        )
        return await self.call_llm(
            system_prompt="You are a requirement analyst.",
            user_prompt=prompt,
            max_tokens=1000,
        )

    async def _generate_exploration_plan(
        self, summary: str, feedback: str = ""
    ) -> dict[str, Any]:
        feedback_section = f"\n\nPrevious feedback: {feedback}" if feedback else ""

        system_prompt = (
            "You are a code exploration planner. Given a requirement summary, "
            "break it down into concrete code exploration tasks. "
            "Return a JSON object with: modules_to_explore, search_queries, "
            "priority_order, depth_hint."
        )

        user_prompt = (
            f"Requirement summary: {summary}{feedback_section}\n\n"
            "Generate an exploration plan. Focus on finding existing implementations, "
            "data models, API endpoints, and business logic related to the requirements."
        )

        raw = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            max_tokens=2000,
        )
        return json.loads(raw)
