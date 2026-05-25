import json
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState


class Critic(BaseAgent):
    """Critiques generated questions for quality and product-relevance.

    Three-phase critique:
    1. Quality assessment (dedup, ambiguity, coverage, answerability)
    2. Dimension filtering (separate technical questions from product questions)
    3. Overall scoring (pass/fail with feedback)
    """

    async def process(self, state: WorkflowState) -> WorkflowState:
        questions = state.get("questions", [])
        if not questions:
            state["critic_result"] = {
                "filtered_questions": [],
                "tech_questions": [],
                "quality_feedback": "No questions to critique.",
                "passed": False,
            }
            self.add_message(state, "critic_reviewing", {"passed": False, "reason": "no_questions"})
            return state

        critique = await self._critique(questions, state)
        state["critic_result"] = critique
        state["filtered_questions"] = critique["filtered_questions"]
        state["tech_questions"] = critique["tech_questions"]

        if critique["passed"]:
            self.add_message(state, "critic_pass", {
                "count": len(critique["filtered_questions"]),
            })
        else:
            self.add_message(state, "critic_fail", {
                "feedback": critique["quality_feedback"],
            })

        return state

    async def _critique(
        self, questions: list[dict], state: WorkflowState
    ) -> dict[str, Any]:
        analysis = state.get("analysis", {})
        coverage = analysis.get("coverage_score", 0)

        system_prompt = (
            "You are a critical reviewer of product clarification questions. "
            "Evaluate questions across three dimensions: quality, product-relevance, "
            "and overall score. Filter out technical questions (for engineering reference). "
            "Return feedback for the Planner if more exploration is needed."
        )

        user_prompt = (
            f"=== GENERATED QUESTIONS ({len(questions)}) ===\n"
            + json.dumps(questions, indent=2, ensure_ascii=False)
            + f"\n\n=== COVERAGE SCORE: {coverage} ===\n\n"
            "Return a JSON object:\n"
            "{\n"
            '  "filtered_questions": [...],  // product-relevant questions only\n'
            '  "tech_questions": [...],  // technical questions for engineering reference\n'
            '  "quality_feedback": "feedback for Planner about what to explore next",\n'
            '  "passed": true/false\n'
            "}\n\n"
            "Criteria for passed=true:\n"
            "- At least 3 product-relevant questions\n"
            "- No duplicate or overly generic questions\n"
            "- Questions cover multiple dimensions (business, data, edge cases, exceptions)\n"
            f"- Coverage score >= {state.get('settings', {}).get('coverage_threshold', 0.7)}\n\n"
            "Criteria for failed:\n"
            "- Less than 3 quality questions\n"
            "- Questions are too generic or ambiguous\n"
            "- Coverage score too low\n"
            "- Missing key dimensions"
        )

        raw = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            max_tokens=4000,
        )
        return json.loads(raw)
