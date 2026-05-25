import json
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState


class Analyzer(BaseAgent):
    """Analyzes code exploration results against the requirement.

    Determines coverage score, identifies gaps, and provides feedback
    for the Planner to guide further exploration.
    """

    async def process(self, state: WorkflowState) -> WorkflowState:
        summary = state.get("requirement_doc_summary", "")
        code_snippets = state.get("code_snippets", [])
        exploration_rounds = state.get("exploration_rounds", [])

        if not summary or not code_snippets:
            state["analysis"] = {
                "coverage_score": 0.0,
                "covered_points": [],
                "uncovered_gaps": [],
                "needs_more_exploration": True,
                "exploration_feedback": "No code or summary available for analysis.",
            }
            self.add_message(state, "analysis_done", state["analysis"])
            return state

        analysis = await self._analyze_coverage(summary, code_snippets)
        state["analysis"] = analysis
        state["known_implementations"] = analysis.get("covered_points", [])
        state["unresolved_gaps"] = analysis.get("uncovered_gaps", [])

        self.add_message(state, "analysis_done", {
            "coverage_score": analysis["coverage_score"],
            "gaps_count": len(analysis["uncovered_gaps"]),
        })
        return state

    async def _analyze_coverage(
        self, summary: str, code_snippets: list[dict]
    ) -> dict[str, Any]:
        system_prompt = (
            "You are a code coverage analyst. Given a requirement summary and "
            "code snippets from the codebase, analyze how well the existing "
            "code covers the requirements. Identify covered points, uncovered gaps, "
            "and estimate a coverage score (0.0-1.0)."
        )

        user_prompt = (
            f"=== REQUIREMENT SUMMARY ===\n{summary}\n\n"
            f"=== CODE SNIPPETS ({len(code_snippets)} files) ===\n"
        )
        for snippet in code_snippets[:10]:
            user_prompt += f"\n--- {snippet['path']} ---\n{snippet['content'][:500]}\n"

        if len(code_snippets) > 10:
            user_prompt += f"\n... and {len(code_snippets) - 10} more files\n"

        user_prompt += (
            "\n\nReturn a JSON object with: coverage_score (float 0-1), "
            "covered_points (list of strings), uncovered_gaps (list of strings with format 'gap: description'), "
            "needs_more_exploration (bool), exploration_feedback (string for Planner)."
        )

        raw = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            max_tokens=4000,
        )
        return json.loads(raw)
