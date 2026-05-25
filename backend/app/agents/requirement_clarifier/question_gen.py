import json
import uuid
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState


class QuestionGen(BaseAgent):
    """Generates clarification questions from analysis results.

    Each question includes source code references and requirement references
    for traceability.
    """

    async def process(self, state: WorkflowState) -> WorkflowState:
        summary = state.get("requirement_doc_summary", "")
        analysis = state.get("analysis", {})
        code_snippets = state.get("code_snippets", [])
        gaps = analysis.get("uncovered_gaps", [])
        known = analysis.get("covered_points", [])

        if not summary or not gaps:
            state["questions"] = []
            self.add_message(state, "questions_generated", [])
            return state

        questions = await self._generate_questions(summary, gaps, known, code_snippets)
        state["questions"] = questions

        self.add_message(state, "questions_generated", {
            "count": len(questions),
            "categories": list(set(q.get("category", "unknown") for q in questions)),
        })
        return state

    async def _generate_questions(
        self,
        summary: str,
        gaps: list[str],
        covered: list[str],
        code_snippets: list[dict],
    ) -> list[dict]:
        system_prompt = (
            "You are a product requirements analyst. Given uncovered gaps in the "
            "requirement-to-code mapping, generate specific, actionable clarification "
            "questions for the product team. Each question must be traceable to "
            "source code and the requirement document."
        )

        user_prompt = (
            f"=== REQUIREMENT SUMMARY ===\n{summary}\n\n"
            f"=== COVERED AREAS ===\n" + "\n".join(f"- {g}" for g in covered) + "\n\n"
            f"=== UNCOVERED GAPS ===\n" + "\n".join(f"- {g}" for g in gaps) + "\n\n"
            f"=== RELEVANT CODE ({len(code_snippets)} snippets) ===\n"
        )
        for snippet in code_snippets[:5]:
            user_prompt += f"\n--- {snippet['path']} ---\n{snippet['content'][:300]}\n"

        user_prompt += (
            "\n\nReturn a JSON array of ClarificationQuestion objects:\n"
            "[{\n"
            '  "id": "uuid string",\n'
            '  "text": "specific question",\n'
            '  "category": "business_flow|data_model|edge_case|exception_handling",\n'
            '  "severity": "critical|important|normal|low",\n'
            '  "source_code_refs": [{"file": "path", "line": 42, "snippet": "code"}],\n'
            '  "requirement_ref": "which part of the requirement this relates to",\n'
            '  "suggested_options": ["option1", "option2"]\n'
            "}]"
        )

        raw = await self.call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True,
            max_tokens=6000,
        )
        questions = json.loads(raw)

        # Ensure each question has a valid UUID
        for q in questions:
            if not q.get("id"):
                q["id"] = str(uuid.uuid4())

        return questions
