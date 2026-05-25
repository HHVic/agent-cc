from typing import Any, Required, TypedDict


class ExplorationRound(TypedDict):
    round: int
    files_explored: int
    functions_found: int


class ClarificationRound(TypedDict):
    round: int
    questions: list[dict[str, Any]]
    filtered_questions: list[dict[str, Any]]
    passed: bool


class CodeRef(TypedDict):
    file: str
    line: int
    snippet: str


class KnownImpl(TypedDict):
    description: str
    source: str


class UnresolvedGap(TypedDict):
    description: str
    priority: str


class ClarificationQuestion(TypedDict, total=False):
    id: str
    text: str
    category: str
    severity: str
    source_code_refs: list[CodeRef]
    requirement_ref: str
    suggested_options: list[str]


__all__ = [
    "ExplorationRound",
    "ClarificationRound",
    "CodeRef",
    "KnownImpl",
    "UnresolvedGap",
    "ClarificationQuestion",
]
