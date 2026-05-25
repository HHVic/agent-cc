import asyncio
import json
import uuid
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.base.agent import AgentRegistry, BaseAgent, WorkflowState
from app.agents.requirement_clarifier.planner import Planner
from app.agents.requirement_clarifier.explorer import Explorer
from app.agents.requirement_clarifier.analyzer import Analyzer
from app.agents.requirement_clarifier.question_gen import QuestionGen
from app.agents.requirement_clarifier.critic import Critic
from app.core.config import get_settings
from app.core.llm import OpenAICompatibleClient
from app.tools.base import ToolRegistry
from app.workflow.graph import ClarificationWorkflow

router = APIRouter()


class ClarificationRequest(BaseModel):
    session_id: str = ""
    requirement_doc: str
    target_repos: list[str] = []


class ClarificationResponse(BaseModel):
    session_id: str
    status: str
    questions: list[dict]
    tech_questions: list[dict]
    rounds: int


_event_queues: dict[str, asyncio.Queue] = {}
_session_states: dict[str, WorkflowState] = {}


def _get_or_create_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _event_queues:
        _event_queues[session_id] = asyncio.Queue()
    return _event_queues[session_id]


@router.post("/clarification/start")
async def start_clarification(req: ClarificationRequest):
    """Start a new clarification session."""
    session_id = req.session_id or str(uuid.uuid4())
    settings = get_settings()

    queue = asyncio.Queue()
    _get_or_create_queue(session_id)

    state: WorkflowState = {
        "session_id": session_id,
        "requirement_doc": req.requirement_doc,
        "target_repos": req.target_repos,
        "base_dir": ".",
        "_message_queue": queue,
        "settings": {
            "max_exploration_rounds": settings.max_exploration_rounds,
            "max_clarification_rounds": settings.max_clarification_rounds,
            "min_questions": settings.min_questions_threshold,
            "coverage_threshold": settings.coverage_threshold,
        },
    }

    # Run workflow in background
    asyncio.create_task(_run_workflow(session_id, state))

    return ClarificationResponse(
        session_id=session_id,
        status="started",
        questions=[],
        tech_questions=[],
        rounds=0,
    )


async def _run_workflow(session_id: str, state: WorkflowState) -> None:
    """Run the clarification workflow in the background."""
    try:
        settings = get_settings()
        llm_client = OpenAICompatibleClient(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
        )

        planner = Planner("planner", llm_client=llm_client)
        explorer = Explorer("explorer", llm_client=llm_client, tools={})
        analyzer = Analyzer("analyzer", llm_client=llm_client)
        question_gen = QuestionGen("question_gen", llm_client=llm_client)
        critic = Critic("critic", llm_client=llm_client)

        workflow = ClarificationWorkflow(
            planner=planner,
            explorer=explorer,
            analyzer=analyzer,
            question_gen=question_gen,
            critic=critic,
            max_exploration_rounds=settings.max_exploration_rounds,
            max_clarification_rounds=settings.max_clarification_rounds,
            min_questions=settings.min_questions_threshold,
            coverage_threshold=settings.coverage_threshold,
        )

        await workflow.run(state)
    except Exception as e:
        queue = _get_or_create_queue(session_id)
        queue.put_nowait({"type": "error", "content": str(e)})


@router.get("/clarification/{session_id}/events")
async def stream_events(session_id: str):
    """Stream SSE events for a clarification session."""
    queue: asyncio.Queue = _get_or_create_queue(session_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # Keep-alive ping
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/clarification/{session_id}/status")
async def get_status(session_id: str):
    """Get the current status of a clarification session."""
    # In production, check database
    return {"session_id": session_id, "status": "running"}
