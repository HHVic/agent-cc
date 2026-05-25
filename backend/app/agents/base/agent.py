import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from app.core.llm import OpenAICompatibleClient


class WorkflowState(dict):
    """Mutable state dict passed through the agent pipeline.

    Extends dict for LangGraph compatibility while providing type hints
    via TypedDict-style annotations in docstrings.
    """
    pass


class BaseAgent(ABC):
    """Base class for all agents in the system.

    Each agent receives a WorkflowState, processes it, and returns the modified state.
    Agents use an asyncio.Queue to push SSE events during processing.
    """

    def __init__(
        self,
        name: str,
        llm_client: OpenAICompatibleClient,
        tools: Optional[dict[str, Any]] = None,
    ):
        self.name = name
        self.llm = llm_client
        self.tools = tools or {}
        self.logger = logging.getLogger(f"Agent.{name}")

    @abstractmethod
    async def process(self, state: WorkflowState) -> WorkflowState:
        """Process state and return updated state."""
        ...

    async def call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        return await self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def add_message(
        self, state: WorkflowState, event_type: str, content: Any
    ) -> None:
        """Push an SSE event into the state's message queue."""
        message = {
            "type": event_type,
            "agent": self.name,
            "timestamp": datetime.now().isoformat(),
            "content": content,
        }
        state.setdefault("messages", []).append(message)
        queue: Optional[asyncio.Queue] = state.get("_message_queue")
        if queue is not None:
            try:
                queue.put_nowait(message)
            except Exception as e:
                self.logger.warning("Failed to push message to queue: %s", e)


class AgentRegistry:
    """Global registry for agent instances."""

    _agents: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        return cls._agents.get(name)

    @classmethod
    def all(cls) -> dict[str, BaseAgent]:
        return dict(cls._agents)

    @classmethod
    def reset(cls) -> None:
        cls._agents.clear()
