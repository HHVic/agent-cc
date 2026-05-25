import asyncio
import pytest
from app.agents.base.agent import BaseAgent, AgentRegistry, WorkflowState
from app.core.llm import OpenAICompatibleClient


class _ExampleAgent(BaseAgent):
    """Concrete agent for testing."""

    async def process(self, state: WorkflowState) -> WorkflowState:
        state["processed"] = True
        self.add_message(state, "test_event", {"data": "hello"})
        return state


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_process_sets_state(self):
        state = WorkflowState()
        agent = _ExampleAgent("test", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_add_message_to_queue(self):
        queue = asyncio.Queue()
        state: WorkflowState = {"_message_queue": queue}
        agent = _ExampleAgent("test", llm_client=None)  # type: ignore
        agent.add_message(state, "test", {"key": "value"})
        assert len(state["messages"]) == 1
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg["type"] == "test"
        assert msg["agent"] == "test"

    @pytest.mark.asyncio
    async def test_add_message_without_queue(self):
        state: WorkflowState = {}
        agent = _ExampleAgent("test", llm_client=None)  # type: ignore
        agent.add_message(state, "test", {"key": "value"})
        assert len(state["messages"]) == 1


class TestAgentRegistry:
    def test_register_and_get(self):
        AgentRegistry.reset()
        agent = _ExampleAgent("reg_test", llm_client=None)  # type: ignore
        AgentRegistry.register(agent)
        assert AgentRegistry.get("reg_test") is agent

    def test_get_missing_returns_none(self):
        AgentRegistry.reset()
        assert AgentRegistry.get("missing") is None

    def test_all_returns_all_agents(self):
        AgentRegistry.reset()
        AgentRegistry.register(_ExampleAgent("a", llm_client=None))  # type: ignore
        AgentRegistry.register(_ExampleAgent("b", llm_client=None))  # type: ignore
        all_agents = AgentRegistry.all()
        assert len(all_agents) == 2

    def test_reset_clears(self):
        AgentRegistry.reset()
        AgentRegistry.register(_ExampleAgent("x", llm_client=None))  # type: ignore
        AgentRegistry.reset()
        assert AgentRegistry.all() == {}
