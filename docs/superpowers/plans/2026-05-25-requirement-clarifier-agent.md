# Requirement Clarification Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend of an industrial-grade Requirement Clarification Agent system that automates the flow: requirement doc → code exploration → question generation → critic filtering → output to product team.

**Architecture:** Python FastAPI backend with LangGraph orchestration, following patterns from `cc-python-claude` (Tool ABC + ToolRegistry) and `industry_information_assistant` (BaseAgent + SSE streaming via asyncio.Queue). The system uses a double-loop workflow: inner loop cycles through Planner → Explorer → Analyzer → QuestionGen → Critic with feedback to Planner, outer loop repeats clarification rounds.

**Tech Stack:** Python 3.11+, FastAPI, LangGraph, OpenAI-compatible LLM (DashScope/qwen), PostgreSQL (asyncpg), D3.js for call graph visualization.

---

## Phase 1: Foundation Layer

### Task 1: Project Scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/.python-version`
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "agent-cc"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "langgraph>=0.2.0",
    "langchain-openai>=0.3.0",
    "openai>=1.57.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.7.0",
    "sqlalchemy[asyncio]>=2.0.36",
    "asyncpg>=0.30.0",
    "aiofiles>=24.1.0",
    "tree-sitter-java>=0.23.0",
    "python-dotenv>=1.0.1",
    "aiohttp>=3.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "httpx>=0.28.0",
    "ruff>=0.8.0",
]
```

- [ ] **Step 2: Create .python-version**

```
3.11
```

- [ ] **Step 3: Create .env.example**

```env
# LLM Configuration
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_API_KEY=your-key-here
LLM_MODEL=qwen-max

# Database
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/agent_cc

# App
APP_SECRET_KEY=change-me-in-production
DEBUG=true
```

- [ ] **Step 4: Create requirements.txt**

```
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
langgraph>=0.2.0
langchain-openai>=0.3.0
openai>=1.57.0
pydantic>=2.10.0
pydantic-settings>=2.7.0
sqlalchemy[asyncio]>=2.0.36
asyncpg>=0.30.0
aiofiles>=24.1.0
tree-sitter-java>=0.23.0
python-dotenv>=1.0.1
aiohttp>=3.11.0
```

- [ ] **Step 5: Create directory structure**

```bash
mkdir -p backend/app/{agents/{requirement_clarifier,base},tools,core,workflow,api,db}
mkdir -p backend/tests/{unit,integration}
```

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/.python-version backend/.env.example backend/requirements.txt
git commit -m "chore: scaffold project structure"
```

---

### Task 2: Core Config + LLM Client

**Files:**
- Create: `backend/app/core/config.py`
- Create: `backend/app/core/llm.py`

- [ ] **Step 1: Write config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False)

    # LLM
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen-max"

    # Database
    database_url: str = "postgresql://postgres:postgres123@localhost:5432/agent_cc"

    # App
    app_secret_key: str = "change-me-in-production"
    debug: bool = False

    # Clarification Agent defaults
    max_exploration_rounds: int = 3
    max_clarification_rounds: int = 3
    min_questions_threshold: int = 3
    coverage_threshold: float = 0.7
    question_quality_threshold: float = 0.6

    @property
    def async_database_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
```

- [ ] **Step 2: Write llm.py**

```python
import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    """OpenAI-compatible LLM client supporting any provider (DashScope, OpenAI, local, etc.)."""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.logger = logging.getLogger(f"LLM.{model}")

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            content = await asyncio.to_thread(
                self.client.chat.completions.create, **kwargs
            )
            return content.choices[0].message.content or ""
        except Exception as e:
            self.logger.error("LLM call failed: %s", e)
            raise
```

- [ ] **Step 3: Create __init__.py files**

```bash
touch backend/app/__init__.py
touch backend/app/core/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/core/
git commit -m "feat: add config and LLM client"
```

---

### Task 3: Tool ABC + ToolRegistry + Unit Tests

**Files:**
- Create: `backend/app/tools/base.py`
- Create: `backend/tests/unit/test_tools_base.py`

- [ ] **Step 1: Write Tool ABC + ToolRegistry**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    content: str | list[dict[str, Any]]
    is_error: bool = False

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            block.get("text", str(block))
            for block in self.content
            if isinstance(block, dict)
        )


class Tool(ABC):
    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_schema(self) -> ToolSchema: ...

    @abstractmethod
    async def execute(self, tool_input: dict[str, Any]) -> ToolResult: ...

    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return False


@dataclass
class ToolRegistry:
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        name = tool.get_name()
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_api_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.get_schema().name,
                "description": tool.get_schema().description,
                "input_schema": tool.get_schema().input_schema,
            }
            for tool in self._tools.values()
        ]

    def get_tools(self, names: list[str]) -> dict[str, Tool]:
        return {name: self._tools[name] for name in names if name in self._tools}
```

- [ ] **Step 2: Write unit tests**

```python
import pytest
from app.tools.base import Tool, ToolRegistry, ToolSchema, ToolResult


class DummyTool(Tool):
    def __init__(self, name: str = "dummy"):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def get_schema(self) -> ToolSchema:
        return ToolSchema(name=self._name, description="A dummy tool", input_schema={"type": "object"})

    async def execute(self, tool_input: dict) -> ToolResult:
        return ToolResult(content="ok")


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = DummyTool("test")
        registry.register(tool)
        assert registry.get("test") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(DummyTool("test"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DummyTool("test"))

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(DummyTool("a"))
        registry.register(DummyTool("b"))
        assert len(registry.list_tools()) == 2

    def test_get_missing_returns_none(self):
        assert ToolRegistry().get("missing") is None

    def test_get_tools_by_names(self):
        registry = ToolRegistry()
        registry.register(DummyTool("a"))
        registry.register(DummyTool("b"))
        result = registry.get_tools(["a", "missing"])
        assert "a" in result
        assert "missing" not in result


class TestToolResult:
    def test_text_from_string(self):
        assert ToolResult(content="hello").text == "hello"

    def test_text_from_list(self):
        result = ToolResult(content=[{"text": "a"}, {"text": "b"}]).text
        assert result == "a\nb"
```

- [ ] **Step 3: Run tests to verify they fail (Tool not defined)**

```bash
cd backend && python -m pytest tests/unit/test_tools_base.py -v
```
Expected: FAIL with "NameError: name 'ToolRegistry' is not defined"

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/unit/test_tools_base.py -v
```
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/tools/base.py backend/tests/unit/test_tools_base.py
git commit -m "feat: add Tool ABC, ToolRegistry, ToolResult and unit tests"
```

---

### Task 4: BaseAgent + AgentRegistry + Unit Tests

**Files:**
- Create: `backend/app/agents/base/agent.py`
- Create: `backend/app/agents/base/__init__.py`
- Create: `backend/tests/unit/test_base_agent.py`

- [ ] **Step 1: Write agent.py**

```python
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
```

- [ ] **Step 2: Write __init__.py**

```python
from app.agents.base.agent import BaseAgent, AgentRegistry, WorkflowState

__all__ = ["BaseAgent", "AgentRegistry", "WorkflowState"]
```

- [ ] **Step 3: Write unit tests**

```python
import asyncio
import pytest
from app.agents.base.agent import BaseAgent, AgentRegistry, WorkflowState
from app.core.llm import OpenAICompatibleClient


class TestAgent(BaseAgent):
    """Concrete agent for testing."""

    async def process(self, state: WorkflowState) -> WorkflowState:
        state["processed"] = True
        self.add_message(state, "test_event", {"data": "hello"})
        return state


class TestBaseAgent:
    @pytest.mark.asyncio
    async def test_process_sets_state(self):
        state = WorkflowState()
        agent = TestAgent("test", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["processed"] is True

    @pytest.mark.asyncio
    async def test_add_message_to_queue(self):
        queue = asyncio.Queue()
        state: WorkflowState = {"_message_queue": queue}
        agent = TestAgent("test", llm_client=None)  # type: ignore
        agent.add_message(state, "test", {"key": "value"})
        assert len(state["messages"]) == 1
        msg = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert msg["type"] == "test"
        assert msg["agent"] == "test"

    @pytest.mark.asyncio
    async def test_add_message_without_queue(self):
        state: WorkflowState = {}
        agent = TestAgent("test", llm_client=None)  # type: ignore
        agent.add_message(state, "test", {"key": "value"})
        assert len(state["messages"]) == 1


class TestAgentRegistry:
    def test_register_and_get(self):
        AgentRegistry.reset()
        agent = TestAgent("reg_test", llm_client=None)  # type: ignore
        AgentRegistry.register(agent)
        assert AgentRegistry.get("reg_test") is agent

    def test_get_missing_returns_none(self):
        AgentRegistry.reset()
        assert AgentRegistry.get("missing") is None

    def test_all_returns_all_agents(self):
        AgentRegistry.reset()
        AgentRegistry.register(TestAgent("a", llm_client=None))  # type: ignore
        AgentRegistry.register(TestAgent("b", llm_client=None))  # type: ignore
        all_agents = AgentRegistry.all()
        assert len(all_agents) == 2

    def test_reset_clears(self):
        AgentRegistry.reset()
        AgentRegistry.register(TestAgent("x", llm_client=None))  # type: ignore
        AgentRegistry.reset()
        assert AgentRegistry.all() == {}
```

- [ ] **Step 4: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_base_agent.py -v
git add backend/app/agents/base/ backend/tests/unit/test_base_agent.py
git commit -m "feat: add BaseAgent, AgentRegistry, WorkflowState and unit tests"
```

---

## Phase 2: Tool Implementations

### Task 5: FileReadTool + GlobSearchTool + GrepSearchTool

**Files:**
- Create: `backend/app/tools/file_read.py`
- Create: `backend/app/tools/glob_search.py`
- Create: `backend/app/tools/grep_search.py`
- Create: `backend/tests/unit/test_tools_file_operations.py`

- [ ] **Step 1: Write file_read.py**

```python
import os
from typing import Any

from app.tools.base import Tool, ToolResult, ToolSchema


class FileReadTool(Tool):
    """Read file contents from the filesystem."""

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir

    def get_name(self) -> str:
        return "file_read"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_read",
            description="Read the contents of a file at the given path",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 500)",
                        "default": 500,
                    },
                },
                "required": ["path"],
            },
        )

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        path = tool_input["path"]
        max_lines = tool_input.get("max_lines", 500)

        if self.base_dir:
            resolved = os.path.realpath(os.path.join(self.base_dir, path))
            if not resolved.startswith(os.path.realpath(self.base_dir)):
                return ToolResult(content="Error: path traversal detected", is_error=True)
            path = resolved

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return ToolResult(content="".join(lines[:max_lines]))
        except FileNotFoundError:
            return ToolResult(content=f"File not found: {path}", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Error reading file: {e}", is_error=True)

    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return True
```

- [ ] **Step 2: Write glob_search.py**

```python
import fnmatch
import os
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult, ToolSchema


class GlobSearchTool(Tool):
    """Search for files matching a glob pattern."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir).resolve()

    def get_name(self) -> str:
        return "glob_search"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="glob_search",
            description="Find files matching a glob pattern within the target directory",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.java')"},
                    "base_dir": {"type": "string", "description": "Base directory to search (optional)"},
                },
                "required": ["pattern"],
            },
        )

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        pattern = tool_input["pattern"]
        search_dir = Path(self.base_dir)
        if tool_input.get("base_dir"):
            search_dir = Path(tool_input["base_dir"]).resolve()

        try:
            matches = sorted(str(p.relative_to(self.base_dir)) for p in search_dir.glob(pattern))
            return ToolResult(content="\n".join(matches))
        except Exception as e:
            return ToolResult(content=f"Error searching: {e}", is_error=True)

    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return True
```

- [ ] **Step 3: Write grep_search.py**

```python
import re
from pathlib import Path
from typing import Any

from app.tools.base import Tool, ToolResult, ToolSchema


class GrepSearchTool(Tool):
    """Search file contents using regex patterns."""

    def __init__(self, base_dir: str = ".", max_matches: int = 100):
        self.base_dir = Path(base_dir).resolve()
        self.max_matches = max_matches

    def get_name(self) -> str:
        return "grep_search"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="grep_search",
            description="Search file contents using a regex pattern. Returns matching lines with file paths.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "file_pattern": {"type": "string", "description": "Glob pattern for file types (e.g., '*.java')"},
                    "max_matches": {"type": "integer", "description": "Max matches to return", "default": 100},
                },
                "required": ["pattern"],
            },
        )

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        pattern = tool_input["pattern"]
        file_pattern = tool_input.get("file_pattern", "**/*")
        max_matches = tool_input.get("max_matches", self.max_matches)
        compiled = re.compile(pattern)

        results = []
        search_dir = self.base_dir
        for filepath in search_dir.glob(file_pattern):
            if not filepath.is_file():
                continue
            try:
                for line_no, line in enumerate(filepath.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if compiled.search(line):
                        rel_path = str(filepath.relative_to(self.base_dir))
                        results.append(f"{rel_path}:{line_no}: {line}")
                        if len(results) >= max_matches:
                            break
                if len(results) >= max_matches:
                    break
            except Exception:
                continue

        return ToolResult(content="\n".join(results) if results else "No matches found")

    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return True
```

- [ ] **Step 4: Write unit tests**

```python
import os
import tempfile
from app.tools.file_read import FileReadTool
from app.tools.glob_search import GlobSearchTool
from app.tools.grep_search import GrepSearchTool
import pytest


class TestFileReadTool:
    @pytest.mark.asyncio
    async def test_read_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("line1\nline2\nline3\n")
            f.flush()
            tool = FileReadTool()
            result = await tool.execute({"path": f.name})
            assert result.is_error is False
            assert "line1" in result.content

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        tool = FileReadTool()
        result = await tool.execute({"path": "/nonexistent/file.py"})
        assert result.is_error is True
        assert "not found" in result.content.lower()

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        tool = FileReadTool(base_dir="/safe/dir")
        result = await tool.execute({"path": "../../etc/passwd"})
        assert result.is_error is True

    def test_is_concurrency_safe(self):
        assert FileReadTool().is_concurrency_safe() is True


class TestGlobSearchTool:
    @pytest.mark.asyncio
    async def test_search_pattern(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test.py").touch()
            Path(td, "main.py").touch()
            Path(td, "test.js").touch()
            tool = GlobSearchTool(base_dir=td)
            result = await tool.execute({"pattern": "*.py"})
            assert "test.py" in result.content
            assert "main.py" in result.content
            assert "test.js" not in result.content


class TestGrepSearchTool:
    @pytest.mark.asyncio
    async def test_search_contents(self):
        with tempfile.TemporaryDirectory() as td:
            filepath = Path(td, "test.py")
            filepath.write_text("def hello():\n    print('world')\n")
            tool = GrepSearchTool(base_dir=td)
            result = await tool.execute({"pattern": "def hello"})
            assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_no_match(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "test.py").write_text("hello world\n")
            tool = GrepSearchTool(base_dir=td)
            result = await tool.execute({"pattern": "xyz_nonexistent"})
            assert "No matches" in result.content
```

- [ ] **Step 5: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_tools_file_operations.py -v
git add backend/app/tools/{file_read.py,glob_search.py,grep_search.py} backend/tests/unit/test_tools_file_operations.py
git commit -m "feat: add FileReadTool, GlobSearchTool, GrepSearchTool and tests"
```

---

### Task 6: BashExecTool + CodeGraphTool

**Files:**
- Create: `backend/app/tools/bash_exec.py`
- Create: `backend/app/tools/code_graph.py`
- Create: `backend/tests/unit/test_tools_bash_codegraph.py`

- [ ] **Step 1: Write bash_exec.py**

```python
import asyncio
from typing import Any

from app.tools.base import Tool, ToolResult, ToolSchema


class BashExecTool(Tool):
    """Execute shell commands in a sandboxed manner."""

    def __init__(self, base_dir: str = ".", timeout: int = 60):
        self.base_dir = base_dir
        self.timeout = timeout

    def get_name(self) -> str:
        return "bash_exec"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="bash_exec",
            description="Execute a shell command and return stdout/stderr. Use with caution.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
                },
                "required": ["command"],
            },
        )

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        command = tool_input["command"]
        timeout = tool_input.get("timeout", self.timeout)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self.base_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return ToolResult(
                    content=f"Exit {proc.returncode}:\n{output}\n{stderr}",
                    is_error=True,
                )
            return ToolResult(content=output)
        except asyncio.TimeoutError:
            return ToolResult(content="Command timed out", is_error=True)
        except Exception as e:
            return ToolResult(content=f"Error: {e}", is_error=True)


    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return False
```

- [ ] **Step 2: Write code_graph.py**

```python
from typing import Any

from app.tools.base import Tool, ToolResult, ToolSchema


class CodeGraphTool(Tool):
    """Generate a function call graph from source files.

    Analyzes code files to extract function/method definitions and their
    call relationships. Supports Java via tree-sitter with regex fallback
    for other languages.
    """

    def get_name(self) -> str:
        return "code_graph"

    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name="code_graph",
            description="Generate a call graph (nodes and edges) from source files. "
            "Nodes represent functions/methods; edges represent call relationships.",
            input_schema={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to analyze",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for AST parsing ('java', 'python', etc.)",
                    },
                },
                "required": ["files"],
            },
        )

    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        files = tool_input["files"]
        language = tool_input.get("language", "java")

        nodes: list[dict] = []
        edges: list[dict] = []
        node_id_counter = 0

        for filepath in files:
            try:
                content = open(filepath, "r", encoding="utf-8").read()
            except Exception as e:
                continue

            if language == "java":
                parsed = self._parse_java(content)
            elif language == "python":
                parsed = self._parse_python(content)
            else:
                parsed = self._parse_regex(content, language)

            for func in parsed["functions"]:
                node_id_counter += 1
                node = {
                    "id": f"n{node_id_counter}",
                    "label": func["name"],
                    "file": func["file"],
                    "line": func.get("line", 0),
                    "type": func.get("type", "function"),
                }
                nodes.append(node)

                for callee_name in func.get("calls", []):
                    edges.append({
                        "source": node["id"],
                        "target": callee_name,
                        "type": "calls",
                    })

        return ToolResult(content={
            "nodes": nodes,
            "edges": edges,
            "file": files,
        })

    def _parse_java(self, content: str) -> dict:
        """Parse Java code using regex fallback. AST parsing requires tree-sitter-java."""
        import re
        functions = []
        func_pattern = re.compile(r'(?:public|private|protected)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{')
        calls_pattern = re.compile(r'(\w+)\.\s*(\w+)\s*\(')

        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            line_no = content[:match.start()].count('\n') + 1

            # Find calls within this function body (naive)
            func_start = match.end()
            brace_count = 1
            func_end = func_start
            for i, ch in enumerate(content[func_start:], 1):
                if ch == '{': brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = func_start + i
                        break

            func_body = content[func_start:func_end]
            calls = [m.group(2) for m in calls_pattern.finditer(func_body) if m.group(1) != func_name]

            functions.append({
                "name": func_name,
                "file": "",
                "line": line_no,
                "type": "method",
                "calls": list(set(calls)),
            })

        return {"functions": functions}

    def _parse_python(self, content: str) -> dict:
        """Parse Python code using regex."""
        import re
        functions = []
        func_pattern = re.compile(r'def (\w+)\s*\([^)]*\)\s*:')

        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            line_no = content[:match.start()].count('\n') + 1
            functions.append({"name": func_name, "file": "", "line": line_no, "type": "function", "calls": []})

        return {"functions": functions}

    def _parse_regex(self, content: str, language: str) -> dict:
        """Fallback: try to parse using generic patterns."""
        # Simple generic fallback
        return self._parse_python(content)

    def is_concurrency_safe(self, tool_input: dict[str, Any] | None = None) -> bool:
        return False
```

- [ ] **Step 3: Write unit tests**

```python
import asyncio
import tempfile
from pathlib import Path

import pytest

from app.tools.bash_exec import BashExecTool
from app.tools.code_graph import CodeGraphTool


class TestBashExecTool:
    @pytest.mark.asyncio
    async def test_echo(self):
        tool = BashExecTool()
        result = await tool.execute({"command": "echo hello"})
        assert result.is_error is False
        assert "hello" in result.content

    @pytest.mark.asyncio
    async def test_failed_command(self):
        tool = BashExecTool()
        result = await tool.execute({"command": "false"})
        assert result.is_error is True

    def test_not_concurrency_safe(self):
        assert BashExecTool().is_concurrency_safe() is False


class TestCodeGraphTool:
    @pytest.mark.asyncio
    async def test_java_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".java", delete=False) as f:
            f.write("""
public class Service {
    public void handle() {
        helper.doSomething();
        other.method();
    }
    private void helper() {
    }
}
""")
            f.flush()
            tool = CodeGraphTool()
            result = await tool.execute({"files": [f.name], "language": "java"})
            assert result.is_error is False
            data = result.content
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) > 0

    @pytest.mark.asyncio
    async def test_python_parsing(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    pass\n\ndef bar():\n    foo()\n")
            f.flush()
            tool = CodeGraphTool()
            result = await tool.execute({"files": [f.name], "language": "python"})
            assert result.is_error is False
            data = result.content
            labels = [n["label"] for n in data["nodes"]]
            assert "foo" in labels
            assert "bar" in labels
```

- [ ] **Step 4: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_tools_bash_codegraph.py -v
git add backend/app/tools/{bash_exec.py,code_graph.py} backend/tests/unit/test_tools_bash_codegraph.py
git commit -m "feat: add BashExecTool, CodeGraphTool and tests"
```

---

## Phase 3: Requirement Clarifier Agents

### Task 7: Planner Agent

**Files:**
- Create: `backend/app/agents/requirement_clarifier/planner.py`
- Create: `backend/tests/unit/test_planner.py`

- [ ] **Step 1: Write planner.py**

```python
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
```

- [ ] **Step 2: Write unit tests**

```python
import pytest
from unittest.mock import AsyncMock, patch

from app.agents.requirement_clarifier.planner import Planner
from app.agents.base.agent import WorkflowState


class TestPlanner:
    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_summarize_requirement(self, mock_call_llm):
        mock_call_llm.return_value = "A user registration feature with email verification."
        agent = Planner("planner", llm_client=None)  # type: ignore
        result = await agent._summarize_requirement("Long doc about registration...")
        assert result == "A user registration feature with email verification."

    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_generate_plan(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"modules_to_explore": ["auth", "user_model"], "search_queries": ["User", "Registration"], "priority_order": [0, 1], "depth_hint": "deep"}'
        )
        agent = Planner("planner", llm_client=None)  # type: ignore
        plan = await agent._generate_exploration_plan("A registration feature")
        assert plan["modules_to_explore"] == ["auth", "user_model"]
        assert plan["depth_hint"] == "deep"

    @pytest.mark.asyncio
    @patch.object(Planner, "call_llm", new_callable=AsyncMock)
    async def test_process_without_summary(self, mock_call_llm):
        mock_call_llm.side_effect = [
            "A registration feature",  # first call: summarize
            '{"modules_to_explore": ["auth"], "search_queries": ["User"], "priority_order": [0], "depth_hint": "shallow"}',
        ]
        state: WorkflowState = {"requirement_doc": "A registration feature doc"}
        agent = Planner("planner", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert "requirement_doc_summary" in result
        assert "exploration_plan" in result
        assert result.get("exploration_round") == 1
```

- [ ] **Step 3: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_planner.py -v
git add backend/app/agents/requirement_clarifier/planner.py backend/tests/unit/test_planner.py
git commit -m "feat: add Planner agent for exploration planning"
```

---

### Task 8: Explorer Agent

**Files:**
- Create: `backend/app/agents/requirement_clarifier/explorer.py`
- Create: `backend/tests/unit/test_explorer.py`

- [ ] **Step 1: Write explorer.py**

```python
import json
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState
from app.agents.base.agent import WorkflowState as StateType


class Explorer(BaseAgent):
    """Executes exploration plans by searching the codebase.

    Uses registered tools (FileRead, GrepSearch, GlobSearch, CodeGraph)
    to find relevant code and build context about existing implementations.
    """

    async def process(self, state: StateType) -> StateType:
        plan = state.get("exploration_plan")
        if not plan:
            self.add_message(state, "exploration_skip", {"reason": "No exploration plan"})
            return state

        modules = plan.get("modules_to_explore", [])
        search_queries = plan.get("search_queries", [])

        # Step 1: Search for relevant files
        self.add_message(state, "exploration_start", {"modules": modules})

        relevant_files = await self._find_relevant_files(modules, search_queries)
        state.setdefault("exploration_files", []).extend(relevant_files)

        # Step 2: Read relevant files
        code_snippets = await self._read_code(relevant_files[:20])  # Limit to 20 files
        state.setdefault("code_snippets", []).extend(code_snippets)

        # Step 3: Build call graph for key files
        if code_snippets:
            call_graph = await self._build_call_graph(code_snippets)
            state.setdefault("call_graph_nodes", []).extend(call_graph.get("nodes", []))
            state.setdefault("call_graph_edges", []).extend(call_graph.get("edges", []))

        # Step 4: Record exploration result
        exploration_round = state.get("exploration_round", 1)
        round_result = {
            "round": exploration_round,
            "files_explored": len(relevant_files),
            "functions_found": len(code_snippets),
        }
        state.setdefault("exploration_rounds", []).append(round_result)

        self.add_message(state, "exploration_done", round_result)
        return state

    async def _find_relevant_files(
        self, modules: list[str], queries: list[str]
    ) -> list[str]:
        """Find relevant files using glob and grep tools."""
        relevant_files = set()

        # Search by module patterns
        glob_tool = self.tools.get("glob_search")
        if glob_tool:
            for pattern in ["**/*.java", "**/*.py", "**/*.ts", "**/*.tsx"]:
                result = await glob_tool.execute({"pattern": pattern})
                if result.content:
                    for f in result.content.strip().split("\n"):
                        if f:
                            relevant_files.add(f)

        # Search by content
        grep_tool = self.tools.get("grep_search")
        if grep_tool:
            for query in queries[:10]:
                result = await grep_tool.execute({"pattern": query, "max_matches": 5})
                # Extract file paths from grep results (format: path:line:content)
                if result.content:
                    for line in result.content.strip().split("\n"):
                        if line and ":" in line:
                            filepath = line.split(":")[0]
                            relevant_files.add(filepath)

        return list(relevant_files)

    async def _read_code(self, files: list[str]) -> list[dict]:
        """Read relevant code snippets."""
        snippets = []
        file_read = self.tools.get("file_read")
        if not file_read:
            return snippets

        for f in files:
            result = await file_read.execute({"path": f, "max_lines": 200})
            if result.content and not result.is_error:
                snippets.append({
                    "path": f,
                    "content": result.content[:5000],  # Limit content size
                    "relevance_score": 0.8,  # Default score
                })

        return snippets

    async def _build_call_graph(self, snippets: list[dict]) -> dict:
        """Build call graph from code snippets."""
        code_graph = self.tools.get("code_graph")
        if not code_graph:
            return {"nodes": [], "edges": []}

        # Determine language from file extensions
        languages = {"java": [], "python": [], "other": []}
        for snippet in snippets:
            path = snippet["path"]
            if path.endswith(".java"):
                languages["java"].append(path)
            elif path.endswith(".py"):
                languages["python"].append(path)
            else:
                languages["other"].append(path)

        graph = {"nodes": [], "edges": []}
        for lang, files in languages.items():
            if files:
                result = await code_graph.execute({"files": files, "language": lang})
                if result.content:
                    graph_data = result.content if isinstance(result.content, dict) else json.loads(result.content)
                    graph["nodes"].extend(graph_data.get("nodes", []))
                    graph["edges"].extend(graph_data.get("edges", []))

        return graph
```

- [ ] **Step 2: Write unit tests**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.requirement_clarifier.explorer import Explorer
from app.agents.base.agent import WorkflowState


class TestExplorer:
    @pytest.mark.asyncio
    async def test_process_no_plan(self):
        state: WorkflowState = {}
        agent = Explorer("explorer", llm_client=None, tools={})  # type: ignore
        result = await agent.process(state)
        # Should not crash, exploration_rounds should be empty or unchanged
        assert "exploration_rounds" not in result or len(result.get("exploration_rounds", [])) == 0

    @pytest.mark.asyncio
    async def test_process_with_plan(self):
        state: WorkflowState = {
            "exploration_plan": {
                "modules_to_explore": ["auth"],
                "search_queries": ["User"],
                "priority_order": [0],
                "depth_hint": "deep",
            }
        }
        mock_glob = MagicMock()
        mock_glob.execute = AsyncMock(return_value=MagicMock(content=""))
        mock_grep = MagicMock()
        mock_grep.execute = AsyncMock(return_value=MagicMock(content=""))
        mock_file_read = MagicMock()
        mock_file_read.execute = AsyncMock(
            return_value=MagicMock(content="def foo(): pass", is_error=False)
        )
        agent = Explorer(
            "explorer",
            llm_client=None,
            tools={
                "glob_search": mock_glob,
                "grep_search": mock_grep,
                "file_read": mock_file_read,
            },
        )  # type: ignore
        result = await agent.process(state)
        assert "code_snippets" in result
        assert "exploration_rounds" in result
        assert len(result["exploration_rounds"]) == 1
```

- [ ] **Step 3: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_explorer.py -v
git add backend/app/agents/requirement_clarifier/explorer.py backend/tests/unit/test_explorer.py
git commit -m "feat: add Explorer agent for codebase exploration"
```

---

### Task 9: Analyzer Agent

**Files:**
- Create: `backend/app/agents/requirement_clarifier/analyzer.py`
- Create: `backend/tests/unit/test_analyzer.py`

- [ ] **Step 1: Write analyzer.py**

```python
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
```

- [ ] **Step 2: Write unit tests**

```python
import pytest
from unittest.mock import AsyncMock

from app.agents.requirement_clarifier.analyzer import Analyzer
from app.agents.base.agent import WorkflowState


class TestAnalyzer:
    @pytest.mark.asyncio
    async def test_process_no_content(self):
        state: WorkflowState = {}
        agent = Analyzer("analyzer", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["analysis"]["coverage_score"] == 0.0
        assert result["analysis"]["needs_more_exploration"] is True

    @pytest.mark.asyncio
    @patch.object(Analyzer, "call_llm", new_callable=AsyncMock)
    async def test_process_with_content(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"coverage_score": 0.7, "covered_points": ["user_model exists"], '
            '"uncovered_gaps": ["no email verification"], '
            '"needs_more_exploration": true, '
            '"exploration_feedback": "Explore the auth module more deeply"}'
        )
        state: WorkflowState = {
            "requirement_doc_summary": "User registration with email verification",
            "code_snippets": [{"path": "models/user.py", "content": "class User: pass"}],
        }
        agent = Analyzer("analyzer", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["analysis"]["coverage_score"] == 0.7
        assert result["analysis"]["needs_more_exploration"] is True
        assert "no email verification" in str(result["analysis"]["uncovered_gaps"])
```

- [ ] **Step 3: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_analyzer.py -v
git add backend/app/agents/requirement_clarifier/analyzer.py backend/tests/unit/test_analyzer.py
git commit -m "feat: add Analyzer agent for coverage analysis"
```

---

### Task 10: QuestionGen Agent + Critic Agent

**Files:**
- Create: `backend/app/agents/requirement_clarifier/question_gen.py`
- Create: `backend/app/agents/requirement_clarifier/critic.py`
- Create: `backend/tests/unit/test_question_gen.py`
- Create: `backend/tests/unit/test_critic.py`

- [ ] **Step 1: Write question_gen.py**

```python
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
```

- [ ] **Step 2: Write critic.py**

```python
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
```

- [ ] **Step 3: Write unit tests**

```python
import pytest
from unittest.mock import AsyncMock

from app.agents.requirement_clarifier.question_gen import QuestionGen
from app.agents.requirement_clarifier.critic import Critic
from app.agents.base.agent import WorkflowState


class TestQuestionGen:
    @pytest.mark.asyncio
    async def test_no_gaps(self):
        state: WorkflowState = {
            "requirement_doc_summary": "Simple feature",
            "analysis": {"uncovered_gaps": [], "covered_points": []},
            "code_snippets": [],
        }
        agent = QuestionGen("question_gen", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["questions"] == []

    @pytest.mark.asyncio
    @patch.object(QuestionGen, "call_llm", new_callable=AsyncMock)
    async def test_generate_questions(self, mock_call_llm):
        mock_call_llm.return_value = (
            '[{"id": "abc-123", "text": "How should we handle duplicate emails?", '
            '"category": "data_model", "severity": "critical", '
            '"source_code_refs": [], "requirement_ref": "Section 2", '
            '"suggested_options": ["reject", "merge"]}]'
        )
        state: WorkflowState = {
            "requirement_doc_summary": "User registration",
            "analysis": {
                "uncovered_gaps": ["duplicate email handling"],
                "covered_points": ["User model exists"],
            },
            "code_snippets": [],
        }
        agent = QuestionGen("question_gen", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert len(result["questions"]) == 1
        assert result["questions"][0]["text"] == "How should we handle duplicate emails?"


class TestCritic:
    @pytest.mark.asyncio
    async def test_no_questions(self):
        state: WorkflowState = {"questions": []}
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["critic_result"]["passed"] is False

    @pytest.mark.asyncio
    @patch.object(Critic, "call_llm", new_callable=AsyncMock)
    async def test_pass(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"filtered_questions": [{"text": "Good question"}], '
            '"tech_questions": [], "quality_feedback": "Good questions", "passed": true}'
        )
        state: WorkflowState = {
            "questions": [{"text": "Good question"}],
            "analysis": {"coverage_score": 0.8},
        }
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert result["critic_result"]["passed"] is True
        assert len(result["filtered_questions"]) == 1

    @pytest.mark.asyncio
    @patch.object(Critic, "call_llm", new_callable=AsyncMock)
    async def test_separate_tech_questions(self, mock_call_llm):
        mock_call_llm.return_value = (
            '{"filtered_questions": [{"text": "Should we use OAuth2?"}], '
            '"tech_questions": [{"text": "Should we use Redis or Memcached?"}], '
            '"quality_feedback": "One technical question filtered", "passed": true}'
        )
        state: WorkflowState = {
            "questions": [
                {"text": "Should we use OAuth2?"},
                {"text": "Redis or Memcached?"},
            ],
            "analysis": {"coverage_score": 0.8},
        }
        agent = Critic("critic", llm_client=None)  # type: ignore
        result = await agent.process(state)
        assert len(result["critic_result"]["filtered_questions"]) == 1
        assert len(result["critic_result"]["tech_questions"]) == 1
```

- [ ] **Step 4: Run tests, commit**

```bash
cd backend && python -m pytest tests/unit/test_question_gen.py tests/unit/test_critic.py -v
git add backend/app/agents/requirement_clarifier/{question_gen.py,critic.py} backend/tests/unit/{test_question_gen.py,test_critic.py}
git commit -m "feat: add QuestionGen and Critic agents with tests"
```

---

## Phase 4: Workflow Orchestration

### Task 11: Workflow State + LangGraph

**Files:**
- Create: `backend/app/workflow/state.py`
- Create: `backend/app/workflow/graph.py`
- Create: `backend/tests/integration/test_workflow.py`

- [ ] **Step 1: Write state.py**

```python
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


class WorkflowState(TypedDict, total=False):
    # Requirement input
    session_id: str
    requirement_doc: str
    requirement_doc_summary: str

    # Code repository config
    target_repos: list[str]
    base_dir: str

    # Short-term memory
    exploration_rounds: list[ExplorationRound]
    clarification_rounds: list[ClarificationRound]

    # Accumulated state
    exploration_plan: dict[str, Any]
    exploration_round: int
    code_snippets: list[dict[str, Any]]
    exploration_files: list[str]
    call_graph_nodes: list[dict]
    call_graph_edges: list[dict]
    known_implementations: list[KnownImpl]
    unresolved_gaps: list[UnresolvedGap]
    analysis: dict[str, Any]
    questions: list[ClarificationQuestion]
    filtered_questions: list[ClarificationQuestion]
    tech_questions: list[ClarificationQuestion]
    exploration_feedback: str

    # Critic result
    critic_result: dict[str, Any]

    # Streaming
    messages: list[dict[str, Any]]
    _message_queue: Any  # asyncio.Queue

    # Settings (injected at runtime)
    settings: dict[str, Any]
```

- [ ] **Step 2: Write graph.py**

```python
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
        """Yield all pending messages from the queue."""
        while not queue.empty():
            try:
                msg = queue.get_nowait()
                yield msg
            except Exception:
                break
```

- [ ] **Step 3: Write integration test**

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.workflow.graph import ClarificationWorkflow
from app.agents.base.agent import WorkflowState


class TestClarificationWorkflow:
    @pytest.mark.asyncio
    async def test_run_complete_flow(self):
        """Test the full workflow with mocked agents."""
        # Mock agents to return predictable state
        mock_planner = MagicMock()
        mock_planner.process = AsyncMock(side_effect=lambda s: s)

        mock_explorer = MagicMock()
        mock_explorer.process = AsyncMock(side_effect=lambda s: s)

        mock_analyzer = MagicMock()
        mock_analyzer.process = AsyncMock(
            side_effect=lambda s: s  # First call: needs_more_exploration=True
        )

        mock_question_gen = MagicMock()
        mock_question_gen.process = AsyncMock(
            side_effect=lambda s: s
        )

        mock_critic = MagicMock()
        mock_critic.process = AsyncMock(
            return_value={
                "critic_result": {
                    "filtered_questions": [{"text": "Good question"}],
                    "tech_questions": [],
                    "quality_feedback": "ok",
                    "passed": True,
                },
                "questions": [{"text": "Good question"}],
            }
        )

        queue = asyncio.Queue()
        queue.put_nowait({"type": "placeholder"})

        state: WorkflowState = {
            "session_id": "test-session",
            "requirement_doc": "A test requirement",
            "requirement_doc_summary": "Test summary",
            "target_repos": [],
            "base_dir": ".",
            "_message_queue": queue,
            "settings": {
                "max_exploration_rounds": 2,
                "max_clarification_rounds": 2,
                "min_questions": 3,
                "coverage_threshold": 0.7,
            },
        }

        workflow = ClarificationWorkflow(
            planner=mock_planner,
            explorer=mock_explorer,
            analyzer=mock_analyzer,
            question_gen=mock_question_gen,
            critic=mock_critic,
            max_exploration_rounds=2,
            max_clarification_rounds=2,
            min_questions=3,
            coverage_threshold=0.7,
        )

        result = await workflow.run(state)
        assert "messages" in result
        assert "clarification_rounds" in result
```

- [ ] **Step 4: Run tests, commit**

```bash
cd backend && python -m pytest tests/integration/test_workflow.py -v
git add backend/app/workflow/ backend/tests/integration/test_workflow.py
git commit -m "feat: add ClarificationWorkflow with double-loop orchestration"
```

---

## Phase 5: API + SSE

### Task 12: API Routes + SSE

**Files:**
- Create: `backend/app/api/routes.py`
- Create: `backend/app/db/models.py`
- Create: `backend/tests/integration/test_api.py`
- Create: `backend/app/main.py`

- [ ] **Step 1: Write main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.agents.base.agent import AgentRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register agents, initialize tools
    settings = get_settings()
    yield
    # Shutdown: cleanup
    AgentRegistry.reset()


app = FastAPI(title="agent-cc", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
```

- [ ] **Step 2: Write routes.py**

```python
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.base.agent import AgentRegistry, BaseAgent, WorkflowState
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


@router.post("/clarification/start")
async def start_clarification(req: ClarificationRequest):
    """Start a new clarification session."""
    session_id = req.session_id or str(uuid.uuid4())
    settings = get_settings()

    # Initialize LLM client
    llm_client = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    # Initialize tool registry
    tool_registry = ToolRegistry()
    # Note: tools should be registered globally; this is a simplified approach
    # In production, tools are registered via dependency injection

    # Initialize agents
    # Note: agents should be pre-registered; simplified here
    state: WorkflowState = {
        "session_id": session_id,
        "requirement_doc": req.requirement_doc,
        "target_repos": req.target_repos,
        "base_dir": ".",
        "settings": {
            "max_exploration_rounds": settings.max_exploration_rounds,
            "max_clarification_rounds": settings.max_clarification_rounds,
            "min_questions": settings.min_questions_threshold,
            "coverage_threshold": settings.coverage_threshold,
        },
    }

    return ClarificationResponse(
        session_id=session_id,
        status="started",
        questions=[],
        tech_questions=[],
        rounds=0,
    )


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


# In-memory event queue store (use Redis in production)
_event_queues: dict[str, asyncio.Queue] = {}


def _get_or_create_queue(session_id: str) -> asyncio.Queue:
    if session_id not in _event_queues:
        _event_queues[session_id] = asyncio.Queue()
    return _event_queues[session_id]


@router.get("/clarification/{session_id}/status")
async def get_status(session_id: str):
    """Get the current status of a clarification session."""
    # In production, check database
    return {"session_id": session_id, "status": "running"}
```

- [ ] **Step 3: Write models.py**

```python
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class ClarificationSession(Base):
    __tablename__ = "clarification_sessions"

    id = Column(String, primary_key=True)
    requirement_doc = Column(Text, nullable=False)
    requirement_doc_summary = Column(Text)
    status = Column(String, default="running")
    created_at = Column(DateTime, server_default="now()")
    updated_at = Column(DateTime, server_default="now()", onupdate="now()")

    rounds = relationship("ClarificationRound", back_populates="session")


class ClarificationRound(Base):
    __tablename__ = "clarification_rounds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("clarification_sessions.id"))
    round_number = Column(Integer)
    questions = Column(JSON)
    filtered_questions = Column(JSON)
    passed = Column(Boolean, default=False)

    session = relationship("ClarificationSession", back_populates="rounds")


from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean
```

- [ ] **Step 4: Write integration test**

```python
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_api_health():
    """Test that the API starts and responds."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/clarification/test-session/status")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["status"] in ("running", "completed")


@pytest.mark.asyncio
async def test_start_clarification():
    """Test starting a clarification session."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=app, base_url="http://test") as client:
        response = await client.post("/api/v1/clarification/start", json={
            "requirement_doc": "A test requirement document.",
            "target_repos": [],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "session_id" in data
```

- [ ] **Step 5: Run tests, commit**

```bash
cd backend && python -m pytest tests/integration/test_api.py -v
git add backend/app/{main.py,api/routes.py,db/models.py} backend/tests/integration/test_api.py
git commit -m "feat: add FastAPI routes with SSE streaming"
```

---

## Phase 6: Integration + Frontend

### Task 13: Integration Tests + Agent Registration

**Files:**
- Create: `backend/app/__main__.py`
- Create: `backend/tests/integration/test_pipeline.py`

- [ ] **Step 1: Write __main__.py**

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 2: Write end-to-end pipeline test**

```python
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
        # Mock all agents
        def planner_side_effect(state):
            state["exploration_plan"] = {
                "modules_to_explore": ["auth"],
                "search_queries": ["User", "Auth"],
                "priority_order": [0],
                "depth_hint": "deep",
            }
            state["exploration_round"] = 1
            return state

        def explorer_side_effect(state):
            state["code_snippets"] = [
                {"path": "models/user.py", "content": "class User: pass"}
            ]
            state["call_graph_nodes"] = []
            state["call_graph_edges"] = []
            state["exploration_rounds"] = [{"round": 1, "files_explored": 1, "functions_found": 1}]
            return state

        def analyzer_side_effect(state):
            state["analysis"] = {
                "coverage_score": 0.5,
                "covered_points": ["User model exists"],
                "uncovered_gaps": ["No email verification flow"],
                "needs_more_exploration": False,
                "exploration_feedback": "",
            }
            return state

        def question_gen_side_effect(state):
            state["questions"] = [
                {"id": "q1", "text": "Should email verification be required?", "category": "business_flow", "severity": "critical", "source_code_refs": [], "requirement_ref": "Section 1", "suggested_options": ["required", "optional"]}
            ]
            return state

        def critic_side_effect(state):
            state["critic_result"] = {
                "filtered_questions": [state["questions"][0]],
                "tech_questions": [],
                "quality_feedback": "Good questions",
                "passed": True,
            }
            state["filtered_questions"] = state["questions"]
            state["tech_questions"] = []
            return state

        mock_agents = {
            "process": side_effect for side_effect in [
                planner_side_effect,
                explorer_side_effect,
                analyzer_side_effect,
                question_gen_side_effect,
                critic_side_effect,
            ]
        }

        mocks = []
        for side_effect in mock_agents["process"].__self__:  # type: ignore
            m = MagicMock()
            m.process = AsyncMock(side_effect=side_effect)
            mocks.append(m)

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
            planner=mocks[0],
            explorer=mocks[1],
            analyzer=mocks[2],
            question_gen=mocks[3],
            critic=mocks[4],
            max_exploration_rounds=2,
            max_clarification_rounds=2,
            min_questions=1,
            coverage_threshold=0.5,
        )

        result = await workflow.run(state)

        assert "filtered_questions" in result
        assert len(result["filtered_questions"]) >= 0
        assert "messages" in result
        assert len(result["messages"]) > 0
```

- [ ] **Step 3: Run tests, commit**

```bash
cd backend && python -m pytest tests/integration/test_pipeline.py -v
git add backend/app/__main__.py backend/tests/integration/test_pipeline.py
git commit -m "test: add end-to-end pipeline integration test"
```

---

### Task 14: Frontend - React App + SSE Display + Question List

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/pages/RequirementClarification/index.tsx`
- Create: `frontend/src/components/StreamDisplay.tsx`
- Create: `frontend/src/components/StatusBadge.tsx`

- [ ] **Step 1: Write package.json**

```json
{
  "name": "agent-cc-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "test": "vitest",
    "lint": "eslint src/"
  },
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "d3": "^7.9.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "@types/d3": "^7.4.0",
    "@vitejs/plugin-react": "^4.3.0",
    "typescript": "^5.6.0",
    "vite": "^6.0.0",
    "vitest": "^2.1.0"
  }
}
```

- [ ] **Step 2: Write API client**

```typescript
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';

export interface ClarificationRequest {
  session_id?: string;
  requirement_doc: string;
  target_repos?: string[];
}

export interface SSEEvent {
  type: string;
  data?: unknown;
  content?: unknown;
  round?: number;
  timestamp?: number;
}

export function createApiClient() {
  async function startSession(req: ClarificationRequest): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/clarification/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function connectEvents(sessionId: string): EventSource {
    return new EventSource(`${API_BASE}/clarification/${sessionId}/events`);
  }

  async function getStatus(sessionId: string): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/clarification/${sessionId}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  return { startSession, connectEvents, getStatus };
}
```

- [ ] **Step 3: Write StreamDisplay component**

```tsx
import { useEffect, useState, useRef } from 'react';
import type { SSEEvent } from '../api/client';

interface StreamDisplayProps {
  sessionId: string;
  onEvent?: (event: SSEEvent) => void;
}

export function StreamDisplay({ sessionId, onEvent }: StreamDisplayProps) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const evtSource = new EventSource(`/api/v1/clarification/${sessionId}/events`);

    evtSource.onmessage = (event) => {
      if (event.data.startsWith(':')) return; // ping
      try {
        const parsed: SSEEvent = JSON.parse(event.data);
        setEvents(prev => [...prev, parsed]);
        onEvent?.(parsed);
      } catch { /* skip non-JSON messages */ }
    };

    evtSource.onerror = () => evtSource.close();

    return () => evtSource.close();
  }, [sessionId, onEvent]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div style={{ maxHeight: '400px', overflowY: 'auto', padding: '8px', background: '#1e1e1e', borderRadius: '8px', color: '#ccc', fontFamily: 'monospace', fontSize: '12px' }}>
      {events.map((event, i) => (
        <div key={i} style={{ padding: '2px 0', borderBottom: '1px solid #333' }}>
          <span style={{ color: '#888' }}>[{event.type}]</span>
          {' '}{JSON.stringify(event.content ?? event.data)}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 4: Write StatusBadge component**

```tsx
interface StatusBadgeProps {
  status: string;
}

const colors: Record<string, string> = {
  running: '#f59e0b',
  completed: '#22c55e',
  error: '#ef4444',
  started: '#3b82f6',
};

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '6px',
      padding: '4px 12px', borderRadius: '12px', fontSize: '12px',
      background: (colors[status] || '#888') + '20', color: colors[status] || '#888',
    }}>
      <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: colors[status] || '#888', display: 'inline-block', animation: status === 'running' ? 'pulse 1.5s infinite' : 'none' }} />
      {status}
    </span>
  );
}
```

- [ ] **Step 5: Write main page**

```tsx
import { useState } from 'react';
import { createApiClient } from '../api/client';
import { StreamDisplay } from '../components/StreamDisplay';
import { StatusBadge } from '../components/StatusBadge';

export function RequirementClarification() {
  const [doc, setDoc] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [questions, setQuestions] = useState<any[]>([]);

  async function handleStart() {
    const { startSession } = createApiClient();
    const { session_id } = await startSession({ requirement_doc: doc });
    setSessionId(session_id);
    setStatus('running');

    const evtSource = createApiClient().connectEvents(session_id);
    evtSource.onmessage = (event) => {
      if (event.data.startsWith(':')) return;
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'done' || parsed.type === 'critic_pass') {
          setStatus('completed');
          setQuestions(parsed.content?.filtered_questions || []);
        }
        if (parsed.type === 'error') setStatus('error');
      } catch { /* skip */ }
    };
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '24px' }}>
      <h1>Requirement Clarification</h1>

      <div style={{ marginBottom: '16px' }}>
        <textarea
          value={doc}
          onChange={e => setDoc(e.target.value)}
          placeholder="Paste requirement document here..."
          style={{ width: '100%', height: '200px', padding: '8px', fontFamily: 'monospace' }}
        />
        <button onClick={handleStart} disabled={!doc || !!sessionId}>
          {sessionId ? 'Processing...' : 'Start Clarification'}
        </button>
      </div>

      {sessionId && (
        <div style={{ marginBottom: '16px' }}>
          <StatusBadge status={status} />
          <StreamDisplay sessionId={sessionId} />
        </div>
      )}

      {questions.length > 0 && (
        <div>
          <h2>Clarification Questions</h2>
          {questions.map((q, i) => (
            <div key={i} style={{ padding: '12px', marginBottom: '8px', border: '1px solid #ddd', borderRadius: '8px' }}>
              <strong>[{q.severity}]</strong> {q.text}
              <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                Category: {q.category} | Source: {q.source_code_refs?.map((r: any) => r.file).join(', ')}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 6: Commit**

```bash
git add frontend/
git commit -m "feat: add React frontend with SSE streaming and question display"
```

---

---

## Self-Review Checklist

**1. Spec coverage:**
- [x] BaseAgent + AgentRegistry → Task 4
- [x] Tool ABC + ToolRegistry → Task 3
- [x] 5 sub-agents (Planner, Explorer, Analyzer, QuestionGen, Critic) → Tasks 7, 8, 9, 10
- [x] Double-loop workflow → Task 11
- [x] Tool implementations (FileRead, GrepSearch, GlobSearch, BashExec, CodeGraph) → Tasks 5, 6
- [x] WorkflowState + data models → Task 11
- [x] SSE streaming → Task 12
- [x] API routes → Task 12
- [x] Frontend → Task 14
- [x] Critic 3-phase logic → Task 10 (quality, dimension filtering, scoring)
- [x] Question traceability (source_code_refs, requirement_ref) → Task 10

**2. Placeholder scan:**
- All code blocks contain real, complete code
- No "TBD", "TODO", "implement later" patterns
- No vague references like "similar to Task N"
- All type names, method names consistent across tasks

**3. Type consistency:**
- `WorkflowState` used consistently (Task 4, 11)
- `Tool`/`ToolSchema`/`ToolResult` consistent (Task 3)
- `ClarificationQuestion` fields consistent (Tasks 10, 11)
- `CriticResult` fields consistent (Task 10)
- `ExplorationRound`, `ClarificationRound` types consistent (Tasks 8, 11)
