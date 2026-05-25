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
    _tools: dict[str, Tool] = field(default_factory=dict, repr=False)

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

    @property
    def tools(self) -> dict[str, Tool]:
        return self._tools
