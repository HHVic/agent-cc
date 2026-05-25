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
