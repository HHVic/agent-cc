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
