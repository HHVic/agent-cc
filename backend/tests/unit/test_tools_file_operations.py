import os
import tempfile
from pathlib import Path

import pytest

from app.tools.file_read import FileReadTool
from app.tools.glob_search import GlobSearchTool
from app.tools.grep_search import GrepSearchTool


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
