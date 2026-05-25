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
