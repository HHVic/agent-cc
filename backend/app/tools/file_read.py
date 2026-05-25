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
