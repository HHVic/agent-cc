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
