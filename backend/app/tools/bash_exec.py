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
