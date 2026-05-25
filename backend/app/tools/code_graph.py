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
            "files": files,
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
