import json
from typing import Any

from app.agents.base.agent import BaseAgent, WorkflowState


class Explorer(BaseAgent):
    """Executes exploration plans by searching the codebase.

    Uses registered tools (FileRead, GrepSearch, GlobSearch, CodeGraph)
    to find relevant code and build context about existing implementations.
    """

    async def process(self, state: WorkflowState) -> WorkflowState:
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
