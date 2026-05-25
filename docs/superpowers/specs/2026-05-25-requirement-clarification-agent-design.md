# 需求澄清 Agent 系统设计文档

> 版本: v1.0 | 日期: 2026-05-25

## 1. 项目概述

构建一个工业级、可扩展的 Agent 系统，第一版实现"需求澄清"能力。产品人员撰写产品文档后，Agent 自动分析代码库，生成带溯源的澄清问题，并通过批判机制过滤质量问题和技术问题，最终输出给产品人员进行需求澄清。

**核心目标：**
- 需求 → 代码探索 → 问题生成 → 批判过滤 → 产品澄清 的自动化流程
- Agent 架构可扩展，后续可添加日志排查 Agent、数据分析 Agent 等
- 工程化实现，非 Demo

## 2. 整体架构

```
agent-cc/
├── backend/                          # Python FastAPI
│   ├── app/
│   │   ├── agents/                   # Agent 层（可扩展核心）
│   │   │   ├── base.py              # BaseAgent ABC + AgentRegistry
│   │   │   ├── requirement_clarifier/  # 需求澄清 Agent（第一版）
│   │   │   │   ├── planner.py       #   规划器：制定探索计划
│   │   │   │   ├── explorer.py      #   探索器：代码库探索
│   │   │   │   ├── analyzer.py      #   分析器：覆盖度分析
│   │   │   │   ├── question_gen.py  #   问题生成器
│   │   │   │   └── critic.py        #   批判器
│   │   │   └── (future/)            # 日志排查Agent、数据分析Agent
│   │   ├── tools/                   # 工具层
│   │   │   ├── base.py              # Tool ABC + ToolRegistry
│   │   │   ├── file_read.py         # 文件读取
│   │   │   ├── grep_search.py       # 内容搜索
│   │   │   ├── glob_search.py       # 文件匹配
│   │   │   ├── bash_exec.py         # Shell 执行
│   │   │   └── code_graph.py        # 调用链路图生成
│   │   ├── workflow/                # 编排层（LangGraph）
│   │   │   ├── state.py            # WorkflowState TypedDict
│   │   │   ├── graph.py            # 图定义 + 条件路由
│   │   │   └── service.py          # 执行入口 + SSE
│   │   ├── core/
│   │   │   ├── llm.py              # LLM 客户端（OpenAI 兼容）
│   │   │   └── config.py           # 配置管理
│   │   ├── db/
│   │   │   └── models.py           # 数据库模型
│   │   └── api/
│   │       └── routes.py           # REST + SSE 接口
│   └── requirements.txt
├── frontend/                         # React + TypeScript
│   ├── src/
│   │   ├── pages/
│   │   │   └── RequirementClarification/
│   │   │       ├── index.tsx        # 主页面
│   │   │       ├── DocUpload.tsx    # 需求文档上传
│   │   │       ├── CallGraph.tsx    # 可交互调用链图
│   │   │       ├── QuestionList.tsx # 澄清问题列表
│   │   │       └── QuestionDetail.tsx # 单问题详情
│   │   ├── components/
│   │   │   ├── StreamDisplay.tsx    # SSE 实时流展示
│   │   │   └── StatusBadge.tsx      # 状态指示
│   │   └── api/
│   │       └── client.ts            # API 客户端
│   └── package.json
└── docs/
```

### 设计原则

1. **Agent 层与编排层解耦**：每个 Agent 实现 `process(state) -> state`，不依赖 LangGraph
2. **工具层独立**：Tool ABC + ToolRegistry，每个 Agent 按需获取工具
3. **配置驱动**：不同 Agent 可配置不同 LLM 模型
4. **最小权限**：每个 Agent 只获取其需要的工具子集

## 3. Agent 层设计

### 3.1 BaseAgent + AgentRegistry

参考 `industry_information_assistant` 的设计：

```python
class BaseAgent(ABC):
    def __init__(self, name: str, config: AgentConfig):
        self.name = name
        self.llm = OpenAICompatibleClient(config.llm_base_url, config.llm_api_key, config.model)
        self.tools: dict[str, Tool] = {}

    @abstractmethod
    async def process(self, state: WorkflowState) -> WorkflowState:
        """每个 Agent 实现此方法，接收 state，修改后返回"""
        ...

class AgentRegistry:
    _agents: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent):
        cls._agents[agent.name] = agent

    @classmethod
    def get(cls, name: str) -> BaseAgent: ...
    @classmethod
    def all(cls) -> dict[str, BaseAgent]: ...
```

### 3.2 需求澄清子 Agent 架构

需求澄清是一个**子工作流**，由 5 个 Agent 组成：

| Agent | 职责 | 核心输入 | 核心输出 |
|-------|------|----------|----------|
| **Planner** | 解析需求文档，拆分成探索任务 | 需求文档 + 累积上下文 | 探索计划 |
| **Explorer** | 执行探索任务，调用工具读取代码 | 探索计划 | 代码片段 + 调用链 |
| **Analyzer** | 对比需求 vs 现有实现，判断覆盖度 | 探索结果 + 需求 | 覆盖分析 + 缺口 |
| **QuestionGen** | 生成澄清问题（带溯源） | 分析结果 + 需求 | 结构化问题列表 |
| **Critic** | 质疑问题质量 + 过滤技术题 | 问题列表 | 过滤后的问题 |

### 3.3 双层循环工作流

```
                    ┌──────────────┐
                    │   产品上传    │
                    │  需求文档     │
                    └──────┬───────┘
                           ▼
          ┌─────────────────────────────────┐
          │  外层：澄清循环（最多 N 轮）      │
          │                                 │
          │  ┌─────────┐    ┌──────────┐    │
          │  │ Planner │───▶│ Explorer │    │  ← 内层：探索循环（最多 M 轮）
          │  └─────────┘    └──────────┘    │
          │         ┌──────────┐            │
          │         │ Analyzer │            │
          │         └──────────┘            │
          │                │                │
          │                ▼                │
          │         ┌──────────┐            │
          │         │ Question │            │
          │         │   Gen    │            │
          │         └──────────┘            │
          │                │                │
          │                ▼                │
          │          ┌─────────┐            │
          │          │ Critic  │─不通过─────┤  ← 回 Planner 继续探索
          │          └─────────┘            │
          │                │                │
          │          通过 ✓                   │
          └────────────────┬────────────────┘
                           ▼
                    ┌──────────────┐
                    │  输出澄清     │
                    │    问题       │
                    └──────────────┘
```

**内层循环触发条件（Critic 判定需要继续探索）：**
1. 生成的问题数量不足（< 3 个）
2. 问题质量差（太泛、重复、有歧义）
3. 覆盖度不够（Analyzer 发现需求关键点无对应代码探索）

**循环反馈机制：**
- Critic 给 Planner 提供 feedback（如"缺少对 XX 模块的探索"、"问题太泛"）
- Planner 根据 feedback 调整下一轮探索计划
- Analyzer 每轮评估覆盖度，若 < 阈值则触发新一轮探索

**循环终止条件：**
- Critic 判定通过
- 达到最大迭代次数（默认探索 3 轮，澄清 3 轮）

## 4. 记忆机制设计

参考 `industry_information_assistant` 的双层记忆架构：

### 4.1 短期记忆（Session Memory）

- 存储在内存 dict 中，生命周期为一次澄清会话
- 存储：探索过程日志、每轮澄清问题、产品回答

### 4.2 持久化记忆（Persistent Memory）

- 持久化到 PostgreSQL
- 存储：需求文档摘要、已确认实现、历史澄清记录
- 支持向量检索（Milvus，可选），用于"上次讨论过 XX 模块"的语义检索

### 4.3 WorkflowState 中的记忆字段

```python
class WorkflowState(TypedDict):
    # 需求输入
    session_id: str
    requirement_doc: str
    requirement_doc_summary: str

    # 代码库配置
    target_repos: list[str]

    # 短期记忆
    exploration_rounds: list[ExplorationRound]
    clarification_rounds: list[ClarificationRound]

    # 累积状态
    accumulated_context: str
    known_implementations: list[KnownImpl]
    unresolved_gaps: list[UnresolvedGap]
    call_graph: list[CallGraphEdge]
```

### 4.4 记忆持久化时机

- 每轮澄清完成后，自动持久化到数据库
- 每轮探索完成后，更新 `known_implementations` 和 `unresolved_gaps`

## 5. 工具层设计

参考 `cc-python-claude/cc/tools/base.py` 的工具协议。

### 5.1 工具接口

```python
class ToolSchema(dataclass):
    name: str
    description: str
    input_schema: dict[str, Any]

class ToolResult(dataclass):
    content: str | list[dict[str, Any]]
    is_error: bool = False

class Tool(ABC):
    def get_name(self) -> str: ...
    def get_schema(self) -> ToolSchema: ...
    async def execute(self, tool_input: dict[str, Any]) -> ToolResult: ...
    def is_concurrency_safe(self) -> bool: ...  # 只读工具返回 True

class ToolRegistry:
    def register(self, tool: Tool): ...
    def get(self, name: str) -> Tool | None: ...
    def list_tools(self) -> list[Tool]: ...
```

### 5.2 需求澄清 Agent 需要的工具

| 工具 | 作用 | 并发安全 | 来源 |
|------|------|----------|------|
| `FileReadTool` | 读取代码文件 | 是 | cc-python-claude 复用 |
| `GrepSearchTool` | 内容搜索 | 是 | cc-python-claude 复用 |
| `GlobSearchTool` | 文件模式匹配 | 是 | cc-python-claude 复用 |
| `BashExecTool` | Shell 命令执行 | 否 | cc-python-claude 复用 |
| `CodeGraphTool` | 调用链路图生成 | 否 | **新增** |

### 5.3 CodeGraphTool 设计

- **功能**：分析代码，生成函数/类调用关系图
- **语言支持**：Java AST（Python 端调用 tree-sitter-java 解析 Java 代码）
- **输入**：文件路径列表或模块名
- **输出**：`{nodes: [...], edges: [...]}`，直接匹配前端可交互图的数据格式
- **Fallback**：AST 解析失败时，使用正则文本匹配提取函数调用

## 6. 数据流设计

### 6.1 Planner 输出

```python
ExplorationPlan = TypedDict('ExplorationPlan', {
    'modules_to_explore': list[str],        # 要探索的模块
    'search_queries': list[str],            # 搜索关键词
    'priority_order': list[int],            # 优先级
    'depth_hint': str,                      # deep | shallow
})
```

### 6.2 Explorer 输出

```python
ExplorationResult = TypedDict('ExplorationResult', {
    'code_snippets': list[FileContext],      # 代码片段 {path, content, relevance_score}
    'call_graph': CallGraphData,             # 调用链路图数据
    'relevant_functions': list[FunctionContext],  # 相关函数 {name, file, signature, callers, callees}
    'search_queries_executed': list[str],
})
```

### 6.3 Analyzer 输出

```python
AnalysisResult = TypedDict('AnalysisResult', {
    'coverage_score': float,                 # 0.0 - 1.0
    'covered_points': list[CoveredPoint],
    'uncovered_gaps': list[UnresolvedGap],
    'needs_more_exploration': bool,
    'exploration_feedback': str,             # 给 Planner 的反馈
})
```

### 6.4 问题生成与批判

```python
ClarificationQuestion = TypedDict('ClarificationQuestion', {
    'id': str,
    'text': str,                            # 问题内容
    'category': str,                        # 分类：业务流程/数据模型/边界条件/异常处理
    'severity': str,                        # 优先级：critical/important/normal/low
    'source_code_refs': list[CodeRef],      # 溯源：代码位置
    'requirement_ref': str,                 # 溯源：需求文档对应段落
    'suggested_options': list[str],
})

CriticResult = TypedDict('CriticResult', {
    'filtered_questions': list[ClarificationQuestion],  # 保留给产品的问题
    'tech_questions': list[ClarificationQuestion],       # 技术问题（不展示给产品）
    'quality_feedback': str,                            # 给 Planner 的反馈
    'passed': bool,                                     # 是否通过
})
```

### 6.5 Critic 批判逻辑

**第一轮：质量评估**
- 检查问题是否重复（语义相似性）
- 检查问题是否有歧义（太泛、不具体）
- 检查覆盖度（是否涵盖业务流程、数据模型、边界条件、异常处理）
- 检查问题是否可回答（产品人员能否回答）

**第二轮：维度过滤**
- 判断问题是否为产品维度（产品人员可回答）
- 如果是技术问题（如"用 Redis 还是 Memcached"），标记为 tech_questions，不展示给产品
- 技术备注可在报告底部单独展示，供技术参考

**第三轮：整体质量评分**
- 计算问题整体质量分
- 如果质量分低于阈值，标记 passed=False，触发内层循环

## 7. SSE 实时流设计

### 7.1 事件类型

```python
SSEEvent = {
    'type': str,
    'data': dict,
    'round': int,
    'timestamp': float,
}
```

**事件类型及对应前端渲染：**

| type | 前端渲染 |
|------|----------|
| `exploration_start` | 显示"开始探索 XX 模块" |
| `exploration_progress` | 实时显示正在探索的文件 |
| `exploration_done` | 更新调用链路图 |
| `analysis_start` | 显示"正在分析覆盖度" |
| `analysis_done` | 显示覆盖度报告 |
| `question_generating` | 显示"正在生成问题" |
| `critic_reviewing` | 显示"正在审查问题质量" |
| `round_complete` | 显示本轮生成的问题概览 |
| `round_transition` | 标记"第 2 轮澄清"等 |
| `error` | 显示错误信息 |
| `done` | 切换到最终结果页 |

## 8. 错误处理

| 场景 | 策略 |
|------|------|
| 代码库不可访问 | 跳过该仓库，记录 error 到 logs，继续其他仓库 |
| LLM 调用失败 | 重试 2 次（指数退避），仍失败则标记 skip，记录 error |
| AST 解析失败 | 记录 warning，用 fallback 的文本匹配方式 |
| SSE 断开 | 前端自动重连，从上次 checkpoint 继续 |
| 探索超时（单轮>5分钟） | 终止当前轮，输出当前结果，标记"探索不完整" |

## 9. 未来 Agent 扩展设计

每个新 Agent 只需实现 `BaseAgent.process()` 接口：

```python
class LogTroubleshooterAgent(BaseAgent):
    async def process(self, state: WorkflowState) -> WorkflowState:
        tools = self.registry.get_tools(['log_search', 'grep_search', 'file_read'])
        # 实现自己的逻辑
        return state

AgentRegistry.register(LogTroubleshooterAgent('log_troubleshooter', config))
```

在 LangGraph 中，新 Agent 只需要：
1. 作为 node 添加到图中
2. 添加边定义
3. 通过状态字段（`active_agent_type`）路由到对应 Agent

## 10. 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 后端语言 | Python 3.11+ | 与参考项目一致 |
| Web 框架 | FastAPI | 原生支持 SSE |
| 编排 | LangGraph | 状态管理、checkpoint、条件路由 |
| LLM 客户端 | OpenAI SDK（兼容协议） | base_url + api_key 可配置 |
| 数据库 | PostgreSQL | 会话、记忆持久化 |
| 向量库 | Milvus（可选） | 语义检索 |
| Java AST | tree-sitter-java | Python 库，支持 Java AST 解析 |
| 前端 | React + TypeScript + D3.js | 可交互图表 |
