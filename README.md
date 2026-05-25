# agent-cc

Requirement Clarification Agent — 将需求文档自动转化为代码探索、缺口分析、澄清问题的工业级系统。

**流程：** 需求文档 → 代码探索 → 覆盖率分析 → 问题生成 → 质量评审 → 输出给产品团队

## 架构

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Backend                    │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │  Planner  │→│ Explorer  │→│   Analyzer        │   │
│  └──────────┘  └──────────┘  └────────┬─────────┘   │
│                                        │             │
│  ┌──────────┐  ┌──────────┐  ┌────────▼─────────┐   │
│  │   Critic  │←│ Question  │←│    Inner Loop     │   │
│  └──────────┘  │   Gen     │  └──────────────────┘   │
│                └──────────┘                           │
│                                                       │
│  Outer Loop: 探索 → 分析 → 提问 → 评审 → (再探索?)    │
│                                                       │
│  Tools: FileRead │ GlobSearch │ GrepSearch            │
│         BashExec  │ CodeGraph                            │
└──────────────────────────┬────────────────────────────┘
                           │ SSE Streaming
                           ▼
┌──────────────────────────────────────────────────────┐
│              React Frontend (Vite)                    │
│                                                       │
│  RequirementClarification → StreamDisplay → StatusBadge │
└──────────────────────────────────────────────────────┘
```

## 技术栈

| 层 | 技术 |
|---|------|
| Backend | Python 3.11+, FastAPI, Pydantic, asyncpg |
| LLM | OpenAI-compatible (DashScope/qwen, OpenAI, local) |
| Orchestration | 自定义双循环状态机 (LangGraph-compatible) |
| Database | PostgreSQL (SQLAlchemy ORM) |
| Frontend | React 18, TypeScript, Vite, D3.js |

## 快速开始

### 前置条件

- Python >= 3.11
- PostgreSQL 14+
- Node.js >= 18

### 1. 启动数据库

```bash
# 使用 Docker
docker run -d --name agent-cc-db \
  -e POSTGRES_PASSWORD=postgres123 \
  -e POSTGRES_DB=agent_cc \
  -p 5432:5432 \
  postgres:16

# 或检查本地是否已有 PostgreSQL
psql -U postgres -c "SELECT 1"
```

### 2. 后端

```bash
cd backend

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 LLM API Key 和数据库地址

# 运行测试
pytest -v

# 启动服务
python -m app
# → http://localhost:8000/docs (Swagger UI)
```

### 3. 前端

```bash
cd frontend

# 安装依赖
npm install

# 开发模式 (默认代理到 localhost:8000)
npm run dev
# → http://localhost:5173
```

## Docker 部署

### 方式一：一键 Docker Compose（推荐）

```bash
# 1. 配置环境变量
cp backend/.env.example backend/.env
# 编辑 .env，填入 LLM_API_KEY

# 2. 启动所有服务
chmod +x scripts/start-services.sh
./scripts/start-services.sh start

# 访问地址
#   前端 + API:   http://localhost:3000
#   后端 Swagger: http://localhost:8000/docs
#   数据库:       localhost:5432

# 管理命令
./scripts/start-services.sh status    # 查看状态
./scripts/start-services.sh logs      # 查看日志
./scripts/start-services.sh stop      # 停止服务
./scripts/start-services.sh restart   # 重启
./scripts/start-services.sh clean     # 清理（含数据卷）
```

**服务架构：**

| 容器 | 端口 | 说明 |
|------|------|------|
| agent-cc-db | 5432 | PostgreSQL 数据库 |
| agent-cc-backend | 8000 | FastAPI 后端 |
| agent-cc-frontend | 3000 | React 前端 + Nginx 反向代理 |

### 方式二：手动 Docker 构建

```bash
# 后端
cd backend
docker build -t agent-cc-backend .
docker run -d --name agent-cc-backend \
  --network host \
  --env-file .env \
  -e DATABASE_URL=postgresql+asyncpg://postgres:postgres123@localhost:5432/agent_cc \
  -p 8000:8000 \
  agent-cc-backend

# 前端
cd frontend
docker build -t agent-cc-frontend .
docker run -d --name agent-cc-frontend \
  --network host \
  -p 3000:80 \
  agent-cc-frontend
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/clarification/start` | 启动澄清会话 |
| GET | `/api/v1/clarification/{session_id}/events` | SSE 事件流 |
| GET | `/api/v1/clarification/{session_id}/status` | 查询会话状态 |

## 目录结构

```
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── base/          # BaseAgent, AgentRegistry, WorkflowState
│   │   │   └── requirement_clarifier/
│   │   │       ├── planner.py       # 需求解析 + 探索计划
│   │   │       ├── explorer.py      # 代码探索
│   │   │       ├── analyzer.py      # 覆盖率分析
│   │   │       ├── question_gen.py  # 澄清问题生成
│   │   │       └── critic.py        # 问题质量评审
│   │   ├── core/
│   │   │   ├── config.py        # 配置管理
│   │   │   └── llm.py           # LLM 客户端
│   │   ├── tools/
│   │   │   ├── base.py          # Tool ABC, ToolRegistry
│   │   │   ├── file_read.py     # 文件读取
│   │   │   ├── glob_search.py   # 文件搜索
│   │   │   ├── grep_search.py   # 内容搜索
│   │   │   ├── bash_exec.py     # 命令执行
│   │   │   └── code_graph.py    # 调用图分析
│   │   ├── workflow/
│   │   │   ├── state.py         # 类型定义
│   │   │   └── graph.py         # 双循环工作流
│   │   ├── api/routes.py        # FastAPI 路由
│   │   ├── db/models.py         # SQLAlchemy 模型
│   │   └── main.py              # 应用入口
│   └── tests/
│       ├── unit/                # 单元测���
│       └── integration/         # 集成测试
└── frontend/
    ├── src/
    │   ├── api/client.ts        # API 客户端
    │   ├── components/
    │   │   ├── StreamDisplay.tsx   # SSE 流展示
    │   │   └── StatusBadge.tsx     # 状态徽章
    │   └── pages/
    │       └── RequirementClarification/index.tsx
    └── package.json
```

## License

MIT
