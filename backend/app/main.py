from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.agents.base.agent import AgentRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: register agents, initialize tools
    settings = get_settings()
    yield
    # Shutdown: cleanup
    AgentRegistry.reset()


app = FastAPI(title="agent-cc", version="0.1.0", lifespan=lifespan)
app.include_router(router, prefix="/api/v1")
