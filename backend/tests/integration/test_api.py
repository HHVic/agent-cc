import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_api_health():
    """Test that the API starts and responds."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/clarification/test-session/status")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["status"] in ("running", "completed")


@pytest.mark.asyncio
async def test_start_clarification():
    """Test starting a clarification session."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/clarification/start", json={
            "requirement_doc": "A test requirement document.",
            "target_repos": [],
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "session_id" in data
