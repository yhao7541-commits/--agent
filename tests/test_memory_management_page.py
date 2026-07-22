from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import router


def test_memory_management_page_renders_controls():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/memory")

    assert response.status_code == 200
    html = response.text
    assert "客户记忆管理" in html
    assert "Memory Operations" not in html
    assert "Load Memory" not in html
    assert "Approve" not in html
    assert 'id="memory-table"' in html
    assert 'data-action="edit-memory"' in html
    assert 'data-action="approve-memory"' in html
    assert 'data-action="reject-memory"' in html
    assert 'data-action="delete-memory"' in html
    assert "/api/memory/users/" in html
    assert "/api/memory/memories/" in html
