from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import router


def test_operations_console_renders_runtime_panels():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/operations")

    assert response.status_code == 200
    html = response.text
    assert "Operations Console" in html
    assert 'id="confirmation-panel"' in html
    assert 'id="tool-calls-panel"' in html
    assert 'id="memory-panel"' in html
    assert 'id="rag-panel"' in html
    assert 'id="trace-panel"' in html
