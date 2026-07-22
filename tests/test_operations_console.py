from pathlib import Path

from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from fastapi.templating import Jinja2Templates


TEMPLATES = Jinja2Templates(
    directory=str(Path(__file__).resolve().parents[1] / "web" / "templates")
)


def test_operations_console_renders_runtime_panels():
    app = FastAPI()

    @app.get("/operations", response_class=HTMLResponse)
    async def operations_console(request: Request):
        return TEMPLATES.TemplateResponse(request, "operations_console.html")

    client = TestClient(app)

    response = client.get("/operations")

    assert response.status_code == 200
    html = response.text
    assert "运营控制台" in html
    assert "Operations Console" not in html
    assert "Run Turn" not in html
    assert "Confirm Pending Action" not in html
    assert 'id="confirmation-panel"' in html
    assert 'id="tool-calls-panel"' in html
    assert 'id="memory-panel"' in html
    assert 'id="rag-panel"' in html
    assert 'id="trace-panel"' in html
    assert 'id="decision-panel"' in html
    assert 'id="decision-source-value"' in html
    assert 'id="decision-confidence-value"' in html
    assert 'id="decision-json"' in html
