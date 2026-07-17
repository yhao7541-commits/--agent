from fastapi import FastAPI
from fastapi.testclient import TestClient

from web.routes import router


def test_home_page_renders_chinese_operational_ui():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "本地生活服务预约台" in html
    assert "常用入口" in html
    assert "运营控制台" in html
    assert "客户记忆" in html
    assert "AI聊天机器人" not in html


def test_home_page_keeps_chat_contract():
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert 'id="chat-form"' in html
    assert 'id="chat-box"' in html
    assert 'id="user-input"' in html
    assert 'id="send-btn"' in html
    assert 'id="clear-btn"' in html
    assert "fetch('/chat/stream'" in html
