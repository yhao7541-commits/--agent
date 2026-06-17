from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.operations import router


def make_client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_operations_chat_returns_trace_id():
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_001",
            "conversation_id": "conv_001",
            "message": "你好",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"]
    assert "reply" in body
    assert "raw_prompt" not in body


def test_operations_chat_returns_confirmation_request_for_write_action():
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["confirmation_required"] is True
    assert body["confirmation_request"]["tool_name"] == "create_booking"
    assert body["trace_id"]
    assert body["tool_calls"]


def test_operations_chat_supports_booking_slot_follow_up():
    client = make_client()
    first_turn = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_010",
            "conversation_id": "conv_010",
            "message": "我想约一个肩颈放松",
        },
    ).json()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_010",
            "conversation_id": "conv_010",
            "message": "明天下午3点",
            "booking_slots": first_turn["booking_slots"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "booking"
    assert body["missing_slots"] == []
    assert body["booking_slots"]["service_type"] == "肩颈放松"
    assert body["confirmation_required"] is True


def test_operations_chat_executes_confirmed_booking():
    client = make_client()
    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        },
    ).json()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["confirmation_required"] is False
    assert body["executed_tools"][0]["tool_name"] == "create_booking"
    assert body["executed_tools"][0]["success"] is True
    assert "预约已创建" in body["reply"]


def test_operations_chat_rejects_pending_booking_without_write():
    client = make_client()
    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "我想明天下午3点约肩颈放松",
        },
    ).json()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_002",
            "conversation_id": "conv_002",
            "message": "不用了",
            "confirmation_decision": "rejected",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "confirmation_rejected"
    assert body["confirmation_required"] is False
    assert body["confirmation_request"] == {}
    assert body["executed_tools"] == []
    assert "create_booking" not in {call["tool_name"] for call in body["tool_calls"]}
    assert "未执行" in body["reply"]


def test_operations_chat_exposes_human_escalation():
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_003",
            "conversation_id": "conv_003",
            "message": "按摩后肩膀受伤了，现在很疼怎么办？",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["escalated"] is True
    assert body["intent"] == "escalation"
    assert any(call["tool_name"] == "escalate_to_human" for call in body["tool_calls"])
    assert "raw_prompt" not in body


def test_operations_chat_exposes_memory_proposal_confirmation():
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_004",
            "conversation_id": "conv_004",
            "message": "我以后都喜欢安静一点的房间",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["memory_proposals"][0]["content"] == "喜欢安静房间"
    assert body["confirmation_required"] is True
    assert body["confirmation_request"]["tool_name"] == "write_customer_preference"
    assert "raw_prompt" not in body


def test_operations_router_is_registered_in_api_router_list():
    from api import api_routers

    assert router in api_routers
