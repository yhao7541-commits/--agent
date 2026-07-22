from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.operations import router
from observability.trace_store import JsonlTraceStore
from tools.customer_tools import reset_customer_memory_store
from tools.customer_tools import get_customer_memory_store
from memory.customer_memory import MemoryProposal
from security.guardrails import reset_confirmation_token_registry


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
    assert body["intent"] == "greeting"
    assert body["escalated"] is False
    assert "raw_prompt" not in body


@pytest.mark.parametrize(
    "confirmation_metadata",
    [
        {"confirmed_tool_arguments": {}},
        {"confirmation_token": None},
    ],
)
def test_operations_chat_fails_closed_for_explicit_empty_confirmation_metadata(
    confirmation_metadata,
):
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_explicit_empty_confirmation",
            "conversation_id": "conv_explicit_empty_confirmation",
            "message": "你好",
            **confirmation_metadata,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "escalation"
    assert body["escalated"] is True
    assert [call["tool_name"] for call in body["tool_calls"]] == [
        "escalate_to_human"
    ]
    assert body["tool_calls"][0]["arguments"]["reason"] == (
        "unsafe_tool_confirmation"
    )


def test_operations_chat_persists_trace_events_when_store_is_configured(tmp_path, monkeypatch):
    trace_path = tmp_path / "operations-traces.jsonl"
    monkeypatch.setenv("OPERATIONS_TRACE_STORE_PATH", str(trace_path))
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_trace",
            "conversation_id": "conv_trace",
            "message": "你好",
        },
    )

    assert response.status_code == 200
    trace_id = response.json()["trace_id"]
    events = JsonlTraceStore(trace_path).read_trace(trace_id)
    nodes = [event.node for event in events]
    assert "initialize_turn" in nodes
    assert "finalize_turn" in nodes
    assert all(event.conversation_id == "conv_trace" for event in events)


def test_operations_trace_endpoint_returns_events_and_replay(tmp_path, monkeypatch):
    trace_path = tmp_path / "operations-traces.jsonl"
    monkeypatch.setenv("OPERATIONS_TRACE_STORE_PATH", str(trace_path))
    client = make_client()
    chat_response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_trace_api",
            "conversation_id": "conv_trace_api",
            "message": "浣犲ソ",
        },
    )
    trace_id = chat_response.json()["trace_id"]

    response = client.get(f"/api/operations/traces/{trace_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["trace_id"] == trace_id
    assert body["conversation_id"] == "conv_trace_api"
    assert body["events"]
    assert body["events"][0]["node"] == "initialize_turn"
    assert body["replay"].startswith(f"Trace: {trace_id}")
    assert "raw_prompt" not in str(body)


def test_operations_trace_endpoint_returns_controlled_404_when_unconfigured(monkeypatch):
    monkeypatch.delenv("OPERATIONS_TRACE_STORE_PATH", raising=False)
    client = make_client()

    response = client.get("/api/operations/traces/missing_trace")

    assert response.status_code == 404
    assert response.json() == {"detail": "Trace store is not configured."}


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
    assert body["booking_slot_sources"]["service_type"] == "user"


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


def test_operations_chat_replayed_confirmation_executes_write_only_once():
    reset_confirmation_token_registry()
    client = make_client()
    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_replay",
            "conversation_id": "conv_replay",
            "message": "我想明天下午3点约肩颈放松",
        },
    ).json()
    confirmed_payload = {
        "user_id": "user_replay",
        "conversation_id": "conv_replay",
        "message": "确认",
        "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
        "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
        "confirmation_token": pending["confirmation_request"]["confirmation_token"],
    }

    first = client.post("/api/operations/chat", json=confirmed_payload).json()
    replay = client.post("/api/operations/chat", json=confirmed_payload).json()

    assert [result["tool_name"] for result in first["executed_tools"]] == [
        "create_booking"
    ]
    assert first["executed_tools"][0]["success"] is True
    assert replay["intent"] == "escalation"
    assert [result["tool_name"] for result in replay["executed_tools"]] == [
        "escalate_to_human"
    ]
    assert replay["tool_calls"][0]["arguments"]["reason"] == (
        "unsafe_tool_confirmation"
    )


@pytest.mark.parametrize(
    "malformed_token",
    ["v2.é.abc", "v2.eA.é", f"v2.{'a' * 5_000}.{'b' * 64}"],
)
def test_operations_chat_fails_closed_for_non_ascii_confirmation_token(
    malformed_token,
):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_unicode_token",
            "conversation_id": "conv_unicode_token",
            "message": "确认取消",
            "confirmed_tool_name": "cancel_booking",
            "confirmed_tool_arguments": {
                "booking_id": "booking_123",
                "customer_name": "user_unicode_token",
            },
            "confirmation_token": malformed_token,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "escalation"
    assert body["escalated"] is True
    assert body["tool_calls"][0]["arguments"]["reason"] == (
        "unsafe_tool_confirmation"
    )


@pytest.mark.parametrize(
    ("user_id", "conversation_id"),
    [
        ("user_oversized_api", "c" * 257),
        ("u" * 257, "conv_oversized_api"),
    ],
)
def test_operations_chat_fails_closed_when_draft_binding_is_oversized(
    user_id,
    conversation_id,
):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": user_id,
            "conversation_id": conversation_id,
            "message": "我想明天下午3点约肩颈放松",
        },
    )

    assert response.status_code == 422


@pytest.mark.parametrize("field", ["user_id", "conversation_id"])
def test_operations_chat_rejects_lone_surrogate_identifiers(field):
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app, raise_server_exceptions=False)
    payload = (
        '{"user_id":"\\ud800","conversation_id":"conv_surrogate",'
        '"message":"你好"}'
        if field == "user_id"
        else '{"user_id":"user_surrogate","conversation_id":"\\ud800",'
        '"message":"你好"}'
    )

    response = client.post(
        "/api/operations/chat",
        content=payload,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422


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


def test_operations_chat_returns_rag_citation_metadata_for_policy_question():
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_rag_api",
            "conversation_id": "conv_rag_api",
            "message": "如果我迟到20分钟会怎么样？",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["rag_used"] is True
    assert body["rag_citations"]["rag_used"] is True
    assert body["rag_citations"]["chunks"][0]["source"] == "booking_policy.md"


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


def test_operations_chat_exposes_applied_customer_memory_for_booking():
    reset_customer_memory_store()
    client = make_client()

    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_memory_api",
            "conversation_id": "conv_memory_api",
            "message": "我以后都喜欢安静一点的房间",
        },
    ).json()
    client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_memory_api",
            "conversation_id": "conv_memory_api",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        },
    )

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_memory_api",
            "conversation_id": "conv_memory_api_booking",
            "message": "我想明天下午3点约肩颈放松",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["memory_used"] is True
    assert "喜欢安静房间" in body["customer_context"]["known_preferences"]
    assert any(
        memory["content"] == "喜欢安静房间"
        and memory["applied_to"] == "booking_slots.special_requests"
        for memory in body["applied_customer_memories"]
    )
    assert "raw_prompt" not in body


def test_operations_chat_keeps_sensitive_memory_pending_review_after_confirmation():
    reset_customer_memory_store()
    client = make_client()
    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_sensitive_memory",
            "conversation_id": "conv_sensitive_memory",
            "message": "我对精油过敏，请以后不要用",
        },
    ).json()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_sensitive_memory",
            "conversation_id": "conv_sensitive_memory",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["executed_tools"][0]["output"]["review_status"] == "pending"
    assert get_customer_memory_store().list_user_memories("user_sensitive_memory") == []
    pending_review = get_customer_memory_store().list_user_memories(
        "user_sensitive_memory",
        include_inactive=True,
    )
    assert pending_review[0].status == "pending_review"


def test_operations_chat_routes_real_chinese_delete_memory_to_confirmation():
    reset_customer_memory_store()
    store = get_customer_memory_store()
    created = store.upsert(
        user_id="user_delete_memory",
        proposal=MemoryProposal(
            type="room_preference",
            content="喜欢安静房间",
            evidence="用户说喜欢安静房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    ).memory
    client = make_client()

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_delete_memory",
            "conversation_id": "conv_delete_memory",
            "message": "请删除安静房间这条记忆",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "delete_memory"
    assert body["confirmation_required"] is True
    assert body["confirmation_request"]["tool_name"] == "delete_customer_memory"
    assert body["confirmation_request"]["arguments"]["memory_id"] == created.id


def test_operations_chat_can_delete_pending_sensitive_memory_without_using_it_as_context():
    reset_customer_memory_store()
    client = make_client()
    pending = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_delete_pending_memory",
            "conversation_id": "conv_delete_pending_memory",
            "message": "我对精油过敏，请以后不要用",
        },
    ).json()
    client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_delete_pending_memory",
            "conversation_id": "conv_delete_pending_memory",
            "message": "确认",
            "confirmed_tool_name": pending["confirmation_request"]["tool_name"],
            "confirmed_tool_arguments": pending["confirmation_request"]["arguments"],
            "confirmation_token": pending["confirmation_request"]["confirmation_token"],
        },
    )

    response = client.post(
        "/api/operations/chat",
        json={
            "user_id": "user_delete_pending_memory",
            "conversation_id": "conv_delete_pending_memory",
            "message": "请删除过敏这条记忆",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["customer_context"]["known_preferences"] == []
    assert body["confirmation_required"] is True
    assert body["confirmation_request"]["tool_name"] == "delete_customer_memory"


def test_operations_router_is_registered_in_api_router_list():
    from api import api_routers

    assert router in api_routers
