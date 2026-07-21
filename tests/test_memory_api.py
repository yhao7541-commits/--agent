from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.memory import router
from memory.customer_memory import MemoryProposal
from tools.customer_tools import get_customer_memory_store, reset_customer_memory_store


def make_client(tmp_path):
    reset_customer_memory_store(tmp_path / "customer-memory.sqlite3")
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _proposal(
    *,
    content: str = "quiet room",
    memory_type: str = "room_preference",
    sensitivity: str = "normal",
) -> MemoryProposal:
    return MemoryProposal(
        type=memory_type,
        content=content,
        evidence=f"user said: {content}",
        confidence=0.9,
        sensitivity=sensitivity,
        requires_confirmation=sensitivity == "sensitive",
    )


def test_memory_api_lists_pending_and_active_memories(tmp_path):
    client = make_client(tmp_path)
    store = get_customer_memory_store()
    store.upsert(user_id="user_001", proposal=_proposal())
    store.upsert(
        user_id="user_001",
        proposal=_proposal(
            content="essential oil allergy",
            memory_type="service_contraindication",
            sensitivity="sensitive",
        ),
    )

    response = client.get("/api/memory/users/user_001/memories")

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user_001"
    assert {memory["status"] for memory in body["memories"]} == {"active", "pending_review"}


def test_memory_api_edit_increments_version_and_records_event(tmp_path):
    client = make_client(tmp_path)
    created = get_customer_memory_store().upsert(
        user_id="user_001",
        proposal=_proposal(),
    ).memory

    response = client.patch(
        f"/api/memory/memories/{created.id}",
        json={
            "user_id": "user_001",
            "content": "very quiet room",
            "actor": "ops",
            "reason": "customer clarified preference",
        },
    )
    events = client.get(f"/api/memory/memories/{created.id}/events").json()["events"]

    assert response.status_code == 200
    body = response.json()
    assert body["memory"]["content"] == "very quiet room"
    assert body["memory"]["version"] == created.version + 1
    assert events[-1]["event_type"] == "memory_updated"
    assert events[-1]["actor"] == "ops"


def test_memory_api_approve_and_reject_review_items(tmp_path):
    client = make_client(tmp_path)
    store = get_customer_memory_store()
    approved_candidate = store.upsert(
        user_id="user_001",
        proposal=_proposal(
            content="essential oil allergy",
            memory_type="service_contraindication",
            sensitivity="sensitive",
        ),
    ).memory
    rejected_candidate = store.upsert(
        user_id="user_001",
        proposal=_proposal(
            content="do not send marketing messages",
            memory_type="marketing_consent",
            sensitivity="sensitive",
        ),
    ).memory

    approve_response = client.post(
        f"/api/memory/memories/{approved_candidate.id}/approve",
        json={"user_id": "user_001", "actor": "ops", "reason": "confirmed by customer"},
    )
    reject_response = client.post(
        f"/api/memory/memories/{rejected_candidate.id}/reject",
        json={"user_id": "user_001", "actor": "ops", "reason": "not enough evidence"},
    )

    assert approve_response.status_code == 200
    assert approve_response.json()["memory"]["review_status"] == "approved"
    assert reject_response.status_code == 200
    assert reject_response.json()["memory"]["status"] == "rejected"
    assert get_customer_memory_store().list_user_memories("user_001")[0].content == "essential oil allergy"


def test_memory_api_delete_is_soft_delete(tmp_path):
    client = make_client(tmp_path)
    created = get_customer_memory_store().upsert(
        user_id="user_001",
        proposal=_proposal(),
    ).memory

    response = client.delete(
        f"/api/memory/memories/{created.id}",
        params={"user_id": "user_001", "actor": "ops", "reason": "customer requested deletion"},
    )
    list_response = client.get(
        "/api/memory/users/user_001/memories",
        params={"include_deleted": True},
    )

    assert response.status_code == 200
    assert response.json()["deleted"] is True
    deleted_memory = list_response.json()["memories"][0]
    assert deleted_memory["status"] == "deleted"
    assert deleted_memory["deleted_at"] is not None


def test_memory_api_returns_controlled_404_for_missing_memory(tmp_path):
    client = make_client(tmp_path)

    response = client.patch(
        "/api/memory/memories/missing_memory",
        json={"user_id": "user_001", "content": "updated"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Memory not found."}
