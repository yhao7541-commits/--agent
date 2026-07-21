from datetime import timedelta

from memory.customer_memory import MemoryProposal, utc_now
from memory.memory_store import MemoryStore


def _proposal(
    content: str = "喜欢安静房间",
    memory_type: str = "room_preference",
    sensitivity: str = "normal",
    expires_at=None,
) -> MemoryProposal:
    return MemoryProposal(
        type=memory_type,
        content=content,
        evidence=f"用户说：{content}",
        confidence=0.9,
        sensitivity=sensitivity,
        requires_confirmation=sensitivity == "sensitive",
        expires_at=expires_at,
    )


def test_sqlite_memory_store_persists_across_instances(tmp_path):
    db_path = tmp_path / "customer-memory.sqlite3"
    first_store = MemoryStore(db_path=db_path)
    created = first_store.upsert(
        user_id="user_001",
        proposal=_proposal(),
        trace_id="trace_001",
        conversation_id="conv_001",
    )

    second_store = MemoryStore(db_path=db_path)
    memories = second_store.list_user_memories("user_001")

    assert [memory.id for memory in memories] == [created.memory.id]
    assert memories[0].source_trace_id == "trace_001"
    assert memories[0].source_conversation_id == "conv_001"


def test_expired_memory_is_not_returned_for_agent_lookup(tmp_path):
    store = MemoryStore(db_path=tmp_path / "customer-memory.sqlite3")
    store.upsert(
        user_id="user_001",
        proposal=_proposal(expires_at=utc_now() - timedelta(days=1)),
    )

    assert store.list_user_memories("user_001") == []
    assert store.list_user_memories("user_001", include_inactive=True)[0].status == "expired"


def test_sensitive_memory_is_pending_review_until_approved(tmp_path):
    store = MemoryStore(db_path=tmp_path / "customer-memory.sqlite3")
    result = store.upsert(
        user_id="user_001",
        proposal=_proposal(
            content="精油过敏",
            memory_type="service_contraindication",
            sensitivity="sensitive",
        ),
    )

    assert result.memory.status == "pending_review"
    assert result.memory.review_status == "pending"
    assert store.list_user_memories("user_001") == []

    approved = store.approve(
        user_id="user_001",
        memory_id=result.memory.id,
        actor="ops",
        reason="confirmed allergy constraint",
    )

    assert approved is not None
    assert approved.status == "active"
    assert approved.review_status == "approved"
    assert store.list_user_memories("user_001")[0].content == "精油过敏"


def test_edit_increments_version_and_records_event(tmp_path):
    store = MemoryStore(db_path=tmp_path / "customer-memory.sqlite3")
    created = store.upsert(user_id="user_001", proposal=_proposal()).memory

    updated = store.update(
        user_id="user_001",
        memory_id=created.id,
        content="喜欢特别安静的房间",
        actor="ops",
        reason="customer clarified room preference",
    )

    assert updated is not None
    assert updated.version == created.version + 1
    assert updated.content == "喜欢特别安静的房间"
    events = store.list_memory_events(created.id)
    assert events[-1]["event_type"] == "memory_updated"
    assert events[-1]["previous_value"]["content"] == "喜欢安静房间"
    assert events[-1]["new_value"]["content"] == "喜欢特别安静的房间"


def test_delete_is_soft_delete_and_records_event(tmp_path):
    store = MemoryStore(db_path=tmp_path / "customer-memory.sqlite3")
    created = store.upsert(user_id="user_001", proposal=_proposal()).memory

    deleted = store.delete(
        user_id="user_001",
        memory_id=created.id,
        actor="ops",
        reason="customer requested deletion",
    )

    assert deleted is True
    assert store.list_user_memories("user_001") == []
    deleted_record = store.list_user_memories(
        "user_001",
        include_inactive=True,
        include_deleted=True,
    )[0]
    assert deleted_record.status == "deleted"
    events = store.list_memory_events(created.id)
    assert events[-1]["event_type"] == "memory_deleted"
    assert events[-1]["reason"] == "customer requested deletion"
