from memory.customer_memory import CustomerMemory, MemoryProposal
from memory.memory_proposals import extract_memory_proposals
from memory.memory_store import MemoryStore


def test_explicit_preference_creates_memory_proposal():
    proposals = extract_memory_proposals("我以后都喜欢安静一点的房间")

    assert len(proposals) == 1
    assert proposals[0].type == "preference"
    assert proposals[0].content == "喜欢安静房间"
    assert proposals[0].evidence == "我以后都喜欢安静一点的房间"
    assert proposals[0].requires_confirmation is False


def test_vague_expression_does_not_create_memory_proposal():
    proposals = extract_memory_proposals("随便吧，可能安静一点也行")

    assert proposals == []


def test_sensitive_memory_requires_confirmation():
    proposals = extract_memory_proposals("我对精油过敏，请以后不要用")

    assert len(proposals) == 1
    assert proposals[0].type == "constraint"
    assert proposals[0].sensitivity == "sensitive"
    assert proposals[0].requires_confirmation is True


def test_duplicate_preference_updates_existing_memory():
    store = MemoryStore()
    existing = store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢安静房间",
            evidence="第一次说喜欢安静房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    )

    result = store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢安静房间",
            evidence="再次确认喜欢安静房间",
            confidence=0.95,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    )

    assert result.action == "updated"
    assert result.memory.id == existing.memory.id
    assert result.memory.evidence == "再次确认喜欢安静房间"


def test_conflicting_preference_returns_conflict_result():
    store = MemoryStore()
    store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢安静房间",
            evidence="喜欢安静房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    )

    result = store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢热闹房间",
            evidence="这次说喜欢热闹房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    )

    assert result.action == "conflict"
    assert result.conflict_with is not None


def test_user_can_delete_memory():
    store = MemoryStore()
    created = store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢安静房间",
            evidence="喜欢安静房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
    )

    deleted = store.delete(user_id="user_001", memory_id=created.memory.id)

    assert deleted is True
    assert store.list_user_memories("user_001") == []


def test_memory_write_generates_trace_event():
    store = MemoryStore()
    trace_events = []

    result = store.upsert(
        user_id="user_001",
        proposal=MemoryProposal(
            type="preference",
            content="喜欢安静房间",
            evidence="喜欢安静房间",
            confidence=0.9,
            sensitivity="normal",
            requires_confirmation=False,
        ),
        trace_id="trace_001",
        conversation_id="conv_001",
        trace_events=trace_events,
    )

    assert result.action == "created"
    assert trace_events[-1]["event_type"] == "memory_written"
    assert trace_events[-1]["metadata"]["memory_id"] == result.memory.id


def test_customer_memory_schema_keeps_evidence():
    memory = CustomerMemory(
        id="memory_001",
        user_id="user_001",
        type="preference",
        content="喜欢安静房间",
        evidence="用户明确说喜欢安静房间",
        confidence=0.9,
        sensitivity="normal",
    )

    assert memory.evidence
