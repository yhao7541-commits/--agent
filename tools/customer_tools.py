from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from pydantic import BaseModel

from memory.customer_memory import MemoryProposal
from memory.memory_policy import memory_requires_confirmation, sensitivity_for_memory_type
from memory.memory_store import MemoryStore


_memory_store: MemoryStore | None = None


def reset_customer_memory_store(db_path: str | Path | None = None) -> None:
    global _memory_store
    _memory_store = MemoryStore(db_path=db_path)


def get_customer_memory_store() -> MemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore.from_env()
    return _memory_store


@contextmanager
def isolated_customer_memory_store(db_path: str | Path) -> Iterator[MemoryStore]:
    """Temporarily install and then close an evaluation-only memory store.

    This process-global seam is intended only for sequential offline evaluation.
    """
    global _memory_store
    previous_store = _memory_store
    evaluation_store = MemoryStore(db_path=db_path)
    _memory_store = evaluation_store
    try:
        yield evaluation_store
    finally:
        evaluation_store.close()
        _memory_store = previous_store


def write_customer_preference(arguments: BaseModel, context) -> dict:
    memory_type = getattr(arguments, "preference_type")
    sensitivity = sensitivity_for_memory_type(memory_type)
    proposal = MemoryProposal(
        type=memory_type,
        content=getattr(arguments, "preference_value"),
        evidence=getattr(arguments, "evidence"),
        confidence=0.9,
        sensitivity=sensitivity,
        requires_confirmation=memory_requires_confirmation(memory_type, sensitivity),
    )
    result = get_customer_memory_store().upsert(
        user_id=getattr(arguments, "user_id", context.user_id),
        proposal=proposal,
        trace_id=context.trace_id,
        conversation_id=context.conversation_id,
        trace_events=context.trace_events,
    )
    return {
        "memory_id": result.memory.id,
        "status": result.action,
        "review_status": result.memory.review_status,
    }


def delete_customer_memory(arguments: BaseModel, context) -> dict:
    memory_id = getattr(arguments, "memory_id")
    deleted = get_customer_memory_store().delete(
        user_id=getattr(arguments, "user_id", context.user_id),
        memory_id=memory_id,
        trace_id=context.trace_id,
        conversation_id=context.conversation_id,
        trace_events=context.trace_events,
    )
    return {
        "memory_id": memory_id,
        "status": "deleted" if deleted else "not_found",
    }


def lookup_customer_profile(arguments: BaseModel, context) -> dict:
    user_id = getattr(arguments, "user_id", context.user_id)
    memories = get_customer_memory_store().list_user_memories(user_id)
    return {
        "user_id": user_id,
        "known_preferences": [memory.content for memory in memories],
        "memories": [
            {
                "memory_id": memory.id,
                "type": memory.type,
                "content": memory.content,
                "sensitivity": memory.sensitivity,
                "status": memory.status,
                "review_status": memory.review_status,
                "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
                "source_conversation_id": memory.source_conversation_id,
                "source_trace_id": memory.source_trace_id,
                "version": memory.version,
            }
            for memory in memories
        ],
    }
