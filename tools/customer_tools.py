from pydantic import BaseModel

from memory.customer_memory import MemoryProposal
from memory.memory_policy import memory_requires_confirmation, sensitivity_for_memory_type
from memory.memory_store import MemoryStore


_memory_store = MemoryStore()


def reset_customer_memory_store() -> None:
    global _memory_store
    _memory_store = MemoryStore()


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
    result = _memory_store.upsert(
        user_id=getattr(arguments, "user_id", context.user_id),
        proposal=proposal,
        trace_id=context.trace_id,
        conversation_id=context.conversation_id,
        trace_events=context.trace_events,
    )
    return {
        "memory_id": result.memory.id,
        "status": result.action,
    }


def delete_customer_memory(arguments: BaseModel, context) -> dict:
    memory_id = getattr(arguments, "memory_id")
    deleted = _memory_store.delete(
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
    memories = _memory_store.list_user_memories(user_id)
    return {
        "user_id": user_id,
        "known_preferences": [memory.content for memory in memories],
        "memories": [
            {
                "memory_id": memory.id,
                "type": memory.type,
                "content": memory.content,
                "sensitivity": memory.sensitivity,
            }
            for memory in memories
        ],
    }
