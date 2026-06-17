from __future__ import annotations

import uuid
from datetime import timezone
from typing import Any

from .customer_memory import CustomerMemory, MemoryProposal, MemoryWriteResult, utc_now
from .memory_deduper import find_conflict, find_duplicate


class MemoryStore:
    def __init__(self) -> None:
        self._memories_by_user: dict[str, list[CustomerMemory]] = {}

    def list_user_memories(self, user_id: str) -> list[CustomerMemory]:
        return [
            memory
            for memory in self._memories_by_user.get(user_id, [])
            if memory.deleted_at is None
        ]

    def upsert(
        self,
        user_id: str,
        proposal: MemoryProposal,
        trace_id: str = "",
        conversation_id: str = "",
        trace_events: list[dict[str, Any]] | None = None,
    ) -> MemoryWriteResult:
        existing_memories = self._memories_by_user.setdefault(user_id, [])
        duplicate = find_duplicate(proposal, existing_memories)
        if duplicate:
            duplicate.evidence = proposal.evidence
            duplicate.confidence = proposal.confidence
            duplicate.updated_at = utc_now()
            result = MemoryWriteResult(action="updated", memory=duplicate)
            _append_trace(trace_events, trace_id, conversation_id, "memory_updated", duplicate)
            return result

        candidate = CustomerMemory(
            id=f"memory_{uuid.uuid4().hex[:8]}",
            user_id=user_id,
            type=proposal.type,
            content=proposal.content,
            evidence=proposal.evidence,
            confidence=proposal.confidence,
            sensitivity=proposal.sensitivity,
        )
        conflict = find_conflict(proposal, existing_memories)
        if conflict:
            result = MemoryWriteResult(action="conflict", memory=candidate, conflict_with=conflict)
            _append_trace(trace_events, trace_id, conversation_id, "memory_conflict", candidate)
            return result

        existing_memories.append(candidate)
        result = MemoryWriteResult(action="created", memory=candidate)
        _append_trace(trace_events, trace_id, conversation_id, "memory_written", candidate)
        return result

    def delete(
        self,
        user_id: str,
        memory_id: str,
        trace_id: str = "",
        conversation_id: str = "",
        trace_events: list[dict[str, Any]] | None = None,
    ) -> bool:
        for memory in self._memories_by_user.get(user_id, []):
            if memory.id == memory_id and memory.deleted_at is None:
                memory.deleted_at = utc_now()
                memory.updated_at = memory.deleted_at
                _append_trace(trace_events, trace_id, conversation_id, "memory_deleted", memory)
                return True
        return False


def _append_trace(
    trace_events: list[dict[str, Any]] | None,
    trace_id: str,
    conversation_id: str,
    event_type: str,
    memory: CustomerMemory,
) -> None:
    if trace_events is None:
        return
    trace_events.append(
        {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "node": "memory_store",
            "event_type": event_type,
            "timestamp": utc_now().astimezone(timezone.utc).isoformat(),
            "metadata": {
                "memory_id": memory.id,
                "memory_type": memory.type,
                "sensitivity": memory.sensitivity,
            },
            "error": None,
        }
    )
